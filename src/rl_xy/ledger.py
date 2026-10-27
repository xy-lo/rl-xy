"""Auditable storage and capacity ledgers for the frozen toy workload.

The ledger deliberately separates physical DRAM occupancy from the logical KV
view.  ``memory_snapshots`` contains the real resource accounting (persistent
state plus KV materialised so far), while ``kv_snapshots`` exposes the KV-only
curve without pretending that it is a second allocation.  A reservation is a
future-growth budget: materialising KV moves bytes from ``unmaterialized`` to
``allocated`` and never charges both twice.

This module is a deterministic accounting model, not a replacement for the
AstraSim execution log.  Object lifetime markers are tied to operation ids when
the operation list is supplied; validation independently checks the realised
AstraSim events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .model import enumerate_parameters, kv_bytes_per_processed_token_per_die, parameter_totals
from .types import CaseConfig, MemorySnapshot, ObjectRecord, Operation
from .execution_contract import prompt_for


@dataclass(frozen=True)
class LedgerBundle:
    """Complete accounting output for one configured run."""

    objects: tuple[ObjectRecord, ...]
    memory_snapshots: tuple[MemorySnapshot, ...]
    kv_snapshots: tuple[MemorySnapshot, ...]
    kv_bytes_written_per_die: dict[str, int]

    @property
    def all_snapshots(self) -> tuple[MemorySnapshot, ...]:
        """Return physical snapshots followed by the explicitly-labelled view."""

        return self.memory_snapshots + self.kv_snapshots


_STATE_LAYOUT = (
    ("actor", "bf16", "actor_dtype_bytes"),
    ("reference", "bf16", "reference_dtype_bytes"),
    ("master", "fp32", "master_dtype_bytes"),
    ("adam_m", "fp32", "optimizer_dtype_bytes"),
    ("adam_v", "fp32", "optimizer_dtype_bytes"),
    ("gradient", "fp32", "gradient_dtype_bytes"),
)


def _last_op_id(
    operations: tuple[Operation, ...],
    *,
    round_id: int | None = None,
    phase: str | None = None,
    name: str | None = None,
    fallback: str,
) -> str:
    candidates = [
        op
        for op in operations
        if (round_id is None or op.round_id == round_id)
        and (phase is None or op.phase == phase)
        and (name is None or op.name == name)
    ]
    return candidates[-1].op_id if candidates else fallback


def _persistent_objects(
    case: CaseConfig, operations: tuple[Operation, ...]
) -> list[ObjectRecord]:
    result: list[ObjectRecord] = []
    final_round = case.workload.round_id + case.rounds - 1
    optimizer_last = _last_op_id(
        operations,
        round_id=final_round,
        phase="optimizer",
        fallback=f"r{final_round:02d}:optimizer",
    )
    reference_last = _last_op_id(
        operations,
        round_id=final_round,
        phase="reference",
        fallback=f"r{final_round:02d}:reference",
    )
    for rank in range(case.hardware.dies):
        location = f"d{rank}.dram"
        for parameter in enumerate_parameters(case):
            for kind, dtype, byte_field in _STATE_LAYOUT:
                nbytes = parameter.local_elements_per_die * getattr(case.model, byte_field)
                if kind == "reference":
                    last_consumer = reference_last
                    version: int | None = case.workload.behavior_version
                elif kind == "gradient":
                    last_consumer = optimizer_last
                    version = None
                else:
                    last_consumer = optimizer_last
                    version = case.workload.behavior_version
                result.append(
                    ObjectRecord(
                        storage_id=f"d{rank}.{kind}.{parameter.parameter_id}",
                        tensor_id=f"{kind}.{parameter.parameter_id}",
                        kind=kind,
                        location=location,
                        nbytes=nbytes,
                        dtype=dtype,
                        version=version,
                        created_at="initialization",
                        ready_at="initialization",
                        last_consumer=last_consumer,
                        released_at=None,
                        persistent=True,
                        owner_rank=rank,
                    )
                )
    return result


def _kv_objects(
    case: CaseConfig, operations: tuple[Operation, ...]
) -> list[ObjectRecord]:
    """Create one storage row per real KV allocation increment.

    The final generated token is not fed back into the model.  Therefore A
    processes 2+1+1 positions and B processes 2+1 positions in each round.
    """

    per_token = kv_bytes_per_processed_token_per_die(case)
    prompt_tokens = len(case.workload.prompt_token_ids)
    result: list[ObjectRecord] = []
    for offset in range(case.rounds):
        round_id = case.workload.round_id + offset
        version = case.workload.behavior_version + offset
        for trajectory in case.workload.trajectories:
            prompt_tokens = len(prompt_for(case, trajectory))
            trajectory_id = trajectory.trajectory_id
            increments = [("prefill", prompt_tokens)] + [
                (f"decode{decode_index}", 1)
                for decode_index in range(1, len(trajectory.response_token_ids))
            ]
            final_stage = increments[-1][0]
            release_op = _last_op_id(
                operations,
                round_id=round_id,
                phase="rollout",
                name=f"rollout_{trajectory_id}_{final_stage}.sample_token_rank0_to_rank1",
                fallback=f"r{round_id:02d}:rollout.{trajectory_id}.{final_stage}.complete",
            )
            for rank in range(case.hardware.dies):
                for stage, processed_tokens in increments:
                    ready_op = _last_op_id(
                        operations,
                        round_id=round_id,
                        phase="rollout",
                        name=f"rollout_{trajectory_id}_{stage}.sample_token_rank0_to_rank1",
                        fallback=f"r{round_id:02d}:rollout.{trajectory_id}.{stage}.complete",
                    )
                    result.append(
                        ObjectRecord(
                            storage_id=(
                                f"r{round_id:02d}.d{rank}.kv."
                                f"{trajectory_id}.{stage}"
                            ),
                            tensor_id=f"r{round_id:02d}.kv.{trajectory_id}",
                            kind="kv",
                            location=f"d{rank}.dram",
                            nbytes=processed_tokens * per_token,
                            dtype="bf16",
                            version=version,
                            created_at=ready_op,
                            ready_at=ready_op,
                            last_consumer=release_op,
                            released_at=release_op,
                            persistent=False,
                            owner_rank=rank,
                        )
                    )
    return result


def _snapshot(
    *,
    step: int,
    phase: str,
    location: str,
    allocated: int,
    valid: int,
    unmaterialized: int,
    capacity: int,
    reason: str,
) -> MemorySnapshot:
    return MemorySnapshot(
        step=step,
        phase=phase,
        location=location,
        allocated_bytes=allocated,
        valid_bytes=valid,
        unmaterialized_bytes=unmaterialized,
        capacity_bytes=capacity,
        reason=reason,
    )


def _memory_curves(
    case: CaseConfig,
    operations: tuple[Operation, ...],
) -> tuple[list[MemorySnapshot], list[MemorySnapshot], dict[str, int]]:
    totals = parameter_totals(case)
    resident = totals["resident_with_gradient_bytes_per_die"]
    per_token = kv_bytes_per_processed_token_per_die(case)
    prompt_tokens = len(case.workload.prompt_token_ids)
    trajectories = {item.trajectory_id: item for item in case.workload.trajectories}

    # The named toy contract fixes A=3 response tokens and B=2.  Generalising
    # this curve without an event-driven allocator would hide scheduling
    # choices, so reject it here rather than silently fabricate a curve.
    if set(trajectories) != {"A", "B"}:
        raise ValueError("KV ledger requires the frozen A/B toy trajectories")
    if len(trajectories["A"].response_token_ids) != 3 or len(
        trajectories["B"].response_token_ids
    ) != 2:
        raise ValueError("KV ledger requires toy response lengths A=3 and B=2")

    per_trajectory_reservation = per_token * (
        prompt_tokens + case.workload.max_response_length - 1
    )
    reservation = len(trajectories) * per_trajectory_reservation
    physical: list[MemorySnapshot] = []
    kv_view: list[MemorySnapshot] = []
    cumulative: dict[str, int] = {}
    step = 0
    # (reason, materialised KV, still-unmaterialised reservation)
    curve = (
        ("kv_admission", 0, reservation),
        ("after_prefill", 4 * per_token, reservation - 4 * per_token),
        ("after_decode1", 6 * per_token, reservation - 6 * per_token),
        ("after_release_B", 3 * per_token, per_trajectory_reservation - 3 * per_token),
        ("after_decode2", 4 * per_token, 0),
        ("after_release_A", 0, 0),
    )
    written_per_round = per_token * (prompt_tokens * 2 + 3)
    for offset in range(case.rounds):
        round_id = case.workload.round_id + offset
        for rank in range(case.hardware.dies):
            cumulative[f"r{round_id}.d{rank}"] = written_per_round
            for reason, kv_allocated, unmaterialized in curve:
                phase = f"r{round_id:02d}.rollout"
                physical.append(
                    _snapshot(
                        step=step,
                        phase=phase,
                        location=f"d{rank}.dram",
                        allocated=resident + kv_allocated,
                        valid=resident + kv_allocated,
                        unmaterialized=unmaterialized,
                        capacity=case.hardware.dram_bytes_per_die,
                        reason=reason,
                    )
                )
                kv_view.append(
                    _snapshot(
                        step=step,
                        phase=phase,
                        location=f"d{rank}.dram.kv_view",
                        allocated=kv_allocated,
                        valid=kv_allocated,
                        unmaterialized=unmaterialized,
                        capacity=case.hardware.dram_bytes_per_die - resident,
                        reason=f"{reason}; logical_view_no_extra_allocation",
                    )
                )
                step += 1

    # Explicit zero-occupancy rows keep the other independently-capacitated
    # resources visible in the output.  AstraSim times the transfers; it does
    # not turn an arbitrary full payload into a hidden NIC/SRAM allocation.
    for rank in range(case.hardware.dies):
        tx_payloads = [
            min(item.payload_bytes, case.hardware.d2d_packet_bytes)
            for item in operations
            if item.kind == "p2p" and item.src_rank == rank
        ]
        rx_payloads = [
            min(item.payload_bytes, case.hardware.d2d_packet_bytes)
            for item in operations
            if item.kind == "p2p" and item.dst_rank == rank
        ]
        tx_peak = max(tx_payloads, default=0)
        rx_peak = max(rx_payloads, default=0)
        physical.extend(
            [
                _snapshot(
                    step=step,
                    phase="initialization",
                    location=f"d{rank}.c0.sram",
                    allocated=0,
                    valid=0,
                    unmaterialized=0,
                    capacity=case.hardware.sram_bytes_per_core,
                    reason="independent_resource_visible",
                ),
                _snapshot(
                    step=step + 1,
                    phase="initialization",
                    location=f"d{rank}.nic.tx",
                    allocated=tx_peak,
                    valid=tx_peak,
                    unmaterialized=0,
                    capacity=case.hardware.nic_tx_bytes_per_die,
                    reason="planned_p2p_packet_peak; native_collective_buffers_not_reported",
                ),
                _snapshot(
                    step=step + 2,
                    phase="initialization",
                    location=f"d{rank}.nic.rx",
                    allocated=rx_peak,
                    valid=rx_peak,
                    unmaterialized=0,
                    capacity=case.hardware.nic_rx_bytes_per_die,
                    reason="planned_p2p_packet_peak; native_collective_buffers_not_reported",
                ),
            ]
        )
        step += 3
    return physical, kv_view, cumulative


def build_ledger(
    case: CaseConfig, operations: Iterable[Operation] | None = None
) -> LedgerBundle:
    """Build object lifetimes and per-resource capacity snapshots.

    This function never suppresses accounting errors.  An unsupported toy
    trace raises ``ValueError``; realised checks are returned by
    :func:`rl_xy.validation.validate_run` as structured ``CheckResult`` rows.
    """

    op_tuple = tuple(operations or ())
    if case.schema_version == "1.1":
        # Generic runs take full peak/lifetime evidence from actual native
        # resource callbacks, not the historical hard-coded A/B KV curve.
        resident=parameter_totals(case)["resident_with_gradient_bytes_per_die"]
        snapshots=tuple(_snapshot(step=rank,phase="initialization",location=f"d{rank}.dram",
            allocated=resident,valid=resident,unmaterialized=0,
            capacity=case.hardware.dram_bytes_per_die,reason="persistent_initial_only; see runtime memory.csv")
            for rank in range(case.hardware.dies))
        per_round=sum(len(prompt_for(case,t))+len(t.response_token_ids)-1 for t in case.workload.trajectories)*kv_bytes_per_processed_token_per_die(case)
        return LedgerBundle(tuple(_persistent_objects(case,op_tuple)+_kv_objects(case,op_tuple)), snapshots, (),
            {f"r{case.workload.round_id+offset}.d{rank}":per_round for offset in range(case.rounds)
             for rank in range(case.hardware.dies)})
    physical, kv_view, cumulative = _memory_curves(case, op_tuple)
    objects = _persistent_objects(case, op_tuple) + _kv_objects(case, op_tuple)
    return LedgerBundle(
        objects=tuple(objects),
        memory_snapshots=tuple(physical),
        kv_snapshots=tuple(kv_view),
        kv_bytes_written_per_die=cumulative,
    )

