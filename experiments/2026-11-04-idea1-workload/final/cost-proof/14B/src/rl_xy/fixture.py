"""Independent 21 ns transfer fixture from the MVP design document."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import CheckResult, MemorySnapshot, ObjectRecord, Operation


class FixtureError(ValueError):
    """The fixture configuration or resource allocation is invalid."""


FIXTURE_FIELDS = {
    "schema_version",
    "case_id",
    "evidence_kind",
    "input_bytes",
    "dram_bytes_per_die",
    "sram_bytes_per_core",
    "nic_tx_bytes_per_die",
    "nic_rx_bytes_per_die",
    "local_dma_bytes_per_ns",
    "d2d_bytes_per_ns",
    "d2d_latency_ns",
    "astra_time_scale",
    "compute_duration_ns",
    "expected_completion_ns",
    "expected_sram_peak_bytes",
    "expected_d2d_payload_bytes",
    "expected_dram_traffic_bytes",
}


@dataclass(frozen=True)
class TransferFixture:
    source_path: Path
    config: dict[str, Any]
    operations: tuple[Operation, ...]
    objects: tuple[ObjectRecord, ...]
    memory_snapshots: tuple[MemorySnapshot, ...]


def load_fixture(path: str | Path, *, sram_override: int | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FixtureError("fixture root must be an object")
    unknown = sorted(set(raw) - FIXTURE_FIELDS)
    missing = sorted(FIXTURE_FIELDS - set(raw))
    if unknown or missing:
        raise FixtureError(f"fixture fields: unknown={unknown}, missing={missing}")
    for key in FIXTURE_FIELDS - {"schema_version", "case_id", "evidence_kind"}:
        if not isinstance(raw[key], int) or isinstance(raw[key], bool) or raw[key] <= 0:
            raise FixtureError(f"{key} must be a positive integer")
    if raw["schema_version"] != "1.0":
        raise FixtureError("unsupported fixture schema")
    if raw["evidence_kind"] != "synthetic_fixture":
        raise FixtureError("transfer fixture must be labelled synthetic_fixture")
    if sram_override is not None:
        if sram_override <= 0:
            raise FixtureError("sram_override must be positive")
        raw["sram_bytes_per_core"] = sram_override
    return raw


def _op(
    node_id: int,
    name: str,
    duration_ns: int,
    *,
    executor: str,
    tensor_bytes: int,
    rank: int,
) -> Operation:
    dependency = () if node_id == 1 else (node_id - 1,)
    return Operation(
        node_id=node_id,
        op_id=f"fixture.n{node_id:02d}.{name}",
        round_id=0,
        phase="transfer_fixture",
        name=name,
        kind="compute",
        duration_ns=duration_ns,
        tensor_bytes=tensor_bytes,
        dependencies=dependency,
        metadata={
            "logical_executor": executor,
            "adapter_execution": "rank_local_synthetic_replay",
            "evidence_kind": "synthetic_fixture",
        },
        participant_ranks=(rank,),
        dependencies_by_rank={rank: dependency},
    )


def build_transfer_fixture(
    path: str | Path, *, sram_override: int | None = None
) -> TransferFixture:
    config = load_fixture(path, sram_override=sram_override)
    nbytes = config["input_bytes"]
    required_sram = 2 * nbytes
    if config["sram_bytes_per_core"] < required_sram:
        raise FixtureError(
            "out-of-place compute requires simultaneous input and output: "
            f"required={required_sram}, capacity={config['sram_bytes_per_core']}"
        )
    if config["nic_tx_bytes_per_die"] < nbytes or config["nic_rx_bytes_per_die"] < nbytes:
        raise FixtureError("NIC endpoint buffer cannot hold the fixture payload")
    if config["dram_bytes_per_die"] < 4 * nbytes:
        raise FixtureError("DRAM capacity cannot hold the frozen fixture objects")
    dma_ns = (nbytes + config["local_dma_bytes_per_ns"] - 1) // config[
        "local_dma_bytes_per_ns"
    ]
    compute_ns = config["compute_duration_ns"]
    operations: list[Operation] = [
        _op(1, "d0_dram_to_sram", dma_ns, executor="rank0.dma", tensor_bytes=nbytes, rank=0),
        _op(2, "d0_compute", compute_ns, executor="rank0.core0", tensor_bytes=2 * nbytes, rank=0),
        _op(3, "d0_sram_to_dram", dma_ns, executor="rank0.dma", tensor_bytes=nbytes, rank=0),
        _op(4, "d0_dram_to_tx", dma_ns, executor="rank0.nic_tx", tensor_bytes=nbytes, rank=0),
        Operation(
            node_id=5,
            op_id="fixture.n05.d0_to_d1_packet",
            round_id=0,
            phase="transfer_fixture",
            name="d0_to_d1_packet",
            kind="p2p",
            payload_bytes=nbytes,
            src_rank=0,
            dst_rank=1,
            dependencies=(4,),
            metadata={"route": "D0.DRAM->TX->D2D->RX->D1.DRAM"},
            participant_ranks=(0, 1),
            dependencies_by_rank={0: (4,), 1: ()},
        ),
        _op(6, "rx_to_d1_dram", dma_ns, executor="rank1.nic_rx", tensor_bytes=nbytes, rank=1),
        _op(7, "d1_dram_to_sram", dma_ns, executor="rank1.dma", tensor_bytes=nbytes, rank=1),
        _op(8, "d1_compute", compute_ns, executor="rank1.core0", tensor_bytes=2 * nbytes, rank=1),
        _op(9, "d1_sram_to_dram", dma_ns, executor="rank1.dma", tensor_bytes=nbytes, rank=1),
    ]
    capacity = config["sram_bytes_per_core"]
    memory = (
        MemorySnapshot(0, "transfer_fixture", "rank0.sram0", nbytes, nbytes, 0, capacity, "input_ready"),
        MemorySnapshot(1, "transfer_fixture", "rank0.sram0", required_sram, required_sram, 0, capacity, "out_of_place_compute"),
        MemorySnapshot(2, "transfer_fixture", "rank0.sram0", 0, 0, 0, capacity, "released_after_store"),
        MemorySnapshot(3, "transfer_fixture", "rank1.sram0", nbytes, nbytes, 0, capacity, "input_ready"),
        MemorySnapshot(4, "transfer_fixture", "rank1.sram0", required_sram, required_sram, 0, capacity, "out_of_place_compute"),
        MemorySnapshot(5, "transfer_fixture", "rank1.sram0", 0, 0, 0, capacity, "released_after_store"),
        MemorySnapshot(6, "transfer_fixture", "rank0.nic.tx", nbytes, nbytes, 0, config["nic_tx_bytes_per_die"], "p2p_packet_peak"),
        MemorySnapshot(7, "transfer_fixture", "rank1.nic.rx", nbytes, nbytes, 0, config["nic_rx_bytes_per_die"], "p2p_packet_peak"),
    )
    objects = (
        ObjectRecord("fx.in.d0", "input", "activation", "rank0.dram", nbytes, "bytes", None, "start", "start", "fixture.n01", "fixture.n01", False, 0),
        ObjectRecord("fx.mid.d0", "d0_result", "activation", "rank0.dram", nbytes, "bytes", None, "fixture.n02", "fixture.n03", "fixture.n04", "fixture.n05", False, 0),
        ObjectRecord("fx.rx.d1", "d1_input", "activation", "rank1.dram", nbytes, "bytes", None, "fixture.n05", "fixture.n06", "fixture.n07", "fixture.n07", False, 1),
        ObjectRecord("fx.out.d1", "output", "activation", "rank1.dram", nbytes, "bytes", None, "fixture.n08", "fixture.n09", "external", None, False, 1),
    )
    return TransferFixture(Path(path).resolve(), config, tuple(operations), objects, memory)


def validate_transfer_fixture(
    fixture: TransferFixture, total_ns: float, p2p_duration_ns: float
) -> list[CheckResult]:
    config = fixture.config
    peak = max(item.allocated_bytes for item in fixture.memory_snapshots)
    actual_payload = sum(op.payload_bytes for op in fixture.operations if op.kind == "p2p")
    # D0: three reads/writes, D1: three reads/writes, each one payload wide.
    dram_traffic = 6 * config["input_bytes"]
    expected_p2p = config["d2d_latency_ns"] + config["input_bytes"] / config[
        "d2d_bytes_per_ns"
    ]
    return [
        CheckResult("fixture_completion", total_ns == config["expected_completion_ns"], config["expected_completion_ns"], total_ns, "AstraSim final callback", "synthetic_fixture"),
        CheckResult("fixture_p2p_latency", p2p_duration_ns == expected_p2p, expected_p2p, p2p_duration_ns, "AstraSim node 5 callback-issue", "synthetic_fixture"),
        CheckResult("fixture_sram_peak", peak == config["expected_sram_peak_bytes"], config["expected_sram_peak_bytes"], peak, "explicit out-of-place allocation", "synthetic_fixture"),
        CheckResult("fixture_d2d_payload", actual_payload == config["expected_d2d_payload_bytes"], config["expected_d2d_payload_bytes"], actual_payload, "p2p operation payload", "synthetic_fixture"),
        CheckResult("fixture_dram_traffic", dram_traffic == config["expected_dram_traffic_bytes"], config["expected_dram_traffic_bytes"], dram_traffic, "six explicit DRAM endpoint transfers", "synthetic_fixture"),
    ]


def validate_63b_negative(path: str | Path) -> CheckResult:
    try:
        build_transfer_fixture(path, sram_override=63)
    except FixtureError as exc:
        return CheckResult("fixture_63b_rejected", True, "allocation failure", "allocation failure", str(exc), "synthetic_fixture_negative")
    return CheckResult("fixture_63b_rejected", False, "allocation failure", "accepted", "63 B must not fit simultaneous 32 B input and output", "synthetic_fixture_negative")
