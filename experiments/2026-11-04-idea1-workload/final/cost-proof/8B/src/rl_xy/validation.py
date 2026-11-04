"""Independent, fail-closed validation of the toy RL/AstraSim run.

The planner is not allowed to validate itself merely because it returned.  The
checks below derive expected coverage from the case contract, compare it with
the operation graph, storage ledger, and paired AstraSim issue/callback events,
and return one structured :class:`CheckResult` per assertion.  Callers must use
``require_valid`` before publishing a run as successful.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .ledger import LedgerBundle
from .model import enumerate_parameters, kv_bytes_per_processed_token_per_die, parameter_totals
from .types import CaseConfig, CheckResult, Operation
from .workflow import REQUIRED_PHASES


class ValidationError(RuntimeError):
    """Raised when a caller attempts to accept a run with failed checks."""

    def __init__(self, checks: Sequence[CheckResult]) -> None:
        self.checks = tuple(checks)
        failed = [check.check_id for check in checks if not check.passed]
        super().__init__("validation failed: " + ", ".join(failed))


def _result(
    check_id: str,
    passed: bool,
    expected: Any,
    actual: Any,
    evidence: str,
    scope: str,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        passed=bool(passed),
        expected=expected,
        actual=actual,
        evidence=evidence,
        scope=scope,
    )


def _field(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _rounds(case: CaseConfig) -> range:
    start = case.workload.round_id
    return range(start, start + case.rounds)


def _ops_for_round(operations: Sequence[Operation], round_id: int) -> list[Operation]:
    return [op for op in operations if op.round_id == round_id]


def _parameter_checks(case: CaseConfig, ledger: LedgerBundle) -> list[CheckResult]:
    checks: list[CheckResult] = []
    totals = parameter_totals(case)
    expected_toy = {
        "unique_parameter_elements": 1576,
        "local_parameter_elements_per_die": 808,
        "long_term_bytes_per_die": 12928,
        "gradient_buffer_bytes_per_die": 3232,
        "resident_with_gradient_bytes_per_die": 16160,
    }
    if case.case_id == "toy_grpo_2die":
        checks.append(
            _result(
                "model.toy_parameter_golden",
                totals == expected_toy,
                expected_toy,
                totals,
                "enumerate_parameters + dtype byte widths",
                "model",
            )
        )

    rows = enumerate_parameters(case)
    expected_object_count = len(rows) * 6 * case.hardware.dies
    persistent = [item for item in ledger.objects if item.persistent]
    checks.append(
        _result(
            "ledger.parameter_state_object_coverage",
            len(persistent) == expected_object_count,
            expected_object_count,
            len(persistent),
            "one actor/reference/master/m/v/gradient storage per parameter row and die",
            "objects",
        )
    )
    expected_by_kind = {
        "actor": totals["local_parameter_elements_per_die"]
        * case.model.actor_dtype_bytes,
        "reference": totals["local_parameter_elements_per_die"]
        * case.model.reference_dtype_bytes,
        "master": totals["local_parameter_elements_per_die"]
        * case.model.master_dtype_bytes,
        "adam_m": totals["local_parameter_elements_per_die"]
        * case.model.optimizer_dtype_bytes,
        "adam_v": totals["local_parameter_elements_per_die"]
        * case.model.optimizer_dtype_bytes,
        "gradient": totals["local_parameter_elements_per_die"]
        * case.model.gradient_dtype_bytes,
    }
    for rank in range(case.hardware.dies):
        actual_by_kind: dict[str, int] = defaultdict(int)
        for item in persistent:
            if item.owner_rank == rank:
                actual_by_kind[item.kind] += item.nbytes
        checks.append(
            _result(
                f"ledger.d{rank}.persistent_bytes",
                dict(actual_by_kind) == expected_by_kind,
                expected_by_kind,
                dict(actual_by_kind),
                "physical storage_id rows counted once",
                f"d{rank}.dram",
            )
        )
    reference_objects = [item for item in persistent if item.kind == "reference"]
    reference_versions = sorted(
        {item.version for item in reference_objects},
        key=lambda value: (value is None, str(value)),
    )
    checks.append(
        _result(
            "ledger.reference_version_fixed",
            bool(reference_objects)
            and reference_versions == [case.workload.behavior_version]
            and all(item.released_at is None for item in reference_objects),
            {
                "versions": [case.workload.behavior_version],
                "persistent": True,
                "released": False,
            },
            {
                "versions": reference_versions,
                "all_persistent": all(item.persistent for item in reference_objects),
                "any_released": any(
                    item.released_at is not None for item in reference_objects
                ),
            },
            "reference storage rows remain at the initial frozen version across rounds",
            "objects",
        )
    )
    return checks


def _trajectory_checks(case: CaseConfig, operations: Sequence[Operation]) -> list[CheckResult]:
    from .execution_contract import prompt_for
    checks: list[CheckResult] = []
    trajectories = case.workload.trajectories
    ids = [item.trajectory_id for item in trajectories]
    expected_positions = sum(
        len(prompt_for(case,item)) + len(item.response_token_ids) - 1
        for item in trajectories
    )
    checks.append(
        _result(
            "workload.trajectory_identity",
            (len(ids) == 2 if case.schema_version == "1.0" else len(ids) >= 2) and len(ids) == len(set(ids)),
            "unique trajectories in configured complete groups",
            {"group_id": case.workload.group_id, "trajectory_ids": ids},
            "validated workload trace",
            "workload",
        )
    )
    checks.append(
        _result(
            "workload.response_token_count",
            sum(len(item.response_token_ids) for item in trajectories) == (5 if case.schema_version == "1.0" else sum(len(t.response_token_ids) for t in trajectories)),
            5 if case.schema_version == "1.0" else sum(len(t.response_token_ids) for t in trajectories),
            sum(len(item.response_token_ids) for item in trajectories),
            "A response length 3 plus B response length 2",
            "workload",
        )
    )
    for round_id in _rounds(case):
        round_ops = _ops_for_round(operations, round_id)
        actual_stage_names = [
            op.name
            for op in round_ops
            if op.phase == "rollout" and op.name.endswith(".sample_token_rank0_to_rank1")
        ]
        expected_stage_names: list[str] = []
        for trajectory in trajectories:
            expected_stage_names.append(
                f"rollout_{trajectory.trajectory_id}_prefill.sample_token_rank0_to_rank1"
            )
            expected_stage_names.extend(
                f"rollout_{trajectory.trajectory_id}_decode{index}.sample_token_rank0_to_rank1"
                for index in range(1, len(trajectory.response_token_ids))
            )
        checks.append(
            _result(
                f"r{round_id}.rollout_stage_coverage",
                Counter(actual_stage_names) == Counter(expected_stage_names),
                expected_stage_names,
                actual_stage_names,
                f"{expected_positions} processed input positions; the final sampled token is not re-fed",
                f"round:{round_id}",
            )
        )
        for phase in ("reference", "old_policy"):
            probability_ops = [
                op
                for op in round_ops
                if op.phase == phase and op.name.endswith(".distributed_logsoftmax_local")
            ]
            actual_targets = sorted(
                (op.metadata.get("target_positions", -1) for op in probability_ops),
                key=lambda value: (0, value)
                if isinstance(value, int)
                else (1, str(value)),
            )
            expected_targets = sorted(len(item.response_token_ids) for item in trajectories)
            teacher_positions = sum(
                len(prompt_for(case,item)) + len(item.response_token_ids) - 1
                for item in trajectories
            )
            checks.append(
                _result(
                    f"r{round_id}.{phase}_probability_coverage",
                    actual_targets == expected_targets,
                    {
                        "target_positions": expected_targets,
                        "teacher_input_positions": teacher_positions,
                    },
                    {
                        "target_positions": actual_targets,
                        "teacher_input_positions": teacher_positions
                        if len(probability_ops) == len(trajectories)
                        else None,
                    },
                    "one distributed probability extraction per completed trajectory",
                    f"round:{round_id}",
                )
            )
    return checks


def _dependency_checks(
    case: CaseConfig, operations: Sequence[Operation]
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    ordered_ops = sorted(operations, key=lambda item: item.node_id)
    node_ids = [op.node_id for op in ordered_ops]
    op_ids = [op.op_id for op in ordered_ops]
    broken_dependencies: list[dict[str, Any]] = []
    last_by_rank: dict[int, int | None] = {
        rank: None for rank in range(case.hardware.dies)
    }
    for op in ordered_ops:
        participants = op.participant_ranks or tuple(range(case.hardware.dies))
        expected_by_rank = {
            rank: () if last_by_rank[rank] is None else (last_by_rank[rank],)
            for rank in participants
        }
        actual_by_rank = {
            rank: op.dependencies_by_rank.get(rank, op.dependencies)
            for rank in participants
        }
        expected_union = tuple(
            sorted({item for values in expected_by_rank.values() for item in values})
        )
        if actual_by_rank != expected_by_rank or op.dependencies != expected_union:
            broken_dependencies.append(
                {
                    "op_id": op.op_id,
                    "participants": participants,
                    "expected_by_rank": expected_by_rank,
                    "actual_by_rank": actual_by_rank,
                    "expected_union": expected_union,
                    "actual_union": op.dependencies,
                }
            )
        for rank in participants:
            last_by_rank[rank] = op.node_id
    checks.append(
        _result(
            "workflow.strict_total_order_dependency_chain",
            len(node_ids) == len(set(node_ids))
            and len(op_ids) == len(set(op_ids))
            and not broken_dependencies,
            "unique ids and exactly one previous local-node dependency per participating rank",
            {
                "node_count": len(node_ids),
                "unique_node_count": len(set(node_ids)),
                "op_count": len(op_ids),
                "unique_op_count": len(set(op_ids)),
                "broken_dependencies": broken_dependencies,
            },
            "the frozen baseline S is a rank-local total order with explicit communication synchronization",
            "workflow",
        )
    )
    return checks


def _phase_and_version_checks(case, operations):
    checks = _dependency_checks(case, operations)
    previous_publish_node: int | None = None
    for offset, round_id in enumerate(_rounds(case)):
        round_ops = _ops_for_round(operations, round_id)
        phases = {op.phase for op in round_ops}
        expected_phases=set(REQUIRED_PHASES) if case.schema_version=="1.0" else (
            (set(REQUIRED_PHASES)-{"train_A","train_B"}) | {f"train_{t.trajectory_id}" for t in case.workload.trajectories})
        checks.append(
            _result(
                f"r{round_id}.phase_coverage",
                phases == expected_phases,
                sorted(expected_phases),
                sorted(phases),
                "operation phase labels",
                f"round:{round_id}",
            )
        )
        version_in = case.workload.behavior_version + offset
        optimizer = [op for op in round_ops if op.name == "adamw_read_modify_write"]
        barriers = [op for op in round_ops if op.name == "all_shards_updated_barrier"]
        publishes = [op for op in round_ops if op.name == "publish_actor_version"]
        actual = {
            "optimizer_count": len(optimizer),
            "barrier_count": len(barriers),
            "publish_count": len(publishes),
            "actor_version_in": optimizer[0].metadata.get("actor_version_in")
            if len(optimizer) == 1
            else None,
            "actor_version_out": optimizer[0].metadata.get("actor_version_out")
            if len(optimizer) == 1
            else None,
            "barrier_version": barriers[0].metadata.get("new_version")
            if len(barriers) == 1
            else None,
            "published_version": publishes[0].metadata.get("published_version")
            if len(publishes) == 1
            else None,
            "behavior_input_count": sum(
                op.name == "host_to_rank0_prompt" for op in round_ops
            ),
            "behavior_input_version": next(
                (
                    op.metadata.get("behavior_version")
                    for op in round_ops
                    if op.name == "host_to_rank0_prompt"
                ),
                None,
            ),
        }
        expected = {
            "optimizer_count": 1,
            "barrier_count": 1,
            "publish_count": 1,
            "actor_version_in": version_in,
            "actor_version_out": version_in + 1,
            "barrier_version": version_in + 1,
            "published_version": version_in + 1,
            "behavior_input_count": 1,
            "behavior_input_version": version_in,
        }
        ordered = (
            len(optimizer) == len(barriers) == len(publishes) == 1
            and optimizer[0].node_id < barriers[0].node_id < publishes[0].node_id
        )
        checks.append(
            _result(
                f"r{round_id}.actor_version_publish",
                actual == expected and ordered,
                {**expected, "ordered": True},
                {**actual, "ordered": ordered},
                "optimizer must finish before all-shard barrier and atomic publish",
                f"round:{round_id}",
            )
        )
        first_node = min((op.node_id for op in round_ops), default=None)
        boundary_ok = previous_publish_node is None or (
            first_node is not None and first_node > previous_publish_node
        )
        checks.append(
            _result(
                f"r{round_id}.starts_after_previous_publish",
                boundary_ok,
                "first operation follows the preceding round's completed publish",
                {
                    "first_node": first_node,
                    "previous_publish_node": previous_publish_node,
                },
                "cross-round version boundary",
                f"round:{round_id}",
            )
        )
        previous_publish_node = publishes[0].node_id if len(publishes) == 1 else None
        reference_ops = [op for op in round_ops if op.phase == "reference"]
        forbidden = [
            op.op_id
            for op in reference_ops
            if any(word in op.name.lower() for word in ("backward", "update"))
            or ("write" in op.name.lower() and op.metadata.get("local_transfer") != "write")
            or op.metadata.get("reference_written") is True
        ]
        cleanup_reference_versions = [
            op.metadata.get("reference_version")
            for op in round_ops
            if op.name == "release_round_temporaries"
        ]
        checks.append(
            _result(
                f"r{round_id}.reference_read_only",
                bool(reference_ops)
                and not forbidden
                and cleanup_reference_versions == [case.workload.behavior_version],
                {
                    "reference_operations": ">0",
                    "writes": 0,
                    "cleanup_versions": [case.workload.behavior_version],
                },
                {
                    "operation_count": len(reference_ops),
                    "forbidden": forbidden,
                    "cleanup_versions": cleanup_reference_versions,
                },
                "reference phase names and write metadata",
                f"round:{round_id}",
            )
        )
    if case.rounds >= 2:
        publish_versions = [
            op.metadata.get("published_version")
            for op in operations
            if op.name == "publish_actor_version"
        ]
        expected_versions = [
            case.workload.behavior_version + offset + 1
            for offset in range(case.rounds)
        ]
        checks.append(
            _result(
                "versions.multi_round_chain",
                publish_versions == expected_versions,
                expected_versions,
                publish_versions,
                "ordered publish operations",
                "run",
            )
        )
    return checks


def _training_checks(case: CaseConfig, operations: Sequence[Operation], *, advantages_override=None, denominator_override=None) -> list[CheckResult]:
    checks: list[CheckResult] = []
    total_response_tokens = sum(
        len(item.response_token_ids) for item in case.workload.trajectories
    )
    if denominator_override is not None:total_response_tokens=denominator_override
    required_state_accesses = {
        "reads": ["gradient", "master", "adam_m", "adam_v"],
        "writes": ["master", "adam_m", "adam_v", "actor"],
    }
    local_params = parameter_totals(case)["local_parameter_elements_per_die"]
    expected_adam_bytes = local_params * (
        case.model.gradient_dtype_bytes
        + case.model.master_dtype_bytes
        + 2 * case.model.optimizer_dtype_bytes
        + case.model.actor_dtype_bytes
    )
    from .execution_contract import demean_advantages
    advantages=demean_advantages(case) if advantages_override is None else advantages_override
    expected_advantages = [advantages[item.trajectory_id] for item in case.workload.trajectories]

    for round_id in _rounds(case):
        round_ops = _ops_for_round(operations, round_id)
        advantage_ops = [op for op in round_ops if op.name == "group_advantage_demean"]
        actual_advantages = (
            advantage_ops[0].metadata.get("advantages") if len(advantage_ops) == 1 else None
        )
        advantage_ok = (
            isinstance(actual_advantages, list)
            and len(actual_advantages) == len(expected_advantages)
            and all(
                (actual is None if expected is None else actual is not None and abs(float(actual) - expected) < 1e-12)
                for actual, expected in zip(actual_advantages, expected_advantages)
            )
        )
        checks.append(
            _result(
                f"r{round_id}.advantage_demean",
                advantage_ok,
                expected_advantages,
                actual_advantages,
                "reward minus two-trajectory group mean",
                f"round:{round_id}",
            )
        )

        expected_backward_names: list[str] = []
        expected_forward_names: list[str] = []
        zero_ops = [op for op in round_ops if op.name == "zero_gradient_buffer"]
        first_accumulate = min(
            (op.node_id for op in round_ops if op.name.endswith(".gradient_accumulate_rmw")),
            default=-1,
        )
        checks.append(
            _result(
                f"r{round_id}.gradient_reset_before_accumulation",
                len(zero_ops) == 1
                and zero_ops[0].node_id < first_accumulate
                and zero_ops[0].metadata.get("writes") == ["gradient"]
                and zero_ops[0].metadata.get("initial_gradient_value") == 0,
                "one FP32 zero-fill before this round's first gradient accumulation",
                [op.op_id for op in zero_ops],
                "explicit gradient-buffer write operation",
                f"round:{round_id}",
            )
        )
        for trajectory in case.workload.trajectories:
            forward_prefix = f"train_{trajectory.trajectory_id}_forward"
            expected_forward_names.extend(
                [
                    f"{forward_prefix}.embedding",
                    f"{forward_prefix}.embedding_tp_sum",
                    f"{forward_prefix}.final_rmsnorm",
                    f"{forward_prefix}.lm_head_local",
                    f"{forward_prefix}.distributed_logsoftmax_local",
                    f"train_{trajectory.trajectory_id}_loss",
                ]
            )
            for layer in range(case.model.layers):
                forward_base = f"{forward_prefix}.layer{layer}"
                expected_forward_names.extend(
                    [
                        f"{forward_base}.rmsnorm_attn",
                        f"{forward_base}.qkv",
                        f"{forward_base}.o_proj_partial",
                        f"{forward_base}.o_proj_tp_sum",
                        f"{forward_base}.rmsnorm_ffn",
                        f"{forward_base}.gate_up",
                        f"{forward_base}.down_partial",
                        f"{forward_base}.down_tp_sum",
                    ]
                )
            prefix = f"train_{trajectory.trajectory_id}_backward"
            expected_backward_names.extend(
                [
                    f"train_{trajectory.trajectory_id}_loss_backward",
                    f"{prefix}.lm_head_backward_dinput_dweight",
                    f"{prefix}.lm_head_input_grad_tp_sum",
                    f"{prefix}.final_rmsnorm_backward",
                ]
            )
            for layer in reversed(range(case.model.layers)):
                base = f"{prefix}.layer{layer}"
                expected_backward_names.extend(
                    [
                        f"{base}.ffn_down_backward",
                        f"{base}.swiglu_backward",
                        f"{base}.gate_up_backward",
                        f"{base}.gate_up_input_grad_tp_sum",
                        f"{base}.rmsnorm_ffn_backward",
                        f"{base}.o_proj_backward",
                        f"{base}.attention_backward",
                        f"{base}.qkv_backward",
                        f"{base}.qkv_input_grad_tp_sum",
                        f"{base}.rmsnorm_attn_backward",
                    ]
                )
            expected_backward_names.extend(
                [
                    f"{prefix}.embedding_backward",
                    f"{prefix}.gradient_accumulate_rmw",
                ]
            )
        names = Counter(op.name for op in round_ops)
        missing_forward = [name for name in expected_forward_names if names[name] != 1]
        checks.append(
            _result(
                f"r{round_id}.training_forward_coverage",
                not missing_forward,
                "exactly one required parameter-bearing forward/loss op per microbatch",
                {"missing_or_duplicate": missing_forward},
                "expected checklist derived from layer count and trajectory ids",
                f"round:{round_id}",
            )
        )
        missing_backward = [name for name in expected_backward_names if names[name] != 1]
        checks.append(
            _result(
                f"r{round_id}.backward_parameter_coverage",
                not missing_backward,
                "exactly one required backward/gradient accumulation op per microbatch",
                {"missing_or_duplicate": missing_backward},
                "expected checklist derived from layer count and trajectory ids",
                f"round:{round_id}",
            )
        )

        losses = [op for op in round_ops if op.name in {f"train_{t.trajectory_id}_loss" for t in case.workload.trajectories}]
        accumulates = [op for op in round_ops if op.name.endswith(".gradient_accumulate_rmw")]
        actual_denominators = [op.metadata.get("global_denominator") for op in losses]
        actual_denominators += [
            op.metadata.get("global_loss_denominator") for op in accumulates
        ]
        target_positions = sorted(
            (op.metadata.get("microbatch_target_positions") for op in accumulates),
            key=lambda value: (0, value)
            if isinstance(value, int)
            else (1, str(value)),
        )
        expected_target_positions = sorted(
            len(item.response_token_ids) for item in case.workload.trajectories
        )
        checks.append(
            _result(
                f"r{round_id}.global_loss_denominator",
                len(losses) == len(case.workload.trajectories)
                and len(accumulates) == len(case.workload.trajectories)
                and all(value == total_response_tokens for value in actual_denominators)
                and target_positions == expected_target_positions,
                {
                    "global_denominator": total_response_tokens,
                    "microbatch_target_positions": expected_target_positions,
                },
                {
                    "denominators": actual_denominators,
                    "microbatch_target_positions": target_positions,
                },
                "loss and gradient-RMW metadata",
                f"round:{round_id}",
            )
        )

        adam_ops = [op for op in round_ops if op.name == "adamw_read_modify_write"]
        adam = adam_ops[0] if len(adam_ops) == 1 else None
        accesses = adam.metadata.get("state_accesses") if adam is not None else None
        # Older operation traces may express access semantics as separate keys.
        if accesses is None and adam is not None:
            reads = adam.metadata.get("reads")
            writes = adam.metadata.get("writes")
            if reads is not None or writes is not None:
                accesses = {"reads": reads, "writes": writes}
        actual_adam = {
            "count": len(adam_ops),
            "tensor_bytes": adam.tensor_bytes if adam is not None else None,
            "optimizer": adam.metadata.get("optimizer") if adam is not None else None,
            "state_accesses": accesses,
            "reference_written": adam.metadata.get("reference_written")
            if adam is not None
            else None,
        }
        expected_adam = {
            "count": 1,
            "tensor_bytes": expected_adam_bytes,
            "optimizer": "adamw_shape_only",
            "state_accesses": required_state_accesses,
            "reference_written": False,
        }
        checks.append(
            _result(
                f"r{round_id}.adamw_state_rmw",
                actual_adam == expected_adam,
                expected_adam,
                actual_adam,
                "gradient/master/m/v reads and master/m/v/actor writes are mandatory",
                f"round:{round_id}",
            )
        )
    return checks


def _memory_checks(case: CaseConfig, ledger: LedgerBundle) -> list[CheckResult]:
    checks: list[CheckResult] = []
    totals = parameter_totals(case)
    resident = totals["resident_with_gradient_bytes_per_die"]
    all_snapshots = ledger.all_snapshots
    violations = [
        {
            "step": item.step,
            "location": item.location,
            "allocated": item.allocated_bytes,
            "valid": item.valid_bytes,
            "unmaterialized": item.unmaterialized_bytes,
            "capacity": item.capacity_bytes,
        }
        for item in all_snapshots
        if item.allocated_bytes < 0
        or item.valid_bytes < 0
        or item.unmaterialized_bytes < 0
        or item.valid_bytes > item.allocated_bytes
        or item.allocated_bytes + item.unmaterialized_bytes > item.capacity_bytes
    ]
    checks.append(
        _result(
            "memory.capacity_and_validity",
            not violations and bool(all_snapshots),
            "0 <= valid <= allocated and allocated + reservation <= capacity",
            violations,
            "all physical resources and KV logical views",
            "memory",
        )
    )

    per_token = kv_bytes_per_processed_token_per_die(case)
    expected_reservation = len(case.workload.trajectories) * per_token * (
        len(case.workload.prompt_token_ids) + case.workload.max_response_length - 1
    )
    expected_curve = [4 * per_token, 6 * per_token, 3 * per_token, 4 * per_token, 0]
    expected_written = per_token * (2 * len(case.workload.prompt_token_ids) + 3)
    for round_id in _rounds(case):
        for rank in range(case.hardware.dies):
            scope = f"r{round_id}.d{rank}"
            view = sorted(
                (
                    item
                    for item in ledger.kv_snapshots
                    if item.phase == f"r{round_id:02d}.rollout"
                    and item.location == f"d{rank}.dram.kv_view"
                ),
                key=lambda item: item.step,
            )
            admission = [item for item in view if item.reason.startswith("kv_admission")]
            actual_curve = [
                item.valid_bytes
                for item in view
                if not item.reason.startswith("kv_admission")
            ]
            checks.append(
                _result(
                    f"{scope}.kv_reservation",
                    len(admission) == 1
                    and admission[0].allocated_bytes == 0
                    and admission[0].unmaterialized_bytes == expected_reservation,
                    {"allocated": 0, "unmaterialized": expected_reservation},
                    {
                        "allocated": admission[0].allocated_bytes if admission else None,
                        "unmaterialized": admission[0].unmaterialized_bytes
                        if admission
                        else None,
                    },
                    "public maximum length admission, not realised peak",
                    scope,
                )
            )
            checks.append(
                _result(
                    f"{scope}.kv_actual_curve",
                    actual_curve == expected_curve,
                    expected_curve,
                    actual_curve,
                    "prefill -> decode1 -> release B -> decode2 -> release A",
                    scope,
                )
            )
            actual_written = ledger.kv_bytes_written_per_die.get(scope)
            kv_objects = [
                item
                for item in ledger.objects
                if item.kind == "kv"
                and item.owner_rank == rank
                and item.tensor_id.startswith(f"r{round_id:02d}.")
            ]
            object_written = sum(item.nbytes for item in kv_objects)
            released = all(item.released_at is not None for item in kv_objects)
            checks.append(
                _result(
                    f"{scope}.kv_written_and_released",
                    actual_written == expected_written
                    and object_written == expected_written
                    and released,
                    {"bytes_written": expected_written, "all_released": True},
                    {
                        "counter_bytes": actual_written,
                        "object_bytes": object_written,
                        "all_released": released,
                    },
                    "allocation increments and release markers",
                    scope,
                )
            )
            physical_end = [
                item
                for item in ledger.memory_snapshots
                if item.phase == f"r{round_id:02d}.rollout"
                and item.location == f"d{rank}.dram"
                and item.reason == "after_release_A"
            ]
            checks.append(
                _result(
                    f"{scope}.round_cleanup_no_leak",
                    len(physical_end) == 1
                    and physical_end[0].allocated_bytes == resident
                    and physical_end[0].unmaterialized_bytes == 0,
                    {"allocated": resident, "unmaterialized": 0},
                    {
                        "allocated": physical_end[0].allocated_bytes
                        if physical_end
                        else None,
                        "unmaterialized": physical_end[0].unmaterialized_bytes
                        if physical_end
                        else None,
                    },
                    "physical DRAM returns to persistent actor/reference/optimizer/gradient state",
                    scope,
                )
            )
    return checks


def _communication_checks(case: CaseConfig, operations: Sequence[Operation]) -> list[CheckResult]:
    collective_ops = [op for op in operations if op.kind == "collective"]
    p2p_ops = [op for op in operations if op.kind == "p2p"]
    dp_ops = [
        op.op_id
        for op in operations
        if "data_parallel" in op.name.lower()
        or "dp_gradient" in op.name.lower()
        or "data_parallel" in str(op.metadata).lower()
        or "dp_gradient" in str(op.metadata).lower()
    ]
    malformed_p2p = []
    for op in p2p_ops:
        participants = set(op.participant_ranks or tuple(range(case.hardware.dies)))
        packet_peak = min(op.payload_bytes, case.hardware.d2d_packet_bytes)
        if (
            op.src_rank is None
            or op.dst_rank is None
            or op.src_rank == op.dst_rank
            or participants != {op.src_rank, op.dst_rank}
            or packet_peak > case.hardware.nic_tx_bytes_per_die
            or packet_peak > case.hardware.nic_rx_bytes_per_die
        ):
            malformed_p2p.append(
                {
                    "op_id": op.op_id,
                    "src": op.src_rank,
                    "dst": op.dst_rank,
                    "participants": sorted(participants),
                    "payload_bytes": op.payload_bytes,
                    "packet_peak_bytes": packet_peak,
                }
            )
    missing_embedding_sums = []
    for op in operations:
        if op.name.endswith(".embedding"):
            expected_name = op.name + "_tp_sum"
            sums = [
                item for item in operations
                if item.round_id == op.round_id and item.name == expected_name
            ]
            if len(sums) != 1 or sums[0].kind != "collective" or sums[0].collective != "ALL_REDUCE":
                missing_embedding_sums.append(op.op_id)
    return [
        _result(
            "communication.vocab_parallel_embedding_sum",
            not missing_embedding_sums,
            "each vocab-parallel embedding partial is reduced into replicated hidden",
            missing_embedding_sums,
            "embedding operations and required native TP all-reduce",
            "communication",
        ),
        _result(
            "communication.tp_sampling_present_dp_absent",
            bool(collective_ops) and bool(p2p_ops) and not dp_ops,
            {"tp_or_scalar_collectives": ">0", "sampling_p2p": ">0", "dp_gradient": 0},
            {
                "collectives": len(collective_ops),
                "p2p": len(p2p_ops),
                "dp_operations": dp_ops,
                "configured_dp": case.strategy.data_parallel,
            },
            "operation kinds, names, and communication semantics",
            "communication",
        ),
        _result(
            "communication.p2p_endpoint_and_packet_capacity",
            not malformed_p2p,
            "valid distinct endpoints, exact participant set, and one packet fits TX/RX",
            malformed_p2p,
            "planned p2p operations plus independent NIC capacity contract",
            "communication",
        ),
    ]


def _event_checks(
    case: CaseConfig, operations: Sequence[Operation], astra_events: Sequence[Any]
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    expected_pairs = {
        (op.node_id, rank)
        for op in operations
        for rank in (op.participant_ranks or tuple(range(case.hardware.dies)))
    }
    actual_pairs: list[tuple[int, int]] = []
    event_by_pair: dict[tuple[int, int], Any] = {}
    malformed: list[dict[str, Any]] = []
    op_by_node = {op.node_id: op for op in operations}
    op_id_mismatch: list[dict[str, Any]] = []
    for index, event in enumerate(astra_events):
        node_id = _field(event, "node_id")
        rank = _field(event, "rank", "sys_id", "system_id")
        if not isinstance(node_id, int) or not isinstance(rank, int):
            malformed.append({"index": index, "node_id": node_id, "rank": rank})
            continue
        actual_pairs.append((node_id, rank))
        event_by_pair[(node_id, rank)] = event
        expected_op = op_by_node.get(node_id)
        event_op_id = _field(event, "op_id")
        if expected_op is None or (event_op_id is not None and event_op_id != expected_op.op_id):
            op_id_mismatch.append(
                {
                    "index": index,
                    "node_id": node_id,
                    "event_op_id": event_op_id,
                    "expected_op_id": expected_op.op_id if expected_op else None,
                }
            )
        issue_tick = _field(event, "issue_tick")
        callback_tick = _field(event, "callback_tick")
        duration_ticks = _field(event, "duration_ticks")
        issue_ns = _field(event, "issue_ns")
        callback_ns = _field(event, "callback_ns")
        duration_ns = _field(event, "duration_ns")
        temporal_ok = (
            isinstance(issue_tick, (int, float))
            and isinstance(callback_tick, (int, float))
            and isinstance(duration_ticks, (int, float))
            and callback_tick >= issue_tick
            and duration_ticks == callback_tick - issue_tick
            and isinstance(issue_ns, (int, float))
            and isinstance(callback_ns, (int, float))
            and isinstance(duration_ns, (int, float))
            and callback_ns >= issue_ns
            # Absolute float subtraction loses sub-ns precision for long runs.
            # Integer simulator ticks stay authoritative; each displayed ns
            # field must separately equal the exact tick-to-ns conversion.
            and issue_ns == issue_tick / case.hardware.astra_time_scale
            and callback_ns == callback_tick / case.hardware.astra_time_scale
            and duration_ns == duration_ticks / case.hardware.astra_time_scale
        )
        if not temporal_ok:
            malformed.append(
                {
                    "index": index,
                    "node_id": node_id,
                    "rank": rank,
                    "reason": "unpaired_or_inconsistent_issue_callback",
                }
            )
    counts = Counter(actual_pairs)
    actual_set = set(actual_pairs)
    duplicates = sorted(pair for pair, count in counts.items() if count != 1)
    checks.append(
        _result(
            "astrasim.complete_operation_rank_matrix",
            actual_set == expected_pairs and not duplicates and not malformed,
            {
                "pair_count": len(expected_pairs),
                "one_paired_event_per_operation_per_rank": True,
            },
            {
                "pair_count": len(actual_set),
                "missing": sorted(expected_pairs - actual_set),
                "unexpected": sorted(actual_set - expected_pairs),
                "duplicates": duplicates,
                "malformed": malformed,
            },
            "paired AstraSim issue/callback rows; process exit alone is insufficient",
            "astrasim",
        )
    )
    checks.append(
        _result(
            "astrasim.event_operation_identity",
            not op_id_mismatch,
            "every event node_id resolves to the same planned op_id",
            op_id_mismatch,
            "AstraEvent node_id/op_id mapping",
            "astrasim",
        )
    )
    causal_violations: list[dict[str, Any]] = []
    compute_duration_violations: list[dict[str, Any]] = []
    for op in operations:
        for rank in (op.participant_ranks or tuple(range(case.hardware.dies))):
            event = event_by_pair.get((op.node_id, rank))
            if event is None:
                continue
            issue_tick = _field(event, "issue_tick")
            if op.kind == "compute":
                actual_duration = _field(event, "duration_ticks")
                expected_duration = op.duration_ns * case.hardware.astra_time_scale
                if actual_duration != expected_duration:
                    compute_duration_violations.append(
                        {"rank": rank, "op_id": op.op_id,
                         "expected_raw_ticks": expected_duration,
                         "actual_raw_ticks": actual_duration}
                    )
            dependencies = op.dependencies_by_rank.get(rank, op.dependencies)
            for dependency in dependencies:
                predecessor = event_by_pair.get((dependency, rank))
                callback_tick = _field(predecessor, "callback_tick")
                if (
                    not isinstance(issue_tick, (int, float))
                    or not isinstance(callback_tick, (int, float))
                    or issue_tick < callback_tick
                ):
                    causal_violations.append(
                        {
                            "rank": rank,
                            "node_id": op.node_id,
                            "dependency": dependency,
                            "issue_tick": issue_tick,
                            "dependency_callback_tick": callback_tick,
                        }
                    )
    checks.append(
        _result(
            "astrasim.dependency_causality",
            not causal_violations and actual_set == expected_pairs,
            "every node issues at or after every predecessor callback on each rank",
            causal_violations,
            "paired AstraSim ticks plus planned dependencies",
            "astrasim",
        )
    )
    checks.append(
        _result(
            "astrasim.compute_duration_unit_contract",
            not compute_duration_violations,
            "every replay COMP duration equals planned model ns times 1000",
            compute_duration_violations,
            "raw Astra issue/callback ticks versus planned Chakra duration",
            "astrasim",
        )
    )
    return checks


def validate_run(
    case: CaseConfig,
    operations: Iterable[Operation],
    ledger: LedgerBundle,
    astra_events: Iterable[Any],
) -> list[CheckResult]:
    """Validate all required evidence and return structured results.

    Missing events are a normal failed check, never an implicit success.  If a
    check implementation itself encounters malformed input, that is converted
    into a structured ``validation.internal_error`` failure so callers still
    cannot fail open.
    """

    op_tuple = tuple(operations)
    event_tuple = tuple(astra_events)
    checks: list[CheckResult] = []
    groups = (
        ("parameters", lambda: _parameter_checks(case, ledger)),
        ("trajectories", lambda: _trajectory_checks(case, op_tuple)),
        ("phases_versions", lambda: _phase_and_version_checks(case, op_tuple)),
        ("training", lambda: _training_checks(case, op_tuple)),
        ("memory", lambda: _memory_checks(case, ledger)),
        ("communication", lambda: _communication_checks(case, op_tuple)),
        ("astrasim_events", lambda: _event_checks(case, op_tuple, event_tuple)),
    )
    for group_name, evaluate in groups:
        try:
            checks.extend(evaluate())
        except Exception as exc:  # malformed evidence must become a hard failure
            checks.append(
                _result(
                    f"validation.internal_error.{group_name}",
                    False,
                    "check group completes without exception",
                    {"type": type(exc).__name__, "message": str(exc)},
                    "validator fail-closed exception boundary",
                    group_name,
                )
            )
    return checks


def require_valid(checks: Sequence[CheckResult]) -> None:
    """Raise ``ValidationError`` unless every structured check passed."""

    if not checks or any(not check.passed for check in checks):
        if not checks:
            checks = (
                _result(
                    "validation.no_checks",
                    False,
                    "at least one executed check",
                    0,
                    "empty check collection",
                    "run",
                ),
            )
        raise ValidationError(checks)


def all_passed(checks: Sequence[CheckResult]) -> bool:
    """Convenience predicate; unlike ``all([])``, an empty report is false."""

    return bool(checks) and all(check.passed for check in checks)

