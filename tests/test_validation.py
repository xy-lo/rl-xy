from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from rl_xy.config import load_case
from rl_xy.ledger import build_ledger
from rl_xy.validation import ValidationError, all_passed, require_valid, validate_run
from rl_xy.workflow import build_workflow


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "configs" / "toy_grpo_2die" / "case.json"


def paired_events(case, operations):
    events = []
    for operation in operations:
        participants = operation.participant_ranks or tuple(
            range(case.hardware.dies)
        )
        for rank in participants:
            issue = operation.node_id * 16_000_000
            duration = (
                operation.duration_ns * case.hardware.astra_time_scale
                if operation.kind == "compute"
                else case.hardware.astra_time_scale
            )
            events.append(
                {
                    "rank": rank,
                    "node_id": operation.node_id,
                    "op_id": operation.op_id,
                    "phase": operation.phase,
                    "node_type": operation.kind,
                    "issue_tick": issue,
                    "callback_tick": issue + duration,
                    "duration_ticks": duration,
                    "issue_ns": issue / case.hardware.astra_time_scale,
                    "callback_ns": (issue + duration) / case.hardware.astra_time_scale,
                    "duration_ns": duration / case.hardware.astra_time_scale,
                }
            )
    return events


class ValidationTests(unittest.TestCase):
    def test_large_absolute_ticks_preserve_exact_integer_contract(self):
        from rl_xy.validation import _event_checks
        case=load_case(CASE)
        op=build_workflow(case)[0]
        events=paired_events(case,[op])
        for event in events:
            event['issue_tick']=2962804633776
            event['callback_tick']=event['issue_tick']+event['duration_ticks']
            event['issue_ns']=event['issue_tick']/1000
            event['callback_ns']=event['callback_tick']/1000
        checks=_event_checks(case,[op],events)
        self.assertTrue(all(x.passed for x in checks))
        events[0]['duration_ns']+=.001
        self.assertFalse(all(x.passed for x in _event_checks(case,[op],events)))

    def _evidence(self, rounds=1):
        case = load_case(CASE, rounds_override=rounds)
        operations = build_workflow(case)
        ledger = build_ledger(case, operations)
        events = paired_events(case, operations)
        return case, operations, ledger, events

    def test_complete_two_round_evidence_passes(self) -> None:
        case, operations, ledger, events = self._evidence(rounds=2)
        checks = validate_run(case, operations, ledger, events)
        failed = [(item.check_id, item.actual) for item in checks if not item.passed]
        self.assertTrue(all_passed(checks), failed)
        require_valid(checks)

    def test_missing_astrasim_event_is_structured_failure(self) -> None:
        case, operations, ledger, events = self._evidence()
        events.pop()
        checks = validate_run(case, operations, ledger, events)
        result = next(
            item
            for item in checks
            if item.check_id == "astrasim.complete_operation_rank_matrix"
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.actual["missing"])
        with self.assertRaises(ValidationError):
            require_valid(checks)

    def test_missing_backward_is_not_masked_by_valid_event_dag(self) -> None:
        case, operations, _ledger, _events = self._evidence()
        operations = [op for op in operations if not op.name.endswith(".qkv_backward")]
        ledger = build_ledger(case, operations)
        checks = validate_run(case, operations, ledger, paired_events(case, operations))
        result = next(
            item
            for item in checks
            if item.check_id == "r0.backward_parameter_coverage"
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.actual["missing_or_duplicate"])

    def test_missing_adam_state_access_is_detected(self) -> None:
        case, operations, _ledger, _events = self._evidence()
        changed = []
        for operation in operations:
            if operation.name == "adamw_read_modify_write":
                metadata = dict(operation.metadata)
                metadata["state_accesses"] = {
                    "reads": ["gradient", "master", "adam_m"],
                    "writes": ["master", "adam_m", "actor"],
                }
                operation = replace(operation, metadata=metadata)
            changed.append(operation)
        ledger = build_ledger(case, changed)
        checks = validate_run(case, changed, ledger, paired_events(case, changed))
        result = next(item for item in checks if item.check_id == "r0.adamw_state_rmw")
        self.assertFalse(result.passed)

    def test_capacity_violation_is_structured_failure(self) -> None:
        case, operations, ledger, events = self._evidence()
        snapshots = list(ledger.memory_snapshots)
        snapshots[0] = replace(snapshots[0], capacity_bytes=1)
        ledger = replace(ledger, memory_snapshots=tuple(snapshots))
        checks = validate_run(case, operations, ledger, events)
        result = next(
            item for item in checks if item.check_id == "memory.capacity_and_validity"
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.actual)

    def test_deleted_dependency_is_detected_even_with_complete_events(self) -> None:
        case, operations, _ledger, _events = self._evidence()
        operations = list(operations)
        operations[-1] = replace(operations[-1], dependencies=())
        ledger = build_ledger(case, operations)
        checks = validate_run(case, operations, ledger, paired_events(case, operations))
        result = next(
            item
            for item in checks
            if item.check_id == "workflow.strict_total_order_dependency_chain"
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.actual["broken_dependencies"])

    def test_empty_check_report_fails_closed(self) -> None:
        self.assertFalse(all_passed([]))
        with self.assertRaises(ValidationError):
            require_valid([])

    def test_embedding_partial_sum_cannot_be_deleted(self) -> None:
        case, operations, _, _ = self._evidence()
        operations = [
            op for op in operations
            if op.name != "rollout_A_prefill.embedding_tp_sum"
        ]
        checks = validate_run(
            case, operations, build_ledger(case, operations),
            paired_events(case, operations),
        )
        check = next(row for row in checks if row.check_id == "communication.vocab_parallel_embedding_sum")
        self.assertFalse(check.passed)

    def test_each_round_requires_gradient_reset(self) -> None:
        case, operations, _, _ = self._evidence(rounds=2)
        operations = [
            op for op in operations
            if not (op.round_id == 1 and op.name == "zero_gradient_buffer")
        ]
        checks = validate_run(
            case, operations, build_ledger(case, operations),
            paired_events(case, operations),
        )
        check = next(row for row in checks if row.check_id == "r1.gradient_reset_before_accumulation")
        self.assertFalse(check.passed)

    def test_wrong_raw_compute_duration_is_rejected(self) -> None:
        case, operations, ledger, events = self._evidence()
        event = events[0]
        event["duration_ticks"] += 1000
        event["callback_tick"] += 1000
        event["duration_ns"] += 1
        event["callback_ns"] += 1
        checks = validate_run(case, operations, ledger, events)
        check = next(row for row in checks if row.check_id == "astrasim.compute_duration_unit_contract")
        self.assertFalse(check.passed)


if __name__ == "__main__":
    unittest.main()

