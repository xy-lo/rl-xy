"""End-to-end orchestration for the transfer fixture and toy GRPO round."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .astra_adapter import (
    generate_traces,
    parse_astra_output,
    run_astrasim,
    validate_astra_configs,
)
from .config import load_case
from .fixture import (
    TransferFixture,
    build_transfer_fixture,
    validate_63b_negative,
    validate_transfer_fixture,
)
from .reporting import (
    build_toy_metrics,
    copy_inputs,
    phase_metrics,
    toy_parameters,
    write_json,
    write_manifest,
    write_summary_markdown,
    write_tables,
    write_csv,
)
from .types import (
    AstraEvent,
    CheckResult,
    MemorySnapshot,
    ObjectRecord,
    Operation,
    TransferRecord,
)
from .workflow import build_workflow


class RunError(RuntimeError):
    """A run failed before an all-pass evidence bundle could be produced."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _prepare_output(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists():
        raise RunError(f"refusing to overwrite existing run directory: {output}")
    output.mkdir(parents=True)
    write_json(
        output / "status.json",
        {"status": "running", "started_at_utc": _utc_now()},
    )
    return output


def _fail(output: Path, exc: BaseException) -> None:
    previous = {}
    try:
        previous = json.loads((output / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    failure = {
        "status": "failed",
        "started_at_utc": previous.get("started_at_utc"),
        "failed_at_utc": _utc_now(),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    write_json(output / "status.json", failure)
    execution = getattr(exc, "result", None)
    if execution is not None:
        write_json(
            output / "astra_command.json",
            {"argv": list(execution.command), "exit_status": execution.return_code},
        )
    write_json(output / "summary.json", {**failure, "status": "FAIL"})


def _success(output: Path) -> None:
    previous = json.loads((output / "status.json").read_text(encoding="utf-8"))
    write_json(
        output / "status.json",
        {
            "status": "success",
            "started_at_utc": previous.get("started_at_utc"),
            "finished_at_utc": _utc_now(),
        },
    )


def _transfers(operations: Iterable[Operation]) -> list[TransferRecord]:
    result: list[TransferRecord] = []
    for op in operations:
        if op.kind == "p2p":
            result.append(
                TransferRecord(
                    round_id=op.round_id,
                    op_id=op.op_id,
                    phase=op.phase,
                    transfer_type="p2p",
                    src=f"rank{op.src_rank}",
                    dst=f"rank{op.dst_rank}",
                    route=str(op.metadata.get("route", "direct_d2d")),
                    payload_bytes=op.payload_bytes,
                    node_id=op.node_id,
                )
            )
        elif op.kind == "collective":
            result.append(
                TransferRecord(
                    round_id=op.round_id,
                    op_id=op.op_id,
                    phase=op.phase,
                    transfer_type=op.collective or "collective",
                    src="tp_group",
                    dst="tp_group",
                    route="astra_native_collective",
                    payload_bytes=op.payload_bytes,
                    node_id=op.node_id,
                )
            )
    return result


def _check_or_fail(output: Path, checks: list[CheckResult]) -> None:
    failed = [row.check_id for row in checks if not row.passed]
    write_json(output / "checks.json", {"checks": [asdict(row) for row in checks]})
    if failed:
        raise RunError(f"semantic checks failed: {failed}")


def _write_summary_json(
    output: Path,
    metrics: dict[str, Any],
    checks: list[CheckResult],
    *,
    evidence_kind: str,
) -> None:
    write_json(
        output / "summary.json",
        {
            "status": "PASS",
            "evidence_kind": evidence_kind,
            "metric_scope": metrics.get("metric_scope"),
            "checks_passed": sum(row.passed for row in checks),
            "checks_failed": sum(not row.passed for row in checks),
            "metrics": metrics,
        },
    )


def _fixture_metrics(
    fixture: TransferFixture,
    operations: list[Operation],
    events: list[AstraEvent],
) -> dict[str, Any]:
    total_ns = max(event.callback_ns for event in events)
    sends = [event for event in events if event.node_id == 5 and event.rank == 0]
    receives = [event for event in events if event.node_id == 5 and event.rank == 1]
    return {
        "metric_scope": "synthetic_fixture_not_hardware_calibrated",
        "total_simulated_time_ns": total_ns,
        "phase_timing": phase_metrics(events),
        "operation_count": len(operations),
        "operation_rank_event_count": len(events),
        "collective_logical_payload_bytes": 0,
        "p2p_logical_payload_bytes": sum(
            item.payload_bytes for item in operations if item.kind == "p2p"
        ),
        "p2p_send_service_ns": sends[0].duration_ns if len(sends) == 1 else None,
        "p2p_receive_occupancy_ns": (
            receives[0].duration_ns if len(receives) == 1 else None
        ),
        "dram_traffic_bytes": fixture.config["expected_dram_traffic_bytes"],
        "sram_peak_bytes": max(
            item.allocated_bytes for item in fixture.memory_snapshots
        ),
    }


def run_transfer_fixture(case_path: str | Path, output_path: str | Path) -> Path:
    fixture = build_transfer_fixture(case_path)
    output = _prepare_output(output_path)
    try:
        copied = copy_inputs(
            fixture.source_path.parent,
            output,
            ("case.json", "astra_system.json", "astra_network.yml", "astra_remote_memory.json"),
        )
        adapter_contract = validate_astra_configs(
            system_path=copied["astra_system.json"],
            network_path=copied["astra_network.yml"],
            remote_memory_path=copied["astra_remote_memory.json"],
            ranks=2,
            astra_time_scale=fixture.config["astra_time_scale"],
            d2d_bytes_per_model_ns=fixture.config["d2d_bytes_per_ns"],
            d2d_latency_model_ns=fixture.config["d2d_latency_ns"],
        )
        write_json(output / "astra_config_contract.json", adapter_contract)
        write_json(output / "resolved_config.json", fixture.config)
        operations = list(fixture.operations)
        trace_prefix = output / "inputs/workload"
        traces = generate_traces(
            operations,
            trace_prefix,
            ranks=2,
            astra_time_scale=fixture.config["astra_time_scale"],
        )
        copied.update({f"workload_rank{rank}": path for rank, path in enumerate(traces)})
        execution = run_astrasim(
            trace_prefix=trace_prefix,
            system_config=copied["astra_system.json"],
            network_config=copied["astra_network.yml"],
            remote_memory_config=copied["astra_remote_memory.json"],
            output_dir=output,
        )
        write_json(
            output / "astra_command.json",
            {"argv": list(execution.command), "exit_status": execution.return_code},
        )
        events, summaries = parse_astra_output(
            execution,
            operations,
            ranks=2,
            astra_time_scale=fixture.config["astra_time_scale"],
        )
        total_ns = max(row.wall_cycles for row in summaries) / fixture.config[
            "astra_time_scale"
        ]
        p2p_events = [row for row in events if row.node_id == 5 and row.rank == 0]
        p2p_duration = p2p_events[0].duration_ns if len(p2p_events) == 1 else -1.0
        checks = validate_transfer_fixture(fixture, total_ns, p2p_duration)
        checks.append(validate_63b_negative(case_path))
        expected_event_rows = sum(
            len(op.participant_ranks or (0, 1)) for op in operations
        )
        checks.append(
            CheckResult(
                "astra_event_coverage",
                len(events) == expected_event_rows,
                expected_event_rows,
                len(events),
                "strict parser paired issue/callback for every rank and node",
                "astrasim",
            )
        )
        write_tables(
            output,
            operations=operations,
            events=events,
            summaries=summaries,
            objects=fixture.objects,
            snapshots=fixture.memory_snapshots,
            transfers=_transfers(operations),
            checks=checks,
        )
        _check_or_fail(output, checks)
        metrics = _fixture_metrics(fixture, operations, events)
        write_json(output / "metrics.json", metrics)
        _write_summary_json(
            output,
            metrics,
            checks,
            evidence_kind=fixture.config["evidence_kind"],
        )
        write_summary_markdown(
            output,
            title="21 ns 跨 die 资源链路 fixture",
            metrics=metrics,
            checks=checks,
        )
        _success(output)
        write_manifest(
            output,
            case_id=fixture.config["case_id"],
            evidence_kind=fixture.config["evidence_kind"],
            run_kind="transfer_fixture",
            workflow_scope="hand-checkable two-die transfer-chain fixture",
            execution=execution,
            metrics=metrics,
            checks=checks,
            input_paths=copied,
        )
        return output
    except BaseException as exc:
        _fail(output, exc)
        raise


def run_toy_case(
    case_path: str | Path,
    output_path: str | Path,
    *,
    rounds_override: int | None = None,
    local_cost_model: str = "legacy",
    resource_model: bool = False,
) -> Path:
    case = load_case(case_path, rounds_override=rounds_override)
    if case.schema_version == "1.1" and not resource_model:
        raise RunError("schema 1.1 requires --resource-model; no incomplete legacy memory path")
    output = _prepare_output(output_path)
    try:
        copied = copy_inputs(
            case.source_path.parent,
            output,
            ("case.json", "astra_system.json", "astra_network.yml", "astra_remote_memory.json"),
        )
        # Compiled probes keep their original model metadata and selection
        # contract inside each hash-addressed bundle, not only beside the case.
        optional=[name for name in ("model_config.json","compilation.json","H_SCOPE.md")
                  if (case.source_path.parent/name).is_file()]
        if optional:copied.update(copy_inputs(case.source_path.parent,output,optional))
        adapter_contract = validate_astra_configs(
            system_path=copied["astra_system.json"],
            network_path=copied["astra_network.yml"],
            remote_memory_path=copied["astra_remote_memory.json"],
            ranks=case.hardware.dies,
            astra_time_scale=case.hardware.astra_time_scale,
            d2d_bytes_per_model_ns=case.hardware.d2d_bytes_per_ns_per_direction,
            d2d_latency_model_ns=case.hardware.d2d_latency_ns,
        )
        write_json(output / "astra_config_contract.json", adapter_contract)
        write_json(output / "resolved_config.json", case.resolved)
        operations = build_workflow(case)
        plan=None
        if resource_model:
            from .communication_lowering import lower_communication_endpoints
            from .tensor_plan import build_tensor_plan,finalize_tensor_costs
            operations=lower_communication_endpoints(case,operations)
            plan=build_tensor_plan(case,operations,str(output/'astra_logs/resources'))
            finalize_tensor_costs(plan,case)
            operations=plan.operations
            local_cost_model="mapped"
            write_json(output/'resource_contract.json',plan.contract)
            write_json(output/'tensor_specs.json',plan.tensors)
        elif local_cost_model == "mapped":
            from .mapped_costs import apply_mapped_costs
            operations = apply_mapped_costs(case, operations)
        elif local_cost_model != "legacy":
            raise RunError("unknown local cost model")
        if local_cost_model == "mapped":
            write_json(output / "cost_estimates.json", [
                {"op_id": op.op_id, "kernel": asdict(op.kernel_spec) if op.kernel_spec else None,
                 "estimate": op.metadata.get("cost_estimate")}
                for op in operations if op.kind == "compute"
            ])
        # Imported here so the ledger module stays independently unit-testable.
        from .ledger import build_ledger
        from .validation import validate_run

        ledger = build_ledger(case, operations)
        trace_prefix = output / "inputs/workload"
        traces = generate_traces(
            operations,
            trace_prefix,
            ranks=case.hardware.dies,
            astra_time_scale=case.hardware.astra_time_scale,
        )
        copied.update({f"workload_rank{rank}": path for rank, path in enumerate(traces)})
        execution = run_astrasim(
            trace_prefix=trace_prefix,
            system_config=copied["astra_system.json"],
            network_config=copied["astra_network.yml"],
            remote_memory_config=copied["astra_remote_memory.json"],
            output_dir=output,
            resource_contract=output/'resource_contract.json' if resource_model else None,
        )
        write_json(
            output / "astra_command.json",
            {"argv": list(execution.command), "exit_status": execution.return_code},
        )
        events, summaries = parse_astra_output(
            execution,
            operations,
            ranks=case.hardware.dies,
            astra_time_scale=case.hardware.astra_time_scale,
        )
        if case.schema_version == "1.1":
            from .execution_validation import validate_execution
            checks = validate_execution(case, operations, ledger, events)
        else:
            checks = validate_run(case, operations, ledger, events)
        runtime_snapshots=None
        runtime_peaks=None
        if plan is not None:
            from .resource_validation import validate_resources
            extra,runtime_snapshots,runtime_peaks=validate_resources(plan,case,events)
            checks.extend(extra)
        write_tables(
            output,
            operations=operations,
            events=events,
            summaries=summaries,
            objects=ledger.objects,
            snapshots=ledger.all_snapshots,
            transfers=_transfers(operations),
            checks=checks,
            parameters=toy_parameters(case),
        )
        _check_or_fail(output, checks)
        metrics = build_toy_metrics(
            case, operations, events, summaries, ledger.all_snapshots
        )
        metrics["kv_bytes_written_per_die_per_round"] = ledger.kv_bytes_written_per_die
        metrics["local_cost_model"] = local_cost_model
        if case.schema_version == "1.1":
            from .execution_contract import group_for
            metrics["group_count_per_round"]=len({group_for(case,t) for t in case.workload.trajectories})
            from .training_schedule import updates_per_round
            metrics["execution_scope"]=(f'Qwen2-shape synchronous replay; TP2, {case.hardware.cores_per_die} private cores/die, '
                f'one trajectory per microbatch, {updates_per_round(case)} updates/round; not numerical training or calibrated H')
        if plan is not None:
            write_csv(output/'legacy_memory.csv',ledger.all_snapshots)
            write_csv(output/'memory.csv',runtime_snapshots)
            write_csv(output/'tensor_objects.csv',plan.tensors)
            metrics['memory_peak_bytes_by_location']=runtime_peaks
            metrics['tensor_storage_count']=len(plan.tensors)
            metrics['memory_accounting_scope']={
                'dram':'persistent states + declared out-of-place forward tape, backward products, trajectory/KV and communication outputs',
                'retention_policy':'conservative: forward invocation for inference; full microbatch tape/partials until gradient accumulation',
                'native_collective_internal_buffers':'not exposed by Astra; excluded',
                'allocator_fragmentation':'not modeled',
                'activation_tape_included':True,'local_dma_timing_in_toy':True,
                'sram_workspace':'analytic OS tiles and 16B vector streaming workspace; checked at runtime',
                'event_source':'actual Astra issue/callback hooks with independent replay',
                'not_claimed':'hardware-calibrated full-device peak or fine-grained bank/NoC contention'}
            metrics['resource_model']='astra_callback_admission_v1'
            metrics['local_dram_read_bytes']=sum(o.metadata.get('cost_estimate',{}).get('dram_read_bytes',0)*len(o.participant_ranks) for o in operations)
            metrics['local_dram_write_bytes']=sum(o.metadata.get('cost_estimate',{}).get('dram_write_bytes',0)*len(o.participant_ranks) for o in operations)
        metrics["local_cost_scope"] = (
            "analytic OS GEMM + conservative streaming vector traffic; serial local DMA/compute; not calibrated"
            if local_cost_model == "mapped" else "legacy synthetic FLOP/vector-only costs"
        )
        write_json(output / "metrics.json", metrics)
        _write_summary_json(
            output,
            metrics,
            checks,
            evidence_kind=case.evidence_kind,
        )
        write_summary_markdown(
            output,
            title="两 die TP=2 同步 GRPO 算子回放",
            metrics=metrics,
            checks=checks,
        )
        _success(output)
        write_manifest(
            output,
            case_id=case.case_id,
            evidence_kind=case.evidence_kind,
            run_kind="toy_grpo",
            workflow_scope="full synthetic synchronous-GRPO control/data-flow replay",
            execution=execution,
            metrics=metrics,
            checks=checks,
            input_paths=copied,
        )
        return output
    except BaseException as exc:
        _fail(output, exc)
        raise
