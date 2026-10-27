"""Write a self-contained, hash-addressed evidence bundle for one run."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model import enumerate_parameters, parameter_totals
from .training_schedule import updates_per_round, minibatches
from .types import (
    AstraEvent,
    AstraRankSummary,
    AstraRunResult,
    CaseConfig,
    CheckResult,
    MemorySnapshot,
    ObjectRecord,
    Operation,
    ParameterRow,
    TransferRecord,
)
from .astra_adapter.runner import sha256_file
from . import __version__


class ReportingError(RuntimeError):
    """The evidence bundle could not be completed without ambiguity."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _flatten_row(value: Any) -> dict[str, Any]:
    row = asdict(value) if is_dataclass(value) else dict(value)
    for key, item in list(row.items()):
        if isinstance(item, (dict, list, tuple)):
            row[key] = json.dumps(item, ensure_ascii=False, sort_keys=True)
        elif isinstance(item, Path):
            row[key] = str(item)
    return row


def write_csv(path: str | Path, rows: Iterable[Any], fieldnames: list[str] | None = None) -> None:
    target = Path(path)
    values = [_flatten_row(row) for row in rows]
    if fieldnames is None:
        fieldnames = list(values[0]) if values else []
    with target.open("w", encoding="utf-8", newline="") as stream:
        if not fieldnames:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            value = asdict(row) if is_dataclass(row) else dict(row)
            stream.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )


def copy_inputs(case_dir: Path, output_dir: Path, names: Iterable[str]) -> dict[str, Path]:
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for name in names:
        source = (case_dir / name).resolve()
        if not source.is_file():
            raise ReportingError(f"missing required input {source}")
        target = input_dir / name
        shutil.copy2(source, target)
        copied[name] = target
    return copied


def deterministic_tree_hash(root: Path) -> str:
    # Reports embed the hash itself. Exclude these derived artifacts to
    # avoid a circular provenance definition; their contents are not runtime inputs.
    generated_reports = {
        "docs/工程文件与完整流程.md",
        "docs/完整流程预览.html",
        "docs/资源版工程说明与完整流程.md",
        "docs/资源版完整流程预览.html",
    }
    digest = hashlib.sha256()
    included: list[Path] = []
    for relative in (
        ".dockerignore",
        ".gitignore",
        "LICENSE",
        "Makefile",
        "README.md",
        "compose.yaml",
        "pyproject.toml",
        "configs",
        "docker",
        "docs",
        "examples",
        "scripts",
        "src",
        "tests",
    ):
        path = root / relative
        if path.is_file():
            included.append(path)
        elif path.is_dir():
            included.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts
            )
    for path in sorted(included):
        relative_name = path.relative_to(root).as_posix()
        if relative_name in generated_reports:
            continue
        relative = relative_name.encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _mem_total_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def phase_metrics(events: Iterable[AstraEvent]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[AstraEvent]] = {}
    for event in events:
        buckets.setdefault(f"r{event.round_id:02d}.{event.phase}", []).append(event)
    result: dict[str, dict[str, float]] = {}
    for phase, rows in buckets.items():
        start = min(row.issue_ns for row in rows)
        end = max(row.callback_ns for row in rows)
        result[phase] = {
            "start_ns": start,
            "end_ns": end,
            "span_ns": end - start,
            "operation_rank_events": len(rows),
        }
    return result


def build_toy_metrics(
    case: CaseConfig,
    operations: list[Operation],
    events: list[AstraEvent],
    summaries: list[AstraRankSummary],
    snapshots: Iterable[MemorySnapshot],
) -> dict[str, Any]:
    total_ticks = max(item.wall_cycles for item in summaries)
    scale = case.hardware.astra_time_scale
    peaks: dict[str, int] = {}
    for snapshot in snapshots:
        if snapshot.reason == "independent_resource_visible":
            continue
        peaks[snapshot.location] = max(
            peaks.get(snapshot.location, 0), snapshot.allocated_bytes
        )
    response_tokens = sum(
        len(item.response_token_ids) for item in case.workload.trajectories
    )
    return {
        "metric_scope": "synthetic_fixture_not_hardware_calibrated",
        "total_simulated_time_ns": total_ticks / scale,
        "rounds": case.rounds,
        "phase_timing": phase_metrics(events),
        "operation_count": len(operations),
        "operation_rank_event_count": len(events),
        "compute_operation_count": sum(op.kind == "compute" for op in operations),
        "collective_operation_count": sum(op.kind == "collective" for op in operations),
        "p2p_operation_count": sum(op.kind == "p2p" for op in operations),
        "collective_logical_payload_bytes": sum(
            op.payload_bytes for op in operations if op.kind == "collective"
        ),
        "p2p_logical_payload_bytes": sum(
            op.payload_bytes for op in operations if op.kind == "p2p"
        ),
        "response_tokens_per_round": response_tokens,
        "global_loss_denominator": response_tokens if updates_per_round(case) == 1 else None,
        "optimizer_updates": case.rounds * updates_per_round(case),
        "training_epochs": case.strategy.training_epochs,
        "minibatch_loss_denominators": [sum(len(t.response_token_ids) for t in b) for b in minibatches(case)],
        "actor_version_start": case.workload.behavior_version,
        "actor_version_end": case.workload.behavior_version + case.rounds * updates_per_round(case),
        "reference_version_start": case.workload.behavior_version,
        "reference_version_end": case.workload.behavior_version,
        "parameter_accounting": parameter_totals(case),
        "memory_peak_bytes_by_location": peaks,
        "memory_accounting_scope": {
            "dram": "persistent parameter/optimizer/gradient states plus rollout KV only",
            "kv_view": "logical view of the same KV storage; never add to physical DRAM",
            "nic": "planned p2p packet peak; native collective buffers excluded",
            "activation_tape_included": False,
            "sram_workspace_peak_bytes": None,
            "local_dma_timing_in_toy": False,
            "note": "tracked DRAM subtotal is not whole-step or full-device peak memory",
        },
        "rank_timing": [
            {
                "rank": row.rank,
                "wall_ns": row.wall_cycles / scale,
                "astra_compute_lane_busy_ns": row.gpu_cycles / scale,
                "astra_comm_node_occupied_ns": row.communication_cycles / scale,
                "exposed_communication_ns": row.exposed_communication_cycles / scale,
                "astra_compute_lane_busy_ratio": row.gpu_cycles / row.wall_cycles,
                "astra_comm_node_occupied_ratio": row.communication_cycles / row.wall_cycles,
                "timing_note": "compute lane includes synthetic host/local-control replay; comm occupancy may include posted receive wait",
            }
            for row in summaries
        ],
    }


def _artifact_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    return hashes


def write_tables(
    output_dir: Path,
    *,
    operations: Iterable[Operation],
    events: Iterable[AstraEvent],
    summaries: Iterable[AstraRankSummary],
    objects: Iterable[ObjectRecord],
    snapshots: Iterable[MemorySnapshot],
    transfers: Iterable[TransferRecord],
    checks: Iterable[CheckResult],
    parameters: Iterable[Any] = (),
) -> None:
    operation_rows = list(operations)
    event_rows = list(events)
    summary_rows = list(summaries)
    object_rows = list(objects)
    snapshot_rows = list(snapshots)
    transfer_rows = list(transfers)
    check_rows = list(checks)
    parameter_rows = list(parameters)
    write_jsonl(output_dir / "operations.jsonl", operation_rows)
    write_jsonl(output_dir / "events.jsonl", event_rows)
    write_csv(
        output_dir / "astra_rank_summary.csv",
        summary_rows,
        [item.name for item in fields(AstraRankSummary)],
    )
    write_csv(
        output_dir / "objects.csv",
        object_rows,
        [item.name for item in fields(ObjectRecord)],
    )
    write_csv(
        output_dir / "memory.csv",
        snapshot_rows,
        [item.name for item in fields(MemorySnapshot)],
    )
    write_csv(
        output_dir / "transfers.csv",
        transfer_rows,
        [item.name for item in fields(TransferRecord)],
    )
    write_csv(
        output_dir / "checks.csv",
        check_rows,
        [item.name for item in fields(CheckResult)],
    )
    write_csv(
        output_dir / "parameters.csv",
        parameter_rows,
        [item.name for item in fields(ParameterRow)],
    )


def write_manifest(
    output_dir: Path,
    *,
    case_id: str,
    evidence_kind: str,
    run_kind: str,
    workflow_scope: str,
    execution: AstraRunResult,
    metrics: Mapping[str, Any],
    checks: Iterable[CheckResult],
    input_paths: Mapping[str, Path],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    check_rows = list(checks)
    if not check_rows or not all(item.passed for item in check_rows):
        raise ReportingError("success manifest requires a non-empty all-pass check set")
    inside_docker = Path("/.dockerenv").exists()
    container_image_tag = os.environ.get("RLXY_IMAGE")
    container_image_id = os.environ.get("RLXY_IMAGE_ID")
    if not inside_docker:
        raise ReportingError(
            "success manifest is restricted to the formal Docker execution path"
        )
    if not container_image_tag or not container_image_id:
        raise ReportingError(
            "formal Docker run is missing RLXY_IMAGE/RLXY_IMAGE_ID provenance; "
            "use scripts/docker.sh"
        )
    manifest = {
        "schema_version": "1.0",
        "status": "success",
        "run_id": output_dir.name,
        "case_id": case_id,
        "run_kind": run_kind,
        "evidence_kind": evidence_kind,
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {
            "workflow": workflow_scope,
            "simulator": "official AstraSim analytical congestion-unaware backend",
            "not_claimed": [
                "real model training",
                "learning quality",
                "hardware-calibrated wafer throughput",
                "contention-accurate full-wafer performance",
            ],
        },
        "software": {
            "rl_xy_version": __version__,
            "rl_xy_git_revision": _git_revision(project_root),
            "rl_xy_source_tree_sha256": deterministic_tree_hash(project_root),
            "rl_xy_source_tree_hash_scope": "source/config/tests/scripts/manual docs; excludes four generated v0.1/v0.2 walkthrough artifacts, caches and execution outputs",
            "astrasim_commit": execution.astra_commit,
            "astrasim_local_resource_patches": Path('/opt/astra-sim/.rlxy-resource-patch').read_text()
            if Path('/opt/astra-sim/.rlxy-resource-patch').is_file() else None,
            "astrasim_multicore_patch": Path('/opt/astra-sim/.rlxy-multicore-patch').read_text()
            if Path('/opt/astra-sim/.rlxy-multicore-patch').is_file() else None,
            "astrasim_gpu_patch": Path('/opt/astra-sim/.rlxy-gpu-patch').read_text()
            if Path('/opt/astra-sim/.rlxy-gpu-patch').is_file() else None,
            "astrasim_binary": str(execution.binary_path),
            "astrasim_binary_sha256": execution.binary_sha256,
            "python": platform.python_version(),
            "container_image_tag": container_image_tag,
            "container_image_id": container_image_id,
        },
        "execution_host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory_total_bytes": _mem_total_bytes(),
            "inside_docker": inside_docker,
            "gpu_used_by_simulator": False,
        },
        "command": list(execution.command),
        "exit_status": execution.return_code,
        "started_at_utc": execution.started_at_utc,
        "simulator_finished_at_utc": execution.finished_at_utc,
        "raw_logs": {
            "stdout": execution.stdout_path.name,
            "stderr": execution.stderr_path.name,
            "debug": execution.debug_log_path.relative_to(output_dir).as_posix(),
        },
        "inputs": {
            name: {"path": path.relative_to(output_dir).as_posix(), "sha256": sha256_file(path)}
            for name, path in sorted(input_paths.items())
        },
        "metrics_file": "metrics.json",
        "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        "checks_passed": len(check_rows),
        "checks_failed": 0,
        "headline_metrics": {
            key: metrics[key]
            for key in (
                "total_simulated_time_ns",
                "operation_count",
                "collective_logical_payload_bytes",
                "p2p_logical_payload_bytes",
            )
            if key in metrics
        },
        "artifact_sha256": _artifact_hashes(output_dir),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def write_summary_markdown(
    output_dir: Path,
    *,
    title: str,
    metrics: Mapping[str, Any],
    checks: Iterable[CheckResult],
) -> None:
    rows = list(checks)
    phase_rows = metrics.get("phase_timing", {})
    lines = [
        f"# {title}",
        "",
        "本次运行成功通过官方 AstraSim analytical backend；所有成功判据均来自本次原始日志与结构化检查。",
        "",
        f"- 总仿真时间：`{metrics.get('total_simulated_time_ns', 'n/a')} ns`",
        f"- 逻辑操作数：`{metrics.get('operation_count', 'n/a')}`",
        f"- 检查：`{sum(item.passed for item in rows)}/{len(rows)}` 通过",
        f"- 证据边界：`{metrics.get('metric_scope', 'synthetic_fixture')}`",
    ]
    if phase_rows:
        lines.extend(["", "## 阶段时间", "", "| 阶段 | 起点/ns | 终点/ns | 跨度/ns |", "|---|---:|---:|---:|"])
        for phase, values in phase_rows.items():
            lines.append(
                f"| {phase} | {values['start_ns']} | {values['end_ns']} | {values['span_ns']} |"
            )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "该结果证明 toy 配置、Chakra ET 生成、AstraSim 调用、日志解析、资源账本和语义校验能够闭环。它不是实机训练，也不证明真实晶圆吞吐、学习质量或硬件校准精度。",
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def toy_parameters(case: CaseConfig) -> list[Any]:
    return enumerate_parameters(case)
