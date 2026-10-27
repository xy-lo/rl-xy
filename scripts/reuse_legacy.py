#!/usr/bin/env python3
"""CPU-only audit/reuse of existing RLInfra runs; never launch training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

from rl_xy.hardware_profile import inspect_hardware_file
from rl_xy.legacy_workload import import_legacy_run, read_legacy_overrides, synthesize_lengths
from rl_xy.migration import preflight_migration
from rl_xy.model_import import inspect_hf_model
from rl_xy.verl_trace import load_verl_trace


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=["E1_gsm8k_qwen05b", "E3_dapo_qwen05b_r2048"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "inside_docker": Path("/.dockerenv").is_file(),
              "new_gpu_training": False, "runs": [], "hardware": inspect_hardware_file(args.hardware)}
    write(out / "report.json", report)
    base = args.source_root / "experiments/full_paper_queue"
    # Archived overrides are inspected as scalar data, never executed.
    matches = {}
    for path in base.glob("outputs/*/*/.hydra/overrides.yaml"):
        lines = set(path.read_text().splitlines())
        if not any(f"- trainer.experiment_name={run}" in lines for run in args.runs):
            continue
        values = read_legacy_overrides(path)
        name = values["trainer.experiment_name"]
        if name in args.runs:
            if name in matches:
                raise ValueError(f"multiple archived configs for {name}; explicit disambiguation required")
            matches[name] = path
    try:
        for run in args.runs:
            if run not in matches:
                raise ValueError(f"archived overrides not found for {run}")
            target = out / run
            target.mkdir()
            inputs = target / "inputs"
            inputs.mkdir()
            originals = {"run.log": base / f"logs/tasks/{run}.log", "overrides.yaml": matches[run],
                         "summary.json": base / f"results/{run}.summary.json", "steps.csv": base / f"results/{run}.steps.csv",
                         "state.txt": base / f"state/{run}.state"}
            for name, path in originals.items():
                shutil.copyfile(path, inputs / name)
            imported = import_legacy_run(inputs / "run.log", inputs / "overrides.yaml", run)
            write(target / "statistics.json", imported)
            summary = json.loads((inputs / "summary.json").read_text())
            csv_rows = list(csv.DictReader((inputs / "steps.csv").open()))
            by_step = {int(r["step"]): r for r in csv_rows}
            if len(by_step) != len(csv_rows):
                raise ValueError("duplicate steps in historical CSV")
            compared = []
            aliases = {"gen": "timing_s/gen", "old_log_prob": "timing_s/old_log_prob", "ref": "timing_s/ref",
                       "update_actor": "timing_s/update_actor", "update_weights": "timing_s/update_weights",
                       "step_time": "timing_s/step", "resp_mean": "response_length/mean", "resp_max": "response_length/max",
                       "resp_min": "response_length/min", "resp_clip": "response_length/clip_ratio", "prompt_mean": "prompt_length/mean"}
            for step in imported["steps"]:
                row = by_step.get(step["step"])
                if row is None or any(abs(float(row[k]) - step["metrics"][v]) > 1e-9 for k, v in aliases.items()):
                    raise ValueError(f"raw log/CSV disagreement at {run}/{step['step']}")
                compared.append(step["step"])
            mean = statistics.mean(s["metrics"]["timing_s/step"] for s in imported["steps"])
            if len(compared) != len(csv_rows) or summary["steps"] != len(compared) or abs(summary["step_time_s"]["mean"] - mean) > 1e-9:
                raise ValueError("summary/raw log/CSV coverage or mean disagreement")
            state = dict(line.split("=", 1) for line in (inputs / "state.txt").read_text().splitlines() if "=" in line)
            if state.get("run_exit") != "0" or int(state.get("completed_steps", "0")) != len(compared):
                raise ValueError("historical exit/step coverage not successful")
            model_name = Path(imported["overrides"]["actor_rollout_ref.model.path"]).name
            model_config = args.source_root / "models" / model_name / "config.json"
            shutil.copyfile(model_config, inputs / "model_config.json")
            write(target / "model.json", inspect_hf_model(inputs / "model_config.json"))
            synthetic = []
            for seed in args.seeds:
                folder = target / f"seed{seed}"
                folder.mkdir()
                rows, synthesis = synthesize_lengths(imported, seed=seed, selected_steps=compared, clip_reference="observed_max")
                trace = folder / "lengths.jsonl"
                trace.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
                normalized = load_verl_trace(trace, origin="statistics_constrained_synthetic")
                write(folder / "synthesis.json", synthesis)
                write(folder / "normalized.json", normalized)
                # Admission does not allocate a model or run the GPU/wafer.
                write(folder / "preflight.json", preflight_migration(trace_path=trace,
                    trace_origin="statistics_constrained_synthetic", hardware_path=args.hardware,
                    model_path=inputs / "model_config.json"))
                synthetic.append({"seed": seed, "steps": len(normalized["steps"]),
                                  "trajectories": sum(c["trajectory_count"] for c in synthesis["checks"]),
                                  "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()})
            row = {"run_id": run, "observed_steps": len(compared), "gpu_step_mean_s": mean,
                   "raw_log_csv_summary_state_consistent": True,
                   "synthetic_workloads": synthetic, "sources": {k: {"original_path": str(v),
                   "sha256": hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in originals.items()},
                   "remaining_provenance_gaps": ["historical software commit/versions and full resolved config not frozen in this audit",
                                                 "current model config is not proof of historical weight revision"]}
            report["runs"].append(row)
            write(out / "report.json", report)
            print(f"{run}: {len(compared)} observed steps; {len(synthetic)} synthetic seeds", flush=True)
        report["status"] = "PASS"
        report["artifact_sha256"] = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(out.rglob("*")) if p.is_file() and p != out / "report.json"}
        write(out / "report.json", report)
        return 0
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write(out / "report.json", report)
        raise


if __name__ == "__main__":
    sys.exit(main())
