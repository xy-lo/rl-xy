#!/usr/bin/env python3
"""Host orchestrator: all acceptance work executes in Docker, without GPUs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--image", default="rlxy-astrasim:0.3.0")
    p.add_argument("--container", default="rl-xy-inputs")
    p.add_argument("--legacy-root", default=Path.home()/"RLInfra", type=Path)
    a = p.parse_args()
    if Path(a.run_id).name != a.run_id or a.run_id in ("", ".", ".."):
        p.error("invalid run id")
    out = ROOT / "acceptance" / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "commands": [], "checks": [], "image": a.image, "new_gpu_training": False}

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def command(label, argv, expected=0):
        save()
        log = out / f"{label}.log"
        with log.open("w") as stream:
            r = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=900, env={**os.environ, "RLXY_CONTAINER": a.container})
        report["commands"].append({"label": label, "argv": argv, "exit_status": r.returncode,
                                   "expected_exit": expected, "log": str(log.relative_to(ROOT)),
                                   "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        save()
        print(f"{label}: rc={r.returncode} (expected {expected})", flush=True)
        if r.returncode != expected:
            raise ValueError(f"unexpected exit: {label}")

    def check(label, passed, value):
        report["checks"].append({"check": label, "passed": bool(passed), "evidence": value})
        save()
        if not passed:
            raise ValueError(label)

    try:
        command("build_inputs_image", ["docker", "build", "-t", a.image, "-f", "docker/Dockerfile.inputs", "."])
        command("create_container", ["bash", "scripts/docker.sh", "create-container", a.image])
        image = subprocess.check_output(["docker", "inspect", "--format", "{{.Image}}", a.container], text=True).strip()
        report["image_id"] = image
        docker = ["docker", "exec", "-e", f"RLXY_IMAGE={a.image}", "-e", f"RLXY_IMAGE_ID={image}", a.container, "python3"]
        command("unit_tests", docker + ["-m", "unittest", "discover", "-s", "tests", "-v"])
        hrel = out.relative_to(ROOT).as_posix() + "/hardware.json"
        command("formal_h_contract", docker + ["-m", "rl_xy.cli", "inspect-hardware", "--profile",
            "configs/hardware/xy_h1h2h3_54.json", "--out", hrel])
        command("unsupported_h_rejected", docker + ["-m", "rl_xy.cli", "inspect-hardware", "--profile",
            "configs/hardware/xy_h1h2h3_54.json", "--require-executable"], expected=2)
        reuse = f"runs/legacy_reuse/{a.run_id}"
        command("reuse_actual_legacy_inputs", ["docker", "run", "--rm", "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{ROOT}:/workspace/rl-xy",
            "-v", f"{a.legacy_root.resolve()}:/source/RLInfra:ro", "-w", "/workspace/rl-xy",
            "--entrypoint", "python3", image, "scripts/reuse_legacy.py", "--source-root", "/source/RLInfra",
            "--hardware", "configs/hardware/xy_h1h2h3_54.json", "--out", reuse])
        report["legacy_reuse"] = reuse
        raw = json.loads((ROOT / reuse / "report.json").read_text())
        check("legacy_log_csv_summary_exit_consistent", raw["status"] == "PASS" and raw["inside_docker"]
              and [r["observed_steps"] for r in raw["runs"]] == [116, 30], raw["runs"])
        check("six_generated_workloads", sum(len(r["synthetic_workloads"]) for r in raw["runs"]) == 6,
              [r["synthetic_workloads"] for r in raw["runs"]])
        for run in raw["runs"]:
            for seed in (0, 1, 2):
                folder = ROOT / reuse / run["run_id"] / f"seed{seed}"
                normalized = json.loads((folder / "normalized.json").read_text())
                synthesis = json.loads((folder / "synthesis.json").read_text())
                passed = normalized["origin_declared"] == "statistics_constrained_synthetic" and not normalized["available"]["reward_values"]
                passed &= all(abs(c["response_mean_source"] - c["response_mean_generated"]) <= 1 / 512
                              and abs(c["prompt_mean_source"] - c["prompt_mean_generated"]) <= 1 / 64
                              and abs(c["legacy_clip_ratio_source"] - c["legacy_clip_ratio_generated"]) <= 1 / 512 for c in synthesis["checks"])
                check(f"{run['run_id']}.seed{seed}.moments_and_origin", passed, len(normalized["steps"]))
        command("existing_resource_flow", ["bash", "scripts/docker.sh", "toy-resource", a.image, a.run_id])
        bundle = ROOT / "runs/toy_grpo_2die_resource" / a.run_id
        command("existing_resource_integrity", docker + ["scripts/check_result.py", str(bundle.relative_to(ROOT))])
        metrics = json.loads((bundle / "metrics.json").read_text())
        check("existing_native_flow_unchanged", metrics["total_simulated_time_ns"] == 108832.27,
              metrics["total_simulated_time_ns"])
        report["regression_bundle"] = str(bundle.relative_to(ROOT))
        manifest = json.loads((bundle / "manifest.json").read_text())
        report["source_tree_sha256"] = manifest["software"]["rl_xy_source_tree_sha256"]
        report["status"] = "PASS"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        return 0
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
