#!/usr/bin/env python3
"""Run Docker acceptance gates and preserve every command, exit code and log."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="rlxy-astrasim:0.1.0")
    parser.add_argument("--container", default="rl-xy")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a single non-empty path component")
    output = root / "acceptance" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "started_at_utc": utc_now(),
        "image": args.image,
        "container": args.container,
        "commands": [],
        "checks": [],
    }
    environment = {**os.environ, "RLXY_CONTAINER": args.container}
    def save() -> None:
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    def command(label: str, argv: list[str]) -> None:
        log = output / f"{label}.log"
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                argv, cwd=root, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, check=False, timeout=900,
            )
        record = {
            "label": label, "argv": argv, "exit_status": result.returncode,
            "wall_seconds": round(time.monotonic() - started, 3),
            "log": log.relative_to(root).as_posix(),
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        }
        report["commands"].append(record)
        save()
        print(f"{label}: rc={result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {log}")
    def check(label: str, passed: bool, evidence: object) -> None:
        report["checks"].append({"check": label, "passed": passed, "evidence": evidence})
        save()
        if not passed:
            raise RuntimeError(f"acceptance assertion failed: {label}")
    def read(relative: str):
        return json.loads((root / relative).read_text(encoding="utf-8"))
    try:
        save()
        for label, action, run_id in (
            ("unit_tests", "test", args.run_id),
            ("transfer", "transfer", args.run_id),
            ("toy", "toy", args.run_id),
            ("two_rounds", "two-rounds", args.run_id),
            ("toy_repeat", "toy", args.run_id + "-repeat"),
        ):
            command(label, ["bash", "scripts/docker.sh", action, args.image, run_id])
        bundles = {
            "transfer": f"runs/fixture_transfer_21ns/{args.run_id}",
            "toy": f"runs/toy_grpo_2die/{args.run_id}",
            "two_rounds": f"runs/toy_grpo_2die_2rounds/{args.run_id}",
            "toy_repeat": f"runs/toy_grpo_2die/{args.run_id}-repeat",
        }
        report["bundles"] = bundles
        for name, relative in bundles.items():
            command(f"integrity_{name}", [sys.executable, "scripts/check_result.py", relative])
        transfer = read(bundles["transfer"] + "/metrics.json")
        check("hand_calculated_transfer_21ns", transfer["total_simulated_time_ns"] == 21, transfer)
        check("sender_service_5ns", transfer["p2p_send_service_ns"] == 5, transfer["p2p_send_service_ns"])
        check("posted_receive_includes_wait", transfer["p2p_receive_occupancy_ns"] == 13, transfer["p2p_receive_occupancy_ns"])
        toy = read(bundles["toy"] + "/metrics.json")
        two = read(bundles["two_rounds"] + "/metrics.json")
        check("actor_version_0_to_1_to_2",
              toy["actor_version_start"] == 0 and toy["actor_version_end"] == 1
              and two["actor_version_start"] == 0 and two["actor_version_end"] == 2,
              {"single": toy["actor_version_end"], "two": two["actor_version_end"]})
        check("reference_frozen", two["reference_version_start"] == two["reference_version_end"] == 0,
              {"start": two["reference_version_start"], "end": two["reference_version_end"]})
        first = root / bundles["toy"] / "events.jsonl"
        repeated = root / bundles["toy_repeat"] / "events.jsonl"
        check("repeated_astra_event_ticks_identical", first.read_bytes() == repeated.read_bytes(),
              {"first_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
               "repeat_sha256": hashlib.sha256(repeated.read_bytes()).hexdigest()})
        check("two_round_wall_time_doubles",
              two["total_simulated_time_ns"] == 2 * toy["total_simulated_time_ns"],
              {"single_ns": toy["total_simulated_time_ns"], "two_ns": two["total_simulated_time_ns"]})
        sys.path.insert(0, str(root / "src"))
        from rl_xy.reporting import deterministic_tree_hash
        source_hash = deterministic_tree_hash(root)
        run_hashes = {
            name: read(relative + "/manifest.json")["software"]["rl_xy_source_tree_sha256"]
            for name, relative in bundles.items()
        }
        check("all_runs_match_current_source_tree",
              all(value == source_hash for value in run_hashes.values()),
              {"current": source_hash, "runs": run_hashes})
        report["status"] = "PASS"
        report["finished_at_utc"] = utc_now()
        save()
        print(f"PASS: {output}")
        return 0
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at_utc"] = utc_now()
        save()
        print(report["error"], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
