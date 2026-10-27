#!/usr/bin/env python3
"""Small, dependency-free run-bundle integrity check for operators and CI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = {
    "manifest.json",
    "astra_config_contract.json",
    "SUMMARY.md",
    "resolved_config.json",
    "parameters.csv",
    "objects.csv",
    "operations.jsonl",
    "events.jsonl",
    "memory.csv",
    "transfers.csv",
    "checks.json",
    "summary.json",
    "metrics.json",
    "status.json",
    "astra_stdout.log",
    "astra_stderr.log",
    "astra_command.json",
    "astra_rank_summary.csv",
    "checks.csv",
    "astra_logs/log.log",
}
PINNED_ASTRA_COMMIT = "4ce9ecfe746a3ee5fe47559a11d91644d641fa7f"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    missing = sorted(name for name in REQUIRED if not (args.run_dir / name).is_file())
    if missing:
        raise SystemExit(f"missing run files: {missing}")
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
    checks = json.loads((args.run_dir / "checks.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise SystemExit(f"manifest status is {manifest.get('status')!r}")
    if summary.get("status") != "PASS":
        raise SystemExit(f"summary status is {summary.get('status')!r}")
    if status.get("status") != "success":
        raise SystemExit(f"run status is {status.get('status')!r}")
    if manifest.get("evidence_kind") != "synthetic_fixture":
        raise SystemExit("formal MVP result must be labelled synthetic_fixture")
    if manifest.get("exit_status") != 0:
        raise SystemExit(f"AstraSim exit status is {manifest.get('exit_status')!r}")
    software = manifest.get("software", {})
    host = manifest.get("execution_host", {})
    if software.get("astrasim_commit") != PINNED_ASTRA_COMMIT:
        raise SystemExit("AstraSim commit is not the pinned MVP revision")
    if not software.get("container_image_tag") or not software.get("container_image_id"):
        raise SystemExit("Docker image tag/ID provenance is missing")
    if host.get("inside_docker") is not True:
        raise SystemExit("success bundle was not produced by the formal Docker path")
    check_rows = checks.get("checks")
    if not isinstance(check_rows, list) or not check_rows:
        raise SystemExit("checks.json must contain a non-empty checks list")
    failed = [item["check_id"] for item in check_rows if not item["passed"]]
    if failed:
        raise SystemExit(f"failed checks: {failed}")
    if manifest.get("checks_passed") != len(check_rows) or manifest.get("checks_failed") != 0:
        raise SystemExit("manifest check counters do not match checks.json")
    if summary.get("checks_passed") != len(check_rows) or summary.get("checks_failed") != 0:
        raise SystemExit("summary check counters do not match checks.json")
    metrics_digest = hashlib.sha256(
        (args.run_dir / "metrics.json").read_bytes()
    ).hexdigest()
    if manifest.get("metrics_sha256") != metrics_digest:
        raise SystemExit("metrics_sha256 does not match metrics.json")
    input_records = manifest.get("inputs")
    if not isinstance(input_records, dict) or not input_records:
        raise SystemExit("manifest inputs must be a non-empty object")
    for name, record in input_records.items():
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise SystemExit(f"malformed manifest input record: {name}")
        path = args.run_dir / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"input integrity failure: {name}")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise SystemExit("manifest artifact_sha256 must be a non-empty object")
    actual_artifacts = {
        path.relative_to(args.run_dir).as_posix()
        for path in args.run_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(artifact_hashes) != actual_artifacts:
        raise SystemExit(
            "artifact inventory mismatch: "
            f"missing={sorted(actual_artifacts - set(artifact_hashes))}, "
            f"unexpected={sorted(set(artifact_hashes) - actual_artifacts)}"
        )
    mismatched = []
    for relative, expected in artifact_hashes.items():
        path = args.run_dir / relative
        if not path.is_file():
            mismatched.append(f"{relative}:missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatched.append(f"{relative}:sha256")
    if mismatched:
        raise SystemExit(f"artifact integrity failure: {mismatched}")
    digest = hashlib.sha256((args.run_dir / "manifest.json").read_bytes()).hexdigest()
    print(json.dumps({"status": "PASS", "manifest_sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

