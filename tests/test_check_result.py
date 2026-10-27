from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_result.py"


class CheckResultScriptTests(unittest.TestCase):
    def _bundle(self, root: Path) -> None:
        text_files = {
            "astra_config_contract.json": "{}\n",
            "SUMMARY.md": "# summary\n",
            "resolved_config.json": "{}\n",
            "parameters.csv": "parameter_id\n",
            "objects.csv": "storage_id\n",
            "operations.jsonl": "{}\n",
            "events.jsonl": "{}\n",
            "memory.csv": "step\n",
            "transfers.csv": "op_id\n",
            "metrics.json": "{}\n",
            "astra_stdout.log": "ok\n",
            "astra_stderr.log": "",
            "astra_command.json": '{"argv": [], "exit_status": 0}\n',
            "astra_rank_summary.csv": "rank\n",
            "checks.csv": "check_id,passed\nc,true\n",
            "astra_logs/log.log": "event\n",
            "inputs/case.json": "{}\n",
        }
        for relative, value in text_files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        checks = {"checks": [{"check_id": "c", "passed": True}]}
        summary = {"status": "PASS", "checks_passed": 1, "checks_failed": 0}
        status = {"status": "success"}
        (root / "checks.json").write_text(json.dumps(checks), encoding="utf-8")
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (root / "status.json").write_text(json.dumps(status), encoding="utf-8")

        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts = {
            path.relative_to(root).as_posix(): digest(path)
            for path in root.rglob("*")
            if path.is_file()
        }
        manifest = {
            "status": "success",
            "evidence_kind": "synthetic_fixture",
            "exit_status": 0,
            "checks_passed": 1,
            "checks_failed": 0,
            "software": {
                "astrasim_commit": "4ce9ecfe746a3ee5fe47559a11d91644d641fa7f",
                "container_image_tag": "rlxy-astrasim:0.1.0",
                "container_image_id": "sha256:test",
            },
            "execution_host": {"inside_docker": True},
            "metrics_sha256": digest(root / "metrics.json"),
            "inputs": {
                "case": {
                    "path": "inputs/case.json",
                    "sha256": digest(root / "inputs/case.json"),
                }
            },
            "artifact_sha256": artifacts,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_complete_bundle_passes_and_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            passed = subprocess.run(
                [sys.executable, str(CHECKER), str(root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("manifest_sha256", passed.stdout)
            (root / "metrics.json").write_text('{"tampered": true}\n', encoding="utf-8")
            failed = subprocess.run(
                [sys.executable, str(CHECKER), str(root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
