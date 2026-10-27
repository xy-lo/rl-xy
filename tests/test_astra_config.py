from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_xy.astra_adapter.config import AstraConfigError, validate_astra_configs


ROOT = Path(__file__).resolve().parents[1]


class AstraConfigTests(unittest.TestCase):
    def _validate(self, case_dir: Path):
        return validate_astra_configs(
            system_path=case_dir / "astra_system.json",
            network_path=case_dir / "astra_network.yml",
            remote_memory_path=case_dir / "astra_remote_memory.json",
            ranks=2,
            astra_time_scale=1000,
            d2d_bytes_per_model_ns=8,
            d2d_latency_model_ns=1,
        )

    def test_both_frozen_cases_match_unit_contract(self) -> None:
        for name in ("fixture_transfer", "toy_grpo_2die"):
            contract = self._validate(ROOT / "configs" / name)
            self.assertFalse(contract["contention_modelled"])
            self.assertAlmostEqual(contract["effective_raw_bytes_per_ns"], 0.008)

    def test_bandwidth_drift_is_rejected(self) -> None:
        source = ROOT / "configs" / "fixture_transfer"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in (
                "astra_system.json",
                "astra_network.yml",
                "astra_remote_memory.json",
            ):
                (target / name).write_bytes((source / name).read_bytes())
            network = target / "astra_network.yml"
            network.write_text(
                network.read_text(encoding="utf-8").replace(
                    "0.007450580596923828", "8.0"
                ),
                encoding="utf-8",
            )
            with self.assertRaises(AstraConfigError):
                self._validate(target)

    def test_json_boolean_cannot_impersonate_integer_contract(self) -> None:
        source = ROOT / "configs" / "fixture_transfer"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in (
                "astra_system.json",
                "astra_network.yml",
                "astra_remote_memory.json",
            ):
                (target / name).write_bytes((source / name).read_bytes())
            system_path = target / "astra_system.json"
            system = json.loads(system_path.read_text(encoding="utf-8"))
            system["trace-enabled"] = True
            system_path.write_text(json.dumps(system), encoding="utf-8")
            with self.assertRaises(AstraConfigError):
                self._validate(target)


if __name__ == "__main__":
    unittest.main()
