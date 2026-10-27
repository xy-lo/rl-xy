from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_xy.config import ConfigError, load_case
from rl_xy.model import kv_bytes_per_processed_token_per_die, parameter_totals


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "configs/toy_grpo_2die/case.json"


class ConfigModelTest(unittest.TestCase):
    def test_frozen_parameter_and_kv_accounting(self) -> None:
        case = load_case(CASE)
        self.assertEqual(
            parameter_totals(case),
            {
                "unique_parameter_elements": 1576,
                "local_parameter_elements_per_die": 808,
                "long_term_bytes_per_die": 12928,
                "gradient_buffer_bytes_per_die": 3232,
                "resident_with_gradient_bytes_per_die": 16160,
            },
        )
        self.assertEqual(kv_bytes_per_processed_token_per_die(case), 32)

    def test_unknown_field_is_rejected(self) -> None:
        value = json.loads(CASE.read_text(encoding="utf-8"))
        value["silent_default"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_case(path)

    def test_two_round_override_is_resolved(self) -> None:
        case = load_case(CASE, rounds_override=2)
        self.assertEqual(case.rounds, 2)
        self.assertEqual(case.resolved["rounds"], 2)

    def test_non_synthetic_evidence_label_is_rejected(self) -> None:
        value = json.loads(CASE.read_text(encoding="utf-8"))
        value["evidence_kind"] = "calibrated"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_case(path)

    def test_strategy_fields_are_consumed_strictly(self) -> None:
        value = json.loads(CASE.read_text(encoding="utf-8"))
        value["strategy"]["rollout_max_concurrency"] = 999
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_case(path)


if __name__ == "__main__":
    unittest.main()
