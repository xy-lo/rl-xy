from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from rl_xy.hardware_profile import HardwareProfileError, inspect_hardware, load_hardware_profile
from rl_xy.model_import import inspect_hf_model
from rl_xy.verl_trace import TraceError, load_verl_trace


ROOT = Path(__file__).resolve().parents[1]


def capture():
    meta = {"type": "run_meta", "run_id": "explicit-synthetic-adapter-test",
            "model": {"family": "Qwen", "name": "fixture", "revision": "fixture", "variant": "Base"},
            "data": {"dataset": "fixture", "revision": "fixture", "split": "train"},
            "algo": {"name": "grpo", "framework": "verl", "framework_version": "fixture", "framework_commit": "fixture",
                     "group_n": 3, "batch_prompts": 2, "temperature": 1.0, "top_p": 1.0,
                     "max_prompt_tokens": 8, "max_response_tokens": 8},
            "hardware": {"gpu": "fixture", "n_gpu": 1}, "environment": {"enabled": False, "mode": "single-turn"},
            "steps_recorded": [1, 2]}
    rows = [meta]
    for step in (1, 2):
        for group in range(2):
            for i in range(3):
                rows.append({"type": "traj", "step": step, "traj_id": f"s{step}g{group}t{i}",
                             "group_id": f"g{group}", "prompt_len": 2 + group, "resp_len": i + 1,
                             "behavior_version": step - 1, "finish_reason": "eos"})
        rows.append({"type": "stage_times", "step": step,
                     "stage_times_s": {k: 1.0 for k in ("rollout", "old_logprob", "ref", "actor", "sync")}})
    return rows


class MigrationInputTest(unittest.TestCase):
    def _read(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return load_verl_trace(path, origin="synthetic_fixture")

    def test_54_module_contract_aggregates_and_unresolved(self):
        h = load_hardware_profile(ROOT / "configs/hardware/xy_h1h2h3_54.json")
        report = inspect_hardware(h)
        self.assertEqual(report["aggregates"]["cores"], 13824)
        self.assertEqual(report["aggregates"]["dram_bytes"], 3456000000000)
        self.assertEqual(report["aggregates"]["sram_bytes_sum_not_shared_pool"], 17280000000)
        self.assertEqual(report["aggregates"]["peak_fp16_flops_per_ns"], 14155776)
        self.assertIsNone(report["aggregates"]["peak_bf16_flops_per_ns"])
        self.assertFalse(report["current_full_grpo_executable"])
        self.assertIn("core_bf16_flops_per_ns", report["unresolved_parameters"])
        self.assertFalse(report["geometry_errors"])

    def test_geometry_does_not_accept_area_quota_as_placement(self):
        h = load_hardware_profile(ROOT / "configs/hardware/xy_h1h2h3_54.json")
        h["parameters"]["module_width_mm"]["value"] = 33
        h["parameters"]["module_height_mm"]["value"] = 26
        self.assertTrue(inspect_hardware(h)["geometry_errors"])

    def test_hardware_unknown_fields_and_unit_or_missing_source_rejected(self):
        original = json.loads((ROOT / "configs/hardware/xy_h1h2h3_54.json").read_text())
        for key, bad in (("unit", "MB"), ("value", 0), ("source", ""), ("value", True)):
            h = copy.deepcopy(original)
            h["parameters"]["sram_bytes_per_core"][key] = bad
            with tempfile.TemporaryDirectory() as directory:
                p = Path(directory) / "h.json"
                p.write_text(json.dumps(h))
                with self.assertRaises(HardwareProfileError):
                    load_hardware_profile(p)

    def test_arbitrary_groups_and_lengths_preserved_without_fake_reward(self):
        r = self._read(capture())
        self.assertEqual(len(r["steps"]), 2)
        self.assertEqual(r["steps"][0]["global_response_tokens"], 12)
        self.assertEqual([g["prompt_length"] for g in r["steps"][0]["groups"]], [2, 3])
        self.assertEqual(r["steps"][1]["behavior_version"], 1)
        self.assertFalse(r["available"]["reward_values"])
        self.assertFalse(r["available"]["token_ids"])
        self.assertFalse(r["capture_authenticity_verified"])

    def test_gpu_timing_cannot_create_wafer_costs(self):
        rows = capture()
        rows[7]["stage_times_s"]["actor"] = 99999999
        r = self._read(rows)
        self.assertEqual(r["steps"][0]["gpu_observed_stage_times_s"]["actor"], 99999999)
        self.assertNotIn("duration_ns", json.dumps(r))

    def test_malformed_semantics_rejected(self):
        mutations = [
            lambda r: r.pop(1),
            lambda r: r[2].update(traj_id=r[1]["traj_id"]),
            lambda r: r[2].update(prompt_len=7),
            lambda r: r[1].update(behavior_version=1),
            lambda r: r[1].update(resp_len=0),
            lambda r: r[1].update(resp_len=9),
            lambda r: r[1].update(finish_reason="drop"),
            lambda r: r.insert(8, copy.deepcopy(r[7])),
            lambda r: r[7]["stage_times_s"].update(actor=float("nan")),
            lambda r: r[0]["algo"].update(group_n=True),
            lambda r: r[0]["environment"].update(enabled=True),
            lambda r: r[1].update(prompt_text="must-not-be-silently-accepted"),
            lambda r: r[0]["algo"].update(top_p=2),
        ]
        for mutate in mutations:
            rows = capture()
            mutate(rows)
            with self.subTest(mutation=mutate), self.assertRaises(TraceError):
                self._read(rows)

    def test_truncated_capture_and_zero_timings_explicit(self):
        rows = capture()[:8]
        rows[7]["stage_times_s"]["sync"] = 0
        r = self._read(rows)
        self.assertEqual(len(r["warnings"]), 2)

    def test_qwen2_parameter_identity_with_gqa_bias_and_tied_embedding(self):
        raw = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
               "hidden_size": 896, "num_attention_heads": 14, "num_key_value_heads": 2,
               "num_hidden_layers": 24, "intermediate_size": 4864, "vocab_size": 151936,
               "max_position_embeddings": 32768, "tie_word_embeddings": True,
               "hidden_act": "silu", "attention_dropout": 0.0, "use_sliding_window": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw))
            result = inspect_hf_model(path)
            self.assertEqual(result["unique_parameter_elements"], 494032768)
            self.assertEqual(result["bf16_kv_bytes_per_processed_token_all_layers"], 12288)
            self.assertFalse(result["current_full_grpo_executable"])
            raw["tie_word_embeddings"] = False
            path.write_text(json.dumps(raw))
            # Untying adds exactly the 151936x896 output matrix.
            self.assertEqual(inspect_hf_model(path)["unique_parameter_elements"], 630167424)


if __name__ == "__main__":
    unittest.main()
