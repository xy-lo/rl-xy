import json
from pathlib import Path
import random
import tempfile
import unittest

from rl_xy.legacy_workload import _lengths, import_legacy_run, synthesize_lengths
from rl_xy.verl_trace import load_verl_trace


class LegacyWorkloadTest(unittest.TestCase):
    def source(self, directory):
        metrics = {"prompt_length/min": 2, "prompt_length/mean": 3, "prompt_length/max": 4,
                   "response_length/min": 1, "response_length/mean": 4, "response_length/max": 8,
                   "response_length/clip_ratio": 0.125,
                   "timing_s/gen": 1, "timing_s/old_log_prob": 2, "timing_s/ref": 3,
                   "timing_s/update_actor": 4, "timing_s/update_weights": 1, "timing_s/step": 12}
        overrides = {"data.train_batch_size": 4, "data.max_prompt_length": 8, "data.max_response_length": 8,
                     "actor_rollout_ref.rollout.n": 2, "algorithm.adv_estimator": "grpo", "trainer.experiment_name": "test",
                     "actor_rollout_ref.model.path": "/models/Qwen2.5-0.5B-Instruct", "data.train_files": "/data/test/train.parquet",
                     "trainer.n_gpus_per_node": 2, "trainer.nnodes": 1}
        log, override = Path(directory) / "run.log", Path(directory) / "overrides.yaml"
        log.write_text("\x1b[36m(TaskRunnerV1 pid=1)\x1b[0m step:1 - " + " - ".join(f"{k}:{v}" for k, v in metrics.items()) + "\n")
        override.write_text("".join(f"- {k}={v}\n" for k, v in overrides.items()))
        return log, override

    def test_reuse_scalars_and_synthesize_bounded_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            log, override = self.source(directory)
            source = import_legacy_run(log, override, "test")
            rows, report = synthesize_lengths(source, seed=42, selected_steps=[1], clip_reference="public_cap")
            trace = Path(directory) / "trace.jsonl"
            trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
            normalized = load_verl_trace(trace, origin="statistics_constrained_synthetic")
            self.assertEqual(normalized["steps"][0]["global_response_tokens"], 32)
            self.assertFalse(normalized["available"]["reward_values"])
            self.assertIsNone(normalized["run_meta"]["algo"]["temperature"])
            self.assertEqual(report["checks"][0]["response_mean_generated"], 4)
            self.assertEqual(report["checks"][0]["legacy_clip_ratio_generated"], 0.125)
            self.assertEqual(synthesize_lengths(source, seed=42, selected_steps=[1], clip_reference="public_cap"), (rows, report))

    def test_batch_max_ratio_cannot_be_mislabeled_as_public_cap_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            log, override = self.source(directory)
            log.write_text(log.read_text().replace("response_length/max:8", "response_length/max:7"))
            source = import_legacy_run(log, override, "test")
            self.assertFalse(source["clip_metric_audit"]["public_cap_interpretation_consistent"])
            with self.assertRaises(ValueError):
                synthesize_lengths(source, seed=1, selected_steps=[1], clip_reference="public_cap")
            _, report = synthesize_lengths(source, seed=1, selected_steps=[1], clip_reference="observed_max")
            self.assertEqual(report["checks"][0]["public_cap_hit_ratio_generated"], 0)
            self.assertEqual(report["checks"][0]["legacy_clip_ratio_generated"], 0.125)

    def test_infeasible_joint_statistics_rejected(self):
        with self.assertRaises(ValueError):
            _lengths(8, 1, 8, 1, random.Random(0), cap=8, clipped=4)
        with self.assertRaises(ValueError):
            _lengths(8, 1, 8, 4, random.Random(0), cap=8, clipped=0)

    def test_exact_integer_moments_over_seeds(self):
        for seed in range(20):
            x = _lengths(512, 10, 512, 322.27734375, random.Random(seed), cap=512, clipped=79)
            self.assertEqual(sum(x), 165006)
            self.assertEqual(sum(t == 512 for t in x), 79)
            self.assertEqual((min(x), max(x)), (10, 512))

    def test_duplicate_or_wrong_run_or_missing_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            log, override = self.source(directory)
            with self.assertRaises(ValueError):
                import_legacy_run(log, override, "other")
            text = log.read_text()
            log.write_text(text + text.replace("timing_s/step:12", "timing_s/step:13"))
            with self.assertRaises(ValueError):
                import_legacy_run(log, override, "test")
            log.write_text(text.replace("prompt_length/min:2 - ", ""))
            with self.assertRaises(ValueError):
                import_legacy_run(log, override, "test")


if __name__ == "__main__":
    unittest.main()
