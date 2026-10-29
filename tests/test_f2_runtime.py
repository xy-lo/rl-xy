import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rl_xy.f2_runtime import ASSETS, INPUTS, make_operator, sha, verify_assets


class F2ReleaseTest(unittest.TestCase):
    def test_all_pinned_assets_and_five_input_windows(self):
        manifest = verify_assets()
        self.assertFalse(manifest['physical_end_to_end_calibrated'])
        for model in ('4B', '8B', '14B'):
            for name, expected in manifest['input_hashes'][model].items():
                self.assertEqual(sha(INPUTS / ('Qwen3-' + model) / name), expected)
        for model, expected in [('4B', 128), ('8B', 64), ('14B', 128)]:
            policy = json.loads((ASSETS / 'policies' / (model + '-wafer.json')).read_text())
            self.assertEqual(policy['cap'], expected)

    def test_relocated_gpu_and_wafer_local_costs_match_original_smoke(self):
        for platform, expected in [('gpu', 392379.864794568), ('wafer', 384493.0920343137)]:
            op = make_operator('14B', platform)
            cost = op.mm(128, op.model.vocab_size // 4, op.model.hidden_size) + op.attention((1,), (1000,), 4)
            self.assertAlmostEqual(cost.ns, expected, delta=0.001)

    def test_f2_rejects_unsupported_models(self):
        with self.assertRaises(ValueError):
            make_operator('32B', 'gpu')

    def test_screen_can_select_the_128_concurrency_baseline(self):
        from rl_xy.qwen3_experiment import Screen
        op = SimpleNamespace(info={'config':{'model_type':'fixture'}},
                             admission=lambda *args:{'passed':True, 'peak_bytes':1})
        screen = Screen(op, [None] * 1536)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(screen, 'generation', side_effect=lambda tp, cap:128-cap), \
             patch.object(screen, 'teacher', return_value=1), \
             patch.object(screen, 'fixed', return_value={'weight_sync':0,'samples':0,'optimizer':0}), \
             patch('rl_xy.qwen3_experiment.advantage_plan', return_value=None), \
             patch('rl_xy.qwen3_experiment.simple_plan_time', return_value=0):
            rows = screen.run(folder)
            self.assertEqual(rows[0]['policy']['cap'], 128)
            self.assertTrue(any(r['policy']['cap'] == 128 for r in rows))


if __name__ == '__main__':
    unittest.main()
