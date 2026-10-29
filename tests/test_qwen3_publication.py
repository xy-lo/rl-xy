import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rl_xy.qwen3_paths import input_root, native_root, native_binary, generation_cache, output_path, campaign_lock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen3_public_checker', ROOT / 'scripts/check_qwen3_results.py')
checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)


class Qwen3PublicationTest(unittest.TestCase):
    def test_public_results_and_claim_boundaries(self):
        result = checker.check()
        self.assertEqual(result['formal_runs'], 24)
        self.assertEqual(result['current_vs_historical_cost_file_changes'], [])
        self.assertFalse(result['hardware_calibrated'])

    def test_scope_promotion_is_rejected_before_checksum_validation(self):
        real_read = checker.read
        def promoted(path):
            value = real_read(path)
            if path.name == 'provenance.json': value['hardware_calibrated'] = True
            return value
        with patch.object(checker, 'read', side_effect=promoted):
            with self.assertRaisesRegex(ValueError, 'hardware_calibrated'):
                checker.check()

    def test_output_cannot_overwrite_frozen_inputs_or_source(self):
        for value in [ROOT, ROOT.parent, ROOT/'experiments/new-run', ROOT/'src/run']:
            with self.assertRaises(ValueError): output_path(ROOT, value)
        self.assertEqual(output_path(ROOT, ROOT/'runs/new-run'), ROOT/'runs/new-run')

    def test_portable_paths_and_explicit_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict('os.environ', {'ASTRA_SIM_ROOT':tmp, 'ASTRA_SIM_BIN':str(Path(tmp)/'binary')}, clear=True):
                self.assertEqual(native_root(ROOT), Path(tmp).resolve())
                self.assertEqual(native_binary(ROOT), (Path(tmp)/'binary').resolve())
                self.assertEqual(generation_cache(ROOT), ROOT/'runs/qwen3/generation_cache')
        self.assertEqual(input_root(ROOT), ROOT/'experiments/2026-10-29-qwen3/inputs')

    def test_campaign_lock_rejects_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'campaign.lock'
            with campaign_lock(path):
                with self.assertRaisesRegex(RuntimeError, 'already active'):
                    with campaign_lock(path): pass
            with campaign_lock(path): self.assertTrue(path.exists())


if __name__ == '__main__': unittest.main()
