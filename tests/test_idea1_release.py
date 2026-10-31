"""Published Idea1 controller smoke check; no hardware or end-to-end claim."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

class Idea1ReleaseTests(unittest.TestCase):
    def test_estimator_has_no_future_response_oracle(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'scripts/idea1'), env.get('PYTHONPATH', '')])
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/idea1/test_locality_no_oracle.py')], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS:', result.stdout)
