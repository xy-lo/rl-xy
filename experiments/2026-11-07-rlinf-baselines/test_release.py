"""Fast integrity checks for packaged test1 inputs and RLinf partition behavior."""
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from rlinf_balance import get_seqlen_balanced_partitions

class ReleaseIntegrity(unittest.TestCase):
    def test_inputs_and_rewards(self):
        manifest = json.loads((ROOT / 'inputs/manifest.json').read_text())
        for model, info in manifest['models'].items():
            with self.subTest(model=model):
                data = ROOT / 'inputs' / f'{model}-test1.json'
                bank = ROOT / 'inputs' / f'{model}-reward.jsonl'
                self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), info['test1_input_sha256'])
                self.assertEqual(hashlib.sha256(bank.read_bytes()).hexdigest(), info['test1_reward_sha256'])
                groups = json.loads(data.read_text())
                self.assertEqual(len(groups), 192)
                self.assertTrue(all(x['window'] == 'test1' and len(x['samples']) >= 8 for x in groups))
                expected = Counter((str(g['group_id']), r['sample']) for g in groups for r in g['samples'][:8])
                rows = [json.loads(x) for x in bank.read_text().splitlines()]
                self.assertEqual(Counter((str(r['group_id']), r['sample']) for r in rows), expected)
                self.assertTrue(all(r['status'] == 'PASS_MEASUREMENT' and r['seconds'] >= 0 for r in rows))
    def test_partition_coverage_and_balance(self):
        lengths = [12, 11, 10, 9, 8, 7, 6, 5]
        parts = get_seqlen_balanced_partitions(lengths, 4, True)
        self.assertEqual(sorted(i for p in parts for i in p), list(range(8)))
        self.assertTrue(all(len(p) == 2 for p in parts))
        self.assertEqual([sum(lengths[i] for i in p) for p in parts], [17] * 4)

if __name__ == '__main__':
    unittest.main()
