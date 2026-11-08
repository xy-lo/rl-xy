import hashlib
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

class Math32InputTest(unittest.TestCase):
    def test_groups_and_zero_modelled_reward(self):
        groups = json.loads((HERE / 'inputs/32B-test1.json').read_text())
        rewards = [json.loads(line) for line in (HERE / 'inputs/32B-reward.jsonl').read_text().splitlines()]
        self.assertEqual(len(groups), 192)
        self.assertEqual(len({x['group_id'] for x in groups}), 192)
        self.assertTrue(all([x['sample'] for x in g['samples']] == list(range(8)) for g in groups))
        self.assertEqual(len(rewards), 1536)
        self.assertTrue(all(x['status'] == 'MODELLED_ZERO' and x['seconds'] == 0 for x in rewards))

    def test_model_config_hash(self):
        root = HERE.parents[1]
        config = HERE / 'inputs/Qwen3-32B/config.json'
        self.assertEqual(hashlib.sha256(config.read_bytes()).hexdigest(), '97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb')

if __name__ == '__main__':
    unittest.main()
