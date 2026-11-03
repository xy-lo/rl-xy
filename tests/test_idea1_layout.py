"""Check TP4-to-TP8 GQA layout and accounted migration for each Qwen3 size."""
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/idea1'))
from rl_xy.f2_runtime import make_operator
# Both historical TP4->TP8 and layered TP2->TP4 drivers have this module name.
# Other contract tests may import the layered kernel first. Bind this test to
# its explicit historical file instead of depending on global import order.
spec=importlib.util.spec_from_file_location('historical_tp4_tp8_layout_test',ROOT/'scripts/idea1/attention_tp8_program.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
head_layout,migration,compile_attention_tp8=module.head_layout,module.migration,module.compile_attention_tp8

class Idea1LayoutTests(unittest.TestCase):
    def test_gqa_heads_and_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in ('4B', '8B', '14B'):
                with self.subTest(model=model):
                    op = make_operator(model, 'wafer', cache_dir=Path(tmp) / model)
                    src, dst, contexts = (0, 1, 2, 3), (6, 7, 8, 9), (41, 100, 257)
                    layout = head_layout(op, src, dst)
                    for lane in layout:
                        self.assertEqual(len(lane['kv_heads']), 1)
                        self.assertTrue(all(q // (op.model.query_heads // op.model.kv_heads) == lane['kv_heads'][0] for q in lane['query_heads']))
                    move = migration(op, src, dst, contexts, tag_start=9000)
                    self.assertEqual(move['head_layout'], layout)
                    self.assertEqual(move['pack_extra_read_bytes_per_lane'], move['kv_bytes_per_lane'])
                    self.assertEqual(move['pack_extra_write_bytes_per_lane'], move['kv_bytes_per_lane'])
                    program = compile_attention_tp8(op, src, dst, contexts, group_id=99, tag_start=10000)
                    self.assertEqual(program['layers'], op.model.layers)
                    self.assertEqual(sum(x['kind'] == 'p2p_direct' for x in program['source']), 4 * op.model.layers)
                    self.assertEqual(sum(x['kind'] == 'AR' for x in program['helper']), op.model.layers)
