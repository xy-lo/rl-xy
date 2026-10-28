from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest

from rl_xy.config import load_case
from rl_xy.distributed_optimizer import (build_layout, layout_for_case,
    global_loss_scaling, validate_publication)

ROOT = Path(__file__).resolve().parents[1]


def param(name, elements, replicated=False):
    return SimpleNamespace(parameter_id=name, local_elements_per_die=elements, replicated=replicated)


class DistributedOptimizerTests(unittest.TestCase):
    def test_DP48_and_DP24_padding_and_intersections_cover_exact_parameters(self):
        rows = [param('a', 131), param('b', 70), param('norm', 9, True)]
        for dp in [24, 48]:
            layout = build_layout(rows, dp_size=dp, tp_size=2)
            self.assertEqual(layout.padded_elements % 384, 0)
            self.assertTrue(all(p.start % 64 == 0 for p in layout.parameters))
            intervals = [layout.shard_interval(i) for i in range(dp)]
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], layout.padded_elements)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])))
            actual = {p.parameter_id: 0 for p in rows}
            for rank in range(dp):
                for piece in layout.intersections(rank):
                    actual[piece['parameter_id']] += piece['elements']
            self.assertEqual(actual, {'a': 131, 'b': 70, 'norm': 9})
            self.assertEqual(sum(layout.shard_valid_elements(i) for i in range(dp)), 210)

    def test_working_gradient_shard_alias_does_not_allocate_again(self):
        layout = build_layout([param('weight', 1000)], dp_size=24)
        state = layout.state_bytes(0, include_reference=True)
        expected = 2 * layout.padded_elements + 4 * layout.padded_elements + 12 * layout.shard_valid_elements(0) + 2000
        self.assertEqual(state['total_allocated_bytes'], expected)
        self.assertEqual(state['main_gradient_shard_view']['additional_allocated_bytes'], 0)
        self.assertEqual(state['main_gradient_shard_view']['logical_bytes'], 4 * layout.shard_elements)
        self.assertTrue(state['full_working_gradient_buffer_retained'])

    def test_norm_counts_unique_TP_parameters_once_and_excludes_padding(self):
        layout = build_layout([param('sharded', 7), param('norm', 3, True)], dp_size=4, tp_size=2)
        norm = sum(layout.shard_valid_elements(dp, tp_rank=tp, norm_only=True)
            for tp in range(2) for dp in range(4))
        self.assertEqual(norm, 2 * 7 + 3)
        self.assertEqual(sum(layout.shard_valid_elements(dp) for dp in range(4)), 10)

    def test_numeric_norm_excludes_duplicate_TP_norms_and_padding(self):
        layout = build_layout([param('sharded', 2), param('norm', 3, True)], dp_size=4, tp_size=2)
        values = [{'sharded': [1, 2], 'norm': [5, 6, 7]},
                  {'sharded': [3, 4], 'norm': [5, 6, 7]}]
        total = 0
        for tp in range(2):
            for dp in range(4):
                for piece in layout.intersections(dp, tp_rank=tp, norm_only=True):
                    data = values[tp][piece['parameter_id']]
                    total += sum(v * v for v in data[piece['parameter_local_start']:piece['parameter_local_end']])
        self.assertEqual(total, 140)
        # Actual master/Adam state allocates only valid intersections, not the
        # padded communication shard. Across DP their count is the exact model.
        self.assertEqual(sum(layout.state_bytes(dp)['allocations']['master_FP32_shard'] for dp in range(4)), 5 * 4)

    def test_RS_AG_payloads_and_ring_wire_are_typed(self):
        layout = build_layout([param('weight', 1000)], dp_size=24)
        sizes = layout.collective_sizes()
        self.assertEqual(sizes['reduce_scatter']['full_input_bytes_per_rank'], 4 * layout.padded_elements)
        self.assertEqual(sizes['reduce_scatter']['output_shard_bytes_per_rank'] * 24, 4 * layout.padded_elements)
        self.assertEqual(sizes['parameter_all_gather']['input_shard_bytes_per_rank'] * 24, 2 * layout.padded_elements)
        rs, ag = layout.native_collectives(tuple(range(0, 48, 2)), group_id=2, round_id=1, actor_version=3)
        self.assertEqual((rs.collective, ag.collective), ('REDUCE_SCATTER', 'ALL_GATHER'))
        self.assertEqual(rs.payload_bytes, ag.payload_bytes * 2 * layout.dp_size)
        self.assertEqual(ag.metadata['logical_full_output_bytes_per_rank'],2*layout.padded_elements)
        self.assertEqual(ag.metadata['parameter_version_out'], 4)
        with self.assertRaises(ValueError):
            layout.native_collectives((0, 1), group_id=1, round_id=1, actor_version=0)

    def test_global_token_loss_times_DP_and_mean_RS_cancel_exactly(self):
        for dp in [2, 24, 48]:
            scaling = global_loss_scaling(dp, 313)
            self.assertEqual(Fraction(scaling['local_token_sum_loss_multiplier']) *
                Fraction(scaling['pre_RS_scale_for_native_SUM']), Fraction(1, 313))
            # Independent local token-gradient sums are unequal: reduce their
            # DP/T scaled contributions with a DP mean, not mean-local-means.
            local_sums = [Fraction(i * i + 1) for i in range(dp)]
            result = sum(x * Fraction(dp, 313) for x in local_sums) / dp
            self.assertEqual(result, sum(local_sums) / 313)
        with self.assertRaises(ValueError):
            global_loss_scaling(24, 0)

    def test_parameter_publication_requires_current_gradients_and_all_TP_columns(self):
        result = validate_publication(actor_version=3, gradient_version=3, reference_version=0,
            completed_parameter_all_gather_tp_groups=[0, 1], tp_size=2)
        self.assertEqual(result['actor_version_out'], 4)
        self.assertEqual(result['reference_version_out'], 0)
        with self.assertRaises(ValueError):
            validate_publication(actor_version=3, gradient_version=2, reference_version=0,
                completed_parameter_all_gather_tp_groups=[0, 1], tp_size=2)
        with self.assertRaises(ValueError):
            validate_publication(actor_version=3, gradient_version=3, reference_version=0,
                completed_parameter_all_gather_tp_groups=[0], tp_size=2)

    def test_real_3B_TP1_2_layouts_have_materialized_valid_shards(self):
        case = load_case(ROOT / 'configs/verl_sync_standard/qwen3b/case.json')
        for tp, dp in [(1, 48), (2, 24)]:
            local = replace(case, strategy=replace(case.strategy, tensor_parallel=tp))
            layout = layout_for_case(local, dp_size=dp)
            self.assertEqual(layout.tp_size, tp)
            self.assertTrue(all(layout.shard_valid_elements(i) > 0 for i in range(dp)))
            self.assertEqual(layout.padded_elements % dp, 0)
            self.assertLess(layout.state_bytes(0)['total_allocated_bytes'], 64_000_000_000)


if __name__ == '__main__':
    unittest.main()
