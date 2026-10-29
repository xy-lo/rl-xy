import math
import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace

from rl_xy.communication_service import operator_cost_identity, protocol_overhead_ns
from rl_xy.qwen3_cost import Block, Cost
from rl_xy.qwen3_native import Plan
from rl_xy.qwen3_sequence import compile_blocks


def profile():
    return {'communication_service_overhead': {
        'schema_version': 'rlxy.gpu_protocol_overhead/1',
        'status': 'VALIDATION_CANDIDATE', 'calibrated_ranks': [2, 3],
        'allow_rank_extrapolation': False,
        'by_kind': {'ALL_REDUCE': {'fixed_ns': 10000., 'per_peer_ns': 200.,
                                  'per_input_byte_ns': .01},
                    'GATHER': {'fixed_ns': 7000.}, 'BCAST': {'fixed_ns': 3000.}}}}


class DummyOp:
    platform = 'gpu'

    def __init__(self, settings):
        self.profile = settings

    def vec(self, *args):
        return Cost()


@dataclass(frozen=True)
class Hardware:
    sram_bytes_per_core: int = 1250000
    d2d_latency_ns: int = 100


class CommunicationServiceTests(unittest.TestCase):
    def test_absent_profile_preserves_legacy_cost(self):
        self.assertEqual(protocol_overhead_ns(DummyOp({}), 8, 'ALL_REDUCE', 4096), 0)
        self.assertEqual(protocol_overhead_ns(DummyOp(None), 8, 'ALL_REDUCE', 4096), 0)

    def test_single_rank_needs_no_protocol_cost(self):
        self.assertEqual(protocol_overhead_ns(DummyOp(profile()), 1, 'ALL_REDUCE', 4096), 0)

    def test_rank_extrapolation_must_be_explicit(self):
        p = profile(); op = DummyOp(p)
        with self.assertRaisesRegex(ValueError, 'rank count'):
            protocol_overhead_ns(op, 8, 'ALL_REDUCE', 4096)
        p['communication_service_overhead']['allow_rank_extrapolation'] = True
        self.assertAlmostEqual(protocol_overhead_ns(op, 8, 'ALL_REDUCE', 4096), 11440.96)

    def test_missing_primitive_is_not_silently_borrowed(self):
        with self.assertRaisesRegex(ValueError, 'missing collective'):
            protocol_overhead_ns(DummyOp(profile()), 2, 'REDUCE_SCATTER', 4096)

    def test_protocol_bandwidth_residual_uses_collective_wire_volume(self):
        p = profile(); model = p['communication_service_overhead']
        model['by_kind'] = {kind: {'per_wire_byte_ns': 1.} for kind in
                            ('ALL_REDUCE', 'ALL_GATHER', 'REDUCE_SCATTER')}
        op = DummyOp(p)
        self.assertEqual(protocol_overhead_ns(op, 3, 'ALL_GATHER', 300), 600)
        self.assertEqual(protocol_overhead_ns(op, 3, 'ALL_REDUCE', 300), 400)
        self.assertEqual(protocol_overhead_ns(op, 3, 'REDUCE_SCATTER', 300), 200)

    def test_invalid_residual_cannot_hide_double_counting(self):
        for bad in (-1., float('nan'), float('inf')):
            p = profile(); p['communication_service_overhead']['by_kind']['ALL_REDUCE']['fixed_ns'] = bad
            with self.assertRaises(ValueError):
                protocol_overhead_ns(DummyOp(p), 2, 'ALL_REDUCE', 4096)

    def test_measured_curve_has_explicit_bounds_and_no_double_residual(self):
        p = profile(); row = {
            'wire_residual_knots': [[1024, 10.], [2048, 30.], [4096, 50.]],
            'wire_residual_boundary_policy': 'constant_below_nonnegative_last_above'}
        p['communication_service_overhead']['by_kind']['ALL_REDUCE'] = row
        op = DummyOp(p)
        self.assertEqual(protocol_overhead_ns(op, 2, 'ALL_REDUCE', 512), 10)
        self.assertEqual(protocol_overhead_ns(op, 2, 'ALL_REDUCE', 1536), 20)
        self.assertEqual(protocol_overhead_ns(op, 2, 'ALL_REDUCE', 8192), 90)
        self.assertEqual(protocol_overhead_ns(op, 3, 'ALL_REDUCE', 3072), 50)
        row['fixed_ns'] = 1.
        with self.assertRaisesRegex(ValueError, 'also charge'):
            protocol_overhead_ns(op, 2, 'ALL_REDUCE', 4096)

    def test_compact_and_expanded_charge_same_residual_once(self):
        op = DummyOp(profile()); ranks = (0, 1)
        for kind in ('ALL_REDUCE', 'GATHER', 'BCAST'):
            plan = Plan(op); plan.comm(ranks, kind, 4096, 'probe')
            charged = [s['compute_ns'] for s in plan.stats.values()
                       if s['name'].endswith('.protocol_overhead')]
            steps = compile_blocks(op, ranks, [Block('probe', Cost(), kind, 4096)])
            local_costs = [s['compute'] for s in steps if s['kind'] == 'local' and s['compute']]
            self.assertEqual(len(charged), 1)
            self.assertEqual(local_costs, charged)
            self.assertEqual(charged[0], protocol_overhead_ns(op, 2, kind, 4096))

    def test_runtime_hardware_and_provider_change_cache_identity(self):
        op = SimpleNamespace(hw=Hardware(), profile=None,
                             local_timing_identity={'version': 1})
        first = operator_cost_identity(op)
        self.assertEqual(first, operator_cost_identity(op))
        op.hw = replace(op.hw, d2d_latency_ns=200)
        self.assertNotEqual(first['sha256'], operator_cost_identity(op)['sha256'])
        op.hw = Hardware(); op.local_timing_identity['version'] = 2
        self.assertNotEqual(first['sha256'], operator_cost_identity(op)['sha256'])
        self.assertEqual(first['fields']['local_timing_identity']['version'], 1)

    def test_gpu_protocol_is_part_of_cost_identity(self):
        op = SimpleNamespace(hw=Hardware(), profile=profile())
        first = operator_cost_identity(op)
        op.profile['communication_service_overhead']['by_kind']['BCAST']['fixed_ns'] += 1
        self.assertNotEqual(first['sha256'], operator_cost_identity(op)['sha256'])


if __name__ == '__main__':
    unittest.main()
