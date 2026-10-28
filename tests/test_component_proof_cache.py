"""Independent invalid-clock and mutation cases for the serial proof cache."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from rl_xy.component_proof_cache import ComponentProofCache
from rl_xy.local_chain_contraction import contract_local_chains, expand_native_events
from rl_xy.tensor_plan import TensorPlan
from rl_xy.types import AstraEvent, Operation


def source_fixture():
    ops = [Operation(1, 'a', 0, 'forward', 'a', 'compute', duration_ns=2, participant_ranks=(0,)),
        Operation(2, 'b', 0, 'forward', 'b', 'compute', duration_ns=3, participant_ranks=(0,), dependencies=(1,)),
        Operation(3, 'peer', 0, 'prepare', 'peer', 'compute', duration_ns=1, participant_ranks=(1,)),
        Operation(4, 'send', 0, 'transfer', 'send', 'p2p', payload_bytes=8, src_rank=0, dst_rank=1,
            participant_ranks=(0, 1), dependencies_by_rank={0: (2,), 1: (3,)}),
        Operation(5, 'finish0', 0, 'end', 'finish0', 'compute', duration_ns=4, participant_ranks=(0,), dependencies=(4,)),
        Operation(6, 'finish1', 0, 'end', 'finish1', 'compute', duration_ns=2, participant_ranks=(1,), dependencies=(4,))]
    def node():
        return dict(reads=['w'], allocations={}, ready_outputs=[], frees=[], workspace_bytes=2,
            sram_capacity=16, local_memory=True, reserve_before=0, reserve_after=0)
    ranks = {str(r): dict(initial_objects={'w': 4}, dram_capacity=64,
        nodes={str(o.node_id): node() for o in ops if r in o.participant_ranks}) for r in (0, 1)}
    ranks['0']['nodes']['1'].update(allocations={'t': 8}, ready_outputs=['t'])
    ranks['0']['nodes']['2'].update(reads=['w', 't'], frees=['t'])
    return TensorPlan(dict(schema_version='1.0', log_prefix='unused', ranks=ranks), [], ops, {})


def observed(result, offset=0, extra_network_delay=0):
    times = {(0, 1): (0, 5000), (1, 3): (0, 1000),
        (0, 4): (5000, 20000+extra_network_delay), (1, 4): (1000, 20000+extra_network_delay),
        (0, 5): (20000+extra_network_delay, 24000+extra_network_delay),
        (1, 6): (20000+extra_network_delay, 22000+extra_network_delay)}
    events = []
    for op in result.macro_plan.operations:
        for rank in op.participant_ranks:
            start, end = times[rank, op.node_id];start += offset;end += offset
            kind = 4 if op.kind == 'compute' else 5 if rank == op.src_rank else 6
            events.append(AstraEvent(rank, op.node_id, op.op_id, op.round_id, op.phase, kind,
                start, end, end-start, start/1000, end/1000, (end-start)/1000))
    return events


def reclock(event, start, end):
    return replace(event, issue_tick=start, callback_tick=end, duration_ticks=end-start,
        issue_ns=start/1000, callback_ns=end/1000, duration_ns=(end-start)/1000)


class ComponentProofCacheTests(unittest.TestCase):
    def setUp(self):
        self.result = contract_local_chains(source_fixture())

    def test_one_full_replay_then_distinct_clocks_and_network_intervals_reuse(self):
        cache = ComponentProofCache();cache.prepare(self.result)
        with patch('rl_xy.component_proof_cache.expand_native_events', wraps=expand_native_events) as replay:
            first = cache.verify(self.result, observed(self.result))
            second = cache.verify(self.result, observed(self.result, offset=13000, extra_network_delay=8000))
            self.assertEqual(replay.call_count, 1)
        self.assertEqual(first['mode'], 'full_template_replay')
        self.assertEqual(second['mode'], 'reused_verified_serial_template')
        self.assertEqual(first['ranks'], second['ranks'])
        self.assertEqual(first['logical_events_represented'], second['logical_events_represented'])
        self.assertEqual(first['expanded_logical_events'], 7)
        self.assertEqual(second['expanded_logical_events'], 0)
        self.assertEqual(second['logical_resource_nodes_replayed_this_invocation'], 0)
        self.assertFalse(second['logical_events_materialized'])
        self.assertTrue(second['all_native_macro_clocks_validated'])
        self.assertTrue(second['clock_bound_by_verified_serial_template'])
        self.assertFalse(second['clock_bound_to_expanded_native_events'])
        self.assertNotEqual(first['native_macro_event_stream_sha256'], second['native_macro_event_stream_sha256'])
        self.assertNotIn('expanded_event_stream_sha256', second)
        self.assertEqual(second['template_expanded_event_stream_sha256'], first['expanded_event_stream_sha256'])
        report = cache.finalize()
        self.assertEqual((report['unique_templates'], report['invocations'], report['full_template_replays']), (1, 2, 1))

    def test_semantically_equal_but_distinct_objects_do_not_skip_full_verification(self):
        cache = ComponentProofCache();clone = deepcopy(self.result)
        with patch('rl_xy.component_proof_cache.expand_native_events', wraps=expand_native_events) as replay:
            cache.verify(self.result, observed(self.result))
            proof = cache.verify(clone, observed(clone))
            self.assertEqual(replay.call_count, 2)
        self.assertEqual(proof['mode'], 'full_template_replay')
        self.assertEqual(cache.finalize()['unique_templates'], 2)

    def test_reuse_rejects_missing_duplicate_unknown_and_wrong_rank_event(self):
        cache = ComponentProofCache();cache.verify(self.result, observed(self.result))
        good = observed(self.result, offset=1000)
        variants = [good[:-1], good+[good[0]], [replace(good[0], node_id=999)]+good[1:],
            [replace(good[0], rank=9)]+good[1:]]
        for bad in variants:
            with self.subTest(coverage=[(e.rank, e.node_id) for e in bad]):
                with self.assertRaisesRegex(ValueError, 'coverage|duplicate/unknown'):
                    cache.verify(self.result, bad)

    def test_all_invocations_validate_compute_kind_and_singleton_duration(self):
        for prime in (False, True):
            for node in (1, 5):
                cache = ComponentProofCache()
                if prime: cache.verify(self.result, observed(self.result))
                good = observed(self.result)
                with self.assertRaisesRegex(ValueError, 'duration differs'):
                    cache.verify(self.result, [reclock(e, e.issue_tick, e.callback_tick+1)
                        if e.rank == 0 and e.node_id == node else e for e in good])
                with self.assertRaisesRegex(ValueError, 'kind differs'):
                    cache.verify(self.result, [replace(e, node_type=7)
                        if e.rank == 0 and e.node_id == node else e for e in good])

    def test_reuse_rejects_rank_overlap_and_malformed_tick_derived_fields(self):
        cache = ComponentProofCache();cache.verify(self.result, observed(self.result))
        good = observed(self.result)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            cache.verify(self.result, [reclock(e, 4000, e.callback_tick)
                if e.rank == 0 and e.node_id == 4 else e for e in good])
        with self.assertRaisesRegex(ValueError, 'duration field'):
            cache.verify(self.result, [replace(good[0], duration_ticks=1)]+good[1:])
        with self.assertRaisesRegex(ValueError, 'model clocks'):
            cache.verify(self.result, [replace(good[0], issue_ns=1)]+good[1:])
        with self.assertRaisesRegex(ValueError, 'nonnegative integers'):
            cache.verify(self.result, [reclock(good[0], -1, 4999)]+good[1:])
        with self.assertRaisesRegex(ValueError, 'kind differs'):
            cache.verify(self.result, [replace(e, node_type=7) if e.node_id == 4 else e for e in good])

    def test_source_contract_operation_certificate_and_macro_mutations_fail_final_guard(self):
        def contract(r): r.original_plan.contract['ranks']['0']['nodes']['1']['allocations']['t'] = 7
        def operation(r): r.original_plan.operations[0] = replace(r.original_plan.operations[0], duration_ns=3)
        def certificate(r): r.certificate['logical_operation_count'] = 999
        def macro_contract(r): r.macro_plan.contract['ranks']['0']['dram_capacity'] += 1
        def macro_operation(r): r.macro_plan.operations[0] = replace(r.macro_plan.operations[0], duration_ns=6)
        for mutation in (contract, operation, certificate, macro_contract, macro_operation):
            with self.subTest(mutation=mutation.__name__):
                result = contract_local_chains(source_fixture());cache = ComponentProofCache()
                cache.prepare(result);cache.verify(result, observed(result));mutation(result)
                # Reuse cannot weaken the final whole-run success gate.
                with self.assertRaisesRegex(ValueError, 'changed during run'):
                    cache.finalize()

    def test_primed_source_mutation_and_unverified_prepared_template_fail(self):
        cache = ComponentProofCache();cache.prepare(self.result)
        self.result.original_plan.contract['ranks']['0']['nodes']['1']['allocations']['t'] = 7
        with self.assertRaisesRegex(ValueError, 'changed after contraction'):
            cache.verify(self.result, observed(self.result))
        fresh = contract_local_chains(source_fixture());cache = ComponentProofCache();cache.prepare(fresh)
        with self.assertRaisesRegex(ValueError, 'no verified invocation'):cache.finalize()

    def test_certificate_cannot_claim_missing_serial_proof_or_changed_block_mapping(self):
        self.result.certificate['strict_per_rank_serialization_proved'] = False
        with self.assertRaisesRegex(ValueError, 'strict per-rank'):ComponentProofCache().prepare(self.result)
        fresh = contract_local_chains(source_fixture());fresh.certificate['blocks'] = []
        with self.assertRaisesRegex(ValueError, 'block mapping'):ComponentProofCache().prepare(fresh)

    def test_finalization_closes_cache_and_empty_graph_is_valid(self):
        cache = ComponentProofCache();cache.verify(self.result, observed(self.result))
        report = cache.finalize();self.assertEqual(cache.finalize(), report)
        with self.assertRaisesRegex(ValueError, 'already finalized'):
            cache.verify(self.result, observed(self.result))
        self.assertEqual(ComponentProofCache().finalize()['invocations'], 0)


if __name__ == '__main__':unittest.main()
