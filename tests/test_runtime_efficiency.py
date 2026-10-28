"""Hand-counted resource contracts for representation-only runtime changes."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl_xy.config import load_case
from rl_xy.communication_lowering import lower_communication_endpoints
from rl_xy.resource_encoding import compact_rank, expand_rank
from rl_xy.resource_validation import validate_resources
from rl_xy.tensor_plan import build_tensor_plan, preflight_contract
from rl_xy.workflow import build_workflow


ROOT = Path(__file__).resolve().parents[1]


def resource_example(prefix):
    """24 persistent bytes, overlapping 8/4-byte temporaries, then RMW."""
    actor = 'd0.actor.weight'
    grad = 'd0.gradient.weight'

    def node(reads, allocations, outputs, frees, workspace, before=0, after=0):
        return dict(reads=reads, allocations=allocations, ready_outputs=outputs,
                    frees=frees, workspace_bytes=sum(workspace),
                    core_workspace_bytes=workspace, sram_capacity=64,
                    local_memory=True, reserve_before=before, reserve_after=after)

    rank = dict(initial_objects={actor: 16, grad: 8}, dram_capacity=40,
                cores_per_die=2, sram_bytes_per_core=64, nodes={
                    '1': node([actor], {'a': 8}, ['a'], [], [4, 0], 4, -4),
                    '2': node(['a', actor], {'b': 4}, ['b'], ['a'], [0, 8]),
                    '3': node(['b', grad], {}, [grad], ['b'], [4, 4]),
                })
    contract = dict(ranks={'0': rank}, log_prefix=str(prefix))
    # These counts are independent of the production resource replay.
    state = [(0, 'initial', 0, 24, [0, 0], 0, 0),
             (1, 'issue', 1, 32, [4, 0], 1, 4),
             (1, 'callback', 2, 32, [0, 0], 0, 0),
             (2, 'issue', 3, 36, [0, 8], 2, 0),
             (2, 'callback', 4, 28, [0, 0], 0, 0),
             (3, 'issue', 5, 28, [4, 4], 3, 0),
             (3, 'callback', 6, 24, [0, 0], 0, 0)]
    rows = [dict(node_id=n, event=e, tick=t, dram_bytes=d,
                 sram_bytes=sum(w), core_sram_bytes=w,
                 local_memory_owner=o, reserved_bytes=r)
            for n, e, t, d, w, o, r in state]
    events = [SimpleNamespace(rank=0, node_id=n, issue_tick=2*n-1,
                              callback_tick=2*n) for n in (1, 2, 3)]
    return contract, rows, events


class RuntimeEfficiencyTests(unittest.TestCase):
    def setUp(self):
        case = load_case(ROOT/'configs/toy_grpo_2die/case.json')
        self.case = replace(case, hardware=replace(case.hardware,
                            cores_per_die=2, sram_bytes_per_core=64),
                            model=replace(case.model, tie_word_embeddings=False))

    def replay(self, directory, contract, rows, events):
        path = Path(str(directory/'resources')+'.0.jsonl')
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        plan = SimpleNamespace(contract=contract, tensors=[])
        return validate_resources(plan, self.case, events, gradient_visits=[])

    def test_preflight_counts_overlap_reservation_and_existing_storage_rmw(self):
        contract, _, _ = resource_example('/unused')
        before = copy.deepcopy(contract)
        self.assertEqual(preflight_contract(contract), {'d0.dram': 36})
        self.assertEqual(contract, before)

    def test_preflight_still_rejects_invalid_lifetimes_and_reservations(self):
        examples = [
            ('DRAM admission overflow', lambda r: r['nodes']['1'].update(reserve_before=9)),
            ('read absent/not ready', lambda r: r['nodes']['1'].update(ready_outputs=[])),
            ('duplicate allocation', lambda r: r['nodes']['2']['allocations'].update({'d0.gradient.weight': 4})),
            ('double free', lambda r: r['nodes']['3']['frees'].append('a')),
            ('READY without storage', lambda r: r['nodes']['1']['ready_outputs'].append('missing')),
            ('immutable reference write', lambda r: (
                r['initial_objects'].update({'d0.reference.weight': 1}),
                r['nodes']['3']['ready_outputs'].append('d0.reference.weight'))),
            ('private SRAM overflow', lambda r: r['nodes']['1'].update(
                core_workspace_bytes=[65, 0], workspace_bytes=65)),
        ]
        for message, mutate in examples:
            with self.subTest(message=message):
                contract, _, _ = resource_example('/unused')
                mutate(contract['ranks']['0'])
                with self.assertRaisesRegex(ValueError, message):
                    preflight_contract(contract)

    def test_delta_output_has_exact_expected_fields_and_owns_its_data(self):
        contract, _, _ = resource_example('/unused')
        rank = contract['ranks']['0'];before = copy.deepcopy(rank)
        compact = compact_rank(rank)
        expected_deltas = [
            dict(base_node=0, add=['d0.actor.weight'], remove=[]),
            dict(base_node=1, add=['a'], remove=[]),
            dict(base_node=2, add=['b', 'd0.gradient.weight'], remove=['a', 'd0.actor.weight']),
        ]
        for number, delta in enumerate(expected_deltas, 1):
            original = rank['nodes'][str(number)]
            row = compact['nodes'][str(number)]
            vector = original['core_workspace_bytes']
            key = hashlib.sha256(json.dumps(vector, separators=(',', ':')).encode()).hexdigest()
            wanted = {k: copy.deepcopy(v) for k, v in original.items()
                      if k not in ('reads', 'core_workspace_bytes')}
            wanted.update(read_delta=delta, core_workspace_profile=key)
            self.assertEqual(row, wanted)
            self.assertEqual(compact['core_workspace_profiles'][key], vector)
        self.assertEqual(expand_rank(compact), rank)
        self.assertEqual(rank, before)
        compact['nodes']['1']['allocations']['a'] = 999
        self.assertEqual(rank['nodes']['1']['allocations']['a'], 8)

    def test_shared_workspace_is_serialized_once_without_mutable_aliases(self):
        contract, _, _ = resource_example('/unused');rank = contract['ranks']['0']
        rank['nodes']['2']['core_workspace_bytes'] = [4, 0]
        rank['nodes']['2']['workspace_bytes'] = 4
        compact = compact_rank(rank)
        self.assertEqual(len(compact['core_workspace_profiles']), 2)
        expanded = expand_rank(compact)
        self.assertEqual(expanded, rank)
        expanded['nodes']['1']['core_workspace_bytes'][0] = 999
        self.assertEqual(expanded['nodes']['2']['core_workspace_bytes'], [4, 0])
        rank['nodes']['1']['core_workspace_bytes'][0] = 111
        key = compact['nodes']['1']['core_workspace_profile']
        self.assertEqual(compact['core_workspace_profiles'][key], [4, 0])

    def test_delta_corruption_and_workspace_tampering_remain_errors(self):
        contract, _, _ = resource_example('/unused')
        compact = compact_rank(contract['ranks']['0'])
        bad = copy.deepcopy(compact)
        bad['nodes']['2']['read_delta']['base_node'] = 0
        with self.assertRaisesRegex(ValueError, 'delta chain'):
            expand_rank(bad)
        bad = copy.deepcopy(compact)
        key = bad['nodes']['1']['core_workspace_profile']
        bad['core_workspace_profiles'][key][0] += 1
        with self.assertRaisesRegex(ValueError, 'profile hash mismatch'):
            expand_rank(bad)

    def test_streaming_replay_matches_hand_counted_snapshots_and_peaks(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name);contract, rows, events = resource_example(directory/'resources')
            # Path.read_text is intentionally forbidden: validation must still
            # inspect every line without materializing the whole log file.
            with patch.object(Path, 'read_text', side_effect=AssertionError('whole file read')):
                checks, snapshots, peaks = self.replay(directory, contract, rows, events)
            self.assertTrue(all(c.passed for c in checks))
            self.assertEqual(peaks, {'d0.dram': 36, 'd0.sram_workspace_sum': 8,
                                     'd0.c0.sram': 4, 'd0.c1.sram': 8})
            self.assertEqual(len(snapshots), 14)
            self.assertEqual([s.allocated_bytes for s in snapshots[::2]], [24, 32, 32, 36, 28, 28, 24])
            self.assertEqual([s.allocated_bytes for s in snapshots[1::2]], [0, 4, 0, 8, 0, 8, 0])

    def test_streaming_replay_does_not_accept_false_counts_or_missing_callbacks(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            for corruption in ('false_count', 'missing_callback', 'wrong_tick', 'unsafe_read', 'duplicate_allocation'):
                with self.subTest(corruption=corruption):
                    contract, rows, events = resource_example(directory/'resources')
                    if corruption == 'false_count': rows[3]['dram_bytes'] += 1
                    elif corruption == 'missing_callback': rows.pop()
                    elif corruption == 'wrong_tick': events[1].callback_tick += 1
                    elif corruption == 'unsafe_read': contract['ranks']['0']['nodes']['2']['reads'].append('missing')
                    else: contract['ranks']['0']['nodes']['2']['allocations']['d0.gradient.weight'] = 4
                    checks, _, _ = self.replay(directory, contract, rows, events)
                    self.assertFalse(checks[0].passed)

    def test_payload_index_keeps_all_live_payloads_and_drops_retired_rounds(self):
        case = load_case(ROOT/'configs/toy_grpo_2die/case.json', rounds_override=2)
        plan = build_tensor_plan(case, lower_communication_endpoints(case, build_workflow(case)), '/unused')
        tensors = {t['storage_id']: t for t in plan.tensors}
        kinds = {'token', 'trajectory', 'reward', 'advantage', 'log_prob'}
        tested = 0
        for rank, state in plan.contract['ranks'].items():
            live = set(state['initial_objects'])
            for op in plan.operations:
                if int(rank) not in op.participant_ranks:
                    continue
                node = state['nodes'][str(op.node_id)]
                if op.kind == 'compute' and op.metadata.get('forward_prefix') and not op.metadata.get('local_transfer'):
                    expected = {s for s in live if tensors[s]['kind'] in kinds and tensors[s]['round_id'] == op.round_id}
                    self.assertTrue(expected <= set(node['reads']))
                    tested += 1
                live.update(node['allocations']);live.difference_update(node['frees'])
            self.assertEqual(live, set(state['initial_objects']))
        self.assertGreater(tested, 20)
        self.assertTrue(preflight_contract(plan.contract))


if __name__ == '__main__':
    unittest.main()
