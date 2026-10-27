import copy
from dataclasses import replace
from pathlib import Path
import unittest
from rl_xy.config import load_case
from rl_xy.workflow import build_workflow
from rl_xy.communication_lowering import lower_communication_endpoints
from rl_xy.tensor_plan import build_tensor_plan,finalize_tensor_costs,preflight_contract

ROOT=Path(__file__).resolve().parents[1]


def planned(rounds=1):
    c=load_case(ROOT/'configs/toy_grpo_2die/case.json',rounds_override=rounds)
    p=build_tensor_plan(c,lower_communication_endpoints(c,build_workflow(c)),'/tmp/unit-only-resources')
    finalize_tensor_costs(p,c)
    return c,p


class TensorResourceTests(unittest.TestCase):
    def test_complete_tape_gradient_shapes_and_two_round_reuse(self):
        c,p=planned();_,two=planned(2)
        self.assertEqual(p.preflight_peaks,two.preflight_peaks)
        self.assertGreater(p.preflight_peaks['d0.dram'],16352)
        self.assertTrue(all(t['persistent'] or t['released_node'] is not None for t in p.tensors))
        for rank in (0,1):
            for phase in ('train_A_backward','train_B_backward'):
                partials=[t for t in p.tensors if t['rank']==rank and phase in t['tensor_id'] and t['kind']=='parameter_gradient']
                self.assertEqual(sum(t['nbytes'] for t in partials),3232)
            self.assertEqual(sum(t['nbytes'] for t in p.tensors if t['rank']==rank and t['kind']=='kv'),224)

    def test_network_arrival_is_not_destination_dram_ready(self):
        c,p=planned()
        for op in p.operations:
            if op.kind!='compute':
                for rank in op.participant_ranks:
                    node=p.contract['ranks'][str(rank)]['nodes'][str(op.node_id)]
                    self.assertEqual(node['ready_outputs'],[])
        delayed=[t for t in p.tensors if t.get('ready_node') is not None and t['ready_node']!=t['created_node']]
        self.assertTrue(delayed)
        self.assertTrue(all(t['ready_node']>t['created_node'] for t in delayed))

    def test_missing_tensor_and_duplicate_free_fail_closed(self):
        _,p=planned()
        bad=copy.deepcopy(p.contract)
        first=next(iter(bad['ranks']['0']['nodes'].values()))
        first['reads'].append('missing_tensor')
        with self.assertRaises(ValueError):preflight_contract(bad)
        bad=copy.deepcopy(p.contract)
        last=list(bad['ranks']['0']['nodes'].values())[-1]
        last['frees'].append('already_gone')
        with self.assertRaises(ValueError):preflight_contract(bad)

    def test_dram_oom_is_rejected_before_astra_launch(self):
        _,p=planned()
        bad=copy.deepcopy(p.contract)
        bad['ranks']['0']['dram_capacity']=20000
        with self.assertRaises(ValueError):preflight_contract(bad)

    def test_reservation_is_not_double_counted_or_leaked(self):
        _,p=planned()
        for r in p.contract['ranks'].values():
            self.assertEqual(sum(n['reserve_before']+n['reserve_after'] for n in r['nodes'].values()),0)
        bad=copy.deepcopy(p.contract)
        first=next(n for n in bad['ranks']['0']['nodes'].values() if n['reserve_before']>0)
        first['reserve_before']=0
        with self.assertRaises(ValueError):preflight_contract(bad)

    def test_endpoint_services_are_real_rank_local_nodes(self):
        _,p=planned()
        endpoints=[o for o in p.operations if o.metadata.get('local_transfer')]
        self.assertEqual(len(endpoints),2*98)
        for op in endpoints:
            self.assertEqual(op.duration_ns,op.metadata['cost_estimate']['local_memory_ns'])
            self.assertEqual(op.flops,0)

    def test_reference_write_and_missing_residual_gradient_are_visible(self):
        _,p=planned()
        bad=copy.deepcopy(p.contract)
        first=next(iter(bad['ranks']['0']['nodes'].values()))
        sid=next(s for s in bad['ranks']['0']['initial_objects'] if '.reference.' in s)
        first['ready_outputs'].append(sid)
        with self.assertRaisesRegex(ValueError,'immutable reference'):preflight_contract(bad)
        self.assertEqual(sum(o.name.endswith('.residual_ffn_grad_add') for o in p.operations),4)
        self.assertEqual(sum(o.name.endswith('.residual_attn_grad_add') for o in p.operations),4)

    def test_final_cost_domain_controls_local_memory_lease(self):
        _,p=planned()
        hosts=[o for o in p.operations if o.metadata.get('execution_domain')=='host']
        self.assertTrue(hosts)
        for op in hosts:
            for rank in op.participant_ranks:
                self.assertFalse(p.contract['ranks'][str(rank)]['nodes'][str(op.node_id)]['local_memory'])


if __name__=='__main__':unittest.main()
