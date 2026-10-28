from pathlib import Path
from dataclasses import replace
import unittest
from rl_xy.config import load_case
from rl_xy.colocation import StageBuilder
from rl_xy.communication_lowering import lower_communication_endpoints
from rl_xy.tensor_plan import build_tensor_plan,preflight_contract
from rl_xy.inference_lifetimes import trim_inference_lifetimes
ROOT=Path(__file__).resolve().parents[1]
class InferenceLifetimesTests(unittest.TestCase):
 def test_two_layers_reclaim_without_changing_work_or_outputs(self):
  c=load_case(ROOT/'configs/wafer4_fixture/case.json');t=c.workload.trajectories[0]
  c=replace(c,workload=replace(c.workload,trajectories=(t,)))
  for selected in [{'input','rollout','cleanup'},{'reference','cleanup'}]:
   b=StageBuilder(c,selected);b.build_round(0);p=build_tensor_plan(c,lower_communication_endpoints(c,b.operations),'/tmp/unused');old=preflight_contract(p.contract)
   q=trim_inference_lifetimes(p,c)
   self.assertEqual(q.operations,p.operations);self.assertEqual([t['nbytes'] for t in q.tensors],[t['nbytes'] for t in p.tensors])
   self.assertTrue(all(q.preflight_peaks[r]<=old[r] for r in old))
   self.assertGreater(sum(q.contract['inference_liveness']['removed_conservative_read_permissions'].values()),0)
   for t in q.tensors:
    if t['kind'] in ('token','log_prob','trajectory') and not t['persistent']:
     self.assertTrue(next(o for o in q.operations if o.node_id==t['released_node']).name=='release_round_temporaries')
 def test_early_scope_matches_late_temporary_lifetimes_and_costs(self):
  c=load_case(ROOT/'configs/wafer4_fixture/case.json');c=replace(c,workload=replace(c.workload,trajectories=c.workload.trajectories[:1]))
  b=StageBuilder(c,{'input','rollout','cleanup'});b.build_round(0);ops=lower_communication_endpoints(c,b.operations)
  ordinary=trim_inference_lifetimes(build_tensor_plan(c,ops,'/test'),c)
  early=trim_inference_lifetimes(build_tensor_plan(c,ops,'/test',inference_layer_scope=True),c)
  self.assertEqual(ordinary.preflight_peaks,early.preflight_peaks);self.assertEqual(ordinary.operations,early.operations)
  persistent={t['storage_id'] for t in ordinary.tensors if t['persistent']}
  for rank,row in ordinary.contract['ranks'].items():
   for nid,node in row['nodes'].items():
    other=early.contract['ranks'][rank]['nodes'][nid]
    self.assertEqual(set(node['reads'])-persistent,set(other['reads'])-persistent)
    self.assertEqual(node['frees'],other['frees'])
 def test_training_is_refused(self):
  c=load_case(ROOT/'configs/wafer4_fixture/case.json');b=StageBuilder(c,{'train','cleanup'});b.build_round(0);p=build_tensor_plan(c,lower_communication_endpoints(c,b.operations),'/tmp/unused')
  with self.assertRaises(ValueError):trim_inference_lifetimes(p,c)
if __name__=='__main__':unittest.main()
