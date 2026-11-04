import unittest,sys
from pathlib import Path
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/idea1_layered/legacy'))
from idea1_controller.native.mesh_cohorts import MeshCohortKernel
from idea1_controller.native.inference_cohorts import CohortKernel

class NoFuture(dict):
 def __getitem__(self,key):
  if key in ('response_length','finish_reason','eos'):raise AssertionError('future output read')
  return super().__getitem__(key)
 def get(self,key,*args):
  if key in ('response_length','finish_reason','eos'):raise AssertionError('future output read')
  return super().get(key,*args)
class Catalog:
 def decode(self,tp,batch,context):return 100.
 def phase(self,stage,tp,lengths):return 100.
class MeshPolicyTests(unittest.TestCase):
 def kernel(self):
  k=object.__new__(MeshCohortKernel);k.tick=100;k.mesh_decisions_enabled=True
  k.op=SimpleNamespace(hw=SimpleNamespace(d2d_bytes_per_ns_per_direction=2000.,d2d_latency_ns=100.,dram_bytes_per_ns=1000000.))
  raw=dict(schema='rlxy.mesh-observed/2',width=6,height=9,native_ticks_per_ns=1000,policy='adaptive_xy_yx',edge_fields=['src','dst','announced_not_started_bytes','queued_here_bytes','actual_queue_wait_ns','committed_service_ns'],edges=[[1,2,1000000000,1000000,100000.,1000000.]])
  k.observe_mesh(raw,100);k.pending_enabled=True;k.paired_tp1=False
  k.inference_owner={0:0,1:1,9:9};k.inference_members={r:(r,) for r in k.inference_owner};k.initial_inference_members=k.inference_members.copy()
  k.services={};k.pending_cohort_leases={};k.logical_roles={r:'ROLLOUT' for r in k.inference_owner};k.trained=set();k.waiting={r:True for r in k.inference_owner}
  items=[NoFuture(group_id='new',trajectory_id='new:'+str(i),prompt_length=10) for i in range(2)]
  k.rep={r:SimpleNamespace(ranks=(2*r,2*r+1),cap=2,active=[],pending=None,requests=deque()) for r in k.inference_owner}
  k.rep[0].requests=deque(items);k.rep[0].active=[(x,11,0) for x in items]
  k.inference_busy=lambda j:False;k.leased=lambda j:False;k.same_domain=lambda a,b:False
  k.helper_history={'new':(1000,)*128};k.mesh_catalog=Catalog()
  return k
 def test_observed_congestion_can_select_farther_helper(self):
  k=self.kernel();choice=k.pending_choice(0)
  self.assertEqual(choice[1],9);self.assertEqual(len(choice[2]),2)
  event=k.mesh_decision_log[-1];self.assertGreater(event['prediction']['gain_ns'],0)
  self.assertEqual(len(event['prediction']['paths']),2)
 def test_cannot_borrow_busy_or_unavailable_helper(self):
  k=self.kernel();k.waiting[9]=False
  self.assertIsNone(k.pending_choice(0))
 def test_mesh_disabled_keeps_base_decision(self):
  k=self.kernel();k.mesh_decisions_enabled=False
  with patch.object(CohortKernel,'pending_choice',return_value='base') as f:
   self.assertEqual(k.pending_choice(0),'base');f.assert_called_once()
 def test_stale_observation_keeps_base_decision(self):
  k=self.kernel();k.tick=101
  with patch.object(CohortKernel,'pending_choice',return_value='base'):
   self.assertEqual(k.pending_choice(0),'base')
 def training_kernel(self):
  k=self.kernel();k.region_of={0:0,1:1,9:9};k.region_members={r:[r] for r in k.region_of}
  k.trained={1,9};k.helper_jobs={};k.resource_layout={'training_help_admission':'region-ready'}
  k.training_scope='wafer';k.done=set();k.ids=set(range(32))
  job=dict(batch_id='b',estimate_local_ns=20000000.,index=0,payload_bytes_per_lane=1000000)
  k.train_queues={0:deque([job])};k.batch_state={'b':'queued'}
  k.portable_training_job=lambda job:dict(estimate_ns=1000000.)
  return k
 def test_training_uses_observed_congestion(self):
  k=self.training_kernel();dst,job=k.training_help_choice(0)
  self.assertEqual(dst,9);self.assertEqual(job['batch_id'],'b')
  self.assertEqual(k.mesh_decision_log[-1]['kind'],'training-proposed')
 def test_training_respects_restored_region_and_lease(self):
  k=self.training_kernel();k.region_members[9]=[1,9,10]
  k.leased=lambda j:j==1
  self.assertIsNone(k.training_help_choice(0))
 def test_training_unstarted_ticket_only(self):
  k=self.training_kernel();k.batch_state['b']='running'
  self.assertIsNone(k.training_help_choice(0))
 def test_training_does_not_cross_region_in_domain_mode(self):
  k=self.training_kernel();k.training_scope='domain'
  self.assertIsNone(k.training_help_choice(0))
 def test_training_stale_falls_back(self):
  k=self.training_kernel();k.tick+=1
  with patch.object(CohortKernel,'training_help_choice',return_value='base'):
   self.assertEqual(k.training_help_choice(0),'base')
 def test_expansion_restores_candidate_set_on_failure(self):
  k=self.kernel();k.elastic_groups=[(0,1),(0,9)];original=k.elastic_groups
  with patch.object(CohortKernel,'inference_resize_choice',side_effect=RuntimeError('injected')):
   with self.assertRaises(RuntimeError):k.inference_resize_choice(0,k.logical_roles)
  self.assertIs(k.elastic_groups,original)

if __name__=='__main__':unittest.main()
