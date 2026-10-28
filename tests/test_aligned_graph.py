from dataclasses import replace
from pathlib import Path
import unittest
from rl_xy.config import load_case
from rl_xy.aligned_graph import build_aligned_graph
from rl_xy.baseline_contract import BaselinePolicy,policy_fingerprint
from rl_xy.cost_model import AnalyticalCostProvider
ROOT=Path(__file__).resolve().parents[1]

def fixture(tp=2,mb=2,rounds=1,compact=True,conservative=False):
 c=load_case(ROOT/'configs/wafer4_fixture/case.json');a=c.workload.trajectories[0]
 c=replace(c,rounds=rounds,hardware=replace(c.hardware,dram_bytes_per_die=10**7),
  workload=replace(c.workload,trajectories=tuple(replace(a,trajectory_id=f'X{i}',group_id=f'G{i//2}',reward=float(i%2)) for i in range(8))))
 p=BaselinePolicy(prompt_batch_size=4,ppo_mini_batch_prompts=4,train_tp=tp,rollout_tp=1,
  train_micro_batch_trajectories=mb,use_kl_loss=True,kl_coefficient=.001)
 g=build_aligned_graph(c,6,p,AnalyticalCostProvider(),'/test',active=4,compact_components=compact,conservative_bridge_reads=conservative)
 return c,p,g

class AlignedGraphTests(unittest.TestCase):
 def test_microbatches_have_real_aggregation_and_same_DO_states(self):
  states=[];counts=[]
  for mb in (1,2,4):
   c,p,g=fixture(mb=mb,compact=False)
   self.assertTrue(all(x.passed for x in g.component_checks))
   states.append([r['initial_objects'] for r in g.plan.contract['ranks'].values()])
   counts.append(len(g.plan.operations))
   for op in g.plan.operations:
    if op.name.endswith('_loss'):
     self.assertEqual(op.metadata['local_loss_numerator_scale'],2)
     self.assertEqual(op.metadata['global_denominator'],24)
  self.assertEqual(states[0],states[1]);self.assertEqual(states[1],states[2])
  self.assertGreater(counts[0],counts[1]);self.assertGreater(counts[1],counts[2])
 def test_idle_ranks_have_only_marker_and_no_state_or_barrier(self):
  _,_,g=fixture(compact=False)
  for rank in (4,5):
   self.assertFalse(g.plan.contract['ranks'][str(rank)]['initial_objects'])
   self.assertEqual(g.plan.preflight_peaks[f'd{rank}.dram'],0)
   self.assertTrue(all(o.metadata.get('idle_rank_marker') for o in g.plan.operations if rank in o.participant_ranks))
 def test_AG_input_shard_and_RS_full_input_bytes(self):
  _,_,g=fixture(compact=False);d=g.audit['distributed_optimizer']
  ag=[o for o in g.plan.operations if o.name.startswith('do_parameter_AG_') and o.kind=='collective']
  rs=[o for o in g.plan.operations if o.name.startswith('do_gradient_RS_') and o.kind=='collective']
  self.assertEqual(len(ag),2);self.assertEqual(len(rs),2)
  self.assertTrue(all(o.payload_bytes*2==o.metadata['full_output_bytes_per_rank'] for o in ag))
  self.assertTrue(all(o.payload_bytes==2*ag[0].metadata['full_output_bytes_per_rank'] for o in rs))
  export=[o for o in g.plan.operations if o.name.startswith('export_') and o.kind=='collective']
  self.assertTrue(export)
  self.assertTrue(all(o.collective=='ALL_GATHER' for o in export))
 def test_version_publication_and_reference_immutable_two_rounds(self):
  _,_,g=fixture(rounds=2,compact=False)
  self.assertEqual([o.metadata['actor_version_out'] for o in g.plan.operations if o.name=='Adam_owned_shard_0'],[1,2])
  self.assertEqual(len([o for o in g.plan.operations if o.name=='publish_actor_version']),2)
  self.assertFalse(any('.reference.' in s for c in g.plan.contract['ranks'].values() for n in c['nodes'].values() for s in n['ready_outputs']))
 def test_compaction_preserves_network_payloads_and_compute_sum(self):
  for tp,mb in [(1,1),(1,2),(2,1),(2,4)]:
   _,p,a=fixture(tp,mb,compact=False);_,_,b=fixture(tp,mb,compact=True)
   def net(g):return [(o.kind,o.collective,o.payload_bytes,o.participant_ranks,o.src_rank,o.dst_rank) for o in g.plan.operations if o.kind!='compute']
   self.assertEqual(net(a),net(b))
   self.assertEqual(sum(o.duration_ns for o in a.plan.operations if o.kind=='compute'),sum(o.duration_ns for o in b.plan.operations if o.kind=='compute'))
   self.assertLess(len(b.plan.operations),len(a.plan.operations))
   self.assertEqual(b.audit['policy_fingerprint'],policy_fingerprint(p))
   self.assertTrue(b.component_expansions)
   self.assertLess(b.audit['component_template_count'],b.audit['component_invocation_count'])
 def test_training_and_rollout_memory_representative_states(self):
  _,_,g=fixture(compact=False)
  self.assertTrue(any('owned_intersections' in o.metadata for o in g.plan.operations if o.name.startswith('Adam_owned_shard_')))
  norms=[o for o in g.plan.operations if o.name.startswith('do_gradient_square_sum_')]
  self.assertTrue(all(o.metadata['scalar_slot']==0 and o.metadata['other_slots_zero_initialized'] for o in norms))
 def test_same_local_rewards_do_not_alias_different_global_advantages(self):
  from rl_xy.grpo_math import group_advantages
  c=load_case(ROOT/'configs/wafer4_fixture/case.json');a=c.workload.trajectories[0];rewards=[0.,1.,0.,1.,0.,1.,2.,3.]
  ts=tuple(replace(a,trajectory_id=f'X{i}',group_id=f'G{i//4}',reward=r) for i,r in enumerate(rewards))
  c=replace(c,hardware=replace(c.hardware,dram_bytes_per_die=10**7),workload=replace(c.workload,trajectories=ts))
  p=BaselinePolicy(prompt_batch_size=2,responses_per_prompt=4,ppo_mini_batch_prompts=2,train_tp=2,rollout_tp=1,train_micro_batch_trajectories=2,use_kl_loss=True,kl_coefficient=.001)
  g=build_aligned_graph(c,4,p,AnalyticalCostProvider(),'/test',active=4,compact_components=False)
  expected=dict(zip([t.trajectory_id for t in ts],group_advantages(rewards,[t.group_id for t in ts],normalize_std=True)))
  observed={}
  for o in g.plan.operations:
   if o.name.endswith('_loss') or o.name.endswith('.loss'):
    inverse={v:k for k,v in o.metadata['source_to_canonical_trajectories'].items()}
    for term in o.metadata.get('training_batch_terms',[]):observed[inverse[term['trajectory_id']]]=term['advantage']
  self.assertEqual(expected,observed)
  p=replace(p,train_micro_batch_trajectories=1)
  g=build_aligned_graph(c,4,p,AnalyticalCostProvider(),'/test',active=4,compact_components=False)
  observed={}
  for o in g.plan.operations:
   if o.name.endswith('_loss'):
    source=next(iter(o.metadata['source_to_canonical_trajectories']))
    observed[source]=o.metadata['advantage']
  self.assertEqual(expected,observed)
 def test_immutable_external_read_pruning_changes_no_allocation_cost_or_dependency(self):
  for compact in (False,True):
   _,_,old=fixture(compact=compact,conservative=True);_,_,new=fixture(compact=compact)
   self.assertEqual(old.plan.operations,new.plan.operations)
   self.assertEqual(old.plan.preflight_peaks,new.plan.preflight_peaks)
   for r,c in old.plan.contract['ranks'].items():
    self.assertEqual(c['initial_objects'],new.plan.contract['ranks'][r]['initial_objects'])
    for nid,n in c['nodes'].items():
     after=new.plan.contract['ranks'][r]['nodes'][nid]
     self.assertTrue(set(after['reads'])<=set(n['reads']))
     self.assertEqual({k:v for k,v in after.items() if k!='reads'},{k:v for k,v in n.items() if k!='reads'})
   counts=new.audit['external_input_read_permission_counts']
   self.assertGreater(counts['before'],counts['after'])
 def test_external_carrier_write_reallocation_or_early_free_is_rejected(self):
  import copy
  from rl_xy.aligned_graph import audit_immutable_carriers
  _,_,g=fixture();sid=next(t['storage_id'] for t in g.plan.tensors if t['storage_id'].startswith('trajectory.'))
  audit_immutable_carriers(g.plan.contract,{sid})
  for field in ('ready_outputs','allocations','frees'):
   contract=copy.deepcopy(g.plan.contract);row=contract['ranks']['0'];nid=next(iter(row['nodes']))
   if field=='allocations':row['nodes'][nid][field][sid]=4
   else:row['nodes'][nid][field].append(sid)
   with self.assertRaises(ValueError):audit_immutable_carriers(contract,{sid})
 def test_generation_prefill_projects_only_last_hidden_row_teacher_unchanged(self):
  _,_,g=fixture(compact=False)
  generation=[o for o in g.plan.operations if o.name.endswith('.lm_head_local') and o.metadata.get('colocation_stage')=='generation']
  self.assertTrue(generation)
  for o in generation:
   self.assertEqual(o.kernel_spec.matmuls[0].m,o.metadata['batch_request_count'])
  selections=[o for o in g.plan.operations if o.name.endswith('.last_hidden_row_select')]
  self.assertTrue(selections);self.assertTrue(all(o.metadata['selected_hidden_rows']==1 for o in selections))
  heads=[o for o in g.plan.operations if o.name.endswith('.lm_head_local') and o.metadata.get('colocation_stage') in ('old_policy','reference')]
  self.assertTrue(all(o.kernel_spec.matmuls[0].m==4 for o in heads))
 def test_component_validation_ids_are_unique_across_cache_variants(self):
  _,_,g=fixture(rounds=2)
  ids=[c.check_id for c in g.component_checks]
  self.assertEqual(len(ids),len(set(ids)))
 def test_unsupported_configuration_rejected(self):
  with self.assertRaises(ValueError):fixture(tp=2,mb=3)
if __name__=='__main__':unittest.main()
