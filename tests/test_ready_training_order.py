import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/idea1_layered/legacy'))
from idea1_controller.native.regions import order_ready_training_jobs,RegionKernel

class ReadyOrderTests(unittest.TestCase):
 def jobs(self):
  before=[('region_data',[],['copy'],None),('old_policy',[],['teacher'],None),('reference',[],['ref'],None),('advantage',[],['adv'],None),('gradient_zero',[],['zero'],None)]
  jobs=[('training',[str(i)],['body'+str(i)],dict(batch_id=str(i),index=i,estimate_local_ns=cost)) for i,cost in enumerate((2.,10.,3.,10.))]
  return before+jobs
 def test_keeps_advantage_quorum_and_bodies_while_prioritizing_cost(self):
  jobs=self.jobs();ordered=order_ready_training_jobs(jobs,'ready_lpt')
  self.assertEqual(ordered[:5],jobs[:5]);self.assertEqual([x[3]['batch_id'] for x in ordered[5:]],['1','3','2','0'])
  for row in ordered[5:]:self.assertIn(row,jobs[5:]);self.assertIs(row,jobs[int(row[3]['batch_id'])+5])
  self.assertEqual([x[3]['batch_id'] for x in jobs[5:]],['0','1','2','3'])
 def test_base_cached_and_adaptive_actual_builders_use_same_ordering(self):
  from types import SimpleNamespace
  from rl_xy.qwen3_cost import Cost
  from idea1_controller.native.cached_complete import CachedCompleteKernel
  from idea1_controller.native.adaptive import AdaptiveKernel
  for cls in (RegionKernel,CachedCompleteKernel,AdaptiveKernel):
   with self.subTest(builder=cls.__name__):
    k=object.__new__(cls);k.resource_layout={'training_ready_order':'ready_lpt'};k.region_programs={}
    k.groups={0:[dict(trajectory_id=str(i),prompt_length=1,response_length=c) for i,c in enumerate((2,10,3,10))]}
    k.region_ranks={0:(0,1)};k.region_members={0:(0,)};k.logprob_mb=1;k.train_mb=1;k.tick=0;k.batch_manifest={};k.batch_state={}
    k.op=SimpleNamespace(vec=lambda *a:Cost(compute=1),state=lambda *a:{'layout':SimpleNamespace(padded_elements=16)},forward=lambda *a:1,training=lambda lengths,tp,tokens:tokens)
    k.memory=SimpleNamespace(compact_gradient=lambda tp:64,grad2=64)
    k._compile=lambda r,body:[dict(kind='local',compute=body)]
    k._batch_program=lambda r,stage,batch:[dict(kind='local',compute=sum(x['response_length'] for x in batch) if stage=='training' else 1)]
    k._batches=lambda items,stage,tp:[[x] for x in items]
    k._restore_gradient=lambda r:[];k.payload=lambda batch:32;k._synchronized=lambda r,i,steps:steps
    cls._build_region_program(k,0)
    rows=k.region_programs[0]
    self.assertEqual([x[3]['batch_id'] for x in rows if x[0]=='training'],['0:1','0:3','0:2','0:0'])
    self.assertEqual(rows[-1][0],'gradient_restore')
 def test_shared_finalize_orders_before_numbering_tp_gates(self):
  k=object.__new__(RegionKernel);k.resource_layout={'training_ready_order':'ready_lpt'};k.region_programs={}
  k._synchronized=lambda r,i,body:('sync',r,i,body)
  k._finalize_region_program(0,self.jobs())
  rows=k.region_programs[0]
  self.assertEqual([x[3]['batch_id'] for x in rows[5:]],['1','3','2','0'])
  self.assertEqual([x[2][2] for x in rows],list(range(len(rows))))
 def test_default_is_exact_unchanged_object(self):
  jobs=self.jobs();self.assertIs(order_ready_training_jobs(jobs,'input'),jobs)
 def test_refuses_cross_phase_reordering_and_invalid_work(self):
  jobs=self.jobs();jobs.insert(6,('gradient_restore',[],[],None))
  with self.assertRaises(ValueError):order_ready_training_jobs(jobs,'ready_lpt')
  jobs=self.jobs();jobs[5][3]['estimate_local_ns']=float('nan')
  with self.assertRaises(ValueError):order_ready_training_jobs(jobs,'ready_lpt')
if __name__=='__main__':unittest.main()
