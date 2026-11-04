import sys,unittest
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/idea1_layered/legacy'))
from idea1_controller.native.adaptive import AdaptiveKernel

@dataclass
class Certificate:
 peak_bytes:int=1
class Memory:
 def batches(self,items,stage,tp,maximum):
  return [items[i:i+maximum] for i in range(0,len(items),maximum)]
 def phase(self,*args):return Certificate()
class LateBatchTests(unittest.TestCase):
 def kernel(self,cap=2,complete=False):
  k=object.__new__(AdaptiveKernel);k.train_mb=4;k.logprob_mb=8;k.memory=Memory();k.op=SimpleNamespace(capacity=10)
  k.resource_layout={'late_region_batch_cap_per_pair':cap};k.done={0,1} if complete else {0};k.ids={0,1};k.tick=100
  k.targets={str(i):0 for i in range(8)};k.allocation_events=[];k.mb_choices=[];return k
 def items(self):return [dict(trajectory_id=str(i)) for i in range(8)]
 def test_early_region_keeps_original_training_efficiency(self):
  k=self.kernel();b=list(k._batches(self.items(),'training',2));self.assertEqual([len(x) for x in b],[4,4]);self.assertFalse(k.mb_choices[0]['late_region_cap_applied'])
 def test_observed_global_eos_splits_only_training_with_pair_scaled_cap(self):
  for tp,sizes in [(2,[2,2,2,2]),(4,[4,4]),(8,[8])]:
   k=self.kernel(complete=True);b=list(k._batches(self.items(),'training',tp));self.assertEqual([len(x) for x in b],sizes)
   self.assertEqual([x for batch in b for x in batch],self.items());self.assertEqual(k.mb_choices[0]['late_region_cap_applied'],tp<8);self.assertTrue(k.mb_choices[0]['late_region_cap_eligible'])
 def test_teacher_unchanged_and_default_does_not_add_control_metadata(self):
  k=self.kernel(complete=True);self.assertEqual([len(x) for x in k._batches(self.items(),'old_policy',2)],[8])
  k=self.kernel(cap=0,complete=True);self.assertEqual([len(x) for x in k._batches(self.items(),'training',2)],[4,4]);self.assertNotIn('late_region_cap_applied',k.mb_choices[0])
class AdaptivePackingTests(unittest.TestCase):
 kernel=LateBatchTests.kernel
 items=LateBatchTests.items
 def packed(self,complete=False):
  k=self.kernel(cap=1);k.resource_layout['adaptive_training_batch_max']=8
  k.done={x['trajectory_id'] for x in self.items()};k.ids=set(k.done) if complete else set(k.done)|{'future'}
  return k
 def test_local_eos_can_pack_while_other_regions_still_generate(self):
  k=self.packed();self.assertEqual([len(b) for b in k._batches(self.items(),'training',2)],[8])
  self.assertEqual(k.mb_choices[0]['baseline_maximum'],4);self.assertFalse(k.mb_choices[0]['late_region_cap_eligible'])
 def test_global_eos_fine_tail_overrides_bulk_packing(self):
  k=self.packed(True);self.assertEqual([len(b) for b in k._batches(self.items(),'training',2)],[1]*8)
  self.assertTrue(all(x['maximum']==1 and x['original_maximum']==8 for x in k.mb_choices))
 def test_packing_leaves_teacher_program_unchanged(self):
  k=self.packed();self.assertEqual([len(b) for b in k._batches(self.items(),'old_policy',2)],[8])
  self.assertNotIn('adaptive_training_batch_max',k.mb_choices[0])
 def test_packing_rejects_unobserved_local_response(self):
  k=self.packed();k.done.remove('7')
  with self.assertRaises(ValueError):list(k._batches(self.items(),'training',2))
  self.assertEqual(k.allocation_events,[])
 def test_larger_batch_still_must_pass_phase_capacity(self):
  k=self.packed();k.memory.phase=lambda *args:Certificate(peak_bytes=11)
  with self.assertRaises(AssertionError):list(k._batches(self.items(),'training',2))
 def test_config_rejects_invalid_maximum_or_nonadaptive_builder(self):
  from idea1_controller.native.regions import RegionKernel
  for maximum in [True,8.,16]:
   with self.subTest(maximum=maximum),self.assertRaises(ValueError):
    object.__new__(RegionKernel).configure_regions({}, {}, {'adaptive_training_batch_max':maximum})
  with self.assertRaises(ValueError):object.__new__(RegionKernel).configure_regions({}, {}, {'adaptive_training_batch_max':8})

class PackingAuditTests(unittest.TestCase):
 def fixture(self):
  import sys
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
  from audit_idea1_inference import audit_training_packing
  ids=[str(i) for i in range(8)]
  return audit_training_packing,[dict(stage='training',tp=2,maximum=8,actual=8,adaptive_training_batch_max=8,baseline_maximum=4,packing_tids=ids,decision_tick=10)],{'adaptive_training_batch_max':8},{t:5 for t in ids}
 def test_independent_eos_and_sample_conservation(self):
  audit,c,l,e=self.fixture();self.assertEqual(audit(c,l,e,4)['larger_than_baseline_batches'],1)
  e['0']=11
  with self.assertRaises(AssertionError):audit(c,l,e,4)
 def test_rejects_larger_bound_missing_sample_and_false_baseline(self):
  import copy
  audit,c,l,e=self.fixture()
  for mutate in [lambda x:x.update(maximum=9),lambda x:x.update(packing_tids=x['packing_tids'][:-1],actual=7),lambda x:x.update(baseline_maximum=2)]:
   bad=copy.deepcopy(c);mutate(bad[0])
   with self.assertRaises(AssertionError):audit(bad,l,e,4)
 def test_legacy_has_no_new_declaration(self):
  audit,c,l,e=self.fixture();self.assertIsNone(audit([dict(stage='training')],{},e,4))
  with self.assertRaises(AssertionError):audit(c,{},e,4)

if __name__=='__main__':unittest.main()
