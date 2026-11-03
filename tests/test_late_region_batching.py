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
if __name__=='__main__':unittest.main()
