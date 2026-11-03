import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/workload_planner'))
from final_campaign import select_tune_policy

class FinalCampaignTests(unittest.TestCase):
 def case(self,root,name,seconds,window='tune',complete=True):
  d=root/name;(d/'native-run').mkdir(parents=True)
  (d/'native-run/result.json').write_text(json.dumps(dict(model='4B',window=window,requests=1536,completed=1536,trained_samples=1536,capacity_passed=True,total_s=seconds)))
  (d/'inference-audit.json').write_text(json.dumps(dict(status='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT')))
  (root/(name+'-status.json')).write_text(json.dumps(dict(complete=complete,stages=[dict(returncode=0)])))
  return dict(run=name,home_policy='score',planner_mode='one_scene')
 def test_selects_faster_completed_policy_and_retains_tie_order(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);a=self.case(root,'a',72);b=self.case(root,'b',71);b['home_policy']='service'
   rows,best=select_tune_policy(root,[a,b],73.8);self.assertEqual(best['run'],'b');self.assertEqual(len(rows),2)
 def test_incomplete_peer_blocks_even_when_first_is_fast(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);a=self.case(root,'a',70);b=self.case(root,'b',69,complete=False)
   with self.assertRaises(ValueError):select_tune_policy(root,[a,b],73.8)
 def test_final_window_time_forbidden(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);a=self.case(root,'a',60,window='test1')
   with self.assertRaises(AssertionError):select_tune_policy(root,[a],73.8)
 def test_faster_than_gpu_is_insufficient_to_weaken_existing_ceiling(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);a=self.case(root,'a',75)
   with self.assertRaises(ValueError):select_tune_policy(root,[a],73.8)
 def test_failed_audit_and_nan_are_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);a=self.case(root,'a',float('nan'))
   with self.assertRaises(ValueError):select_tune_policy(root,[a],73.8)
   (root/'a/inference-audit.json').write_text(json.dumps(dict(status='FAIL')))
   with self.assertRaises(AssertionError):select_tune_policy(root,[a],73.8)
if __name__=='__main__':unittest.main()
