"""Negative tests for final result tables; no constructed native timings as evidence."""
import copy,csv,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_idea1_workload import verify_comparison,sha

class FinalComparisonTests(unittest.TestCase):
    def fixture(self,root):
        baseline=root/'baseline.csv'
        with baseline.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['model','window','gpu_s','h0_s','original_generation_s','previous_innovation3_s','device_s']);w.writeheader()
            for m in ['4B','8B','14B']:w.writerow(dict(model=m,window='test1',gpu_s=100,h0_s=120,original_generation_s=90,previous_innovation3_s=80,device_s=60))
        final=[dict(model=m,window='test1',total_device_s=50) for m in ['4B','8B','14B']]
        rows=[dict(model=m,window='test1',device_s=50,gpu_s=100,h0_s=120,original_generation_s=90,previous_innovation3_s=80,old_complete_system_device_s=60,speedup_gpu_device=2,reduction_old_complete_device=1-50/60) for m in ['4B','8B','14B']]
        return dict(status='FINAL_NATIVE_RESULTS_READY_FOR_REVIEW',comparison=rows,baseline_csv_sha256=sha(baseline)),final,baseline
    def test_valid_table_and_all_baseline_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            report,final,base=self.fixture(Path(d));verify_comparison(report,final,base)
            for field in ['gpu_s','h0_s','original_generation_s','previous_innovation3_s','old_complete_system_device_s']:
                with self.subTest(field=field):
                    bad=copy.deepcopy(report);bad['comparison'][0][field]+=1
                    with self.assertRaises(AssertionError):verify_comparison(bad,final,base)
    def test_repeated_model_cannot_hide_missing_final_model(self):
        with tempfile.TemporaryDirectory() as d:
            report,final,base=self.fixture(Path(d));report['comparison'][2]=copy.deepcopy(report['comparison'][0])
            with self.assertRaises(AssertionError):verify_comparison(report,final,base)
    def test_stale_baseline_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            report,final,base=self.fixture(Path(d));base.write_text(base.read_text()+'\n')
            with self.assertRaises(AssertionError):verify_comparison(report,final,base)
    def test_device_time_and_ratio_must_match_case(self):
        with tempfile.TemporaryDirectory() as d:
            report,final,base=self.fixture(Path(d))
            for field in ['device_s','speedup_gpu_device','reduction_old_complete_device']:
                with self.subTest(field=field):
                    bad=copy.deepcopy(report);bad['comparison'][0][field]+=1
                    with self.assertRaises(AssertionError):verify_comparison(bad,final,base)
