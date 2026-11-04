"""Negative tests for final result tables; no constructed native timings as evidence."""
import copy,hashlib,csv,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_idea1_workload import verify_comparison,verify_deployment,sha

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

class DeploymentBindingTests(unittest.TestCase):
    def fixture(self,packed=True):
        fields=dict(adaptive_training_batch_max=8,training_tail_fine_samples=8,late_region_batch_cap_per_pair=1,training_ready_order='input') if packed else {}
        assignment=dict(fields,forecasts={'g':{'score':1.0}})
        raw=json.dumps(assignment,indent=2).encode();inf=json.dumps({'groups':{'0':[0]}}).encode()
        digest=lambda b:hashlib.sha256(b).hexdigest()
        deploy=dict(model='4B',window='test1',hashes={'selected-plan.json':digest(raw),'inference-layout.json':digest(inf),'forecast.json':digest(json.dumps(assignment['forecasts'],indent=2).encode())})
        if packed:deploy.update(adaptive_training_batch_max=8,training_tail_fine_samples=8,late_region_batch_cap=1,training_ready_order='input',runtime_training_packing_unscored=True)
        return deploy,[dict(model='4B',window='test1'),raw+b'\n',assignment,inf+b'\n']
    def test_legacy_and_new_runtime_options_accept_exact_recording(self):
        for packed in (False,True):
            d,args=self.fixture(packed);verify_deployment(d,*args)
    def test_false_options_and_input_hashes_are_rejected(self):
        d,args=self.fixture()
        for key,value in [('adaptive_training_batch_max',4),('training_tail_fine_samples',0),('late_region_batch_cap',2),('training_ready_order','ready_lpt'),('runtime_training_packing_unscored',False),('training_tail_fine_samples',True)]:
            with self.subTest(key=key,value=value):
                bad=copy.deepcopy(d);bad[key]=value
                with self.assertRaises(AssertionError):verify_deployment(bad,*args)
        for key in d['hashes']:
            bad=copy.deepcopy(d);bad['hashes'][key]='0'*64
            with self.assertRaises(AssertionError):verify_deployment(bad,*args)
        # A mutated execution flag cannot be hidden behind the original input.
        bad_args=copy.deepcopy(args);bad_args[2]['training_tail_fine_samples']=0
        bad=copy.deepcopy(d);bad['training_tail_fine_samples']=0
        with self.assertRaises(AssertionError):verify_deployment(bad,*bad_args)
    def test_resource_bank_is_bound_with_normalized_fraction(self):
        d,args=self.fixture();d['resource_candidates']=['box4_top@1:row1']
        verify_deployment(d,*args,constraint=['box4_top@1.0:row1'])
        with self.assertRaises(AssertionError):verify_deployment(d,*args,constraint=['box4@1.0:row1'])
        d['resource_candidates']=[]
        with self.assertRaises(AssertionError):verify_deployment(d,*args)
