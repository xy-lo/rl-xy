import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import test_qwen_execution as qwen_test

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('comparison_under_test',ROOT/'scripts/compare_gpu_wafer.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


class MatchedComparisonTest(unittest.TestCase):
    def fixture(self,directory,cap=None):
        compiler,model,trace=qwen_test.QwenExecutionTest().compiler_inputs(directory)
        gpu=directory/'gpu';gpu.mkdir()
        trace_path=gpu/'capture.jsonl';trace_path.write_bytes(trace.read_bytes())
        # The fixture has two steps; compilation explicitly selects step 1.
        bundle=directory/'bundle';bundle.mkdir()
        compiler(qwen_test.CASE.parent,model,trace_path,'framework_capture',1,['g0','g1'],bundle/'inputs',cap)
        (gpu/'launch.log').write_text('explicit synthetic unit fixture, not GPU measurement\n')
        (gpu/'report.json').write_text(json.dumps({'status':'PASS','exit_status':0,
            'model_config_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
            'launch_log_sha256':hashlib.sha256((gpu/'launch.log').read_bytes()).hexdigest()}))
        cfg={'algorithm':{'adv_estimator':'grpo','norm_adv_by_std_in_grpo':False},
            'actor_rollout_ref':{'actor':{'loss_agg_mode':'token-mean','ppo_epochs':1,'ppo_mini_batch_size':2},
                'model':{'enable_gradient_checkpointing':False,'use_remove_padding':True}}}
        (gpu/'execution_step_1.json').write_text(json.dumps({'timing_raw_s':{'step':.05},'resolved_training_config':cfg}))
        (bundle/'metrics.json').write_text(json.dumps({'total_simulated_time_ns':1000000,'optimizer_updates':1,
            'rounds':1,'dies':54,'cores_per_die':256,'data_parallel':27,'tensor_parallel':2,'assumptions':{}}))
        (bundle/'summary.json').write_text(json.dumps({'status':'PASS'}))
        return gpu,bundle

    def test_matching_contract_still_labels_prediction_not_measured_speedup(self):
        with tempfile.TemporaryDirectory() as d:
            gpu,bundle=self.fixture(Path(d));r=comparison.compare(gpu,bundle)
            self.assertEqual(r['status'],'MATCHED_SCENARIO_PREDICTION')
            self.assertEqual(r['scenario_time_ratio_NOT_validated_speedup'],50)
            self.assertFalse(r['hardware_calibrated']);self.assertFalse(r['equal_resource_budget'])

    def test_counterfactual_cap_rejects_ratio(self):
        with tempfile.TemporaryDirectory() as d:
            gpu,bundle=self.fixture(Path(d),1);r=comparison.compare(gpu,bundle)
            self.assertEqual(r['status'],'NOT_COMPARABLE');self.assertIsNone(r['scenario_time_ratio_NOT_validated_speedup'])

    def test_wrong_gpu_identity_and_log_and_update_contract_rejected(self):
        mutations=[('gpu','report.json','model_config_sha256','wrong'),
            ('gpu','report.json','launch_log_sha256','wrong'),('bundle','metrics.json','optimizer_updates',2)]
        for target,name,key,value in mutations:
            with tempfile.TemporaryDirectory() as d:
                gpu,bundle=self.fixture(Path(d));path=(gpu if target=='gpu' else bundle)/name
                raw=json.loads(path.read_text());raw[key]=value;path.write_text(json.dumps(raw))
                with self.subTest(key=key):
                    r=comparison.compare(gpu,bundle);self.assertEqual(r['status'],'NOT_COMPARABLE')
                    self.assertIsNone(r['scenario_time_ratio_NOT_validated_speedup'])
