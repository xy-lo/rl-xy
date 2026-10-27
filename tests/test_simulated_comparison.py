import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import test_comparison as measured_test
from rl_xy.gpu_model import load_gpu_profile

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('sim_comparison',ROOT/'scripts/compare_simulated_platforms.py')
comparison=importlib.util.module_from_spec(s);s.loader.exec_module(comparison)


class SimulatedComparisonTest(unittest.TestCase):
    def fixture(self,path):
        _,w=measured_test.MatchedComparisonTest().fixture(path)
        (w/'inputs/gpu_provenance.json').write_text('{}')
        h={'cores_per_die':256,'gemm_flops_per_ns':1020,'gemm_fp32_flops_per_ns':512,
           'vector_ops_per_ns':256,'dram_bytes_per_die':64000000000,'dram_bytes_per_ns':2000,'sram_bytes_per_core':1250000}
        (w/'resolved_config.json').write_text(json.dumps({'effective_hardware':h}))
        g=path/'sim_gpu';g.mkdir();shutil.copytree(w/'inputs',g/'inputs')
        m=json.loads((w/'metrics.json').read_text());m.update(platform='GPU',total_simulated_time_ns=2000000,
            gpu_profile=load_gpu_profile(ROOT/'configs/gpu/a100_48_reference.json'))
        (g/'metrics.json').write_text(json.dumps(m));(g/'summary.json').write_text('{"status":"PASS"}')
        for b in (w,g):(b/'manifest.json').write_text(json.dumps({'exit_status':0,'software':{'rl_xy_source_tree_sha256':'synthetic-unit-fixture'}}))
        return w,g

    def test_result_is_simulated_near_budget_not_observed_or_proven(self):
        with tempfile.TemporaryDirectory() as d:
            w,g=self.fixture(Path(d));r=comparison.compare(w,g)
            self.assertEqual(r['status'],'NEAR_BUDGET_SIMULATION_COMPARISON')
            self.assertEqual(r['gpu_over_wafer_time_ratio_SCENARIO_ONLY'],2)
            self.assertFalse(r['observed_GPU_step_seconds_used_for_prediction']);self.assertFalse(r['feasibility_proven'])
            m=json.loads((g/'metrics.json').read_text());m['total_simulated_time_ns']=500000
            (g/'metrics.json').write_text(json.dumps(m))
            self.assertFalse(comparison.compare(w,g)['wafer_faster_in_this_scenario'])

    def test_different_capture_and_non_simulated_platform_reject_ratio(self):
        for mutation in ('capture','platform'):
            with tempfile.TemporaryDirectory() as d:
                w,g=self.fixture(Path(d))
                if mutation=='capture':(g/'inputs/compilation.json').write_text('{}')
                else:
                    m=json.loads((g/'metrics.json').read_text());m['platform']='OBSERVED_GPU'
                    (g/'metrics.json').write_text(json.dumps(m))
                r=comparison.compare(w,g);self.assertEqual(r['status'],'NOT_COMPARABLE')
                self.assertIsNone(r['gpu_over_wafer_time_ratio_SCENARIO_ONLY'])

    def test_insufficient_gpu_budget_rejects_ratio(self):
        with tempfile.TemporaryDirectory() as d:
            w,g=self.fixture(Path(d));m=json.loads((g/'metrics.json').read_text())
            m['gpu_profile']['dram_bytes_per_gpu']=32000000000;(g/'metrics.json').write_text(json.dumps(m))
            self.assertEqual(comparison.compare(w,g)['status'],'NOT_COMPARABLE')
