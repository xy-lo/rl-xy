import copy
import hashlib,json
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
import unittest
from rl_xy.gpu_model import load_gpu_profile,GPUCostProvider,effective_gpu_case,cta_mapping,budget_audit
from rl_xy.config import load_case
from rl_xy.operator_ir import Matmul,KernelSpec

ROOT=Path(__file__).resolve().parents[1]


class GPUModelTest(unittest.TestCase):
    def setUp(self):
        self.p=load_gpu_profile(ROOT/'configs/gpu/a100_48_reference.json')
        self.case=effective_gpu_case(load_case(ROOT/'configs/wafer4_fixture/case.json'),self.p)

    def test_ragged_ctas_conserve_and_leave_small_grid_idle(self):
        p=Matmul(5,7,11,batches=3)
        f,t,w,r,wr,n=cta_mapping(p,108,164*1024,4,4,4)
        self.assertEqual(sum(f),p.flops);self.assertEqual(n,12)
        self.assertEqual(sum(x>0 for x in f),12)
        self.assertEqual(sum(t),r+wr)
        self.assertTrue(all(x<=164*1024 for x in w))

    def test_private_smem_capacity_cannot_be_pooled(self):
        with self.assertRaises(ValueError):cta_mapping(Matmul(128,128,128),108,64,128,128,32)

    def test_adaptive_cta_never_worse_than_fixed_legal_reference(self):
        for shape in ((100,100,64),(100,2432,896),(896,75968,100)):
            spec=KernelSpec('adaptive',matmuls=(Matmul(*shape),))
            auto=GPUCostProvider(self.p).estimate(spec,self.case.hardware)
            fixed=copy.deepcopy(self.p);fixed.pop('cta_candidates')
            base=GPUCostProvider(fixed).estimate(spec,self.case.hardware)
            self.assertLessEqual(auto.duration_ns,base.duration_ns)
            self.assertIn(auto.partitions[0]['selected_cta_tile'],self.p['cta_candidates'])

    def test_gpu_profile_aggregate_rate_and_memory_bottleneck(self):
        e=GPUCostProvider(self.p).estimate(KernelSpec('gemm',matmuls=(Matmul(256,256,256),)),self.case.hardware)
        self.assertEqual(len(e.core_workspace_bytes),108)
        self.assertEqual(sum(e.core_workspace_bytes),e.sram_workspace_bytes)
        self.assertGreaterEqual(e.local_memory_ns*self.p['hbm_bytes_per_ns'],e.dram_read_bytes+e.dram_write_bytes)
        self.assertEqual(e.duration_ns,e.compute_ns+e.local_memory_ns)
        self.assertEqual(e.partitions[0]['flops_sum'],2*256**3)

    def test_precision_sensitivity_not_same_as_real_fp32_peak(self):
        spec=KernelSpec('fp32',matmuls=(Matmul(1024,1024,1024,a_bytes=4,b_bytes=4),))
        strict=GPUCostProvider(self.p).estimate(spec,self.case.hardware)
        optimistic=load_gpu_profile(ROOT/'configs/gpu/a100_48_optimistic.json')
        other=GPUCostProvider(optimistic).estimate(spec,self.case.hardware)
        self.assertLess(other.duration_ns,strict.duration_ns)
        self.assertFalse(optimistic['physical_sku_prediction'])

    def test_budget_is_near_not_equal_and_gpu_not_stronger_everywhere(self):
        h={'cores_per_die':256,'gemm_flops_per_ns':1020,'gemm_fp32_flops_per_ns':512,
           'vector_ops_per_ns':256,'dram_bytes_per_die':64000000000,'dram_bytes_per_ns':2000,'sram_bytes_per_core':1250000}
        a=budget_audit({'dies':54},{'effective_hardware':h},self.p)
        self.assertTrue(a['gpu_no_less_fp16_and_dram_capacity'])
        self.assertAlmostEqual(a['gpu_minus_wafer_percent']['dram_capacity_bytes'],100/9)
        self.assertGreater(a['gpu_minus_wafer_percent']['bf16_peak_flops_per_ns'],6)
        self.assertLess(a['gpu_minus_wafer_percent']['dram_bandwidth_bytes_per_ns'],0)
        self.assertLess(a['gpu_minus_wafer_percent']['private_sram_capacity_bytes'],0)
    def test_measured_gemm_has_no_second_hbm_charge_and_domain_is_closed(self):
        mat=Matmul(64,128,256);key=hashlib.sha256(json.dumps(asdict(mat),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        p={**self.p,'gemm_measurements':{key:{'duration_ns':5000}}}
        e=GPUCostProvider(p).estimate(KernelSpec('measured',matmuls=(mat,)),self.case.hardware)
        self.assertEqual(e.local_memory_ns,0)
        self.assertEqual(e.duration_ns,5000+self.p['kernel_launch_ns'])
        with self.assertRaises(ValueError):GPUCostProvider(p).estimate(KernelSpec('unseen',matmuls=(Matmul(65,128,256),)),self.case.hardware)
