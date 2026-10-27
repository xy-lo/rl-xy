import dataclasses
from pathlib import Path
import unittest
from rl_xy.config import load_case
from rl_xy.operator_ir import KernelSpec, Matmul, TensorSpec
from rl_xy.mapping import os_mapping, choose_os_mapping
from rl_xy.cost_model import AnalyticalCostProvider, LookupCostProvider, profile_key
from rl_xy.workflow import build_workflow
from rl_xy.mapped_costs import apply_mapped_costs

ROOT = Path(__file__).resolve().parents[1]


class MappingTests(unittest.TestCase):
    def test_all_compute_shapes_and_hardware_sensitivity(self):
        case=load_case(ROOT/'configs/toy_grpo_2die/case.json')
        ops=build_workflow(case)
        base=apply_mapped_costs(case,ops)
        slow=apply_mapped_costs(dataclasses.replace(case,hardware=dataclasses.replace(case.hardware,dram_bytes_per_ns=1)),ops)
        small=apply_mapped_costs(dataclasses.replace(case,hardware=dataclasses.replace(case.hardware,sram_bytes_per_core=64)),ops)
        self.assertTrue(all(o.kernel_spec is not None for o in base if o.flops))
        self.assertGreater(sum(o.duration_ns for o in slow),sum(o.duration_ns for o in base))
        self.assertTrue(any(a.metadata.get('cost_estimate')!=b.metadata.get('cost_estimate') for a,b in zip(base,small)))
        self.assertEqual([(o.op_id,o.dependencies_by_rank) for o in base],[(o.op_id,o.dependencies_by_rank) for o in ops])
        for o in base:
            if o.kernel_spec is not None:
                self.assertEqual(o.flops,sum(p.flops for p in o.kernel_spec.matmuls))

    def test_hand_calculated_os_traffic(self):
        p=Matmul(8,8,8)
        a=os_mapping(p,(4,4,4),128)
        b=os_mapping(p,(2,2,2),32)
        self.assertEqual((a.flops,a.workspace_bytes,a.dram_read_bytes,a.dram_write_bytes),(1024,128,512,128))
        self.assertEqual((b.flops,b.workspace_bytes,b.dram_read_bytes,b.dram_write_bytes),(1024,32,1024,128))

    def test_ragged_matches_independent_loops(self):
        p=Matmul(7,9,5,batches=2)
        a=os_mapping(p,(4,4,4),128)
        reads=writes=0
        for batch in range(2):
            for i in range(0,7,4):
                for j in range(0,9,4):
                    mi,nj=min(4,7-i),min(4,9-j)
                    for k in range(0,5,4):
                        kk=min(4,5-k)
                        reads+=2*(mi*kk+kk*nj)
                    writes+=2*mi*nj
        self.assertEqual((a.dram_read_bytes,a.dram_write_bytes),(reads,writes))

    def test_sram_fixed_mapping_rejected_and_auto_remaps(self):
        with self.assertRaises(ValueError):os_mapping(Matmul(8,8,8),(4,4,4),127)
        small=choose_os_mapping(Matmul(8,8,8),32)
        large=choose_os_mapping(Matmul(8,8,8),4096)
        self.assertLessEqual(small.workspace_bytes,32)
        self.assertGreater(small.dram_read_bytes,large.dram_read_bytes)
        with self.assertRaises(ValueError):choose_os_mapping(Matmul(8,8,8),7)

    def test_dtype_widths_and_invalid_shapes(self):
        self.assertEqual(TensorSpec('x',(2,3),4).nbytes,24)
        for bad in (0,-1,1.5,True):
            with self.assertRaises(ValueError):Matmul(bad,8,8)
        with self.assertRaises(ValueError):TensorSpec('x',(0,),2)

    def test_bandwidth_affects_inclusive_local_time(self):
        hw=load_case(ROOT/'configs/toy_grpo_2die/case.json').hardware
        s=KernelSpec('gemm',(Matmul(8,8,8),))
        p=AnalyticalCostProvider()
        a=p.estimate(s,hw)
        b=p.estimate(s,dataclasses.replace(hw,dram_bytes_per_ns=1))
        self.assertGreater(b.duration_ns,a.duration_ns)
        self.assertEqual(a.duration_ns,a.compute_ns+a.local_memory_ns)
        self.assertIn('not_hardware_calibrated',a.source)

    def test_profile_miss_provenance_and_no_silent_fallback(self):
        hw=load_case(ROOT/'configs/toy_grpo_2die/case.json').hardware
        s=KernelSpec('gemm',(Matmul(8,8,8),))
        with self.assertRaises(ValueError):LookupCostProvider({}).estimate(s,hw)
        key=profile_key(s,hw)
        with self.assertRaises(ValueError):LookupCostProvider({key:{}}).estimate(s,hw)
        estimate=AnalyticalCostProvider().estimate(s,hw)
        records={key:{'estimate':dataclasses.asdict(estimate),'label_origin':'analytic_fixture', 'reference_version':'test-v1'}}
        self.assertEqual(LookupCostProvider(records).estimate(s,hw),estimate)
        with self.assertRaises(ValueError):KernelSpec('bad',timing_contract='explicit_dma_plus_inclusive')


if __name__=='__main__':unittest.main()
