from dataclasses import asdict,replace
from pathlib import Path
import unittest
from rl_xy.memoized_costs import MemoizedCostProvider
from rl_xy.cost_model import AnalyticalCostProvider
from rl_xy.operator_ir import KernelSpec,Matmul
from rl_xy.config import load_case
class MemoizedCostTests(unittest.TestCase):
 def test_complete_key_and_identical_readonly_metadata(self):
  c=load_case(Path(__file__).resolve().parents[1]/'configs/wafer4_fixture/case.json');p=AnalyticalCostProvider();q=MemoizedCostProvider(p)
  a=KernelSpec('x',matmuls=(Matmul(8,8,8),));b=replace(a,read_bytes=16)
  self.assertEqual(q.estimate(a,c.hardware),p.estimate(a,c.hardware));self.assertEqual(q.metadata_for(a,c.hardware),asdict(p.estimate(a,c.hardware)))
  self.assertIs(q.metadata_for(a,c.hardware),q.metadata_for(a,c.hardware))
  self.assertEqual(q.estimate(b,c.hardware),p.estimate(b,c.hardware));self.assertEqual(q.misses,2)
  h=replace(c.hardware,dram_bytes_per_ns=1);self.assertEqual(q.estimate(a,h),p.estimate(a,h));self.assertEqual(q.misses,3)
if __name__=='__main__':unittest.main()
