import json
from dataclasses import replace
from pathlib import Path
import unittest
from rl_xy.config import load_case
from rl_xy.colocation import build_colocated_graph,load_colocation
from rl_xy.capacity_mapping import CapacityAwareCostProvider
from rl_xy.cost_model import AnalyticalCostProvider
from rl_xy.operator_ir import KernelSpec,Matmul

ROOT=Path(__file__).resolve().parents[1]


class ColocationTests(unittest.TestCase):
    def setUp(self):
        self.case=load_case(ROOT/'configs/wafer4_fixture/case.json')
        self.path=ROOT/'configs/colocation/tp2_to_tp1.json'
        self.config=load_colocation(self.path,self.case,4)
    def graph(self,case=None,world=4,config=None):
        return build_colocated_graph(case or self.case,world,config or self.config,'/tmp/test-colocation')
    def test_stages_join_all_replicas_before_training(self):
        g=self.graph();by_id={o.node_id:o for o in g.plan.operations}
        rollout={o.node_id for o in g.plan.operations if o.metadata.get('colocation_stage')=='generation'}
        first={}
        for o in g.plan.operations:
            if o.phase.startswith('train_'):
                for r in o.participant_ranks:first.setdefault(r,o.node_id)
        for node in first.values():
            seen=set();todo=list(by_id[node].dependencies)
            while todo:
                n=todo.pop()
                if n in seen:continue
                seen.add(n);todo.extend(by_id[n].dependencies)
            self.assertTrue(rollout<=seen)
    def test_distinct_rollout_storage_and_round_cleanup(self):
        one=self.graph();two=self.graph(replace(self.case,rounds=2))
        self.assertEqual(one.plan.preflight_peaks,two.plan.preflight_peaks)
        self.assertTrue(all(c.passed for c in two.component_checks))
        exports=[o for o in one.plan.operations if o.phase=='weight_reshard' and o.kind=='collective']
        self.assertTrue(exports)
        self.assertEqual(sum(o.payload_bytes for o in exports),one.audit['weight_export_logical_payload_bytes'])
        for o in exports:self.assertEqual(o.collective,'ALL_GATHER')
    def test_general_tp4_contiguous_and_standby(self):
        m=replace(self.case.model,hidden_size=16,query_heads=4,kv_heads=4,head_dim=4,ffn_size=32,vocab_size=64)
        c=replace(self.case,model=m)
        g=self.graph(c,world=10,config={**self.config,'train_tp':4,'rollout_tp':2})
        self.assertEqual(g.placements[0]['rollout_tp_groups'],[[0,1],[2,3]])
        self.assertEqual(g.audit['idle_devices'],[8,9])
        self.assertTrue(all(x.passed for x in g.component_checks))
    def test_illegal_gqa_tp_rejected(self):
        self.assertEqual(self.config['rollout_tp'],1)
        self.assertFalse(__import__('rl_xy.colocation',fromlist=['legal_tp']).legal_tp(self.case.model,3))
    def test_generation_batches_requests_without_mixing_attention_keys(self):
        g=self.graph(config={**self.config,'rollout_tp':2})
        grouped=[o for o in g.plan.operations if o.metadata.get('colocation_stage')=='generation' and o.metadata.get('batch_request_count',0)>1]
        self.assertTrue(grouped)
        linear=[o for o in grouped if o.name.endswith('.qkv')]
        self.assertTrue(linear)
        self.assertTrue(all(len(o.kernel_spec.matmuls)==1 for o in linear))
        attention=[o for o in grouped if o.name.endswith('.attention_scores')]
        self.assertTrue(attention)
        for o in attention:
            self.assertEqual(o.flops,sum(p.flops for p in o.kernel_spec.matmuls))
            self.assertTrue(all(p.n>=p.m for p in o.kernel_spec.matmuls))
        self.assertTrue(all(c.passed for c in g.component_checks))
    def test_capacity_mapping_cost_not_above_reference(self):
        h=replace(self.case.hardware,cores_per_die=16,sram_bytes_per_core=1250000)
        for p in [Matmul(65,99,127),Matmul(7,1024,128),Matmul(128,2048,512)]:
            spec=KernelSpec('mapping_test',matmuls=(p,));a=CapacityAwareCostProvider().estimate(spec,h);b=AnalyticalCostProvider().estimate(spec,h)
            self.assertLessEqual(a.duration_ns,b.duration_ns)
            self.assertEqual(a.partitions[0]['flops_sum'],p.flops)
            self.assertTrue(all(x<=h.sram_bytes_per_core for x in a.core_workspace_bytes))


if __name__=='__main__':unittest.main()
