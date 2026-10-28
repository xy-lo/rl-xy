"""Independent exact-weight census checks; no GPU or fabricated timing data."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('collect_aligned_shapes', ROOT / 'scripts/collect_aligned_shapes.py')
census = importlib.util.module_from_spec(_spec);_spec.loader.exec_module(census)
from rl_xy.operator_ir import Matmul, KernelSpec
from rl_xy.types import Operation
from rl_xy.config import load_case
from rl_xy.baseline_contract import BaselinePolicy
from rl_xy.aligned_graph import build_aligned_graph


class AlignedShapesTests(unittest.TestCase):
    def op(self, matmul, participants=(0,1), component=None):
        return Operation(1,'x',0,'train','GEMM','compute',duration_ns=1,
            participant_ranks=participants,kernel_spec=KernelSpec('test',matmuls=(matmul,)),
            metadata={} if component is None else {'stage_component': component})

    def test_component_invocations_TP_weights_and_global_kernel_are_exact(self):
        m=Matmul(3,5,7);g=Matmul(2,3,4,direction='dweight',output_bytes=4)
        template=SimpleNamespace(operations=[self.op(m)])
        contraction=SimpleNamespace(original_plan=template)
        graph=SimpleNamespace(component_expansions=[{'contraction':contraction,'physical':(0,1)},
            {'contraction':contraction,'physical':(2,3)}],
            plan=SimpleNamespace(operations=[self.op(m,component='ignored macro'),self.op(g,(1,))]))
        r=census.census_graph(graph,active=4)
        row=r['signatures'][census.signature(census.asdict(m))]
        self.assertEqual(row['occurrences'],4)
        self.assertEqual(row['logical_matmul_occurrences'],2)
        self.assertEqual(row['weighted_flops'],m.flops*4)
        self.assertEqual(r['totals']['weighted_flops'],m.flops*4+g.flops)
        self.assertEqual(r['totals']['rank_local_matmul_occurrences'],5)
        self.assertTrue(all(type(x['weighted_flops']) is int for x in r['signatures'].values()))

    def test_missing_exact_signature_has_no_fallback(self):
        m=Matmul(3,5,7);graph=SimpleNamespace(plan=SimpleNamespace(operations=[self.op(m)]))
        r=census.census_graph(graph,active=2);d=census.measured_domain(r,{})
        self.assertFalse(d['exact_domain_complete']);self.assertEqual(d['missing_count'],1)
        self.assertEqual(d['missing_rank_local_occurrences'],2)
        self.assertFalse(d['fallback_permitted'])
        k=next(iter(r['shapes']))
        labels={k:{'problem':r['shapes'][k],'duration_ns':100,'repeat_relative_difference':0}}
        self.assertTrue(census.measured_domain(r,labels)['exact_domain_complete'])
        labels[k]['problem']={**labels[k]['problem'],'direction':'dinput'}
        with self.assertRaisesRegex(ValueError,'signature integrity'):
            census.measured_domain(r,labels)

    def test_idle_or_missing_participant_is_rejected(self):
        m=Matmul(3,5,7)
        for ps in [(),(0,2),(0,0)]:
            with self.assertRaises(ValueError):
                census.census_graph(SimpleNamespace(plan=SimpleNamespace(operations=[self.op(m,ps)])),active=2)

    def test_placeholder_provider_has_core_vectors_and_no_real_cost(self):
        case=load_case(ROOT/'configs/wafer4_fixture/case.json')
        h=replace(case.hardware,cores_per_die=108)
        r=census.CensusProvider().estimate(KernelSpec('test'),h)
        self.assertEqual(r.duration_ns,1)
        self.assertEqual(r.core_workspace_bytes,(0,)*108)
        self.assertEqual(r.sram_workspace_bytes,0)
        self.assertIn('NOT_A_PERFORMANCE',r.source)

    def test_source_snapshot_includes_nested_modules(self):
        files=census.source_snapshot()['files']
        nested=list((ROOT/'src/rl_xy').rglob('*.py'))
        self.assertTrue(any(p.parent!=ROOT/'src/rl_xy' for p in nested))
        self.assertTrue(all(str(p.relative_to(ROOT)) in files for p in nested))

    def test_compact_census_matches_independent_uncontracted_rank_sum(self):
        c=load_case(ROOT/'configs/wafer4_fixture/case.json');a=c.workload.trajectories[0]
        c=replace(c,hardware=replace(c.hardware,dram_bytes_per_die=10**8,cores_per_die=2),
            workload=replace(c.workload,trajectories=tuple(replace(a,trajectory_id=f'X{i}',group_id=f'G{i//2}',reward=float(i%2)) for i in range(8))))
        p=BaselinePolicy(prompt_batch_size=4,ppo_mini_batch_prompts=4,train_tp=2,rollout_tp=1,
            train_micro_batch_trajectories=2,logprob_micro_batch_trajectories=2,use_kl_loss=True,kl_coefficient=.001)
        compact=build_aligned_graph(c,6,p,census.CensusProvider(),'/census',active=4)
        raw=build_aligned_graph(c,6,p,census.CensusProvider(),'/census',active=4,compact_components=False)
        measured=census.census_graph(compact,active=4)
        expected={}
        for op in raw.plan.operations:
            if op.kernel_spec:
                for m in op.kernel_spec.matmuls:
                    key=census.signature(census.asdict(m))
                    expected[key]=expected.get(key,0)+len(op.participant_ranks)
        self.assertEqual({k:v['occurrences'] for k,v in measured['signatures'].items()},expected)
        self.assertEqual(measured['totals']['weighted_flops'],sum(sum(m.flops for m in o.kernel_spec.matmuls)*len(o.participant_ranks) for o in raw.plan.operations if o.kernel_spec))
        self.assertTrue(compact.component_expansions)


if __name__=='__main__':unittest.main()
