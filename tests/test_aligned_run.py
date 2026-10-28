"""Runner gates and clock/proof checks; no mocked result is a native run."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from rl_xy.aligned_run import load_policy, run_aligned, validate_aligned_clocks, expand_component_proofs
from rl_xy.cli import build_parser
from rl_xy.config import load_case
from rl_xy.local_chain_contraction import contract_local_chains
from rl_xy.tensor_plan import TensorPlan
from rl_xy.types import AstraEvent, Operation

ROOT=Path(__file__).resolve().parents[1]


def event(rank,node,start,end,op_id='observed',phase='observed'):
    return AstraEvent(rank,node,op_id,0,phase,4,start,end,end-start,start/1000,end/1000,(end-start)/1000)


def stage_graph():
    names=['switch_to_rollout','weight_reshard','generation','switch_to_train','reward','old_policy',
           'reference','advantage','train','dp_sync','clip','optimizer','optimizer_sync','publish']
    ops=[];events=[]
    for number,stage in enumerate(names,1):
        name='Adam_owned_shard_0' if stage=='optimizer' else 'publish_actor_version' if stage=='publish' else stage
        meta={'colocation_stage':stage}
        if stage in ('optimizer','publish'):meta.update(actor_version_in=0,actor_version_out=1)
        op=Operation(number,name,0,stage,name,'compute',duration_ns=1,participant_ranks=(0,1),metadata=meta)
        ops.append(op)
        events.extend(event(r,number,(number-1)*1000,number*1000,name,stage) for r in (0,1))
    graph=SimpleNamespace(plan=SimpleNamespace(operations=ops),audit={'active_devices':2})
    return graph,events


class AlignedRunTests(unittest.TestCase):
    def setUp(self):self.case=load_case(ROOT/'configs/aligned_fixture/case.json')

    def test_cli_exposes_selected_runner_and_explicit_contraction_scope(self):
        args=build_parser().parse_args(['run-aligned','--case','c','--scenario','s','--policy','p',
            '--active','4','--gpu-profile','g','--out','o','--uncontracted-components'])
        self.assertEqual((args.command,args.active,args.uncontracted_components),('run-aligned',4,True))
        self.assertFalse(args.contract_local_chains)
        policy=load_policy(ROOT/'configs/aligned_fixture/policy.json')
        self.assertEqual((policy.trajectory_batch_size,policy.train_micro_batch_trajectories),(4,2))

    def test_whole_graph_flag_cannot_silently_claim_implemented_support(self):
        with self.assertRaisesRegex(ValueError,'whole-graph.*unsupported'):
            run_aligned('/not-read','/not-read','/not-read','/not-created',contract_local_chains=True)

    def test_actual_stage_clocks_are_checked_instead_of_python_graph_order(self):
        graph,events=stage_graph();checks,spans=validate_aligned_clocks(self.case,graph,events)
        self.assertTrue(all(c.passed for c in checks));self.assertIn('r0.optimizer_sync',spans)
        generation=next(o for o in graph.plan.operations if o.phase=='generation')
        shifted=[replace(e,issue_tick=0,issue_ns=0) if e.node_id==generation.node_id else e for e in events]
        checks,_=validate_aligned_clocks(self.case,graph,shifted)
        self.assertFalse(checks[0].passed)

    def test_publication_versions_and_active_world_are_not_implicit(self):
        graph,events=stage_graph();publish=next(o for o in graph.plan.operations if o.name=='publish_actor_version')
        graph.plan.operations=[replace(o,metadata={**o.metadata,'actor_version_out':3}) if o is publish else o for o in graph.plan.operations]
        checks,_=validate_aligned_clocks(self.case,graph,events);self.assertFalse(checks[1].passed)
        graph,events=stage_graph();graph.audit['active_devices']=3
        checks,_=validate_aligned_clocks(self.case,graph,events);self.assertFalse(checks[1].passed)

    def test_component_native_rank_node_mapping_and_independent_logical_replay(self):
        ops=[Operation(1,'a',0,'train','a','compute',duration_ns=2,participant_ranks=(0,)),
             Operation(2,'b',0,'train','b','compute',duration_ns=3,participant_ranks=(0,),dependencies=(1,))]
        def node(**kw):return dict(reads=['w'],allocations={},ready_outputs=[],frees=[],workspace_bytes=0,
            sram_capacity=16,local_memory=True,reserve_before=0,reserve_after=0,**kw)
        contract={'schema_version':'1.0','ranks':{'0':{'initial_objects':{'w':4},'dram_capacity':64,
            'nodes':{'1':node(),'2':node()}}},'log_prefix':'/not-executed'}
        contraction=contract_local_chains(TensorPlan(contract,[],ops,{}))
        graph=SimpleNamespace(component_expansions=[dict(label='component',contraction=contraction,
            physical=(3,),global_to_local={100:1})])
        checks,proofs=expand_component_proofs(graph,[event(3,100,5000,10000)])
        self.assertTrue(checks[0].passed)
        self.assertEqual(proofs['component']['expanded_logical_events'],2)
        self.assertTrue(proofs['component']['clock_bound_to_expanded_native_events'])
        self.assertFalse(proofs['component']['native_per_object_hook_claimed'])
        with self.assertRaisesRegex(ValueError,'incomplete'):
            expand_component_proofs(graph,[])
        with self.assertRaisesRegex(ValueError,'duration differs'):
            expand_component_proofs(graph,[event(3,100,5000,11000)])

    def test_fixture_budgets_are_small_and_inputs_are_declared_synthetic(self):
        self.assertEqual(self.case.evidence_kind,'synthetic_fixture')
        self.assertEqual(len(self.case.workload.trajectories),4)
        self.assertEqual(self.case.model.layers,1)
        gpu=json.loads((ROOT/'configs/aligned_fixture/gpu.json').read_text())
        self.assertEqual(gpu['gpu_count'],4)


if __name__=='__main__':unittest.main()
