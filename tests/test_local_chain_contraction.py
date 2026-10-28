import copy
import hashlib
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from rl_xy.local_chain_contraction import (contract_local_chains, expand_native_events,
    replay_logical_resources)
from rl_xy.tensor_plan import TensorPlan
from rl_xy.types import Operation, AstraEvent


def event(rank, node, start, end, op_id='observed', phase='observed'):
    return AstraEvent(rank, node, op_id, 1, phase, 1, start, end, end-start,
        start/1000, end/1000, (end-start)/1000)


def plan_fixture():
    ops = [Operation(1, 'a', 1, 'forward', 'a', 'compute', duration_ns=2, participant_ranks=(0,),flops=128,vector_ops=8,tensor_bytes=32),
        Operation(2, 'b', 1, 'forward', 'b', 'compute', duration_ns=3, participant_ranks=(0,), dependencies=(1,),flops=256,vector_ops=16,tensor_bytes=64),
        Operation(3, 'other', 1, 'prepare', 'other', 'compute', duration_ns=1, participant_ranks=(1,)),
        Operation(4, 'send', 1, 'transport', 'send', 'p2p', payload_bytes=64,
            src_rank=0, dst_rank=1, participant_ranks=(0,1), dependencies_by_rank={0:(2,),1:(3,)}),
        Operation(5, 'c', 1, 'train', 'c', 'compute', duration_ns=4, participant_ranks=(0,), dependencies=(4,)),
        Operation(6, 'd', 1, 'train', 'd', 'compute', duration_ns=5, participant_ranks=(0,), dependencies=(5,)),
        Operation(7, 'finish1', 1, 'cleanup', 'finish1', 'compute', duration_ns=2, participant_ranks=(1,), dependencies=(4,))]
    ranks = {}
    for rank in (0,1):
        nodes = {}
        for op in ops:
            if rank not in op.participant_ranks:continue
            nodes[str(op.node_id)] = {'reads':['weight'], 'allocations':{}, 'ready_outputs':[], 'frees':[],
                'workspace_bytes':4, 'sram_capacity':16, 'local_memory':op.kind=='compute',
                'reserve_before':0, 'reserve_after':0}
        ranks[str(rank)] = {'initial_objects':{'weight':4},'dram_capacity':64,'nodes':nodes}
    ranks['0']['nodes']['1'].update(allocations={'a':16},ready_outputs=['a'])
    ranks['0']['nodes']['2'].update(reads=['weight','a'],frees=['a'])
    ranks['0']['nodes']['5'].update(allocations={'x':8},ready_outputs=['x'])
    ranks['0']['nodes']['6'].update(reads=['weight','x'],frees=['x'])
    ranks['1']['nodes']['3'].update(allocations={'other':8},ready_outputs=['other'])
    ranks['1']['nodes']['4']['reads'].append('other')
    ranks['1']['nodes']['7'].update(reads=['weight','other'],frees=['other'])
    return TensorPlan({'schema_version':'1.1','log_prefix':'/not-executed/resources','ranks':ranks}, [], ops, {})


class LocalChainContractionTests(unittest.TestCase):
    def test_accelerated_canonical_hash_matches_streaming_bytes(self):
        from rl_xy.local_chain_contraction import _digest
        value = {'unicode': '晶圆\n\"\\', 'floating': [-0.0, 1.25e-7],
                 'resource': plan_fixture().contract, 'private_cores': [i % 257 for i in range(10000)]}
        before = hashlib.sha256()
        for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':')).iterencode(value):
            before.update(chunk.encode())
        self.assertEqual(_digest(value), before.hexdigest())

    def test_exact_prefix_expansion_and_network_boundaries(self):
        original = plan_fixture();result = contract_local_chains(original)
        self.assertEqual([b.logical_node_ids for b in result.blocks],[(1,2),(3,),(4,),(5,6),(7,)])
        self.assertEqual(result.certificate['network_operations_deleted'],0)
        network = next(o for o in result.macro_plan.operations if o.kind=='p2p')
        self.assertEqual(network.node_id,4)
        self.assertEqual(network.payload_bytes,64)
        self.assertEqual(network.dependencies_by_rank,{0:(1,),1:(3,)})
        observed = [event(0,1,0,5000),event(1,3,0,1000),event(0,4,5000,20000),
            event(1,4,1000,20000),event(0,5,20000,29000),event(1,7,20000,22000)]
        expanded,proof = expand_native_events(result,observed)
        clocks = {(e.rank,e.node_id):(e.issue_tick,e.callback_tick) for e in expanded}
        self.assertEqual(clocks,{(0,1):(0,2000),(0,2):(2000,5000),(1,3):(0,1000),
            (0,4):(5000,20000),(1,4):(1000,20000),(0,5):(20000,24000),
            (0,6):(24000,29000),(1,7):(20000,22000)})
        self.assertTrue(proof['clock_bound_to_expanded_native_events'])
        self.assertEqual(proof['ranks']['0']['allocated_dram_peak_bytes'],20)
        self.assertEqual(proof['ranks']['1']['allocated_dram_peak_bytes'],12)

    def test_initial_identity_is_retained_and_temp_arena_not_duplicate_weights(self):
        result = contract_local_chains(plan_fixture(),retain_initial_objects=True)
        rank = result.macro_plan.contract['ranks']['0']
        self.assertEqual(rank['initial_objects'],{'weight':4,'d0.logical_plan_arena':16})
        self.assertEqual(result.macro_plan.preflight_peaks['d0.dram'],20)
        self.assertEqual(result.certificate['arena_reservations_by_rank']['0']['retained_initial_resident_bytes'],4)
        self.assertFalse(result.certificate['logical_resources']['native_per_object_hook_claimed'])

    def test_logical_outputs_live_across_macro_and_network_are_checked(self):
        plan = plan_fixture()
        # 'other' is created in a singleton before a network boundary, read by
        # the network, and freed after it. Its logical lifetime must survive.
        result = contract_local_chains(plan)
        self.assertEqual(result.certificate['logical_resources']['ranks']['1']['allocated_dram_peak_bytes'],12)
        broken = copy.deepcopy(plan);broken.contract['ranks']['1']['nodes']['3']['ready_outputs']=[]
        with self.assertRaises(ValueError):contract_local_chains(broken)

    def test_rank_parallelism_and_float_clock_are_rejected(self):
        plan = plan_fixture();plan.operations[1]=replace(plan.operations[1],dependencies=())
        with self.assertRaisesRegex(ValueError,'interleaving'):contract_local_chains(plan)
        plan = plan_fixture();plan.operations[0]=replace(plan.operations[0],duration_ns=2.5)
        with self.assertRaisesRegex(ValueError,'integer'):contract_local_chains(plan)

    def test_explicit_contraction_boundary_preserves_cleanup_node(self):
        plan=plan_fixture();plan.operations[1]=replace(plan.operations[1],
            metadata={'contraction_boundary':True})
        result=contract_local_chains(plan)
        self.assertEqual(result.blocks[0].logical_node_ids,(1,))
        self.assertEqual(result.blocks[1].logical_node_ids,(2,))
        self.assertEqual(result.macro_plan.operations[1].metadata['contraction_boundary'],True)

    def test_host_and_round_boundaries_never_merge(self):
        plan = plan_fixture();plan.operations[1]=replace(plan.operations[1],metadata={'execution_domain':'host'})
        result = contract_local_chains(plan)
        self.assertEqual(result.blocks[0].logical_node_ids,(1,))
        self.assertEqual(result.blocks[1].logical_node_ids,(2,))
        plan = plan_fixture();plan.operations[1]=replace(plan.operations[1],round_id=2)
        result = contract_local_chains(plan)
        self.assertEqual(result.blocks[0].logical_node_ids,(1,))

    def test_multi_rank_compute_preserves_each_ranks_own_start_tick(self):
        source=plan_fixture();node=copy.deepcopy(source.contract['ranks']['0']['nodes']['4'])
        node.update(reads=['weight'],workspace_bytes=1,local_memory=True)
        ops=[Operation(1,'r0',1,'local','r0','compute',duration_ns=2,participant_ranks=(0,)),
             Operation(2,'r1',1,'local','r1','compute',duration_ns=4,participant_ranks=(1,)),
             Operation(3,'bothA',1,'local','bothA','compute',duration_ns=3,participant_ranks=(0,1),dependencies_by_rank={0:(1,),1:(2,)}),
             Operation(4,'bothB',1,'local','bothB','compute',duration_ns=5,participant_ranks=(0,1),dependencies=(3,))]
        ranks={str(r):{'initial_objects':{'weight':4},'dram_capacity':64,
            'nodes':{str(o.node_id):copy.deepcopy(node) for o in ops if r in o.participant_ranks}} for r in (0,1)}
        plan=TensorPlan({'schema_version':'1.1','log_prefix':'unused','ranks':ranks},[],ops,{})
        result=contract_local_chains(plan)
        self.assertEqual(result.blocks[-1].logical_node_ids,(3,4))
        expanded,_=expand_native_events(result,[event(0,1,0,2000),event(1,2,0,4000),
            event(0,3,2000,10000),event(1,3,4000,12000)])
        times={(e.rank,e.node_id):(e.issue_tick,e.callback_tick) for e in expanded}
        self.assertEqual(times[0,3],(2000,5000))
        self.assertEqual(times[1,3],(4000,7000))
        self.assertEqual(times[0,4],(5000,10000))
        self.assertEqual(times[1,4],(7000,12000))

    def test_extra_external_entering_dependency_breaks_macro(self):
        plan=plan_fixture()
        # Host node1 precedes device nodes2/3 on rank0; node3 also has a
        # redundant external edge from node1. The conservative contract keeps
        # device2 and3 separate rather than deleting the entering edge.
        ops=[Operation(1,'host',1,'host','host','compute',duration_ns=1,participant_ranks=(0,),metadata={'execution_domain':'host'}),
             Operation(2,'a',1,'local','a','compute',duration_ns=2,participant_ranks=(0,),dependencies=(1,)),
             Operation(3,'b',1,'local','b','compute',duration_ns=3,participant_ranks=(0,),dependencies=(1,2))]
        node={'reads':['weight'],'allocations':{},'ready_outputs':[],'frees':[],
              'workspace_bytes':0,'sram_capacity':16,'local_memory':False,'reserve_before':0,'reserve_after':0}
        plan=TensorPlan({'schema_version':'1.1','log_prefix':'unused','ranks':{'0':{'initial_objects':{'weight':4},
            'dram_capacity':64,'nodes':{str(o.node_id):copy.deepcopy(node) for o in ops}}}},[],ops,{})
        result=contract_local_chains(plan)
        self.assertEqual([b.logical_node_ids for b in result.blocks],[(1,),(2,),(3,)])

    def test_macro_duration_tampering_missing_events_and_resource_mutation_rejected(self):
        result = contract_local_chains(plan_fixture())
        with self.assertRaisesRegex(ValueError,'duration'):
            expand_native_events(result,[event(0,1,0,5001)])
        with self.assertRaisesRegex(ValueError,'coverage'):
            expand_native_events(result,[event(0,1,0,5000)])
        result.original_plan.contract['ranks']['0']['nodes']['1']['allocations']['a']=15
        with self.assertRaisesRegex(ValueError,'changed'):
            expand_native_events(result,[])

    def test_reservations_are_in_arena_admission_peak(self):
        plan = plan_fixture()
        plan.contract['ranks']['0']['nodes']['1']['reserve_before']=8
        plan.contract['ranks']['0']['nodes']['2']['reserve_after']=-8
        result = contract_local_chains(plan)
        proof = result.certificate['logical_resources']['ranks']['0']
        self.assertEqual(proof['allocated_dram_peak_bytes'],20)
        self.assertEqual(proof['admission_dram_peak_including_reservations'],28)
        self.assertEqual(sum(result.macro_plan.contract['ranks']['0']['initial_objects'].values()),28)

    def test_private_workspace_maximum_does_not_hide_conservative_reservation(self):
        plan = plan_fixture()
        for r in plan.contract['ranks'].values():
            r.update(cores_per_die=2,sram_bytes_per_core=8)
            for n in r['nodes'].values():n['core_workspace_bytes']=[n['workspace_bytes'],0]
        plan.contract['ranks']['0']['nodes']['2']['core_workspace_bytes']=[0,4]
        result = contract_local_chains(plan)
        macro = result.macro_plan.contract['ranks']['0']['nodes']['1']
        self.assertEqual(macro['core_workspace_bytes'],[4,4])
        self.assertEqual(macro['workspace_bytes'],8)
        self.assertEqual(result.certificate['logical_resources']['ranks']['0']['logical_sram_peak_bytes'],4)
        self.assertTrue(result.certificate['native_SRAM_reservation_sum_can_exceed_logical_simultaneous_peak'])

    @unittest.skipUnless(os.environ.get('RLXY_TEST_NATIVE_CONTRACTION')=='1','explicit bounded Docker native test only')
    def test_native_original_and_contracted_logical_ticks_are_identical(self):
        from rl_xy.astra_adapter import generate_traces,run_astrasim,parse_astra_output
        root = Path(__file__).resolve().parents[1]
        persistent = os.environ.get('RLXY_CONTRACTION_NATIVE_OUT')
        if persistent:
            Path(persistent).mkdir(parents=True,exist_ok=False)
        context = nullcontext(persistent) if persistent else tempfile.TemporaryDirectory(prefix='rlxy-contraction-native-')
        with context as directory:
            output = Path(directory);plan = plan_fixture();result = contract_local_chains(plan)
            (output/'certificate.json').write_text(json.dumps(result.certificate,indent=2))
            (output/'logical_resource_contract.json').write_text(json.dumps(plan.contract,indent=2))
            (output/'logical_operations.json').write_text(json.dumps([asdict(o) for o in plan.operations],indent=2))
            (output/'macro_operations.json').write_text(json.dumps([asdict(o) for o in result.macro_plan.operations],indent=2))
            def execute(name,p):
                folder=output/name;folder.mkdir()
                generate_traces(p.operations,folder/'workload',ranks=2,astra_time_scale=1000)
                for label,source in [('system','astra_system.json'),('network','astra_network.yml'),('remote','astra_remote_memory.json')]:
                    (folder/(label+'.json' if label!='network' else 'network.yml')).write_bytes((root/'configs/toy_grpo_2die'/source).read_bytes())
                contract=copy.deepcopy(p.contract);contract['log_prefix']=str(folder/'astra_logs/resources')
                # Direct raw C++ resource format is1.0; logical/index format
                # version is independent and remains bound by the certificate.
                contract['schema_version']='1.0'
                (folder/'resources.json').write_text(json.dumps(contract))
                run=run_astrasim(trace_prefix=folder/'workload',system_config=folder/'system.json',
                    network_config=folder/'network.yml',remote_memory_config=folder/'remote.json',
                    resource_contract=folder/'resources.json',output_dir=folder,
                    binary='/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware',timeout_seconds=60)
                events,_=parse_astra_output(run,p.operations,ranks=2,astra_time_scale=1000)
                return events
            original=execute('original',plan);native=execute('contracted',result.macro_plan)
            expanded,proof=expand_native_events(result,native)
            (output/'original_events.json').write_text(json.dumps([asdict(e) for e in original],indent=2))
            (output/'macro_events.json').write_text(json.dumps([asdict(e) for e in native],indent=2))
            (output/'expanded_events.json').write_text(json.dumps([asdict(e) for e in expanded],indent=2))
            (output/'logical_resource_replay.json').write_text(json.dumps(proof,indent=2))
            key=lambda e:(e.rank,e.node_id,e.issue_tick,e.callback_tick)
            self.assertEqual(sorted(map(key,expanded)),sorted(map(key,original)))
            self.assertEqual(replay_logical_resources(plan,original)['ranks'],proof['ranks'])


if __name__=='__main__':unittest.main()
