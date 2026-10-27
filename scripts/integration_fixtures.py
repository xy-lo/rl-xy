#!/usr/bin/env python3
"""Real AstraSim resource-admission and four-rank network counterfactuals.

Run inside the mapped Docker image. This is NOT an alternative RL simulator:
it generates tiny Chakra ETs and checks the pinned native binary's callbacks.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.astra_adapter.generator import generate_traces
from rl_xy.astra_adapter.runner import run_astrasim, AstraExecutionError, sha256_file
from rl_xy.astra_adapter.parser import EVENT_RE, FINISHED_RE
from rl_xy.reporting import deterministic_tree_hash, write_json
from rl_xy.types import Operation

ROOT=Path(__file__).resolve().parents[1]
BIN=Path('/opt/astra-sim/build/astra_analytical/build/bin')


def execute(folder, operations, ranks, *, aware=False, contract=None):
    folder.mkdir(parents=True, exist_ok=False)
    system=json.loads((ROOT/'configs/toy_grpo_2die/astra_system.json').read_text())
    if ranks==4:
        for k in ('all-reduce-implementation','all-gather-implementation',
                  'reduce-scatter-implementation','all-to-all-implementation'):
            system[k]=['ring','ring']
    write_json(folder/'system.json',system)
    # Four nodes, two radix-2 Ring dimensions: square adjacency. This is a
    # network fixture, not a generalized arbitrary-mesh RL configuration.
    (folder/'network.yml').write_text(
        'topology: [ Ring, Ring ]\nnpus_count: [ 2, 2 ]\n'
        'bandwidth: [ 0.007450580596923828, 0.007450580596923828 ]\n'
        'latency: [ 1000.0, 1000.0 ]\n' if ranks==4 else
        (ROOT/'configs/toy_grpo_2die/astra_network.yml').read_text())
    write_json(folder/'remote.json',json.loads((ROOT/'configs/toy_grpo_2die/astra_remote_memory.json').read_text()))
    write_json(folder/'operations.json',[asdict(o) for o in operations])
    generate_traces(operations,folder/'workload',ranks=ranks,astra_time_scale=1000)
    if contract is not None:
        contract=copy.deepcopy(contract)
        contract['log_prefix']=str(folder/'astra_logs/resources')
        write_json(folder/'resource_contract.json',contract)
    failure=None
    try:
        result=run_astrasim(trace_prefix=folder/'workload',system_config=folder/'system.json',
            network_config=folder/'network.yml',remote_memory_config=folder/'remote.json',
            output_dir=folder,binary=BIN/('AstraSim_Analytical_Congestion_Aware' if aware
                else 'AstraSim_Analytical_Congestion_Unaware'),
            resource_contract=folder/'resource_contract.json' if contract else None,timeout_seconds=60,
            network_model='mesh2d' if ranks==4 and aware else None)
    except AstraExecutionError as exc:
        if exc.result is None:raise
        result=exc.result;failure=str(exc)
    write_json(folder/'astra_command.json',{'argv':list(result.command),'exit_status':result.return_code})
    write_json(folder/'execution.json',{'binary':str(result.binary_path),'binary_sha256':result.binary_sha256,
        'astra_commit':result.astra_commit,'exit_status':result.return_code,'failure':failure,
        'network_model':'mesh2d' if ranks==4 and aware else 'native_yaml',
        'network_note':'2x2 Ring dimensions have identical adjacency to a 2x2 mesh; no wraparound extra neighbor'})
    return result


def ticks(result,operations,ranks):
    if result.return_code or result.stderr_path.read_text().strip():
        raise ValueError('successful fixture must have rc=0 and empty stderr')
    text=result.debug_log_path.read_text()
    records={}
    for m in EVENT_RE.finditer(text):
        key=(int(m['rank']),int(m['node']),m['action'])
        if key in records:raise ValueError('duplicate native event')
        records[key]=int(m['tick'])
    wanted={(r,o.node_id,a) for o in operations for r in o.participant_ranks for a in ('issue','callback')}
    if set(records)!=wanted:raise ValueError('incomplete native event coverage')
    finished=list(FINISHED_RE.finditer(result.stdout_path.read_text()))
    if len(finished)!=ranks or {int(m['rank']) for m in finished}!=set(range(ranks)):
        raise ValueError('incomplete native rank termination')
    for o in operations:
        for r in o.participant_ranks:
            start=records[r,o.node_id,'issue'];end=records[r,o.node_id,'callback']
            if end<start:raise ValueError('callback before issue')
            for dep in o.dependencies_by_rank.get(r,o.dependencies):
                if records[r,dep,'callback']>start:raise ValueError('dependency violated')
    write_json(result.stdout_path.parent/'paired_ticks.json',[
        {'rank':r,'node':o.node_id,'issue_tick':records[r,o.node_id,'issue'],
         'callback_tick':records[r,o.node_id,'callback']} for o in operations for r in o.participant_ranks])
    return records


def resource_fixture(out):
    ops=[Operation(1,'gpu_local',0,'fixture','gpu_local','compute',duration_ns=64,participant_ranks=(0,1)),
         Operation(2,'cpu_local',0,'fixture','cpu_local','compute',duration_ns=64,
                   metadata={'execution_domain':'host'},participant_ranks=(0,1)),
         Operation(3,'retire',0,'fixture','retire','compute',duration_ns=1,dependencies=(1,2),
                   metadata={'execution_domain':'host'},participant_ranks=(0,1))]
    node={'reads':['input'],'allocations':{},'ready_outputs':[],'frees':[],
          'workspace_bytes':0,'sram_capacity':64,'local_memory':False,'reserve_before':0,'reserve_after':0}
    ranks={}
    for r in (0,1):
        nodes={str(i):copy.deepcopy(node) for i in (1,2,3)}
        for i in (1,2):
            nodes[str(i)].update(allocations={f'out{i}':8},ready_outputs=[f'out{i}'],
                                 workspace_bytes=16,local_memory=True)
        nodes['3'].update(reads=['out1','out2'],frees=['out1','out2'])
        ranks[str(r)]={'initial_objects':{'input':8},'dram_capacity':128,'nodes':nodes}
    base={'schema_version':'1.0','ranks':ranks}
    checks=[];observed={}
    for name,locked in [('independent_engines',False),('shared_local_memory',True)]:
        contract=copy.deepcopy(base)
        if not locked:
            for r in contract['ranks'].values():
                for n in r['nodes'].values():n['local_memory']=False
        result=execute(out/name,ops,2,contract=contract)
        t=ticks(result,ops,2)
        observed[name]=t
    for rank in (0,1):
        a=observed['independent_engines'];b=observed['shared_local_memory']
        checks.append({'check':f'local_memory_serializes_distinct_engines_rank{rank}',
            'passed':a[rank,1,'issue']==a[rank,2,'issue']==0
                and b[rank,2,'issue']==b[rank,1,'callback']==64000
                and b[rank,3,'callback']==129000 and a[rank,3,'callback']==65000,
            'unlocked_end_tick':a[rank,3,'callback'],'locked_end_tick':b[rank,3,'callback']})
    ready_contract=copy.deepcopy(base)
    for r in ready_contract['ranks'].values():
        for n in r['nodes'].values():n['local_memory']=False
        r['nodes']['2']['reads']=['out1']
    result=execute(out/'ready_gate_without_dependency',ops,2,contract=ready_contract)
    t=ticks(result,ops,2)
    checks.append({'check':'allocated_not_ready_blocks_independent_lane_without_DAG_edge',
        'passed':all(t[r,2,'issue']==t[r,1,'callback']==64000 for r in (0,1)),
        'producer_callback_tick':t[0,1,'callback'],'consumer_issue_tick':t[0,2,'issue']})
    # Bypass the Python preflight deliberately: these must be rejected by C++
    # BEFORE successful completion, not only by a post-hoc Python ledger.
    for name,message in [('dram_oom','DRAM capacity exceeded'),('sram_oom','SRAM capacity exceeded'),
                         ('read_absent','read absent/freed tensor'),('double_free','double free'),
                         ('reference_write','immutable reference write'),
                         ('free_active_reader','free with active access'),('read_after_free','read absent/freed tensor')]:
        bad=copy.deepcopy(base);r=bad['ranks']['0'];n=r['nodes']['1']
        case_ops=ops
        if name=='dram_oom':r['dram_capacity']=12
        elif name=='sram_oom':n['workspace_bytes']=65
        elif name=='read_absent':n['reads']=['nonexistent']
        elif name=='double_free':r['nodes']['3']['frees'].append('out1')
        elif name=='reference_write':
            n['allocations']={'d0.reference.weights':8};n['ready_outputs']=['d0.reference.weights']
        elif name=='free_active_reader':
            from dataclasses import replace
            case_ops=[replace(o,duration_ns=1) if o.node_id==2 else o for o in ops]
            for nr in r['nodes'].values():nr['local_memory']=False
            r['nodes']['2']['frees']=['input']
        elif name=='read_after_free':
            from dataclasses import replace
            case_ops=[replace(o,dependencies=(1,)) if o.node_id==2 else o for o in ops]
            n['frees']=['input']
        result=execute(out/name,case_ops,2,contract=bad)
        checks.append({'check':f'cpp_rejects_{name}','passed':result.return_code!=0
            and message in result.stderr_path.read_text(),'exit_status':result.return_code,
            'expected_error':message})
    return checks


def network_fixture(out):
    observed={};checks=[]
    for name,interferer in [('solo',None),('disjoint',(2,0)),('shared',(1,3))]:
        ops=[Operation(1,'start',0,'network','start','compute',duration_ns=1,participant_ranks=(0,1,2,3)),
             Operation(2,'probe_0_to_3',0,'network','probe','p2p',payload_bytes=8192,
                       src_rank=0,dst_rank=3,dependencies=(1,),participant_ranks=(0,3))]
        if interferer:
            src,dst=interferer
            # Native aware backend is store-and-forward. Delay the one-hop
            # interferer by 512ns so its 1024ns serialization overlaps the
            # probe's second hop. Without this, same-time flows do NOT contend.
            ops.append(Operation(6,'offset_interferer',0,'network','offset','compute',duration_ns=512,
                                 dependencies=(1,),participant_ranks=(src,)))
            ops.append(Operation(3,'interferer',0,'network','interferer','p2p',payload_bytes=8192,
                                 src_rank=src,dst_rank=dst,participant_ranks=(src,dst),
                                 dependencies_by_rank={src:(6,),dst:(1,)}))
        # Independent completion markers; do not create a false global barrier.
        deps={r:tuple(o.node_id for o in ops if r in o.participant_ranks) for r in range(4)}
        ops.append(Operation(4,'finish',0,'network','finish','compute',duration_ns=1,
                   participant_ranks=(0,1,2,3),dependencies_by_rank=deps))
        for aware in (False,True):
            key=name+('_aware' if aware else '_unaware')
            result=execute(out/key,ops,4,aware=aware)
            t=ticks(result,ops,4)
            observed[key]={'probe_receive_service_ticks':t[3,2,'callback']-t[0,2,'issue'],
                           'send_issue_tick':t[0,2,'issue'],'receive_callback_tick':t[3,2,'callback']}
    solo=observed['solo_aware']['probe_receive_service_ticks']
    shared=observed['shared_aware']['probe_receive_service_ticks']
    disjoint=observed['disjoint_aware']['probe_receive_service_ticks']
    checks.append({'check':'aware_shared_route_slows_probe','passed':shared>solo,'solo':solo,'shared':shared})
    checks.append({'check':'aware_disjoint_route_preserves_probe','passed':disjoint==solo,'solo':solo,'disjoint':disjoint})
    unaware=[observed[n+'_unaware']['probe_receive_service_ticks'] for n in ('solo','shared','disjoint')]
    checks.append({'check':'unaware_cannot_model_route_contention','passed':len(set(unaware))==1,'ticks':unaware})
    write_json(out/'network_comparison.json',observed)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--section',choices=['all','network','resources'],default='all')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    checks=[]
    if args.section in ('all','resources'):checks.extend(resource_fixture(out/'resources'))
    if args.section in ('all','network'):checks.extend(network_fixture(out/'network'))
    report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':checks,
        'source_tree_sha256':deterministic_tree_hash(ROOT),'image':os.environ.get('RLXY_IMAGE'),
        'image_id':os.environ.get('RLXY_IMAGE_ID'),'scope':'synthetic native-Astra counterfactuals; not hardware calibration',
        'artifact_sha256':{p.relative_to(out).as_posix():sha256_file(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out/'report.json',report);print(json.dumps(report['checks'],indent=2))
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
