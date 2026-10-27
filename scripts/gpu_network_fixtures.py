#!/usr/bin/env python3
"""Native shared node-uplink contention, not all-to-all independent links."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.astra_adapter import generate_traces,run_astrasim,parse_astra_output
from rl_xy.types import Operation
from rl_xy.reporting import deterministic_tree_hash


def fixture(out,pairs):
    out.mkdir(parents=True,exist_ok=False);ops=[];last={}
    def add(op):
        from dataclasses import replace
        deps={r:(last[r],) if r in last else () for r in op.participant_ranks}
        op=replace(op,dependencies_by_rank=deps,dependencies=tuple(sorted({i for v in deps.values() for i in v})))
        ops.append(op)
        for r in op.participant_ranks:last[r]=op.node_id
    add(Operation(1,'initial',0,'network','initial','compute',duration_ns=1,participant_ranks=tuple(range(8))))
    for i,(a,b) in enumerate(pairs,2):
        add(Operation(i,f'pair{i}',0,'network',f'pair{i}','p2p',payload_bytes=64,
            src_rank=a,dst_rank=b,participant_ranks=tuple(sorted((a,b)))))
    add(Operation(len(ops)+1,'finish',0,'network','finish','compute',duration_ns=1,participant_ranks=tuple(range(8))))
    prefix=out/'workload';generate_traces(ops,prefix,ranks=8,astra_time_scale=1000)
    template=Path('configs/wafer4_fixture/astra_system.json');system=json.loads(template.read_text())
    for k in ('all-reduce-implementation','all-gather-implementation','reduce-scatter-implementation','all-to-all-implementation'):
        system[k]=['ring','ring']
    (out/'system.json').write_text(json.dumps(system))
    rates=[8/1000*1e9/2**30,4/1000*1e9/2**30]
    (out/'network.yml').write_text(f'topology: [Ring, Ring]\nnpus_count: [2, 4]\nbandwidth: {rates}\nlatency: [1000, 1000]\n')
    (out/'remote.json').write_text('{"memory-type":"NO_MEMORY_EXPANSION"}')
    result=run_astrasim(trace_prefix=prefix,system_config=out/'system.json',network_config=out/'network.yml',
        remote_memory_config=out/'remote.json',output_dir=out,network_model='gpu_cluster',
        binary='/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware')
    events,_=parse_astra_output(result,ops,ranks=8,astra_time_scale=1000)
    (out/'events.json').write_text(json.dumps([asdict(e) for e in events],indent=2))
    return {'max_tick':max(e.callback_tick for e in events),'binary_sha256':result.binary_sha256,'native_exit':result.return_code}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    # Every rank communicates, preserving the strict statistics coverage gate.
    # Reverse direction uses a different full-duplex link, not the same queue.
    shared=fixture(out/'shared_uplink',[(0,2),(1,3),(4,6),(5,7)])
    disjoint=fixture(out/'independent_uplinks',[(0,2),(3,1),(4,6),(7,5)])
    intra=fixture(out/'intra_node',[(0,1),(2,3),(4,5),(6,7)])
    checks=[{'check':'shared_node_uplink_contention','passed':shared['max_tick']>disjoint['max_tick']},
        {'check':'intra_node_bypasses_spine','passed':intra['max_tick']<disjoint['max_tick']},
        {'check':'same_native_binary_and_success','passed':len({r['binary_sha256'] for r in (shared,disjoint,intra)})==1
            and all(r['native_exit']==0 for r in (shared,disjoint,intra))}]
    r={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','checks':checks,
       'shared':shared,'disjoint':disjoint,'intra':intra,'source_tree_sha256':deterministic_tree_hash(Path.cwd())}
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(r['status'])
    return 0 if r['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
