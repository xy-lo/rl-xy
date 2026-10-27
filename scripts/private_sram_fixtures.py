#!/usr/bin/env python3
"""Native two-core SRAM placement, malformed contracts and lossless encoding."""
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from integration_fixtures import execute,ticks
from rl_xy.resource_encoding import compact_rank
from rl_xy.reporting import write_json,deterministic_tree_hash
from rl_xy.types import Operation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    ops=[Operation(1,'a',0,'fixture','a','compute',duration_ns=64,participant_ranks=(0,1)),
         Operation(2,'b',0,'fixture','b','compute',duration_ns=64,participant_ranks=(0,1),metadata={'execution_domain':'host'})]
    n={'reads':['input'],'allocations':{},'ready_outputs':[],'frees':[],'workspace_bytes':16,
       'sram_capacity':16,'local_memory':False,'reserve_before':0,'reserve_after':0,'core_workspace_bytes':[16,0]}
    base={'schema_version':'1.0','ranks':{str(r):{'initial_objects':{'input':4},'dram_capacity':32,
        'cores_per_die':2,'sram_bytes_per_core':16,'nodes':{'1':copy.deepcopy(n),'2':copy.deepcopy(n)}} for r in (0,1)}}
    observed={};checks=[]
    for name,second in [('same_core',[16,0]),('different_core',[0,16])]:
        c=copy.deepcopy(base)
        for r in c['ranks'].values():r['nodes']['2']['core_workspace_bytes']=second
        result=execute(a.out/name,ops,2,contract=c);t=ticks(result,ops,2)
        observed[name]=t
    checks.append({'check':'private_core_contention_not_shared_pool','passed':all(
        observed['same_core'][r,2,'issue']==64000 and observed['different_core'][r,2,'issue']==0 for r in (0,1))})
    for name,values,total,message in [('one_core_overflow',[17,0],17,'private SRAM capacity exceeded'),
        ('sum_mismatch',[16,0],15,'private SRAM sum mismatch'),('core_count',[16],16,'core count mismatch')]:
        c=copy.deepcopy(base);c['ranks']['0']['nodes']['1'].update(core_workspace_bytes=values,workspace_bytes=total)
        result=execute(a.out/name,ops,2,contract=c)
        checks.append({'check':name,'passed':result.return_code==2 and message in result.stderr_path.read_text(),
                       'exit_status':result.return_code})
    chain=[ops[0],replace(ops[1],dependencies=(1,))]
    raw=execute(a.out/'raw',chain,2,contract=base);t_raw=ticks(raw,chain,2)
    compact={**base,'ranks':{r:compact_rank(c) for r,c in base['ranks'].items()}}
    encoded=execute(a.out/'encoded',chain,2,contract=compact);t_encoded=ticks(encoded,chain,2)
    checks.append({'check':'lossless_delta_preserves_actual_native_ticks','passed':t_raw==t_encoded})
    report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':checks,
        'source_tree_sha256':deterministic_tree_hash(Path(__file__).resolve().parents[1])}
    write_json(a.out/'report.json',report)
    if report['status']!='PASS':raise RuntimeError('native private SRAM fixtures failed')
    print('private SRAM/encoding fixtures PASS')


if __name__=='__main__':main()
