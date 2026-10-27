#!/usr/bin/env python3
"""Host orchestrator: all simulations and tests execute inside the Docker image."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container',default='rl-xy-mapped')
    p.add_argument('--image',default='rlxy-astrasim:0.2.0')
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    if Path(a.run_id).name!=a.run_id or a.run_id in ('','.','..'):p.error('invalid run-id')
    out=ROOT/'acceptance'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'container':a.container,'image':a.image,'commands':[],'checks':[],'bundles':{}}
    def save():
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    def command(name,argv,*,expected_failure=False):
        with (out/(name+'.log')).open('w') as stream:
            r=subprocess.run(argv,cwd=ROOT,env={**os.environ,'RLXY_CONTAINER':a.container},
                             stdout=stream,stderr=subprocess.STDOUT,timeout=900)
        report['commands'].append({'label':name,'argv':argv,'exit_status':r.returncode,
            'expected_failure':expected_failure,'log':(out/(name+'.log')).relative_to(ROOT).as_posix(),
            'log_sha256':hashlib.sha256((out/(name+'.log')).read_bytes()).hexdigest()})
        save();print(f'{name}: rc={r.returncode}',flush=True)
        if (r.returncode!=0)!=expected_failure:raise ValueError(f'unexpected exit: {name}')
    def check(name,passed,evidence):
        report['checks'].append({'check':name,'passed':bool(passed),'evidence':evidence});save()
        if not passed:raise ValueError(name)
    def read(path):return json.loads((ROOT/path).read_text())
    try:
        save()
        image_id=subprocess.check_output(['docker','inspect','--format','{{.Image}}',a.container],text=True).strip()
        report['image_id']=image_id
        docker=['docker','exec','-e',f'RLXY_IMAGE={a.image}','-e',f'RLXY_IMAGE_ID={image_id}',a.container,'python3']
        command('unit_tests',docker+['-m','unittest','discover','-s','tests','-v'])
        for name,action,run_id in [('transfer','transfer',a.run_id),('toy','toy-resource',a.run_id),
            ('two_rounds','two-rounds-resource',a.run_id),('toy_repeat','toy-resource',a.run_id+'-repeat')]:
            command(name,['bash','scripts/docker.sh',action,a.image,run_id])
            case={'transfer':'fixture_transfer_21ns','toy':'toy_grpo_2die_resource',
                  'two_rounds':'toy_grpo_2die_resource_2rounds','toy_repeat':'toy_grpo_2die_resource'}[name]
            report['bundles'][name]=f'runs/{case}/{run_id}'
        variants={}
        for name,field,value in [('slow_dram','dram_bytes_per_ns',1),('small_sram','sram_bytes_per_core',64),
                                 ('dram_oom','dram_bytes_per_die',20000)]:
            folder=out/'variants'/name
            shutil.copytree(ROOT/'configs/toy_grpo_2die',folder)
            c=json.loads((folder/'case.json').read_text());c['hardware'][field]=value
            (folder/'case.json').write_text(json.dumps(c,indent=2)+'\n')
            result=f'runs/resource_sensitivity/{a.run_id}-{name}'
            variants[name]=result
            command(name,docker+['-m','rl_xy.cli','run','--resource-model',
                '--case',folder.relative_to(ROOT).as_posix()+'/case.json','--out',result],
                expected_failure=name=='dram_oom')
            if name!='dram_oom':report['bundles'][name]=result
        command('native_resource_and_network_fixtures',docker+['scripts/integration_fixtures.py',
            '--output',f'runs/resource_integration/{a.run_id}'])
        report['integration']=f'runs/resource_integration/{a.run_id}'
        fixture=read(report['integration']+'/report.json')
        check('native_fixture_checks',fixture['status']=='PASS' and all(c['passed'] for c in fixture['checks']),fixture['checks'])
        for name,bundle in report['bundles'].items():
            command('integrity_'+name,docker+['scripts/check_result.py',bundle])
        m={name:read(bundle+'/metrics.json') for name,bundle in report['bundles'].items()}
        check('hand_calculated_21ns',m['transfer']['total_simulated_time_ns']==21,m['transfer']['total_simulated_time_ns'])
        check('hardware_dram_bandwidth_changes_actual_time',m['slow_dram']['total_simulated_time_ns']>m['toy']['total_simulated_time_ns'],
              {n:m[n]['total_simulated_time_ns'] for n in ('toy','slow_dram')})
        check('sram_remaps_actual_workload',m['small_sram']['local_dram_read_bytes']>m['toy']['local_dram_read_bytes']
              and m['small_sram']['total_simulated_time_ns']>m['toy']['total_simulated_time_ns'],
              {n:{k:m[n][k] for k in ('total_simulated_time_ns','local_dram_read_bytes','memory_peak_bytes_by_location')} for n in ('toy','small_sram')})
        check('oom_fail_closed_before_astra',read(variants['dram_oom']+'/status.json')['status']=='failed'
              and not (ROOT/variants['dram_oom']/'astra_command.json').exists(),variants['dram_oom'])
        check('actor_version_and_frozen_reference',m['two_rounds']['actor_version_end']==2
              and m['two_rounds']['reference_version_end']==m['toy']['reference_version_start']==0,
              {n:[m[n]['actor_version_end'],m[n]['reference_version_end']] for n in ('toy','two_rounds')})
        check('two_round_no_growth_in_runtime_peak',m['toy']['memory_peak_bytes_by_location']==m['two_rounds']['memory_peak_bytes_by_location'],
              m['two_rounds']['memory_peak_bytes_by_location'])
        first=(ROOT/report['bundles']['toy']/'events.jsonl').read_bytes()
        repeat=(ROOT/report['bundles']['toy_repeat']/'events.jsonl').read_bytes()
        check('repeat_event_ticks_identical',first==repeat,hashlib.sha256(first).hexdigest())
        sys.path.insert(0,str(ROOT/'src'))
        from rl_xy.reporting import deterministic_tree_hash
        tree=deterministic_tree_hash(ROOT)
        hashes={n:read(b+'/manifest.json')['software']['rl_xy_source_tree_sha256'] for n,b in report['bundles'].items()}
        check('frozen_source_matches_every_run',all(v==tree for v in hashes.values()) and fixture['source_tree_sha256']==tree,hashes)
        report['source_tree_sha256']=tree;report['status']='PASS'
    except Exception as exc:
        report['status']='FAIL';report['error']=f'{type(exc).__name__}: {exc}'
        print(report['error'],file=sys.stderr)
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
