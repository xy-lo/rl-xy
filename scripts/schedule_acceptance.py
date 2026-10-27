#!/usr/bin/env python3
"""CPU/Docker acceptance for H0 transcription and optimizer schedule extension."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rl_xy.reporting import deterministic_tree_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--image', default='rlxy-astrasim:0.4.1')
    a=parser.parse_args()
    if Path(a.run_id).name != a.run_id or a.run_id in ('','.','..'):
        parser.error('run-id must be a unique path component')
    out=ROOT/'acceptance'/a.run_id
    out.mkdir(parents=True, exist_ok=False)
    report={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'image':a.image,'container':a.container,'commands':[],'checks':[],'bundles':{},
        'new_gpu_training':False,'hardware_calibrated':False,'full_wafer_migration_complete':False}
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    def command(label, argv, expected=0):
        log=out/(label+'.log');save()
        with log.open('w') as stream:
            result=subprocess.run(argv,cwd=ROOT,env={**os.environ,'RLXY_CONTAINER':a.container},
                                  stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
        report['commands'].append({'label':label,'argv':argv,'exit_status':result.returncode,
            'expected_exit':expected,'log':str(log.relative_to(ROOT)),
            'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
        save();print(f'{label}: {result.returncode} (expected {expected})',flush=True)
        if result.returncode != expected:raise ValueError(label)
    def check(name, passed, evidence):
        report['checks'].append({'check':name,'passed':bool(passed),'evidence':evidence});save()
        if not passed:raise ValueError(name)
    def read(path):return json.loads((ROOT/path).read_text())
    try:
        source=deterministic_tree_hash(ROOT)
        command('build',['docker','build','-t',a.image,'-f','docker/Dockerfile.schedule','.'])
        command('create',['bash','scripts/docker.sh','create-container',a.image])
        image_id=subprocess.check_output(['docker','inspect','--format','{{.Image}}',a.container],text=True).strip()
        report['image_id']=image_id
        docker=['docker','exec','-e',f'RLXY_IMAGE={a.image}','-e',f'RLXY_IMAGE_ID={image_id}',a.container,'python3']
        command('unit_tests',docker+['-m','unittest','discover','-s','tests','-v'])
        hfile=str((out/'h0.json').relative_to(ROOT))
        command('h0_contract_and_expected_gate',docker+['-m','rl_xy.cli','inspect-hardware',
            '--profile','configs/hardware/xy_h0_wsc_case3.json','--out',hfile,'--require-executable'],2)
        base='runs/schedule/'+a.run_id
        for label,case,rounds in [('legacy','toy_grpo_2die',1),('gqa_baseline','qwen2_group_fixture',1),
                                  ('updates','qwen2_updates_fixture',1),('two_rounds','qwen2_updates_fixture',2),
                                  ('repeat','qwen2_updates_fixture',1)]:
            target=base+'-'+label
            command(label,docker+['-m','rl_xy.cli','run','--case',f'configs/{case}/case.json',
                '--rounds',str(rounds),'--resource-model','--out',target])
            report['bundles'][label]=target
        command('native_fixtures',docker+['scripts/integration_fixtures.py','--output',base+'-native'])
        report['integration']=base+'-native'
        for label,bundle in report['bundles'].items():
            command('integrity_'+label,docker+['scripts/check_result.py',bundle])
        metrics={label:read(bundle+'/metrics.json') for label,bundle in report['bundles'].items()}
        check('legacy_toy_unchanged',metrics['legacy']['total_simulated_time_ns']==108832.27,metrics['legacy']['total_simulated_time_ns'])
        check('gqa_baseline_unchanged',metrics['gqa_baseline']['total_simulated_time_ns']==214196.1,metrics['gqa_baseline']['total_simulated_time_ns'])
        check('eight_updates_two_rounds_frozen_reference',metrics['two_rounds']['optimizer_updates']==8
            and metrics['two_rounds']['actor_version_end']==8 and metrics['two_rounds']['reference_version_end']==0,
            {k:metrics['two_rounds'][k] for k in ['optimizer_updates','actor_version_end','reference_version_end']})
        check('rounds_do_not_leak_resources',metrics['two_rounds']['memory_peak_bytes_by_location']==metrics['updates']['memory_peak_bytes_by_location'],
            metrics['two_rounds']['memory_peak_bytes_by_location'])
        check('partial_batch_denominators',metrics['updates']['minibatch_loss_denominators']==[6,3],metrics['updates']['minibatch_loss_denominators'])
        check('repeat_native_events_identical',(ROOT/report['bundles']['updates']/'events.jsonl').read_bytes()==
            (ROOT/report['bundles']['repeat']/'events.jsonl').read_bytes(),report['bundles']['repeat'])
        h=read(hfile)
        check('h0_source_keeps_54_and_geometry_conflict',h['aggregates']['modules']==54 and h['aggregates']['cores']==13824
            and h['aggregates']['peak_fp16_flops_per_ns']==14100480 and len(h['geometry_errors'])==3
            and not h['current_full_grpo_executable'],h)
        native=read(report['integration']+'/report.json')
        check('native_mesh_resources_unchanged',native['status']=='PASS' and all(c['passed'] for c in native['checks']),native['checks'])
        check('all_bundles_bound_to_unchanged_source',deterministic_tree_hash(ROOT)==source
            and all(read(b+'/manifest.json')['software']['rl_xy_source_tree_sha256']==source for b in report['bundles'].values())
            and native['source_tree_sha256']==source,source)
        report.update(status='PASS',source_tree_sha256=source,metrics=metrics)
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()
