#!/usr/bin/env python3
"""Docker acceptance plus complete actual-GPU-step wafer replay and comparison."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.reporting import deterministic_tree_hash

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--container',required=True);p.add_argument('--gpu',required=True,type=Path)
    a=p.parse_args()
    if Path(a.run_id).name!=a.run_id or a.run_id in ('','.','..'):p.error('unique run-id component required')
    out=ROOT/'acceptance'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'image':'rlxy-astrasim:0.5.0','container':a.container,'commands':[],'checks':[],'bundles':{},
        'gpu_report':str(a.gpu/'report.json'),'hardware_calibrated':False,'physical_chip_deployment':False,
        'full_verl_feature_migration_complete':False,'artifact_storage_root':os.environ.get('RLXY_RUNS_ROOT')}
    def save():(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    def command(label,argv,expected=0,timeout=1800):
        save();log=out/(label+'.log')
        with log.open('w') as stream:
            r=subprocess.run(argv,cwd=ROOT,env={**os.environ,'RLXY_CONTAINER':a.container},stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        report['commands'].append({'label':label,'argv':argv,'exit_status':r.returncode,'expected_exit':expected,
            'log':str(log.relative_to(ROOT)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
        save()
        try:print(f'{label}: {r.returncode} (expected {expected})',flush=True)
        except BrokenPipeError:
            # The evidence is persisted before progress output. Losing an SSH
            # stdout consumer must not invalidate a completed simulator run.
            sys.stdout=open(os.devnull,'w')
        if r.returncode!=expected:raise ValueError(label)
    def check(name,condition,evidence):
        report['checks'].append({'check':name,'passed':bool(condition),'evidence':evidence});save()
        if not condition:raise ValueError(name)
    def read(path):return json.loads((ROOT/path).read_text())
    try:
        source=deterministic_tree_hash(ROOT)
        gpu=read(a.gpu/'report.json')
        check('real_gpu_completed',gpu['status']=='PASS' and gpu.get('exit_status')==0,
            {'report':str(a.gpu/'report.json'),'log_sha256':gpu.get('launch_log_sha256')})
        command('build',['docker','build','-t',report['image'],'-f','docker/Dockerfile.wafer','.'])
        command('create',['bash','scripts/docker.sh','create-container',report['image']])
        image=subprocess.check_output(['docker','inspect','--format','{{.Image}}',a.container],text=True).strip();report['image_id']=image
        docker=['docker','exec','-e','RLXY_IMAGE='+report['image'],'-e','RLXY_IMAGE_ID='+image,a.container,'python3']
        command('unit_tests',docker+['-m','unittest','discover','-s','tests','-v'])
        command('native_private_sram',docker+['scripts/private_sram_fixtures.py','--out','runs/wafer/'+a.run_id+'-private'])
        report['private_fixtures']='runs/wafer/'+a.run_id+'-private'
        command('native_old_regressions',docker+['scripts/integration_fixtures.py','--output','runs/wafer/'+a.run_id+'-native'])
        report['native_fixtures']='runs/wafer/'+a.run_id+'-native'
        for label,directory in [('mesh4','wafer4_fixture'),('mesh54','wafer54_fixture')]:
            target='runs/wafer/'+a.run_id+'-'+label
            command(label,docker+['-m','rl_xy.cli','run-wafer','--case',f'configs/{directory}/case.json',
                '--scenario',f'configs/{directory}/wafer.json','--out',target])
            report['bundles'][label]=target
            command('early_integrity_'+label,docker+['scripts/check_result.py',target])
        target='runs/wafer/'+a.run_id+'-legacy'
        command('legacy',docker+['-m','rl_xy.cli','run','--case','configs/toy_grpo_2die/case.json','--resource-model','--out',target])
        report['bundles']['legacy']=target
        compiled='runs/wafer/'+a.run_id+'-matched-case'
        command('compile_full_gpu_step',docker+['scripts/prepare_matched_case.py','--gpu',str(a.gpu),
            '--template','configs/wafer54_fixture','--model-config',
            'runs/legacy_reuse/inputs-20261026-final/E1_gsm8k_qwen05b/inputs/model_config.json','--out',compiled])
        target='runs/wafer/'+a.run_id+'-matched'
        command('matched_54_full_lengths',docker+['-m','rl_xy.cli','run-wafer','--case',compiled+'/case.json',
            '--scenario','configs/wafer54_fixture/wafer.json','--out',target])
        report['bundles']['matched']=target
        comparison='runs/comparison/'+a.run_id
        command('matched_comparison',docker+['scripts/compare_gpu_wafer.py','--gpu',str(a.gpu),'--wafer',target,'--out',comparison])
        report['comparison']=comparison
        for label,bundle in report['bundles'].items():command('integrity_'+label,docker+['scripts/check_result.py',bundle])
        metrics={label:read(bundle+'/metrics.json') for label,bundle in report['bundles'].items()}
        check('legacy_time_unchanged',metrics['legacy']['total_simulated_time_ns']==108832.27,metrics['legacy']['total_simulated_time_ns'])
        m=metrics['matched'];cp=read(compiled+'/compilation.json')
        check('all_54_dies_and_private_cores',m['dies']==54 and m['cores_per_die']==256 and m['data_parallel']==27 and m['tensor_parallel']==2,
            {k:m[k] for k in ('dies','cores_per_die','data_parallel','tensor_parallel')})
        check('full_64_real_trajectories_no_modified_lengths',m['trajectory_count']==64 and m['response_tokens']==256
            and cp['full_step_replayed'] and not cp['response_lengths_modified'] and cp['counterfactual_response_cap'] is None,cp)
        check('gpu_identity_and_algorithms_matched',read(comparison+'/comparison.json')['status']=='MATCHED_SCENARIO_PREDICTION',comparison)
        check('all_evidence_bound_to_same_source',deterministic_tree_hash(ROOT)==source and all(
            read(bundle+'/manifest.json')['software']['rl_xy_source_tree_sha256']==source for bundle in report['bundles'].values()),source)
        report.update(status='PASS',source_tree_sha256=source,metrics=metrics)
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()
