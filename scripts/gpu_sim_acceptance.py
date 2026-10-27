#!/usr/bin/env python3
"""48-A100 near-budget simulation, GPU-favoring and wafer-rate sensitivities."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--container',required=True)
    p.add_argument('--wafer',type=Path,default=Path('runs/wafer/wafer-20261027-final-matched'))
    p.add_argument('--case',type=Path,default=Path('runs/wafer/wafer-20261027-final-matched-case/case.json'))
    a=p.parse_args()
    if Path(a.run_id).name!=a.run_id or a.run_id in ('','.','..'):p.error('unique run-id required')
    out=ROOT/'acceptance'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'image':'rlxy-astrasim:0.6.0','container':a.container,'commands':[],'checks':[],'bundles':{},
        'frozen_wafer_control':str(a.wafer),'comparisons':{},'hardware_calibrated':False,
        'physical_chip_deployment':False,'observed_GPU_seconds_used':False,
        'artifact_storage_root':os.environ.get('RLXY_RUNS_ROOT')}
    def save():(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    def progress(s):
        try:print(s,flush=True)
        except BrokenPipeError:sys.stdout=open(os.devnull,'w')
    def record(label,argv,log,rc,allow_fail=False):
        report['commands'].append({'label':label,'argv':argv,'exit_status':rc,'expected_exit':0,
            'log':str(log.relative_to(ROOT)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
        save();progress(label+': '+str(rc))
        if rc and not allow_fail:raise ValueError(label)
    def command(label,argv):
        save();log=out/(label+'.log')
        with log.open('w') as f:r=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=5400,
            env={**os.environ,'RLXY_CONTAINER':a.container})
        record(label,argv,log,r.returncode)
    def read(path):return json.loads((ROOT/path).read_text())
    def check(name,passed,evidence):
        report['checks'].append({'check':name,'passed':bool(passed),'evidence':evidence});save()
        if not passed:raise ValueError(name)
    try:
        source=deterministic_tree_hash(ROOT)
        check('frozen_control_already_valid',read(a.wafer/'summary.json')['status']=='PASS',str(a.wafer))
        report['frozen_wafer_manifest_sha256']=hashlib.sha256((ROOT/a.wafer/'manifest.json').read_bytes()).hexdigest()
        command('build',['docker','build','-t',report['image'],'-f','docker/Dockerfile.gpu','.'])
        command('create',['bash','scripts/docker.sh','create-container',report['image']])
        image=subprocess.check_output(['docker','inspect','--format','{{.Image}}',a.container],text=True).strip();report['image_id']=image
        docker=['docker','exec','-e','RLXY_IMAGE='+report['image'],'-e','RLXY_IMAGE_ID='+image,a.container,'python3']
        command('unit_tests',docker+['-m','unittest','discover','-s','tests','-v'])
        native='runs/gpu_sim/'+a.run_id+'-native';report['native_fixtures']=native
        command('gpu_network_fixtures',docker+['scripts/gpu_network_fixtures.py','--out',native])
        private='runs/gpu_sim/'+a.run_id+'-private';report['private_fixtures']=private
        command('private_sram',docker+['scripts/private_sram_fixtures.py','--out',private])
        old='runs/gpu_sim/'+a.run_id+'-old-native';report['old_native_fixtures']=old
        command('old_native',docker+['scripts/integration_fixtures.py','--output',old])
        small=read(Path('configs/gpu/a100_48_reference.json'));small.update(gpu_count=4,gpus_per_node=2,name='4GPU_native_fixture_not_primary_budget')
        profile=out/'gpu4.json';profile.write_text(json.dumps(small,indent=2)+'\n')
        target='runs/gpu_sim/'+a.run_id+'-gpu4'
        command('gpu4',docker+['-m','rl_xy.cli','run-gpu','--case','configs/wafer4_fixture/case.json',
            '--scenario','configs/wafer4_fixture/wafer.json','--gpu-profile',str(profile.relative_to(ROOT)),'--out',target])
        report['bundles']['gpu4']=target
        command('early_gpu4_integrity',docker+['scripts/check_result.py',target])
        target='runs/gpu_sim/'+a.run_id+'-wafer4'
        command('wafer4_regression',docker+['-m','rl_xy.cli','run-wafer','--case','configs/wafer4_fixture/case.json',
            '--scenario','configs/wafer4_fixture/wafer.json','--out',target]);report['bundles']['wafer4']=target
        command('early_wafer4_integrity',docker+['scripts/check_result.py',target])
        check('wafer4_clock_unchanged',read(Path(target)/'metrics.json')['total_simulated_time_ns']==1675.97,target)
        # Separate output directories and one CPU process per simulation; no GPU
        # device access or background numerical training is requested.
        jobs=[]
        for label,profile_name in [('reference','a100_48_reference'),('optimistic','a100_48_optimistic'),('wafer_rates','wafer_typed_rate_sensitivity')]:
            target='runs/gpu_sim/'+a.run_id+'-'+label;report['bundles'][label]=target
            if label=='wafer_rates':
                argv=docker+['-m','rl_xy.cli','run-wafer','--case',str(a.case),'--scenario','configs/gpu/'+profile_name+'.json','--out',target]
            else:
                argv=docker+['-m','rl_xy.cli','run-gpu','--case',str(a.case),'--scenario','configs/wafer54_fixture/wafer.json',
                    '--gpu-profile','configs/gpu/'+profile_name+'.json','--out',target]
            log=out/(label+'.log');f=log.open('w');proc=subprocess.Popen(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            jobs.append((label,argv,log,f,proc));save()
        failures=[]
        for label,argv,log,f,proc in jobs:
            rc=proc.wait(timeout=5400);f.close();record(label,argv,log,rc,allow_fail=True)
            if rc:failures.append(label)
        if failures:raise ValueError('full simulations failed: '+','.join(failures))
        for label in ('reference','optimistic','wafer_rates'):
            command('integrity_'+label,docker+['scripts/check_result.py',report['bundles'][label]])
        for label,w,g in [('primary',str(a.wafer),report['bundles']['reference']),
            ('gpu_favoring',str(a.wafer),report['bundles']['optimistic']),
            ('wafer_rate_sensitivity',report['bundles']['wafer_rates'],report['bundles']['reference']),
            ('joint_sensitivity',report['bundles']['wafer_rates'],report['bundles']['optimistic'])]:
            target='runs/sim_comparison/'+a.run_id+'-'+label
            command('compare_'+label,docker+['scripts/compare_simulated_platforms.py','--wafer',w,'--gpu',g,'--out',target])
            report['comparisons'][label]=target
        all_metrics={label:read(Path(path)/'metrics.json') for label,path in report['bundles'].items()}
        check('both_GPU_runs_are_48devices_TP2_DP24_64real_lengths',all(
            all_metrics[k]['dies']==48 and all_metrics[k]['cores_per_die']==108 and all_metrics[k]['data_parallel']==24
            and all_metrics[k]['trajectory_count']==64 and all_metrics[k]['response_tokens']==256
            and all_metrics[k]['optimizer_updates']==1 for k in ('reference','optimistic')),'complete same captured workload')
        check('same_current_source_for_all_new_bundles',deterministic_tree_hash(ROOT)==source and all(
            read(Path(path)/'manifest.json')['software']['rl_xy_source_tree_sha256']==source for path in report['bundles'].values()),source)
        report.update(status='PASS',source_tree_sha256=source,metrics=all_metrics,
            outcomes={k:read(Path(path)/'comparison.json') for k,path in report['comparisons'].items()})
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()
