#!/usr/bin/env python3
"""Pinned protocol regression and controlled 0.5/3/7B dual-platform native runs."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.reporting import deterministic_tree_hash
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--container',required=True)
    p.add_argument('--skip-full',action='store_true');a=p.parse_args();os.chdir(ROOT)
    out=ROOT/'acceptance'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    source=deterministic_tree_hash(ROOT);image='rlxy-astrasim:0.7.0'
    report={'status':'running','source_tree_sha256':source,'started_at_utc':datetime.now(timezone.utc).isoformat(),'commands':[],'bundles':{},'comparisons':{}}
    env={**os.environ,'RLXY_CONTAINER':a.container,'RLXY_RUNS_ROOT':str((ROOT/'runs').resolve())}
    def save():(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    def run(label,argv,timeout=None):
        timeout=int(timeout or os.environ.get('RLXY_COMMAND_TIMEOUT_SECONDS','7200'))
        path=out/(label+'.log');start=time.monotonic();save()
        with path.open('w') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=timeout)
        report['commands'].append(dict(label=label,argv=argv,exit_status=r.returncode,elapsed_wall_s=time.monotonic()-start,
          log=str(path.relative_to(ROOT)),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest()));save();print(label,r.returncode,flush=True)
        if r.returncode:raise ValueError(label+' failed; see '+str(path))
    def bundle(label,case,layout,platform='wafer',scenario='configs/wafer54_fixture/wafer.json',profile='configs/gpu/a100_48_reference.json'):
        if shutil.disk_usage(ROOT/'runs').free<8*1024**3:raise ValueError('less than 8GiB artifact space; preserve existing evidence')
        target='runs/colocation/'+a.run_id+'-'+label;report['bundles'][label]=target;save()
        argv=docker+['-m','rl_xy.cli','run-'+platform,'--case',str(case),'--scenario',scenario,'--colocation',str(layout),'--out',target]
        if platform=='gpu':argv+=['--gpu-profile',profile]
        run(label,argv);run(label+'_integrity',docker+['scripts/check_result.py',target]);return target
    try:
        run('build',['docker','build','-f','docker/Dockerfile.colocation','-t',image,'.'])
        run('create',['bash','scripts/docker.sh','create-container',image])
        identity=subprocess.check_output(['docker','inspect','--format','{{.Image}}',a.container],text=True).strip()
        report.update(image=image,image_id=identity,container=a.container)
        docker=['docker','exec','-e','RLXY_IMAGE='+image,'-e','RLXY_IMAGE_ID='+identity,a.container,'python3']
        for _k in ('RLXY_ASTRA_TIMEOUT_SECONDS',):
            if os.environ.get(_k):docker[6:6]=['-e',_k+'='+os.environ[_k]]
        run('unit_tests',docker+['-m','unittest','discover','-s','tests','-v'])
        for label,world,tr,ge,rounds,platform in [('wafer4_two',4,2,1,2,'wafer'),('gpu4_batch',4,2,2,1,'gpu'),('gpu10_standby',10,4,2,1,'gpu')]:
            c=out/(label+'_case');c.mkdir();raw=json.loads((ROOT/'configs/wafer4_fixture/case.json').read_text());raw['rounds']=rounds
            if tr==4:raw['model'].update(hidden_size=16,query_heads=4,kv_heads=4,head_dim=4,ffn_size=32,vocab_size=64)
            (c/'case.json').write_text(json.dumps(raw,indent=2)+'\n')
            for name in ['astra_system.json','astra_network.yml','astra_remote_memory.json']:shutil.copy2(ROOT/'configs/wafer4_fixture'/name,c/name)
            config=json.loads((ROOT/'configs/colocation/tp2_to_tp1.json').read_text());config.update(train_tp=tr,rollout_tp=ge)
            layout=out/(label+'_layout.json');layout.write_text(json.dumps(config,indent=2)+'\n')
            profile=out/(label+'_gpu.json');gp=json.loads((ROOT/'configs/gpu/a100_48_reference.json').read_text());gp.update(gpu_count=world,gpus_per_node=2)
            profile.write_text(json.dumps(gp,indent=2)+'\n')
            bundle(label,(c/'case.json').relative_to(ROOT),layout.relative_to(ROOT),platform,'configs/wafer4_fixture/wafer.json',str(profile.relative_to(ROOT)))
        for name,script in [('gpu_network','gpu_network_fixtures.py'),('private_sram','private_sram_fixtures.py')]:
            run(name,docker+['scripts/'+script,'--out','runs/colocation/'+a.run_id+'-'+name])
        if not a.skip_full:
            for label,case in [('05b','configs/verl_sync_cases/qwen05b/case.json'),('3b','configs/verl_sync_standard/qwen3b/case.json'),('7b','configs/verl_sync_standard/qwen7b/case.json')]:
                layout='configs/colocation/tp4_to_tp2.json' if label=='7b' else 'configs/colocation/tp2_to_tp1.json'
                w=bundle(label+'_wafer',case,layout)
                g=bundle(label+'_gpu',case,layout,'gpu')
                target='runs/sim_comparison/'+a.run_id+'-'+label;report['comparisons'][label]=target
                run('compare_'+label,docker+['scripts/compare_colocated_platforms.py','--wafer',w,'--gpu',g,'--out',target])
        if deterministic_tree_hash(ROOT)!=source:raise ValueError('source changed during acceptance')
        for target in report['bundles'].values():
            m=json.loads((ROOT/target/'manifest.json').read_text())
            if m['software']['rl_xy_source_tree_sha256']!=source:raise ValueError('bundle source identity differs')
        report['status']='PASS'
    except BaseException as e:report.update(status='FAIL',error=type(e).__name__+': '+str(e));raise
    finally:report['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
if __name__=='__main__':main()
