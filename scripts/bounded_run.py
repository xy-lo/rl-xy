#!/usr/bin/env python3
"""Run one owned CPU/Docker command with explicit resource and evidence bounds.

This is an engineering runner, not permission to label an unsupported strategy
as a paper baseline. Formal runs require a separate successful alignment gate.
"""
import argparse,hashlib,json,os,subprocess,time,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def resolve_artifact_paths(repo_root,runs_root,record):
    """Admit only the existing default root or this task's real home folder.

    Existing ROOT/runs may be a managed artifact symlink; it is never changed.
    The alternative must be the direct, nonsymlink ROOT/runs-home-T42 folder.
    Record containment is checked after symlink/.. resolution before mkdir.
    """
    repo=Path(repo_root).resolve();default=(repo/'runs').resolve()
    requested=repo/'runs' if runs_root is None else Path(runs_root).expanduser()
    if not requested.is_absolute():requested=repo/requested
    alternative=repo/'runs-home-T42'
    requested_lexical=Path(os.path.abspath(requested))
    if requested_lexical==alternative and alternative.is_symlink():
        raise ValueError('runs-home-T42 must be a real dedicated directory, not a symlink')
    actual=requested.resolve()
    if actual in (repo,Path('/')) or repo.is_relative_to(actual):
        raise ValueError('artifact root must not be the repository, filesystem root or their ancestor')
    allowed_alternative=(actual==alternative and not alternative.is_symlink())
    if actual!=default and not allowed_alternative:
        raise ValueError('artifact root must be default runs resolved or direct ROOT/runs-home-T42')
    if not actual.is_dir():raise ValueError('artifact root must already be an existing directory')
    selected=Path(record).expanduser().resolve()
    if selected==actual or not selected.is_relative_to(actual):
        raise ValueError('record must be a strict descendant of the actual artifact root')
    return actual,selected


def artifact_mount_args(repo_root,runs,*,image_source):
    """Mount admitted output paths without hiding immutable image sources."""
    result=[] if image_source else ['--volume',str(repo_root)+':/workspace/rl-xy:ro']
    result+=['--volume',str(runs)+':'+str(runs)]
    if image_source:result+=['--volume',str(runs)+':/workspace/rl-xy/runs']
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--image',default='rlxy-astrasim:0.9.5')
    p.add_argument('--memory-gib',type=int,default=32);p.add_argument('--cpus',type=int,default=2);p.add_argument('--timeout-s',type=int,default=900)
    p.add_argument('--pids-limit',type=int,default=128);p.add_argument('--record',type=Path,required=True);p.add_argument('--runs-root',type=Path,help='existing default runs or direct ROOT/runs-home-T42 dedicated artifact directory');p.add_argument('--purpose',choices=['engineering_regression','formal_baseline'],default='engineering_regression')
    p.add_argument('--max-output-mib',type=int,default=2048);p.add_argument('--minimum-free-gib',type=int,default=30);p.add_argument('--disk-reserve-gib',type=int,default=20)
    p.add_argument('--image-source',action='store_true',help='execute immutable source inside image; mount only experiment outputs')
    p.add_argument('--alignment-gate',type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if not a.command or a.command[0]!='python3':p.error('explicit python3 argv required')
    if not 1<=a.memory_gib<=64 or not 1<=a.cpus<=8 or not 1<=a.timeout_s<=14400:p.error('resource limit outside admitted range')
    if not a.name.startswith(('rl-xy-T40-','rl-xy-T42-')):p.error('dedicated rl-xy-T40- container name required')
    if a.purpose=='formal_baseline':
        if not a.alignment_gate:p.error('formal baseline requires explicit alignment gate')
        g=json.loads(a.alignment_gate.read_text())
        if g.get('runtime_alignment_ready') is not True or g.get('cost_validation_ready') is not True:
            p.error('runtime and cost gates must both pass; configuration validity alone is insufficient')
    runs,a.record=resolve_artifact_paths(ROOT,a.runs_root,a.record)
    a.record.mkdir(parents=True,exist_ok=False)
    def run(argv,**kw):return subprocess.run(argv,check=True,**kw)
    image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',a.image],text=True).strip()
    if min(a.max_output_mib,a.minimum_free_gib,a.disk_reserve_gib,a.pids_limit)<=0:raise ValueError('positive output/disk/PID budgets required')
    artifact_fs_start=shutil.disk_usage(runs)
    if artifact_fs_start.free<a.minimum_free_gib*1024**3:raise ValueError('free-disk start admission failed on '+str(runs))
    if subprocess.run(['docker','container','inspect',a.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False).returncode==0:raise ValueError('container name already exists')
    argv=['docker','create','--name',a.name,'--network','none','--memory',str(a.memory_gib)+'g','--memory-swap',str(a.memory_gib)+'g',
       '--cpus',str(a.cpus),'--pids-limit',str(a.pids_limit),'--user',str(os.getuid())+':'+str(os.getgid()),
       *artifact_mount_args(ROOT,runs,image_source=a.image_source),
       '--workdir','/workspace/rl-xy','--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONHASHSEED=0',
       '--env','PYTHONPATH=/workspace/rl-xy/src:/opt/astra-sim','--env','ASTRA_SIM_ROOT=/opt/astra-sim',
       '--env','ASTRA_SIM_BIN=/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware',
       '--env','RLXY_IMAGE='+a.image,'--env','RLXY_IMAGE_ID='+image_id,'--entrypoint','/usr/bin/env',a.image,*a.command]
    report={'status':'STARTING','purpose':a.purpose,'orchestrator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'PYTHONHASHSEED':'0','image':a.image,'image_id':image_id,'container':a.name,
        'source_execution_mode':'immutable_image' if a.image_source else 'host_readonly_bind',
        'artifact_root_actual':str(runs),'artifact_root_requested':str(a.runs_root) if a.runs_root is not None else str(ROOT/'runs'),
        'record_actual':str(a.record),'artifact_filesystem_device_id':runs.stat().st_dev,
        'artifact_filesystem_total_bytes':artifact_fs_start.total,'artifact_filesystem_free_start_bytes':artifact_fs_start.free,
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'memory_limit_GiB':a.memory_gib,'cpus':a.cpus,
        'output_limit_bytes':a.max_output_mib*1024**2,'minimum_free_start_bytes':a.minimum_free_gib*1024**3,'disk_reserve_bytes':a.disk_reserve_gib*1024**3,
        'timeout_s':a.timeout_s,'network':'none','GPU_devices_requested':False,'create_argv':argv,'command':a.command}
    def save():(a.record/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save();created=False;start=time.monotonic()
    try:
        run(argv,stdout=subprocess.PIPE);created=True
        with (a.record/'console.log').open('w') as log:
            proc=subprocess.Popen(['docker','start','--attach',a.name],stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                used=sum(p.stat().st_size for p in a.record.rglob('*') if p.is_file())
                free=shutil.disk_usage(runs).free
                reason=('timeout' if time.monotonic()-start>=a.timeout_s else
                    'artifact_size' if used>a.max_output_mib*1024**2 else
                    'free_disk_reserve' if free<a.disk_reserve_gib*1024**3 else None)
                if reason:
                    # A grace period lets a serializer write hundreds of MB
                    # after a disk guard fires. Stop only this owned container
                    # immediately; partial evidence is intentionally retained.
                    subprocess.run(['docker','kill',a.name],stdout=subprocess.DEVNULL,check=False)
                    proc.wait(timeout=30);report.update(budget_failed=reason,artifact_bytes_at_stop=used,free_disk_bytes_at_stop=free)
                    report['budget_stop_method']='immediate_docker_kill'
                    if reason=='timeout':report['timeout_reached']=True
                    break
                time.sleep(.5)
            rc=124 if report.get('budget_failed') else proc.returncode
        state=json.loads(subprocess.check_output(['docker','inspect',a.name],text=True))[0]
        report.update(status='PASS' if rc==0 and not state['State'].get('OOMKilled') else 'FAIL',
            exit_status=rc,docker_exit_status=state['State']['ExitCode'],OOMKilled=state['State'].get('OOMKilled'),
            memory_limit_bytes=state['HostConfig']['Memory'],memory_swap_limit_bytes=state['HostConfig']['MemorySwap'])
    except BaseException as e:
        report.update(status='FAIL',error=type(e).__name__+': '+str(e));raise
    finally:
        report['artifact_bytes']=sum(p.stat().st_size for p in a.record.rglob('*') if p.is_file())
        report['artifact_filesystem_free_end_bytes']=shutil.disk_usage(runs).free
        report['elapsed_wall_s']=time.monotonic()-start;report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        if (a.record/'console.log').exists():report['console_sha256']=hashlib.sha256((a.record/'console.log').read_bytes()).hexdigest()
        save()
    print(json.dumps({k:report.get(k) for k in ['status','exit_status','OOMKilled','elapsed_wall_s']}));return 0 if report['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
