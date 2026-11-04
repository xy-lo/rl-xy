#!/usr/bin/env python3
"""Remote-only final acceptance dispatcher, gated by complete 4B evidence.

No configuration is selected using final-window response lengths or timing.
Failures preserve logs and require diagnosis; no automatic benchmark weakening.
GitHub publication follows human/agent review, outside this dispatcher.
"""
import argparse,json,os,time,subprocess,hashlib,fcntl,concurrent.futures,re
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def select_tune_policy(root,variants,ceiling):
 """Choose only complete, audited 4B TUNE policies before final-window access."""
 if not variants:raise ValueError('no 4B candidate policy')
 accepted=[]
 for variant in variants:
  run_name=variant['run'];policy=variant.get('home_policy','score')
  if policy not in ('score','service'):raise ValueError('unsupported selected home policy')
  driver=read(root/(run_name+'-status.json'))
  if not driver.get('complete') or not driver.get('stages') or any(x['returncode']!=0 for x in driver['stages']):
   raise ValueError('candidate TUNE driver incomplete or failed')
  tune=read(root/run_name/'native-run/result.json')
  assert tune['model'] in ('4B','Qwen3-4B') and tune['window']=='tune','selection must not use final-window timing'
  ready_order=variant.get('training_ready_order','input')
  assert ready_order in ('input','ready_lpt')
  cap=variant.get('late_region_batch_cap',0)
  assert type(cap) is int and cap in (0,1,2,4)
  tail=variant.get('training_tail_fine_samples',0)
  assert type(tail) is int and tail in (0,8,16)
  assert read(root/run_name/'native-run/resource-layout.json').get('training_tail_fine_samples',0)==tail
  packing=variant.get('adaptive_training_batch_max',0)
  assert type(packing) is int and packing in (0,1,2,4,8)
  assert read(root/run_name/'native-run/resource-layout.json').get('adaptive_training_batch_max',0)==packing
  assert read(root/run_name/'native-run/resource-layout.json').get('late_region_batch_cap_per_pair',0)==cap
  assert read(root/run_name/'native-run/resource-layout.json').get('training_ready_order','input')==ready_order
  assert tune['requests']==tune['completed']==tune['trained_samples']==1536 and tune['capacity_passed']
  assert read(root/run_name/'inference-audit.json')['status']=='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT'
  if 'offline_plan' in variant:
   offline=Path(variant['offline_plan']);plan=read(offline/'result.json')
   assert plan['model']=='4B' and plan['window']=='tune'
   assert plan.get('home_policy','score')==policy and plan['mode']==variant['planner_mode']
   assert plan['forecast_mode']=='controlled-proxy' and read(offline/'distribution.json')['exact_weight']==1
  import math
  if not math.isfinite(tune['total_s']) or tune['total_s']<=0:raise ValueError('invalid TUNE time')
  accepted.append(dict(variant,device_s=tune['total_s']))
 chosen=min(accepted,key=lambda x:(x['device_s'],x['run']))
 if chosen['device_s']>ceiling:raise ValueError('4B full TUNE exceeds frozen prior-system device ceiling; iterate before final models')
 return accepted,chosen

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config)
 root=Path(c['evidence_root']);name=c.get('campaign_name','final-campaign-01')
 if not re.fullmatch('[A-Za-z0-9_-]+',name):raise ValueError('invalid campaign name')
 base=root/name;base.mkdir(exist_ok=True)
 lock=(base/'dispatcher.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('final dispatcher already active')
 source=Path(c['source']);python=c['python'];native=Path(c['native_root'])
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
 assert commit==c['source_commit'],'immutable source differs'
 assert hashlib.sha256(Path(c['tune_ceiling_source']).read_bytes()).hexdigest()==c['tune_ceiling_source_sha256'],'prior TUNE evidence differs'
 assert hashlib.sha256(Path(c['baseline_csv']).read_bytes()).hexdigest()==c['baseline_csv_sha256'],'baseline changed before final acceptance'
 if (base/'status.json').exists() and read(base/'status.json').get('phase')!='waiting-4b-gates':
  raise RuntimeError('dispatcher already advanced; inspect evidence rather than duplicate jobs')
 state=dict(phase='waiting-4b-gates',complete=False,source_commit=commit,config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
  dispatcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pid=os.getpid())
 def status(phase,**updates):
  state.update(phase=phase,**updates);write(base/'status.json',state)
 def run(cmd,folder,name,env):
  folder.mkdir(exist_ok=True,parents=True);start=time.time()
  with (folder/(name+'.log')).open('w') as log:result=subprocess.run(list(map(str,cmd)),cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
  row=dict(argv=list(map(str,cmd)),returncode=result.returncode,wall_s=time.time()-start)
  write(folder/(name+'-status.json'),row)
  if result.returncode:raise RuntimeError('stage failed: '+str(folder/name))
  return row
 try:
  while True:
   gates=[]
   for name in c['required_4b_runs']:
    path=root/(name+'-status.json')
    if not path.exists():gates.append(False);continue
    s=read(path)
    if any(x['returncode']!=0 for x in s.get('stages',[])):raise RuntimeError('4B gate failed: '+name)
    gates.append(s.get('complete',False))
   status('waiting-4b-gates',ready_gates=sum(gates),total_gates=len(gates))
   if all(gates):break
   time.sleep(30)
  variants=c.get('candidate_4b_policies',[dict(run=c['tune_run'],home_policy=c.get('home_policy','score'),planner_mode=c['planner_mode'])])
  accepted,chosen=select_tune_policy(root,variants,c['max_tune_device_s'])
  write(base/'tune-policy-selection.json',dict(candidates=accepted,selected=chosen,
   selection='native audited TUNE only; complete all declared variants before final-window planning; no global optimum claim'))
  # Independently reconstruct declared forecast and native binding for each
  # candidate. Ranking from final-window outcomes is never fed back here.
  import sys
  sys.path.insert(0,str(source/'scripts'))
  from audit_idea1_planning import audit_planning
  for variant in variants:
   if 'offline_plan' in variant:
    proof=audit_planning(root/variant['run']/'native-run',Path(variant['offline_plan']))
    write(base/(variant['run']+'-planning-audit.json'),proof)
  selected_home=chosen.get('home_policy','score');selected_mode=chosen.get('planner_mode',c['planner_mode']);selected_order=chosen.get('training_ready_order','input');selected_cap=chosen.get('late_region_batch_cap',0);selected_packing=chosen.get('adaptive_training_batch_max',0);selected_tail=chosen.get('training_tail_fine_samples',0)
  stress=root/c['training_stress_run']/'native-run'
  events=read(stress/'mesh-coordination-events.json')
  if not any(e['kind']=='training-proposed' for e in events):raise RuntimeError('training mesh selection not exercised')
  # Decisions may be wrapped by a summary; inspect the true native transfer list.
  stress_result=read(stress/'result.json')
  if stress_result['training_transfers']<1:raise RuntimeError('training fixture did not exercise donation')
  assert read(root/c['training_stress_run']/'inference-audit.json')['status']=='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT'
  status('preparing-final-models',tune_device_s=chosen['device_s'],selected_home_policy=selected_home,selected_tune=chosen['run'],tune_ceiling_s=c['max_tune_device_s'])
  envbase=dict(os.environ,LD_LIBRARY_PATH=c['ld_library_path'])
  cost14=Path(c['cost14_root'])
  if not cost14.exists():
   run([python,source/'scripts/prepare_idea1_joint.py','--model','14B','--output',cost14],base,'prepare14',envbase)
  else:
   # Verify immutable source bytes even when preparation previously completed.
   expected=read(source/'experiments/2026-11-02-idea1-joint/14B/baseline-metadata.json')['cost_source_files']
   assert all(hashlib.sha256((cost14/k).read_bytes()).hexdigest()==v for k,v in expected.items())
  def execute_model(model):
   folder=base/(model+'-test1');folder.mkdir(exist_ok=False)
   cost=cost14 if model=='14B' else Path(c['cost4_root'])
   env=dict(envbase,RLXY_COST_ROOT=str(cost),PYTHONPATH=str(cost/'src'))
   args=[python,source/'scripts/workload_planner/plan_system.py','--model',model,'--window','test1','--mode',selected_mode,'--home-policy',selected_home,'--training-ready-order',selected_order,'--late-region-batch-cap',selected_cap,'--adaptive-training-batch-max',selected_packing,'--training-tail-fine-samples',selected_tail,
    '--forecast-mode','controlled-proxy','--allow-controlled-proxy','--proxy-weight','1','--output',folder/'plan']
   # Optional bank geometries are declared before final execution; the planner
   # still reconstructs capacity, forecast and placement for this batch.
   candidates=c.get('resource_candidates_by_model',{}).get(model,[])
   if not isinstance(candidates,list) or any(not isinstance(x,str) for x in candidates):
    raise ValueError('resource candidate bank must contain strings')
   for candidate in candidates:args.extend(['--resource-candidate',candidate])
   run(args,folder,'planning',env)
   deploy=read(folder/'plan/deployment.json');files=deploy['files']
   layout=read(files['selected-plan.json'])
   assert layout.get('adaptive_training_batch_max',0)==selected_packing
   assert layout.get('training_tail_fine_samples',0)==selected_tail
   assert layout.get('late_region_batch_cap_per_pair',0)==selected_cap
   cmd=[python,source/'scripts/workload_planner/run_mesh_system.py','--model',model,'--window','test1','--training-scope','wafer',
    '--assignment',files['selected-plan.json'],'--resource-layout',files['selected-plan.json'],'--inference-layout',files['inference-layout.json'],
    '--workload-forecast',files['forecast.json'],'--native-root',native,'--fast-cost-validation',c['cost_validation'],
    '--rollout-cap','64','--mesh-routing','adaptive_xy_yx','--mesh-controller','--output',folder/'native-run']
   run(cmd,folder,'native',env)
   run([python,source/'scripts/audit_idea1_inference.py',folder/'native-run','--output',folder/'inference-audit.json'],folder,'audit',env)
   result=read(folder/'native-run/result.json');assert result['requests']==result['completed']==result['trained_samples']==1536
   assert read(folder/'inference-audit.json')['status']=='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT'
   return dict(model=model,window='test1',device_s=result['total_s'],planning_host_s=deploy['host_wall_s'],run=str(folder),
    inference_tp_by_group=result['inference_tp_by_group'],training_tp_by_region=result['training_tp_by_region'])
  status('final-three-models-running')
  results=[];failures=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   jobs={pool.submit(execute_model,m):m for m in ('4B','8B','14B')}
   for future in concurrent.futures.as_completed(jobs):
    try:results.append(future.result())
    except BaseException as error:failures.append(dict(model=jobs[future],error=repr(error)))
    write(base/'partial-results.json',dict(results=results,failures=failures))
    status('final-three-models-running',finished_models=len(results)+len(failures),failures=failures)
  if failures:raise RuntimeError('one or more final models failed; completed results retained')
  import csv
  baseline=Path(c['baseline_csv']);rows=list(csv.DictReader(baseline.open()))
  compare=[]
  for x in sorted(results,key=lambda x:int(x['model'][:-1])):
   old=next(r for r in rows if r['model']==x['model'] and r['window']=='test1')
   gpu=float(old['gpu_s'])
   compare.append(dict(x,gpu_s=gpu,h0_s=float(old['h0_s']),original_generation_s=float(old['original_generation_s']),
    previous_innovation3_s=float(old['previous_innovation3_s']),old_complete_system_device_s=float(old['device_s']),
    speedup_gpu_device=gpu/x['device_s'],reduction_old_complete_device=1-x['device_s']/float(old['device_s'])))
  report=dict(status='FINAL_NATIVE_RESULTS_READY_FOR_REVIEW',comparison=compare,
   selected_tune_policy=chosen,source_commit=commit,baseline_csv_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
   information='offline controlled same-prompt proxy alpha1, current evaluation samples excluded; NOT chronological RL history',
   evaluation='historically used test1 window, not pristine held-out',
   timing='native device seconds separate from host planning; old Mac planning proxy excluded from old device comparison',
   scope='54 wafer dies vs48 A100 baseline; not equal power/area/cost; calibrated trace cost simulation, no numerical training/learning or physical wafer')
  write(base/'comparison.json',report)
  lines=['# 三模型原生验收：待最终审查','','|模型|GPU/s|H0/s|原生成机制/s|旧创新三/s|旧完整系统设备/s|新系统设备/s|规划主机/s|GPU加速|较旧完整缩短|',
   '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  for x in compare:
   lines.append('|%s|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3fx|%.2f%%|'%(x['model'],x['gpu_s'],x['h0_s'],x['original_generation_s'],x['previous_innovation3_s'],x['old_complete_system_device_s'],x['device_s'],x['planning_host_s'],x['speedup_gpu_device'],100*x['reduction_old_complete_device']))
  lines+=['',report['information'],'',report['evaluation'],'',report['timing'],'',report['scope'],'','远端等价检查、最终瓶颈审查和证据交付由主任务继续；用户已暂缓GitHub上传，当前未标记整体完成。']
  (base/'comparison.md').write_text('\n'.join(lines)+'\n')
  status('results-ready-review-required',results_ready=True,next='audit same-window/cost identities, bottlenecks and regressions; finalize remote code/evidence and equivalent checks; GitHub upload deferred by user')
 except BaseException as error:
  status('needs-diagnosis',error=repr(error),next='inspect failed stage; do not duplicate or weaken benchmark');raise

if __name__=='__main__':main()
