#!/usr/bin/env python3
"""Auditable composition: offline candidates -> binding -> optional PS.

Planning uses only public prompts and frozen distributions. Binding actual
responses happens afterwards for simulator replay identity. A deployment is
not a native experiment result or a claim of global optimality.
"""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--model',choices=['4B','8B','14B'],required=True)
 p.add_argument('--window',choices=['tune','test1','test2','test3','test4'],default='tune')
 p.add_argument('--forecast-mode',choices=['distribution','controlled-proxy'],default='distribution')
 p.add_argument('--forecast-summary',choices=['pooled','legacy'],default='pooled')
 p.add_argument('--allow-controlled-proxy',action='store_true');p.add_argument('--proxy-weight',type=float,default=.75)
 p.add_argument('--mode',choices=['one_scene','shortlist6','full'],default='full')
 p.add_argument('--late-region-batch-cap',type=int,choices=[0,1,2,4],default=0)
 p.add_argument('--training-ready-order',choices=['input','ready_lpt'],default='input')
 p.add_argument('--adaptive-training-batch-max',type=int,choices=[0,1,2,4,8],default=0)
 p.add_argument('--home-policy',choices=['score','service'],default='score')
 p.add_argument('--resource-candidate',action='append',help='explicit preselected offline bank entry inference@fraction:training')
 p.add_argument('--positions',choices=['pb','ps'],default='ps');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.forecast_mode=='controlled-proxy' and not a.allow_controlled_proxy:p.error('offline controlled proxy requires explicit opt-in')
 a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();stages=[]
 def run(name,argv):
  before=time.perf_counter();cmd=[sys.executable,str(Path(__file__).parent/argv[0]),*map(str,argv[1:])]
  with (a.output/(name+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  stages.append(dict(name=name,argv=cmd,returncode=r.returncode,wall_s=time.perf_counter()-before))
  (a.output/'stages.json').write_text(json.dumps(stages,indent=2))
  if r.returncode:raise RuntimeError('stage failed: '+name+'; evidence retained')
 offline=a.output/'offline';pb=a.output/'pb';ps=a.output/'ps'
 args=['run_distribution.py','--model',a.model,'--window',a.window,'--mode',a.mode,'--placement','release',
  '--home-policy',a.home_policy,'--forecast-mode',a.forecast_mode,'--forecast-summary',a.forecast_summary,'--proxy-weight',a.proxy_weight,'--output',offline]
 if a.allow_controlled_proxy:args+=['--allow-controlled-proxy']
 for candidate in a.resource_candidate or []:args+=['--resource-candidate',candidate]
 run('offline',args)
 run('bind',['materialize.py','--offline-run',offline,'--model',a.model,'--window',a.window,'--training-ready-order',a.training_ready_order,'--late-region-batch-cap',a.late_region_batch_cap,'--adaptive-training-batch-max',a.adaptive_training_batch_max,'--output',pb])
 selected=pb;position_reason='PB explicitly selected'
 if a.positions=='ps':
  run('positions',['materialize_spatial.py','--binding',pb,'--public',offline/'public.json','--model',a.model,'--output',ps])
  selected=ps;position_reason=json.loads((ps/'position-proof.json').read_text())['reason']
 files={name:str((selected/name).resolve()) for name in ('selected-plan.json','inference-layout.json','forecast.json')}
 result=dict(schema='idea1.deployment/1',status='PLANNED_REQUIRES_NATIVE_ACCEPTANCE',model=a.model,window=a.window,
  adaptive_training_batch_max=a.adaptive_training_batch_max,runtime_training_packing_unscored=bool(a.adaptive_training_batch_max),resource_candidates=a.resource_candidate,late_region_batch_cap=a.late_region_batch_cap,training_ready_order=a.training_ready_order,home_policy=a.home_policy,forecast_mode=a.forecast_mode,forecast_summary=a.forecast_summary,position_reason=position_reason,files=files,
  hashes={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in files.items()},
  phases=stages,host_wall_s=time.perf_counter()-start,
  runtime_contract=dict(inference='legal TP1/2/4/8 and whole TP expansion until EOS; no active shrink',
   mesh='observed shared-queue helper/PENDING/training selection plus bounded XY/YX routing',
   gradient='canonical full DP27 RS; no sparse contributor certificate assumed',
   publication='canonical owner AG then actor_v1 commit; cut-aware tree not enabled'),
  limits=['finite resource templates and bounded PS search, not global optimality','planning wall time is separate from simulator device time','controlled proxy is offline, not chronological actor history','adaptive training packing is an EOS-time native refinement; fixed training surrogate coverage is unchanged'])
 (a.output/'deployment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
