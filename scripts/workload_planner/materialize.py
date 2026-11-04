#!/usr/bin/env python3
"""Bind an already-selected, public-only offline plan to an execution trace.

Actual responses are used solely for post-selection replay identity, not scoring.
"""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from idea1_controller.workload import digest


def main():
 p=argparse.ArgumentParser();p.add_argument('--offline-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--late-region-batch-cap',type=int,choices=[0,1,2,4],default=0)
 p.add_argument('--training-ready-order',choices=['input','ready_lpt'],default='input')
 p.add_argument('--adaptive-training-batch-max',type=int,choices=[0,1,2,4,8],default=0)
 p.add_argument('--model',default='4B');p.add_argument('--window',default='tune');p.add_argument('--subset-prompts',type=int,default=0)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 artifact=json.loads((a.offline_run/'offline-candidates.json').read_text())
 if digest({k:v for k,v in artifact.items() if k!='artifact_digest'})!=artifact['artifact_digest']:raise ValueError('artifact changed')
 result=artifact['result'];inf=result['inference'];home=result['training']
 catalog=ROOT/f'experiments/2026-11-03-idea1-elastic/assets/inference-catalog-{a.model}.json'
 expected=hashlib.sha256(catalog.read_bytes()).hexdigest()
 if result['input_hashes'].get(str(catalog))!=expected or expected!=artifact['catalog_sha256']:
  raise ValueError('model or service catalogue differs from offline evaluation')
 forecast=json.loads((a.offline_run/'forecast.json').read_text());public=json.loads((a.offline_run/'public.json').read_text())
 if digest(public)!=artifact['public_digest']:raise ValueError('public identity mismatch')
 path=ROOT/f'experiments/2026-10-29-qwen3/inputs/Qwen3-{a.model}/{a.window}.json'
 requests=json.loads(path.read_text())['trajectories']
 if a.subset_prompts:requests=requests[:a.subset_prompts*8]
 ids={str(x['group_id']) for x in requests}
 known={str(x['group_id']):x for x in public}
 if not ids<=set(known):raise ValueError('offline plan does not support this batch; rebuild offline')
 for x in requests:
  if x['prompt_length']!=known[str(x['group_id'])]['prompt_length']:raise ValueError('public prompt changed')
 home['queues']={str(r):[g for g in gs if g in ids] for r,gs in home['queues'].items()}
 inf['execution_queues']={str(r):[g for g in gs if g in ids] for r,gs in inf['execution_queues'].items()}
 forecast={g:f for g,f in forecast.items() if g in ids}
 if 'long_quota_by_region' in home:
  home['long_quota_by_region']={str(r):sum(forecast[g]['label']=='proxy-long' for w in ws for g in home['queues'][str(w)]) for r,ws in home['region_members'].items()}
 # Retain existing online mechanisms, never infer FFN-only support from TP expansion.
 inf.update(schema='idea1.inference-partition/1',elastic=not inf['paired_tp1'],elastic_data_wait=True,pending_enabled=True,pending_data_wait=True)
 home.update(model=a.model,window=a.window,source_requests_digest=digest(requests),forecasts=forecast,
  plan_id=digest([artifact['artifact_digest'],sorted(ids)]),planning_wall_s=0,
  workload_distribution_version=artifact['distribution_version'],workload_forecast_summary=artifact.get('forecast_summary','legacy'),workload_forecast_digest=digest(forecast),
  workload_offline_artifact=artifact['artifact_digest'],planner_source='distribution-controlled-proxy' if result.get('forecast_evidence')=='offline-controlled-same-prompt-proxy' else 'distribution-offline-evaluated-candidates',
  forecast_evidence=result.get('forecast_evidence','offline-calibration-proxy'),
  planning_cost_scope='offline preparation/search and host admission measured separately; not injected as device compute',
  training_help_admission='region-ready',helper_domains=home['region_members'])
 if a.training_ready_order!='input':
  home['training_ready_order']=a.training_ready_order;home['plan_id']=digest([home['plan_id'],a.training_ready_order])
 if a.late_region_batch_cap:
  home['late_region_batch_cap_per_pair']=a.late_region_batch_cap;home['plan_id']=digest([home['plan_id'],'late-region-batch-cap',a.late_region_batch_cap])
 if a.adaptive_training_batch_max:
  home['adaptive_training_batch_max']=a.adaptive_training_batch_max
  home['plan_id']=digest([home['plan_id'],'adaptive-training-pack',a.adaptive_training_batch_max])
 (a.output/'selected-plan.json').write_text(json.dumps(home,indent=2))
 (a.output/'inference-layout.json').write_text(json.dumps(inf,indent=2))
 (a.output/'forecast.json').write_text(json.dumps(forecast))
 print(json.dumps(dict(status='BOUND_FOR_REPLAY',model=a.model,window=a.window,groups=len(ids),plan=home['plan_id'],offline_selection_fixed=True)))

if __name__=='__main__':main()
