#!/usr/bin/env python3
"""Make a PB TUNE candidate from a fixed, already admitted resource layout."""
from pathlib import Path
import argparse,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from idea1_controller.region_release_placement import balanced_release_placement
from idea1_controller.workload import digest
p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--public',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
plan=json.loads((a.binding/'selected-plan.json').read_text());inf=json.loads((a.binding/'inference-layout.json').read_text());f=json.loads((a.binding/'forecast.json').read_text());public=json.loads(a.public.read_text())
if set(str(g['group_id']) for g in public)!=set(f):raise ValueError('public batch differs from bound forecast')
pb=balanced_release_placement(public,f,plan['region_members'],{w:len(gs) for w,gs in plan['queues'].items()},inf['groups'])
plan['queues']=pb['queues'];plan['long_quota_by_region']=pb['proof']['long_quota_by_region'];plan['placement_proof']=pb['proof'];plan['source_offline_plan_id']=plan['plan_id'];plan['plan_id']=digest([plan['source_offline_plan_id'],pb]);plan['planner_source']='distribution-region-release-placement';plan['placement_stage']='PB-TUNE-candidate-not-yet-adopted';inf['execution_queues']=pb['execution_queues']
for name,value in [('selected-plan.json',plan),('inference-layout.json',inf),('forecast.json',f),('placement-proof.json',pb['proof'])]:(a.output/name).write_text(json.dumps(value,indent=2))
print(json.dumps(dict(status='PB_CANDIDATE',plan=plan['plan_id'],mean_region_dependencies=sum(len(v) for v in pb['proof']['region_dependencies'].values())/len(plan['region_members']))))
