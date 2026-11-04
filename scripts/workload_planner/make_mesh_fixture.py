#!/usr/bin/env python3
"""Adversarial 4B protocol fixture; never a deployment/performance candidate."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from idea1_controller.workload import digest
from search import templates
h=Path.home();o=h/'outputs/idea1-final-t162';dest=o/'mesh-functional-binding-01';dest.mkdir(exist_ok=False)
old=o/'m1-native-tune-full-01/binding';plan=json.loads((old/'selected-plan.json').read_text());f=json.loads((old/'forecast.json').read_text())
requests=json.loads((ROOT/'experiments/2026-10-29-qwen3/inputs/Qwen3-4B/tune.json').read_text())['trajectories'][:24]
ids=list(dict.fromkeys(str(x['group_id']) for x in requests));f={g:f[g] for g in ids}
regions=templates()['box4'];queues={str(w):ids if w==0 else [] for w in range(27)}
inf=dict(schema='idea1.inference-partition/1',groups={str(w):[w] for w in range(27)},caps={str(w):16 for w in range(27)},execution_queues=queues,paired_tp1=False,elastic=True,elastic_data_wait=True,pending_enabled=True,pending_data_wait=True)
plan.update(queues=queues,region_members=regions,helper_domains=regions,forecasts=f,source_requests_digest=digest(requests),workload_forecast_digest=digest(f),
 long_quota_by_region={str(r):sum(f[g]['label']=='proxy-long' for w in ws for g in queues[str(w)]) for r,ws in regions.items()},
 planner_source='mesh-adversarial-functional-fixture',fixture_scope='24 original TUNE samples in one busy TP2; complete training TP8 region gives three legal idle peers; no claimed placement gain',plan_id=digest(['mesh-fixture',ids]))
for name,value in [('selected-plan.json',plan),('inference-layout.json',inf),('forecast.json',f)]:(dest/name).write_text(json.dumps(value,indent=2))
print(json.dumps(dict(status='FUNCTIONAL_FIXTURE',groups=len(ids),samples=len(requests),scope=plan['fixture_scope'])))
