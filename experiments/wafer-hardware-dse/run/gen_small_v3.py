import os,sys,json,time
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
from rl_xy.f2_runtime import INPUTS,make_operator
from idea1_controller.hardware_operator import make_candidate_operator
from idea1_controller.cost_catalog import ServiceCatalog
from scenario_tp4 import ScenarioRuntime
h,model=sys.argv[1:];start=time.perf_counter()
if (h,model)==('H1','8B'):
 os.environ.update(RLXY_REFERENCE_STAGING='1',RLXY_IDLE_OWNER_PAIRS='1',RLXY_MIN_TRAINING_TP='4')
paths={('H1','4B'):'plans/H1-4B-test1-bound',('H1','8B'):'plans/H1-8B-staged-parallel-bound',('H2','4B'):'plans/H2-4B-parallel-bound',('H2','8B'):'plans/H2-8B-parallel-bound',('H3','4B'):'plans/H3-4B-parallel-bound',('H3','8B'):'plans/H3-8B-parallel-bound'}
plan=json.load(open(paths[h,model]+'/selected-plan.json')); p=json.load(open('inputs/profiles.json'));profile={**next(x for x in p['candidates'] if x['id']==h),**p['shared']}
op,_=make_candidate_operator(make_operator,model,profile,cache_dir=Path('coarse-probe/cache-'+h),require_active_dies_support=True);op.rlxy_owner_tp=2
catpath=('catalogs/H1-4B.json' if (h,model)==('H1','4B') else 'catalogs/'+h+'/'+model+('-staged' if (h,model)==('H1','8B') else '')+'.json'); data=json.load(open(catpath));cat=ServiceCatalog(data);cap=min(64,data['inference_capacity_frontier']['2'],data['inference_capacity_frontier']['4']);assert cap>0
req=json.load(open(INPUTS/('Qwen3-'+model)/'test1.json'))['trajectories'];bygroup=defaultdict(list)
for x in req:bygroup[str(x['group_id'])].append(x)
assert len(bygroup)==192 and all(len(v)==8 for v in bygroup.values())
public={g:{'prompt_length':v[0]['prompt_length']} for g,v in bygroup.items()};actual={g:[x['response_length'] for x in v] for g,v in bygroup.items()};forecast={g:SimpleNamespace(**v) for g,v in plan['forecasts'].items()}
regions={int(r):tuple(ws) for r,ws in plan['region_members'].items()};queues={int(w):[str(g) for g in gs] for w,gs in plan['queues'].items()};home={w:r for r,ws in regions.items() for w in ws};ghome={g:home[w] for w,gs in queues.items() for g in gs};assert set(ghome)==set(bygroup)
completed=[];ends,counts=ScenarioRuntime(cat,op,cap,interval=64,helper=True).run(regions,queues,public,forecast,0,actual_lengths=actual,completion_sink=completed)
rows=[(bygroup[g][i]['trajectory_id'],t,ghome[g]) for g,i,t in completed];assert len(rows)==1536 and len(set(x[0] for x in rows))==1536
result=dict(hardware=h,model=model,cap=cap,generation_seconds=max(ends.values())/1e9,groups=192,samples=1536,completions=rows,replica_end_ns=ends,counts=counts,scope='coarse TP2-to-TP4 replay using observed EOS and historical forecasts; not native',wall_seconds=time.perf_counter()-start)
Path('coarse-probe/'+h+'-'+model+'-v3-mixed.json').write_text(json.dumps(result));print({k:v for k,v in result.items() if k not in ('completions','replica_end_ns')})
