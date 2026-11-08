import os,sys,json,time
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
from rl_xy.f2_runtime import INPUTS,make_operator
from idea1_controller.hardware_operator import make_candidate_operator
from idea1_controller.mesh_geometry import MeshGeometry
from idea1_controller.cost_catalog import ServiceCatalog
from scenario_tp4 import ScenarioRuntime
h=sys.argv[1]; start=time.perf_counter(); os.environ['RLXY_IDLE_OWNER_PAIRS']='3' if h=='H1' else '0'
p=json.load(open('inputs/profiles.json')); c={**next(x for x in p['candidates'] if x['id']==h),**p['shared']}
op,identity=make_candidate_operator(make_operator,'14B',c,cache_dir=Path('coarse-probe/cache-wide-'+h),require_active_dies_support=True)
op.rlxy_owner_tp=4; mesh=MeshGeometry(c['columns'],c['rows']);path=[mesh.die_at(y,x) for y in range(mesh.rows) for x in (range(mesh.columns) if y%2==0 else range(mesh.columns-1,-1,-1))];pairs=tuple(tuple(path[i:i+4]) for i in range(0,56,4));op.rlxy_pair_mapping=SimpleNamespace(mesh=mesh,pairs=pairs)
catdata=json.load(open('coarse-probe/'+h+'-14B-tp8-catalog.json')); catdata['table'].update(json.load(open('coarse-probe/'+h+'-14B-tp4-cap'+('25' if h=='H1' else '32')+'.json'))['table']);cat=ServiceCatalog(catdata)
layout=json.load(open('coarse-probe/'+h+'-14B-wide-layout.json')); f=json.load(open('plans/H3-14B-frozen-bound/selected-plan.json'))['forecasts'];forecast={g:SimpleNamespace(**v) for g,v in f.items()}
req=json.load(open(INPUTS/'Qwen3-14B'/'test1.json'))['trajectories']; bygroup=defaultdict(list)
for x in req:bygroup[str(x['group_id'])].append(x)
assert len(bygroup)==192 and all(len(v)==8 for v in bygroup.values())
public={g:{'prompt_length':v[0]['prompt_length']} for g,v in bygroup.items()};actual={g:[x['response_length'] for x in v] for g,v in bygroup.items()}
regions={r:(2*r,2*r+1) for r in range(7)};queues={w:[] for w in range(14)};homes={}
for key,gids in layout['execution_queues'].items():
 r=int(key); loads=[0.,0.]
 for g in gids:
  g=str(g);j=min(range(2),key=lambda z:(loads[z],z));queues[2*r+j].append(g);homes[g]=r;loads[j]+=sum(sum(x) for x in f[g]['response_scenarios'])/len(f[g]['response_scenarios'])
assert set(homes)==set(bygroup)
completed=[];ends,counts=ScenarioRuntime(cat,op,(25 if h=='H1' else 32),interval=64,helper=True).run(regions,queues,public,forecast,0,actual_lengths=actual,completion_sink=completed)
rows=[(bygroup[g][i]['trajectory_id'],t,homes[g]) for g,i,t in completed];assert len(rows)==1536 and len(set(x[0] for x in rows))==1536
result={'hardware':h,'model':'14B','generation_seconds':max(ends.values())/1e9,'groups':192,'samples':1536,'completions':rows,'replica_end_ns':ends,'counts':counts,'scope':'coarse 64-token scenario with actual EOS, pending transfer and approximate TP4-to-TP8 helper; training not yet run','wall_seconds':time.perf_counter()-start}
Path('coarse-probe/'+h+'-14B-v4-mixed.json').write_text(json.dumps(result));print({k:v for k,v in result.items() if k not in ('completions','replica_end_ns')})
