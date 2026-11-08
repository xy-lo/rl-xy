import sys,json,time,os
from pathlib import Path
from collections import defaultdict
from coarse_training import simulate_training
from rl_xy.f2_runtime import INPUTS,make_operator
from idea1_controller.hardware_operator import make_candidate_operator
from idea1_controller.lifecycle_memory import LifecycleMemory
from rl_xy.qwen3_experiment import Policy,optimizer_plan,simple_plan_time
from rl_xy.qwen3_native import NetworkProfiles
from rl_xy.model import enumerate_parameters
from idea1_controller.cost_catalog import ServiceCatalog
h=sys.argv[1];model=sys.argv[2];start=time.perf_counter();b={('4B','H0'):'plans/H0-4B-test1-bound',('4B','H1'):'plans/H1-4B-test1-bound',('4B','H2'):'plans/H2-4B-parallel-bound',('4B','H3'):'plans/H3-4B-parallel-bound',('8B','H0'):'plans/H0-8B-parallel-bound',('8B','H1'):'plans/H1-8B-staged-parallel-bound',('8B','H2'):'plans/H2-8B-parallel-bound',('8B','H3'):'plans/H3-8B-parallel-bound',('14B','H0'):'plans/H0-14B-staged-parallel-bound',('14B','H3'):'plans/H3-14B-frozen-bound'}[(model,h)]
if (model,h) in [('8B','H1'),('14B','H0')]:os.environ['RLXY_REFERENCE_STAGING']='1'
if (model,h)==('8B','H1'):os.environ['RLXY_IDLE_OWNER_PAIRS']='1';os.environ['RLXY_MIN_TRAINING_TP']='4'
p=json.load(open(b+'/selected-plan.json'));layout=json.load(open(b+'/inference-layout.json'));gen=json.load(open('coarse-probe/'+h+'-'+model+'-v3-mixed.json'));req=json.load(open(INPUTS/('Qwen3-'+model)/'test1.json'))['trajectories'];c=ServiceCatalog(json.load(open(('catalogs/H1-4B.json' if h=='H1' else 'catalogs/'+h+'/4B.json') if model=='4B' else 'catalogs/'+h+'/'+model+('-staged' if (model,h) in [('8B','H1'),('14B','H0')] else '')+'.json')))
regions={int(k):v for k,v in p['region_members'].items()};home={w:r for r,ws in regions.items() for w in ws};group_home={str(g):home[int(w)] for w,gs in p['queues'].items() for g in gs};byid={z['trajectory_id']:z for z in req};bygroup=defaultdict(list);times=defaultdict(list)
for z in req:bygroup[str(z['group_id'])].append(z)
for tid,t,r in gen['completions']:times[str(byid[tid]['group_id'])].append(t)
assert len(times)==192 and all(len(x)==8 for x in times.values())
region_ready={r:float(gen['replica_end_ns'][str(r)]) for r in regions}
profile=c.data['hardware_identity']['profile'];w=profile['columns'];pairs=[]
from idea1_controller.mesh_geometry import MeshGeometry,PairMapping
m=PairMapping.with_idle_tail_pairs(MeshGeometry(w,profile['rows']),int(os.environ.get('RLXY_IDLE_OWNER_PAIRS','0')))
coord={r:m.pairs[ws[0]][0] for r,ws in regions.items()}
def route(a,b):
 cur=coord[a];dst=coord[b];edges=[]
 while cur%w!=dst%w:
  nx=cur+(1 if dst%w>cur%w else -1);edges.append((cur,nx));cur=nx
 while cur//w!=dst//w:
  nx=cur+(w if dst//w>cur//w else -w);edges.append((cur,nx));cur=nx
 return edges
maximum=max(sum(len(p['queues'][str(q)])*8 for q in ws) for ws in regions.values());maxprompt=max(z['prompt_length'] for z in req);reserve=2*(1536+max(64,maximum))*(maxprompt+8192)*4
op,_=make_candidate_operator(make_operator,model,profile,cache_dir=Path('coarse-probe/cache-'+h),require_active_dies_support=True);mem=LifecycleMemory(op,reserve)
net=NetworkProfiles(op,'unused',native=False);params=sum(v.unique_elements for v in enumerate_parameters(op.case_tp(2)))
state=op.state(2,2);prefix=net.time(op.active_dies//2,'ALL_GATHER',state['persistent_reference_shard_bytes']) if 'persistent_reference_shard_bytes' in state else 0
region_ready={r:t+prefix for r,t in region_ready.items()}
for group in groups if False else []:pass
def service(g,r):
 items=bygroup[g['id']];tp=2*len(regions[r]);total=0
 for stage,maxbatch,factor in [('teacher',min(8,{'4B':8,'8B':1,'14B':1}[model]*(tp//2)),2),('training',({'4B':4,'8B':2,'14B':1}[model] if tp==2 else min(8,{'4B':4,'8B':2,'14B':1}[model]*tp)),1)]:
  for batch in mem.batches(items,stage,tp,maxbatch):
   lengths=[z['prompt_length']+z['response_length']-1 for z in batch];total+=factor*c.phase(stage,tp,lengths)
 if g['owner']!=r:
  hops=len(route(g['owner'],r));total+=params*4/(op.hw.d2d_bytes_per_ns_per_direction)+2*params*4/(tp*op.hw.dram_bytes_per_ns)+hops*op.hw.d2d_latency_ns
 return total
groups=[dict(id=g,owner=group_home[g],ready_ns=max(ts)+prefix,bytes=sum((z['prompt_length']+z['response_length'])*4 for z in bygroup[g])) for g,ts in times.items()];neighbors={r:tuple(q for q in regions if q!=r and len(regions[q])==len(regions[r])) for r in regions}
out=simulate_training(groups,region_ready,neighbors,service,route,profile['d2d_TB_s_per_edge_direction']*1000);region_end={r:max([region_ready[r]]+[d['finish_ns'] for d in out['decisions'] if d['region']==r]) for r in regions}
restore={}
for r,ws in regions.items():
 tp=2*len(ws);hops=max([len(route(r,q)) for q in regions if q!=r] or [0]);restore[r]=0 if tp==2 else params*4*(1-2/tp)/op.hw.d2d_bytes_per_ns_per_direction+params*16/op.hw.dram_bytes_per_ns+hops*op.hw.d2d_latency_ns
policy={'4B':Policy(2,2,4,8,32),'8B':Policy(2,2,2,1,16),'14B':Policy(2,2,1,1,16)}[model]
optimizer=simple_plan_time(optimizer_plan(op,policy),net);complete=max(region_end[r]+restore[r] for r in regions)+optimizer
out.update(complete_ns=complete,optimizer_ns=optimizer,prefix_ns=prefix,restore_ns=restore,hardware=h,model=model,groups=len(groups),samples=1536,wall_seconds=time.perf_counter()-start,scope='diagnostic: coarse regional replay with analytic restore, helper gradient transfer and optimizer; fixed initial generation layout; not native execution');Path('coarse-probe/'+h+'-'+model+'-v3-training.json').write_text(json.dumps(out));print({k:v for k,v in out.items() if k!='decisions'})
