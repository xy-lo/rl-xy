import os,sys,json,time
from pathlib import Path
from collections import defaultdict
from coarse_training import simulate_training
from rl_xy.f2_runtime import INPUTS,make_operator
from rl_xy.model import enumerate_parameters
from idea1_controller.hardware_operator import make_candidate_operator
from idea1_controller.wide_owner_memory import WideOwnerMemory
from idea1_controller.cost_catalog import ServiceCatalog
from rl_xy.qwen3_experiment import Policy,optimizer_plan,simple_plan_time
from rl_xy.qwen3_native import NetworkProfiles

h=sys.argv[1]; start=time.perf_counter()
os.environ['RLXY_IDLE_OWNER_PAIRS']='3' if h in ('H0','H1') else '0'
profile=json.load(open('inputs/profiles.json')); profile={**next(x for x in profile['candidates'] if x['id']==h),**profile['shared']}
op,ident=make_candidate_operator(make_operator,'14B',profile,cache_dir=Path('coarse-probe/cache-wide-'+h),require_active_dies_support=True)
assert op.active_dies==(56 if h in ('H1','H2') else 48)
mem=WideOwnerMemory(op,4); net=NetworkProfiles(op,'unused',native=False)
layout=json.load(open('coarse-probe/'+h+'-14B-wide-layout.json')); gen=json.load(open('coarse-probe/'+h+'-14B-v4-mixed.json'))
req=json.load(open(INPUTS/'Qwen3-14B'/'test1.json'))['trajectories']; byid={z['trajectory_id']:z for z in req}; bygroup=defaultdict(list); ends=defaultdict(list)
for z in req: bygroup[str(z['group_id'])].append(z)
for tid,t,r in gen['completions']: ends[str(byid[tid]['group_id'])].append(t)
n=len(layout['groups']); owners={str(g):int(r) for r,gs in layout['execution_queues'].items() for g in gs}
assert len(bygroup)==len(ends)==len(owners)==192 and all(len(x)==8 for x in ends.values())
catpath='coarse-probe/'+h+'-14B-tp8-catalog.json' if h in ('H1','H2') else ('catalogs/H0/14B-staged.json' if h=='H0' else 'catalogs/H3/14B.json')
cat=ServiceCatalog(json.load(open(catpath))); params=sum(x.unique_elements for x in enumerate_parameters(op.case_tp(4)))
regions={int(r):t for r,t in gen['replica_end_ns'].items()}; groups=[{'id':g,'owner':owners[g],'ready_ns':max(ts),'bytes':sum((z['prompt_length']+z['response_length'])*4 for z in bygroup[g])} for g,ts in ends.items()]
cols=profile['columns']; rows=profile['rows']; path=[y*cols+x for y in range(rows) for x in (range(cols) if y%2==0 else range(cols-1,-1,-1))]; roots=[path[8*i] for i in range(n)]
def route(a,b):
    y,x=divmod(roots[a],cols); ty,tx=divmod(roots[b],cols); out=[]
    while x!=tx:
        nx=x+(1 if tx>x else -1); out.append(((y,x),(y,nx))); x=nx
    while y!=ty:
        ny=y+(1 if ty>y else -1); out.append(((y,x),(ny,x))); y=ny
    return out
def service(g,r):
    xs=bygroup[g['id']]; total=0.0
    for stage,maximum,mult in [('teacher',8,2),('training',8,1)]:
        for batch in mem.batches(xs,stage,8,maximum):
            lengths=[z['prompt_length']+z['response_length']-1 for z in batch]; total+=mult*cat.phase(stage,8,lengths)
    if r!=g['owner']: total+=params*4/8/op.hw.d2d_bytes_per_ns_per_direction+len(route(g['owner'],r))*op.hw.d2d_latency_ns
    return total
neighbors={r:tuple(q for q in regions if q!=r) for r in regions}
out=simulate_training(groups,regions,neighbors,service,route,op.hw.d2d_bytes_per_ns_per_direction)
restore=params*4*(1-4/8)/op.hw.d2d_bytes_per_ns_per_direction+params*16/op.hw.dram_bytes_per_ns
opt=simple_plan_time(optimizer_plan(op,Policy(4,8,1,1,32)),net)
out.update(hardware=h,model='14B',complete_ns=out['finish_ns']+restore+opt,restore_ns=restore,optimizer_ns=opt,scope='TP4 owners TP8 regions; coarse dynamic generation and regional helper; simulated costs, not native or hardware calibrated',groups=192,samples=1536,wall_seconds=time.perf_counter()-start)
Path('coarse-probe/'+h+'-14B-v4-training.json').write_text(json.dumps(out))
print({k:v for k,v in out.items() if k!='decisions'})
