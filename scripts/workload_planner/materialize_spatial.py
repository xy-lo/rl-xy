#!/usr/bin/env python3
"""Compile a bounded PS candidate; native TUNE acceptance remains mandatory.

Only complete inference cohorts contained in a training region are supported.
Crossing layouts and PENDING queues keep PB, rather than scoring fictional help.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from search import Environment
from idea1_controller.spatial_release import Slot,optimize_positions
from idea1_controller.mesh_observation import MeshObservation
from idea1_controller.inference_groups import elastic_rectangles,inference_ranks,kv_reshard_moves,InferenceAdmission
from idea1_controller.workload import digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--public',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='4B',choices=['4B','8B','14B']);p.add_argument('--margin',type=float,default=.02)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 plan=json.loads((a.binding/'selected-plan.json').read_text());inf=json.loads((a.binding/'inference-layout.json').read_text());forecast=json.loads((a.binding/'forecast.json').read_text());public=json.loads(a.public.read_text())
 if plan['model']!=a.model or {str(g['group_id']) for g in public}!=set(forecast):raise ValueError('model/public forecast binding mismatch')
 home_digest=digest(plan['queues']);regions_digest=digest(plan['region_members']);queue_digest=digest(sorted(tuple(q) for q in inf['execution_queues'].values()))
 region_of={w:str(r) for r,ws in plan['region_members'].items() for w in ws};slots=[];reason=None
 for root,ws in inf['groups'].items():
  containing={region_of[w] for w in ws}
  if len(containing)!=1:reason='PB_RETAINED_CROSSING_COHORTS_REQUIRE_FULL_RELEASE_MODEL';break
  slots.append(Slot(str(root),next(iter(containing)),tuple(ws),2*len(ws),int(inf['caps'][str(root)])))
 if inf.get('paired_tp1'):reason='PB_RETAINED_TP1_HAS_NO_SUPPORTED_RESIZE_PROTOCOL'
 if reason:
  result=dict(execution_queues=inf['execution_queues'],logical_queue_at_slot={str(r):str(r) for r in inf['groups']},accepted_maps=0,checked_maps=0,reason=reason,scope='no unsupported position gain is assumed')
 else:
  env=Environment(a.public,a.output/'mapper-cache',model=a.model,injected_forecasts=forecast)
  mesh=MeshObservation.parse(dict(schema='rlxy.mesh-observed/2',width=6,height=9,native_ticks_per_ns=1000,policy='xy',edges=[],edge_fields=['src','dst','announced_not_started_bytes','queued_here_bytes','actual_queue_wait_ns','committed_service_ns']),0)
  legal=set(elastic_rectangles());reserve=2*(len(public)*8+max(64,max(sum(len(plan['queues'][str(w)])*8 for w in ws) for ws in plan['region_members'].values())))*(max(env.op.prompt_cap,max(g['prompt_length'] for g in public))+8192)*4
  admission=InferenceAdmission(env.op,reserve);certs={}
  def can_expand(src,helper):
   union=tuple(sorted(src.members+helper.members))
   if src.tp!=helper.tp or src.tp not in (2,4) or union not in legal:return False
   key=src.tp*2,src.cap
   if key not in certs:certs[key]=admission.evaluate(*key,public)['passed']
   return certs[key]
  def local(read,write):return env.op.vec(0,int(read),int(write),'offline-move').ns
  def network(src,dst,size):return mesh.transfer(src,dst,int(size),env.op.hw.d2d_bytes_per_ns_per_direction,env.op.hw.d2d_latency_ns)
  def migration(src,helper,state):
   union=tuple(sorted(src.members+helper.members));moves=kv_reshard_moves(inference_ranks(src.members),inference_ranks(union),state['contexts'],env.op.model);ns=0.;edges=set()
   for move in moves:
    size=move['bytes']
    if move['src']==move['dst']:ns+=local(size,size)
    else:
     ns+=2*local(size,size)+local(size,0)+local(0,size)
     value=network(move['src'],move['dst'],size);ns+=value['service_ns'];edges.update(zip(value['path'],value['path'][1:]))
   size=len(state['contexts'])*68
   for worker in helper.members:
    for dst in (2*worker,2*worker+1):
     value=network(2*int(src.key),dst,size);ns+=local(size,0)+local(0,size)+value['service_ns'];edges.update(zip(value['path'],value['path'][1:]))
   return dict(duration_ns=ns,edges=sorted(edges))
  def drain(slot,profile):
   root=int(slot.region)
   if int(slot.key)==root:return 0. # The real protocol retains a compatible local view.
   size=sum(4*(p+r) for p,r in profile['items']);ns=0.
   for lane in (0,1):ns+=local(size,0)+local(0,size)+network(2*int(slot.key)+lane,2*root+lane,size)['service_ns']
   return ns
  from fast_replica import HeapScenarioScorer
  from idea1_controller.lifecycle_memory import SystemMemory
  scorer=HeapScenarioScorer(env.op,env.policy,env.catalog);scorer.memory=SystemMemory(env.op,reserve)
  home=dict(region_members=plan['region_members'],queues=plan['queues']);public_by_id={str(g['group_id']):g for g in public}
  train={}
  for scene in range(17):
   f=forecast if scene<16 else {g:dict(x,response_scenarios=[max(x['response_scenarios'],key=lambda v:sum(v)+max(v))]*16) for g,x in forecast.items()}
   train[scene]=scorer.training_services(home,public_by_id,f,scene%16)
  def training(region,scene):return train[scene][int(region)]
  result=optimize_positions(slots,inf['execution_queues'],public_by_id,forecast,env.catalog,can_expand=can_expand,migration=migration,drain=drain,training=training,margin_fraction=a.margin)
  result['movement_model']='frozen local copy costs plus packet pipeline; shared background queues require native evaluation'
  result['capacity_certificates']={str(k):v for k,v in certs.items()}
 inf['execution_queues']=result['execution_queues'];inf['logical_queue_at_slot']=result['logical_queue_at_slot']
 if digest(plan['queues'])!=home_digest or digest(plan['region_members'])!=regions_digest or digest(sorted(tuple(q) for q in inf['execution_queues'].values()))!=queue_digest:raise AssertionError('PS changed frozen PB contract')
 proof=dict(home_queues_digest=home_digest,training_regions_digest=regions_digest,logical_queue_multiset_digest=queue_digest,accepted_maps=result['accepted_maps'],reason=result['reason'],scope='position-only candidate; same PB task/home/training order and runtime capabilities')
 plan['source_pb_plan_id']=plan['plan_id'];plan['plan_id']=digest([plan['plan_id'],inf['logical_queue_at_slot']]);plan['position_proof']=proof;plan['placement_stage']='PS-candidate-requires-native-acceptance' if result['accepted_maps'] else 'PB-retained';plan['planner_source']='distribution-spatial-region-release'
 for name,value in [('selected-plan.json',plan),('inference-layout.json',inf),('forecast.json',forecast),('position-proof.json',proof),('position-evaluation.json',result)]:(a.output/name).write_text(json.dumps(value,indent=2))
 print(json.dumps(dict(status=plan['placement_stage'],reason=result['reason'],accepted_maps=result['accepted_maps'],host_wall_s=time.perf_counter()-start)))

if __name__=='__main__':main()
