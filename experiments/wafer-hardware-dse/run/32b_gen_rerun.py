import json,sys,time,heapq
from collections import deque
from pathlib import Path
from idea1_controller.cost_catalog import ServiceCatalog
h=sys.argv[1]; model='32B'; start=time.perf_counter(); n=3
inp=json.load(open('coarse-probe/32b-input.json'))['groups']; data=[]
for i,(prompt,pred,lengths) in enumerate(inp):
    for j,length in enumerate(lengths): data.append({'group_id':str(i),'trajectory_id':str(i)+'-'+str(j),'prompt_length':prompt,'response_length':length})
load=[0.0]*n; queues={str(i):[] for i in range(n)}
for i in sorted(range(len(inp)),key=lambda k:(-inp[k][1],k)):
    r=min(range(n),key=lambda z:(load[z],z)); queues[str(r)].append(str(i));load[r]+=inp[i][1]
layout={'groups':{str(i):list(range(8*i,8*i+8)) for i in range(n)},'execution_queues':queues,'caps':{str(i):16 for i in range(n)},'paired_tp1':False}
cat=ServiceCatalog(json.load(open('coarse-probe/'+h+'-32B-tp16-catalog.json')))
Path('coarse-probe/'+h+'-32B-rerun-layout.json').write_text(json.dumps(layout))
by_group={}
for row in data:by_group.setdefault(str(row['group_id']),[]).append(row)
events=[];states={};seq=0;done=[]
def issue(r,now):
 global seq
 s=states[r];q,a,tp,cap=s['q'],s['active'],s['tp'],s['cap']
 if q and len(a)<cap:
  batch=[q.popleft() for _ in range(min(cap-len(a),len(q)))];dt=cat.phase('prefill',tp,[z['prompt_length'] for z in batch]);kind='prefill';val=batch
 elif a:
  n=len(a);mean=sum(z[1] for z in a)/n;val=min(64,min(z[2] for z in a));d=cat.decode(tp,n,mean);slope=cat.decode(tp,n,mean+1)-d;dt=val*d+val*(val-1)*slope/2;kind='decode'
 else:s['end']=now;return
 seq+=1;heapq.heappush(events,(now+dt,seq,r,kind,val))
for r,workers in layout['groups'].items():
 tp=1 if layout['paired_tp1'] else 2*len(workers);gids=layout['execution_queues'][r];q=deque(z for gid in gids for z in by_group[str(gid)]);states[r]={'q':q,'active':[],'tp':tp,'cap':layout['caps'][r],'end':0.0}
for r in sorted(states):issue(r,0)
while events:
 now,_,r,kind,val=heapq.heappop(events);s=states[r]
 if kind=='prefill':
  for row in val:
   if row['response_length']==1:done.append((row['trajectory_id'],now,r))
   else:s['active'].append((row,row['prompt_length']+1,row['response_length']-1))
 else:
  active=[]
  for row,ctx,left in s['active']:
   if left==val:done.append((row['trajectory_id'],now,r))
   else:active.append((row,ctx+val,left-val))
  s['active']=active
 issue(r,now)
assert sorted(z[0] for z in done)==sorted(z['trajectory_id'] for z in data)
result={'hardware':h,'model':model,'generation_seconds':max(s['end'] for s in states.values())/1e9,'groups':192,'samples':len(done),'wall_seconds':time.perf_counter()-start,'scope':'fixed initial mixed-TP layout, actual EOS; no migration or training; diagnostic only','completions':done,'replica_end_ns':{r:s['end'] for r,s in states.items()}}
Path('coarse-probe/'+h+'-'+model+'-rerun-mixed.json').write_text(json.dumps(result));print({k:v for k,v in result.items() if k not in ('completions','replica_end_ns')})
