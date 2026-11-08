from collections import deque
import heapq
def simulate_training(groups,regions,neighbors,service_ns,route,bandwidth_bytes_ns,hop_latency_ns=100):
 events=[];serial=0;ready={r:deque() for r in regions};idle=set();edges={};decisions=[]
 def push(t,k,x):
  nonlocal serial
  serial+=1;heapq.heappush(events,(t,serial,k,x))
 assert len({g['id'] for g in groups})==len(groups)
 for g in groups:push(g['ready_ns'],'group',g)
 for r,t in regions.items():push(t,'idle',r)
 while events:
  now=events[0][0]
  while events and events[0][0]==now:
   _,_,k,x=heapq.heappop(events)
   if k=='group':ready[x['owner']].append(x)
   else:idle.add(x)
  while idle:
   chosen=None
   for r in sorted(idle):
    if ready[r]:chosen=(r,r,ready[r].popleft());break
   if chosen is None:
    for r in sorted(idle):
     options=[o for o in sorted(neighbors.get(r,())) if ready[o]]
     if options:
      o=min(options,key=lambda x:(ready[x][0]['ready_ns'],x));chosen=(r,o,ready[o].popleft());break
   if chosen is None:break
   r,o,g=chosen;idle.remove(r);arrival=now;path=tuple(route(o,r)) if o!=r else ()
   if o!=r and not path:raise ValueError('missing route')
   for e in path:
    start=max(arrival,edges.get(e,0));end=start+g['bytes']/bandwidth_bytes_ns;edges[e]=end;arrival=end+hop_latency_ns
   duration=service_ns(g,r);assert duration>0;finish=arrival+duration
   decisions.append(dict(group=g['id'],owner=o,region=r,dispatch_ns=now,arrival_ns=arrival,finish_ns=finish,helper=o!=r));push(finish,'idle',r)
 assert len(decisions)==len(groups) and not any(ready.values())
 return dict(finish_ns=max((d['finish_ns'] for d in decisions),default=max(regions.values())),decisions=decisions,helper_groups=sum(d['helper'] for d in decisions))
