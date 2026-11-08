"""Fast scenario replay of normal admission, packed PENDING and full-layer help.

The workload owns sampled remaining lengths for EOS. Policy tests below use
queue occupancy, completed history and elapsed contexts only. This search model
uses alpha-beta transport and calibrated operator services; native execution
resolves real mesh contention and enforces the final experiment contract.
"""
from collections import deque
from bisect import bisect_right
import heapq,math

class ScenarioRuntime:
    def __init__(self,catalog,op,cap,interval=64,helper=True):
        self.cat=catalog;self.op=op;self.cap=cap;self.interval=interval;self.helper=helper
    def worker_hops(self,a,b):
        mapping=getattr(self.op,'rlxy_pair_mapping',None)
        if mapping is None:
            return max(abs((2*a)//6-(2*b)//6)+abs((2*a)%6-(2*b)%6),1)
        return max(mapping.mesh.distance(mapping.pairs[a][0],mapping.pairs[b][0]),1)
    def run(self,regions,queues,public,forecast,scenario,pressure=False):
        pools={g:tuple(sorted(v for vec in f.response_scenarios for v in vec)) for g,f in forecast.items()}
        owner={w:r for r,ws in regions.items() for w in ws};rep={};events=[];idle=set();paired={};as_helper={};history=[];serial=0;done=0;needed=sum(len(q)*8 for q in queues.values());tlast={};counts={'pending':0,'helper':0}
        for w,gids in queues.items():
            q=deque()
            for g in gids:
                vec=(8192,)*8 if pressure else forecast[g].response_scenarios[scenario%16]
                for index,length in enumerate(vec):q.append((g,public[g]['prompt_length'],int(length),index))
            rep[w]=dict(queue=q,active=[],pending=None)
        def schedule(w,now,duration,pending):
            nonlocal serial
            serial+=1;rep[w]['pending']=pending;heapq.heappush(events,(now+max(1.,duration),serial,w))
        def decode_service(w):
            a=rep[w]['active'];n=len(a);mean=sum(x[1] for x in a)/n
            tp=4 if w in paired else 2
            value=self.cat.decode(tp,n,mean);delta=self.cat.decode(tp,n,mean+1)-value
            if tp==4:
                # Charge source head/activation/KV extra; the service fit is a
                # surrogate, never zero-cost communication or free widening.
                src,dst=w,paired[w];hops=self.worker_hops(src,dst)
                value+=self.op.model.layers*(n*self.op.model.hidden_size*2/self.op.hw.d2d_bytes_per_ns_per_direction*hops+self.op.hw.d2d_latency_ns)
            return value,delta
        def help_gain(w,j):
            a=rep[w]['active'];batch=len(a);mean=sum(x[1] for x in a)/batch;residual=[]
            for item,ctx,_ in a:
                g,prompt,length,index=item;elapsed=ctx-prompt
                # Frozen historical pool, not this scenario's future endpoint.
                pool=pools[g]
                i=bisect_right(pool,elapsed);n=len(pool)-i
                residual.append(max(1,pool[i+(9*(n-1))//10]-elapsed) if n>=32 else max(64,min(2048,elapsed//4)))
            horizon=max(residual);old=self.cat.decode(2,batch,mean);new=self.cat.decode(4,batch,mean)
            hops=self.worker_hops(w,j);bw=self.op.hw.d2d_bytes_per_ns_per_direction
            m=self.op.model
            size=max(1,int(mean))*batch*m.layers*2*(m.kv_heads//4)*m.head_dim*m.actor_dtype_bytes
            migration=size/bw*hops+4*size/self.op.hw.dram_bytes_per_ns
            extra=self.op.model.layers*(batch*self.op.model.hidden_size*2/bw*hops+self.op.hw.d2d_latency_ns)
            gain=horizon*(old-new-extra)-migration
            return gain,migration
        def issue(w,now):
            nonlocal serial
            r=rep[w]
            if w in as_helper:return
            q,a=r['queue'],r['active']
            # Own prefill has priority over donation. Full normal-cap overflow
            # can move atomically to an otherwise idle same-region instance.
            free=sorted(j for j in idle if j!=w and owner[j]==owner[w] and j not in as_helper)
            if q and len(a)>=self.cap and free:
                dst=free[0];items=[q.popleft() for _ in range(min(self.cap,len(q)))];idle.remove(dst)
                payload=sum(x[1]*4+64 for x in items);hops=self.worker_hops(w,dst)
                duration=payload/self.op.hw.d2d_bytes_per_ns_per_direction*hops+2*payload/self.op.hw.dram_bytes_per_ns+2*self.op.hw.d2d_latency_ns*hops
                schedule(w,now,duration,('transfer',(dst,items)));counts['pending']+=1;return
            if q and len(a)<self.cap:
                add=[q.popleft() for _ in range(min(self.cap-len(a),len(q)))];idle.discard(w)
                schedule(w,now,self.cat.phase('prefill',2,[x[1] for x in add]),('prefill',add));return
            if a:
                idle.discard(w)
                if self.helper and not q and w not in paired and free:
                    choices=[(help_gain(w,j),j) for j in free];(gain,migration),j=max(choices,key=lambda x:(x[0][0],-x[1]))
                    if gain>max(1000,migration*.05):
                        paired[w]=j;as_helper[j]=w;idle.discard(j);counts['helper']+=1
                        schedule(w,now,migration,('acquire',None));return
                steps=min(self.interval,min(x[2] for x in a));value,delta=decode_service(w)
                schedule(w,now,steps*value+steps*(steps-1)/2*delta,('decode',steps));return
            if w in paired:
                j=paired.pop(w);as_helper.pop(j);issue(j,now)
            idle.add(w);tlast[w]=now;r['pending']=None
        for w in sorted(rep):issue(w,0.)
        while events:
            now,_,w=heapq.heappop(events);r=rep[w];kind,data=r['pending'];r['pending']=None
            if kind=='transfer':
                dst,items=data;rep[dst]['queue'].extend(items);issue(dst,now);issue(w,now);continue
            if kind=='prefill':
                for item in data:
                    if item[2]==1:done+=1;history.append(1)
                    else:r['active'].append([item,item[1]+1,item[2]-1])
            elif kind=='decode':
                kept=[]
                for item,ctx,remaining in r['active']:
                    if remaining==data:done+=1;history.append(item[2])
                    else:kept.append([item,ctx+data,remaining-data])
                r['active']=kept
            elif kind!='acquire':raise ValueError('unknown search event')
            issue(w,now)
        if done!=needed or paired or as_helper or any(r['queue'] or r['active'] for r in rep.values()):raise ValueError('search scenario did not conserve work')
        return {r:max(tlast[w] for w in ws) for r,ws in regions.items()},counts
