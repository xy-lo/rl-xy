"""Bounded FIFO length replay with exact affine decode-period integration.

Known response lengths determine when recorded EOS events occur. They never
change admission, queue ordering, placement, or the conservative KV reservation.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from .qwen3_cost import Cost


class Rollout:
    def __init__(self,operators,network):self.op=operators;self.net=network
    @lru_cache(maxsize=512)
    def decode_affine(self,tp,batch):
        one=self.op.forward((1,)*batch,tp,'generation',(1,)*batch)
        two=self.op.forward((1,)*batch,tp,'generation',(2,)*batch)
        a=sum((b.cost for b in one),Cost());b=sum((b.cost for b in two),Cost())
        slope=Cost((b.compute-a.compute)/batch,(b.memory-a.memory)/batch,(b.flops-a.flops)//batch,(b.read-a.read)//batch,(b.write-a.write)//batch)
        network=sum(self.net.time(tp,x.collective,x.payload) for x in one if x.collective)
        return a,slope,network
    def replica(self,requests,tp,cap,*,records=False,token_step=False):
        queue=deque(requests);active=[];time=compute=memory=network=0.;peak=0;done={};periods=[]
        kv_bytes=2*self.op.model.layers*(self.op.model.kv_heads//tp)*self.op.model.head_dim*2
        while queue or active:
            add=[]
            while queue and len(active)+len(add)<cap:add.append(queue.popleft())
            if add:
                blocks=self.op.forward(tuple(x['prompt_length'] for x in add),tp,'generation')
                c=sum((b.cost for b in blocks),Cost());n=sum(self.net.time(tp,b.collective,b.payload) for b in blocks if b.collective)
                start=time;time+=c.ns+n;compute+=c.compute;memory+=c.memory;network+=n
                peak=max(peak,(sum(x[1] for x in active)+sum(r['prompt_length']+1 for r in add))*kv_bytes)
                for r in add:
                    if r['response_length']==1:done[r['trajectory_id']]=time
                    else:active.append([r['trajectory_id'],r['prompt_length']+1,r['response_length']-1])
                peak=max(peak,sum(x[1] for x in active)*kv_bytes)
                if records:periods.append({'kind':'prefill','start_ns':start,'end_ns':time,'requests':[x['trajectory_id'] for x in add],'batch':len(add),'completed':[x['trajectory_id'] for x in add if x['response_length']==1]})
                if len(active)<cap and queue:continue
            if not active:continue
            batch=len(active);context=sum(x[1] for x in active);steps=1 if token_step else min(x[2] for x in active)
            a,b,n=self.decode_affine(tp,batch)
            # Sum s_0 + (s_0+B) + ... over an unchanged active set.
            total_context=steps*context+batch*steps*(steps-1)/2
            c=a*steps+b*(total_context-steps*batch);start=time
            time+=c.ns+n*steps;compute+=c.compute;memory+=c.memory;network+=n*steps
            for x in active:x[1]+=steps;x[2]-=steps
            peak=max(peak,sum(x[1] for x in active)*kv_bytes)
            if records:periods.append({'kind':'decode','start_ns':start,'end_ns':time,'batch':batch,'iterations':steps,'sum_context_start':context,'requests':[x[0] for x in active],'completed':[x[0] for x in active if x[2]==0]})
            for x in active:
                if x[2]==0:done[x[0]]=time
            active=[x for x in active if x[2]]
        if len(done)!=len(requests):raise ValueError('request completion coverage')
        return {'model_ns':time,'compute_ns':compute,'memory_ns':memory,'communication_service_ns':network,'actual_KV_peak_bytes':peak,'completion_ns':done,'periods':periods}
    def run(self,requests,tp,cap,*,records=False):
        dp=48//tp;replicas=[]
        for i in range(dp):replicas.append(self.replica(requests[i::dp],tp,cap,records=records))
        end=max(x['model_ns'] for x in replicas)
        return {'model_ns':end,'replicas':replicas,'replica_count':dp,'tp':tp,'cap':cap,
                'barrier_wait_ns_by_replica':[end-x['model_ns'] for x in replicas],
                'mean_replica_barrier_wait_fraction':sum(end-x['model_ns'] for x in replicas)/(dp*end),
                'request_count':sum(len(x['completion_ns']) for x in replicas)}
