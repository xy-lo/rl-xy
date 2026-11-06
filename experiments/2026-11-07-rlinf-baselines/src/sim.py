"""Conditional RollPacker replay. EOS lengths are private simulator outcomes.

Policy sees request load, observed prefixes, completion/reward callbacks, and
declared capacity only. This is not numerical learning or an H800 calibration.
"""
from __future__ import annotations
import argparse, hashlib, heapq, json, math, time
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rl_xy.qwen3_cost import Operators, Cost
from rl_xy.qwen3_native import NetworkProfiles
from rl_xy.qwen3_experiment import Policy, weight_plan, optimizer_plan
from rl_xy.model import enumerate_parameters

NS=1e9

class Backend:
    def __init__(self, root, model, policy, host_bw=25e9):
        self.op=Operators(root,Path(root)/f'experiments/2026-10-29-qwen3/inputs/Qwen3-{model}/config.json','gpu')
        self.net=NetworkProfiles(self.op,'unused',native=False)
        self.p=policy; self.world=48; self.host_bw=host_bw
        self.params=self.op.info['unique_parameter_elements']
        self.kv=2*self.op.model.layers*(self.op.model.kv_heads//policy.rollout_tp)*self.op.model.head_dim*2
        self.state=self.op.state(policy.train_tp,policy.rollout_tp)
        self.capacity=self.op.capacity
        self.max_param=max(r.local_elements_per_die for r in enumerate_parameters(self.op.case_tp(policy.train_tp)))
        self.history_response_bound=8192

    def blocks(self, blocks, tp): return self.net.blocks(tp,blocks)/NS

    @lru_cache(maxsize=8192)
    def prefill(self, lengths):
        return self.blocks(self.op.forward(lengths,self.p.rollout_tp,'generation'),self.p.rollout_tp)

    @lru_cache(maxsize=256)
    def affine(self,b):
        tp=self.p.rollout_tp
        one=self.blocks(self.op.forward((1,)*b,tp,'generation',(1,)*b),tp)
        two=self.blocks(self.op.forward((1,)*b,tp,'generation',(2,)*b),tp)
        assert two>=one
        return one,(two-one)/b

    @lru_cache(maxsize=16384)
    def work(self, lengths, response_lengths):
        return self.work_schedule(lengths,response_lengths)[0]

    def memory(self,lengths,training):
        m=self.op.model; tp=self.p.train_tp; tokens=sum(lengths)
        base=self.state['training_base_max']; mx=self.max_param
        staging=max(4*mx,4*tokens*m.hidden_size)
        if not training:
            return base+tokens*(m.hidden_size*4+(m.vocab_size//tp)*8)+staging
        checkpoint=tokens*m.hidden_size*2*m.layers
        layer=tokens*(m.hidden_size*12+2*m.ffn_size//tp*6+(m.query_width+2*m.kv_heads*m.head_dim)//tp*6)
        head=tokens*(m.vocab_size//tp)*8
        return base+checkpoint+max(layer,head)+mx*8+staging

    def fit_chunks(self,lengths,limit,training):
        chunks=[]; i=0
        while i<len(lengths):
            n=min(limit,len(lengths)-i)
            while self.memory(lengths[i:i+n],training)>self.capacity:
                if n==1: raise ValueError('single observed sample exceeds phase capacity')
                n=max(1,n//2)
            chunks.append((i,i+n)); i+=n
        return chunks

    @lru_cache(maxsize=16384)
    def work_schedule(self,lengths,response_lengths):
        p=self.p; total=0.
        logs=self.fit_chunks(lengths,p.logprob_mb,False); trains=self.fit_chunks(lengths,p.train_mb,True)
        for i,j in logs: total+=getattr(self,'teacher_forwards',2)*self.blocks(self.op.forward(lengths[i:j],p.train_tp),p.train_tp)
        for i,j in trains: total+=self.blocks(self.op.training(lengths[i:j],p.train_tp,sum(response_lengths[i:j])),p.train_tp)
        return total,logs,trains

    def fixed_plan(self, plan):
        """Conservative plan replay with node-shared interconnect reservations.

        Collective participants and per-rank ordering come from frozen rl-xy.
        All nodes used by a collective share one serialized uplink per node.
        """
        ready=[0.]*self.world; links=[0.]*(self.world//8)
        prof=self.op.profile
        for o in plan.operations:
            ranks=o.participant_ranks; nodes=sorted(set(r//8 for r in ranks))
            start=max(ready[r] for r in ranks)
            if o.kind=='compute': duration=o.duration_ns/NS
            else:
                start=max(start,max(links[n] for n in nodes))
                if len(nodes)==1:
                    duration=self.net.time(len(ranks),o.collective,o.payload_bytes)/NS if o.kind=='collective' else (prof['nvlink_latency_ns']+o.payload_bytes/prof['nvlink_bytes_per_ns_per_direction'])/NS
                else:
                    n=len(nodes); factor=2*(n-1)/n if o.collective=='ALL_REDUCE' else (n-1)/n
                    volume=o.payload_bytes*factor
                    duration=(2*(n-1)*prof['uplink_latency_ns']+volume/prof['node_uplink_bytes_per_ns_per_direction'])/NS
                    duration+=self.net.time(min(8,len(ranks)),o.collective,o.payload_bytes)/NS if o.kind=='collective' else 0
                for node in nodes: links[node]=start+duration
            for rank in ranks: ready[rank]=start+duration
        return max(ready)

    @lru_cache(maxsize=1)
    def publish(self): return self.fixed_plan(weight_plan(self.op,self.p))
    @lru_cache(maxsize=1)
    def optimize(self): return self.fixed_plan(optimizer_plan(self.op,self.p))

    def host_gradient(self, participating_dp):
        # FP32 TP-local gradient buffers, one copy per streaming DP replica.
        # A node shares its host link across all 8 GPU endpoints.
        perrank=self.params*4/self.p.train_tp
        ranks=participating_dp*self.p.train_tp
        return min(8,ranks)*perrank/self.host_bw

@dataclass
class Req:
    rid: str
    gid: str
    sample: int
    prompt: int
    length: int             # private outcome, accessed by execution engine only
    version: int
    generated: int=0
    compute_tokens: int=0
    status: str='queued'
    cancel_at: float|None=None
    finish_at: float|None=None
    reward_at: float|None=None
    accepted: bool=False
    migrations: int=0
    replica: int=-1

@dataclass
class Period:
    start: float
    end: float
    kind: str
    ids: tuple
    steps: int
    a: float=0.
    b: float=0.
    add: tuple=()
    epoch: int=0
    slowdown: float=1.

class Replica:
    def __init__(self,sim,index):
        self.s=sim; self.i=index; self.queue=deque(); self.active=[]
        self.period=None; self.epoch=0; self.halted=False; self.halt_requested=False
        self.busy=0.; self.peak_kv=0; self.segments=[];self.preemptions=0

    def kv_bytes(self,rs,extra=0):
        if self.s.kv_budget(self) is None:return sum(r.prompt+r.generated+extra for r in rs)*self.s.be.kv
        block=16
        return sum(math.ceil((r.prompt+r.generated+extra)/block)*block for r in rs)*self.s.be.kv

    def start(self,t):
        if self.period or self.halted or not self.s.actor_can_start(self,t): return
        self.queue=deque(r for r in self.queue if r.cancel_at is None)
        add=[]
        while self.queue and len(self.active)+len(add)<self.s.cap:
            budget=self.s.kv_budget(self)
            candidate=self.queue[0]
            if budget is not None:
                # Existing active requests wait while new prompts are prefilling;
                # only the incoming requests append their first/next token.
                need=self.kv_bytes(self.active)+self.kv_bytes(add+[candidate],1)
                if need>budget:
                    if not self.active and not add:raise ValueError('one observed prefix cannot fit paged KV budget')
                    break
            r=self.queue.popleft(); add.append(r); r.status='active'; r.replica=self.i
        if add:
            lengths=tuple(r.prompt+r.generated for r in add)
            duration=(self.s.be.prefill_group(lengths,self.i) if hasattr(self.s.be,'prefill_group') else self.s.be.prefill(lengths))*self.s.actor_slowdown(self,t)
            # Migration recomputes KV, including prefix, but emits only the next token.
            ids=tuple(r.rid for r in self.active+add)
            p=Period(t,t+duration,'prefill',ids,1,add=tuple(r.rid for r in add))
            self.active.extend(add)
        elif self.active:
            budget=self.s.kv_budget(self)
            if budget is not None:
                # Recompute preemption uses only known prefix pages and FIFO
                # priority. No response EOS length is consulted to choose a victim.
                while self.kv_bytes(self.active,1)>budget:
                    if len(self.active)==1:raise ValueError('single live sequence exceeds KV pages')
                    victim=self.active.pop();victim.status='queued';self.queue.appendleft(victim);self.preemptions+=1
                    self.s.log(t,'KV_PREEMPT',request=victim.rid,generation_observed=victim.generated,replica=self.i)
            steps=min(r.length-r.generated for r in self.active)
            if budget is not None:
                lo,hi=1,steps
                while lo<hi:
                    mid=(lo+hi+1)//2
                    if self.kv_bytes(self.active,mid)<=budget:lo=mid
                    else:hi=mid-1
                steps=lo
            b=len(self.active); contexts=sum(r.prompt+r.generated for r in self.active)
            base,slope=(self.s.be.affine_group(b,self.i) if hasattr(self.s.be,'affine_group') else self.s.be.affine(b)); factor=self.s.actor_slowdown(self,t)
            a=(base+slope*(contexts-b))*factor; delta=slope*b*factor
            duration=steps*a+delta*steps*(steps-1)/2
            p=Period(t,t+duration,'decode',tuple(r.rid for r in self.active),steps,a,delta)
        else: return
        p.slowdown=self.s.actor_slowdown(self,t)
        self.epoch+=1; p.epoch=self.epoch; self.period=p
        self.peak_kv=max(self.peak_kv,self.kv_bytes(self.active))
        self.s.push(p.end,'replica',(self.i,p.epoch))

    def interrupt(self,t,halt=False):
        if halt: self.halt_requested=True
        p=self.period
        if p is None:
            if halt: self.halted=True
            return
        if p.kind=='prefill' or p.end<=t+1e-12: return
        # Abort acknowledgements occur at the next token boundary, never for free.
        elapsed=max(0.,t-p.start); lo,hi=1,p.steps
        while lo<hi:
            mid=(lo+hi)//2
            if mid*p.a+p.b*mid*(mid-1)/2>=elapsed-1e-12: hi=mid
            else: lo=mid+1
        n=lo; self.epoch+=1
        p.steps=n; p.end=p.start+n*p.a+p.b*n*(n-1)/2; p.epoch=self.epoch
        self.s.push(p.end,'replica',(self.i,p.epoch))

    def finish(self,t,epoch):
        p=self.period
        if p is None or epoch!=p.epoch: return
        self.busy+=t-p.start
        self.segments.append([p.start,t,p.kind,len(p.ids),p.steps])
        if p.slowdown!=1:self.s.log(p.start,'MPS_ACTOR_SERVICE',replica=self.i,end=t,slowdown=p.slowdown,kind=p.kind)
        if p.kind=='prefill':
            for rid in p.add:
                r=self.s.reqs[rid]
                self.s.prefill_tokens+=r.prompt+r.generated
                if r.generated: self.s.recomputed_prefix_tokens+=r.prompt+r.generated
                r.generated+=1; r.compute_tokens+=1
        else:
            for rid in p.ids:
                r=self.s.reqs[rid]; r.generated+=p.steps; r.compute_tokens+=p.steps
        self.peak_kv=max(self.peak_kv,self.kv_bytes(self.active))
        remain=[]
        for r in self.active:
            assert 0<r.generated<=r.length
            if r.generated==r.length:
                r.finish_at=t; r.status='finished'
                self.s.push(t,'reward',r.rid)
            elif r.cancel_at is not None:
                r.status='cancelled'; self.s.log(t,'CANCEL_ACK',request=r.rid,generated=r.generated)
            else: remain.append(r)
        self.active=remain; self.period=None
        if self.halt_requested: self.halted=True
        if self.s.scaling and all(x.halted for x in self.s.replicas): self.s.migrate(t)
        else: self.start(t)
        self.s.after_replica_finish(self,t)

class Trainer:
    def __init__(self,sim):
        self.s=sim; self.pending=[]; self.ready={}; self.tasks=[]; self.used=set(); self.started=False
        self.unit=max(sim.be.p.train_mb,sim.be.p.logprob_mb)
    def enable(self,t,full=False):
        dp=self.s.be.world//self.s.be.p.train_tp
        workers=range(dp) if full else range(dp//2,dp)
        new=[w for w in workers if w not in self.ready]
        ready=self.s.prepare_training(t,new)
        for w in new: self.ready[w]=ready[w]
        self.started=True; self.drain(t,flush=full)
    def add(self,rs,t):
        self.pending.extend(rs)
        if self.started: self.drain(t)
    def drain(self,t,flush=False):
        while len(self.pending)>=self.unit or (flush and self.pending):
            if not self.ready: return
            # Actual lengths are available now; stable token balancing is legal.
            self.pending.sort(key=lambda r:(-(r.prompt+r.generated),r.rid))
            batch=self.pending[:self.unit]; del self.pending[:self.unit]
            w=min(self.ready,key=lambda w:(self.ready[w],w)); start=max(t,self.ready[w])
            ls=tuple(r.prompt+r.generated-1 for r in batch); rs=tuple(r.generated for r in batch)
            duration=self.s.be.work(ls,rs)
            for r in batch:
                assert r.rid not in self.used and r.accepted; self.used.add(r.rid)
            end=start+duration; self.ready[w]=end
            task={'worker':w,'start':start,'end':end,'requests':[r.rid for r in batch], 'stream':not self.s.finalizing}
            if hasattr(self.s.be,'work_schedule'):
                _,logs,trains=self.s.be.work_schedule(ls,rs)
                task.update(logprob_slices=logs,training_slices=trains)
            self.tasks.append(task)
            self.s.log(start,'BACKWARD_START',worker=w,count=len(batch),end=end)
    @property
    def end(self): return max(self.ready.values(),default=0.)

class Sim:
    def __init__(self,be,groups,version,variant,target,kind,cap=None,stream_policy='fixed-code-40'):
        self.be=be; self.variant=variant; self.version=version; self.kind=kind
        self.target=target; self.cap=cap or be.p.cap
        self.reqs={}; self.group_reqs=defaultdict(list); self.rewards=defaultdict(list)
        self.accepted=[]; self.accepted_set=set(); self.events=[]; self.heap=[]; self.serial=0
        self.prefill_tokens=0; self.recomputed_prefix_tokens=0
        self.stopped=False; self.finalizing=False; self.scaled=False; self.scaling=False
        self.migration_seconds=0.; self.scale_time=None; self.generation_end=None
        self.stream_policy=stream_policy; self.completed_requests=0; self.next_scale_ratio=.2
        self.replicas=[Replica(self,i) for i in range(be.world//be.p.rollout_tp)]
        self.trainer=Trainer(self)
        nr=10 if kind=='short' and variant!='verl' else 8
        for g in groups:
            for row in g['samples'][:nr]:
                rid=f'v{version}:{g["group_id"]}:{row["sample"]}'
                r=Req(rid,g['group_id'],row['sample'],row['prompt_length'],row['response_length'],version)
                self.reqs[rid]=r; self.group_reqs[r.gid].append(r)
                rep=min(self.replicas,key=lambda x:(len(x.queue),x.i))
                rep.queue.append(r); r.replica=rep.i
        self.log(0.,'ROUND_START',kind=kind,target=target,candidates=len(groups),requests=len(self.reqs),version=version)

    def log(self,t,event,**kw): self.events.append({'time':t,'event':event,**kw})
    def actor_can_start(self,replica,t):return True
    def actor_slowdown(self,replica,t):return 1.
    def kv_budget(self,replica):return None
    def initial_time(self):return 0.
    def prepare_training(self,t,workers):return {w:t for w in workers}
    def optimizer_ready(self,t):return t
    def publish_ready(self,t):return t
    def finish_state(self,t):return t
    def after_replica_finish(self,replica,t):pass
    def final_training_time(self):return self.generation_end
    def rollout_available(self):
        return self.be.capacity-self.be.state['training_base_max']-self.be.state['rollout_weights_per_rank']-1024**2
    def gradient_transfer(self):
        return 2*self.be.host_gradient((self.be.world//2)//self.be.p.train_tp) if self.scale_time is not None else 0.
    def decorate_summary(self,summary):return summary
    def handle_event(self,t,kind,data):
        if kind=='replica':self.replicas[data[0]].finish(t,data[1])
        elif kind=='reward':self.reward(data,t)
        elif kind=='resume':self.replicas[data].start(t)
    def push(self,t,kind,data):
        self.serial+=1; heapq.heappush(self.heap,(t,self.serial,kind,data))

    def cancel(self,rs,t):
        touched=set()
        for r in rs:
            if r.cancel_at is not None or r.status in ('finished','cancelled'): continue
            r.cancel_at=t
            if r.status=='queued': r.status='cancelled'
            else:
                rep=self.replicas[r.replica]
                if rep.halted and rep.period is None:
                    # The token-boundary halt has already paid for the prefix.
                    # Cancellation while waiting for other replicas to halt
                    # needs an immediate ACK, otherwise migration discards the
                    # request object while leaving its lifecycle "active".
                    if r in rep.active:rep.active.remove(r)
                    r.status='cancelled'
                    self.log(t,'CANCEL_ACK',request=r.rid,generated=r.generated,reason='already-halted')
                else:touched.add(r.replica)
            self.log(t,'CANCEL_REQUEST',request=r.rid,generated_observed=r.generated)
        for i in touched: self.replicas[i].interrupt(t)

    def reward(self,rid,t):
        r=self.reqs[rid]; r.reward_at=t
        self.completed_requests+=1
        self.log(t,'REWARD_READY',request=rid,generated=r.generated)
        if self.stream_policy=='paper-adaptive' and self.variant=='tail-stream' and not self.scaled and not self.scaling and not self.stopped:
            ratio=self.completed_requests/len(self.reqs)
            if ratio>=self.next_scale_ratio and self.next_scale_ratio<=.500001:
                self.try_scale(t); self.next_scale_ratio+=.05
        if self.stopped or r.gid in self.accepted_set: return
        self.rewards[r.gid].append(r)
        if len(self.rewards[r.gid])<8: return
        chosen=self.rewards[r.gid][:8]
        self.accepted_set.add(r.gid); self.accepted.append(r.gid)
        for x in chosen: x.accepted=True
        self.log(t,'GROUP_READY',group=r.gid,requests=[x.rid for x in chosen])
        self.trainer.add(chosen,t)
        if self.variant!='verl': self.cancel([x for x in self.group_reqs[r.gid] if not x.accepted],t)
        if len(self.accepted)==self.target:
            self.stopped=True; self.cancel(list(self.reqs.values()),t)
            self.log(t,'ACCEPTED_BATCH_READY',groups=len(self.accepted))
        elif self.stream_policy=='fixed-code-40' and self.variant=='tail-stream' and not self.scaled and not self.scaling and len(self.accepted)>=math.ceil(.4*self.target):
            self.try_scale(t)
        elif self.stream_policy=='paper-prompts' and self.variant=='tail-stream' and not self.scaled and not self.scaling and len(self.accepted)/self.target>=self.next_scale_ratio and self.next_scale_ratio<=.500001:
            self.try_scale(t);self.next_scale_ratio+=.05

    def try_scale(self,t):
        # Whole TP and train DP groups must fit exactly into the second half.
        half=self.be.world//2; tr=self.be.p.train_tp; tp=self.be.p.rollout_tp
        if half%tr or half%tp:
            self.log(t,'SCALE_REJECT',reason='communication-group split'); self.scaled=True; return
        remaining=sum(r.cancel_at is None and r.status not in ('finished','cancelled') for r in self.reqs.values())
        replicas=len(self.replicas)//2
        slots=math.ceil(remaining/replicas)
        length=8192
        if self.stream_policy in ('paper-adaptive','paper-prompts'):
            slots=min(self.cap,slots)
            observed=max((r.generated for r in self.reqs.values() if r.status=='active'),default=0)
            length=max(self.be.history_response_bound,observed)
        reserve=slots*(self.be.op.prompt_cap+length)*self.be.kv
        available=self.rollout_available()
        if reserve>available:
            self.log(t,'SCALE_REJECT',reason='KV reservation',reserve=reserve,available=available)
            if self.stream_policy=='fixed-code-40': self.scaled=True
            return
        self.scaling=True; self.log(t,'SCALE_REQUEST',reserve=reserve,available=available)
        for rep in self.replicas: rep.interrupt(t,halt=True)
        if all(rep.halted for rep in self.replicas): self.migrate(t)

    def migrate(self,t):
        self.scaling=False; self.scaled=True; self.scale_time=t
        half=len(self.replicas)//2; receivers=self.replicas[:half]; moved=[]
        for rep in self.replicas[half:]:
            moved.extend(rep.active); moved.extend(rep.queue); rep.active=[]; rep.queue.clear()
        node_bytes=defaultdict(int)
        for r in moved:
            if r.cancel_at is not None:
                if r.status not in ('finished','cancelled'):
                    r.status='cancelled';self.log(t,'CANCEL_ACK',request=r.rid,generated=r.generated,reason='migration-barrier')
                continue
            if r.status in ('finished','cancelled'):continue
            dst=min(receivers,key=lambda x:(len(x.active)+len(x.queue),x.i))
            r.migrations+=1; r.status='queued'; r.replica=dst.i; dst.queue.append(r)
            node_bytes[(dst.i*self.be.p.rollout_tp)//8]+=(r.prompt+r.generated)*4
        transfer=max(node_bytes.values(),default=0)/(self.be.op.profile['node_uplink_bytes_per_ns_per_direction']*NS)
        self.migration_seconds=transfer
        self.log(t,'MIGRATE',requests=len(moved),token_transfer_seconds=transfer)
        for rep in receivers:
            rep.halted=False; rep.halt_requested=False; self.push(t+transfer,'resume',rep.i)
        self.trainer.enable(t)

    def run(self):
        initial=self.initial_time()
        for rep in self.replicas: rep.start(initial)
        last=0.
        while self.heap:
            t,_,kind,data=heapq.heappop(self.heap); assert t>=last-1e-10; last=t
            self.handle_event(t,kind,data)
        assert self.stopped and len(self.accepted)==self.target
        assert all(r.status in ('finished','cancelled') for r in self.reqs.values()),Counter(r.status for r in self.reqs.values())
        # Stale invalidated events can be later than the real cancellation ACK.
        self.generation_end=max((max((seg[1] for seg in rep.segments),default=0.) for rep in self.replicas),default=0.)
        self.finalizing=True; final_ready=self.final_training_time();self.trainer.enable(final_ready,full=True)
        assert not self.trainer.pending and len(self.trainer.used)==self.target*8
        offload=self.gradient_transfer()
        # Conservatively charge host offload+reload after stream/final backward.
        backward_end=max(final_ready,self.trainer.end)
        reduce_start=self.optimizer_ready(backward_end+offload)
        end=self.finish_state(self.publish_ready(reduce_start+self.be.optimize())+self.be.publish())
        self.log(reduce_start,'GRADIENT_REDUCE_AND_UPDATE',samples=len(self.trainer.used),global_denominator=self.target*8)
        self.log(end,'PUBLISH_VERSION',version=self.version+1)
        accepted=[r for r in self.reqs.values() if r.accepted]
        assert all(r.version==self.version and r.reward_at is not None for r in accepted)
        generated=sum(r.compute_tokens for r in self.reqs.values())
        accepted_tokens=sum(r.generated for r in accepted)
        tasks=self.trainer.tasks
        for w in self.trainer.ready:
            tt=sorted((x for x in tasks if x['worker']==w),key=lambda x:x['start'])
            assert all(a['end']<=b['start']+1e-9 for a,b in zip(tt,tt[1:]))
        summary={'variant':self.variant,'version':self.version,'kind':self.kind,'seconds':end,
                 'generation_seconds':self.generation_end,'backward_end_seconds':backward_end,
                 'post_backward_seconds':end-backward_end,'host_gradient_seconds':offload,
                 'optimizer_seconds':self.be.optimize(),'publish_seconds':self.be.publish(),
                 'accepted_prompts':self.accepted,'accepted_samples':len(accepted),'submitted_samples':len(self.reqs),
                 'generated_tokens':generated,'accepted_tokens':accepted_tokens,'discarded_tokens':generated-accepted_tokens,
                 'prefill_tokens':self.prefill_tokens,'recomputed_prefix_tokens':self.recomputed_prefix_tokens,
                 'stream_scale_time':self.scale_time,'stream_samples':sum(len(x['requests']) for x in tasks if x['start']<self.generation_end),
                 'stream_samples_finished':sum(len(x['requests']) for x in tasks if x['end']<=self.generation_end),
                 'backward_gpu_seconds':sum((x['end']-x['start'])*self.be.p.train_tp for x in tasks),
                 'rollout_gpu_seconds':sum(rep.busy*self.be.p.rollout_tp for rep in self.replicas),
                 'kv_peak_bytes':max(rep.peak_kv for rep in self.replicas),
                 'preemptions':sum(rep.preemptions for rep in self.replicas),
                 'audit':{'accepted_once':True,'full_gradient_coverage':True,'single_version':True,'single_update':True,
                          'no_unfinished_requests':True,'no_worker_overlap':True,'charged_cancel_prefixes':True},
                 'requests':[vars(r) for r in self.reqs.values()],'events':self.events,'training_tasks':tasks,
                 'replicas':[{'replica':r.i,'busy_seconds':r.busy,'peak_kv':r.peak_kv,'segments':r.segments} for r in self.replicas]}
        return self.decorate_summary(summary)

def run_period(be,groups,variant,out,stream_policy='fixed-code-40',feedback=None,backend_factory=None,prompts_per_step=192,history_feedback=False,sim_factory=Sim,period_finalizer=None,long_shuffle_seed=None):
    out.mkdir(parents=True,exist_ok=False); remaining=deque(groups); waiting=deque(); results=[]; accepted=[]
    groupmap={g['group_id']:g for g in groups}; p=prompts_per_step; version=0; history=[]
    import random
    long_rng=random.Random(long_shuffle_seed) if long_shuffle_seed is not None else None
    def retry_group(gid):
        g=groupmap[gid]
        return {**g,'samples':g['retry_samples'],'attempt_role':'fresh-retry'} if 'retry_samples' in g else g
    while remaining or waiting:
        if variant=='verl':
            batch=[remaining.popleft() for _ in range(min(p,len(remaining)))]; kind='baseline'
        elif len(waiting)>=p:
            batch=[retry_group(waiting.popleft()) for _ in range(p)]; kind='long'
        elif len(remaining)>=math.ceil(1.25*p):
            batch=[remaining.popleft() for _ in range(math.ceil(1.25*p))]; kind='short'
        else:
            # Finite-window drain; must retain exact accepted batch size and bill retries.
            batch=[]
            while waiting and len(batch)<p: batch.append(retry_group(waiting.popleft()))
            while remaining and len(batch)<p: batch.append(remaining.popleft())
            kind='terminal-long'
        assert len(batch)>=p, 'workload must contain a whole number of accepted steps'
        if long_rng is not None and kind in ('long','terminal-long'):long_rng.shuffle(batch)
        sim=sim_factory(be,batch,version,variant,p,kind,stream_policy=stream_policy); result=sim.run()
        if long_rng is not None and kind in ('long','terminal-long'):
            result['long_shuffle']={'seed':long_shuffle_seed,'scope':'released source long shuffle, also applied to explicit finite-window terminal drain',
                'group_order_sha256':hashlib.sha256(json.dumps([g['group_id'] for g in batch]).encode()).hexdigest()}
        result['rollout_tp']=be.p.rollout_tp; result['rollout_cap']=be.p.cap
        if feedback is not None:
            result['tp_feedback']=feedback.observe(0)
            result['tp_feedback']['scope']='zero preemptions under conservative full-length KV admission; no fabricated evictions'
        accepted.extend(result['accepted_prompts'])
        if variant!='verl': waiting.extend(g['group_id'] for g in batch if g['group_id'] not in sim.accepted_set)
        result['long_queue_after']=len(waiting); result['fresh_remaining']=len(remaining)
        if history_feedback:
            history.extend(r['generated'] for r in result['requests'] if r['finish_at'] is not None)
            ordered=sorted(history);be.history_response_bound=ordered[min(len(ordered)-1,int(.95*len(ordered)))]
            result['next_history_response_p95']=be.history_response_bound
        (out/f'step-{version:03}.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
        compact={k:v for k,v in result.items() if k not in ('requests','events','training_tasks','replicas','accepted_prompts','reward_tasks','host_transfers')}
        results.append(compact); print(json.dumps({'progress':str(out),'step':version,**compact}),flush=True)
        version+=1
        if feedback is not None and backend_factory is not None and (remaining or waiting) and feedback.tp!=be.p.rollout_tp:
            be=backend_factory(feedback.tp)
    assert len(accepted)==len(groups) and Counter(accepted)==Counter(groupmap.keys())
    summary={'variant':variant,'closed_period':True,'steps':results,'total_seconds':sum(r['seconds'] for r in results),
             'prompts':len(accepted),'accepted_samples':len(accepted)*8,'remaining_long_queue':0,
             'total_generated_tokens':sum(r['generated_tokens'] for r in results),
             'total_discarded_tokens':sum(r['discarded_tokens'] for r in results),
             'mean_step_seconds':sum(r['seconds'] for r in results)/len(results),'status':'PASS_CONDITIONAL_EVENT_REPLAY'}
    if period_finalizer is not None:period_finalizer(summary)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--model',choices=['4B','8B','14B'],required=True); ap.add_argument('--variant',choices=['verl','tail','tail-stream'],required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--planner',choices=['frozen','offline-feedback'],default='frozen'); a=ap.parse_args()
    policy_path=a.inputs/f'{a.model}-gpu-policy.json'
    policy=Policy(**json.loads(policy_path.read_text())); be=Backend(a.root,a.model,policy); planner=None; feedback=None; factory=None
    if a.planner=='offline-feedback':
        from policy import offline, Feedback, admitted_cap
        from dataclasses import replace
        tune=json.loads((a.root/f'experiments/2026-10-29-qwen3/inputs/Qwen3-{a.model}/tune.json').read_text())['trajectories']
        be,planner=offline(a.root,a.model,policy,tune)
        if a.variant!='verl': feedback=Feedback(be.p.rollout_tp)
        def factory(tp):
            new=Backend(a.root,a.model,replace(policy,rollout_tp=tp)); cap,_=admitted_cap(new)
            new.p=replace(new.p,cap=cap); new.history_response_bound=be.history_response_bound
            return new
        policy=be.p
    groups=json.loads((a.inputs/(a.model+'-candidates.json')).read_text())
    original=[{'trajectory_id':f'{g["group_id"]}:{x["sample"]}',**x} for g in groups for x in g['samples'][:8]]
    admission=be.op.admission(original,policy.train_tp,policy.rollout_tp,policy.train_mb,policy.logprob_mb,policy.cap)
    assert all(admission['phase_peak_bytes'][phase]<=be.capacity for phase in ('generation','weight_export')),admission
    start=time.monotonic(); summary=run_period(be,groups,a.variant,a.out,
        stream_policy='paper-adaptive' if a.planner=='offline-feedback' else 'fixed-code-40',feedback=feedback,backend_factory=factory)
    (a.out/'identity.json').write_text(json.dumps({'model':a.model,'policy':vars(policy),'admission':admission,
         'source_root':str(a.root),'planner':planner,'input_sha256':hashlib.sha256((a.inputs/(a.model+'-candidates.json')).read_bytes()).hexdigest(),
         'sim_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wall_seconds':time.monotonic()-start,
         'scope':'conditional zero-reward-time projection, fixed output outcomes across actor versions; PP=CP=1; not paper reproduction'},indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
