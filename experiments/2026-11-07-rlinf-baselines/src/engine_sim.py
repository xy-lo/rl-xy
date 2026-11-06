"""Paged-prefix admission and bounded mixed prefill for the common engine.

Models the V1 FIFO scheduling contract (block16, decode before new prefills),
not CUDA kernels or exact vLLM implementation. EOS is private to execution.
2048 is an upper sequence limit, never a promise that all sequences fit KV.
"""
from collections import deque
from functools import lru_cache
import math
from sim import Replica,Period
from rl_xy.qwen3_cost import Cost


def suffix_attention(op,queries,contexts,tp,backward=False):
    """Upstream tiled causal attention extended to a known cached prefix.

    A query tile ending at e attends through (context-query)+e. For q=c
    and q=1 this is exactly the frozen provider's arithmetic/tile accounting.
    No calibration interpolation is silently extrapolated to the new shapes.
    """
    assert len(queries)==len(contexts) and all(0<q<=c for q,c in zip(queries,contexts))
    m=op.model;heads=m.query_heads//tp;hd=m.head_dim;cap=op.hw.sram_bytes_per_core
    candidates=[]
    for q in (16,32,64,128,256):
        for k in (16,32,64,128,256):
            qq=min(q,max(queries));kk=min(k,max(contexts))
            work=(6*(qq+2*kk)*hd if backward else 2*(qq+2*kk)*hd+4*qq*hd)+4*qq*kk+16*qq
            if work<=cap:candidates.append((qq,kk,work))
    if not candidates:raise ValueError('attention has no legal private SRAM tile')
    qtile,ktile,workspace=max(candidates,key=lambda x:(x[0],x[1]));pairs=kv_reads=tasks=0
    for q,c in zip(queries,contexts):
        if q==1:pairs+=c;kv_reads+=c;tasks+=heads
        else:
            for start in range(0,q,qtile):
                count=min(qtile,q-start);end=c-q+start+count
                pairs+=count*end;kv_reads+=end;tasks+=heads
    flops=pairs*heads*hd*(10 if backward else 4);vector_ops=pairs*heads*(20 if backward else 5);units=min(op.units,tasks)
    if op.profile:
        unit_rate=op.profile['bf16_flops_per_ns']/op.units;vector_rate=op.profile['vector_ops_per_ns']/op.units;launch=op.profile['kernel_launch_ns']
    else:unit_rate=op.hw.gemm_flops_per_ns;vector_rate=op.hw.vector_ops_per_ns;launch=0
    compute=(flops/(units*unit_rate)+vector_ops/(units*vector_rate))/op.compute_scale+launch
    tokens=sum(queries);reads=2*kv_reads*heads*hd*2+tokens*heads*hd*2;writes=tokens*heads*hd*2+tokens*heads*8
    if backward:reads=reads*2+tokens*heads*hd*6;writes=tokens*(heads+2*m.kv_heads//tp)*hd*4
    traffic=reads+writes
    memory=max(traffic/op.hw.dram_bytes_per_ns,traffic/op.hw.module_path_bytes_per_ns,traffic/(units*op.hw.sram_bytes_per_ns))
    assert not getattr(op,'attention_calibration',None),'partial-prefix shapes require an explicitly validated calibration'
    return Cost(compute,memory,flops,int(reads),int(writes),workspace*units)


class PagedReplica(Replica):
    def __init__(self,*args):
        super().__init__(*args);self.cached={};self.peak_scheduled_tokens=0

    def kv_bytes(self,rs,extra=0):
        return sum(math.ceil((self.cached.get(r.rid,0)+extra)/16)*16 for r in rs)*self.s.be.kv

    def preempt(self,t,r):
        self.active.remove(r);r.status='queued';self.cached.pop(r.rid,None)
        self.queue.appendleft(r);self.preemptions+=1
        self.s.log(t,'KV_PREEMPT',request=r.rid,generation_observed=r.generated,replica=self.i)

    def start(self,t):
        if self.period or self.halted or not self.s.actor_can_start(self,t):return
        self.queue=deque(r for r in self.queue if r.cancel_at is None and r.status not in ('finished','cancelled'))
        budget=self.s.kv_budget(self);assert budget is not None
        # A pure decode interval may be coalesced until an EOS or KV boundary.
        # Incoming callbacks interrupt it at an actual next-token boundary.
        decode=bool(self.active) and all(self.cached.get(r.rid,0)==r.prompt+r.generated-1 and r.generated for r in self.active)
        can_admit=bool(self.queue) and len(self.active)<self.s.cap and self.kv_bytes(self.active)+16*self.s.be.kv<=budget
        if decode and not can_admit:
            while self.kv_bytes(self.active,1)>budget:
                if len(self.active)==1:raise ValueError('single observed context exceeds paged KV pool')
                self.preempt(t,self.active[-1])
            steps=min(r.length-r.generated for r in self.active)
            lo,hi=1,steps
            while lo<hi:
                mid=(lo+hi+1)//2
                if self.kv_bytes(self.active,mid)<=budget:lo=mid
                else:hi=mid-1
            steps=lo;b=len(self.active);contexts=sum(self.cached[r.rid]+1 for r in self.active)
            base,slope=(self.s.be.affine_group(b,self.i) if hasattr(self.s.be,'affine_group') else self.s.be.affine(b));factor=self.s.actor_slowdown(self,t)
            a=(base+slope*(contexts-b))*factor;delta=slope*b*factor
            p=Period(t,t+steps*a+delta*steps*(steps-1)/2,'decode',tuple(r.rid for r in self.active),steps,a,delta)
            p.engine_updates=tuple((r.rid,steps,steps,0) for r in self.active)
            self.peak_scheduled_tokens=max(self.peak_scheduled_tokens,b)
        else:
            remaining=self.s.max_batched_tokens;scheduled=[]
            # Active decoding first, then unfinished prefixes, then FIFO waiting.
            order=sorted(self.active,key=lambda r:not bool(r.generated and self.cached.get(r.rid,0)==r.prompt+r.generated-1))
            used=self.kv_bytes(self.active)
            for r in order:
                if r not in self.active:continue
                if not remaining:break
                old=self.cached.get(r.rid,0);target=r.prompt+r.generated
                q=min(target-old,remaining);assert q>0
                def extra(n):return (math.ceil((old+n)/16)-math.ceil(old/16))*16*self.s.be.kv
                while q and used+extra(q)>budget:q-=1
                if not q:
                    # Recompute preemption follows FIFO priority. Release the
                    # lowest-priority active prefix, including its KV pages.
                    while used+extra(1)>budget and self.active:
                        victim=self.active[-1]
                        if any(x[0]==victim.rid for x in scheduled):break
                        released=self.kv_bytes([victim]);self.preempt(t,victim);used-=released
                        if victim is r:break
                    if r not in self.active:continue
                    q=min(target-old,remaining)
                    while q and used+extra(q)>budget:q-=1
                if not q:continue
                used+=extra(q);remaining-=q
                scheduled.append((r.rid,q,int(old+q==target),q if old+q==target and r.generated else 0))
            while self.queue and remaining and len(self.active)<self.s.cap:
                r=self.queue[0];target=r.prompt+r.generated;q=min(target,remaining)
                # Prefix is allocated only as scheduled, never by future EOS.
                free=int((budget-used)//(16*self.s.be.kv))*16;q=min(q,free)
                if q<=0:break
                self.queue.popleft();r.status='active';r.replica=self.i;self.cached[r.rid]=0;self.active.append(r)
                used+=math.ceil(q/16)*16*self.s.be.kv;remaining-=q
                scheduled.append((r.rid,q,int(q==target),q if r.generated else 0))
            if not scheduled:
                if self.active or self.queue:raise ValueError('paged scheduler made no progress')
                return
            qs=tuple(x[1] for x in scheduled);cs=tuple(self.cached.get(x[0],0)+x[1] for x in scheduled)
            outputs=sum(x[2] for x in scheduled)
            duration=self.s.be.mixed_prefill(qs,cs,outputs,self.i)*self.s.actor_slowdown(self,t)
            p=Period(t,t+duration,'mixed',tuple(x[0] for x in scheduled),1)
            p.engine_updates=tuple(scheduled)
            self.peak_scheduled_tokens=max(self.peak_scheduled_tokens,sum(qs))
            self.s.log(t,'ENGINE_BATCH',replica=self.i,tokens=sum(qs),outputs=outputs,end=p.end)
        p.slowdown=self.s.actor_slowdown(self,t);self.epoch+=1;p.epoch=self.epoch;self.period=p
        self.s.push(p.end,'replica',(self.i,p.epoch))

    def interrupt(self,t,halt=False):
        # Chunked prefill/mixed forward is an atomic engine step. Token-only
        # intervals preserve the parent's next-token acknowledgement rule.
        if self.period and self.period.kind=='mixed':
            if halt:self.halt_requested=True
            return
        return super().interrupt(t,halt)

    def finish(self,t,epoch):
        p=self.period
        if p is None or epoch!=p.epoch:return
        self.busy+=t-p.start;self.segments.append([p.start,t,p.kind,len(p.ids),p.steps])
        for rid,q,emits,recompute in p.engine_updates:
            r=self.s.reqs[rid]
            if p.kind=='decode':q=emits=p.steps
            else:
                # Decode tokens mixed with new prefixes are not prefill work.
                if q>1 or not r.generated or self.cached.get(rid,0)!=r.prompt+r.generated-1:
                    self.s.prefill_tokens+=q
                    if r.generated:self.s.recomputed_prefix_tokens+=q
            self.cached[rid]=self.cached.get(rid,0)+q;r.generated+=emits;r.compute_tokens+=emits
        self.peak_kv=max(self.peak_kv,self.kv_bytes(self.active))
        assert self.peak_kv<=self.s.kv_budget(self)+1
        remain=[]
        for r in self.active:
            assert 0<=r.generated<=r.length
            if r.generated==r.length:
                r.finish_at=t;r.status='finished';self.s.push(t,'reward',r.rid);self.cached.pop(r.rid,None)
            elif r.cancel_at is not None:
                r.status='cancelled';self.cached.pop(r.rid,None);self.s.log(t,'CANCEL_ACK',request=r.rid,generated=r.generated)
            else:remain.append(r)
        self.active=remain;self.period=None
        if self.halt_requested:self.halted=True
        if self.s.scaling and all(x.halted for x in self.s.replicas):self.s.migrate(t)
        else:self.start(t)
        self.s.after_replica_finish(self,t)


class F2FIFOReplica(Replica):
    """Old common FIFO service shape, preserving f2 calibrated FA geometry.

    Observed-prefix KV admission/cancellation/migration still use the shared
    lifecycle. The prefill service covers only newly admitted full prefixes;
    existing decodes wait, matching the accepted f2 FIFO batching convention.
    """
    def __init__(self,*args):
        super().__init__(*args);self.peak_scheduled_tokens=0

    def start(self,t):
        before=self.period;super().start(t);p=self.period
        if p is not None and p is not before:
            tokens=sum(self.s.reqs[r].prompt+self.s.reqs[r].generated for r in p.add) if p.kind=='prefill' else len(p.ids)
            self.peak_scheduled_tokens=max(self.peak_scheduled_tokens,tokens)


class EngineBackendMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if getattr(self,'cost_contract',None)=='f2':
            # Preserve calibrated FA and direction-SRAM attention. The f2 FIFO
            # engine never issues uncalibrated partial-prefix query geometry.
            return
        from types import MethodType
        self.op.attention=lru_cache(maxsize=65536)(MethodType(suffix_attention,self.op))

    @lru_cache(maxsize=16384)
    def mixed_prefill(self,queries,contexts,outputs,replica=0):
        tp=self.p.rollout_tp
        blocks=self.op.forward(queries,tp,'generation',contexts)
        head_count=2 if tp>1 else 1
        # Partial prefixes produce no logits/sampling; completed prefixes and
        # decoded requests do. Keep the exact upstream per-layer cost model.
        blocks=blocks[:-head_count]
        if outputs:blocks+=self.op.forward((1,)*outputs,tp,'generation',(1,)*outputs)[-head_count:]
        if hasattr(self,'_group_blocks'):return self._group_blocks(tuple(blocks),tp,replica)
        return self.blocks(blocks,tp)
