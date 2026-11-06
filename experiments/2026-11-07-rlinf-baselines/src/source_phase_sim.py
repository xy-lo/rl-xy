"""Source-shaped DP training rounds rather than a global work-stealing queue.

veRL: frozen input shards, global old/ref/train phase barriers.
RP: ready prompt groups prefetched in multiples of 2*train_MB*active_DP,
chunked over active DP workers with barriers for each prefetched round.
The common GRPO contract charges real old/reference passes, not source fake
random log-probs. No future-length sorting or cross-DP microbatch stealing.
"""
import math
from collections import Counter
from platform_backend import PlatformSim
from sim import Trainer


class SourceTrainer(Trainer):
    def __init__(self,sim):
        super().__init__(sim);self.round_index=0;self.ready_event=None;self.prefetch_rounds=0

    def enable(self,t,full=False):
        dp=self.s.be.world//self.s.be.p.train_tp
        first=self.s.retained_ranks//self.s.be.p.train_tp
        workers=range(dp) if full else range(first,dp)
        new=[w for w in workers if w not in self.ready]
        ready=self.s.prepare_training(t,new)
        for w in new:self.ready[w]=ready[w]
        self.started=True;self.drain(t,flush=full)

    def drain(self,t,flush=False):
        if not self.ready or not self.pending:return
        if not flush:
            if self.s.stopped:return
            available=max(self.ready.values())
            if available>t+1e-10:
                if self.ready_event!=available:
                    self.ready_event=available;self.s.push(available,'training_round_ready',available)
                return
        workers=sorted(self.ready);dp=len(workers);p=self.s.be.p
        if not flush:
            # Source pg_prompt_count converts this sample multiple into whole
            # 8-answer prompt groups. Never begin on an incomplete prompt.
            unit=math.lcm(2*p.train_mb*dp,8)
            n=len(self.pending)//unit*unit
            if getattr(self.s,'source_prefetch_limit',False):
                if len(self.s.accepted)>=self.s.target-1:return
                already=len(self.used)
                remaining=max(0,(self.s.target-self.s.be.world)*8-already)
                n=min(n,remaining//unit*unit)
            if getattr(self.s,'source_training_order',False) and self.prefetch_rounds:
                # Released BaseWorker starts with prefetch_prompt_count=-1,
                # then asks for exactly one pg_prompt_count per subsequent fetch.
                n=min(n,unit)
            if not n:return
        else:n=len(self.pending)
        if getattr(self.s,'source_prefetch_limit',False) and getattr(self.s,'training_assignment','input')=='input':
            order={gid:i for i,gid in enumerate(self.s.group_reqs)}
            self.pending.sort(key=lambda r:(order[r.gid],r.reward_at,r.rid))
        batch=self.pending[:n];del self.pending[:n]
        if getattr(self.s,'source_training_order',False) and not flush:
            before=[r.rid for r in batch]
            permutation=self.s.be.source_prefetch_rng.permutation(len(batch)).tolist()
            batch=[batch[i] for i in permutation]
            self.s.log(t,'SOURCE_PREFETCH_ORDER',prefetch_round=self.prefetch_rounds,unit_samples=unit,
                before=before,permutation=permutation,after=[r.rid for r in batch],seed=42)
            self.prefetch_rounds+=1
        if self.s.variant=='verl' and getattr(self.s,'training_assignment','input')=='input':
            order={rid:i for i,rid in enumerate(self.s.reqs)}
            batch.sort(key=lambda r:order[r.rid])
        assignment=getattr(self.s,'training_assignment','input')
        if assignment in ('arrival_block','arrival_rr'):
            assert len(batch)%8==0 and all(len({r.gid for r in batch[i:i+8]})==1 for i in range(0,len(batch),8))
            request_order={rid:i for i,rid in enumerate(self.s.reqs)}
            groups=[sorted(batch[i:i+8],key=lambda r:request_order[r.rid]) for i in range(0,len(batch),8)]
            if assignment=='arrival_rr':
                assert len(groups)%dp==0
                groups=[group for w in range(dp) for group in groups[w::dp]]
            batch=[r for group in groups for r in group]
        elif assignment!='input':
            raise ValueError(assignment)
        self.s.log(t,'TRAIN_GROUP_ASSIGNMENT',assignment=assignment,groups=[r.gid for r in batch[::8]],dp=dp)
        # Contiguous data.chunk(DP) is the source dispatch. Uneven terminal
        # shards on54dies are explicit; no padding, dropping or duplicate loss.
        q,extra=divmod(n,dp);shards={};i=0
        for k,w in enumerate(workers):
            size=q+int(k<extra);shards[w]=batch[i:i+size];i+=size
        start=max(t,max(self.ready.values()));round_start=start
        phases=['old_logprob']+(['reference_logprob'] if getattr(self.s.be,'teacher_forwards',2)==2 else [])+['advantage','backward']
        for phase in phases:
            if phase=='backward' and getattr(self.s,'dp_balance',False):
                from rlinf_balance import get_seqlen_balanced_partitions
                all_samples=[r for w in workers for r in shards[w]]
                parts=get_seqlen_balanced_partitions([r.prompt+r.generated for r in all_samples],dp,True)
                assert len(parts)==dp and all(len(part)==len(all_samples)//dp for part in parts)
                new_shards={w:[all_samples[j] for j in parts[k]] for k,w in enumerate(workers)}
                assert len({r.rid for rs in new_shards.values() for r in rs})==len(all_samples)
                # Explicit padded tensor model for RLinf all_gather_object path.
                seq_len=16384;prompt_cap=self.s.be.op.prompt_cap;response_cap=seq_len-prompt_cap
                bytes_per_sample=18*seq_len+12*response_cap+21
                local_payload=(len(all_samples)//dp)*bytes_per_sample+65536
                ranks=[r for w in workers for r in range(w*p.train_tp,(w+1)*p.train_tp)]
                balance_start=start
                down=self.s.transfer(ranks,lambda _:local_payload,'D2H',start,'DP_BALANCE_BATCH_TO_CPU')
                stage=self.s.transfer(ranks,lambda _:local_payload,'H2D',max(down.values()),'DP_BALANCE_OBJECT_STAGE_GPU')
                plan=self.s.be.make_plan()
                for shard in range(p.train_tp):
                    group=tuple(w*p.train_tp+shard for w in workers)
                    plan.comm(group,'ALL_GATHER',local_payload,'dp_balance_all_gather_object')
                allgather=self.s.be.fixed_plan(plan);allgather_start=max(stage.values());allgather_end=allgather_start+allgather
                full_payload=local_payload*dp
                gather=self.s.transfer(ranks,lambda _:full_payload,'D2H',allgather_end,'DP_BALANCE_OBJECT_TO_CPU')
                selected=self.s.transfer(ranks,lambda _:local_payload,'H2D',max(gather.values()),'DP_BALANCE_SELECTED_TO_GPU')
                start=max(selected.values())+getattr(self.s,'balance_control_overhead',0.)
                self.s.log(balance_start,'DP_BALANCE_DONE',end=start,algorithm='RLinf-karmarkar-karp',seq_len=seq_len,prompt_cap=prompt_cap,response_cap=response_cap,bytes_per_sample=bytes_per_sample,local_payload=local_payload,allgather_start=allgather_start,allgather_end=allgather_end,control_overhead=getattr(self.s,'balance_control_overhead',0.),partitions=[[r.rid for r in new_shards[w]] for w in workers])
                shards=new_shards
            if phase=='reference_logprob' and getattr(self.s,'reference_swap',False):
                ranks=[r for w in workers for r in range(w*p.train_tp,(w+1)*p.train_tp)]
                payload=lambda r:self.s.allocs[r]['actor_BF16_parameter_buffer']
                down=self.s.transfer(ranks,payload,'D2H',start,'REF_SWAP_ACTOR_TO_CPU')
                up=self.s.transfer(ranks,payload,'H2D',max(down.values()),'REF_SWAP_REFERENCE_TO_GPU')
                start=max(up.values())
            ends=[]
            for w in workers:
                rs=shards[w]
                if not rs:continue
                lengths=tuple(r.prompt+r.generated-1 for r in rs)
                training=phase=='backward';limit=p.train_mb if training else p.logprob_mb
                slices=self.s.be.fit_chunks(lengths,limit,training)
                duration=0.
                if phase=='advantage':
                    duration=self.s.be.op.vec(5*len(rs)+4*(len(rs)//8),8*len(rs),4*len(rs),'grpo_group_advantage').ns/1e9
                for a,b in ([] if phase=='advantage' else slices):
                    blocks=(self.s.be.op.training(lengths[a:b],p.train_tp,sum(r.generated for r in rs[a:b])) if training
                            else self.s.be.op.forward(lengths[a:b],p.train_tp))
                    duration+=(self.s.be.worker_blocks(blocks,p.train_tp,w) if hasattr(self.s.be,'worker_blocks') else self.s.be.blocks(blocks,p.train_tp))
                end=start+duration;ends.append(end)
                task={'worker':w,'phase':phase,'round':self.round_index,'start':start,'end':end,
                    'requests':[r.rid for r in rs],'stream':not self.s.finalizing,
                    ('training_slices' if training else 'logprob_slices'):slices}
                self.tasks.append(task)
                self.s.log(start,'BACKWARD_START' if training else phase.upper()+'_START',worker=w,count=len(rs),end=end,round=self.round_index)
                if training:
                    for r in rs:
                        assert r.accepted and r.rid not in self.used;self.used.add(r.rid)
            start=max(ends,default=start)
            if phase=='reference_logprob' and getattr(self.s,'reference_swap',False):
                ranks=[r for w in workers for r in range(w*p.train_tp,(w+1)*p.train_tp)]
                payload=lambda r:self.s.allocs[r]['actor_BF16_parameter_buffer']
                restore=self.s.transfer(ranks,payload,'H2D',start,'REF_SWAP_ACTOR_RESTORE')
                start=max(restore.values())
            self.s.log(start,'TRAIN_PHASE_BARRIER',phase=phase,workers=workers,round=self.round_index)
        for w in workers:self.ready[w]=start
        self.s.log(start,'TRAIN_ROUND_DONE',round=self.round_index,start=round_start,samples=n,workers=workers)
        self.round_index+=1


class SourcePhaseSim(PlatformSim):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.trainer=SourceTrainer(self)

    def handle_event(self,t,kind,data):
        if kind=='training_round_ready':
            if self.trainer.ready_event==data:
                self.trainer.ready_event=None
                if not self.stopped:self.trainer.drain(t)
            return
        return super().handle_event(t,kind,data)

    def decorate_summary(self,d):
        d=super().decorate_summary(d);tasks=d['training_tasks'];tp=self.be.p.train_tp
        fb=[t for t in tasks if t.get('phase')=='backward']
        d['stream_samples']=sum(len(t['requests']) for t in fb if t['start']<d['generation_seconds'])
        d['stream_samples_finished']=sum(len(t['requests']) for t in fb if t['end']<=d['generation_seconds'])
        for phase,key in [('old_logprob','old_policy_gpu_seconds'),('reference_logprob','reference_gpu_seconds'),('backward','backward_gpu_seconds')]:
            d[key]=sum((t['end']-t['start'])*tp for t in tasks if t.get('phase')==phase)
        d['training_dispatch']=getattr(self,'training_assignment','input')+'; fixed-DP full-batch simulation, explicit reference swap when requested'
        d['training_rounds']=self.trainer.round_index
        return d
