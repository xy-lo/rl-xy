"""Compact-gradient training arenas and capacity-gated regional microbatches."""
from .cached_complete import CachedCompleteKernel
from .complete import CompleteAdapter
from .batch_catalog import FrozenBatchCatalog,OperatorBatchCatalog
from ..memory import RegionalMemory
from .context import digest
from rl_xy.qwen3_sequence import local
from rl_xy.qwen3_experiment import chunks,teacher_lengths
from dataclasses import asdict
import json

class AdaptiveKernel(CachedCompleteKernel):
    def init_adaptive(self,record_reserve):
        self.memory=RegionalMemory(self.op,record_reserve)
        self.allocation_events=[];self.mb_choices=[];self.actual_allocation_events=[]
    def _batches(self,items,stage,tp):
        factor=tp//2
        maximum=(self.train_mb*factor*2 if stage=='training' else self.logprob_mb*factor)
        # K remains8. Max training batch8 avoids crossing unrelated groups in
        # this first implementation and preserves small frozen TP2 batches.
        if stage=='training':maximum=min(8,maximum) if tp>2 else self.train_mb
        cap=getattr(self,'resource_layout',{}).get('late_region_batch_cap_per_pair',0)
        original_maximum=maximum
        late=bool(cap and stage=='training' and len(self.done)==len(self.ids))
        # The complete rollout state is observed here. Early regions retain
        # their original batch efficiency; late regions create finer atomic
        # jobs so a busy donor reaches helper dispatch points sooner.
        if late:maximum=min(maximum,cap*factor)
        for batch in self.memory.batches(items,stage,tp,maximum):
            proof=asdict(self.memory.phase(stage,tp,batch));assert proof['peak_bytes']<=self.op.capacity
            self.allocation_events.append(dict(certificate_tick=self.tick,region=self.targets[batch[0]['trajectory_id']],stage=stage,tp=tp,items=[x['trajectory_id'] for x in batch],**proof))
            choice=dict(stage=stage,tp=tp,maximum=maximum,actual=len(batch))
            if cap and stage=='training':
                choice.update(late_region_cap_eligible=late,late_region_cap_applied=late and maximum<original_maximum,original_maximum=original_maximum,decision_tick=self.tick,
                    region=self.targets[batch[0]['trajectory_id']],global_rollout_complete=len(self.done)==len(self.ids))
            self.mb_choices.append(choice)
            yield batch

    def _build_region_program(self,r):
        items=self.groups[r];ranks=self.region_ranks[r];tp=len(ranks);jobs=[]
        # Existing EOS copies deliver the complete dataset to the root TP2.
        # Redistribute to every additional training member, preserving lane.
        size=sum((x['prompt_length']+x['response_length'])*4 for x in items)
        steps=[];tag=440000000+r*100
        for w in (self.region_members[r][1:] if size else []):
            for src,dst in zip(self.rep[r].ranks,self.rep[w].ranks):
                steps += [local(self.op.vec(0,size,0,'region_record_read'),ranks=[src]),
                          dict(kind='p2p_direct',src=src,dst=dst,bytes=size,tag=tag),
                          local(self.op.vec(0,0,size,'region_record_write'),ranks=[dst])];tag+=1
        jobs.append(('region_data',[],steps,None))
        for stage in ('old_policy','reference'):
            for batch in self._batches(items,stage,tp):
                jobs.append((stage,[x['trajectory_id'] for x in batch],self._batch_program(r,stage,batch),None))
        n=len(items);jobs.append(('advantage',[],[local(self.op.vec(5*n+4*(n//8),8*n,4*n,'grpo_group_advantage'))],None))
        jobs.append(('gradient_zero',[],[local(self.op.vec(0,0,self.memory.compact_gradient(tp),'gradient_zero'))],None))
        for i,batch in enumerate(self._batches(items,'training',tp)):
            steps=self._batch_program(r,'training',batch)
            bid=f'{r}:{i}';tids=[x['trajectory_id'] for x in batch]
            proof=dict(batch_id=bid,home=r,index=i,tids=tids,ready_tick=self.tick,training_tp=tp,physical_ranks=ranks,
                program_sha256=digest(dict(repeat=1,steps=steps)),payload_bytes_per_lane=self.payload(batch),estimate_local_ns=sum(s.get('compute',0)+s.get('memory',0) for s in steps))
            self.batch_manifest[bid]=proof;self.batch_state[bid]='queued'
            jobs.append(('training',tids,steps,dict(**proof,steps=steps)))
        jobs.append(('gradient_restore',[],self._restore_gradient(r) if items else ([local(self.op.vec(0,0,self.memory.grad2,'zero_empty_expanded_TP2_gradients'))] if tp>2 else []),None))
        self._finalize_region_program(r,jobs)


    def commit_stage(self,w):
        pending=self.pending_stage.get(w)
        if pending and w==self.region_of[w] and pending[0] in ('old_policy','reference','training'):
            stage,tids,start=pending
            cert=next(x for x in self.allocation_events if x['region']==w and x['stage']==stage and x['items']==tids)
            self.actual_allocation_events.append(dict(**cert,start_tick=start,end_tick=self.tick))
        super().commit_stage(w)
