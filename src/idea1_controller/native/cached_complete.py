"""Batch program cache independent of plan and native runtime outcomes."""
from .complete import CompleteKernel
from rl_xy.qwen3_sequence import local
from rl_xy.qwen3_experiment import chunks,teacher_lengths
from .context import digest

class CachedCompleteKernel(CompleteKernel):
    def _batch_program(self,r,stage,batch):
        steps=self.batch_catalog.body(stage,len(self.region_ranks[r]),batch)
        for step in steps:
            if step['kind']=='AR':step['group']=self.region_communicators[r]
        return steps

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
            for batch in chunks(items,self.logprob_mb):
                jobs.append((stage,[x['trajectory_id'] for x in batch],self._batch_program(r,stage,batch),None))
        n=len(items);jobs.append(('advantage',[],[local(self.op.vec(5*n+4*(n//8),8*n,4*n,'grpo_group_advantage'))],None))
        jobs.append(('gradient_zero',[],[local(self.op.vec(0,0,self.op.state(2,2)['layout'].padded_elements*4,'gradient_zero'))],None))
        for i,batch in enumerate(chunks(items,self.train_mb)):
            steps=self._batch_program(r,'training',batch)
            bid=f'{r}:{i}';tids=[x['trajectory_id'] for x in batch]
            proof=dict(batch_id=bid,home=r,index=i,tids=tids,ready_tick=self.tick,training_tp=tp,physical_ranks=ranks,
                program_sha256=digest(dict(repeat=1,steps=steps)),payload_bytes_per_lane=self.payload(batch),estimate_local_ns=sum(s.get('compute',0)+s.get('memory',0) for s in steps))
            self.batch_manifest[bid]=proof;self.batch_state[bid]='queued'
            jobs.append(('training',tids,steps,dict(**proof,steps=steps)))
        jobs.append(('gradient_restore',[],self._restore_gradient(r) if items else [],None))
        self.region_programs[r]=tuple((stage,tids,self._synchronized(r,i,steps),job) for i,(stage,tids,steps,job) in enumerate(jobs))

