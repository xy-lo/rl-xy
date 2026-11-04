"""Real TP2/4/8 training regions over immutable inference TP2 replicas.

Native programs run on every physical member, including region collectives.
Gradients are explicitly restored to the original TP2 distributed-optimizer
contract. This keeps optimizer owners and publication fixed while changing DP.
No numeric tensors are executed; all costs/traffic use frozen operator services.
"""
from collections import deque
from dataclasses import replace
from pathlib import Path
import hashlib,json
from training_coop import TrainingCoopRuntime
from joint_runtime import JointRuntime
from rl_xy.qwen3_sequence import compile_blocks,local
from rl_xy.qwen3_cost import Cost
from rl_xy.qwen3_experiment import chunks,teacher_lengths
from rl_xy.model import enumerate_parameters
from .context import NativeContext,digest


def order_ready_training_jobs(jobs,policy):
    """Reorder only observed-shape training bodies, keeping phase boundaries.

    Called after EOS data readiness when a region is compiled. The unchanged
    generator exposes the training queue only after native advantage quorum.
    Sample sets, batch bodies, ticket identities and update weights are intact.
    """
    if policy=='input':return jobs
    if policy!='ready_lpt':raise ValueError('unknown ready training order')
    import math
    positions=[i for i,row in enumerate(jobs) if row[0]=='training']
    if positions and positions!=list(range(min(positions),max(positions)+1)):
        raise ValueError('training bodies must have one contiguous phase')
    bodies=[jobs[i] for i in positions]
    for _,_,_,job in bodies:
        cost=job['estimate_local_ns']
        if not math.isfinite(cost) or cost<=0:raise ValueError('invalid observed-shape training estimate')
    bodies.sort(key=lambda row:(-row[3]['estimate_local_ns'],row[3]['index']))
    result=list(jobs)
    for i,row in zip(positions,bodies):result[i]=row
    return result


class RegionKernel(TrainingCoopRuntime):
    def configure_regions(self, regions, assigned, plan):
        if plan.get('training_ready_order','input') not in ('input','ready_lpt'):raise ValueError('unknown ready training order')
        cap=plan.get('late_region_batch_cap_per_pair',0)
        if type(cap) is not int or cap not in (0,1,2,4):raise ValueError('invalid late-region batch cap')
        if cap and not hasattr(self,'_batches'):raise ValueError('late-region batch cap requires adaptive memory-gated builder')
        packing=plan.get('adaptive_training_batch_max',0)
        if type(packing) is not int or packing not in (0,1,2,4,8):raise ValueError('invalid adaptive training batch maximum')
        if packing and not hasattr(self,'_batches'):raise ValueError('adaptive packing requires memory-gated builder')
        self.resource_layout=plan
        self.region_members={int(r):tuple(ws) for r,ws in regions.items()}
        self.region_of={w:r for r,ws in self.region_members.items() for w in ws}
        assert sorted(self.region_of)==sorted(self.rep)
        self.region_ranks={r:tuple(self.rep[w].ranks[h] for h in range(2) for w in ws) for r,ws in self.region_members.items()}
        self.groups=[[] for _ in self.rep]
        for r,ws in self.region_members.items():self.groups[r]=[x for w in ws for x in assigned[w]]
        self.targets={x['trajectory_id']:r for r,xs in enumerate(self.groups) for x in xs}
        self.region_programs={};self.region_communicators={};self.restore_proofs=[]
        self.region_events=[];self.region_program_hashes={};self.region_started=set()

    def step(self,*args,**kwargs):raise RuntimeError('Use hierarchical policies, not the monolithic step')

    def same_domain(self,a,b):return self.region_of[a]==self.region_of[b]
    def domain_drained(self,w):
        return all(self.empty(j) and not self.leased(j) for j in self.region_members[self.region_of[w]])

    def _synchronized(self,r,index,steps):
        # All physical TP ranks finish this job before any next callback.
        key=f'region-{r}-job-{index}'
        return [dict(kind='gate_signal',key=key+'-begin'),dict(kind='gate_wait',key=key+'-begin',threshold=len(self.region_ranks[r]))]+steps+[dict(kind='gate_signal',key=key),dict(kind='gate_wait',key=key,threshold=len(self.region_ranks[r]))]

    def _compile(self,r,blocks):
        steps=compile_blocks(self.op,self.region_ranks[r],blocks)
        for s in steps:
            if s['kind']=='AR':s['group']=self.region_communicators[r]
        return steps

    def _restore_gradient(self,r):
        """Reverse-order streaming expansion into the original TP2 buffers.

        Logical shards: t=h*m+k -> old lane h, designated replica root r.
        Root local pieces expand backward, then other k pieces arrive; other
        original DP replicas become zero contributors. FP32 Q/K norm partials
        are summed across k; original suffix merges the two h contributions.
        """
        ranks=self.region_ranks[r];ws=self.region_members[r];m=len(ws);tp=2*m
        steps=[];tag=460000000+r*100000
        for row_index,row in reversed(list(enumerate(enumerate_parameters(self.op.case_tp(tp))))):
            partial=row.layout=='replicated_per_head_dimension' if hasattr(row,'layout') else row.sharding=='replicated_per_head_dimension'
            if row.replicated and not partial:continue # equal replica: root already has correct value
            size=row.local_elements_per_die*4
            for h in range(2):
                dst=self.rep[r].ranks[h]
                # Local repack is explicit; source shard has a valid alias only
                # for TP2. Reverse row traversal prevents in-place expansion
                # from overwriting a later row's unread compact producer data.
                if m>1:steps.append(local(self.op.vec(0,size,size,'gradient_local_expand'),ranks=[dst]))
                for k in range(1,m):
                    src=self.rep[ws[k]].ranks[h]
                    # Bounded pack/unpack, all endpoint service explicitly paid.
                    steps += [local(self.op.vec(0,size,size,'gradient_pack'),ranks=[src]),
                              local(self.op.vec(0,size,0,'gradient_send_read'),ranks=[src]),
                              dict(kind='p2p_direct',src=src,dst=dst,bytes=size,tag=tag),
                              local(self.op.vec(0,0,size,'gradient_receive_write'),ranks=[dst]),
                              local(self.op.vec(size//4 if partial else 0,size,size,'gradient_unpack_or_partial_sum'),ranks=[dst])]
                    self.restore_proofs.append(dict(region=r,parameter=row.parameter_id,logical_lane=h,subshard=k,subshards=m,
                        src=src,dst=dst,bytes=size,semantics='partial_sum' if partial else 'canonical_shard_concat',tag=tag))
                    tag+=1
        # All data has arrived before any non-root producer clears its buffer.
        key=f'gradient-restore-{r}';steps += [dict(kind='gate_signal',key=key),dict(kind='gate_wait',key=key,threshold=tp)]
        padded=self.op.state(2,2)['layout'].padded_elements*4
        for w in ws[1:]:
            steps.append(local(self.op.vec(0,0,padded,'zero_noncontributing_dp_gradient'),ranks=list(self.rep[w].ranks)))
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
                jobs.append((stage,[x['trajectory_id'] for x in batch],self._compile(r,self.op.forward(teacher_lengths(batch),tp)),None))
        n=len(items);jobs.append(('advantage',[],[local(self.op.vec(5*n+4*(n//8),8*n,4*n,'grpo_group_advantage'))],None))
        jobs.append(('gradient_zero',[],[local(self.op.vec(0,0,self.op.state(2,2)['layout'].padded_elements*4,'gradient_zero'))],None))
        for i,batch in enumerate(chunks(items,self.train_mb)):
            steps=self._compile(r,self.op.training(teacher_lengths(batch),tp,sum(x['response_length'] for x in batch)))
            bid=f'{r}:{i}';tids=[x['trajectory_id'] for x in batch]
            proof=dict(batch_id=bid,home=r,index=i,tids=tids,ready_tick=self.tick,training_tp=tp,physical_ranks=ranks,
                program_sha256=digest(dict(repeat=1,steps=steps)),payload_bytes_per_lane=self.payload(batch),estimate_local_ns=sum(s.get('compute',0)+s.get('memory',0) for s in steps))
            self.batch_manifest[bid]=proof;self.batch_state[bid]='queued'
            jobs.append(('training',tids,steps,dict(**proof,steps=steps)))
        jobs.append(('gradient_restore',[],self._restore_gradient(r) if items else [],None))
        self._finalize_region_program(r,jobs)

    def _finalize_region_program(self,r,jobs):
        # All base/cached/adaptive builders call this shared boundary. Sorting
        # before synchronization keeps every TP rank on the same gate order.
        jobs=order_ready_training_jobs(jobs,self.resource_layout.get('training_ready_order','input'))
        self.region_programs[r]=tuple((stage,tids,self._synchronized(r,i,steps),job) for i,(stage,tids,steps,job) in enumerate(jobs))

    def training_jobs(self,w):
        r=self.region_of[w]
        if r not in self.region_programs:self._build_region_program(r)
        for stage,tids,steps,job in self.region_programs[r]:
            if stage=='training':
                self.pending_batch[w]=job
                if w==r:self.batch_state[job['batch_id']]='running'
            yield stage,tids if w==r else [],steps

    def commit_stage(self,w):
        r=self.region_of[w];pending=self.pending_stage.get(w)
        if pending and pending[0]=='training':
            if w==r:
                job=self.pending_batch.pop(w);bid=job['batch_id'];assert self.batch_state[bid]=='running'
                self.batch_state[bid]='completed';self.batch_executions.append(dict(batch_id=bid,home=r,worker=r,tids=job['tids'],
                    physical_ranks=self.region_ranks[r],training_tp=len(self.region_ranks[r]),start_tick=pending[2],end_tick=self.tick,program_sha256=job['program_sha256']))
            else:self.pending_batch.pop(w)
        JointRuntime.commit_stage(self,w)

    def region_step(self,w,it,c):
        assert c.region_eligible(w);r=self.region_of[w];c.roles[w]='TRAINING'
        expected={x['trajectory_id'] for x in self.groups[r]}
        # DRAIN is a real data-wait state for every member, not generation.
        # Only the first data-ready member activates/wakes the TP quorum.
        if not expected<=self.received[r]:self.stages[w]='data_wait';return self.wait(w)
        if r not in self.region_started:
            self.region_started.add(r)
            prefix=[self.wake(j,self.rep[w].ranks[0]) for j in self.region_members[r] if j!=w and self.waiting[j]]
            if prefix:return {'steps':prefix+[local(Cost(compute=1000)),self.rpc(w,'step')]}
        if w not in self.jobs:self.jobs[w]=iter(self.training_jobs(w))
        if w not in self.trained:
            try:stage,tids,steps=next(self.jobs[w])
            except StopIteration:self.trained.add(w);self.stages[w]='gradient_ready'
            else:
                self.stages[w]=stage;self.pending_stage[w]=(stage,tids,self.tick);self.record_program(w,stage,steps)
                return {'steps':steps}
        if self.finished_all():
            prefix=[self.wake(j,self.rep[w].ranks[0]) for j in self.rep if self.waiting[j]]
            if prefix:return {'steps':prefix+[self.rpc(w,'joint_finish')]}
            self.returned.add(w);return {'steps':[],'finish':True}
        return self.wait(w)


class RegionContext(NativeContext):
    def region_eligible(self,w):
        k=self.kernel;r=k.region_of[w]
        return super().region_eligible(w)
    def region_id(self,w):return f'R{self.kernel.region_of[w]}'
    def _fixed_plan(self):
        from ..contracts import TrainingRegion
        plan=super()._fixed_plan();k=self.kernel
        training=tuple(TrainingRegion(f'R{r}',k.region_ranks[r],len(k.region_ranks[r])) for r in sorted(k.region_members))
        candidate=replace(plan.placement.candidate,candidate_id='native-regions',training_tp_by_region={r.region_id:r.tp for r in training})
        placement=replace(plan.placement,plan_id=digest(k.resource_layout),candidate=candidate,training=training,
            admission_domains={f'R{r}':ranks for r,ranks in k.region_ranks.items()},
            drain_dependencies={f'R{r}':tuple(f'I{w}' for w in ws) for r,ws in k.region_members.items()})
        return replace(plan,step_plan_id='planned-'+self.model_id,placement=placement,gradient_routing_digest=digest(dict(layout=k.resource_layout,route='explicit-TPp-to-original-TP2-suffix')))
    def snapshot(self):
        snap=super().snapshot();dies={d:replace(s,execution_group_id=self.region_id(d//2) if s.assigned_role=='TRAINING' else s.execution_group_id) for d,s in snap.dies.items()}
        return replace(snap,dies=dies)
