"""Ticketed regional training donation into restored TP2 execution pairs.

Heterogeneous regions keep their original quorum/gradient protocol. A TP2/4/8 donor may move an unstarted, advantage-ready batch to a completed
TP2 pair using capacity-bounded sequential TP2 microbatches. For a heterogeneous receiver, every member must first
finish its canonical TP2 restoration. The receiver accumulates into its already restored TP2
buffer; the unchanged global SUM includes it once. Larger TP regions never
silently execute a TP2 body. Native record transport is charged in full.
"""
from dataclasses import asdict
from training_coop import TrainingCoopRuntime
from .context import digest

class RegionalTrainingHelp:
    def training_jobs(self,w):
        r=self.region_of[w]
        if r not in self.region_programs:self._build_region_program(r)
        for stage,tids,steps,job in self.region_programs[r]:
            if stage=='gradient_zero' and w==r:
                from collections import deque
                # Published only after the full native advantage quorum.
                self.train_queues[w]=deque(x[3] for x in self.region_programs[r] if x[0]=='training')
            if stage=='training':
                bid=job['batch_id'];key='training-ticket-'+bid
                if not hasattr(self,'training_tickets'):self.training_tickets={}
                if w==r:
                    run=self.batch_state[bid]=='queued';self.training_tickets[bid]=run
                    if not hasattr(self,'training_ticket_events'):self.training_ticket_events=[]
                    self.training_ticket_events.append(dict(batch_id=bid,region=r,run_here=run,tick=self.tick))
                    if run:
                        self.train_queues[w].remove(job);self.batch_state[bid]='running'
                # Followers cannot inspect a skip/run decision before the
                # root publishes it. A positive native control event prevents
                # a same-clock callback/cache cycle. Signals come from one rank.
                from rl_xy.qwen3_sequence import local
                from rl_xy.qwen3_cost import Cost
                yield 'training_ticket',[],[local(Cost(compute=1000)),
                    dict(kind='gate_signal',key=key,rank=self.rep[r].ranks[0]),
                    dict(kind='gate_wait',key=key,threshold=1)]
                if not self.training_tickets[bid]:continue
                self.pending_batch[w]=job
            yield stage,tids if w==r else [],steps

    def portable_training_job(self,job):
        if not hasattr(self,'portable_jobs'):self.portable_jobs={}
        bid=job['batch_id']
        if bid not in self.portable_jobs:
            items=[self.items[t] for t in job['tids']];body=[];subbatches=[];certs=[]
            for batch in self.memory.batches(items,'training',2,self.train_mb):
                body.extend(self.batch_catalog.body('training',2,batch))
                subbatches.append([x['trajectory_id'] for x in batch]);certs.append(asdict(self.memory.phase('training',2,batch)))
            assert all(s['kind'] in ('local','AR') and not set(s)&{'group','src','dst','ranks'} for s in body)
            local_ns=sum(s.get('compute',0)+s.get('memory',0) for s in body)
            net_ns=sum(3*s.get('bytes',0)/self.op.hw.d2d_bytes_per_ns_per_direction+3*self.op.hw.d2d_latency_ns for s in body if s['kind']=='AR')
            self.portable_jobs[bid]=dict(batch_id=bid,tids=job['tids'],steps=body,subbatches=subbatches,
                peak_allocation=max(certs,key=lambda x:x['peak_bytes']),estimate_ns=local_ns+net_ns,
                source_training_tp=job['training_tp'],sha256=digest(dict(repeat=1,steps=body)))
        return self.portable_jobs[bid]

    def training_help_choice(self,w):
        if w!=self.region_of[w]:return None
        if self.leased(w) or not self.train_queues.get(w):return None
        if self.resource_layout.get('training_help_admission','global-eos')!='region-ready' and len(self.done)!=len(self.ids):return None
        free=[j for j in self.trained if j!=w and all(v in self.trained for v in self.region_members[self.region_of[j]])
              and self.waiting[j] and j not in self.services and j not in self.helper_jobs and not self.leased(j)
              and (self.training_scope=='wafer' or self.same_domain(j,w))]
        if not free:return None
        def hops(j):return max(abs(a//6-b//6)+abs(a%6-b%6) for a,b in zip(self.rep[w].ranks,self.rep[j].ranks))
        dst=min(free,key=lambda j:(hops(j),j))
        job=max(self.train_queues[w],key=lambda x:(x['estimate_local_ns'],x['index']))
        # All token/advantage data is already observed, so this check uses no
        # future EOS. A positive service margin also avoids tiny bad transfers.
        size=job['payload_bytes_per_lane'];transport=size/self.op.hw.d2d_bytes_per_ns_per_direction*max(1,hops(dst))+2*size/self.op.hw.dram_bytes_per_ns
        remote=self.portable_training_job(job)
        remaining=sum(x['estimate_local_ns'] for x in self.train_queues[w])
        if remaining<=(remote['estimate_ns']+transport)*1.05:return None
        return dst,job

    def dispatch_training(self,w):
        choice=self.training_help_choice(w)
        if choice is None:return None
        dst,job=choice;self.train_queues[w].remove(job);bid=job['batch_id']
        assert self.batch_state[bid]=='queued'
        portable=self.portable_training_job(job)
        remote=dict(job,steps=portable['steps'],program_sha256=portable['sha256'],portable=portable)
        self.batch_state[bid]='transferring';self.helper_jobs[dst]=remote
        self.move_id+=1;tx=dict(id=self.move_id,kind='training_batch',src=w,dst=dst,batch_id=bid,
            tids=job['tids'],bytes_per_lane=job['payload_bytes_per_lane'],consumed=set(),start_tick=self.tick)
        self.services[w]=self.services[dst]=tx;self.waiting[w]=False
        return {'steps':[self.wake(dst,self.rep[w].ranks[0]),self.rpc(w,'sample_run')]}

    def commit_stage(self,w):
        pending=self.pending_stage.get(w)
        if pending and pending[0]=='training' and w in self.helper_jobs:
            job=self.helper_jobs[w];cert=job['portable']['peak_allocation'];assert cert['peak_bytes']<=self.op.capacity
            self.actual_allocation_events.append(dict(certificate_tick=pending[2],region=w,stage='training',tp=2,
                items=job['tids'],initial_region=self.region_of[w],execution_group='training-help-I'+str(w),physical_ranks=self.rep[w].ranks,sequential_subbatches=job['portable']['subbatches'],allocation_semantics='maximum of sequential TP2 microbatch allocations',**cert,start_tick=pending[2],end_tick=self.tick,remote_from=job['home']))
            TrainingCoopRuntime.commit_stage(self,w)
            self.batch_executions[-1].update(training_tp=2,physical_ranks=self.rep[w].ranks,remote=True,source_training_tp=job['training_tp'],sequential_subbatches=job['portable']['subbatches'])
            return
        return super().commit_stage(w)

class RegionalTrainingCooperator:
    def __init__(self,c):self.c=c
    def propose_unstarted_batch_takeovers(self,region_id,snapshot):
        c=self.c;k=c.kernel
        if region_id!=c.region_id(c.w) or not c.is_step() or c.w in k.services:return ()
        result=[];choice=k.training_help_choice(c.w)
        if choice:
            dst,job=choice
            result.append(c.propose('training_transfer',20,'TAKE_TRAIN_BATCH',peer=dst,tids=job['tids'],batch_id=job['batch_id']))
        if c.w in k.helper_jobs:
            result.append(c.propose('run_remote_batch',30,'START_BACKWARD',tids=k.helper_jobs[c.w]['tids']))
        return tuple(result)

# Early experiment imports retain this name; execution supports all TP sizes.
SingletonTrainingHelp=RegionalTrainingHelp
