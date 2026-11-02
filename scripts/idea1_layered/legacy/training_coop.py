"""Native safe-point transfer of unstarted frozen training batches.

Same-domain TP2 workers retain actor v0 and accumulate unnormalized gradients;
only token/mask/old-reference-logprob/advantage records move. The global update
waits for all original batch IDs, including remotely executed batches.
"""
from collections import defaultdict, deque
import hashlib
import json
from joint_runtime import JointRuntime
from locality_runtime_v3 import AttentionRuntime
from rl_xy.qwen3_sequence import compile_blocks, local
from rl_xy.qwen3_experiment import chunks, teacher_lengths


class TrainingCoopRuntime(JointRuntime):
    def __init__(self,*args,training_scope='domain',**kwargs):
        super().__init__(*args,**kwargs)
        self.training_scope=training_scope
        self.train_queues={}
        self.batch_manifest={}
        self.batch_state={}
        self.pending_batch={}
        self.helper_jobs={}
        self.helper_ready=set()
        self.batch_executions=[]
        self.train_moves=[]
        self.gradient_zero_programs={}

    def record_program(self,w,stage,steps):
        if stage=='gradient_zero':
            self.gradient_zero_programs[w]=hashlib.sha256(json.dumps(dict(repeat=1,steps=steps),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return super().record_program(w,stage,steps)

    @staticmethod
    def payload(items):
        # FP32 token IDs + byte mask, two FP32 per-response-token logprobs,
        # scalar FP32 advantage and conservative 64-byte identity/version record.
        return sum(5*(x['prompt_length']+x['response_length'])+8*x['response_length']+4+64 for x in items)

    def training_jobs(self,w):
        items=self.groups[w];ranks=self.rep[w].ranks
        for stage in ('old_policy','reference'):
            for i,batch in enumerate(chunks(items,self.logprob_mb)):
                steps=self.frozen[stage][w][i]['steps'] if self.frozen else compile_blocks(self.op,ranks,self.op.forward(teacher_lengths(batch),2))
                yield stage,[x['trajectory_id'] for x in batch],steps
        n=len(items)
        yield 'advantage',[],[local(self.op.vec(5*n+4*(n//8),8*n,4*n,'grpo_group_advantage'))]
        # The generator resumes only at the native completion of advantage.
        queue=deque()
        for i,batch in enumerate(chunks(items,self.train_mb)):
            steps=self.frozen['training'][w][i+1]['steps'] if self.frozen else compile_blocks(self.op,ranks,self.op.training(teacher_lengths(batch),2,sum(x['response_length'] for x in batch)))
            # AR uses the executing TP2's native communicator. No frozen body
            # may contain an absolute rank or a foreign communicator.
            assert all(s['kind'] in ('local','AR') and not any(k in s for k in ('src','dst','ranks','group')) for s in steps)
            bid=f'{w}:{i}';tids=[x['trajectory_id'] for x in batch]
            estimate=sum(s.get('compute',0)+s.get('memory',0) for s in steps)
            proof=dict(batch_id=bid,home=w,index=i,tids=tids,ready_tick=self.tick,
                       program_sha256=hashlib.sha256(json.dumps(dict(repeat=1,steps=steps),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                       payload_bytes_per_lane=self.payload(batch),estimate_local_ns=estimate)
            self.batch_manifest[bid]=proof;self.batch_state[bid]='queued'
            queue.append(dict(**proof,steps=steps))
        self.train_queues[w]=queue
        zero=self.frozen['training'][w][0]['steps'] if self.frozen else [local(self.op.vec(0,0,self.op.state(2,2)['layout'].padded_elements*4,'gradient_zero'))]
        yield 'gradient_zero',[],zero
        while queue:
            job=queue.popleft();self.pending_batch[w]=job;self.batch_state[job['batch_id']]='running'
            yield 'training',job['tids'],job['steps']

    def commit_stage(self,w):
        pending=self.pending_stage.get(w)
        if pending and pending[0]=='training':
            job=self.pending_batch.pop(w);bid=job['batch_id']
            assert self.batch_state[bid]=='running'
            self.batch_state[bid]='completed'
            self.batch_executions.append(dict(batch_id=bid,home=job['home'],worker=w,tids=job['tids'],
                start_tick=pending[2],end_tick=self.tick,program_sha256=job['program_sha256']))
            if w in self.helper_jobs:
                assert self.helper_jobs[w]['batch_id']==bid
                self.helper_jobs.pop(w);self.helper_ready.discard(w)
                self.stages[w]='gradient_ready'
        super().commit_stage(w)

    def dispatch_training(self,w):
        # Source initiates at a native batch boundary. A finished receiver is
        # already blocked, has freed reference/KV, and has a live gradient buffer.
        if len(self.done)!=len(self.ids) or not self.train_queues.get(w):return None
        free=[j for j in self.trained if j!=w and self.waiting[j] and j not in self.services
              and j not in self.helper_jobs and not self.leased(j)
              and (self.training_scope=='wafer' or j//3==w//3)]
        if not free:return None
        dst=min(free,key=lambda j:(abs(self.rep[j].ranks[0]-self.rep[w].ranks[0]),j))
        # All candidates are already EOS and advantage-ready. No hidden EOS or
        # recorded future completion time enters this choice.
        queue=self.train_queues[w];job=max(queue,key=lambda j:(j['estimate_local_ns'],j['index']))
        queue.remove(job);bid=job['batch_id'];assert self.batch_state[bid]=='queued'
        self.batch_state[bid]='transferring';self.helper_jobs[dst]=job
        self.move_id+=1
        tx=dict(id=self.move_id,kind='training_batch',src=w,dst=dst,batch_id=bid,
                tids=job['tids'],bytes_per_lane=job['payload_bytes_per_lane'],consumed=set(),start_tick=self.tick)
        self.services[w]=self.services[dst]=tx;self.waiting[w]=False
        return {'steps':[self.wake(dst,self.rep[w].ranks[0]),self.rpc(w,'sample_run')]}

    def sample_run(self,w):
        tx=self.services[w]
        if tx.get('kind')!='training_batch':return super().sample_run(w)
        steps=[];size=tx['bytes_per_lane']
        for lane,(src,dst) in enumerate(zip(self.rep[tx['src']].ranks,self.rep[tx['dst']].ranks)):
            steps += [local(self.op.vec(0,size,0,'prepared_training_record_read'),ranks=[src]),
                dict(kind='p2p_direct',src=src,dst=dst,bytes=size,tag=430000000+tx['id']*2+lane),
                local(self.op.vec(0,0,size,'prepared_training_record_write'),ranks=[dst])]
        key=f'training-copy-{tx["id"]}'
        steps += [dict(kind='gate_signal',key=key),dict(kind='gate_wait',key=key,threshold=4),
                  self.rpc(w,'sample_done',transfer=tx['id'])]
        return {'steps':steps}

    def sample_done(self,w,transfer):
        tx=self.services[w]
        if tx.get('kind')!='training_batch':return super().sample_done(w,transfer)
        self.services.pop(w);assert tx['id']==transfer and w not in tx['consumed']
        if not tx['consumed']:
            assert self.batch_state[tx['batch_id']]=='transferring'
            self.helper_ready.add(tx['dst']);self.batch_state[tx['batch_id']]='ready_remote'
            self.train_moves.append({k:v for k,v in tx.items() if k!='consumed'}|dict(end_tick=self.tick))
        tx['consumed'].add(w)
        return {'steps':[]}

    def finished_all(self):
        return (len(self.trained)==len(self.rep) and not self.services and not self.helper_jobs
                and all(s=='completed' for s in self.batch_state.values()))

    def step(self,w,it):
        self.waiting[w]=False;self.commit_stage(w)
        if w in self.services:return self.sample_run(w)
        self._observe(w)
        if not self.leased(w):
            transfer=self.schedule_transfer(w)
            if transfer is not None:return transfer
            transfer=self.dispatch_training(w)
            if transfer is not None:return transfer
        if w in self.helper_jobs:
            assert w in self.helper_ready
            job=self.helper_jobs[w];bid=job['batch_id'];assert self.batch_state[bid]=='ready_remote'
            self.batch_state[bid]='running';self.pending_batch[w]=job;self.stages[w]='training'
            self.pending_stage[w]=('training',job['tids'],self.tick)
            self.record_program(w,'training',job['steps'])
            return {'steps':job['steps']}
        if self.domain_drained(w) and (self.early or len(self.done)==len(self.ids)):
            expected={x['trajectory_id'] for x in self.groups[w]}
            if not expected<=self.received[w]:self.stages[w]='data_wait';return self.wait(w)
            if w not in self.jobs:self.jobs[w]=iter(self.training_jobs(w))
            if w not in self.trained:
                try:stage,tids,steps=next(self.jobs[w])
                except StopIteration:self.trained.add(w);self.stages[w]='gradient_ready'
                else:
                    self.stages[w]=stage;self.pending_stage[w]=(stage,tids,self.tick)
                    self.record_program(w,stage,steps);return {'steps':steps}
            if self.finished_all():
                prefix=[self.wake(j,self.rep[w].ranks[0]) for j in self.rep if self.waiting[j]]
                if prefix:return {'steps':prefix+[self.rpc(w,'joint_finish')]}
                self.returned.add(w);return {'steps':[],'finish':True}
            return self.wait(w)
        self.stages[w]='generation'
        result=AttentionRuntime.step(self,w,it)
        if result.get('finish'):return self.wait(w)
        for step in result['steps']:
            if step.get('kind')=='controller_rpc' and step['event']['phase']=='finish':step['event']['phase']='step'
        return result

    def audit(self):
        super().audit()
        assert self.finished_all() and not self.pending_batch
        assert len(self.batch_executions)==len(self.batch_manifest)
        assert {e['batch_id'] for e in self.batch_executions}==set(self.batch_manifest)
        assert all(not q for q in self.train_queues.values())
