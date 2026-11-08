"""Existing Qwen3 workload adaptation: CPU rewards, engine and state lifetimes.

Common numerical-work contract retains real old/reference passes. This is
source-method adaptation to declared hardware costs, not H800 paper numbers.
"""
from collections import defaultdict
import math
from engine_sim import PagedReplica,EngineBackendMixin,F2FIFOReplica
from platform_backend import PlatformBackend
from source_phase_sim import SourcePhaseSim
from sim import Sim
from functools import lru_cache


class FullBackend(EngineBackendMixin,PlatformBackend):
    @lru_cache(maxsize=8192)
    def _group_blocks(self,blocks,tp,group):
        # Unlike the old search-only NetworkProfiles alpha/beta shortcut,
        # execute the pinned Plan's endpoint reads/reductions, protocols and
        # routed collective rounds on the group's actual physical ranks.
        # Concurrent different TP groups are still isolated service profiles;
        # do not call this full native inter-group contention calibration.
        plan=self.make_plan();ranks=tuple(range(group*tp,(group+1)*tp))
        assert ranks[-1]<self.world
        plan.blocks(ranks,blocks,'operator_service')
        # PlatformBackend.fixed_plan already returns seconds (Plan stores ns).
        return self.fixed_plan(plan)

    def blocks(self,blocks,tp):return self._group_blocks(tuple(blocks),tp,0)
    def worker_blocks(self,blocks,tp,worker):return self._group_blocks(tuple(blocks),tp,worker)

    @lru_cache(maxsize=8192)
    def prefill_group(self,lengths,replica):
        return self._group_blocks(tuple(self.op.forward(tuple(lengths),self.p.rollout_tp,'generation')),self.p.rollout_tp,replica)

    @lru_cache(maxsize=4096)
    def affine_group(self,b,replica):
        tp=self.p.rollout_tp
        one=self._group_blocks(tuple(self.op.forward((1,)*b,tp,'generation',(1,)*b)),tp,replica)
        two=self._group_blocks(tuple(self.op.forward((1,)*b,tp,'generation',(2,)*b)),tp,replica)
        assert two>=one
        return one,(two-one)/b


class FullMethodSim(SourcePhaseSim):
    def __init__(self,*args,reward_bank=None,cpu_workers=32,max_batched_tokens=8192,host_bytes_per_second=25e9,cpu_timeline=None,released_source_training=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.f2_engine=getattr(self.be,'cost_contract',None)=='f2'
        self.max_batched_tokens=self.be.p.cap*self.be.model_seq_cap if self.f2_engine else max_batched_tokens
        self.cap=self.be.p.cap if self.f2_engine else 2048
        self.source_prefetch_limit=True
        # Active entry is rlvr_pipeline_async, not the older debug Megatron
        # pipeline. Its completed-round feedback affects the NEXT iteration.
        self.source_training_order=bool(released_source_training and self.variant=='tail-stream')
        self.source_stream_ratio=getattr(self.be,'source_next_stream_ratio',.4) if self.source_training_order else .4
        if self.source_training_order and not hasattr(self.be,'source_prefetch_rng'):
            import numpy as np
            self.be.source_prefetch_rng=np.random.RandomState(42)
        # Rebuild queues before any execution; requests retain their stable FIFO
        # assignment and IDs. No future response length enters this operation.
        old=self.replicas;replica_class=F2FIFOReplica if self.f2_engine else PagedReplica
        self.replicas=[replica_class(self,r.i) for r in old]
        for src,dst in zip(old,self.replicas):dst.queue=src.queue
        self.bank=reward_bank;self.cpu_ready=[0.]*cpu_workers;self.cpu_rr=0
        self.cpu_timeline=cpu_timeline
        if cpu_timeline is not None:
            self.cpu_ready=[max(0.,t-cpu_timeline['clock']) for t in cpu_timeline['ready']]
            self.cpu_rr=cpu_timeline['round_robin']
        self.reward_tasks=[];self.reward_submitted=set();self.accepted_at=None
        self.baseline_reward_dispatched=False
        self.host_bw=host_bytes_per_second;self.host_ready=defaultdict(float);self.host_tasks=[]
        self.loaded=set();self.allocs={r:self.be.state['layout'].state_bytes(r//self.be.p.train_tp)['allocations'] for r in range(self.be.world)}
        self.runtime_bytes=3*1024**3

    def rollout_available(self):
        return self.be.capacity*.85-self.be.state['rollout_weights_per_rank']-self.runtime_bytes

    def kv_budget(self,replica):return self.rollout_available()

    def try_scale(self,t):
        # Released scheduler directly halves the infer ranks at40% of target
        # completed PROMPTS. It has no paper historical-P95 reservation gate.
        if not 0<self.retained_ranks<self.be.world:
            self.scaled=True;self.log(t,'SCALE_REJECT',reason='no whole-group half split');return
        self.scaling=True;self.log(t,'SCALE_REQUEST',trigger='source-completed-prompts',ratio=self.source_stream_ratio,retained_ranks=self.retained_ranks)
        for rep in self.replicas:rep.interrupt(t,halt=True)
        if all(rep.halted for rep in self.replicas):self.migrate(t)

    def reward(self,rid,t):
        if self.stopped:
            self.log(t,'LATE_EOS_NO_REWARD_SUBMITTED',request=rid);return
        if self.variant=='verl':
            # Frozen veRL invokes reward computation after generate returns.
            if any(r.status not in ('finished','cancelled') for r in self.reqs.values()):return
            if self.baseline_reward_dispatched:return
            self.baseline_reward_dispatched=True
            for request in self.reqs:self.submit_reward(request,t)
        else:self.submit_reward(rid,t)

    def submit_reward(self,rid,t):
        assert rid not in self.reward_submitted;self.reward_submitted.add(rid)
        r=self.reqs[rid]
        duration=0. if self.bank is None else float(self.bank[(r.gid,r.sample)]['seconds'])
        if self.bank is not None:assert self.bank[(r.gid,r.sample)]['status'] in ('PASS_MEASUREMENT','MODELLED_ZERO')
        i=self.cpu_rr%len(self.cpu_ready);self.cpu_rr+=1;start=max(t,self.cpu_ready[i]);end=start+duration;self.cpu_ready[i]=end
        self.reward_tasks.append({'request':rid,'domain':'math' if self.be.model_seq_cap==32768 else 'code','resource':'cpu','worker':i,'submitted':t,'start':start,'end':end})
        self.log(t,'REWARD_SUBMIT',request=rid,worker=i);self.push(end,'cpu_reward_done',rid)

    def handle_event(self,t,kind,data):
        if kind=='cpu_reward_done':
            # Keep the source floor(0.4*batch) threshold (76 for192), rather
            # than rounding up or switching to paper request milestones.
            prior=self.stream_policy;self.stream_policy='source-code-40'
            Sim.reward(self,data,t);self.stream_policy=prior
            if self.stopped and self.accepted_at is None:self.accepted_at=t
            if self.variant=='tail-stream' and not self.scaled and not self.scaling and not self.stopped and len(self.accepted)>=int(self.source_stream_ratio*self.target):self.try_scale(t)
            return
        return super().handle_event(t,kind,data)

    def final_training_time(self):
        assert self.accepted_at is not None
        return max(self.generation_end,self.accepted_at)

    def transfer(self,ranks,payload_fn,direction,t,event):
        # Declared6-node host interface shared by8 logical endpoints. Same
        # rate/placement for direct wafer transfer; not calibrated wafer I/O.
        by=defaultdict(list)
        for rank in ranks:by[rank//8].append(rank)
        ends={}
        for node,rs in sorted(by.items()):
            payload=sum(payload_fn(r) for r in rs);start=max(t,self.host_ready[node]);end=start+payload/self.host_bw
            self.host_ready[node]=end;ends.update({r:end for r in rs})
            row={'node':node,'ranks':rs,'start':start,'end':end,'bytes':payload,'direction':direction,'event':event}
            self.host_tasks.append(row);self.log(start,event,node=node,ranks=rs,bytes=payload,end=end,direction=direction)
        return ends

    def prepare_training(self,t,workers):
        ranks=[r for w in workers for r in range(w*self.be.p.train_tp,(w+1)*self.be.p.train_tp)]
        if not ranks:return {}
        assert not self.loaded.intersection(ranks)
        for rank in ranks:
            rep=self.replicas[rank//self.be.p.rollout_tp]
            assert rep.period is None and not rep.active
        off=self.transfer(ranks,lambda r:self.be.state['rollout_weights_per_rank'],'D2H',t,'ROLLOUT_WEIGHTS_OFFLOAD')
        def payload(r):
            a=self.allocs[r];return sum(a[k] for k in ('actor_BF16_parameter_buffer','master_FP32_shard','reference_BF16_valid_parameters'))
        loaded=self.transfer(ranks,payload,'H2D',max(off.values()),'TRAIN_ACTOR_MASTER_REFERENCE_LOAD')
        zero=max(self.allocs[r]['full_FP32_gradient_working_buffer'] for r in ranks)/(self.be.op.hw.dram_bytes_per_ns*1e9)
        self.loaded.update(ranks)
        return {w:max(loaded[r] for r in range(w*self.be.p.train_tp,(w+1)*self.be.p.train_tp))+zero for w in workers}

    def optimizer_ready(self,t):
        assert len(self.loaded)==self.be.world
        ends=self.transfer(range(self.be.world),lambda r:sum(self.allocs[r][k] for k in ('adam_m_FP32_shard','adam_v_FP32_shard')),'H2D',t,'ADAM_STATES_LOAD')
        return max(ends.values())

    def publish_ready(self,t):
        def payload(r):return sum(v for k,v in self.allocs[r].items() if k!='full_FP32_gradient_working_buffer')
        off=self.transfer(range(self.be.world),payload,'D2H',t,'TRAIN_STATES_OFFLOAD')
        up=self.transfer(range(self.be.world),lambda r:self.allocs[r]['actor_BF16_parameter_buffer']+self.be.state['rollout_weights_per_rank'],'H2D',max(off.values()),'PUBLISH_SOURCE_ROLLOUT_LOAD')
        return max(up.values())

    def finish_state(self,t):
        ends=self.transfer(range(self.be.world),lambda r:self.allocs[r]['actor_BF16_parameter_buffer'],'D2H',t,'PUBLISH_SOURCE_OFFLOAD')
        return max(ends.values())

    def decorate_summary(self,d):
        d=super().decorate_summary(d);reward_end=max((x['end'] for x in self.reward_tasks),default=0.)
        if self.source_training_order:
            # With fixed TP, use_large_tp staysFalse in the released pipeline.
            # Only already-accepted responses from this finished round enter
            # feedback; never peek at pending or next-round response lengths.
            observed=max(r.generated for r in self.reqs.values() if r.accepted)
            self.be.source_next_stream_ratio=.4 if observed>=.9*8192 else .5
            d['source_stream_feedback']={'ratio_used':self.source_stream_ratio,'observed_accepted_response_max':observed,
                'response_cap':8192,'next_ratio':self.be.source_next_stream_ratio,'scope':'released fixed-TP40/50 feedback, not paper20..50 KV adaptation'}
            d['source_training_order']='source prefetch response permutation before DP chunk; first fetch variable, subsequent at most one pg unit; common local-epoch shuffleFalse retained'
        d.update(reward_tasks=self.reward_tasks,reward_end_seconds=reward_end,accepted_batch_ready_seconds=self.accepted_at,
                 cpu_reward_workers=len(self.cpu_ready),submitted_reward_samples=len(self.reward_submitted),
                 measured_reward_service_seconds=sum(x['end']-x['start'] for x in self.reward_tasks),
                 host_transfers=self.host_tasks,host_bandwidth={'H2D':self.host_bw,'D2H':self.host_bw},
                 state_boundary='CPU actor/master/reference/Adam; GPU updated rollout; gradients discarded after update',
                 state_lifetime='offloaded during generation, loaded before training; stream gradients retained on GPU',
                 engine_contract={'max_num_seqs':self.cap,'max_num_batched_tokens':self.max_batched_tokens,'block_size':16,'gpu_memory_utilization':.85,'max_model_len':self.be.model_seq_cap,'runtime_reserve_bytes':self.runtime_bytes},
                 generation_resident_peak_bytes=self.be.state['rollout_weights_per_rank']+self.runtime_bytes+d['kv_peak_bytes'],
                 engine_peak_scheduled_tokens=max(r.peak_scheduled_tokens for r in self.replicas),
                 scope='conditional existing-workload method adaptation; no reward measured' if self.bank is None else 'conditional existing-workload method adaptation with measured CPU rewards')
        if self.bank is not None and all(row['status']=='MODELLED_ZERO' for row in self.bank.values()):
            d['modelled_reward_service_seconds']=d.pop('measured_reward_service_seconds')
            d['scope']='conditional math workload simulation; reward service modelled as zero, not measured'
        if self.cpu_timeline is None:d['seconds']=max(d['seconds'],reward_end)
        else:
            # Outstanding stateless CPU rewards continue across actor updates.
            # Preserve their FIFO reservations; do not invent an iteration
            # barrier merely to bill cancelled candidates. Drain once at the
            # terminal closed-window boundary, with every service still paid.
            start=self.cpu_timeline['clock'];d['period_offset_seconds']=start
            self.cpu_timeline['ready']=[start+t for t in self.cpu_ready]
            self.cpu_timeline['round_robin']=self.cpu_rr;self.cpu_timeline['clock']=start+d['seconds']
            d['cpu_queue_persistence']=True;d['outstanding_reward_after_publish_seconds']=max(0.,reward_end-d['seconds'])
        assert d['generation_resident_peak_bytes']<=self.be.capacity*.85+1
        assert d['engine_peak_scheduled_tokens']<=self.max_batched_tokens
        if self.f2_engine:
            d['engine_contract'].update(max_num_batched_tokens=None,prefill='f2-style whole-request FIFO; no mixed or partial-prefix prefill')
            d['cost_contract']=self.be.op.baseline_contract_identity
        return d


def independent_full_audit(path,be):
    """Reconstruct finite queues, state traffic and phase readiness from files."""
    import json
    rows=[];global_cpu=defaultdict(list);last_publication=0.
    for f in sorted(path.glob('step-*.json')):
        d=json.loads(f.read_text());reqs={r['rid']:r for r in d['requests']};tasks=d['reward_tasks']
        assert len({x['request'] for x in tasks})==len(tasks)==d['submitted_reward_samples']
        for task in tasks:
            assert task['submitted']>=reqs[task['request']]['finish_at']-1e-8
            assert task['start']>=task['submitted']-1e-8
            offset=d.get('period_offset_seconds',0.)
            global_cpu[task['worker']].append((offset+task['start'],offset+task['end']))
        for worker in range(d['cpu_reward_workers']):
            spans=sorted((x['start'],x['end']) for x in tasks if x['worker']==worker)
            assert all(a[1]<=b[0]+1e-8 for a,b in zip(spans,spans[1:]))
        transfers=d['host_transfers'];by=defaultdict(list)
        for x in transfers:
            assert math.isclose(x['end']-x['start'],x['bytes']/d['host_bandwidth'][x['direction']],abs_tol=1e-8)
            by[x['node']].append(x)
        for spans in by.values():
            spans.sort(key=lambda x:x['start']);assert all(a['end']<=b['start']+1e-8 for a,b in zip(spans,spans[1:]))
        loads={r:x for x in transfers if x['event']=='TRAIN_ACTOR_MASTER_REFERENCE_LOAD' for r in x['ranks']}
        off={r:x for x in transfers if x['event']=='ROLLOUT_WEIGHTS_OFFLOAD' for r in x['ranks']}
        assert set(loads)==set(off)==set(range(be.world))
        for r in loads:assert loads[r]['start']>=off[r]['end']-1e-8
        for rep in d['replicas']:
            for r in range(rep['replica']*be.p.rollout_tp,(rep['replica']+1)*be.p.rollout_tp):assert all(x[1]<=off[r]['start']+1e-8 for x in rep['segments'])
        for task in d['training_tasks']:
            assert all(task['start']>=reqs[r]['reward_at']-1e-8 for r in task['requests'])
            assert all(task['start']>=loads[r]['end']-1e-8 for r in range(task['worker']*be.p.train_tp,(task['worker']+1)*be.p.train_tp))
        assert sum(x['bytes'] for x in transfers if x['direction']=='H2D')==sum(x['bytes'] for x in transfers if x['direction']=='D2H')
        assert d['seconds']>=max([x['end'] for x in transfers]+([] if d.get('cpu_queue_persistence') else [x['end'] for x in tasks])+[0.])-1e-8
        last_publication=max(last_publication,d.get('period_offset_seconds',0.)+d['seconds'])
        rows.append({'version':d['version'],'reward_samples':len(tasks),'host_intervals':len(transfers),'status':'PASS'})
    for spans in global_cpu.values():
        spans.sort();assert all(a[1]<=b[0]+1e-8 for a,b in zip(spans,spans[1:]))
    summary=json.loads((path/'summary.json').read_text())
    if 'terminal_reward_drain_seconds' in summary:
        end=max([last_publication]+[x[1] for spans in global_cpu.values() for x in spans])
        assert math.isclose(end,summary['total_seconds'],abs_tol=1e-7)
    return {'status':'PASS','steps':rows,'scope':'finite reward queues across updates, state timing/roles, closed bytes and actual callback dependency'}
