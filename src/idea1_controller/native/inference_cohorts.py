"""Actual TP2/4/8 generation with independent TP2 optimizer owners.

Every physical pair executes the same TPp program, with a native full-TP
completion quorum before the owning request engine observes any EOS.
"""
from collections import deque
from dataclasses import replace
import gzip,json
from online_rollout import Replica
from rl_xy.qwen3_sequence import local
from rl_xy.qwen3_cost import Cost
from ..inference_groups import inference_ranks,validate_partition,kv_reshard_moves
from .view_system import ViewKernel,ViewWorker,ViewAdapter
from .context import digest
from .cohort_pending import CohortPending


class CohortReplica(Replica):
    def next(self,iteration):
        physical=self.ranks
        self.ranks=self.execution_ranks
        try:return super().next(iteration)
        finally:self.ranks=physical


class CohortKernel(CohortPending,ViewKernel):
    def init_inference(self,groups,caps,communicators):
        validate_partition(groups)
        self.inference_members={int(r):tuple(ws) for r,ws in groups.items()}
        self.inference_owner={w:r for r,ws in self.inference_members.items() for w in ws}
        self.initial_inference_members=dict(self.inference_members);self.initial_inference_owner=dict(self.inference_owner)
        self.inference_resizes=[];self.inference_releases=[]
        self.inference_role_events=[];self.inference_saved_roles={}
        self.pending_cohort_leases={};self.inference_input_broadcasts={}
        self.inference_programs={};self.inference_program_events=[];self.inference_issued={}
        self.inference_communicators=communicators;self.initial_inference_communicators=dict(communicators)
        self.inference_epoch={r:0 for r in self.inference_members}
        self.inference_tag=480_000_000
        override=getattr(self,'inference_queue_override',None)
        bygroup={}
        for x in self.items.values():bygroup.setdefault(str(x['group_id']),[]).append(x)
        if override is not None:
            from collections import Counter
            assert Counter(g for gs in override.values() for g in gs)==Counter(bygroup.keys())
            assert set(map(int,override))==set(self.inference_members)
        for r,ws in self.inference_members.items():
            items=([x for g in override[str(r)] for x in bygroup[g]] if override is not None else [x for w in ws for x in self.rep[w].requests])
            for w in ws:self.rep[w].requests.clear()
            old=self.rep[r]
            if getattr(self,'paired_tp1',False):
                from .paired_tp1 import PairedTP1Replica
                def tag():
                    self.inference_tag+=1
                    return self.inference_tag
                engine=PairedTP1Replica(self.op,items,old.ranks,int(caps[str(r)]),tag)
            else:engine=CohortReplica(self.op,items,old.ranks,int(caps[str(r)]),safe_interval=16)
            engine.execution_ranks=inference_ranks(ws);self.rep[r]=engine
        self.owners={x['trajectory_id']:w for w,rep in self.rep.items() for x in rep.requests}
        self.initial_owners=dict(self.owners)
        # Stream bodies retain exact cost/collective identity independently of
        # the mutable live source. Gzip keeps long full-run evidence bounded.
        self.inference_log=gzip.open(self.evidence_output/'inference-programs.jsonl.gz','wt')

    def inference_busy(self,w):
        r=self.inference_owner[w];engine=self.rep[r]
        return bool(engine.requests or engine.active or engine.pending)

    def _observe(self,w):
        super()._observe(w)
        for event in reversed(self.inference_program_events):
            if event['group']==w:
                if 'end_tick' not in event:event['end_tick']=self.tick
                break
        r=self.inference_owner[w]
        if w==r and not self.inference_busy(w):
            self.cohort_drained=getattr(self,'cohort_drained',set())
            if r not in self.cohort_drained:
                self.cohort_drained.add(r)
                self.cohort_release_pending=getattr(self,'cohort_release_pending',{})
                self.cohort_release_pending[w]=[j for j in self.inference_members[r] if j!=w or j in self.inference_saved_roles]
                members=self.inference_members[r]
                if members!=self.initial_inference_members[r]:
                    self.inference_releases.append(dict(root=r,tick=self.tick,members=members,reason='all requests EOS; no live KV to restore'))
                    for j in members:self.inference_owner[j]=self.initial_inference_owner[j]
                    self.inference_members[r]=self.initial_inference_members[r]
                    self.rep[r].execution_ranks=inference_ranks(self.inference_members[r])
                    self.inference_communicators[r]=self.initial_inference_communicators[r]

    def empty(self,w):
        return not self.inference_busy(w)

    def leased(self,w):
        return self.pending_lease(w) or self.inference_busy(w) or super().leased(w)

    def roles(self):
        labels=super().roles()
        for w in self.rep:
            if self.inference_busy(w):labels[w]='helper' if self.inference_owner[w]!=self.initial_inference_owner[w] else 'own'
        return labels

    def inference_resize_choice(self,w,roles):
        if not getattr(self,'elastic_enabled',False) or self.inference_owner[w]!=w:return None
        engine=self.rep[w];old=self.inference_members[w]
        if engine.requests or not engine.active or engine.pending or len(old)>=4:return None
        from bisect import bisect_right
        residual=[]
        for item,context,_ in engine.active:
            pool=self.helper_history.get(str(item['group_id']),());elapsed=context-item['prompt_length'];i=bisect_right(pool,elapsed);n=len(pool)-i
            residual.append(max(1,pool[i+9*(n-1)//10]-elapsed) if n>=32 else max(64,min(2048,elapsed//4)))
        horizon=max(residual);batch=len(engine.active);q,e=divmod(sum(x[1] for x in engine.active),batch);contexts=tuple(q+(i<e) for i in range(batch))
        def service(tp):
            blocks=self.op.forward((1,)*batch,tp,'generation',contexts)
            local_ns=sum(b.cost.ns for b in blocks)
            network=sum(2*(tp-1)*(self.op.hw.d2d_latency_ns+b.payload/tp/self.op.hw.d2d_bytes_per_ns_per_direction) for b in blocks if b.collective)
            return local_ns+network
        gain=service(len(old)*2)-service(len(old)*4)
        if gain<=0:return None
        choices=[]
        for target in self.elastic_groups:
            if len(target)!=2*len(old) or not set(old)<=set(target):continue
            newcomers=set(target)-set(old)
            def eligible(j):
                role_ok=roles[j]=='ROLLOUT' or (getattr(self,'elastic_data_wait',False) and roles[j]=='TRAINING' and self.stages[j]=='data_wait' and self.region_of[j] not in self.region_started)
                return role_ok and not self.inference_busy(j) and not self.pending_lease(j) and j not in self.services and j not in self.trained and self.waiting[j]
            if not all(eligible(j) for j in newcomers):continue
            moves=kv_reshard_moves(engine.execution_ranks,inference_ranks(target),[x[1] for x in engine.active],self.op.model)
            transport=0.
            for m in moves:
                hops=abs(m['src']//6-m['dst']//6)+abs(m['src']%6-m['dst']%6)
                transport+=m['bytes']*4/self.op.hw.dram_bytes_per_ns
                if hops:transport+=m['bytes']/self.op.hw.d2d_bytes_per_ns_per_direction*hops+2*hops*self.op.hw.d2d_latency_ns
            score=horizon*gain-transport
            if score>max(5000.,transport*.05):choices.append(dict(target=target,predicted_net_ns=score,predicted_migration_ns=transport,predicted_steps=horizon,gain_per_step_ns=gain,
                role_before={str(j):dict(role=roles[j],stage=self.stages[j]) for j in newcomers}))
        return max(choices,key=lambda x:(x['predicted_net_ns'],tuple(-j for j in x['target']))) if choices else None

    def resize_steps(self,r,choice):
        old=self.inference_members[r];new=tuple(choice['target']);engine=self.rep[r]
        assert set(old)<set(new) and all(not self.inference_busy(j) and not self.pending_lease(j) for j in set(new)-set(old))
        moves=kv_reshard_moves(engine.execution_ranks,inference_ranks(new),[x[1] for x in engine.active],self.op.model)
        old_ranks=set(engine.execution_ranks);steps=[]
        def retired(rank):return f'KV-reshape-{r}-{self.inference_epoch[r]+1}-retired-{rank}'
        for index,move in enumerate(moves):
            src,dst,size=move['src'],move['dst'],move['bytes']
            if dst in old_ranks and dst!=src:
                # Source-list order alone is not a native dependency: an
                # uninvolved rank skips preceding P2P operations. This gate
                # prevents early arrival/write into an unretired old cache.
                steps.append(dict(kind='gate_wait',key=retired(dst),threshold=1))
            if src==dst:steps.append(local(self.op.vec(0,size,size,'inference_KV_inplace_repack'),ranks=[src]))
            else:
                self.inference_tag+=1
                steps.extend([local(self.op.vec(0,size,size,'inference_KV_stream_pack'),ranks=[src]),
                    local(self.op.vec(0,size,0,'inference_KV_read'),ranks=[src]),
                    dict(kind='p2p_direct',src=src,dst=dst,bytes=size,tag=self.inference_tag),
                    local(self.op.vec(0,0,size,'inference_KV_write'),ranks=[dst]),
                    local(self.op.vec(0,size,size,'inference_KV_stream_unpack'),ranks=[dst])])
            if index==len(moves)-1 or moves[index+1]['src']!=src:
                steps.append(dict(kind='gate_signal',key=retired(src),rank=src))
        size=len(engine.active)*68
        for w in sorted(set(new)-set(old)):
            for dst in self.rep[w].ranks:
                src=self.rep[r].ranks[0];self.inference_tag+=1
                steps.extend([local(self.op.vec(0,size,0,'inference_metadata_read'),ranks=[src]),dict(kind='p2p_direct',src=src,dst=dst,bytes=size,tag=self.inference_tag),local(self.op.vec(0,0,size,'inference_metadata_write'),ranks=[dst])])
        for w in new:self.inference_owner[w]=r
        self.inference_members[r]=new;engine.execution_ranks=inference_ranks(new)
        self.inference_communicators[r]=self.elastic_communicators[new]
        self.inference_resizes.append(dict(root=r,tick=self.tick,old_members=old,new_members=new,old_tp=2*len(old),new_tp=2*len(new),
            contexts=[x[1] for x in engine.active],active_ids=[x[0]['trajectory_id'] for x in engine.active],moves=moves,
            metadata_bytes_per_new_die=size,prediction=choice,migration_program_sha256=digest(steps),temporary_KV_pack_bytes=2*1024**2,
            storage_protocol='old destination outgoing data retires before overwrite; remote sends precede local compaction; per-source native retirement gates',
            actor_storage='local canonical TP2 superset; no new weight allocation or weight transfer'))
        return steps

    def cohort_reply(self,w,it,choice=None):
        if self.pending_lease(w):return self.wait(w)
        r=self.inference_owner[w];ws=self.inference_members[r];engine=self.rep[r]
        self.stages[w]='generation'
        if not self.inference_busy(w):return self.wait(w)
        if w==r:
            if not (engine.requests or engine.active or engine.pending):return self.wait(w)
            migration=self.resize_steps(r,choice) if choice else []
            input_broadcast,broadcast_ids=self.pending_input_fanout(r)
            ws=self.inference_members[r]
            # Context applied observe() before proposing; one leader owns EOS.
            reply=engine.next(it);self.inference_epoch[r]+=1;epoch=self.inference_epoch[r]
            steps=reply['steps']
            for s in steps:
                if s['kind']=='AR':s['group']=self.inference_communicators[r]
                if s['kind']=='p2p':
                    # The physical TP2 sequence owners have different default
                    # communicator IDs. Full inference has cross-owner head
                    # gather/broadcast, so use one explicit shared message tag.
                    self.inference_tag+=1
                    if self.inference_tag>=500_000_000:raise ValueError('inference tag space exhausted')
                    s.update(kind='p2p_direct',tag=self.inference_tag)
            event=dict(group=r,epoch=epoch,tp=getattr(engine,'inference_tp',len(engine.execution_ranks)),members=ws,
                ranks=engine.execution_ranks,tick=self.tick,repeat=reply.get('repeat',1),
                kind=engine.pending[0] if engine.pending else 'empty',
                active_ids=[x[0]['trajectory_id'] for x in engine.active],
                actual_contexts=[x[1] for x in engine.active],
                pending_ids=[x['trajectory_id'] for x in engine.pending[1]] if engine.pending and engine.pending[0]=='prefill' else [],
                input_broadcast_ids=broadcast_ids,
                program_sha256=digest(dict(steps=steps,repeat=reply.get('repeat',1))))
            if getattr(self,'paired_tp1',False):event.update(engine.event)
            self.inference_program_events.append(event)
            self.inference_log.write(json.dumps(dict(event=event,steps=steps,migration=migration,input_broadcast=input_broadcast),separators=(',',':'))+'\n');self.inference_log.flush()
            self.inference_programs[r]=dict(steps=steps,repeat=reply.get('repeat',1),epoch=epoch,migration=migration,input_broadcast=input_broadcast)
        program=self.inference_programs.get(r)
        if program is None or self.inference_issued.get(w)==(r,program['epoch']):return self.wait(w)
        self.inference_issued[w]=(r,program['epoch']);epoch=program['epoch']
        # Repeat applies only to the operator body, never repeated gate signals.
        begin=f'infer-{r}-{epoch}-begin';end=f'infer-{r}-{epoch}-end'
        prefix=[]
        if w==r:
            prefix=[self.wake(j,self.rep[w].ranks[0]) for j in ws if j!=w and self.waiting[j]]
        body=[dict(kind='gate_signal',key=begin),dict(kind='gate_wait',key=begin,threshold=len(engine.execution_ranks))]
        if program['input_broadcast']:
            gate=begin+'-input-broadcast'
            body+=program['input_broadcast']+[dict(kind='gate_signal',key=gate),dict(kind='gate_wait',key=gate,threshold=len(engine.execution_ranks))]
        if program['migration']:
            gate=begin+'-migration'
            body+=program['migration']+[dict(kind='gate_signal',key=gate),dict(kind='gate_wait',key=gate,threshold=len(engine.execution_ranks))]
        # Sequence engine supports nested repeated bodies through controller
        # phases: all atoms issue the same body with repeat, then rendezvous.
        self.cohort_pending_finish=getattr(self,'cohort_pending_finish',{})
        self.cohort_pending_finish[w]=(r,epoch)
        if prefix:
            self.cohort_deferred=getattr(self,'cohort_deferred',{});self.cohort_deferred[w]=dict(steps=body,repeat=1)
            return dict(steps=prefix+[local(Cost(compute=1000)),self.rpc(w,'cohort_begin')])
        return dict(steps=body+[self.rpc(w,'cohort_body')])

    def cohort_control(self,w,phase):
        r,epoch=self.cohort_pending_finish[w];program=self.inference_programs[r]
        assert program['epoch']==epoch
        if phase=='cohort_begin':
            x=self.cohort_deferred.pop(w);return dict(steps=x['steps']+[self.rpc(w,'cohort_body')])
        if phase=='cohort_body':
            self.cohort_body_issued=getattr(self,'cohort_body_issued',set());self.cohort_body_issued.add(w)
            return dict(steps=program['steps'],repeat=program['repeat'])
        if phase=='cohort_end':
            key=f'infer-{r}-{epoch}-end'
            return dict(steps=[dict(kind='gate_signal',key=key),dict(kind='gate_wait',key=key,threshold=len(self.rep[r].execution_ranks)),self.rpc(w,'step')])
        raise ValueError(phase)


class CohortWorker(ViewWorker):
    def rollout_reply(self,intent):
        c=self.c;k=c.kernel;choice=intent['details'].get('resize')
        if choice:
            for j,saved in choice.get('role_before',{}).items():
                j=int(j)
                if saved['role']=='TRAINING':
                    assert c.roles[j]=='TRAINING' and k.stages[j]=='data_wait' and k.region_of[j] not in k.region_started
                    k.inference_saved_roles[j]=saved;c.roles[j]='ROLLOUT'
                    k.inference_role_events.append(dict(worker=j,tick=k.tick,old='TRAINING',new='ROLLOUT',reason='DATA_WAIT only; no teacher/reference/gradient active'))
        return k.cohort_reply(c.w,c.req['iteration'],choice)
    def execute(self,intent):
        c=self.c
        if c.is_step() and c.w in getattr(c.kernel,'cohort_release_pending',{}):
            members=c.kernel.cohort_release_pending.pop(c.w)
            restored_self=False
            for j in members:
                saved=c.kernel.inference_saved_roles.pop(j,None)
                if saved:
                    restored_self=restored_self or j==c.w
                    c.roles[j]=saved['role'];c.kernel.stages[j]=saved['stage']
                    c.kernel.inference_role_events.append(dict(worker=j,tick=c.kernel.tick,old='ROLLOUT',new=saved['role'],reason='full inference completion quorum; no live KV'))
            steps=[c.kernel.wake(j,c.kernel.rep[c.w].ranks[0]) for j in members if c.kernel.waiting[j]]
            if steps or restored_self:return dict(steps=steps+[local(Cost(compute=1000)),c.kernel.rpc(c.w,'step')])
        if intent['op']=='cohort_wait':return c.kernel.wait(c.w)
        if intent['op']=='native_control' and c.phase.startswith('cohort_'):return c.kernel.cohort_control(c.w,c.phase)
        if intent['op']=='native_control' and c.phase=='arrive' and c.w in getattr(c.kernel,'cohort_body_issued',set()):
            c.kernel.cohort_body_issued.remove(c.w)
            return c.kernel.cohort_control(c.w,'cohort_end')
        return super().execute(intent)


from .selected_context import SelectedContext,SelectedManifestPlanner
from .complete import CompleteCapabilities
from .region_adapter import PlannedRegion
from .policies import RoleScheduler,Eligibility,ResourceManager,LegacyOrderArbiter
from .region_training_help import RegionalTrainingCooperator
from .backend import NativeExecutor,GradientCoordinator,OptimizerOwner,WeightPublisher
from .context import DisabledPlanningPolicy,StateCatalog
from ..contracts import InferenceInstance,RequestView,MeshView,ModelContract
from ..orchestration import HierarchicalController,PlanningServices,RuntimeServices


class CohortContext(SelectedContext):
    def _fixed_plan(self):
        p=super()._fixed_plan();k=self.kernel
        inference=tuple(InferenceInstance('I'+str(r),inference_ranks(ws),len(ws)*2) for r,ws in k.inference_members.items())
        if getattr(k,'paired_tp1',False):inference=tuple(InferenceInstance('I'+str(d),(d,),1) for d in range(54))
        candidate=replace(p.placement.candidate,rollout_tp=max(x.tp for x in inference),rollout_replica_count=len(inference))
        placement=replace(p.placement,inference=inference,candidate=candidate,
            drain_dependencies={f'R{r}':tuple('I'+str(x) for x in sorted({k.inference_owner[w] for w in ws})) for r,ws in k.region_members.items()})
        if getattr(k,'paired_tp1',False):
            initial={t:'I'+str(d) for e in k.rep.values() for t,d in e.request_rank.items()}
            queues={'I'+str(d):tuple(t for e in k.rep.values() for t,r in e.request_rank.items() if r==d) for d in range(54)}
            assignment=replace(p.assignment,initial_instance_by_request=initial,ordered_requests_by_instance=queues)
            placement=replace(placement,drain_dependencies={f'R{r}':tuple('I'+str(d) for w in ws for d in (2*w,2*w+1)) for r,ws in k.region_members.items()})
            p=replace(p,assignment=assignment)
        else:
            p=replace(p,assignment=replace(p.assignment,ordered_requests_by_instance={x.instance_id:p.assignment.ordered_requests_by_instance[x.instance_id] for x in inference}))
        return replace(p,placement=placement,capability_report_digest='actual-inference-TP1-2-4-8')
    def snapshot(self):
        snap=super().snapshot();k=self.kernel;dies=dict(snap.dies)
        for w in k.rep:
            if self.roles[w]=='ROLLOUT':
                for d in k.rep[w].ranks:dies[d]=replace(dies[d],execution_group_id='I'+str(d if getattr(k,'paired_tp1',False) else k.inference_owner[w]))
            pending=frozenset('cohort-pending:'+str(t) for t,ws in k.pending_cohort_leases.items() if w in ws)
            if pending:
                for d in k.rep[w].ranks:dies[d]=replace(dies[d],lease_ids=dies[d].lease_ids|pending)
        return replace(snap,dies=dies)
    def propose(self,op,*args,**kwargs):
        proposal=super().propose(op,*args,**kwargs)
        if op=='rollout_step':
            choice=kwargs.get('resize');ws=choice['target'] if choice else self.kernel.inference_members[self.kernel.inference_owner[self.w]]
            proposal=replace(proposal,die_ids=inference_ranks(ws))
        elif op=='pending_transfer':
            src,dst=kwargs['src'],kwargs['dst']
            ws=tuple(sorted(set(self.kernel.inference_members[src])|set(self.kernel.initial_inference_members[dst])))
            proposal=replace(proposal,die_ids=inference_ranks(ws))
        return proposal


class CohortCapabilities(CompleteCapabilities):
    def validate(self,placement,model):
        return self.report(all(x.tp in (1,2,4,8) for x in placement.inference) and all(x.tp in (2,4,8) for x in placement.training))


class CohortRollout:
    def __init__(self,c):self.c=c
    def propose_pending_takeovers(self,instance_id,snapshot):
        c=self.c
        if not c.is_step() or c.roles[c.w]!='ROLLOUT' or instance_id!='I'+str(c.kernel.initial_inference_owner[c.w]):return ()
        choice=c.kernel.pending_choice(c.w)
        if choice is None:return ()
        src,dst,items=choice
        return (c.propose('pending_transfer',5,'TAKE_PENDING_RESPONSE',peer=dst,tids=[x['trajectory_id'] for x in items],src=src,dst=dst),)
    def propose_attention_help(self,instance_id,snapshot):
        c=self.c
        if not c.is_step() or c.w in c.kernel.services or c.roles[c.w]!='ROLLOUT':return ()
        expected=2*c.w if getattr(c.kernel,'paired_tp1',False) else c.kernel.initial_inference_owner[c.w]
        if instance_id!='I'+str(expected):return ()
        choice=c.kernel.inference_resize_choice(c.w,c.roles)
        return (c.propose('rollout_step',50,'HELP_ATTENTION' if choice else 'CONTINUE_OWN',selected_helper=None,resize=choice),)


class CohortExecutor(NativeExecutor):
    def __init__(self,c):self.c=c;self.workers={w:CohortWorker(c,w) for w in c.kernel.rep}


class CohortRegion(PlannedRegion):
    def propose(self,region_id,snapshot):
        result=super().propose(region_id,snapshot);c=self.c
        # A DATA_WAIT region can temporarily lend one member to inference.
        # An unborrowed TRAINING member must keep a legal wait action when
        # a record arrival wakes it before the borrowed member returns.
        if region_id==c.region_id(c.w) and c.is_step() and c.w not in c.kernel.services and c.roles[c.w]=='TRAINING' and not c.region_eligible(c.w):
            return tuple(result)+(c.propose('cohort_wait',60),)
        return result


class CohortAdapter(ViewAdapter):
    def __init__(self,kernel,policy,capacity,model_id,output):
        self.kernel=kernel;kernel.helper_completion_quorums=[]
        self.context=c=CohortContext(kernel,policy,capacity,model_id,output)
        kernel.logical_roles=c.roles
        fixed=SelectedManifestPlanner(c);caps=CohortCapabilities(c);disabled=DisabledPlanningPolicy()
        planning=PlanningServices(fixed,fixed,disabled,disabled,disabled,disabled,disabled,disabled,disabled,caps)
        runtime=RuntimeServices(RoleScheduler(c),CohortRegion(c),CohortRollout(c),RegionalTrainingCooperator(c),Eligibility(c),LegacyOrderArbiter(c),
            c,c,c,StateCatalog(c),c,GradientCoordinator(c),OptimizerOwner(c),WeightPublisher(c),CohortExecutor(c),ResourceManager(c))
        self.controller=HierarchicalController(fixed,planning,runtime)
        requests=[RequestView(x['trajectory_id'],str(x['group_id']),str(x['group_id']),x['prompt_length'],8192,'recorded-decoding',x['trajectory_id']) for x in kernel.items.values()]
        mesh=MeshView(tuple(range(54)),{i:(i//6,i%6) for i in range(54)},{i:capacity['capacity_bytes'] for i in range(54)}, {}, {})
        model=ModelContract(model_id,kernel.coordinates.schema_digest,(1,2,4,8),(2,4,8),'frozen-BF16-and-FP32','native-inference-cohorts')
        plan=self.controller.prepare_step(requests,model,mesh,c.plan.algorithm);self.controller.install(plan);self.native_plan_bound=False

    def on_native_complete(self,result):
        self.kernel.inference_log.close()
        super().on_native_complete(result)
        p=self.context.output/'framework-summary.json';v=json.loads(p.read_text())
        v.update(pending_takeover_enabled=getattr(self.kernel,'pending_enabled',False),elastic_tp_enabled=getattr(self.kernel,'elastic_enabled',False),helper_kernel='actual inference TP1/2/4/8; training TP4/8 collaboration')
        p.write_text(json.dumps(v,indent=2)+'\n')
