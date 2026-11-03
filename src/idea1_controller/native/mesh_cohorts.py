"""Runtime helper/endpoint selection using actual mesh observations.

All original role, memory, KV and lease gates remain in the base protocol.
The optional policy never reads a pending item's response_length or EOS.
"""
from statistics import median
from .inference_cohorts import CohortKernel
from ..mesh_observation import MeshObservation
from ..inference_groups import inference_ranks, kv_reshard_moves


class MeshCohortKernel(CohortKernel):
    def observe_mesh(self,raw,tick_ps):
        self.mesh_view=MeshObservation.parse(raw,int(tick_ps))
        if not hasattr(self,'mesh_decision_log'):self.mesh_decision_log=[]

    def _live_mesh(self):
        view=getattr(self,'mesh_view',None)
        return view if getattr(self,'mesh_decisions_enabled',False) and view is not None and view.fresh(self.tick) else None

    def _transport(self,view,src,dst,size):
        return view.transfer(int(src),int(dst),int(size),self.op.hw.d2d_bytes_per_ns_per_direction,self.op.hw.d2d_latency_ns)

    def _remaining(self,item,context=None):
        pool=self.helper_history.get(str(item['group_id']),())
        elapsed=0 if context is None else max(0,context-item['prompt_length'])
        rest=[r-elapsed for r in pool if r>elapsed]
        return median(rest) if len(rest)>=8 else max(64,min(2048,elapsed/4))

    def _pending_cost(self,src,dst,items,view):
        source,target=self.rep[src],self.rep[dst]
        size=sum(x['prompt_length']*4+64 for x in items)
        estimates=[]
        for a,b in zip(source.ranks,target.ranks):estimates.append(self._transport(view,a,b,size))
        for j in self.initial_inference_members[dst]:
            if j!=dst:
                for a,b in zip(target.ranks,self.rep[j].ranks):estimates.append(self._transport(view,a,b,size))
        # The real backend executes these transfers. This is a conservative
        # serial-action ranking, not an asserted exact communication duration.
        transport=sum(e['service_ns'] for e in estimates)+4*len(estimates)*size/self.op.hw.dram_bytes_per_ns
        waits=min(self._remaining(item,context) for item,context,_ in source.active)
        src_tp=2*len(self.inference_members[src]);dst_tp=2*len(self.initial_inference_members[dst])
        context=sum(x[1] for x in source.active)/len(source.active)
        wait=waits*self.mesh_catalog.decode(src_tp,len(source.active),context)
        lengths=[x['prompt_length'] for x in items]
        remaining=median(self._remaining(x) for x in items)
        future_context=sum(lengths)/len(lengths)+remaining/2
        source_cost=self.mesh_catalog.phase('prefill',src_tp,lengths)+remaining*self.mesh_catalog.decode(src_tp,min(source.cap,len(items)),future_context)
        target_cost=self.mesh_catalog.phase('prefill',dst_tp,lengths)+remaining*self.mesh_catalog.decode(dst_tp,len(items),future_context)
        gain=wait+source_cost-target_cost-transport
        return dict(gain_ns=gain,transport_ns=transport,queue_penalty_ns=sum(e['queue_penalty_ns'] for e in estimates),
            source_wait_ns=wait,source_service_ns=source_cost,target_service_ns=target_cost,
            paths=[e['path'] for e in estimates],scope='forecast services and current committed mesh; no future outputs')

    def pending_choice(self,w):
        view=self._live_mesh()
        if view is None:return super().pending_choice(w)
        if not getattr(self,'pending_enabled',False) or getattr(self,'paired_tp1',False):return None
        if self.inference_owner[w]!=w or w in self.services or self.pending_lease(w):return None
        source=self.rep[w]
        if not source.requests or len(source.active)<source.cap or source.pending is not None:return None
        candidates=[];roles=self.logical_roles;old=self.inference_members[w]
        for dst,members in self.initial_inference_members.items():
            if dst==w or set(old)&set(members) or len(members)<len(old):continue
            if any(self.inference_owner[j]!=dst for j in members):continue
            if any(self.inference_busy(j) or self.leased(j) or j in self.services or j in self.trained or not self.waiting[j] for j in members):continue
            def allowed(j):return roles[j]=='ROLLOUT' or (getattr(self,'pending_data_wait',False) and roles[j]=='TRAINING' and self.stages[j]=='data_wait' and self.region_of[j] not in self.region_started)
            if not all(allowed(j) for j in members):continue
            # Limit transfer by both catalogues; do not invent unsupported shape costs.
            count=min(source.cap,self.rep[dst].cap,len(source.requests))
            items=list(source.requests)[:count]
            if not items:continue
            prediction=self._pending_cost(w,dst,items,view)
            if prediction['gain_ns']>max(5000.,.05*prediction['transport_ns']):candidates.append((prediction['gain_ns'],self.same_domain(w,dst),-dst,dst,items,prediction))
        if not candidates:return None
        choice=max(candidates,key=lambda x:x[:3]);dst,items,prediction=choice[3:]
        self.mesh_decision_log.append(dict(kind='pending-proposed',tick_ps=self.tick,source=w,target=dst,sample_ids=[x['trajectory_id'] for x in items],prediction=prediction))
        return w,dst,items

    def inference_resize_choice(self,w,roles):
        view=self._live_mesh()
        if view is None:return super().inference_resize_choice(w,roles)
        old=self.inference_members.get(w,())
        if not old:return None
        targets=[g for g in self.elastic_groups if len(g)==2*len(old) and set(old)<=set(g)]
        original=self.elastic_groups;candidates=[]
        try:
            # Reuse the unchanged base eligibility and numerical-service gates
            # for every bounded neighbouring target, rather than just re-ranking
            # the distance-only winner after it has discarded other helpers.
            for target in targets:
                self.elastic_groups=(target,)
                choice=super().inference_resize_choice(w,roles)
                if choice is None:continue
                engine=self.rep[w]
                moves=kv_reshard_moves(engine.execution_ranks,inference_ranks(target),[x[1] for x in engine.active],self.op.model)
                transport=0.;queue=0.;paths=[]
                for move in moves:
                    size=move['bytes'];transport+=size*4/self.op.hw.dram_bytes_per_ns
                    if move['src']!=move['dst']:
                        estimate=self._transport(view,move['src'],move['dst'],size)
                        transport+=estimate['service_ns'];queue+=estimate['queue_penalty_ns'];paths.append(estimate['path'])
                gain=choice['predicted_steps']*choice['gain_per_step_ns']-transport
                if gain>max(5000.,.05*transport):
                    choice=dict(choice,predicted_net_ns=gain,predicted_migration_ns=transport,
                        mesh_observation_tick_ps=view.tick_ps,mesh_queue_penalty_ns=queue,mesh_paths=paths,
                        prediction_scope='base legal TP expansion plus observed shared network; bounded heuristic')
                    candidates.append(choice)
        finally:self.elastic_groups=original
        if not candidates:return None
        choice=max(candidates,key=lambda x:(x['predicted_net_ns'],tuple(-j for j in x['target'])))
        self.mesh_decision_log.append(dict(kind='resize-proposed',tick_ps=self.tick,source=w,prediction=choice))
        return choice
