"""Native actor-view execution keeps the same global optimizer and publication."""
from .system_kernel import SystemKernel
from .system_adapter import SystemAdapter,QuorumWorker,QuorumExecutor
from .view_attention import ViewAttention
from ..actor_views import ActorViewCatalog

from .region_training_help import RegionalTrainingHelp,RegionalTrainingCooperator

class ViewKernel(RegionalTrainingHelp,SystemKernel):
    def same_domain(self,a,b):
        pools=self.resource_layout.get('helper_domains')
        if pools:return any(a in ws and b in ws for ws in pools.values())
        return super().same_domain(a,b)
    def domain_drained(self,w):
        pools=self.resource_layout.get('helper_domains')
        if pools:
            ws=next(ws for ws in pools.values() if w in ws)
            return all(self.empty(j) and not self.leased(j) for j in ws)
        return super().domain_drained(w)
    def roles(self):
        labels=super().roles()
        for w,stage in self.stages.items():
            if stage=='training_ticket':labels[w]='training'
        return labels
    def init_actor_views(self):self.actor_views=ActorViewCatalog(self.op);assert self.actor_views.validate()
    def estimate_locality(self,w,helper):
        x=super().estimate_locality(w,helper)
        if x is None:return None
        from attention_tp8_program import attention_weight_bytes_per_helper
        size=attention_weight_bytes_per_helper(self.op);bw=self.op.hw.d2d_bytes_per_ns_per_direction;hops=max(1,x['hops'])
        saved=size/bw*hops+self.op.vec(0,size,0,'estimate_migration_read').ns+self.op.vec(0,0,size,'estimate_migration_write').ns
        x['predicted_migration_ns']=max(0,x['predicted_migration_ns']-saved)
        x['predicted_net_ns']+=saved;x['accepted']=x['predicted_net_ns']>max(1000,x['predicted_migration_ns']*.05)
        x['attention_weight_storage']='local immutable actor subview';return x

class ViewWorker(QuorumWorker):
    def rollout_reply(self,intent):
        c=self.c;k=c.kernel;w=c.w;it=c.req['iteration'];k.stages[w]='generation';before=len(k.migrations)
        result=ViewAttention.step(k,w,it);helper=intent['details']['selected_helper']
        if helper is None:assert len(k.migrations)==before
        else:assert len(k.migrations)==before+1 and k.migrations[-1]['helper']==helper
        if result.get('finish'):return k.wait(w)
        for step in result['steps']:
            if step.get('kind')=='controller_rpc' and step['event']['phase']=='finish':step['event']['phase']='step'
        return result

class ViewExecutor(QuorumExecutor):
    def __init__(self,c):self.c=c;self.workers={w:ViewWorker(c,w) for w in c.kernel.rep}

class ViewAdapter(SystemAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Executor binding precedes activate and every native command.
        from dataclasses import replace
        self.controller.runtime=replace(self.controller.runtime,executor=ViewExecutor(self.context),training=RegionalTrainingCooperator(self.context))
    def on_native_complete(self,result):
        assert result['native_exit_status']==0 and self.kernel.finished_all() and not self.kernel.tx
        assert self.kernel.actor_views.validate()
        self.kernel.actor_views.publish_after_native_update()
        super().on_native_complete(result)
        import json
        c=self.context;path=c.output/'framework-summary.json';summary=json.loads(path.read_text())
        summary.update(training_batch_stealing='ticketed TP2/4/8 donors to restored TP2 pairs; sequential capacity-bounded microbatches',step_plan_id=c.plan.step_plan_id,placement_id=c.plan.placement.plan_id,assignment_id=c.plan.assignment.plan_id,
            prediction_digest=c.plan.assignment.prediction_digest,capacity_proof=c.plan.feasibility_proof_digest,
            gradient_route_digest=c.plan.gradient_routing_digest,parameter_schema=k.coordinates.schema_digest if (k:=self.kernel) else None,
            actor_view_version=k.actor_views.version,actor_view_generation=k.actor_views.generation,
            handoff='device drain -> DATA_WAIT -> complete region data -> wake TP quorum -> preparation/backward',
            planning_execution_boundary='SystemPlanner chooses immutable manifest; SelectedManifestPlanner binds it to native facts')
        path.write_text(json.dumps(summary,indent=2)+'\n')
