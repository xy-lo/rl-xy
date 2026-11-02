"""Assembly of planned TP regions into the original hierarchical controller."""
import json
from dataclasses import replace
from .adapter import LayeredNativeAdapter
from .regions import RegionContext
from .context import FixedPlanner,Capabilities,DisabledPlanningPolicy,StateCatalog
from .policies import RoleScheduler,RegionController,RolloutCooperator,Eligibility,ResourceManager,LegacyOrderArbiter,sample_transfer_choice
from .backend import NativeExecutor,NativeWorker,GradientCoordinator,OptimizerOwner,WeightPublisher
from .region_attention import RegionAttention
from ..orchestration import HierarchicalController,PlanningServices,RuntimeServices
from ..contracts import ModelContract,MeshView,RequestView,Readiness

def attention_helper_choice(c):
    k=c.kernel;w=c.w;r=k.rep[w]
    tx=next((v for v in k.tx.values() if w in (v['source'],v['helper'])),None)
    if tx:return None
    if len(k.done)==len(k.ids) and not k.tx:return None
    if not r.requests and not r.active and r.pending is None:return None
    busy={x for v in k.tx.values() for x in (v['source'],v['helper'])}
    free=[j for j in k.rep if k.same_domain(j,w) and j!=w and k.waiting[j] and j not in busy
          and not k.rep[j].requests and not k.rep[j].active and k.rep[j].pending is None]
    free.sort(key=lambda j:(max(abs(s//6-d//6)+abs(s%6-d%6) for s,d in zip(r.ranks,k.rep[j].ranks)),j))
    can_decode=bool(r.active) and not r.requests and r.pending is None
    choices=[]
    if k.priority and free:
        for source,v in k.rep.items():
            if source in busy or not v.active or v.requests or v.pending is not None:continue
            for helper,h in k.rep.items():
                if not k.same_domain(helper,source) or helper==source or helper in busy or not k.waiting[helper]:continue
                if h.requests or h.active or h.pending is not None:continue
                estimate=k.estimate_locality(source,helper)
                if estimate['accepted']:choices.append((estimate['predicted_net_ns'],-source,-helper,source))
    chosen=max(choices)[3] if choices else None
    if w not in busy and free and can_decode and (not k.priority or w==chosen):
        choice=max((k.estimate_locality(w,j) for j in free),key=lambda x:x['predicted_net_ns'])
        if choice['accepted']:return choice['helper']
    return None


class RegionCapabilities(Capabilities):
    def report(self,good=True,reason=()):return replace(super().report(good,reason),heterogeneous_training_tp=True)
    def validate(self,placement,model):
        good=all(x.tp==2 for x in placement.inference) and all(x.tp in (2,4,8) for x in placement.training)
        return self.report(good,() if good else ('unsupported_native_region_tp',))

class PlannedRollout(RolloutCooperator):
    def propose_attention_help(self,instance_id,snapshot):
        c=self.c;k=c.kernel
        if instance_id!=f'I{c.w}' or not c.is_step() or c.w in k.services or c.roles[c.w]!='ROLLOUT':return ()
        helper=attention_helper_choice(c)
        return (c.propose('rollout_step',50,'HELP_ATTENTION' if helper is not None else 'CONTINUE_OWN',peer=helper,selected_helper=helper),)

class PlannedRegion(RegionController):
    def propose(self,region_id,snapshot):
        c=self.c;k=c.kernel
        if region_id!=c.region_id(c.w) or not c.is_step() or c.w in k.services:return ()
        result=[];choice=sample_transfer_choice(c)
        if choice:
            src,dst,tids=choice;result.append(c.propose('sample_transfer',10,peer=src if dst==c.w else dst,tids=tids,src=src,dst=dst))
        if c.roles[c.w]=='TRAINING' and c.region_eligible(c.w):result.append(c.propose('region_step',40))
        return tuple(result)

class NoTrainingSteal:
    def propose_unstarted_batch_takeovers(self,*args):return ()

class PlannedWorker(NativeWorker):
    def execute(self,intent):
        c=self.c;k=c.kernel;w=c.w;it=c.req['iteration'];op=intent['op']
        if op=='region_step':return k.region_step(w,it,c)
        if op=='rollout_step':
            k.stages[w]='generation';before=len(k.migrations)
            result=RegionAttention.step(k,w,it);helper=intent['details']['selected_helper']
            if helper is None:assert len(k.migrations)==before
            else:assert len(k.migrations)==before+1 and k.migrations[-1]['helper']==helper
            if result.get('finish'):return k.wait(w)
            for step in result['steps']:
                if step.get('kind')=='controller_rpc' and step['event']['phase']=='finish':step['event']['phase']='step'
            return result
        return super().execute(intent)

class PlannedExecutor(NativeExecutor):
    def __init__(self,c):self.c=c;self.workers={w:PlannedWorker(c,w) for w in c.kernel.rep}

class RegionAdapter(LayeredNativeAdapter):
    def __init__(self,kernel,policy,capacity,model_id,output):
        self.kernel=kernel;self.context=c=RegionContext(kernel,policy,capacity,model_id,output)
        fixed=FixedPlanner(c);caps=RegionCapabilities(c);disabled=DisabledPlanningPolicy()
        planning=PlanningServices(fixed,fixed,disabled,disabled,disabled,disabled,disabled,disabled,disabled,caps)
        runtime=RuntimeServices(RoleScheduler(c),PlannedRegion(c),PlannedRollout(c),NoTrainingSteal(),Eligibility(c),LegacyOrderArbiter(c),
            c,c,c,StateCatalog(c),c,GradientCoordinator(c),OptimizerOwner(c),WeightPublisher(c),PlannedExecutor(c),ResourceManager(c))
        self.controller=HierarchicalController(fixed,planning,runtime)
        requests=[RequestView(x['trajectory_id'],str(x['group_id']),str(x['group_id']),x['prompt_length'],8192,'recorded-decoding',x['trajectory_id']) for x in kernel.items.values()]
        mesh=MeshView(tuple(range(54)),{i:(i//6,i%6) for i in range(54)},{i:capacity['capacity_bytes'] for i in range(54)}, {}, {})
        model=ModelContract(model_id,'fixed-tp2-owners-native-region-views',(2,),(2,4,8),'frozen-BF16-and-FP32','native-regions')
        plan=self.controller.prepare_step(requests,model,mesh,c.plan.algorithm);self.controller.install(plan);self.native_plan_bound=False
    def on_native_complete(self,result):
        super().on_native_complete(result)
        p=self.context.output/'framework-summary.json';v=json.loads(p.read_text());v.update(fixed_placement=False,fixed_assignment=False,
            training_tp_by_region={str(r):len(xs) for r,xs in self.kernel.region_ranks.items()},training_batch_stealing=False,
            gradient_restore='explicit canonical TPp to original TP2 native optimizer',status='PASS_PLANNED_NATIVE_FRAMEWORK')
        p.write_text(json.dumps(v,indent=2)+'\n')
