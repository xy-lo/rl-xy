"""Optional PENDING policy over the published fixed-TP2 training backend."""
from .adapter import LayeredNativeAdapter,NativeKernel
from .pending import PendingMixin
from .context import NativeContext,FixedPlanner,Capabilities,DisabledPlanningPolicy,StateCatalog
from .policies import RoleScheduler,RegionController,RolloutCooperator,TrainingCooperator,Eligibility,ResourceManager,LegacyOrderArbiter
from .backend import NativeExecutor,NativeWorker,GradientCoordinator,OptimizerOwner,WeightPublisher
from ..orchestration import HierarchicalController,PlanningServices,RuntimeServices
from ..contracts import RequestView,MeshView,ModelContract
from dataclasses import replace
import json

class PendingKernel(PendingMixin,NativeKernel):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.init_pending()

class PendingRollout(RolloutCooperator):
    def propose_pending_takeovers(self,instance_id,snapshot):
        c=self.c;k=c.kernel
        if instance_id!=f'I{c.w}' or not c.is_step() or c.w in k.services:return ()
        choice=k.pending_choice(c.w)
        if choice is None:return ()
        src,dst,items=choice
        return (c.propose('pending_transfer',5,'TAKE_PENDING_RESPONSE',peer=src if dst==c.w else dst,
            tids=[x['trajectory_id'] for x in items],src=src,dst=dst),)

class PendingWorker(NativeWorker):
    def execute(self,intent):
        if intent['op']=='pending_transfer':
            k=self.c.kernel;choice=k.pending_choice(self.c.w);assert choice is not None
            assert (choice[0],choice[1])==(intent['details']['src'],intent['details']['dst'])
            return k.schedule_pending(self.c.w,choice)
        return super().execute(intent)

class PendingExecutor(NativeExecutor):
    def __init__(self,c):self.c=c;self.workers={w:PendingWorker(c,w) for w in c.kernel.rep}

class PendingCapabilities(Capabilities):
    def report(self,good=True,reason=()):
        x=super().report(good,reason)
        return replace(x,pending_response_takeover=True,supported_actions=x.supported_actions|{'TAKE_PENDING_RESPONSE'})

class PendingAdapter(LayeredNativeAdapter):
    def __init__(self,kernel,policy,capacity,model_id,output):
        self.kernel=kernel;self.context=c=NativeContext(kernel,policy,capacity,model_id,output)
        fixed=FixedPlanner(c);caps=PendingCapabilities(c);disabled=DisabledPlanningPolicy()
        planning=PlanningServices(fixed,fixed,disabled,disabled,disabled,disabled,disabled,disabled,disabled,caps)
        runtime=RuntimeServices(RoleScheduler(c),RegionController(c),PendingRollout(c),TrainingCooperator(c),Eligibility(c),LegacyOrderArbiter(c),
            c,c,c,StateCatalog(c),c,GradientCoordinator(c),OptimizerOwner(c),WeightPublisher(c),PendingExecutor(c),ResourceManager(c))
        self.controller=HierarchicalController(fixed,planning,runtime)
        requests=[RequestView(x['trajectory_id'],str(x['group_id']),str(x['group_id']),x['prompt_length'],8192,'recorded-decoding',x['trajectory_id']) for x in kernel.items.values()]
        mesh=MeshView(tuple(range(54)),{i:(i//6,i%6) for i in range(54)},{i:capacity['capacity_bytes'] for i in range(54)}, {}, {})
        model=ModelContract(model_id,'legacy-native-tp2-parameter-layout',(2,),(2,),'original-actor-and-fp32-gradient','uniform-tp2-native')
        plan=self.controller.prepare_step(requests,model,mesh,c.plan.algorithm);self.controller.install(plan);self.native_plan_bound=False
    def on_native_complete(self,result):
        super().on_native_complete(result)
        p=self.context.output/'framework-summary.json';v=json.loads(p.read_text());v.update(fixed_assignment=False,pending_takeover_enabled=True,pending_transfers=len(self.kernel.pending_transfers));p.write_text(json.dumps(v,indent=2)+'\n')
