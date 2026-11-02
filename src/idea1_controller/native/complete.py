"""Planned regions + atomic PENDING + attention/FFN helper, same update."""
from .regions import RegionKernel
from .full_layer import FullLayerAttention
from .pending import PendingMixin
from .region_adapter import RegionAdapter,PlannedRollout,PlannedWorker,RegionCapabilities,PlannedRegion
from .pending_adapter import PendingRollout,PendingWorker
from .context import FixedPlanner,DisabledPlanningPolicy,StateCatalog
from .policies import RoleScheduler,Eligibility,ResourceManager,LegacyOrderArbiter
from .region_adapter import NoTrainingSteal
from .backend import NativeExecutor,GradientCoordinator,OptimizerOwner,WeightPublisher
from .regions import RegionContext
from ..orchestration import HierarchicalController,PlanningServices,RuntimeServices
from ..contracts import RequestView,MeshView,ModelContract
from dataclasses import replace
import json

class CompleteKernel(PendingMixin,FullLayerAttention,RegionKernel):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.init_pending()

class CompleteRollout(PlannedRollout):
    propose_pending_takeovers=PendingRollout.propose_pending_takeovers

class CompleteWorker(PendingWorker,PlannedWorker):pass
class CompleteExecutor(NativeExecutor):
    def __init__(self,c):self.c=c;self.workers={w:CompleteWorker(c,w) for w in c.kernel.rep}
class CompleteCapabilities(RegionCapabilities):
    def report(self,good=True,reason=()):
        x=super().report(good,reason);return replace(x,pending_response_takeover=True,supported_actions=x.supported_actions|{'TAKE_PENDING_RESPONSE'})

class CompleteAdapter(RegionAdapter):
    def __init__(self,kernel,policy,capacity,model_id,output):
        self.kernel=kernel;self.context=c=RegionContext(kernel,policy,capacity,model_id,output)
        fixed=FixedPlanner(c);caps=CompleteCapabilities(c);disabled=DisabledPlanningPolicy()
        planning=PlanningServices(fixed,fixed,disabled,disabled,disabled,disabled,disabled,disabled,disabled,caps)
        runtime=RuntimeServices(RoleScheduler(c),PlannedRegion(c),CompleteRollout(c),NoTrainingSteal(),Eligibility(c),LegacyOrderArbiter(c),
            c,c,c,StateCatalog(c),c,GradientCoordinator(c),OptimizerOwner(c),WeightPublisher(c),CompleteExecutor(c),ResourceManager(c))
        self.controller=HierarchicalController(fixed,planning,runtime)
        requests=[RequestView(x['trajectory_id'],str(x['group_id']),str(x['group_id']),x['prompt_length'],8192,'recorded-decoding',x['trajectory_id']) for x in kernel.items.values()]
        mesh=MeshView(tuple(range(54)),{i:(i//6,i%6) for i in range(54)},{i:capacity['capacity_bytes'] for i in range(54)}, {}, {})
        model=ModelContract(model_id,'fixed-tp2-owners-native-region-views',(2,),(2,4,8),'frozen-BF16-and-FP32','native-complete')
        plan=self.controller.prepare_step(requests,model,mesh,c.plan.algorithm);self.controller.install(plan);self.native_plan_bound=False
    def on_native_complete(self,result):
        super().on_native_complete(result)
        p=self.context.output/'framework-summary.json';v=json.loads(p.read_text());v.update(pending_takeover_enabled=True,
            helper_kernel='attention+FFN-TP4 with TP2 sampling',pending_transfers=len(self.kernel.pending_transfers));p.write_text(json.dumps(v,indent=2)+'\n')
