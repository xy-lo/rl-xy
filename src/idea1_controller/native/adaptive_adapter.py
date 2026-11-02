"""Adaptive arena contracts are installed before native execution begins."""
from .complete import CompleteAdapter,CompleteCapabilities,CompleteRollout,CompleteExecutor
from .regions import RegionContext
from .context import FixedPlanner,DisabledPlanningPolicy,StateCatalog
from .region_adapter import PlannedRegion,NoTrainingSteal
from .policies import RoleScheduler,Eligibility,ResourceManager,LegacyOrderArbiter
from .backend import GradientCoordinator,OptimizerOwner,WeightPublisher
from ..orchestration import HierarchicalController,PlanningServices,RuntimeServices
from ..contracts import RequestView,MeshView,ModelContract
from dataclasses import replace

class AdaptiveContext(RegionContext):
    def _fixed_plan(self):
        p=super()._fixed_plan();k=self.kernel
        mb={r.region_id:(k.train_mb if r.tp==2 else min(8,k.train_mb*(r.tp//2)*2)) for r in p.placement.training}
        candidate=replace(p.placement.candidate,training_microbatch_by_region=mb)
        return replace(p,placement=replace(p.placement,candidate=candidate))

class AdaptiveAdapter(CompleteAdapter):
    def __init__(self,kernel,policy,capacity,model_id,output):
        self.kernel=kernel;self.context=c=AdaptiveContext(kernel,policy,capacity,model_id,output)
        fixed=FixedPlanner(c);caps=CompleteCapabilities(c);disabled=DisabledPlanningPolicy()
        planning=PlanningServices(fixed,fixed,disabled,disabled,disabled,disabled,disabled,disabled,disabled,caps)
        runtime=RuntimeServices(RoleScheduler(c),PlannedRegion(c),CompleteRollout(c),NoTrainingSteal(),Eligibility(c),LegacyOrderArbiter(c),
            c,c,c,StateCatalog(c),c,GradientCoordinator(c),OptimizerOwner(c),WeightPublisher(c),CompleteExecutor(c),ResourceManager(c))
        self.controller=HierarchicalController(fixed,planning,runtime)
        requests=[RequestView(x['trajectory_id'],str(x['group_id']),str(x['group_id']),x['prompt_length'],8192,'recorded-decoding',x['trajectory_id']) for x in kernel.items.values()]
        mesh=MeshView(tuple(range(54)),{i:(i//6,i%6) for i in range(54)},{i:capacity['capacity_bytes'] for i in range(54)}, {}, {})
        model=ModelContract(model_id,'fixed-tp2-owners-native-region-views',(2,),(2,4,8),'frozen-BF16-and-FP32','native-adaptive')
        plan=self.controller.prepare_step(requests,model,mesh,c.plan.algorithm);self.controller.install(plan);self.native_plan_bound=False
