"""Final lease completion quorum before training reuses helper resources."""
from .complete import CompleteWorker,CompleteCapabilities,CompleteRollout
from .adaptive_adapter import AdaptiveAdapter
from .selected_context import SelectedContext as AdaptiveContext,SelectedManifestPlanner
from .context import FixedPlanner,DisabledPlanningPolicy,StateCatalog
from .region_adapter import PlannedRegion,NoTrainingSteal
from .policies import RoleScheduler,Eligibility,ResourceManager,LegacyOrderArbiter
from .backend import NativeExecutor,GradientCoordinator,OptimizerOwner,WeightPublisher
from ..orchestration import HierarchicalController,PlanningServices,RuntimeServices
from ..contracts import RequestView,MeshView,ModelContract
from rl_xy.qwen3_sequence import local
from rl_xy.qwen3_cost import Cost

class QuorumWorker(CompleteWorker):
    def rollout_reply(self,intent):return super().execute(intent)
    def execute(self,intent):
        c=self.c;k=c.kernel;w=c.w
        if intent['op']=='rollout_step':
            tx=next((t for t in k.tx.values() if w in (t['source'],t['helper'])),None)
            if tx and tx['helper']==w:
                issued=tx.get('helper_issued_epoch',0);acked=tx.get('helper_acked_epoch',0)
                if issued>acked:
                    tx['helper_acked_epoch']=issued
                    k.helper_completion_quorums.append(dict(worker=w,source=tx['source'],lease=tx['id'],segment=issued,tick=k.tick))
                    source=tx['source']
                    if k.waiting[source] and not k.rep[source].active:
                        return {'steps':[k.wake(source,k.rep[w].ranks[0]),local(Cost(compute=1000)),k.rpc(w,'step')]}
            if tx and tx['source']==w and not k.rep[w].active and not k.rep[w].requests:
                if tx.get('helper_acked_epoch',0)<tx.get('helper_issued_epoch',0):
                    return k.wait(w)
            result=self.rollout_reply(intent)
            tx=next((t for t in k.tx.values() if t['helper']==w),None)
            if tx and tx.get('helper_started') and tx.get('stage')=='running' and result.get('repeat'):
                tx['helper_issued_epoch']=tx['segment_epoch']
            return result
        return super().execute(intent)

class QuorumExecutor(NativeExecutor):
    def __init__(self,c):self.c=c;self.workers={w:QuorumWorker(c,w) for w in c.kernel.rep}

class SystemAdapter(AdaptiveAdapter):
    def __init__(self,kernel,policy,capacity,model_id,output):
        self.kernel=kernel;kernel.helper_completion_quorums=[]
        self.context=c=AdaptiveContext(kernel,policy,capacity,model_id,output)
        fixed=SelectedManifestPlanner(c);caps=CompleteCapabilities(c);disabled=DisabledPlanningPolicy()
        planning=PlanningServices(fixed,fixed,disabled,disabled,disabled,disabled,disabled,disabled,disabled,caps)
        runtime=RuntimeServices(RoleScheduler(c),PlannedRegion(c),CompleteRollout(c),NoTrainingSteal(),Eligibility(c),LegacyOrderArbiter(c),
            c,c,c,StateCatalog(c),c,GradientCoordinator(c),OptimizerOwner(c),WeightPublisher(c),QuorumExecutor(c),ResourceManager(c))
        self.controller=HierarchicalController(fixed,planning,runtime)
        requests=[RequestView(x['trajectory_id'],str(x['group_id']),str(x['group_id']),x['prompt_length'],8192,'recorded-decoding',x['trajectory_id']) for x in kernel.items.values()]
        mesh=MeshView(tuple(range(54)),{i:(i//6,i%6) for i in range(54)},{i:capacity['capacity_bytes'] for i in range(54)}, {}, {})
        model=ModelContract(model_id,kernel.coordinates.schema_digest,(2,),(2,4,8),'frozen-BF16-and-FP32','native-complete-system')
        plan=self.controller.prepare_step(requests,model,mesh,c.plan.algorithm);self.controller.install(plan);self.native_plan_bound=False
