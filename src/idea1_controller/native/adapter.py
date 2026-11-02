"""Assembly of real native mechanism-3 ports into HierarchicalController."""
import json
from ..contracts import MeshView, ModelContract, RequestView, RuntimeEvent
from ..orchestration import HierarchicalController, PlanningServices, RuntimeServices
from .context import NativeContext, StateCatalog, Capabilities, FixedPlanner, DisabledPlanningPolicy, digest
from .policies import RoleScheduler, RegionController, RolloutCooperator, TrainingCooperator, Eligibility, ResourceManager, LegacyOrderArbiter
from .backend import NativeExecutor, GradientCoordinator, OptimizerOwner, WeightPublisher
from training_coop import TrainingCoopRuntime


class NativeKernel(TrainingCoopRuntime):
    def step(self,*args,**kwargs):
        raise RuntimeError('The monolithic legacy scheduler is disabled; use framework policies')


class LayeredNativeAdapter:
    def __init__(self,kernel,policy,capacity,model_id,output):
        self.kernel=kernel
        self.context=c=NativeContext(kernel,policy,capacity,model_id,output)
        fixed=FixedPlanner(c);caps=Capabilities(c);disabled=DisabledPlanningPolicy()
        planning=PlanningServices(fixed,fixed,disabled,disabled,disabled,disabled,disabled,disabled,disabled,caps)
        runtime=RuntimeServices(RoleScheduler(c),RegionController(c),RolloutCooperator(c),TrainingCooperator(c),
            Eligibility(c),LegacyOrderArbiter(c),c,c,c,StateCatalog(c),c,GradientCoordinator(c),OptimizerOwner(c),
            WeightPublisher(c),NativeExecutor(c),ResourceManager(c))
        self.controller=HierarchicalController(fixed,planning,runtime)
        requests=[RequestView(x['trajectory_id'],str(x['group_id']),x['group_id'],x['prompt_length'],8192,
                    'original-recorded-decoding',x['trajectory_id']) for x in kernel.items.values()]
        mesh=MeshView(tuple(range(54)),{i:(i//6,i%6) for i in range(54)},
            {i:capacity['capacity_bytes'] for i in range(54)}, {}, {})
        model=ModelContract(model_id,'legacy-native-tp2-parameter-layout',(2,),(2,),
            'original-actor-and-fp32-gradient','uniform-tp2-native')
        plan=self.controller.prepare_step(requests,model,mesh,c.plan.algorithm)
        self.controller.install(plan)
        self.native_plan_bound=False

    def __getattr__(self,name):return getattr(self.kernel,name)

    def bind_native_plan(self,plan):
        names=[o.name for o in plan.operations]
        assert sum('publish_actor_v1' in name for name in names)==1
        assert sum('dp_gradient_RS' in o.name and o.kind=='collective' for o in plan.operations)==2
        self.native_plan_bound=True

    def dispatch(self,req):
        c=self.context;w=req['event']['worker']
        assert self.native_plan_bound
        c.serial+=1;c.req=req;c.current_kind=req['event']['phase']
        c.intents={};c.approved=set();c.reply=None;c.cached_snapshot=None
        if self.controller.phase=='STAGING':
            assert c.phase=='arrive'
            # This callback is downstream of the original global-ready prefix,
            # including 14B reference reconstruction. No cost is omitted/added.
            c.prefix_confirmed=True
            self.controller.activate()
        previous=c.pending.pop(w,None)
        command_id,event_id=previous if previous is not None else ('',f'native-initial-{w}')
        event=RuntimeEvent(event_id,req['tick'],c.phase,command_id,
            digest({'event':req['event'],'iteration':req['iteration'],'tick':req['tick']}),c.version)
        self.controller.handle_event(event)
        reply=c.reply
        if reply is None:raise RuntimeError('Native framework produced no RPC reply')
        if reply.get('finish'):
            command_id,event_id=c.pending.pop(w)
            ack=RuntimeEvent(event_id,req['tick'],'FINISH_ACK',command_id,'zero-step-finish-confirmed',c.version)
            self.controller.handle_event(ack)
        if self.kernel.returned==set(self.kernel.rep) and not c.suffix_armed:
            assert not c.pending
            self.controller.finalize_update(c.route)
        return reply

    def on_native_complete(self,native_result):
        c=self.context
        assert native_result['native_exit_status']==0
        result=json.loads((c.output/'native/result.json').read_text())
        assert result['native_coverage_and_dependencies_checked'] and len(result['native_sequence_summaries'])==54
        assert c.suffix_armed and c.publication_armed
        c.native_completed=True
        assert self.controller.finished_and_published()
        c.decision_log.close()
        (c.output/'framework-summary.json').write_text(json.dumps(dict(
            status='PASS_LAYERED_NATIVE_FRAMEWORK',controller_phase=self.controller.phase,
            native_rpc_commands=c.program_count,native_reply_sequence_sha256=c.program_hash.hexdigest(),
            operations=c.action_counts,legacy_top_level_step_enabled=False,
            fixed_placement=True,fixed_assignment=True,pending_takeover_enabled=False,
            training_help_requires_global_eos=True,native_prefix_and_suffix_preserved=True,
            optimizer_suffix_armed_once=c.suffix_armed,publication_confirmed=c.native_completed),indent=2)+'\n')

    def audit(self):
        self.kernel.audit()
        assert self.controller.phase=='FINISHED'
