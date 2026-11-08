"""Controller wiring only. All optimization and backend policies are injected."""
from dataclasses import dataclass, replace
from typing import Optional, Protocol, Sequence
from .contracts import AlgorithmContract, Decision, MeshView, ModelContract, RequestView, RuntimeEvent, StepPlan, GradientRoutePlan, PlacementPlan, ActionProposal, WorldSnapshot
from .execution import BackendCapabilities, CommandExecutor
from .gradient import GradientCoordinator, GradientLayoutPlanner, OptimizerCoordinator, WeightPublisher
from .planning import ParallelismPlanner, PlacementPlanner, PlanCostModel, PlanSelector
from .prediction import BaselineAssigner, HistoryStore, QuotaTaskAssigner, WorkPredictor
from .runtime import ActionArbiter, EligibilityChecker, GlobalRoleScheduler, RegionController, RolloutCooperator, TrainingCooperator
from .state import LeaseManager, SampleLedger, StateCatalog, TaskLedger, WorldState, ResourceManager

@dataclass(frozen=True)
class PlanningServices:
    parallelism: ParallelismPlanner
    placement: PlacementPlanner
    predictor: WorkPredictor
    assigner: QuotaTaskAssigner
    fallback: BaselineAssigner
    history: HistoryStore
    gradient_layout: GradientLayoutPlanner
    cost: PlanCostModel
    selector: PlanSelector
    capabilities: BackendCapabilities

@dataclass(frozen=True)
class RuntimeServices:
    role_scheduler: GlobalRoleScheduler
    regions: RegionController
    rollout: RolloutCooperator
    training: TrainingCooperator
    eligibility: EligibilityChecker
    arbiter: ActionArbiter
    world: WorldState
    tasks: TaskLedger
    samples: SampleLedger
    states: StateCatalog
    leases: LeaseManager
    gradients: GradientCoordinator
    optimizer: OptimizerCoordinator
    publisher: WeightPublisher
    executor: CommandExecutor
    resources: ResourceManager

class StepPlanner(Protocol):
    def build(self, requests: Sequence[RequestView], model: ModelContract, mesh: MeshView, algorithm: AlgorithmContract, services: PlanningServices) -> StepPlan: ...
    # candidate→placement→prediction/assignment→gradient route→容量/能力复核→整步评分。

from .errors import ContractViolation, ExecutionFault, LifecycleError, NotReady, UnsupportedPlan


class HierarchicalController:
    """Plan admission, event dispatch, policy arbitration and one update barrier.

    No default planner, predictor, runtime policy or executor is provided.
    Execution failures retain reservations and fault the controller: reconciliation
    belongs to the supplied backend, never to an automatic blind retry here.
    """
    def __init__(self, planner: StepPlanner, planning: PlanningServices, runtime: RuntimeServices) -> None:
        self.planner = planner
        self.planning = planning
        self.runtime = runtime
        self._phase = 'EMPTY'
        self._prepared = None
        self._plan = None
        self._seen = {}
        self._inflight = {}
        self._publication_id = None
        self._ticket = None

    @property
    def phase(self) -> str:
        return self._phase

    def prepare_step(self, requests: Sequence[RequestView], model: ModelContract, mesh: MeshView, algorithm: AlgorithmContract) -> StepPlan:
        if self._phase not in ('EMPTY', 'FINISHED'):
            raise LifecycleError('The previous step is still active or faulted')
        if {r.request_id for r in requests} != set(algorithm.expected_request_ids):
            raise ContractViolation('Request IDs differ from the algorithm contract')
        plan = self.planner.build(requests, model, mesh, algorithm, self.planning)
        if plan.algorithm != algorithm:
            raise ContractViolation('The planner changed the algorithm contract')
        for proof in (
            self.planning.placement.validate_partition_and_drain(plan.placement),
            self.planning.parallelism.check_phase_capacity(plan.placement.candidate, plan.placement, model, algorithm),
        ):
            if not proof.valid:
                raise UnsupportedPlan('; '.join(proof.reasons))
        capability = self.planning.capabilities.validate(plan.placement, model)
        if not capability.supported:
            raise UnsupportedPlan('; '.join(capability.reasons))
        tps = {region.tp for region in plan.placement.training}
        if len(tps) > 1 and not capability.heterogeneous_training_tp:
            raise UnsupportedPlan('The backend cannot execute heterogeneous training TP')
        if not all((plan.feasibility_proof_digest, plan.gradient_routing_digest, plan.publication_plan_digest)):
            raise ContractViolation('Missing feasibility, gradient or publication contract')
        self._prepared = plan
        return plan

    def install(self, plan: StepPlan) -> None:
        if self._phase not in ('EMPTY', 'FINISHED') or plan != self._prepared:
            raise LifecycleError('Install requires a newly prepared plan')
        self.runtime.world.install(plan)
        self._plan = plan
        self._phase = 'STAGING'
        self._seen.clear()
        self._inflight.clear()
        self._publication_id = None
        self._ticket = None

    def activate(self) -> None:
        if self._phase != 'STAGING':
            raise LifecycleError('Only a staged plan can be activated')
        if self._inflight:
            raise NotReady('Initialization commands still await completion')
        ready = self.runtime.states.plan_ready(self._plan, self.runtime.world.snapshot())
        if not ready.allowed:
            raise NotReady('; '.join(ready.blockers))
        self._phase = 'ACTIVE'

    def _collect(self, snapshot: WorldSnapshot) -> Sequence[ActionProposal]:
        if self._phase == 'STAGING':
            return self.runtime.states.propose_initialization(self._plan, snapshot)
        proposals = list(self.runtime.role_scheduler.propose_role_transitions(snapshot))
        for region in self._plan.placement.training:
            proposals.extend(self.runtime.regions.propose(region.region_id, snapshot))
        for instance in self._plan.placement.inference:
            pending = self.runtime.rollout.propose_pending_takeovers(instance.instance_id, snapshot)
            eligible_pending = [p for p in pending if self.runtime.eligibility.check(p, snapshot).allowed
                                and self.planning.capabilities.supports_action(p).supported
                                and self.runtime.resources.check_action(p, snapshot).allowed]
            proposals.extend(pending)
            if not eligible_pending:
                proposals.extend(self.runtime.rollout.propose_attention_help(instance.instance_id, snapshot))
        for region in self._plan.placement.training:
            proposals.extend(self.runtime.training.propose_unstarted_batch_takeovers(region.region_id, snapshot))
        return proposals

    def _decide(self, snapshot: WorldSnapshot) -> Decision:
        proposed = {}
        for proposal in self._collect(snapshot):
            old = proposed.get(proposal.proposal_id)
            if old is not None and old != proposal:
                raise ContractViolation('Conflicting proposal IDs')
            proposed[proposal.proposal_id] = proposal
        legal = []
        rejected = {}
        for proposal in proposed.values():
            reasons = []
            if proposal.version != self._plan.algorithm.version:
                reasons.append('version_mismatch')
            if self._phase == 'STAGING' and proposal.kind not in ('PREPARE_INITIAL_STATE', 'WAIT'):
                reasons.append('plan_not_active')
            if proposal.based_on_epoch != snapshot.epoch:
                reasons.append('stale_snapshot')
            if not set(proposal.die_ids) <= set(snapshot.dies):
                reasons.append('unknown_die')
            readiness = self.runtime.eligibility.check(proposal, snapshot)
            capability = self.planning.capabilities.supports_action(proposal)
            resources = self.runtime.resources.check_action(proposal, snapshot)
            if not readiness.allowed:
                reasons.extend(readiness.blockers or ('not_ready',))
            if not resources.allowed:
                reasons.extend(resources.blockers or ('resources_unavailable',))
            if not capability.supported:
                reasons.extend(capability.reasons or ('unsupported_action',))
            if reasons:
                rejected[proposal.proposal_id] = tuple(reasons)
            else:
                legal.append(proposal)
        decision = self.runtime.arbiter.arbitrate(tuple(legal), snapshot)
        if decision.based_on_epoch != snapshot.epoch:
            raise ContractViolation('Arbiter returned a stale decision')
        legal_by_id = {p.proposal_id: p for p in legal}
        reserved = set()
        accepted_ids = set()
        for proposal in decision.accepted:
            if legal_by_id.get(proposal.proposal_id) != proposal or proposal.proposal_id in accepted_ids:
                raise ContractViolation('Arbiter accepted an invalid or duplicate proposal')
            if reserved.intersection(proposal.die_ids):
                raise ContractViolation('Two accepted actions share a die')
            reserved.update(proposal.die_ids)
            accepted_ids.add(proposal.proposal_id)
        for proposal in legal:
            if proposal.proposal_id not in accepted_ids:
                rejected.setdefault(proposal.proposal_id, ('not_selected_by_arbiter',))
        rejected.update(decision.rejected_reasons)
        return replace(decision, rejected_reasons=rejected)

    def handle_event(self, event: RuntimeEvent) -> Decision:
        prior = self._seen.get(event.event_id)
        if prior is not None:
            if prior[0] != event:
                raise ContractViolation('Event ID reused with different contents')
            return prior[1]
        if self._phase not in ('STAGING', 'ACTIVE', 'PUBLISHING'):
            raise LifecycleError('No active step accepts events')
        if event.version != self._plan.algorithm.version:
            raise ContractViolation('Event belongs to another iteration or model version')
        try:
            expected = self._inflight.get(event.command_id)
            if expected is not None and event.event_id not in expected:
                raise ContractViolation('Unexpected completion event for a submitted command')
            snapshot = self.runtime.world.apply_backend_event(event)
            if snapshot.step_plan.step_plan_id != self._plan.step_plan_id:
                raise ContractViolation('Backend event switched the active plan')
            if expected is not None:
                expected.remove(event.event_id)
                if not expected:
                    del self._inflight[event.command_id]
            if self._phase == 'PUBLISHING':
                decision = Decision(event.event_id + ':ack', (), {}, snapshot.epoch, '')
            else:
                decision = self._decide(snapshot)
            if decision.accepted:
                self.runtime.world.reserve_decision(decision)
                commands = self.runtime.executor.compile(decision)
                if commands.decision_id != decision.decision_id or not commands.completion_event_ids or commands.version != self._plan.algorithm.version:
                    raise ContractViolation('Command batch has no valid decision/completion contract')
                if commands.command_id in self._inflight:
                    raise ContractViolation('Command ID is already in flight')
                self._inflight[commands.command_id] = set(commands.completion_event_ids)
                self.runtime.executor.submit(commands)
            self._seen[event.event_id] = (event, decision)
            return decision
        except Exception as error:
            self._phase = 'FAULTED'
            raise ExecutionFault('Controller faulted; reconcile reservations and backend status before resuming') from error

    def finalize_update(self, route: GradientRoutePlan, target: Optional[PlacementPlan] = None) -> str:
        if self._phase in ('PUBLISHING', 'FINISHED'):
            return self._publication_id
        if self._phase != 'ACTIVE':
            raise LifecycleError('No active step can finalize')
        if self._inflight:
            raise NotReady('Submitted commands still await completion')
        if route.plan_digest != self._plan.gradient_routing_digest:
            raise ContractViolation('Gradient route differs from the installed plan')
        if target is not None and target != self._plan.placement:
            raise UnsupportedPlan('A different next-step layout requires a separately admitted publication plan')
        snapshot = self.runtime.world.snapshot()
        for ready in (
            self.runtime.tasks.all_training_contributions_complete(self._plan.algorithm),
            self.runtime.gradients.can_finalize(snapshot),
        ):
            if not ready.allowed:
                raise NotReady('; '.join(ready.blockers))
        try:
            self._ticket = self.runtime.gradients.prepare_update(route, self._plan.algorithm, snapshot)
            if self._ticket.iteration != self._plan.algorithm.version.iteration:
                raise ContractViolation('Update ticket belongs to another iteration')
            if not self._ticket.resolved_global_normalizer > 0:
                raise ContractViolation('Global normalizer must be positive')
            expected = self._plan.algorithm.global_normalizer
            if expected is not None and self._ticket.resolved_global_normalizer != expected:
                raise ContractViolation('Update ticket changed the algorithm normalizer')
            self.runtime.optimizer.apply_once(self._ticket)
            self._publication_id = self.runtime.publisher.publish_and_reshard(self._ticket, self._plan.placement)
            self._phase = 'PUBLISHING'
            return self._publication_id
        except Exception as error:
            self._phase = 'FAULTED'
            raise ExecutionFault('Update/publication faulted; no automatic optimizer retry is allowed') from error

    def finished_and_published(self) -> bool:
        if self._phase == 'FINISHED':
            return True
        if self._phase != 'PUBLISHING':
            return False
        ready = self.runtime.publisher.all_target_instances_acknowledged(self._publication_id)
        if ready.allowed and not self._inflight:
            self._phase = 'FINISHED'
            return True
        return False
