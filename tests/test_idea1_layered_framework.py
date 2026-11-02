"""Framework contract tests using fake ports, never a training/simulator run."""
import dataclasses
import unittest
from types import SimpleNamespace
from idea1_controller.contracts import (
    ActionProposal, AlgorithmContract, AssignmentPlan, CapabilityReport, CommandBatch,
    Decision, DieState, GradientRoutePlan, InferenceInstance, MeshView, ModelContract,
    ParallelCandidate, PlacementPlan, Readiness, RequestView, RuntimeEvent,
    StatePlacement, StepPlan, TrainingRegion, UpdateTicket, ValidationReport, Version,
    WorldSnapshot,
)
from idea1_controller import HierarchicalController, PlanningServices, RuntimeServices
from idea1_controller.errors import ContractViolation, ExecutionFault, NotReady, UnsupportedPlan


def capability(supported=True, heterogeneous=False):
    return CapabilityReport(supported, frozenset(), heterogeneous, True, True, () if supported else ('unsupported',))


class Capabilities:
    def __init__(self):
        self.heterogeneous = False
        self.reject = set()

    def validate(self, placement, model):
        return capability(heterogeneous=self.heterogeneous)

    def supports_action(self, action):
        return capability(action.kind not in self.reject)


class LocalPolicies:
    def __init__(self):
        self.pending = []
        self.attention = []
        self.training = []
        self.attention_calls = 0

    def propose_pending_takeovers(self, instance, snapshot):
        return self.pending if instance == 'I0' else []

    def propose_attention_help(self, instance, snapshot):
        self.attention_calls += instance == 'I0'
        return self.attention if instance == 'I0' else []

    def propose_unstarted_batch_takeovers(self, region, snapshot):
        return self.training if region == 'R0' else []


class World:
    def __init__(self, plan, version):
        self.plan = plan
        self.version = version
        self.epoch = 0
        self.events = 0
        self.reservations = []

    def install(self, plan):
        self.plan = plan

    def snapshot(self):
        dies = {i: DieState(i, 'ROLLOUT' if i < 2 else 'TRAINING', None,
                           'OWN' if i < 2 else 'WAIT', 'DECODE' if i < 2 else 'GRAD_READY',
                           'I0' if i < 2 else 'R1', self.version, 0, frozenset()) for i in range(4)}
        return WorldSnapshot(self.epoch, self.epoch * 1000, self.plan, dies, 'tasks', 'samples', 'states', 'fake')

    def apply_backend_event(self, event):
        self.events += 1
        self.epoch += 1
        return self.snapshot()

    def reserve_decision(self, decision):
        self.reservations.append(decision)


class Executor:
    def __init__(self):
        self.submitted = []
        self.fail = False

    def compile(self, decision):
        n = len(self.submitted) + 1
        return CommandBatch('cmd' + str(n), decision.decision_id, 'fake-program', ('complete' + str(n),), decision.accepted[0].version)

    def submit(self, commands):
        self.submitted.append(commands)
        if self.fail:
            raise OSError('unknown submission status')
        return commands.command_id


class Gradients:
    def __init__(self, version):
        self.version = version
        self.ready = True
        self.calls = 0

    def can_finalize(self, snapshot):
        return Readiness(self.ready, () if self.ready else ('unfinished_gradient',), 'fake')

    def prepare_update(self, route, algorithm, snapshot):
        self.calls += 1
        return UpdateTicket('ticket', self.version.iteration, 'all-tasks', 'all-grads', 'owners', 'v1', 2.0)


class Publisher:
    def __init__(self):
        self.calls = 0
        self.ack = False

    def publish_and_reshard(self, ticket, placement):
        self.calls += 1
        return 'publication'

    def all_target_instances_acknowledged(self, publication):
        return Readiness(self.ack, () if self.ack else ('pending_publication',), 'fake')


def fixture():
    e = SimpleNamespace()
    e.version = Version(0, 'v0', 'ref0', 0)
    e.algorithm = AlgorithmContract(e.version, frozenset(('q0', 'q1')), 1, 'weights', 2.0,
                                    'normalizer-recipe', 'optimizer', 'rng')
    candidate = ParallelCandidate('candidate', 2, 2, {'R0': 2, 'R1': 2}, 8,
                                  {'R0': 1, 'R1': 1}, {'rollout': 0, 'training': 0})
    e.placement = PlacementPlan('placement', candidate,
        (InferenceInstance('I0', (0, 1), 2), InferenceInstance('I1', (2, 3), 2)),
        (TrainingRegion('R0', (0, 1), 2), TrainingRegion('R1', (2, 3), 2)),
        {'D0': (0, 1), 'D1': (2, 3)}, {'R0': ('I0',), 'R1': ('I1',)},
        StatePlacement({}, {}, {}, 'migration'), {'R0': ('R1',), 'R1': ('R0',)},
        {'rollout': (), 'training': ()})
    assignment = AssignmentPlan('assignment', {'g0': 'R0', 'g1': 'R1'}, {'q0': 'I0', 'q1': 'I1'},
                                {'I0': ('q0',), 'I1': ('q1',)}, {'R0': 1.0, 'R1': 1.0}, frozenset(), 'prediction')
    e.plan = StepPlan('step', e.placement, assignment, e.algorithm, 'capacity', 'route', 'publication', 'capability')
    e.route = GradientRoutePlan('route', {}, {}, 'normalizer', 'communication')
    e.mesh = MeshView((0, 1, 2, 3), {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)},
                      {i: 1000 for i in range(4)}, {}, {})
    e.model = ModelContract('fake-model', 'params', (2,), (2,), 'fp32', 'fake')
    e.requests = [RequestView('q' + str(i), 'p' + str(i), 'g' + str(i), 1, 2, 'decode', 'rng' + str(i)) for i in range(2)]
    e.capabilities = Capabilities()
    proof = ValidationReport(True, (), {}, 'fake-proof')
    e.planning = PlanningServices(
        SimpleNamespace(check_phase_capacity=lambda *args: proof),
        SimpleNamespace(validate_partition_and_drain=lambda *args: proof),
        object(), object(), object(), object(), object(), object(), object(), e.capabilities)
    e.world = World(e.plan, e.version)
    e.local = LocalPolicies()
    e.executor = Executor()
    e.gradients = Gradients(e.version)
    e.publisher = Publisher()
    e.optimizer_calls = []
    e.task_ready = True
    e.state_ready = True
    e.reject_readiness = set()
    e.arbiter = SimpleNamespace(arbitrate=lambda proposals, snapshot:
        Decision('decision-' + str(snapshot.epoch), tuple(proposals), {}, snapshot.epoch, 'reservation'))
    e.runtime = RuntimeServices(
        SimpleNamespace(propose_role_transitions=lambda snapshot: []),
        SimpleNamespace(propose=lambda region, snapshot: []), e.local, e.local,
        SimpleNamespace(check=lambda proposal, snapshot: Readiness(
            proposal.proposal_id not in e.reject_readiness,
            ('not_ready',) if proposal.proposal_id in e.reject_readiness else (), 'fake')),
        e.arbiter, e.world,
        SimpleNamespace(all_training_contributions_complete=lambda algorithm:
            Readiness(e.task_ready, () if e.task_ready else ('pending_task',), 'fake')),
        object(), SimpleNamespace(
            plan_ready=lambda *args: Readiness(e.state_ready, () if e.state_ready else ('state_not_resident',), 'fake'),
            propose_initialization=lambda *args: []), object(), e.gradients,
        SimpleNamespace(apply_once=lambda ticket: e.optimizer_calls.append(ticket.ticket_id)),
        e.publisher, e.executor, SimpleNamespace(check_action=lambda *args: Readiness(True, (), 'fake')))
    e.planner = SimpleNamespace(build=lambda *args: e.plan)
    e.controller = HierarchicalController(e.planner, e.planning, e.runtime)
    return e


def start(e):
    e.controller.install(e.controller.prepare_step(e.requests, e.model, e.mesh, e.algorithm))
    e.controller.activate()


def action(name, kind='TAKE_TRAIN_BATCH', dies=(2, 3), epoch=1):
    return ActionProposal(name, kind, ('task',), dies, 'R0', 'R1', epoch, 1.0, 1, 0.0, 1.0, Version(0, 'v0', 'ref0', 0))


def boot():
    return RuntimeEvent('boot', 0, 'READY', '', 'payload', Version(0, 'v0', 'ref0', 0))


class ControllerContracts(unittest.TestCase):
    def test_installed_plan_does_not_activate_before_state_confirmation(self):
        e = fixture()
        e.controller.install(e.controller.prepare_step(e.requests, e.model, e.mesh, e.algorithm))
        e.state_ready = False
        with self.assertRaises(NotReady):
            e.controller.activate()
        self.assertEqual(e.controller.phase, 'STAGING')
        e.local.training = [action('help')]
        self.assertFalse(e.controller.handle_event(boot()).accepted)
        self.assertFalse(e.executor.submitted)
        e.state_ready = True
        e.controller.activate()
        self.assertEqual(e.controller.phase, 'ACTIVE')

    def test_shared_resource_manager_can_reject_a_ready_action(self):
        e = fixture(); start(e); e.local.training = [action('help')]
        object.__setattr__(e.runtime, 'resources', SimpleNamespace(
            check_action=lambda *args: Readiness(False, ('memory_busy',), 'fake')))
        decision = e.controller.handle_event(boot())
        self.assertFalse(decision.accepted)
        self.assertIn('memory_busy', decision.rejected_reasons['help'])
        self.assertFalse(e.executor.submitted)

    def test_old_version_event_cannot_modify_current_state(self):
        e = fixture(); start(e)
        old = dataclasses.replace(boot(), version=Version(-1, 'old', 'ref0', -1))
        with self.assertRaises(ContractViolation):
            e.controller.handle_event(old)
        self.assertEqual(e.world.events, 0)
        self.assertFalse(e.executor.submitted)

    def test_non_initialization_action_is_rejected_while_staging(self):
        e = fixture()
        e.controller.install(e.controller.prepare_step(e.requests, e.model, e.mesh, e.algorithm))
        e.runtime.states.propose_initialization = lambda *args: [action('wrong_phase')]
        decision = e.controller.handle_event(boot())
        self.assertFalse(decision.accepted)
        self.assertIn('plan_not_active', decision.rejected_reasons['wrong_phase'])
        self.assertFalse(e.executor.submitted)

    def test_contracts_deeply_freeze_external_mappings(self):
        coords = {0: [0, 0]}
        mesh = MeshView((0,), coords, {0: 1}, {}, {})
        coords[0][0] = 99
        self.assertEqual(mesh.coordinates[0], (0, 0))
        with self.assertRaises(TypeError):
            mesh.coordinates[0] = (1, 1)

    def test_pending_response_is_considered_before_attention(self):
        e = fixture(); start(e)
        e.local.pending = [action('pending', 'TAKE_PENDING_RESPONSE')]
        e.local.attention = [action('attention', 'HELP_ATTENTION')]
        decision = e.controller.handle_event(boot())
        self.assertEqual([p.proposal_id for p in decision.accepted], ['pending'])
        self.assertEqual(e.local.attention_calls, 0)

    def test_ineligible_pending_does_not_hide_attention_candidate(self):
        e = fixture(); start(e)
        e.local.pending = [action('pending', 'TAKE_PENDING_RESPONSE')]
        e.local.attention = [action('attention', 'HELP_ATTENTION')]
        e.reject_readiness.add('pending')
        decision = e.controller.handle_event(boot())
        self.assertEqual([p.proposal_id for p in decision.accepted], ['attention'])
        self.assertIn('pending', decision.rejected_reasons)

    def test_training_help_has_no_hardcoded_global_eos_gate(self):
        e = fixture(); start(e); e.local.training = [action('help')]
        self.assertEqual(e.world.snapshot().dies[0].assigned_role, 'ROLLOUT')
        self.assertEqual(len(e.controller.handle_event(boot()).accepted), 1)
        self.assertEqual(len(e.executor.submitted), 1)

    def test_duplicate_completion_cannot_submit_twice(self):
        e = fixture(); start(e); e.local.training = [action('help')]
        first = e.controller.handle_event(boot())
        self.assertIs(first, e.controller.handle_event(boot()))
        self.assertEqual(e.world.events, 1)
        self.assertEqual(len(e.executor.submitted), 1)
        with self.assertRaises(ContractViolation):
            e.controller.handle_event(dataclasses.replace(boot(), payload_digest='different'))

    def test_stale_candidates_are_rejected(self):
        e = fixture(); start(e); e.local.training = [action('stale', epoch=0)]
        decision = e.controller.handle_event(boot())
        self.assertFalse(decision.accepted)
        self.assertIn('stale_snapshot', decision.rejected_reasons['stale'])
        self.assertFalse(e.executor.submitted)

    def test_arbiter_cannot_allocate_one_die_twice(self):
        e = fixture(); start(e)
        e.local.training = [action('a'), action('b')]
        with self.assertRaises(ExecutionFault):
            e.controller.handle_event(boot())
        self.assertFalse(e.executor.submitted)
        self.assertEqual(e.controller.phase, 'FAULTED')

    def test_unknown_submission_holds_reservation_and_faults(self):
        e = fixture(); start(e); e.local.training = [action('help')]; e.executor.fail = True
        with self.assertRaises(ExecutionFault):
            e.controller.handle_event(boot())
        self.assertEqual(len(e.world.reservations), 1)
        self.assertEqual(len(e.executor.submitted), 1)
        self.assertEqual(e.controller.phase, 'FAULTED')

    def test_update_waits_for_inflight_commands_and_complete_tasks(self):
        e = fixture(); start(e); e.local.training = [action('help')]
        e.controller.handle_event(boot())
        with self.assertRaises(NotReady):
            e.controller.finalize_update(e.route)
        e.local.training = []
        e.controller.handle_event(RuntimeEvent('complete1', 1, 'DONE', 'cmd1', 'done', e.version))
        e.task_ready = False
        with self.assertRaises(NotReady):
            e.controller.finalize_update(e.route)
        self.assertFalse(e.optimizer_calls)

    def test_one_update_and_all_replica_publication_barrier(self):
        e = fixture(); start(e)
        self.assertEqual(e.controller.finalize_update(e.route), 'publication')
        self.assertEqual(e.controller.finalize_update(e.route), 'publication')
        self.assertEqual(e.optimizer_calls, ['ticket'])
        self.assertEqual(e.publisher.calls, 1)
        self.assertFalse(e.controller.finished_and_published())
        e.publisher.ack = True
        self.assertTrue(e.controller.finished_and_published())
        self.assertEqual(e.gradients.calls, 1)

    def test_heterogeneous_tp_requires_backend_capability(self):
        e = fixture()
        regions = (e.placement.training[0], dataclasses.replace(e.placement.training[1], tp=4))
        e.plan = dataclasses.replace(e.plan, placement=dataclasses.replace(e.placement, training=regions))
        with self.assertRaises(UnsupportedPlan):
            e.controller.prepare_step(e.requests, e.model, e.mesh, e.algorithm)

    def test_update_cannot_change_the_algorithm_denominator(self):
        e = fixture(); start(e)
        original = e.gradients.prepare_update
        e.gradients.prepare_update = lambda *args: dataclasses.replace(original(*args), resolved_global_normalizer=4.0)
        with self.assertRaises(ExecutionFault):
            e.controller.finalize_update(e.route)
        self.assertFalse(e.optimizer_calls)


if __name__ == '__main__':
    unittest.main()
