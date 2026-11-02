"""Authoritative native state and fixed-plan views. No whole legacy scheduler."""
import hashlib
import json
from types import SimpleNamespace
from ..contracts import (
    ActionProposal, AlgorithmContract, AssignmentPlan, CapabilityReport, DieState,
    GradientRoutePlan, InferenceInstance, ParallelCandidate, PlacementPlan,
    Readiness, StatePlacement, StepPlan, TrainingRegion, UpdateTicket,
    ValidationReport, Version, WorldSnapshot,
)

STEP_PHASES = frozenset(('step', 'resume', 'attention_helper_ready'))

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class NativeContext:
    def __init__(self, kernel, policy, capacity, model_id, output):
        self.kernel = kernel
        self.policy = policy
        self.capacity = capacity
        self.model_id = model_id
        self.output = output
        self.version = Version(0, 'actor_v0', 'reference_v0', 0)
        self.epoch = 0
        self.serial = 0
        self.req = None
        self.current_kind = None
        self.intents = {}
        self.roles = {w:'ROLLOUT' for w in kernel.rep}
        self.prefix_confirmed = False
        self.suffix_armed = False
        self.publication_armed = False
        self.native_completed = False
        self.approved = set()
        self.reply = None
        self.pending = {}
        self.program_count = 0
        self.program_hash = hashlib.sha256()
        self.action_counts = {}
        self.decision_log = (output/'framework-decisions.jsonl').open('w')
        self.cached_snapshot = None
        self.plan = self._fixed_plan()
        self.route = GradientRoutePlan(self.plan.gradient_routing_digest, {}, {}, 'global-sample-sum', 'native-tp2-dp-suffix')

    def _fixed_plan(self):
        k = self.kernel
        training = tuple(TrainingRegion(f'R{w}', tuple(rep.ranks), 2) for w,rep in k.rep.items())
        inference = tuple(InferenceInstance(f'I{w}', tuple(rep.ranks), 2) for w,rep in k.rep.items())
        candidate = ParallelCandidate('frozen-h0',2,27,{r.region_id:2 for r in training},self.policy.cap,
            {r.region_id:self.policy.train_mb for r in training},{'rollout':0,'training':0})
        layout_digest = digest({'policy':self.policy.__dict__,'capacity':self.capacity,'layout':'legacy-uniform-tp2-native-param-layout'})
        placement = PlacementPlan('fixed-h0-54',candidate,inference,training,
            {f'D{d}':tuple(range(d*6,d*6+6)) for d in range(9)},
            {f'R{w}':tuple(f'I{j}' for j in range(w//3*3,w//3*3+3)) for w in k.rep},
            StatePlacement({}, {}, {}, layout_digest),{}, {'rollout':(),'training':()})
        assignment = AssignmentPlan('frozen-home-and-fifo',
            {x['group_id']:f'R{w}' for w,items in enumerate(k.groups) for x in items},
            {tid:f'I{w}' for tid,w in k.owners.items()},
            {f'I{w}':tuple(x['trajectory_id'] for x in rep.requests) for w,rep in k.rep.items()},
            {},frozenset(x['group_id'] for x in k.items.values()),'disabled-history-frozen-baseline')
        algorithm = AlgorithmContract(self.version,frozenset(k.ids),8,'original-sample-sum',float(len(k.ids)),
            'fixed-number-of-samples','native-original-optimizer','recorded-trajectory-rng-unchanged')
        return StepPlan('legacy-compatible-'+self.model_id,placement,assignment,algorithm,
            digest(self.capacity),layout_digest,'native-publish-once','uniform-tp2-native-capabilities')

    @property
    def w(self):
        return self.req['event']['worker']

    @property
    def phase(self):
        return self.current_kind or self.req['event']['phase']

    def is_step(self):
        return self.req is not None and self.phase in STEP_PHASES

    def region_eligible(self, w):
        k=self.kernel
        return k.domain_drained(w) and (k.early or len(k.done)==len(k.ids))

    def propose(self, op, priority, kind='CONTINUE_OWN', peer=None, tids=(), **details):
        w=self.w
        participants=tuple(self.kernel.rep[w].ranks)
        if peer is not None:
            participants=tuple(sorted(set(participants+self.kernel.rep[peer].ranks)))
        pid=f'{self.serial}:{op}:{w}'
        intent=dict(op=op,priority=priority,w=w,peer=peer,details=details)
        old=self.intents.get(pid)
        if old is not None and old!=intent:
            raise ValueError('Conflicting native intent')
        self.intents[pid]=intent
        return ActionProposal(pid,kind,tuple(tids),participants,f'I{w}' if self.roles[w]=='ROLLOUT' else f'R{w}',
            None if peer is None else f'R{peer}',self.epoch,0.0,0,0.0,1.0,self.version)

    def snapshot(self):
        if self.cached_snapshot is not None:
            return self.cached_snapshot
        k=self.kernel; labels=k.roles();dies={}
        phase_map={'generation':'DECODE','old_policy':'OLD_LOGPROB','reference':'REFERENCE',
            'advantage':'ADVANTAGE','training':'BACKWARD','gradient_zero':'BACKWARD','gradient_ready':'GRAD_READY'}
        for w,rep in k.rep.items():
            label=labels[w]
            activity={'own':'OWN','helper':'HELP','prepare':'OWN','training':'OWN','sample_transfer':'TRANSFER','idle':'WAIT'}[label]
            leases=frozenset(str(t['id']) for t in k.tx.values() if w in (t['source'],t['helper']))
            peak_key={'own':'generation','helper':'helper','prepare':'teacher','training':'training','sample_transfer':'training','idle':'training'}[label]
            reserved=self.capacity['phase_peak_bytes'][peak_key]
            for die in rep.ranks:
                dies[die]=DieState(die,self.roles[w],None,activity,phase_map.get(k.stages[w]),
                    f'I{w}' if self.roles[w]=='ROLLOUT' else f'R{w}',self.version,reserved,leases)
        self.cached_snapshot=WorldSnapshot(self.epoch,k.tick,self.plan,dies,
            f'native-ledger:{self.epoch}',f'native-samples:{self.epoch}',self.plan.feasibility_proof_digest,'uniform-tp2-native')
        return self.cached_snapshot

    def install(self, plan):
        if plan!=self.plan:
            raise ValueError('Native adapter accepts the frozen fixed plan only')

    def apply_backend_event(self, event):
        self.current_kind=event.kind
        self.epoch+=1;self.cached_snapshot=None
        if event.kind!='FINISH_ACK':
            k=self.kernel;k.tick=self.req['tick']
            if self.is_step():
                k.waiting[self.w]=False
                k.commit_stage(self.w)
                if self.w not in k.services:
                    k._observe(self.w)
        return self.snapshot()

    def reserve_decision(self, decision):
        if decision.based_on_epoch!=self.epoch:
            raise ValueError('Native reservation is stale')
        for p in decision.accepted:
            if p.proposal_id not in self.intents:
                raise ValueError('No native execution intent')
            self.approved.add(p.proposal_id)

    def note_reply(self, proposal, reply):
        self.reply=reply
        blob=json.dumps(reply,sort_keys=True,separators=(',',':')).encode()
        self.program_hash.update(len(blob).to_bytes(8,'big'));self.program_hash.update(blob)
        self.program_count+=1
        op=self.intents[proposal.proposal_id]['op']
        self.action_counts[op]=self.action_counts.get(op,0)+1
        self.decision_log.write(json.dumps(dict(serial=self.serial,tick=self.kernel.tick,w=self.w,phase=self.phase,
            role=self.roles[self.w],operation=op,proposal=proposal.proposal_id,
            participants=proposal.die_ids,reply_sha256=hashlib.sha256(blob).hexdigest()),separators=(',',':'))+'\n')
        self.decision_log.flush()
        self.cached_snapshot=None

    def all_training_contributions_complete(self, algorithm):
        k=self.kernel
        good=k.finished_all() and k.returned==set(k.rep) and not k.pending_stage
        return Readiness(good,() if good else ('native_work_or_gradient_still_pending',),'native-task-ledger')


class StateCatalog:
    def __init__(self, context):self.c=context
    def plan_ready(self, plan, snapshot):
        good=self.c.prefix_confirmed and self.c.capacity['passed']
        return Readiness(good,() if good else ('native_prefix_not_confirmed',),plan.feasibility_proof_digest)
    def propose_initialization(self,*args):return ()


class Capabilities:
    def __init__(self, context):self.c=context
    def report(self, good=True, reason=()):
        return CapabilityReport(good,frozenset(('CONTINUE_OWN','HELP_ATTENTION','ENTER_PREPARATION','START_BACKWARD','TAKE_TRAIN_BATCH')),
            False,False,False,reason)
    def validate(self, placement, model):
        good=all(x.tp==2 for x in placement.inference+placement.training)
        return self.report(good,() if good else ('legacy_requires_uniform_tp2',))
    def supports_action(self, action):
        return self.report(action.proposal_id in self.c.intents)


class FixedPlanner:
    def __init__(self, context):self.c=context
    def build(self,*args):return self.c.plan
    def validate_partition_and_drain(self, placement):
        for groups in (placement.inference,placement.training):
            dies=[d for group in groups for d in group.die_ids]
            if sorted(dies)!=list(range(54)) or any(len(g.die_ids)!=g.tp for g in groups):
                return ValidationReport(False,('invalid_fixed_partition',),{},'')
        return ValidationReport(True,(),{},'fixed-h0-partition-checked')
    def check_phase_capacity(self,*args):
        c=self.c.capacity
        peaks={(die,phase):value for die in range(54) for phase,value in c['phase_peak_bytes'].items()}
        good=c['passed'] and max(peaks.values())<=c['capacity_bytes']
        return ValidationReport(good,() if good else ('native_capacity_failed',),peaks,self.c.plan.feasibility_proof_digest)


class DisabledPlanningPolicy:
    def __getattr__(self,name):
        raise NotImplementedError('Innovation 1/2 is disabled in fixed-plan parity mode: '+name)
