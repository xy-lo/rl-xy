"""Read-only mechanism-3 proposals preserving the original branch priorities."""
from ..contracts import Decision, Readiness
from .context import STEP_PHASES


def sample_transfer_choice(c):
    k=c.kernel;w=c.w
    if k.leased(w):return None
    candidates=[]
    if k.empty(w):
        for src in k.rep:
            if src!=w and k.waiting[src] and src not in k.services and not k.leased(src):
                tids=k.sample_candidates(src,w)
                if tids:candidates.append((src,w,tids))
    for dst in k.rep:
        if dst!=w and k.waiting[dst] and dst not in k.services and k.empty(dst) and not k.leased(dst):
            tids=k.sample_candidates(w,dst)
            if tids:candidates.append((w,dst,tids))
    return candidates[0] if candidates else None


def training_transfer_choice(c):
    k=c.kernel;w=c.w
    if k.leased(w) or len(k.done)!=len(k.ids) or not k.train_queues.get(w):return None
    free=[j for j in k.trained if j!=w and k.waiting[j] and j not in k.services
          and j not in k.helper_jobs and not k.leased(j)
          and (k.training_scope=='wafer' or j//3==w//3)]
    if not free:return None
    dst=min(free,key=lambda j:(abs(k.rep[j].ranks[0]-k.rep[w].ranks[0]),j))
    job=max(k.train_queues[w],key=lambda j:(j['estimate_local_ns'],j['index']))
    return dst,job


def attention_helper_choice(c):
    k=c.kernel;w=c.w;r=k.rep[w]
    tx=next((v for v in k.tx.values() if w in (v['source'],v['helper'])),None)
    if tx:return None
    if len(k.done)==len(k.ids) and not k.tx:return None
    if not r.requests and not r.active and r.pending is None:return None
    busy={x for v in k.tx.values() for x in (v['source'],v['helper'])}
    free=[j for j in k.rep if j//3==w//3 and j!=w and k.waiting[j] and j not in busy
          and not k.rep[j].requests and not k.rep[j].active and k.rep[j].pending is None]
    free.sort(key=lambda j:(max(abs(s//6-d//6)+abs(s%6-d%6) for s,d in zip(r.ranks,k.rep[j].ranks)),j))
    can_decode=bool(r.active) and not r.requests and r.pending is None
    choices=[]
    if k.priority and free:
        for source,v in k.rep.items():
            if source in busy or not v.active or v.requests or v.pending is not None:continue
            for helper,h in k.rep.items():
                if helper//3!=source//3 or helper==source or helper in busy or not k.waiting[helper]:continue
                if h.requests or h.active or h.pending is not None:continue
                estimate=k.estimate_locality(source,helper)
                if estimate['accepted']:choices.append((estimate['predicted_net_ns'],-source,-helper,source))
    chosen=max(choices)[3] if choices else None
    if w not in busy and free and can_decode and (not k.priority or w==chosen):
        choice=max((k.estimate_locality(w,j) for j in free),key=lambda x:x['predicted_net_ns'])
        if choice['accepted']:return choice['helper']
    return None


class RoleScheduler:
    def __init__(self,c):self.c=c
    def propose_role_transitions(self,snapshot):
        c=self.c;k=c.kernel
        if c.phase=='FINISH_ACK':return ()
        if not c.is_step():return (c.propose('native_control',0),)
        if c.w in k.services:return (c.propose('progress_service',0),)
        if c.roles[c.w]!='TRAINING' and c.region_eligible(c.w):
            return (c.propose('region_step',40,'ENTER_PREPARATION'),)
        return ()


class RegionController:
    def __init__(self,c):self.c=c
    def can_enter_preparation(self,region_id,snapshot):
        c=self.c;w=int(region_id[1:]);k=c.kernel
        expected={x['trajectory_id'] for x in k.groups[w]}
        good=c.region_eligible(w) and expected<=k.received[w]
        return Readiness(good,() if good else ('domain_or_samples_not_ready',),'native-domain-and-delivery')
    def can_start_backward(self,region_id,snapshot):
        c=self.c;w=int(region_id[1:]);k=c.kernel
        good=w in k.train_queues and any(e['worker']==w and e['phase']=='gradient_zero' for e in k.events)
        return Readiness(good,() if good else ('native_preparation_not_complete',),'native-teacher-advantage-gradient')
    def propose(self,region_id,snapshot):
        c=self.c;k=c.kernel
        if region_id!=f'R{c.w}' or not c.is_step() or c.w in k.services:return ()
        result=[]
        choice=sample_transfer_choice(c)
        if choice:
            src,dst,tids=choice;peer=src if dst==c.w else dst
            result.append(c.propose('sample_transfer',10,peer=peer,tids=tids,src=src,dst=dst))
        if c.roles[c.w]=='TRAINING' and c.region_eligible(c.w):
            result.append(c.propose('region_step',40))
        return tuple(result)


class RolloutCooperator:
    def __init__(self,c):self.c=c
    def propose_pending_takeovers(self,instance_id,snapshot):
        # The reference mechanism does not implement pending-response stealing.
        return ()
    def propose_attention_help(self,instance_id,snapshot):
        c=self.c;k=c.kernel
        if instance_id!=f'I{c.w}' or not c.is_step() or c.w in k.services or c.roles[c.w]!='ROLLOUT':return ()
        helper=attention_helper_choice(c)
        return (c.propose('rollout_step',50,'HELP_ATTENTION' if helper is not None else 'CONTINUE_OWN',peer=helper,
                          selected_helper=helper),)


class TrainingCooperator:
    def __init__(self,c):self.c=c
    def propose_unstarted_batch_takeovers(self,region_id,snapshot):
        c=self.c;k=c.kernel
        if region_id!=f'R{c.w}' or not c.is_step() or c.w in k.services:return ()
        result=[];choice=training_transfer_choice(c)
        if choice:
            dst,job=choice
            result.append(c.propose('training_transfer',20,'TAKE_TRAIN_BATCH',peer=dst,tids=job['tids'],batch_id=job['batch_id']))
        if c.w in k.helper_jobs:
            result.append(c.propose('run_remote_batch',30,'START_BACKWARD',tids=k.helper_jobs[c.w]['tids']))
        return tuple(result)


class Eligibility:
    def __init__(self,c):self.c=c
    def check(self,proposal,snapshot):
        c=self.c;intent=c.intents.get(proposal.proposal_id)
        good=intent is not None and proposal.based_on_epoch==c.epoch and proposal.version==c.version
        if good and intent['op']=='run_remote_batch':good=c.w in c.kernel.helper_ready
        return Readiness(good,() if good else ('native_readiness_or_epoch_failed',),'native-safe-point')


class ResourceManager:
    def __init__(self,c):self.c=c
    def check_action(self,proposal,snapshot):
        c=self.c;intent=c.intents.get(proposal.proposal_id);good=bool(intent) and c.capacity['passed']
        if good and intent['op'] in ('sample_transfer','training_transfer'):
            good=not c.kernel.leased(c.w)
        return Readiness(good,() if good else ('native_resource_conflict',),c.plan.feasibility_proof_digest)


class LegacyOrderArbiter:
    def __init__(self,c):self.c=c
    def arbitrate(self,proposals,snapshot):
        c=self.c
        if c.phase=='FINISH_ACK':return Decision(f'ack-{c.serial}',(),{},snapshot.epoch,'')
        if not proposals:raise ValueError('Native RPC has no legal next action')
        chosen=min(proposals,key=lambda p:(c.intents[p.proposal_id]['priority'],p.proposal_id))
        return Decision(f'native-{c.serial}',(chosen,),
            {p.proposal_id:('reference_branch_priority',) for p in proposals if p is not chosen},
            snapshot.epoch,'native-all-participant-reservation')
