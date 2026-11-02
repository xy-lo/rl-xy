"""Native command compilation and precompiled optimizer-suffix ownership."""
from ..contracts import CommandBatch, Readiness, UpdateTicket
from .context import digest
from locality_runtime_v3 import AttentionRuntime


class NativeExecutor:
    def __init__(self,c):
        self.c=c
        self.workers={w:NativeWorker(c,w) for w in c.kernel.rep}
    def compile(self,decision):
        c=self.c
        if len(decision.accepted)!=1:raise ValueError('One native RPC continuation must be selected')
        proposal=decision.accepted[0]
        if proposal.proposal_id not in c.approved:raise ValueError('Unapproved native command')
        reply=self.workers[c.w].execute(c.intents[proposal.proposal_id])
        c.note_reply(proposal,reply)
        return CommandBatch(f'native-c{c.serial}',decision.decision_id,digest(reply),(f'native-complete-{c.serial}',),c.version)
    def submit(self,commands):
        self.c.pending[self.c.w]=(commands.command_id,commands.completion_event_ids[0])
        return commands.command_id
    def poll(self):return ()

class NativeWorker:
    def __init__(self,c,w):
        self.c=c;self.worker_id=f'I{w}';self.w=w;self.die_ids=tuple(c.kernel.rep[w].ranks)
    def poll(self):return ()

    def execute(self,intent):
        c=self.c;k=c.kernel;w=c.w;it=c.req['iteration'];op=intent['op']
        assert w==self.w and intent['w']==w
        if op=='native_control':
            phase=c.phase
            if phase=='arrive':
                gate=f'joint-worker-{w}-{it}'
                return {'steps':[{'kind':'gate_signal','key':gate},
                    {'kind':'gate_wait','key':gate,'threshold':2},k.rpc(w,'step')]}
            if phase=='sample_run':return k.sample_run(w)
            if phase=='sample_done':return k.sample_done(w,c.req['event']['transfer'])
            if phase=='attention_migrate_source':return k.migration_source(w)
            if phase=='attention_source_ready':return k.source_ready(w,it)
            if phase=='attention_source':return k.attention_source(w)
            if phase=='deferred':return k.deferred.pop(w)
            if phase=='joint_finish':
                k.returned.add(w);return {'steps':[],'finish':True}
            raise ValueError('Unsupported native phase '+phase)
        if op=='progress_service':return k.sample_run(w)
        if op=='sample_transfer':
            result=k.schedule_transfer(w)
            if result is None:raise ValueError('Sample transfer proposal changed before commit')
            tx=k.services[w];detail=intent['details']
            assert (tx['src'],tx['dst'])==(detail['src'],detail['dst'])
            return result
        if op=='training_transfer':
            result=k.dispatch_training(w)
            if result is None:raise ValueError('Training transfer proposal changed before commit')
            assert k.helper_jobs[intent['peer']]['batch_id']==intent['details']['batch_id']
            return result
        if op=='run_remote_batch':
            assert w in k.helper_ready
            job=k.helper_jobs[w];bid=job['batch_id'];assert k.batch_state[bid]=='ready_remote'
            k.batch_state[bid]='running';k.pending_batch[w]=job;k.stages[w]='training'
            k.pending_stage[w]=('training',job['tids'],k.tick)
            k.record_program(w,'training',job['steps'])
            return {'steps':job['steps']}
        if op=='region_step':
            assert c.region_eligible(w)
            c.roles[w]='TRAINING'
            expected={x['trajectory_id'] for x in k.groups[w]}
            if not expected<=k.received[w]:k.stages[w]='data_wait';return k.wait(w)
            if w not in k.jobs:k.jobs[w]=iter(k.training_jobs(w))
            if w not in k.trained:
                try:stage,tids,steps=next(k.jobs[w])
                except StopIteration:k.trained.add(w);k.stages[w]='gradient_ready'
                else:
                    k.stages[w]=stage;k.pending_stage[w]=(stage,tids,k.tick)
                    k.record_program(w,stage,steps);return {'steps':steps}
            if k.finished_all():
                prefix=[k.wake(j,k.rep[w].ranks[0]) for j in k.rep if k.waiting[j]]
                if prefix:return {'steps':prefix+[k.rpc(w,'joint_finish')]}
                k.returned.add(w);return {'steps':[],'finish':True}
            return k.wait(w)
        if op=='rollout_step':
            k.stages[w]='generation';before=len(k.migrations)
            result=AttentionRuntime.step(k,w,it)
            helper=intent['details']['selected_helper']
            if helper is None:assert len(k.migrations)==before
            else:
                assert len(k.migrations)==before+1
                assert k.migrations[-1]['helper']==helper and k.migrations[-1]['source']==w
            if result.get('finish'):return k.wait(w)
            for step in result['steps']:
                if step.get('kind')=='controller_rpc' and step['event']['phase']=='finish':step['event']['phase']='step'
            return result
        raise ValueError('Unknown native intent '+op)


class GradientCoordinator:
    def __init__(self,c):self.c=c
    def can_finalize(self,snapshot):
        return self.c.all_training_contributions_complete(self.c.plan.algorithm)
    def prepare_update(self,route,algorithm,snapshot):
        c=self.c;c.kernel.audit()
        assert algorithm==c.plan.algorithm and route.plan_digest==c.plan.gradient_routing_digest
        return UpdateTicket('native-update-once',c.version.iteration,'native-all-sample-gradients',
            'native-two-lane-dp-RS','native-original-optimizer-owners','actor_v1',float(len(c.kernel.ids)))


class OptimizerOwner:
    def __init__(self,c):self.c=c
    def apply_once(self,ticket):
        c=self.c
        if c.suffix_armed:raise ValueError('Optimizer suffix was already armed')
        c.suffix_armed=True
        # The original ASTRA suffix is already in the static DAG, dependent on
        # all dynamic sequences. Arming does not inject another optimizer step.
        return 'native-precompiled-optimizer-armed'


class WeightPublisher:
    def __init__(self,c):self.c=c
    def publish_and_reshard(self,ticket,target):
        c=self.c
        assert c.suffix_armed and target==c.plan.placement and not c.publication_armed
        c.publication_armed=True
        return 'native-precompiled-actor-v1-publication'
    def all_target_instances_acknowledged(self,publication_id):
        good=self.c.native_completed and self.c.publication_armed
        return Readiness(good,() if good else ('native_suffix_not_completed',),'native-54-rank-coverage')
