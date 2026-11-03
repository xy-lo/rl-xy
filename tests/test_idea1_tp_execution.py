import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/idea1_layered/legacy'))
from rl_xy.qwen3_cost import Block,Cost
from idea1_controller.native.paired_tp1 import PairedTP1Replica
from idea1_controller.native.inference_cohorts import CohortReplica,CohortKernel,CohortRegion,CohortWorker


class Operator:
    def __init__(self):self.tps=[]
    def forward(self,lengths,tp,*args):
        self.tps.append(tp);return [Block('test',Cost(compute=sum(lengths)))]
    def vec(self,flops,read,write,*args):return Cost(compute=flops,memory=read+write)


class ActualTPExecution(unittest.TestCase):
    def items(self):return [dict(trajectory_id='a',group_id=0,prompt_length=3,response_length=2),dict(trajectory_id='b',group_id=1,prompt_length=3,response_length=4)]

    def test_full_cohort_compiles_actual_tp_without_changing_storage_owners(self):
        op=Operator();r=CohortReplica(op,self.items(),(0,1),8,safe_interval=16);r.execution_ranks=(0,2,1,3)
        r.next(0);self.assertEqual(op.tps,[4]);self.assertEqual(r.ranks,(0,1))

    def test_eos_release_restores_communicator_before_cohort_can_receive_new_pending(self):
        k=object.__new__(CohortKernel);k.inference_members={0:(0,1),1:(1,)};k.initial_inference_members={0:(0,),1:(1,)}
        k.inference_owner={0:0,1:0};k.initial_inference_owner={0:0,1:1};k.inference_communicators={0:99,1:12};k.initial_inference_communicators={0:11,1:12}
        k.inference_program_events=[];k.inference_releases=[];k.inference_saved_roles={};k.tick=100
        k.rep={0:SimpleNamespace(requests=[],active=[],pending=None,execution_ranks=(0,2,1,3))}
        with patch('idea1_controller.native.view_system.ViewKernel._observe',return_value=None):k._observe(0)
        self.assertEqual(k.rep[0].execution_ranks,(0,1));self.assertEqual(k.inference_owner,{0:0,1:1})
        self.assertEqual(k.inference_communicators[0],11)

    def test_unborrowed_training_member_can_wait_for_a_lent_region_member(self):
        c=SimpleNamespace(kernel=SimpleNamespace(services={},wait=lambda w:dict(waiting=w)),w=16,roles={16:'TRAINING'},region_id=lambda w:'R13',is_step=lambda:True,region_eligible=lambda w:False,propose=lambda op,p:dict(op=op,priority=p))
        c.kernel.rep={16:SimpleNamespace(ranks=(32,33))}
        with patch('idea1_controller.native.region_adapter.PlannedRegion.propose',return_value=()):
            proposal=CohortRegion(c).propose('R13',None)[0]
        self.assertEqual(proposal['op'],'cohort_wait');self.assertEqual(CohortWorker(c,16).execute(proposal),dict(waiting=16))

    def test_released_data_wait_leader_gets_fresh_training_decision_without_peer_wakeup(self):
        k=SimpleNamespace(cohort_release_pending={0:[0]},inference_saved_roles={0:dict(role='TRAINING',stage='data_wait')},stages={0:'generation'},inference_role_events=[],tick=100,waiting={0:False},rpc=lambda w,p:dict(kind='controller_rpc',event=dict(worker=w,phase=p)))
        k.rep={0:SimpleNamespace(ranks=(0,1))}
        c=SimpleNamespace(kernel=k,w=0,roles={0:'ROLLOUT'},is_step=lambda:True)
        reply=CohortWorker(c,0).execute(dict(op='rollout_step'))
        self.assertEqual(c.roles[0],'TRAINING');self.assertEqual(k.stages[0],'data_wait');self.assertEqual(reply['steps'][-1]['event']['phase'],'step')

    def test_two_tp1_engines_use_common_repeat_and_deliver_eos_only_after_copy(self):
        op=Operator();tags=iter(range(480000000,480000100));r=PairedTP1Replica(op,self.items(),(0,1),8,lambda:next(tags))
        r.next(0);r.observe();self.assertEqual(r.done,[])
        decode=r.next(1);self.assertEqual(decode['repeat'],1)
        self.assertEqual([e.pending[1][0] for e in r.engines.values()],[1,1])
        r.observe();self.assertEqual(r.done,[]);self.assertEqual(r.pending[0],'copy_needed')
        copied=r.next(2);self.assertEqual(sum(s['bytes'] for s in copied['steps'] if s['kind']=='p2p_direct'),4*(3+2))
        r.observe();self.assertEqual(r.done,['a'])
        self.assertEqual(r.next(3)['repeat'],2);r.observe();self.assertEqual(r.done,['a'])
        r.next(4);r.observe();self.assertEqual(r.done,['a','b']);self.assertEqual(set(op.tps),{1})

    def test_only_unstarted_data_wait_regions_can_temporarily_return_to_inference(self):
        k=object.__new__(CohortKernel);k.elastic_enabled=True;k.inference_owner={0:0,1:1};k.inference_members={0:(0,),1:(1,)}
        k.rep={0:SimpleNamespace(requests=[],active=[[dict(group_id='g',prompt_length=10),20,9999]],pending=None,execution_ranks=(0,1)),
               1:SimpleNamespace(requests=[],active=[],pending=None,ranks=(2,3))}
        k.helper_history={'g':(1000,)*64};k.elastic_groups=((0,1),);k.stages={1:'data_wait'};k.region_of={1:1};k.region_started=set()
        k.services={};k.trained=set();k.waiting={1:True};k.op=SimpleNamespace(model=SimpleNamespace(kv_heads=8,layers=1,head_dim=1,actor_dtype_bytes=2),
            hw=SimpleNamespace(d2d_bytes_per_ns_per_direction=2000,dram_bytes_per_ns=2000,d2d_latency_ns=100),
            forward=lambda lengths,tp,*args:[Block('dummy',Cost(compute=10000/tp))])
        roles={0:'ROLLOUT',1:'TRAINING'}
        self.assertIsNone(k.inference_resize_choice(0,roles))
        k.elastic_data_wait=True;self.assertEqual(k.inference_resize_choice(0,roles)['role_before']['1']['stage'],'data_wait')
        k.pending_cohort_leases={99:(1,)};self.assertIsNone(k.inference_resize_choice(0,roles))
        k.pending_cohort_leases.clear()
        k.region_started.add(1);self.assertIsNone(k.inference_resize_choice(0,roles))
        k.region_started.clear();k.stages[1]='reference';self.assertIsNone(k.inference_resize_choice(0,roles))


if __name__=='__main__':unittest.main()
