"""TP followers must observe the root's native run/skip ticket before work."""
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/idea1_layered/legacy'))
from idea1_controller.native.region_training_help import SingletonTrainingHelp


class TicketRegression(unittest.TestCase):
    def kernel(self):
        k=SingletonTrainingHelp();k.tick=0;k.region_of={0:0,1:0};k.region_members={0:(0,1)}
        k.rep={0:SimpleNamespace(ranks=(0,1)),1:SimpleNamespace(ranks=(2,3))}
        jobs=[dict(batch_id='0:'+str(i),tids=['sample'+str(i)],steps=[dict(kind='local',compute=10)]) for i in range(2)]
        k.region_programs={0:[('gradient_zero',[],[],None)]+[('training',j['tids'],j['steps'],j) for j in jobs]}
        k.train_queues={};k.pending_batch={};k.batch_state={j['batch_id']:'queued' for j in jobs}
        return k,jobs

    def test_follower_waits_for_skip_and_does_not_start_a_donated_batch(self):
        k,jobs=self.kernel();root=k.training_jobs(0);follower=k.training_jobs(1)
        self.assertEqual(next(root)[0],'gradient_zero');self.assertEqual(next(follower)[0],'gradient_zero')
        k.train_queues[0].remove(jobs[0]);k.batch_state['0:0']='transferring'
        wait=next(follower)
        self.assertEqual(wait[0],'training_ticket');self.assertNotIn('0:0',k.training_tickets)
        self.assertTrue(any(s['kind']=='local' and s['compute']>0 for s in wait[2]))
        next(root);self.assertFalse(k.training_tickets['0:0'])
        # A real native gate blocks the follower until the preceding publish.
        wait1=next(follower);self.assertEqual(wait1[2][-1]['key'],'training-ticket-0:1')
        next(root);self.assertTrue(k.training_tickets['0:1'])
        body=next(follower);self.assertEqual(body[0],'training');self.assertEqual(body[1],[])
        self.assertEqual(k.pending_batch[1]['batch_id'],'0:1')
        self.assertEqual(next(root)[1],['sample1'])

    def test_followers_never_reset_the_donor_queue(self):
        k,jobs=self.kernel();root=k.training_jobs(0);next(root)
        k.train_queues[0].remove(jobs[0]);k.batch_state['0:0']='ready_remote'
        next(k.training_jobs(1))
        self.assertEqual([j['batch_id'] for j in k.train_queues[0]],['0:1'])

    def test_ready_regions_can_help_before_unrelated_generation_finishes(self):
        k=SingletonTrainingHelp();k.region_of={0:0,1:1};k.region_members={0:(0,),1:(1,)}
        k.rep={0:SimpleNamespace(ranks=(0,1)),1:SimpleNamespace(ranks=(2,3))}
        k.leased=lambda w:False;k.done={'ready'};k.ids={'ready','unrelated_rollout'}
        jobs=[dict(batch_id=str(i),estimate_local_ns=10e6,index=i,payload_bytes_per_lane=1024) for i in range(2)]
        k.train_queues={0:deque(jobs)};k.trained={1};k.waiting={1:True};k.services={};k.helper_jobs={};k.training_scope='wafer'
        k.op=SimpleNamespace(hw=SimpleNamespace(d2d_bytes_per_ns_per_direction=2000,dram_bytes_per_ns=2000))
        k.portable_training_job=lambda job:dict(estimate_ns=5e6)
        k.resource_layout={'training_help_admission':'global-eos'}
        self.assertIsNone(k.training_help_choice(0))
        k.resource_layout['training_help_admission']='region-ready'
        self.assertEqual(k.training_help_choice(0)[0],1)
        k.trained.clear()
        self.assertIsNone(k.training_help_choice(0))


if __name__=='__main__':unittest.main()
