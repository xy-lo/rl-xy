"""Regression for the native DRAIN/data-wait/TP wakeup livelock."""
import unittest,sys
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/idea1_layered/legacy'))
from idea1_controller.native.regions import RegionKernel

class HandoffRegression(unittest.TestCase):
    def kernel(self):
        k=object.__new__(RegionKernel);k.region_of={0:0,1:0,2:0};k.region_members={0:(0,1,2)};k.region_started=set()
        k.groups=[[{'trajectory_id':'sample'}],[],[]];k.received=[set(),set(),set()];k.stages={w:'generation' for w in range(3)}
        k.waiting={0:False,1:True,2:True};k.notify=defaultdict(int);k.rep={w:SimpleNamespace(ranks=(2*w,2*w+1)) for w in range(3)}
        return k,SimpleNamespace(region_eligible=lambda w:True,roles={w:'ROLLOUT' for w in range(3)})
    def test_all_drained_members_wait_for_data_before_activating(self):
        k,c=self.kernel();reply=k.region_step(0,1,c)
        self.assertEqual(k.region_started,set());self.assertEqual(k.stages[0],'data_wait');self.assertTrue(k.waiting[0])
        self.assertEqual(reply['steps'][0]['kind'],'gate_wait')
    def test_last_data_arrival_on_follower_wakes_entire_region_once(self):
        k,c=self.kernel();k.received[0].add('sample');k.waiting={0:True,1:False,2:True}
        reply=k.region_step(1,2,c)
        self.assertEqual(k.region_started,{0})
        signals=[x['key'] for x in reply['steps'] if x['kind']=='gate_signal']
        self.assertEqual(signals,['reclaim-notify-0','reclaim-notify-2'])
        self.assertTrue(any(x['kind']=='local' and x['compute']>0 for x in reply['steps']))
        self.assertEqual(reply['steps'][-1]['event']['phase'],'step')

if __name__=='__main__':unittest.main()
