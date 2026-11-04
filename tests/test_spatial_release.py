import unittest
from idea1_controller.spatial_release import Slot,optimize_positions

class Catalog:
 def phase(self,stage,tp,prompts):return 0.
 def coefficients(self,stage,tp,batch):return 2e9/tp,0.
class SpatialTests(unittest.TestCase):
 def model(self,groups=4):
  keys=[str(i) for i in range(groups)]
  slots=[Slot(k,'r',(i,),2,8) for i,k in enumerate(keys)]
  queues={k:[k] for k in keys};public={k:dict(prompt_length=1) for k in keys}
  forecast={k:dict(response_scenarios=[(40 if int(k)<2 else 20,)*8]*16) for k in keys}
  return slots,queues,public,forecast
 def run_model(self,args,shared=False,**kw):
  return optimize_positions(*args,Catalog(),can_expand=lambda a,b:abs(int(a.key)-int(b.key))==1,
   migration=lambda a,b,s:dict(duration_ns=0.,edges=[(0,1)] if shared else [(min(a.key,b.key),max(a.key,b.key))]),
   drain=lambda slot,p:0.,training=lambda region,scene:0.,**kw)
 def test_two_blockers_need_joint_opportunity(self):
  args=self.model();r=self.run_model(args)
  self.assertEqual(r['baseline']['values'][0],39e9)
  self.assertEqual(r['candidate']['values'][0],35.5e9)
  self.assertEqual(len(r['candidate']['details'][0]['r']['actions']),2)
  self.assertGreater(r['accepted_maps'],0)
  self.assertEqual(sorted(tuple(q) for q in args[1].values()),sorted(tuple(q) for q in r['execution_queues'].values()))
 def test_shared_edge_is_not_counted_twice(self):
  r=self.run_model(self.model(),shared=True)
  self.assertEqual(r['accepted_maps'],0)
 def test_single_helper_cannot_serve_two_blockers(self):
  r=self.run_model(self.model(3))
  self.assertEqual(r['accepted_maps'],0)
 def test_search_budget_is_global(self):
  args=self.model();r=self.run_model(args,max_maps=2,passes=2)
  self.assertLessEqual(r['checked_maps'],4)
 def test_pending_queue_falls_back_without_claimed_gain(self):
  slots,q,p,f=self.model();slots=[Slot(s.key,s.region,s.members,s.tp,4) for s in slots]
  r=self.run_model((slots,q,p,f))
  self.assertEqual(r['reason'],'PB_RETAINED_PENDING_WORK_NOT_MODELED');self.assertEqual(r['execution_queues'],q)
 def test_no_legal_expansion_retains_pb(self):
  args=self.model()
  r=optimize_positions(*args,Catalog(),can_expand=lambda a,b:False,migration=lambda a,b,s:dict(duration_ns=0.,edges=[]),drain=lambda s,p:0.,training=lambda r,s:0.)
  self.assertEqual(r['accepted_maps'],0)
 def test_no_current_outputs_are_needed(self):
  class Public(dict):
   def __getitem__(self,key):
    if key in ('response_length','eos','finish_reason'):raise AssertionError('future read')
    return super().__getitem__(key)
  a,b,p,f=self.model();p={k:Public(v) for k,v in p.items()}
  self.assertGreater(self.run_model((a,b,p,f))['accepted_maps'],0)

if __name__=='__main__':unittest.main()
