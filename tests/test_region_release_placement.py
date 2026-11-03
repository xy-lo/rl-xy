import unittest
from collections import Counter
from idea1_controller.region_release_placement import balanced_release_placement

class ReleasePlacementTests(unittest.TestCase):
 def args(self):
  public=[dict(group_id=str(i)) for i in range(8)]
  f={str(i):dict(score=100 if i<4 else 1,label='proxy-long' if i<4 else 'ordinary') for i in range(8)}
  return public,f
 def test_training_long_first_preserves_home_and_inference_order(self):
  p,f=self.args();p=[p[i] for i in (4,0,5,1,6,2,7,3)]
  args=(p,f,{0:[0,1],2:[2,3]},{0:2,1:2,2:2,3:2},{0:[0,1,2,3]})
  a=balanced_release_placement(*args,training_queue_order='input')
  b=balanced_release_placement(*args)
  self.assertEqual(a['execution_queues'],b['execution_queues'])
  self.assertEqual(a['proof']['group_to_home_digest'],b['proof']['group_to_home_digest'])
  self.assertNotEqual(a['queues'],b['queues'])
  for worker in a['queues']:
   self.assertEqual(Counter(a['queues'][worker]),Counter(b['queues'][worker]))
   scores=[f[g]['score'] for g in b['queues'][worker]]
   self.assertEqual(scores,sorted(scores,reverse=True))
 def test_regional_quota_and_no_long_packing(self):
  p,f=self.args();r=balanced_release_placement(p,f,{0:[0,1],2:[2,3]},{0:2,1:2,2:2,3:2},{0:[0],1:[1],2:[2],3:[3]})
  self.assertEqual(r['proof']['long_quota_by_region'],{'0':2,'2':2})
  self.assertTrue(all(sum(f[g]['label']=='proxy-long' for g in q)==1 for q in r['queues'].values()))
 def test_producer_locality_does_not_require_identical_partitions(self):
  p,f=self.args();r=balanced_release_placement(p,f,{0:[0,1],2:[2,3]},{0:2,1:2,2:2,3:2},{0:[0,1,2,3]})
  self.assertEqual(r['proof']['region_dependencies'],{'0':[0],'2':[0]})
  self.assertEqual(Counter(g for q in r['execution_queues'].values() for g in q),Counter(str(i) for i in range(8)))
 def test_unequal_fixed_quotas_and_stable_replay(self):
  p,f=self.args();args=(p,f,{0:[0,1],2:[2,3]},{0:3,1:1,2:2,3:2},{0:[0,1],2:[2,3]})
  r=balanced_release_placement(*args);self.assertEqual(r,balanced_release_placement(*args))
  self.assertEqual([len(r['queues'][str(i)]) for i in range(4)],[3,1,2,2])
 def test_zero_slots_and_unknown_long_labels(self):
  p,f=self.args()
  for x in f.values():x['label']='unknown'
  r=balanced_release_placement(p,f,{0:[0,1],2:[2,3]},{0:0,1:0,2:4,3:4},{0:[0,1],2:[2,3]})
  self.assertEqual(r['queues']['0'],[]);self.assertEqual(r['proof']['long_quota_by_region'],{'0':0,'2':0})
 def test_reject_missing_groups_partition_and_bad_predictions(self):
  p,f=self.args()
  with self.assertRaises(ValueError):balanced_release_placement(p,f,{0:[0,1]},{0:4,1:4},{0:[0]})
  with self.assertRaises(ValueError):balanced_release_placement(p,f,{0:[0,1]},{0:3,1:4},{0:[0,1]})
  f['0']['score']=float('nan')
  with self.assertRaises(ValueError):balanced_release_placement(p,f,{0:[0,1]},{0:4,1:4},{0:[0,1]})

if __name__=='__main__':unittest.main()
