import unittest
from idea1_controller.mesh_observation import MeshObservation

class MeshTests(unittest.TestCase):
 def raw(self,edges=(),policy='adaptive_xy_yx'):
  return dict(schema='rlxy.mesh-observed/2',width=3,height=2,native_ticks_per_ns=1,policy=policy,edges=list(edges),edge_fields=['src','dst','announced_not_started_bytes','queued_here_bytes','actual_queue_wait_ns','committed_service_ns'])
 def test_xy_default_and_safe_yx(self):
  obs=MeshObservation.parse(self.raw(),10)
  self.assertEqual(obs.paths(0,5),((0,1,2,5),(0,3,4,5)))
  self.assertEqual(obs.transfer(0,5,100,10.,1.)['path'],(0,1,2,5))
  obs=MeshObservation.parse(self.raw([(0,1,1000,500,50,100)]),10)
  self.assertEqual(obs.transfer(0,5,100,10.,1.)['path'],(0,3,4,5))
 def test_fixed_route_observes_same_queue(self):
  obs=MeshObservation.parse(self.raw([(0,1,1000,500,50,100)],'xy'),10)
  r=obs.transfer(0,5,100,10.,1.)
  self.assertEqual(r['path'],(0,1,2,5));self.assertEqual(r['queue_penalty_ns'],100)
 def test_reverse_direction_independent(self):
  obs=MeshObservation.parse(self.raw([(1,0,1000,500,50,100)]),10)
  self.assertEqual(obs.transfer(0,5,100,10.,1.)['queue_penalty_ns'],0)
 def test_freshness_and_zero_payload(self):
  obs=MeshObservation.parse(self.raw(),10)
  self.assertTrue(obs.fresh(10));self.assertFalse(obs.fresh(9));self.assertFalse(obs.fresh(11))
  self.assertEqual(obs.transfer(0,0,100,10.,1.)['service_ns'],0)
  self.assertEqual(obs.transfer(0,5,0,10.,1.)['service_ns'],0)
 def test_packet_pipeline(self):
  obs=MeshObservation.parse(self.raw(),10)
  self.assertEqual(obs.transfer(0,5,65536*2,1024.,2.)['service_ns'],256+6)
 def test_mandatory_cut_remains(self):
  obs=MeshObservation.parse(self.raw([(0,1,100,0,0,10),(3,4,200,0,0,20)]),10)
  for path in obs.paths(0,5):self.assertTrue(any(a%3<1<=b%3 for a,b in zip(path,path[1:])))
  self.assertEqual(obs.cut_work('col',1)['announced_bytes'],300)
 def test_bad_counters_and_identity(self):
  for rows in [[(0,2,1,0,0,1)],[(0,1,1,2,0,1)],[(0,1,1,0,0,float('nan'))],[(0,1,1,0,0,-1)],[(0,1,1,0,0,1)]*2]:
   with self.subTest(rows=rows),self.assertRaises(ValueError):MeshObservation.parse(self.raw(rows),10)
  with self.assertRaises(ValueError):MeshObservation.parse(self.raw(),-1)
 def test_transport_invalid(self):
  obs=MeshObservation.parse(self.raw(),10)
  for size,bw in [(-1,1),(1,0),(1,float('inf'))]:
   with self.subTest(size=size,bw=bw),self.assertRaises(ValueError):obs.transfer(0,5,size,bw,1.)

if __name__=='__main__':unittest.main()
