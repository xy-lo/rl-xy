import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/idea1_layered/legacy'))
from idea1_controller.native.regions import order_ready_training_jobs

class ReadyOrderTests(unittest.TestCase):
 def jobs(self):
  before=[('region_data',[],['copy'],None),('old_policy',[],['teacher'],None),('reference',[],['ref'],None),('advantage',[],['adv'],None),('gradient_zero',[],['zero'],None)]
  jobs=[('training',[str(i)],['body'+str(i)],dict(batch_id=str(i),index=i,estimate_local_ns=cost)) for i,cost in enumerate((2.,10.,3.,10.))]
  return before+jobs
 def test_keeps_advantage_quorum_and_bodies_while_prioritizing_cost(self):
  jobs=self.jobs();ordered=order_ready_training_jobs(jobs,'ready_lpt')
  self.assertEqual(ordered[:5],jobs[:5]);self.assertEqual([x[3]['batch_id'] for x in ordered[5:]],['1','3','2','0'])
  for row in ordered[5:]:self.assertIn(row,jobs[5:]);self.assertIs(row,jobs[int(row[3]['batch_id'])+5])
  self.assertEqual([x[3]['batch_id'] for x in jobs[5:]],['0','1','2','3'])
 def test_default_is_exact_unchanged_object(self):
  jobs=self.jobs();self.assertIs(order_ready_training_jobs(jobs,'input'),jobs)
 def test_refuses_cross_phase_reordering_and_invalid_work(self):
  jobs=self.jobs();jobs.insert(6,('gradient_restore',[],[],None))
  with self.assertRaises(ValueError):order_ready_training_jobs(jobs,'ready_lpt')
  jobs=self.jobs();jobs[5][3]['estimate_local_ns']=float('nan')
  with self.assertRaises(ValueError):order_ready_training_jobs(jobs,'ready_lpt')
if __name__=='__main__':unittest.main()
