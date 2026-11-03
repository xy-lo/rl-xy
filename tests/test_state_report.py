import unittest
from idea1_controller.state_report import snapshot

class StateReportTests(unittest.TestCase):
 def fixture(self):
  roles={str(w):'idle' for w in range(27)}
  roles.update({'0':'own','1':'helper','2':'training','3':'sample_transfer'})
  rows=[dict(tick=1000,roles=roles,stages={})]
  base=dict(dynamic_origin_s=1e-9,dynamic_end_s=10e-9,total_s=11e-9)
  jobs=[dict(remote=True,start_tick=1000,end_tick=5000,physical_ranks=[4,5])]
  return rows,base,jobs
 def test_native_list_roles_and_string_key_stages(self):
  rows,base,jobs=self.fixture()
  rows[0]['roles']=[rows[0]['roles'][str(w)] for w in range(27)]
  rows[0]['stages']={'2':'training'}
  s=snapshot(rows,base,jobs,2e-9)
  self.assertEqual(s['counts']['training'],2)
  self.assertEqual(s['per_die'][4]['stage'],'training')
 def test_conservation_and_helper_subset(self):
  rows,base,jobs=self.fixture();s=snapshot(rows,base,jobs,2e-9)
  self.assertEqual(sum(s['counts'].values()),54)
  self.assertEqual(s['counts']['training'],2);self.assertEqual(s['training_helper_subset_dies'],2)
  self.assertEqual(s['counts']['idle'],46)
 def test_update_and_prefix_are_not_idle(self):
  rows,base,jobs=self.fixture()
  self.assertEqual(snapshot(rows,base,jobs,0)['counts']['prefix'],54)
  self.assertEqual(snapshot(rows,base,jobs,10e-9)['counts']['update'],54)
 def test_last_event_at_same_tick_wins(self):
  rows,base,jobs=self.fixture();roles={str(w):'idle' for w in range(27)}
  rows.append(dict(tick=6000,roles=roles,stages={}))
  self.assertEqual(snapshot(rows,base,jobs,6e-9)['counts']['idle'],54)
 def test_training_helper_role_conflict_is_rejected(self):
  rows,base,jobs=self.fixture();rows[0]['roles']['2']='own'
  with self.assertRaises(ValueError):snapshot(rows,base,jobs,2e-9)
 def test_outside_interval_rejected(self):
  rows,base,jobs=self.fixture()
  for t in (-1,11e-9):
   with self.assertRaises(ValueError):snapshot(rows,base,jobs,t)
if __name__=='__main__':unittest.main()
