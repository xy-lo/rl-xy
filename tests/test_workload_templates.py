import unittest,sys,importlib.util
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/workload_planner'))
spec=importlib.util.spec_from_file_location('workload_script_search',ROOT/'scripts/workload_planner/search.py')
search=importlib.util.module_from_spec(spec);spec.loader.exec_module(search)

class TemplateCoverageTests(unittest.TestCase):
 def test_maximal_tp4_count_is_present(self):
  layouts=search.templates()
  for name in ('packed2_head','packed2_tail'):
   self.assertEqual(Counter(2*len(g) for g in layouts[name].values()),{4:13,2:1})
   self.assertEqual(sorted(w for group in layouts[name].values() for w in group),list(range(27)))
 def test_dominoes_are_physical_neighbors(self):
  for name,groups in search.templates().items():
   if name.startswith('packed2'):
    for group in groups.values():
     if len(group)==2:
      a,b=group;self.assertEqual(abs(a//3-b//3)+abs(a%3-b%3),1)
 def test_top_and_bottom_remainders_are_distinct(self):
  layouts=search.templates();a=layouts['box4'];b=layouts['box4_top']
  self.assertNotEqual(a,b)
  self.assertEqual(Counter(map(len,a.values())),Counter(map(len,b.values())))
  self.assertIn(0,b);self.assertEqual(b[0],[0]);self.assertEqual(b[1],[1,2])
if __name__=='__main__':unittest.main()
