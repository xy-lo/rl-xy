import sys,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/workload_planner'),str(ROOT/'src')]
from placement_services import PlacementServices

class Memory:
 def batches(self,items,stage,tp,maximum):
  return [items[i:i+maximum] for i in range(0,len(items),maximum)]
class Catalog:
 def phase(self,stage,tp,lengths):return sum(x*x for x in lengths)/tp
 def decode(self,tp,cap,context):return 90
class PlacementServiceTests(unittest.TestCase):
 def model(self):
  model=object.__new__(PlacementServices)
  model.env=SimpleNamespace(public={'a':{'prompt_length':100},'b':{'prompt_length':100},'c':{'prompt_length':200}},policy=SimpleNamespace(logprob_mb=4,train_mb=4),catalog=Catalog(),catalog_data={'inference_capacity_frontier':{'1':10,'2':10,'4':20}})
  model.forecasts={g:dict(score=900,response_scenarios=[[100]*8]) for g in model.env.public}
  model.cache={};model.memory=Memory();return model
 def test_prompt_and_tp_service_costs_not_redivided_by_region_size(self):
  m=self.model();self.assertEqual(m.group('a',2),3*8*199**2/2)
  self.assertEqual(m.group('a',4),m.group('a',2)/2)
  self.assertGreater(m.group('c',2),m.group('a',2))
  home,_=m.costs({0:[0,1]},{0:[0,1]})
  self.assertEqual(home[0]['a'],m.group('a',4))
 def test_unknown_same_condition_ignores_per_task_scenario_noise(self):
  m=self.model();m.forecasts['b']['response_scenarios']=[[8192]*8]*16
  self.assertEqual(m.group('a',2),m.group('b',2))
 def test_conditional_raw_pool_averaged_by_vector(self):
  m=self.model();m.forecasts['a']['conditional_lengths']=[10]*8+[20]*8
  self.assertEqual(m.group('a',2),(3*8*109**2/2+3*8*119**2/2)/2)
 def test_malformed_conditional_vector_rejected(self):
  m=self.model();m.forecasts['a']['conditional_lengths']=[1]*7
  with self.assertRaises(ValueError):m.group('a',2)
if __name__=='__main__':unittest.main()
