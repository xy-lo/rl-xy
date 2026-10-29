import json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
from rl_xy.gpu_calibration import GEMMCalibration,CalibratedGPUCostProvider
from rl_xy.operator_ir import Matmul,KernelSpec
from rl_xy.cost_model import CostEstimate

class CalibrationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'profile.json'
  def row(m,t,split):
   return dict(problem=asdict(Matmul(m,64,128)),duration_ns=t,split=split,numeric_sample_passed=True)
  self.data=dict(schema_version='rlxy.gpu_calibration/1',status='PASS',device='NVIDIA A100-SXM4-80GB',scope='fixture only',numeric_fixture_checks=[dict(passed=True)],measurements={'a':row(10,100,'calibration'),'b':row(30,300,'calibration'),'heldout':row(20,9999,'validation')})
  self.save()
 def save(self):self.path.write_text(json.dumps(self.data))
 def tearDown(self):self.tmp.cleanup()
 def test_validation_does_not_leak_into_prediction(self):
  c=GEMMCalibration(self.path)
  self.assertEqual(c.predict(Matmul(20,64,128)),200)
 def test_outside_domain_is_rejected(self):
  c=GEMMCalibration(self.path)
  for p in (Matmul(1,64,128),Matmul(31,64,128),Matmul(20,65,128)):
   with self.assertRaises(ValueError):c.predict(p)
 def test_inclusive_measurement_does_not_add_reference_memory_or_launch(self):
  class Reference:
   def estimate(self,spec,hw):return CostEstimate(9000,1000,8000,20,30,16,(),core_compute_ns=(1000,))
  p=CalibratedGPUCostProvider(Reference(),GEMMCalibration(self.path))
  result=p.estimate(KernelSpec('test',matmuls=(Matmul(20,64,128),)),None)
  self.assertEqual(result.duration_ns,200);self.assertEqual(result.local_memory_ns,0)
  self.assertEqual(result.core_compute_ns,(0,))
 def test_profile_identity_changes_with_measurements(self):
  before=GEMMCalibration(self.path).identity['sha256']
  self.data['measurements']['a']['duration_ns']=101;self.save()
  self.assertNotEqual(before,GEMMCalibration(self.path).identity['sha256'])
 def test_incomplete_or_wrong_hardware_profile_rejected(self):
  for key,value in [('status','RUNNING'),('device','different GPU')]:
   original=self.data[key];self.data[key]=value;self.save()
   with self.assertRaises(ValueError):GEMMCalibration(self.path)
   self.data[key]=original
 def test_decode_attention_remains_affine_for_native_repeat_program(self):
  self.data['attention_models']={'8/2/128/decode':{'coefficients':[100,1,2,3,4,5,6]}};self.save()
  c=GEMMCalibration(self.path)
  a=c.predict_attention((1,1),(11,23),8,2,128)
  b=c.predict_attention((1,1),(12,24),8,2,128)
  end=c.predict_attention((1,1),(21,33),8,2,128)
  self.assertAlmostEqual(end,a+10*(b-a))
 def test_attention_explicit_domain_rejects_unvalidated_geometry(self):
  self.data['attention_models']={'8/2/128/decode':{'coefficients':[100,1,2,3,4,5,6]}}
  self.data['require_attention_domain']=True
  self.data['attention_domains']={'8/2/128/decode':{'batch':[1,4],'query_tokens':[1,4],'context_tokens':[1,1000],'max_query':[1,1],'max_context':[1,500]}}
  self.save();c=GEMMCalibration(self.path)
  self.assertGreater(c.predict_attention((1,1),(11,23),8,2,128),0)
  for q,ctx in [((1,)*5,(20,)*5),((1,),(501,)),((1,2),(20,))]:
   with self.assertRaises(ValueError):c.predict_attention(q,ctx,8,2,128)
 def test_attention_required_domain_cannot_be_omitted(self):
  self.data['attention_models']={'8/2/128/decode':{'coefficients':[100,1,2,3,4,5,6]}}
  self.data['require_attention_domain']=True;self.save();c=GEMMCalibration(self.path)
  with self.assertRaises(ValueError):c.predict_attention((1,),(20,),8,2,128)
if __name__=='__main__':unittest.main()
