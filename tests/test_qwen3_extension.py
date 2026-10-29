import unittest,random
from rl_xy.gpu_model import cta_mapping
from rl_xy.operator_ir import Matmul

class ExactCTATest(unittest.TestCase):
 def test_closed_form_matches_enumeration(self):
  rng=random.Random(41)
  for _ in range(120):
   p=Matmul(rng.randint(1,250),rng.randint(1,250),rng.randint(1,150),batches=rng.randint(1,9))
   sms=rng.randint(1,20);tm,tn,tk=[rng.choice([16,32,64]) for _ in range(3)]
   f=[0]*sms;t=[0]*sms;ws=[0]*sms;r=w=blocks=0
   for b in range(p.batches):
    for m in range(0,p.m,tm):
     mc=min(tm,p.m-m)
     for n in range(0,p.n,tn):
      nc=min(tn,p.n-n);kc=min(tk,p.k);sm=blocks%sms
      rr=mc*p.k*p.a_bytes+p.k*nc*p.b_bytes;ww=mc*nc*p.output_bytes
      f[sm]+=2*mc*nc*p.k;t[sm]+=rr+ww;ws[sm]=max(ws[sm],mc*kc*p.a_bytes+kc*nc*p.b_bytes+mc*nc*p.accumulator_bytes)
      r+=rr;w+=ww;blocks+=1
   self.assertEqual(cta_mapping(p,sms,10**9,tm,tn,tk),(tuple(f),tuple(t),tuple(ws),r,w,blocks))
if __name__=='__main__':unittest.main()

class Qwen3ProtocolTest(unittest.TestCase):
 def setUp(self):
  from pathlib import Path
  self.root=Path(__file__).resolve().parents[1]
  self.inputs=self.root/'experiments/2026-10-29-qwen3/inputs'
 def test_model_counts_and_independent_query_width(self):
  from rl_xy.qwen3_cost import model_case
  from rl_xy.model import enumerate_parameters
  for size,expected in [('4B',4022468096),('8B',8190735360),('14B',14768307200)]:
   case,info=model_case(self.root,self.inputs/f'Qwen3-{size}/config.json')
   self.assertEqual(sum(x.unique_elements for x in enumerate_parameters(case)),expected)
   if size=='4B':self.assertEqual(case.model.query_width,4096);self.assertEqual(case.model.hidden_size,2560)
 def test_decode_event_integration_matches_tokens(self):
  from rl_xy.qwen3_cost import Operators
  from rl_xy.qwen3_native import NetworkProfiles
  from rl_xy.qwen3_rollout import Rollout
  op=Operators(self.root,self.inputs/'Qwen3-4B/config.json','gpu');roll=Rollout(op,NetworkProfiles(op,'unused'))
  rng=random.Random(191)
  for tp in [1,2,4,8]:
   for cap in [1,2,4]:
    xs=[{'trajectory_id':str(i),'prompt_length':rng.randrange(20,150),'response_length':rng.randrange(1,30)} for i in range(8)]
    a=roll.replica(xs,tp,cap);b=roll.replica(xs,tp,cap,token_step=True)
    self.assertLess(abs(a['model_ns']-b['model_ns']),0.01)
    self.assertLess(max(abs(a['completion_ns'][k]-b['completion_ns'][k]) for k in a['completion_ns']),0.01)
    self.assertEqual(a['actual_KV_peak_bytes'],b['actual_KV_peak_bytes'])
 def test_decode_affinity_with_ragged_contexts(self):
  from rl_xy.qwen3_cost import Operators,Cost
  from rl_xy.qwen3_native import NetworkProfiles
  from rl_xy.qwen3_rollout import Rollout
  op=Operators(self.root,self.inputs/'Qwen3-14B/config.json','gpu');r=Rollout(op,NetworkProfiles(op,'unused'))
  for tp in [1,2,4,8]:
   context=(37,184,3000,10000);a,b,_=r.decode_affine(tp,len(context));derived=a+b*(sum(context)-len(context))
   direct=sum((x.cost for x in op.forward((1,)*len(context),tp,'generation',context)),Cost())
   self.assertLess(abs(derived.ns-direct.ns),0.01)
 def test_generation_admission_does_not_use_future_R(self):
  from rl_xy.qwen3_cost import Operators
  op=Operators(self.root,self.inputs/'Qwen3-14B/config.json','wafer')
  a=[{'prompt_length':100,'response_length':1}]*8
  b=[{'prompt_length':100,'response_length':8192}]*8
  aa=op.admission(a,4,2,1,1,8);bb=op.admission(b,4,2,1,1,8)
  self.assertEqual(aa['phase_peak_bytes']['generation'],bb['phase_peak_bytes']['generation'])
