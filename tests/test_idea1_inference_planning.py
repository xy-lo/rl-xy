"""Safety checks for public-only TP selection and sparse service shapes."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/idea1_layered/legacy'))
from types import SimpleNamespace
from idea1_controller.inference_catalog import InferenceServiceCatalog
from idea1_controller.elastic_planner import ElasticSystemPlanner
from idea1_controller.native.inference_cohorts import CohortKernel


class InferencePlanning(unittest.TestCase):
    def test_sparse_prefill_preserves_total_batch_service_at_uniform_lengths(self):
        c=InferenceServiceCatalog(dict(table={'prefill:4:2':[4.,3.,5.],'prefill:4:6':[12.,7.,11.],'decode:4:2':[8.,2.],'decode:4:6':[16.,6.]}))
        expected=(c.phase('prefill',4,[10]*2)+c.phase('prefill',4,[10]*6))/2
        self.assertAlmostEqual(c.phase('prefill',4,[10]*4),expected)
        self.assertEqual(c.decode(4,4,100),12+4*100)
        with self.assertRaises(ValueError):c.decode(4,7,100)

    def test_tp_bank_identity_and_future_response_rejected_before_planning(self):
        catalog=SimpleNamespace(data=dict(service_contract={'frozen':1},model_config_sha256='abc'))
        bank=dict(selection_window='tune',service_contract={'frozen':1},model_config_sha256='abc')
        planner=ElasticSystemPlanner(None,None,catalog,bank)
        with self.assertRaisesRegex(ValueError,'live response'):planner.build([dict(group_id='x',prompt_length=3,response_length=99)],{})
        for key,value in [('selection_window','test2'),('service_contract',{'frozen':2}),('model_config_sha256','wrong')]:
            bad=dict(bank,**{key:value})
            with self.assertRaises(ValueError):ElasticSystemPlanner(None,None,catalog,bad)

    def test_pending_cannot_borrow_reference_or_started_training_cohort(self):
        k=object.__new__(CohortKernel);k.pending_enabled=True;k.paired_tp1=False;k.inference_owner={0:0,1:1}
        k.inference_members=k.initial_inference_members={0:(0,),1:(1,)}
        item=dict(trajectory_id='a',prompt_length=3);k.rep={0:SimpleNamespace(requests=[item],active=[item],cap=1,pending=None,ranks=(0,1)),1:SimpleNamespace(requests=[],active=[],cap=8,pending=None,ranks=(2,3))}
        k.pending_cohort_leases={};k.services={};k.tx={};k.trained=set();k.waiting={1:True};k.logical_roles={0:'ROLLOUT',1:'TRAINING'}
        k.pending_data_wait=True;k.stages={1:'data_wait'};k.region_of={1:1};k.region_started=set();k.same_domain=lambda a,b:True
        self.assertEqual(k.pending_choice(0),(0,1,[item]))
        k.stages[1]='reference';self.assertIsNone(k.pending_choice(0))
        k.stages[1]='data_wait';k.region_started.add(1);self.assertIsNone(k.pending_choice(0))

if __name__=='__main__':unittest.main()
