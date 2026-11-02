import copy
import os
import subprocess
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from idea1_controller.history import digest, validate_history
from idea1_controller.quotas import layout_quotas
from idea1_controller.topology import ring_order,hops
from idea1_controller.system import SystemPlanner
from idea1_controller.cost_catalog import service_contract,validate_catalog


class SystemContracts(unittest.TestCase):
    def test_planning_import_is_independent_of_native_legacy(self):
        root=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=str(root/'src'))
        code="from idea1_controller.system import SystemPlanner; import sys; assert 'training_coop' not in sys.modules; assert 'locality_runtime_v3' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],cwd=root,env=env,check=True,capture_output=True)

    def test_service_directory_rejects_different_hardware_and_math(self):
        @dataclass
        class Hardware:
            bandwidth:int=2000
        op=SimpleNamespace(hw=Hardware(),local_timing_identity={'source':'frozen'},compute_scale=1,network_scale=1,
            runtime_cost_identity={'assets_manifest_sha256':'assets'})
        catalog=dict(model_id='8B',model_config_sha256='config',service_contract=service_contract(op))
        self.assertTrue(validate_catalog(catalog,op,'8B','config'))
        op.hw.bandwidth=4000
        with self.assertRaisesRegex(ValueError,'hardware'):
            validate_catalog(catalog,op,'8B','config')

    def test_ring_changes_physical_order_without_losing_members(self):
        members=(0,1,6,7,12,13,18,19)
        ring=ring_order(members)
        self.assertEqual(set(ring),set(members))
        distance=lambda xs:sum(hops(a,b) for a,b in zip(xs,xs[1:]+xs[:1]))
        self.assertLessEqual(distance(ring),distance(members))

    def test_future_response_never_enters_planner(self):
        planner=SystemPlanner(None,None,None)
        with self.assertRaisesRegex(ValueError,'live response'):
            planner.build([dict(group_id='a',prompt_length=12,response_length=8192)],{})

    def test_region_first_quotas_reserve_real_integer_samples(self):
        regions={0:(0,),1:(1,2),3:(3,4,5,6),7:tuple(range(7,11)),11:tuple(range(11,15)),15:tuple(range(15,19)),19:tuple(range(19,23)),23:tuple(range(23,27))}
        quotas=layout_quotas(192,regions)
        self.assertEqual(sum(quotas.values()),192)
        self.assertEqual(set(quotas),set(range(27)))
        for ws in regions.values():
            self.assertLess(abs(sum(quotas[w] for w in ws)-192*len(ws)/27),1)

    def test_same_task_history_cannot_cross_actor_config_or_prompt(self):
        public=[dict(group_id='1',prompt_length=10,prompt_sha256='abc',library='numpy')]
        h=dict(model='4B',model_config_sha256='cfg',task_bindings={'1':{k:v for k,v in public[0].items() if k!='group_id'}},records={'1':[]})
        h['snapshot_id']=digest(h)
        self.assertTrue(validate_history(public,h,'4B','cfg'))
        with self.assertRaisesRegex(ValueError,'configuration'):
            validate_history(public,h,'4B','other')
        with self.assertRaisesRegex(ValueError,'identity'):
            validate_history([dict(public[0],prompt_sha256='changed')],h,'4B','cfg')
        bad=copy.deepcopy(h);bad['records']['1']=[{'response_length':999}]
        with self.assertRaisesRegex(ValueError,'digest'):
            validate_history(public,bad,'4B','cfg')


if __name__=='__main__':unittest.main()
