from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest

from rl_xy.config import load_case, ConfigError
from rl_xy.communication_lowering import lower_communication_endpoints
from rl_xy.execution_validation import shape_checks
from rl_xy.hardware_profile import load_hardware_profile, inspect_hardware, HardwareProfileError
from rl_xy.tensor_plan import build_tensor_plan, finalize_tensor_costs
from rl_xy.training_schedule import updates_per_round, training_visits
from rl_xy.training_validation import multi_update_checks
from rl_xy.workflow import build_workflow

ROOT = Path(__file__).resolve().parents[1]


def case_with(epochs=2, minibatch=3, rounds=2):
    case = load_case(ROOT/'configs/qwen2_group_fixture/case.json')
    return replace(case, rounds=rounds, strategy=replace(case.strategy,
        training_epochs=epochs, optimizer_minibatch_trajectories=minibatch))


class TrainingScheduleTest(unittest.TestCase):
    def test_tail_minibatch_two_epochs_two_rounds(self):
        c = case_with()
        ops = build_workflow(c)
        self.assertEqual(updates_per_round(c), 4)
        self.assertEqual([o.metadata['published_version'] for o in ops if o.name=='publish_actor_version'], list(range(1,9)))
        self.assertEqual([o.metadata['behavior_version'] for o in ops if o.name=='host_to_rank0_prompt'], [0,4])
        self.assertEqual([o.metadata['minibatch_denominator'] for o in ops if o.name=='adamw_read_modify_write'], [6,3]*4)
        checks = multi_update_checks(c, ops)
        self.assertTrue(all(x.passed for x in checks), [(x.check_id,x.actual) for x in checks if not x.passed])
        self.assertTrue(all(x.passed for x in shape_checks(c,ops)))
        self.assertEqual(len(list(training_visits(c))), 16)

    def test_partial_groups_keep_original_demeaned_advantages(self):
        c = case_with(rounds=1)
        ops = build_workflow(c)
        # Batch ABC splits the second group CD; C/D retain +1/-1, not 0.
        self.assertEqual([o.metadata['advantage'] for o in ops if o.name.endswith('_loss')], [.5,-.5,1.,-1.]*2)
        self.assertEqual(len([o for o in ops if o.name=='group_advantage_demean']),1)

    def test_independent_validator_rejects_six_mutations(self):
        c = case_with(rounds=1)
        ops = build_workflow(c)
        mutations = [
            ('adamw_read_modify_write','actor_version_out',99),
            ('adamw_read_modify_write','behavior_version',99),
            ('zero_gradient_buffer','initial_gradient_value',1),
            ('zero_gradient_buffer','reset_each_update',False),
            ('train_u0_C_loss','advantage',0.),
            ('train_u1_D_loss_backward','global_denominator',9),
        ]
        for name,key,value in mutations:
            bad = list(ops)
            i=next(i for i,o in enumerate(bad) if o.name==name)
            bad[i]=replace(bad[i],metadata={**bad[i].metadata,key:value})
            with self.subTest(name=name,key=key):
                self.assertTrue(any(not x.passed for x in multi_update_checks(c,bad)))
        bad=[o for o in ops if not (o.name=='publish_actor_version' and o.metadata['update_id']==1)]
        self.assertTrue(any(not x.passed for x in multi_update_checks(c,bad)))

    def test_tensor_plan_unique_storage_and_per_visit_tied_gradients(self):
        c=case_with(rounds=1)
        p=build_tensor_plan(c,lower_communication_endpoints(c,build_workflow(c)),'/tmp/not-run')
        finalize_tensor_costs(p,c)
        self.assertEqual(len(p.tensors),len({t['storage_id'] for t in p.tensors}))
        for rid,_,_,_,t,phase in training_visits(c):
            total=sum(x['nbytes'] for x in p.tensors if x['kind']=='parameter_gradient' and x['rank']==0
                      and x['round_id']==rid and phase+'_backward' in x['tensor_id'])
            self.assertEqual(total,(696+64)*4)

    def test_config_rejects_bool_negative_and_legacy_multiupdate(self):
        raw=json.loads((ROOT/'configs/qwen2_group_fixture/case.json').read_text())
        for name,value in [('training_epochs',True),('training_epochs',0),('optimizer_minibatch_trajectories',True),
                           ('optimizer_minibatch_trajectories',-1)]:
            changed=copy.deepcopy(raw);changed['strategy'][name]=value
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'case.json';p.write_text(json.dumps(changed))
                with self.subTest(name=name,value=value),self.assertRaises(ConfigError):load_case(p)
        legacy=json.loads((ROOT/'configs/toy_grpo_2die/case.json').read_text())
        legacy['strategy']['training_epochs']=2
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'case.json';p.write_text(json.dumps(legacy))
            with self.assertRaises(ConfigError):load_case(p)

    def test_new_h0_exact_values_and_geometry_are_not_silently_repaired(self):
        h=load_hardware_profile(ROOT/'configs/hardware/xy_h0_wsc_case3.json')
        r=inspect_hardware(h)
        p=h['parameters']
        self.assertEqual([p['modules_x']['value'],p['modules_y']['value']], [6,9])
        self.assertEqual(p['core_fp16_flops_per_ns']['value'],1020)
        self.assertEqual(r['aggregates']['peak_fp16_flops_per_ns'],14100480)
        self.assertEqual(r['aggregates']['cores'],13824)
        self.assertEqual(r['aggregates']['sram_bytes_sum_not_shared_pool'],17280000000)
        self.assertEqual(r['aggregates']['dram_bytes'],3456000000000)
        self.assertEqual(len(r['geometry_errors']),3)
        self.assertAlmostEqual(r['geometry_details']['unrotated_grid_mm'][0],149.94)
        self.assertAlmostEqual(r['geometry_details']['unrotated_grid_mm'][1],299.25)
        self.assertIsNone(p['wafer_diameter_mm']['value'])
        self.assertIsNone(p['sram_bytes_per_ns_per_core']['value'])
        self.assertIsNone(p['noc_bytes_per_ns_per_module']['value'])
        self.assertFalse(r['current_full_grpo_executable'])

    def test_h0_geometry_source_unit_and_finiteness_fail_closed(self):
        raw=json.loads((ROOT/'configs/hardware/xy_h0_wsc_case3.json').read_text())
        for key,value in [('unit','cm'),('source',''),('value',float('nan')),('value',True)]:
            changed=copy.deepcopy(raw);changed['declared_geometry']['array_width_mm'][key]=value
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'h.json';p.write_text(json.dumps(changed))
                with self.subTest(key=key,value=value),self.assertRaises(HardwareProfileError):load_hardware_profile(p)
