import copy
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import unittest

from rl_xy.baseline_contract import (BaselinePolicy, audit_policy, audit_direct_transfer,
    candidate_policies, capacity_lower_bounds)
from rl_xy.config import load_case
from rl_xy.gpu_model import load_gpu_profile, effective_gpu_case
from rl_xy.operator_ir import Matmul

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validate_gpu_costs', ROOT / 'scripts/validate_gpu_costs.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class BaselineContractTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case(ROOT / 'configs/verl_sync_standard/qwen3b/case.json')
        self.policy = BaselinePolicy(distributed_optimizer=False)

    def test_upstream_distributed_optimizer_default_is_explicitly_blocked(self):
        policy = BaselinePolicy()
        result = audit_policy(self.case.model, policy)
        self.assertTrue(policy.distributed_optimizer)
        self.assertIn('distributed_optimizer_state_and_collectives', result['runtime_capabilities_missing'])
        self.assertFalse(result['runtime_alignment_ready'])
        self.assertFalse(capacity_lower_bounds(self.case, policy, {'GPU': 80_000_000_000})['scope_supported'])

    def test_prompt_units_and_official_DP_minibatch_constraint(self):
        good = audit_policy(self.case.model, self.policy)
        self.assertEqual(good['trajectory_batch_size'], 96)
        self.assertEqual(good['trajectory_minibatch_per_train_dp'], 4)
        self.assertTrue(good['official_batch_constraints_pass'])
        self.assertFalse(good['runtime_alignment_ready'])
        bad = audit_policy(self.case.model, replace(self.policy, prompt_batch_size=32, ppo_mini_batch_prompts=32))
        self.assertFalse(bad['official_batch_constraints_pass'])
        self.assertTrue(any('minibatch_not_divisible' in x for x in bad['blockers']))

    def test_static_microbatch_and_no_KV_replication_are_explicit(self):
        bad = audit_policy(self.case.model, replace(self.policy, train_micro_batch_trajectories=3))
        self.assertTrue(any('static_train_microbatch' in x for x in bad['blockers']))
        bad = audit_policy(self.case.model, replace(self.policy, train_tp=4))
        self.assertTrue(any('KV_heads' in x for x in bad['blockers']))

    def test_microbatch_more_than_one_requires_real_packed_execution(self):
        result = audit_policy(self.case.model, replace(self.policy,
            train_micro_batch_trajectories=2, logprob_micro_batch_trajectories=2))
        self.assertIn('multi_trajectory_train_microbatch', result['runtime_capabilities_missing'])
        self.assertIn('multi_trajectory_logprob_microbatch', result['runtime_capabilities_missing'])

    def test_candidate_enumeration_does_not_select_a_winner(self):
        rows = list(candidate_policies(self.case.model))
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(not row['blockers'] for _, row in rows), 8)
        self.assertTrue(all(not row['runtime_alignment_ready'] for _, row in rows))

    def test_strict_transfer_forbids_extra_DP_or_different_TP(self):
        good = audit_direct_transfer(self.policy, self.policy, barrier_participants=range(48))
        self.assertFalse(good['blockers'])
        self.assertEqual(good['wafer_idle_ranks'], list(range(48, 54)))
        bad = audit_direct_transfer(self.policy, replace(self.policy, train_tp=1), barrier_participants=range(48))
        self.assertIn('software_policy_changed_during_transfer', bad['blockers'])
        bad = audit_direct_transfer(self.policy, self.policy, barrier_participants=range(54))
        self.assertIn('barrier_participants_differ_from_selected_active_world', bad['blockers'])

    def test_capacity_lower_bound_rejects_impossible_selected_TP1_with_reference(self):
        p = replace(self.policy, train_tp=1, use_kl_loss=True, kl_coefficient=.01)
        bounds = capacity_lower_bounds(self.case, p, {'GPU': 80_000_000_000, 'wafer': 64_000_000_000})
        self.assertFalse(bounds['platforms']['GPU']['lower_bound_exceeds_capacity'])
        self.assertTrue(bounds['platforms']['wafer']['lower_bound_exceeds_capacity'])
        self.assertTrue(bounds['passing_lower_bound_is_not_capacity_admission'])
        self.assertEqual(sum(bounds['bytes_per_train_local_parameter_by_role'].values()), 20)

    def test_PP_and_sharded_optimizer_memory_are_not_false_lower_bounds(self):
        for policy in [replace(self.policy, pipeline_parallel=2),
                       replace(self.policy, distributed_optimizer=True), replace(self.policy, offload=True)]:
            self.assertFalse(capacity_lower_bounds(self.case, policy, {'GPU': 80_000_000_000})['scope_supported'])

    def test_invalid_boolean_integer_never_reaches_division(self):
        result = audit_policy(self.case.model, replace(self.policy, train_tp=True))
        self.assertEqual(result['status'], 'INVALID_CONFIGURATION')


class GPUValidationTests(unittest.TestCase):
    def test_error_statistics_keeps_signed_and_absolute_error(self):
        rows = [{'predicted_hot_ns': 200, 'observed_hot_ns': 100},
                {'predicted_hot_ns': 50, 'observed_hot_ns': 100}]
        result = validation.error_statistics(rows)
        self.assertEqual(result['MAPE'], .75)
        self.assertEqual(result['WAPE_unique_shapes'], .75)
        self.assertEqual(result['sum_predicted_over_observed_unique_shapes'], 1.25)
        self.assertEqual(result['mean_signed_relative_error'], .25)

    def test_label_integrity_and_unseen_domain_gate(self):
        mat = asdict(Matmul(32, 64, 64));key = validation.signature(mat)
        labels = {'schema_version': 'rlxy.gemm_measurements/1', 'status': 'PASS',
            'device': 'NVIDIA A100-SXM4-80GB', 'numeric_fixture_checks': [{'passed': True}],
            'measurements': {key: {'problem': mat, 'duration_ns': 5000, 'repeat_relative_difference': .01}}}
        profile = load_gpu_profile(ROOT / 'configs/gpu/a100_48_reference.json')
        case = effective_gpu_case(load_case(ROOT / 'configs/wafer4_fixture/case.json'), profile)
        other = asdict(Matmul(33, 64, 64));other_key = validation.signature(other)
        result = validation.audit(labels, profile, case.hardware, required_shapes={key: mat, other_key: other})
        self.assertEqual(result['required_domain']['covered_signature_fraction'], .5)
        self.assertEqual(result['required_domain']['missing_signature_sha256'], [other_key])
        self.assertFalse(result['paper_performance_ready'])
        bad = copy.deepcopy(labels);bad['measurements'][key]['problem']['m'] = 33
        with self.assertRaises(ValueError):
            validation.validate_labels(bad)
        with self.assertRaises(ValueError):
            validation.audit(labels, {**profile, 'gemm_measurements': labels['measurements']}, case.hardware)

    def test_holdout_family_is_reproducible_and_precision_sensitive(self):
        mat = asdict(Matmul(65, 128, 256))
        adjacent = {**mat, 'm': 66}
        self.assertEqual(validation.shape_family(mat), validation.shape_family(adjacent))
        self.assertEqual(validation.split_for(mat), validation.split_for(adjacent))
        self.assertNotEqual(validation.shape_family(mat), validation.shape_family({**mat, 'output_bytes': 4}))


if __name__ == '__main__':
    unittest.main()
