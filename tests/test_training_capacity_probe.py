"""One exact training stage is only a DRAM lower bound, never workflow fit."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest

from rl_xy.baseline_contract import BaselinePolicy
from rl_xy.config import load_case
from rl_xy.distributed_optimizer import layout_for_case

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('probe_training_capacity',ROOT/'scripts/probe_training_capacity.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def fixture():
    case=load_case(ROOT/'configs/wafer4_fixture/case.json');sample=case.workload.trajectories[0]
    ts=tuple(replace(sample,trajectory_id='X'+str(i),group_id='G'+str(i//2),reward=float(i%2)) for i in range(8))
    case=replace(case,hardware=replace(case.hardware,dram_bytes_per_die=10**7),workload=replace(case.workload,trajectories=ts))
    policy=BaselinePolicy(prompt_batch_size=4,ppo_mini_batch_prompts=4,train_tp=2,rollout_tp=1,
        train_micro_batch_trajectories=2,logprob_micro_batch_trajectories=2,use_kl_loss=True,kl_coefficient=.001)
    return case,policy


class TrainingCapacityProbeTests(unittest.TestCase):
    def test_full_global_DO_resident_and_global_denominator_survive_single_mb_compile(self):
        case,policy=fixture()
        result=probe.measure_training_stage(case,policy,allocated=6,active=4,tp=2,microbatch=2)
        self.assertEqual(result['full_global_trajectory_count'],8)
        self.assertEqual(result['full_global_token_denominator'],24)
        self.assertEqual(result['global_policy']['prompt_batch_size']*result['global_policy']['responses_per_prompt'],8)
        self.assertEqual(len(result['sampled_source_trajectory_ids']),2)
        self.assertEqual(set(result['gradient_zero_per_sampled_rank'].values()),{1})
        self.assertEqual(set(result['gradient_RMW_per_sampled_rank'].values()),{1})
        local=replace(case,hardware=replace(case.hardware,dies=2),strategy=replace(case.strategy,tensor_parallel=2))
        layout=layout_for_case(local,dp_size=2)
        for rank in range(4):
            self.assertEqual(result['initial_resident_bytes_by_rank'][str(rank)],layout.state_bytes(rank//2)['total_allocated_bytes'])
        self.assertEqual(result['initial_resident_bytes_by_rank']['4'],0)
        self.assertEqual(result['initial_resident_bytes_by_rank']['5'],0)
        self.assertGreater(result['peak_required_bytes'],max(result['initial_resident_bytes_by_rank'].values()))
        self.assertFalse(result['full_workflow_fit']);self.assertFalse(result['modeled_performance_result'])
        self.assertFalse(result['native_executed']);self.assertFalse(result['GPU_executed'])
        self.assertFalse(result['SRAM_workspace_validated']);self.assertTrue(result['external_carriers_omitted'])
        self.assertEqual(policy.train_micro_batch_trajectories,2)
        self.assertEqual(case.hardware.dram_bytes_per_die,10**7)

    def test_overflow_gets_exact_required_peak_and_cannot_be_labeled_fit(self):
        case,policy=fixture()
        good=probe.measure_training_stage(case,policy,allocated=4,active=4,tp=2,microbatch=2)
        low=probe.measure_training_stage(case,policy,allocated=4,active=4,tp=2,microbatch=2,target_capacity_bytes=1)
        self.assertEqual(good['peak_required_bytes'],low['peak_required_bytes'])
        self.assertTrue(low['sampled_stage_exceeds_physical_capacity'])
        self.assertFalse(low['sampled_stage_within_guarded_capacity'])
        self.assertEqual(low['physical_target_headroom_bytes'],1-low['peak_required_bytes'])
        self.assertGreater(low['measurement_compiler_admission_ceiling_bytes'],low['physical_target_capacity_bytes'])
        self.assertTrue(low['measurement_ceiling_is_not_hardware_capacity'])

    def test_exact_candidates_are_built_independently_and_guard_is_separate(self):
        case,policy=fixture();peaks=[]
        for mb in (1,2,4):
            row=probe.measure_training_stage(case,policy,allocated=4,active=4,tp=2,microbatch=mb,guard_bytes=10**9)
            self.assertEqual(row['microbatch_trajectories'],mb);self.assertTrue(row['component_checks_pass'])
            self.assertEqual(row['global_policy']['train_micro_batch_trajectories'],mb)
            self.assertEqual(row['full_global_token_denominator'],24)
            self.assertFalse(row['sampled_stage_exceeds_physical_capacity'])
            self.assertFalse(row['sampled_stage_within_guarded_capacity'])
            peaks.append(row['peak_required_bytes'])
        # Each actual shape is materialized; this is not a linear MB extrapolator.
        self.assertGreater(len(set(peaks)),1)
        row=probe.measure_training_stage(case,policy,allocated=4,active=4,tp=1,microbatch=2)
        self.assertEqual((row['train_tp'],row['data_parallel']),(1,4))

    def test_invalid_policy_shape_and_guard_fail_closed(self):
        case,policy=fixture()
        with self.assertRaisesRegex(ValueError,'policy invalid'):
            probe.measure_training_stage(case,policy,allocated=4,active=4,tp=2,microbatch=3)
        with self.assertRaisesRegex(ValueError,'guard bytes'):
            probe.measure_training_stage(case,policy,allocated=4,active=4,tp=2,microbatch=2,guard_bytes=-1)
        ts=list(case.workload.trajectories);ts[-1]=replace(ts[-1],response_token_ids=(1,))
        bad=replace(case,workload=replace(case.workload,trajectories=tuple(ts)))
        with self.assertRaisesRegex(ValueError,'homogeneous'):
            probe.measure_training_stage(bad,policy,allocated=4,active=4,tp=2,microbatch=2)

    def test_cli_declares_requested_matrix_without_starting_it(self):
        args=probe.build_parser().parse_args(['--case','c','--policy','p','--scenario','w','--gpu-profile','g','--out','o'])
        self.assertEqual(args.train_tps,[1,2]);self.assertEqual(args.microbatches,[1,2,4,8,16])
        self.assertEqual(args.platforms,['GPU','WAFER']);self.assertEqual(args.guard_bytes,0)
        self.assertEqual(probe.OVERALL_SECONDS,180)
        self.assertLess(probe.PROCESS_ADDRESS_SPACE_LIMIT,8*1024**3)


if __name__=='__main__':unittest.main()
