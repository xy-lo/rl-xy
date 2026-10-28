from dataclasses import replace
from pathlib import Path
import unittest
from rl_xy.config import load_case
from rl_xy.colocation import StageBuilder
from rl_xy.communication_lowering import lower_communication_endpoints
from rl_xy.tensor_plan import build_tensor_plan,finalize_tensor_costs,preflight_contract
from rl_xy.uniform_generation import replicate_uniform_generation
from rl_xy.batching import fuse_generation
ROOT=Path(__file__).resolve().parents[1]
class UniformGenerationTests(unittest.TestCase):
    def setup_case(self,tp):
        c=load_case(ROOT/'configs/wafer4_fixture/case.json');a=c.workload.trajectories[0]
        c=replace(c,hardware=replace(c.hardware,dies=tp,dram_bytes_per_die=10**8),strategy=replace(c.strategy,tensor_parallel=tp))
        ts=tuple(replace(a,trajectory_id='X'+str(i),group_id='g') for i in range(3))
        return replace(c,workload=replace(c.workload,trajectories=ts))
    def plan(self,c):
        b=StageBuilder(c,{'input','rollout','cleanup'});b.build_round(0)
        return build_tensor_plan(c,lower_communication_endpoints(c,b.operations),'/test')
    def test_uniform_matches_serial_then_fused_cost_and_live_bytes(self):
        for tp in [1,2]:
            c=self.setup_case(tp);single=replace(c,workload=replace(c.workload,trajectories=c.workload.trajectories[:1]))
            new=replicate_uniform_generation(self.plan(single),single,c);old=fuse_generation(self.plan(c),c)
            for plan in [old,new]:
                for r in plan.contract['ranks'].values():
                    used={s for v in r['nodes'].values() for f in ['reads','ready_outputs'] for s in v[f]};r['initial_objects']={s:v for s,v in r['initial_objects'].items() if s in used}
                finalize_tensor_costs(plan,c)
            self.assertEqual(sum(o.flops for o in old.operations),sum(o.flops for o in new.operations))
            self.assertEqual(sum(o.duration_ns for o in old.operations if o.phase=='rollout'),sum(o.duration_ns for o in new.operations if o.phase=='rollout'))
            # The legacy input fixture copies only its first prompt. The new
            # batch explicitly stores every request's prompt (no prefix reuse).
            extra=(len(c.workload.trajectories)-1)*len(c.workload.trajectories[0].prompt_token_ids)*4
            self.assertEqual({r:v+extra for r,v in old.preflight_peaks.items()},new.preflight_peaks)
            self.assertEqual(len(new.tensors),len(self.plan(single).tensors))
            self.assertTrue(all(t['request_count']==3 and t['nbytes']==3*t['request_stride_bytes'] for t in new.tensors if not t['persistent']))
            self.assertTrue(all(o.metadata.get('attention_keys_shared_between_requests') is False for o in new.operations))
    def test_host_prompt_duration_uses_full_batch_bytes(self):
        from rl_xy.mapping import ceildiv
        c=self.setup_case(1);single=replace(c,workload=replace(c.workload,trajectories=c.workload.trajectories[:1]))
        new=replicate_uniform_generation(self.plan(single),single,c)
        op=next(o for o in new.operations if o.name=='host_to_rank0_prompt')
        self.assertEqual(op.tensor_bytes,3*len(single.workload.trajectories[0].prompt_token_ids)*4)
        self.assertEqual(op.duration_ns,c.hardware.host_latency_ns+ceildiv(op.tensor_bytes,c.hardware.host_bytes_per_ns))
    def test_reject_mixed_length_or_version(self):
        c=self.setup_case(1);single=replace(c,workload=replace(c.workload,trajectories=c.workload.trajectories[:1]));p=self.plan(single)
        bad=replace(c,workload=replace(c.workload,trajectories=(c.workload.trajectories[0],replace(c.workload.trajectories[1],response_token_ids=(1,)))))
        with self.assertRaises(ValueError):replicate_uniform_generation(p,single,bad)
        with self.assertRaises(ValueError):replicate_uniform_generation(p,single,replace(c,workload=replace(c.workload,behavior_version=99)))
if __name__=='__main__':unittest.main()
