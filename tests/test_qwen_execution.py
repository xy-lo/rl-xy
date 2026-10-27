from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import importlib.util

from rl_xy.config import load_case, ConfigError
from rl_xy.execution_contract import demean_advantages
from rl_xy.execution_validation import shape_checks
from rl_xy.model import parameter_totals, enumerate_parameters, kv_bytes_per_processed_token_per_die
from rl_xy.workflow import build_workflow
from rl_xy.communication_lowering import lower_communication_endpoints
from rl_xy.tensor_plan import build_tensor_plan, finalize_tensor_costs

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'configs/qwen2_group_fixture/case.json'


def read(raw):
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'case.json'
        p.write_text(json.dumps(raw))
        return load_case(p)


class QwenExecutionTest(unittest.TestCase):
    def compiler_inputs(self, directory):
        # Reuse the independent L1 fixture; no historic files or GPU required.
        from test_migration_inputs import capture
        model=directory/'model.json'
        model.write_text(json.dumps({'model_type':'qwen2','architectures':['Qwen2ForCausalLM'],
            'hidden_size':8,'num_attention_heads':4,'num_key_value_heads':2,'num_hidden_layers':2,
            'intermediate_size':16,'vocab_size':16,'max_position_embeddings':32768,
            'tie_word_embeddings':True,'hidden_act':'silu','attention_dropout':0.,'use_sliding_window':False}))
        trace=directory/'capture.jsonl'
        trace.write_text(''.join(json.dumps(r)+'\n' for r in capture()))
        spec=importlib.util.spec_from_file_location('make_case_under_test',ROOT/'scripts/make_execution_case.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod.compile_case,model,trace

    def test_compiler_preserves_lengths_without_values_or_gpu_costs(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);compile_case,model,trace=self.compiler_inputs(d)
            r=compile_case(CASE.parent,model,trace,'synthetic_fixture',1,['g0','g1'],d/'compiled')
            c=load_case(d/'compiled/case.json')
            self.assertEqual([len(t.response_token_ids) for t in c.workload.trajectories],[1,2,3]*2)
            self.assertEqual([t.reward for t in c.workload.trajectories],[None]*6)
            self.assertEqual(r['input_values'],dict.fromkeys(['token_ids','reward','logprob'],False))
            self.assertTrue(r['full_step_replayed'])
            self.assertFalse(r['gpu_times_used_as_chip_durations'])

    def test_compiler_cap_is_explicit_counterfactual(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);compile_case,model,trace=self.compiler_inputs(d)
            r=compile_case(CASE.parent,model,trace,'synthetic_fixture',1,['g0'],d/'compiled',1)
            self.assertEqual(r['original_response_lengths'],[1,2,3])
            self.assertEqual(r['executed_response_lengths'],[1,1,1])
            self.assertTrue(r['response_lengths_modified'])
            self.assertFalse(r['full_step_replayed'])

    def test_compiler_rejects_missing_duplicate_groups_and_scaled_rope(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);compile_case,model,trace=self.compiler_inputs(d)
            for groups in ([],['absent'],['g0','g0']):
                with self.subTest(groups=groups),self.assertRaises(ValueError):
                    compile_case(CASE.parent,model,trace,'synthetic_fixture',1,groups,d/'compiled')
            raw=json.loads(model.read_text());raw['rope_scaling']={'type':'linear','factor':2}
            model.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):compile_case(CASE.parent,model,trace,'synthetic_fixture',1,['g0'],d/'compiled')

    def test_compiler_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);compile_case,model,trace=self.compiler_inputs(d)
            target=d/'compiled'
            compile_case(CASE.parent,model,trace,'synthetic_fixture',1,['g0'],target)
            before=(target/'case.json').read_bytes()
            with self.assertRaises(FileExistsError):compile_case(CASE.parent,model,trace,'synthetic_fixture',1,['g0'],target)
            self.assertEqual((target/'case.json').read_bytes(),before)

    def test_parameter_and_kv_hand_golden(self):
        c=load_case(CASE)
        self.assertEqual(parameter_totals(c)['unique_parameter_elements'],1352)
        self.assertEqual(parameter_totals(c)['local_parameter_elements_per_die'],696)
        self.assertEqual(kv_bytes_per_processed_token_per_die(c),16)
        rows=enumerate_parameters(c)
        self.assertNotIn('lm_head',[r.parameter_id for r in rows])
        self.assertEqual(sum(r.unique_elements for r in rows if 'bias' in r.parameter_id),32)

    def test_group_demean_is_not_global_demean(self):
        c=load_case(CASE)
        self.assertEqual(demean_advantages(c),{'A':.5,'B':-.5,'C':1.,'D':-1.})
        ops=build_workflow(c)
        self.assertEqual(next(o for o in ops if o.name=='group_advantage_demean').metadata['advantages'],[.5,-.5,1.,-1.])
        self.assertEqual(next(o for o in ops if o.name=='train_C_forward.embedding').metadata['tokens'],4)
        self.assertEqual(next(o for o in ops if o.name=='rollout_C_decode1.embedding').metadata['context_tokens'],4)

    def test_full_tape_kv_and_tied_storage(self):
        c=load_case(CASE)
        p=build_tensor_plan(c,lower_communication_endpoints(c,build_workflow(c)),'/tmp/not-run')
        finalize_tensor_costs(p,c)
        self.assertTrue(all(x.passed for x in shape_checks(c,p.operations)))
        kv=[t for t in p.tensors if t['rank']==0 and t['kind']=='kv']
        # processed positions A4+B2+C4+D5, 2layers * 1KVhead * 2dim * K/V * BF16.
        self.assertEqual(sum(t['nbytes'] for t in kv),15*16)
        self.assertTrue(all('.rope_apply.' in t['tensor_id'] for t in kv))
        self.assertEqual(len([t for t in p.tensors if t['rank']==0 and t['kind']=='gradient' and t['persistent'] and 'token_embedding' in t['storage_id']]),1)
        self.assertFalse(any(t['persistent'] and '.lm_head' in t['storage_id'] for t in p.tensors))
        for tid in 'ABCD':
            products=sum(t['nbytes'] for t in p.tensors if t['rank']==0 and t['kind']=='parameter_gradient' and f'train_{tid}_backward' in t['tensor_id'])
            self.assertEqual(products,(696+64)*4)
        self.assertLess(p.preflight_peaks['d0.dram'],c.hardware.dram_bytes_per_die)

    def test_lengths_only_does_not_invent_values(self):
        raw=json.loads(CASE.read_text())
        for t in raw['workload']['trajectories']:
            t['prompt_length']=len(t.pop('prompt_token_ids'))
            t['response_length']=len(t.pop('response_token_ids'))
            t['reward']=None
        c=read(raw)
        self.assertEqual(demean_advantages(c),dict.fromkeys('ABCD'))
        self.assertFalse(isinstance(c.workload.trajectories[0].response_token_ids,tuple))
        o=next(o for o in build_workflow(c) if o.name=='group_advantage_demean')
        self.assertEqual(o.metadata['advantages'],[None]*4)

    def test_invalid_contracts_fail_closed(self):
        mutations=[lambda r:r['workload']['trajectories'].pop(),
            lambda r:r['workload']['trajectories'][1].update(prompt_token_ids=[1]),
            lambda r:r['workload']['trajectories'][1].update(reward=None),
            lambda r:r['model'].update(kv_heads=3),
            lambda r:r['model'].update(qkv_bias=1),
            lambda r:r['model'].update(head_dim=3,hidden_size=12),
            lambda r:r['workload']['trajectories'][0].update(trajectory_id='A_B'),
            lambda r:r['workload']['trajectories'][0].update(response_length=2)]
        for mutate in mutations:
            raw=json.loads(CASE.read_text());mutate(raw)
            with self.subTest(mutate=mutate), self.assertRaises((ConfigError,ValueError)):
                read(raw)

    def test_shape_validator_detects_missing_gqa_and_wrong_projection(self):
        c=load_case(CASE);ops=build_workflow(c)
        self.assertFalse(all(x.passed for x in shape_checks(c,[o for o in ops if not o.name.endswith('.gqa_reduce_kv_gradients')])))
        changed=[]
        for o in ops:
            if o.name.endswith('.qkv'):
                spec=o.kernel_spec
                spec=replace(spec,matmuls=(replace(spec.matmuls[0],n=spec.matmuls[0].n+1),))
                o=replace(o,kernel_spec=spec)
            changed.append(o)
        self.assertFalse(all(x.passed for x in shape_checks(c,changed)))

    def test_multiple_round_cleanup_and_versions(self):
        c=load_case(CASE,rounds_override=2)
        ops=build_workflow(c)
        p=build_tensor_plan(c,lower_communication_endpoints(c,ops),'/tmp/not-run')
        finalize_tensor_costs(p,c)
        self.assertEqual([o.metadata['published_version'] for o in ops if o.name=='publish_actor_version'],[1,2])

    def test_rope_backward_is_transpose_not_inverse_position(self):
        # Independent scalar adjoint and finite-difference check, no GPU.
        x=[.3,-.5,.7,.2];dy=[1.,-.2,.4,.6];angles=[.4,.07]
        def rotate(v):
            return [v[i]*math.cos(angles[i])-v[i+2]*math.sin(angles[i]) for i in range(2)]+[v[i+2]*math.cos(angles[i])+v[i]*math.sin(angles[i]) for i in range(2)]
        dx=[dy[i]*math.cos(angles[i])+dy[i+2]*math.sin(angles[i]) for i in range(2)]+[dy[i+2]*math.cos(angles[i])-dy[i]*math.sin(angles[i]) for i in range(2)]
        for i in range(4):
            plus=x[:];minus=x[:];plus[i]+=1e-6;minus[i]-=1e-6
            numeric=sum(d*(a-b) for d,a,b in zip(dy,rotate(plus),rotate(minus)))/2e-6
            self.assertAlmostEqual(numeric,dx[i],places=8)

    def test_repeat_kv_gradient_sums_repeated_heads(self):
        # qheads14/kvheads2: seven query heads use each compact KV head.
        kv=[.2,.8];upstream=[(i-3)/10 for i in range(14)]
        def objective(v):return sum(upstream[q]*v[q//7] for q in range(14))
        for k in range(2):
            a=kv[:];b=kv[:];a[k]+=1e-6;b[k]-=1e-6
            self.assertAlmostEqual((objective(a)-objective(b))/2e-6,sum(upstream[k*7:(k+1)*7]),places=8)

    def test_tied_gradient_and_bias_reduction_contributions(self):
        # One shared scalar appears in embedding and head; one bias across tokens.
        def f(w,b):return sum((w*x+b)*w for x in [.2,.7,-.3])
        w,b=.4,.1
        head=sum(w*x+b for x in [.2,.7,-.3]);embedding=sum(w*x for x in [.2,.7,-.3])
        self.assertAlmostEqual((f(w+1e-6,b)-f(w-1e-6,b))/2e-6,head+embedding,places=8)
        self.assertAlmostEqual((f(w,b+1e-6)-f(w,b-1e-6))/2e-6,3*w,places=8)


if __name__=='__main__':unittest.main()
