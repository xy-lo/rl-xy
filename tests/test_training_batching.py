"""Packed training algebra, parameter provenance and actual tensor lifetimes."""
import copy
from dataclasses import replace
from pathlib import Path
import unittest

from rl_xy.colocation import StageBuilder
from rl_xy.communication_lowering import lower_communication_endpoints
from rl_xy.config import load_case
from rl_xy.cost_model import AnalyticalCostProvider
from rl_xy.model import parameter_totals
from rl_xy.operator_ir import Matmul
from rl_xy.tensor_plan import build_tensor_plan, preflight_contract, finalize_tensor_costs
from rl_xy.training_batching import (fuse_training, training_batching_eligibility,
    pack_training_matmuls, validate_training_batching, TrainingBatchingUnsupported,
    prepare_training_gradient_io, validate_training_gradient_io)

ROOT = Path(__file__).resolve().parents[1]


def training_plan(ragged=False, count=4, untied=False):
    case = load_case(ROOT/'configs/qwen2_group_fixture/case.json')
    trajectories = tuple(replace(t, prompt_token_ids=(1, 2), response_token_ids=(3, 4))
                         for t in case.workload.trajectories)
    if ragged:
        trajectories = trajectories[:2]+(
            replace(trajectories[2], prompt_token_ids=(1, 2, 3), response_token_ids=(4, 5)),
            replace(trajectories[3], prompt_token_ids=(1, 2, 3, 4), response_token_ids=(5, 6, 7)))
    case = replace(case, workload=replace(case.workload, trajectories=trajectories[:count]))
    if untied: case = replace(case, model=replace(case.model, tie_word_embeddings=False, qkv_bias=False, rotary_embedding=False))
    builder = StageBuilder(case, {'train', 'clip', 'optimizer', 'publish', 'cleanup'});builder.build_round(0)
    return case, build_tensor_plan(case, lower_communication_endpoints(case, builder.operations), '/not-executed')


def mm(a, b):
    return [[sum(x*y for x,y in zip(row,column)) for column in zip(*b)] for row in a]


def transpose(a): return [list(x) for x in zip(*a)]
def plus(a, b): return [[x+y for x,y in zip(ra,rb)] for ra,rb in zip(a,b)]


class TrainingBatchingTests(unittest.TestCase):
    def test_mb1_is_the_original_plan_without_relabeling_costs_or_boundaries(self):
        for ragged in (False, True):
            case,plan = training_plan(ragged);before = copy.deepcopy(plan)
            self.assertTrue(training_batching_eligibility(plan,case,1)['runtime_ready'])
            self.assertIs(fuse_training(plan,case,1),plan)
            self.assertEqual(plan,before)

    def test_linear_concatenation_has_real_forward_dinput_and_summed_dweight_algebra(self):
        x1=[[1,2,3],[4,5,6]];x2=[[7,8,9]];w=[[2,1],[0,3],[1,4]]
        dy1=[[1,3],[2,4]];dy2=[[5,6]]
        forward=pack_training_matmuls([Matmul(2,2,3),Matmul(1,2,3)],linear=True)[0]
        self.assertEqual((forward.m,forward.n,forward.k),(3,2,3))
        self.assertEqual(mm(x1+x2,w),mm(x1,w)+mm(x2,w))
        din=pack_training_matmuls([Matmul(2,3,2,direction='dinput'),Matmul(1,3,2,direction='dinput')])[0]
        self.assertEqual((din.m,din.n,din.k),(3,3,2))
        self.assertEqual(mm(dy1+dy2,transpose(w)),mm(dy1,transpose(w))+mm(dy2,transpose(w)))
        dw=pack_training_matmuls([Matmul(3,2,2,output_bytes=4,direction='dweight'),Matmul(3,2,1,output_bytes=4,direction='dweight')])[0]
        self.assertEqual((dw.m,dw.n,dw.k,dw.output_bytes),(3,2,3,4))
        self.assertEqual(mm(transpose(x1+x2),dy1+dy2),plus(mm(transpose(x1),dy1),mm(transpose(x2),dy2)))
        # A K-packed call emits ONE 3x2 FP32 output, not two 3x2 outputs.
        self.assertEqual(dw.m*dw.n*dw.output_bytes,24)

    def test_same_shape_mb2_and_mb4_have_one_rmw_and_parameter_sum_per_microbatch(self):
        for size in (2,4):
            case,plan=training_plan();before=copy.deepcopy(plan);batched=fuse_training(plan,case,size)
            self.assertEqual(plan,before)
            self.assertTrue(all(c.passed for c in validate_training_batching(batched,case)))
            self.assertEqual(batched.preflight_peaks,preflight_contract(batched.contract))
            updates=[o for o in batched.operations if o.metadata.get('training_batch_rmw')]
            self.assertEqual(len(updates),4//size)
            for op in updates:
                self.assertEqual(len(op.metadata['training_microbatch_contributors']),size)
                self.assertEqual(op.metadata['global_loss_denominator'],8)
            gradients=[t for t in batched.tensors if t['kind']=='parameter_gradient_aggregate' and t['rank']==0]
            self.assertEqual(sum(t['nbytes'] for t in gradients),4//size*parameter_totals(case)['local_parameter_elements_per_die']*4)
            self.assertLess(len(batched.operations),len(plan.operations))
            for op in batched.operations:
                if op.kind=='compute' and op.kernel_spec:
                    estimate=op.metadata['cost_estimate']
                    self.assertNotIn('temporary_spec',estimate['source'])
                    self.assertEqual(op.duration_ns,max(1,estimate['compute_ns']+estimate['local_memory_ns']))

    def test_dense_dweight_replaces_old_independent_storage_not_phantom_views(self):
        case,plan=training_plan();batched=fuse_training(plan,case,2)
        heads=[o for o in batched.operations if o.name.endswith('.lm_head_backward_dinput_dweight')]
        self.assertEqual(len(heads),2)
        first=heads[0];dw=next(p for p in first.kernel_spec.matmuls if p.direction=='dweight')
        self.assertEqual(dw.k,6) # two teacher sequences of 3 tokens
        tensors={t['storage_id']:t for t in batched.tensors}
        allocated=batched.contract['ranks']['0']['nodes'][str(first.node_id)]['allocations']
        gradients=[tensors[s] for s in allocated if tensors[s]['kind']=='parameter_gradient_component']
        self.assertEqual(len(gradients),1)
        self.assertEqual(gradients[0]['nbytes'],dw.m*dw.n*4)
        self.assertEqual(gradients[0]['contributor_ids'],['A','B'])
        old_head_gradients={t['storage_id'] for t in plan.tensors if t['kind']=='parameter_gradient' and '.lm_head_backward_' in t['storage_id']}
        self.assertFalse(old_head_gradients & set(tensors))

    def test_ragged_attention_keeps_independent_request_blocks(self):
        case,plan=training_plan(ragged=True);batched=fuse_training(plan,case,4)
        score=next(o for o in batched.operations if o.name.endswith('.layer0.attention_scores'))
        self.assertEqual([(m.m,m.n,m.batches) for m in score.kernel_spec.matmuls],[(3,3,4),(4,4,2),(6,6,2)])
        backward=next(o for o in batched.operations if o.name.endswith('.layer0.attention_backward'))
        self.assertEqual(len(backward.kernel_spec.matmuls),12) # four products x three independent shapes
        self.assertTrue(all(m.m in (3,4,6) for m in backward.kernel_spec.matmuls))
        audit=batched.contract['training_batching']
        self.assertEqual(audit['original_gemm_flops'],audit['packed_gemm_flops'])
        losses=[o for o in batched.operations if o.name.endswith('_loss')]
        self.assertEqual(len(losses),1)
        self.assertNotIn('advantage',losses[0].metadata)
        self.assertEqual([x['trajectory_id'] for x in losses[0].metadata['training_batch_terms']],['A','B','C','D'])
        self.assertEqual({x['global_denominator'] for x in losses[0].metadata['training_batch_terms']},{9})

    def test_non_gemm_gradients_have_charged_partial_reduce_and_tied_merge(self):
        case,plan=training_plan();batched=fuse_training(plan,case,4)
        reduces=[o for o in batched.operations if o.metadata.get('training_batch_partial_reduce')]
        self.assertTrue(reduces)
        for op in reduces:
            n=batched.contract['ranks']['0']['nodes'][str(op.node_id)]
            size=sum(n['allocations'].values())
            self.assertEqual(op.kernel_spec.read_bytes,4*size)
            self.assertEqual(op.kernel_spec.write_bytes,size)
            self.assertEqual(op.kernel_spec.vector_ops,3*size//4)
            self.assertEqual(set(n['reads']),set(n['frees']))
            self.assertEqual(len(n['reads']),4)
        embedding=next(o for o in batched.operations if o.name.endswith('.embedding_backward'))
        size=case.model.vocab_size*case.model.hidden_size//case.strategy.tensor_parallel*4
        self.assertEqual(embedding.metadata['explicit_partial_gradient_write_bytes'],4*size)
        bias=next(o for o in batched.operations if o.name.endswith('.layer0.qkv_bias_backward'))
        self.assertEqual(bias.metadata['explicit_partial_gradient_write_bytes'],0)
        tie=next(o for o in batched.operations if o.name.endswith('.tied_embedding_gradient_merge'))
        self.assertTrue(tie.metadata['transpose_head_before_embedding_sum'])
        self.assertEqual((tie.kernel_spec.vector_ops,tie.kernel_spec.read_bytes,tie.kernel_spec.write_bytes),(size//4,2*size,size))
        # Head gradient is HxV; embedding is VxH. The merge must transpose.
        head=[[1,2,3],[4,5,6]];embedding_grad=[[10,20],[30,40],[50,60]]
        self.assertEqual(plus(transpose(head),embedding_grad),[[11,24],[32,45],[53,66]])

    def test_untied_no_bias_variant_and_short_tail_keep_exact_coverage(self):
        for untied,count in [(True,4),(False,3)]:
            case,plan=training_plan(ragged=True,count=count,untied=untied)
            batched=fuse_training(plan,case,2)
            self.assertTrue(all(c.passed for c in validate_training_batching(batched,case)))
            groups=batched.contract['training_batching']['groups']
            self.assertEqual([len(g['contributor_ids']) for g in groups],[2,count-2])
            if untied: self.assertFalse(any(o.name.endswith('.tied_embedding_gradient_merge') for o in batched.operations))

    def test_tp1_tp4_and_two_round_boundaries_remain_executable_contracts(self):
        for tp,rounds in ((1,1),(4,1),(2,2)):
            with self.subTest(tp=tp,rounds=rounds):
                case,_=training_plan(ragged=True)
                case=replace(case,rounds=rounds,hardware=replace(case.hardware,dies=tp),
                    strategy=replace(case.strategy,tensor_parallel=tp),
                    model=replace(case.model,kv_heads=4 if tp==4 else case.model.kv_heads))
                b=StageBuilder(case,{'train','clip','optimizer','publish','cleanup'})
                for offset in range(rounds):b.build_round(offset)
                plan=build_tensor_plan(case,lower_communication_endpoints(case,b.operations),'/not-executed')
                batched=fuse_training(plan,case,2)
                self.assertEqual(len(batched.contract['training_batching']['groups']),2*rounds)
                self.assertTrue(all(c.passed for c in validate_training_batching(batched,case)))
                self.assertTrue(preflight_contract(batched.contract))

    def test_full_workflow_nontraining_objects_survive_and_provider_is_used(self):
        case,_=training_plan(ragged=True)
        b=StageBuilder(case,{'input','rollout','reference','old_policy','reward','advantage','train','clip','optimizer','publish','cleanup'})
        b.build_round(0)
        plan=build_tensor_plan(case,lower_communication_endpoints(case,b.operations),'/not-executed')
        class RecordingProvider(AnalyticalCostProvider):
            def __init__(self):self.seen=[]
            def estimate(self,spec,hardware):
                self.seen.append(spec)
                return super().estimate(spec,hardware)
        provider=RecordingProvider();batched=fuse_training(plan,case,2,cost_provider=provider)
        self.assertTrue(provider.seen)
        self.assertTrue(any(any(p.direction=='dweight' and p.k==6 for p in s.matmuls) for s in provider.seen))
        self.assertEqual([o.name for o in batched.operations if not o.phase.startswith('train_')],
                         [o.name for o in plan.operations if not o.phase.startswith('train_')])
        self.assertTrue(all(c.passed for c in validate_training_batching(batched,case)))
        self.assertTrue(preflight_contract(batched.contract))

    def test_microbatch_peak_is_real_and_can_fail_capacity(self):
        case,plan=training_plan();base=preflight_contract(plan.contract)
        batched=fuse_training(plan,case,4)
        self.assertGreater(batched.preflight_peaks['d0.dram'],base['d0.dram'])
        for rank in plan.contract['ranks'].values():rank['dram_capacity']=base['d0.dram']
        with self.assertRaisesRegex(ValueError,'DRAM admission overflow'):
            fuse_training(plan,case,4)

    def test_only_proven_independent_training_temporary_reads_are_removed(self):
        case,plan=training_plan()
        a_loss=next(o for o in plan.operations if o.name=='train_A_loss')
        sid=next(iter(plan.contract['ranks']['0']['nodes'][str(a_loss.node_id)]['allocations']))
        b_forward=next(o for o in plan.operations if o.name=='train_B_forward.embedding')
        plan.contract['ranks']['0']['nodes'][str(b_forward.node_id)]['reads'].append(sid)
        self.assertTrue(preflight_contract(plan.contract)) # valid but unnecessary old serial dependency
        batched=fuse_training(plan,case,2)
        audit=batched.contract['training_batching']
        self.assertEqual(audit['removed_conservative_read_count'],1)
        self.assertEqual(audit['removed_conservative_read_examples'][0]['storage_id'],sid)
        self.assertTrue(preflight_contract(batched.contract))

    def test_unknown_cross_trajectory_dependency_and_template_mismatch_fail_closed(self):
        case,plan=training_plan()
        altered=[]
        for op in plan.operations:
            if op.name.endswith('_loss'):op=replace(op,name=op.name+'_custom')
            altered.append(op)
        plan.operations=altered
        a_loss=next(o for o in plan.operations if o.name=='train_A_loss_custom')
        b_loss=next(o for o in plan.operations if o.name=='train_B_loss_custom')
        sid=next(iter(plan.contract['ranks']['0']['nodes'][str(a_loss.node_id)]['allocations']))
        plan.contract['ranks']['0']['nodes'][str(b_loss.node_id)]['reads'].append(sid)
        with self.assertRaisesRegex(ValueError,'unproven cross-trajectory'):
            fuse_training(plan,case,2)
        case,plan=training_plan();op=next(o for o in plan.operations if o.name=='train_B_forward.embedding')
        plan.operations=[replace(o,name=o.name+'_unknown') if o.node_id==op.node_id else o for o in plan.operations]
        with self.assertRaisesRegex(ValueError,'templates differ'):fuse_training(plan,case,2)

    def test_independent_coverage_rejects_wrong_contributors_bytes_and_duplicate_rmw(self):
        for corruption in ('contributor','bytes','denominator','duplicate_rmw','loss_advantage'):
            case,plan=training_plan();batched=fuse_training(plan,case,2)
            if corruption in ('contributor','bytes'):
                tensor=next(t for t in batched.tensors if t['kind']=='parameter_gradient_aggregate')
                if corruption=='contributor':tensor['contributor_ids']=['A','A']
                else:tensor['nbytes']+=4
            elif corruption=='loss_advantage':
                op=next(o for o in batched.operations if o.metadata.get('training_batch_loss_kind')=='loss')
                op.metadata['training_batch_terms'][0]['advantage']+=1
            else:
                op=next(o for o in batched.operations if o.metadata.get('training_batch_rmw'))
                if corruption=='duplicate_rmw':batched.operations.append(op)
                else:batched.operations=[replace(o,metadata={**o.metadata,'global_loss_denominator':1}) if o is op else o for o in batched.operations]
            self.assertFalse(all(c.passed for c in validate_training_batching(batched,case)),corruption)

    def test_global_denominator_from_other_dp_replicas_is_not_replaced_by_local_tokens(self):
        case,plan=training_plan()
        plan.operations=[replace(o,metadata={**o.metadata,**{k:64 for k in
            ('global_denominator','global_loss_denominator') if k in o.metadata}}) for o in plan.operations]
        batched=fuse_training(plan,case,4)
        self.assertEqual(batched.contract['training_batching']['source_loss_denominators'],[64])
        self.assertTrue(all(c.passed for c in validate_training_batching(batched,case)))
        self.assertTrue(all(o.metadata['global_loss_denominator']==64 for o in batched.operations if o.metadata.get('training_batch_rmw')))

    def test_invalid_source_lifetime_cannot_be_hidden_by_gradient_coalescing(self):
        case,plan=training_plan()
        rmw=next(o for o in plan.operations if o.name.endswith('.gradient_accumulate_rmw'))
        node=plan.contract['ranks']['0']['nodes'][str(rmw.node_id)]
        node['frees'].append(node['frees'][0])
        with self.assertRaisesRegex(ValueError,'double free'):fuse_training(plan,case,2)

    def test_explicit_gradient_io_is_opt_in_idempotent_and_shared_by_mb1_2_4(self):
        for size in (1,2,4):
            case,plan=training_plan(ragged=True)
            self.assertFalse(all(c.passed for c in validate_training_gradient_io(plan,case)))
            self.assertIs(prepare_training_gradient_io(plan,case),plan)
            before=copy.deepcopy(plan)
            self.assertIs(prepare_training_gradient_io(plan,case),plan)
            self.assertEqual(plan,before)
            if size==1:self.assertIs(fuse_training(plan,case,size),plan)
            batched=fuse_training(plan,case,size)
            finalize_tensor_costs(batched,case)
            self.assertTrue(all(c.passed for c in validate_training_gradient_io(batched,case)))
            for op in batched.operations:
                marker=op.metadata.get('gradient_output_io')
                if marker:
                    self.assertEqual(op.kernel_spec.write_bytes,marker['base_kernel_write_bytes']+marker['additional_write_bytes_per_rank'])
                    self.assertGreaterEqual(op.metadata['cost_estimate']['dram_write_bytes'],op.kernel_spec.write_bytes)
                    if op.name.endswith('.qkv_bias_backward'):
                        self.assertEqual(marker['additional_write_bytes_per_rank'],0)

    def test_explicit_gradient_io_cannot_be_tampered_or_claimed_before_costing(self):
        case,plan=training_plan();prepare_training_gradient_io(plan,case)
        self.assertFalse(all(c.passed for c in validate_training_gradient_io(plan,case)))
        finalize_tensor_costs(plan,case)
        self.assertTrue(all(c.passed for c in validate_training_gradient_io(plan,case)))
        norm=next(o for o in plan.operations if o.name.endswith('.final_rmsnorm_backward'))
        norm.metadata['gradient_output_io']['additional_write_bytes_per_rank']=0
        self.assertFalse(all(c.passed for c in validate_training_gradient_io(plan,case)))
        with self.assertRaisesRegex(ValueError,'marker/spec was changed'):prepare_training_gradient_io(plan,case)

    def test_invalid_size_and_unsupported_update_contract_do_not_mutate(self):
        case,plan=training_plan();before=copy.deepcopy(plan)
        for value in (0,-1,True,1.5,'2',3):
            with self.assertRaisesRegex(ValueError,'supported positive integer'):fuse_training(plan,case,value)
        other=replace(case,strategy=replace(case.strategy,training_epochs=2))
        self.assertFalse(training_batching_eligibility(plan,other,2)['runtime_ready'])
        with self.assertRaises(TrainingBatchingUnsupported):fuse_training(plan,other,2)
        self.assertEqual(plan,before)


if __name__=='__main__':unittest.main()
