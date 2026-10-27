"""Generic synchronous execution checks, separate from the A/B golden fixture.

The existing strict event checker validates actual Astra issue/callback pairs.
Full memory evidence is checked by resource_validation, not a fabricated curve.
"""
from collections import Counter

from .execution_contract import prompt_for
from .types import CheckResult
from .training_schedule import training_visits, enabled
from .validation import (_parameter_checks, _trajectory_checks, _phase_and_version_checks,
                         _training_checks, _communication_checks, _event_checks)


def shape_checks(case, operations):
    m, tp = case.model, case.strategy.tensor_parallel
    errors=[]
    names=Counter((o.round_id,o.name) for o in operations)
    indexed={}
    for o in operations:indexed.setdefault((o.round_id,o.name),[]).append(o)
    qkv=(m.query_heads+2*m.kv_heads)*m.head_dim//tp
    embeddings=[o for o in operations if o.name.endswith('.embedding')]
    for forward in embeddings:
        base=forward.name.removesuffix('.embedding')
        t=forward.metadata['tokens']
        context=forward.metadata['context_tokens']
        if m.rotary_embedding and names[forward.round_id,base+'.rope_table']!=1:
            errors.append(base+'.rope_table')
        for layer in range(m.layers):
            prefix=f'{base}.layer{layer}'
            q=indexed.get((forward.round_id,prefix+'.qkv'),[])
            expected=(t,qkv,m.hidden_size)
            if len(q)!=1 or not q[0].kernel_spec or len(q[0].kernel_spec.matmuls)!=1:
                errors.append(prefix+'.qkv')
            else:
                p=q[0].kernel_spec.matmuls[0]
                if (p.m,p.n,p.k)!=expected:errors.append([prefix,'qkv shape',expected])
            for suffix,enabled in [('qkv_bias_add',m.qkv_bias),('rope_apply',m.rotary_embedding),
                                   ('gqa_expand_kv',m.query_heads!=m.kv_heads),('attention_scale_causal_mask',True)]:
                if names[forward.round_id,prefix+'.'+suffix]!=int(enabled):errors.append(prefix+'.'+suffix)
            scores=indexed.get((forward.round_id,prefix+'.attention_scores'),[])
            if len(scores)!=1 or not scores[0].kernel_spec:
                errors.append(prefix+'.attention_scores')
            else:
                p=scores[0].kernel_spec.matmuls[0]
                if (p.m,p.n,p.k,p.batches)!=(t,context,m.head_dim,m.query_heads//tp):
                    errors.append([prefix,'attention heads/positions'])
    for rid, _, _, _, trajectory, phase in training_visits(case):
            t=len(prompt_for(case,trajectory))+len(trajectory.response_token_ids)-1
            prefix=f'{phase}_backward'
            required=[prefix+'.tied_embedding_gradient_merge'] if m.tie_word_embeddings else []
            for layer in range(m.layers):
                for suffix,enabled in [('qkv_bias_backward',m.qkv_bias),('rope_backward',m.rotary_embedding),
                                       ('gqa_reduce_kv_gradients',m.query_heads!=m.kv_heads)]:
                    if enabled:required.append(f'{prefix}.layer{layer}.{suffix}')
                q=indexed.get((rid,f'{prefix}.layer{layer}.qkv_backward'),[])
                expected=[(t,m.hidden_size,qkv),(m.hidden_size,qkv,t)]
                actual=[(p.m,p.n,p.k) for p in q[0].kernel_spec.matmuls] if len(q)==1 and q[0].kernel_spec else []
                if actual!=expected:errors.append([prefix,'backward shapes',actual,expected])
            for name in required:
                if names[rid,name]!=1:errors.append(name)
    return [CheckResult('execution.qwen_gqa_rope_bias_tied_shapes',not errors,
        'compact QKV, query-head attention, repeated-KV gradient reduction, RoPE/bias and tied merge coverage',
        errors,'model dimensions independently reconstructed from case; not numerical/model-quality validation','shape_contract')]


def validate_execution(case, operations, ledger, events):
    checks=[]
    from .training_validation import multi_update_checks
    for name,fn in [('parameters',lambda:_parameter_checks(case,ledger)),
        ('trajectories',lambda:_trajectory_checks(case,operations)),
        ('versions',lambda:multi_update_checks(case,operations) if enabled(case) else _phase_and_version_checks(case,operations)),
        ('training',lambda:[] if enabled(case) else _training_checks(case,operations)),
        ('communication',lambda:_communication_checks(case,operations)),
        ('events',lambda:_event_checks(case,operations,events)),
        ('shapes',lambda:shape_checks(case,operations))]:
        try:checks.extend(fn())
        except Exception as exc:
            checks.append(CheckResult('execution.internal_error.'+name,False,'complete check',
                                     str(exc),'fail-closed validator','run'))
    return checks
