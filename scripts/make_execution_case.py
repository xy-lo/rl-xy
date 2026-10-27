#!/usr/bin/env python3
"""Compile explicit model shapes and complete selected length groups to TP2.

Inputs remain lengths: no fabricated token IDs/reward/logprob. Selecting groups
is a structural probe, never a representative full-training performance result.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

from rl_xy.config import load_case
from rl_xy.model import parameter_totals
from rl_xy.model_import import inspect_hf_model
from rl_xy.verl_trace import load_verl_trace


def compile_case(template, model_path, trace_path, origin, step, group_ids, output, probe_response_cap=None):
    if not group_ids or len(set(group_ids))!=len(group_ids):
        raise ValueError('explicit unique complete group IDs required')
    model=inspect_hf_model(model_path)
    cfg=model['config']
    if cfg.get('rope_scaling') is not None or cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])%2:
        raise ValueError('only default unscaled even-dimensional RoPE supported')
    trace=load_verl_trace(trace_path,origin=origin)
    if probe_response_cap is not None and (type(probe_response_cap) is not int or probe_response_cap<=0):
        raise ValueError('counterfactual probe cap must be positive')
    found=[s for s in trace['steps'] if s['step']==step]
    if len(found)!=1:raise ValueError('step not available')
    selected=[g for g in found[0]['groups'] if g['group_id'] in group_ids]
    if {g['group_id'] for g in selected}!=set(group_ids):raise ValueError('requested group missing')
    if any(t['prompt_len']+t['resp_len']>cfg['max_position_embeddings'] for g in selected for t in g['trajectories']):
        raise ValueError('model context exceeded')
    raw=json.loads((template/'case.json').read_text())
    raw.update(schema_version='1.1',case_id='qwen2_selected_length_groups',rounds=1)
    mapping={'layers':'num_hidden_layers','hidden_size':'hidden_size','query_heads':'num_attention_heads',
             'kv_heads':'num_key_value_heads','ffn_size':'intermediate_size','vocab_size':'vocab_size'}
    for dst,src in mapping.items():raw['model'][dst]=cfg[src]
    raw['model'].update(head_dim=model['head_dim'],qkv_bias=True,rotary_embedding=True,
                        tie_word_embeddings=cfg['tie_word_embeddings'])
    w=raw['workload']
    w.update(round_id=step,behavior_version=found[0]['behavior_version'],prompt_token_ids=[],
             group_id='selected-complete-groups',max_response_length=min(trace['run_meta']['algo']['max_response_tokens'],probe_response_cap or trace['run_meta']['algo']['max_response_tokens']))
    w['trajectories']=[{'trajectory_id':f'T{i}', 'prompt_length':t['prompt_len'], 'response_length':min(t['resp_len'],probe_response_cap or t['resp_len']),
        'group_id':g['group_id'],'finish_reason':t['finish_reason'],'reward':None}
        for i,(g,t) in enumerate((g,t) for g in selected for t in g['trajectories'])]
    raw['strategy']['rollout_max_concurrency']=len(w['trajectories'])
    output.mkdir(parents=True,exist_ok=False)
    for name in ('astra_system.json','astra_network.yml','astra_remote_memory.json'):
        shutil.copyfile(template/name,output/name)
    (output/'case.json').write_text(json.dumps(raw,indent=2)+'\n')
    case=load_case(output/'case.json')
    if parameter_totals(case)['unique_parameter_elements']!=model['unique_parameter_elements']:
        raise ValueError('imported shape accounting disagrees with HF metadata')
    shutil.copyfile(model_path,output/'model_config.json')
    report={'status':'PASS','origin_declared':origin,'step':step,'selected_groups':group_ids,
        'full_step_replayed':len(selected)==len(found[0]['groups']) and probe_response_cap is None,
        'counterfactual_response_cap':probe_response_cap,
        'response_lengths_modified':any(t['resp_len']>w['max_response_length'] for g in selected for t in g['trajectories']),
        'original_response_lengths':[t['resp_len'] for g in selected for t in g['trajectories']],
        'executed_response_lengths':[t['response_length'] for t in w['trajectories']],
        'selection_scope':'explicit complete-group structural probe; not an unbiased full-step throughput sample',
        'input_values':{'token_ids':False,'reward':False,'logprob':False},
        'model_config_sha256':model['config_sha256'],'length_trace_sha256':trace['source_sha256'],
        'template_sha256':hashlib.sha256((template/'case.json').read_bytes()).hexdigest(),
        'model_identity_verified_against_historical_weights':False,
        'hardware_scope':'caller-supplied TP2 logical template; run-wafer may expand an explicit DP/multicore scenario; no implicit calibrated H',
        'trajectories':len(w['trajectories']), 'groups':len(selected),
        'original_to_execution_ids':[{'original':t['traj_id'],'execution':f'T{i}'} for i,(g,t) in enumerate((g,t) for g in selected for t in g['trajectories'])],
        'parameter_totals':parameter_totals(case),'gpu_observed_stage_times_s':found[0]['gpu_observed_stage_times_s'],
        'gpu_times_used_as_chip_durations':False}
    (output/'compilation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--model-config',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--origin',required=True,choices=('synthetic_fixture','statistics_constrained_synthetic','framework_capture'))
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--groups',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--probe-response-cap',type=int,help='EXPLICIT counterfactual short probe; changes lengths and forbids original-performance interpretation')
    a=p.parse_args()
    print(json.dumps(compile_case(a.template,a.model_config,a.trace,a.origin,a.step,a.groups,a.out,a.probe_response_cap),indent=2))


if __name__=='__main__':main()
