#!/usr/bin/env python3
"""Generate explicit model-scale controls, never pretend larger-model capture."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.model_import import inspect_hf_model
from rl_xy.config import load_case


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prompt-length',type=int);p.add_argument('--response-length',type=int);a=p.parse_args()
    if (a.prompt_length is None)!=(a.response_length is None):p.error('supply both fixed lengths or neither')
    if a.prompt_length is not None and min(a.prompt_length,a.response_length)<1:p.error('positive fixed lengths required')
    root=Path(__file__).resolve().parents[1];template=root/'experiments/2026-10-27-v0.6/inputs'
    for label,model in [('qwen05b',template/'model_config.json'),('qwen3b',root/'configs/models/qwen25_3b.json'),('qwen7b',root/'configs/models/qwen25_7b.json')]:
        out=a.out/label;out.mkdir(parents=True,exist_ok=False);raw=json.loads((template/'case.json').read_text())
        info=inspect_hf_model(model);cfg=info['config']
        for dst,src in {'layers':'num_hidden_layers','hidden_size':'hidden_size','query_heads':'num_attention_heads',
                        'kv_heads':'num_key_value_heads','ffn_size':'intermediate_size','vocab_size':'vocab_size'}.items():raw['model'][dst]=cfg[src]
        raw['model'].update(head_dim=info['head_dim'],tie_word_embeddings=cfg['tie_word_embeddings'])
        raw['case_id']='verl_sync_'+label+'_same_64_length_control'
        if a.prompt_length:
            raw['case_id']=f'verl_sync_{label}_fixed_P{a.prompt_length}_R{a.response_length}'
            raw['workload']['max_response_length']=a.response_length
            for i,t in enumerate(raw['workload']['trajectories']):
                t.update(prompt_length=a.prompt_length,response_length=a.response_length,
                    group_id=f'G{i//2}',finish_reason='fixed_budget_simulation')
        (out/'case.json').write_text(json.dumps(raw,indent=2)+'\n');load_case(out/'case.json')
        for name in ['astra_system.json','astra_network.yml','astra_remote_memory.json']:shutil.copy2(template/name,out/name)
        shutil.copy2(model,out/'model_config.json')
        provenance={'origin':'same captured 0.5B length workload across model shapes',
            'actual_framework_capture_for_this_model':label=='qwen05b' and a.prompt_length is None,'model_scale_counterfactual':label!='qwen05b',
            'source_case_sha256':hashlib.sha256((template/'case.json').read_bytes()).hexdigest(),
            'model_config_sha256':info['config_sha256'],'unique_parameters':info['unique_parameter_elements'],
            'trajectory_count':64,'prompt_groups':32,'response_length':a.response_length or 4,
            'fixed_synthetic_lengths':{'prompt':a.prompt_length,'response':a.response_length},
            'warning':'controlled shape/length performance scenario; not measured larger-model rollouts or learning quality',
            'numerical_training':False}
        (out/'workload_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Three model-scale structural controls prepared')


if __name__=='__main__':main()
