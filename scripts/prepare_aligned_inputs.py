#!/usr/bin/env python3
"""Materialize locked fixed-shape control W; never invent token IDs/rewards."""
import argparse,hashlib,json,shutil
from pathlib import Path
from dataclasses import asdict
from rl_xy.baseline_contract import BaselinePolicy
ROOT=Path(__file__).resolve().parents[1]
def prepare(out,workload='W0',train_tp=2,rollout_tp=1,mb=2,logprob_mb=2):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 prompts,answers,P,R={'W0':(48,2,512,128),'W1':(384,8,512,128),'W2':(384,8,1024,1024)}[workload]
 source=ROOT/'configs/verl_sync_standard/qwen3b';case=json.loads((source/'case.json').read_text())
 case['case_id']='aligned_qwen25_3b_'+workload;case['rounds']=1
 case['workload'].update(prompt_token_ids=[],max_response_length=R,trajectories=[dict(trajectory_id=f'T{i}',group_id=f'G{i//answers}',prompt_length=P,response_length=R,finish_reason='fixed_budget_simulation',reward=None) for i in range(prompts*answers)])
 case['strategy'].update(tensor_parallel=2,rollout_max_concurrency=prompts*answers,training_microbatch_trajectories=1)
 (out/'case.json').write_text(json.dumps(case,indent=2)+'\n')
 p=BaselinePolicy(prompt_batch_size=prompts,responses_per_prompt=answers,ppo_mini_batch_prompts=prompts,train_tp=train_tp,rollout_tp=rollout_tp,
  train_micro_batch_trajectories=mb,logprob_micro_batch_trajectories=logprob_mb,use_kl_loss=True,kl_coefficient=.001)
 (out/'policy.json').write_text(json.dumps(asdict(p),indent=2)+'\n')
 for name in ['astra_system.json','model_config.json']:shutil.copyfile(source/name,out/name)
 (out/'workload_provenance.json').write_text(json.dumps(dict(origin='fixed_length_synthetic_control',W=workload,model='Qwen/Qwen2.5-3B-Instruct',
  numeric_training=False,prompt_ids_and_rewards_known=False,learning_quality_claimed=False,
  legacy_schema_fields='demean_only/adamw_shape_only fields are legacy schema placeholders; run-aligned policy.json governs std-normalized GRPO/DO cost graph',
  prompt_batch=prompts,responses_per_prompt=answers,trajectories=prompts*answers,prompt_tokens=P,response_tokens=R,
  source_case_sha256=hashlib.sha256((source/'case.json').read_bytes()).hexdigest()),indent=2)+'\n')
 return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--workload',choices=['W0','W1','W2'],default='W0');a.add_argument('--train-tp',type=int,default=2);a.add_argument('--rollout-tp',type=int,default=1);a.add_argument('--mb',type=int,default=2);a.add_argument('--logprob-mb',type=int,default=2);x=a.parse_args();print(prepare(x.out,x.workload,x.train_tp,x.rollout_tp,x.mb,x.logprob_mb))
