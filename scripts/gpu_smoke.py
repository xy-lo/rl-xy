#!/usr/bin/env python3
"""One bounded real veRL step, isolated GPU container, existing local weights.

Uses the user's existing read-only veRL environment and length collector;
the activation flag exists ONLY inside this dedicated container, never on host.
Not a learning-quality experiment or a steady-state performance benchmark.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
INFRA=Path(os.environ.get('RLXY_LEGACY_ROOT', str(Path.home()/'RLInfra')))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    for key,value in {'TRITON_CACHE_DIR':'/tmp/rlxy-triton','XDG_CACHE_HOME':'/tmp/rlxy-cache',
            'TORCHINDUCTOR_CACHE_DIR':'/tmp/rlxy-torch','HF_HOME':'/tmp/rlxy-hf',
            'HF_DATASETS_CACHE':'/tmp/rlxy-datasets'}.items():
        os.environ.setdefault(key,value)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    a=p.parse_args()
    if not Path('/.dockerenv').is_file():raise RuntimeError('run only in the dedicated GPU Docker container')
    if Path(a.run_id).name!=a.run_id or a.run_id in ('','.','..'):p.error('unique path component required')
    out=ROOT/'runs/gpu_matched'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    model=INFRA/'models/Qwen2.5-0.5B-Instruct';data=INFRA/'data/gsm8k/train.parquet'
    flag=Path('/root/RLInfra/scripts/rltrace_enabled.json')
    if flag.exists():raise RuntimeError('refuse to replace an existing capture activation')
    args=[sys.executable,'-m','verl.trainer.main_ppo','trainer.use_v1=False',
        f'data.train_files={data}',f'data.val_files={INFRA}/data/gsm8k/test.parquet',
        'data.train_batch_size=32','data.max_prompt_length=128','data.max_response_length=4',
        'data.filter_overlong_prompts=True','data.truncation=error',
        'algorithm.adv_estimator=grpo','algorithm.norm_adv_by_std_in_grpo=False','algorithm.use_kl_in_reward=False',
        f'actor_rollout_ref.model.path={model}','actor_rollout_ref.model.use_remove_padding=True',
        '+actor_rollout_ref.model.override_config.attn_implementation=sdpa',
        'actor_rollout_ref.model.enable_gradient_checkpointing=False',
        'actor_rollout_ref.actor.optim.lr=1e-6','actor_rollout_ref.actor.ppo_mini_batch_size=32',
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1','actor_rollout_ref.actor.use_dynamic_bsz=False',
        'actor_rollout_ref.actor.ppo_epochs=1','actor_rollout_ref.actor.loss_agg_mode=token-mean',
        'actor_rollout_ref.actor.use_kl_loss=True','actor_rollout_ref.actor.kl_loss_coef=0.001',
        'actor_rollout_ref.actor.kl_loss_type=low_var_kl','actor_rollout_ref.actor.entropy_coeff=0',
        'actor_rollout_ref.actor.fsdp_config.param_offload=False','actor_rollout_ref.actor.fsdp_config.optimizer_offload=False',
        'actor_rollout_ref.rollout.name=vllm','actor_rollout_ref.rollout.tensor_model_parallel_size=1',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.3','actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1',
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False','actor_rollout_ref.rollout.enforce_eager=True',
        'actor_rollout_ref.rollout.free_cache_engine=True','actor_rollout_ref.rollout.n=2',
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1','actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False',
        'actor_rollout_ref.ref.fsdp_config.param_offload=False',
        'trainer.critic_warmup=0','trainer.logger=[console]','trainer.project_name=rlxy_matched_smoke',
        f'trainer.experiment_name={a.run_id}','trainer.n_gpus_per_node=1','trainer.nnodes=1',
        'trainer.save_freq=0','trainer.test_freq=0','trainer.val_before_train=False',
        'trainer.total_epochs=1','trainer.total_training_steps=1',f'trainer.default_local_dir={out}/checkpoints',
        'ray_kwargs.ray_init.runtime_env.py_executable=null']
    report={'status':'running','scope':'one real numerical veRL step; smoke not steady-state/quality benchmark',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'argv':args,
        'packages':{name:importlib.metadata.version(name) for name in ('torch','verl','vllm','transformers')},
        'model_config_sha256':digest(model/'config.json'),'dataset_sha256':digest(data),
        'model_weights_sha256':{path.name:digest(path) for path in sorted(model.glob('*.safetensors'))},
        'python_header_sha256':digest(Path('/usr/include/python3.12/Python.h')) if Path('/usr/include/python3.12/Python.h').is_file() else None,
        'collector_sha256':digest(Path('/root/RLInfra/scripts/rltrace_hook.py')),
        'trainer_sha256':digest(INFRA/'setup/venv/lib/python3.12/site-packages/verl/trainer/ppo/ray_trainer.py'),
        'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader'],text=True)}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save();flag.write_text(json.dumps({'out_dir':str(out),'run_id':a.run_id}))
    try:
        start=time.monotonic()
        with (out/'launch.log').open('w') as log:
            r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,cwd=out,timeout=600)
        report.update(exit_status=r.returncode,process_wall_s=time.monotonic()-start,
            status='PASS' if r.returncode==0 and (out/(a.run_id+'.jsonl')).is_file() else 'FAIL')
        report['launch_log_sha256']=digest(out/'launch.log')
        if report['status']!='PASS':raise RuntimeError('veRL or required capture failed; inspect launch.log')
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if flag.exists():flag.unlink()
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()
