#!/usr/bin/env python3
"""Bounded exact DRAM lower bound for ONE aligned training microbatch.

Full global batch/DO resident state is retained; rollout/export/logprob/reward
carriers are absent. Zero-workspace CensusProvider is never a timing model.
A pass is necessary for that sampled training stage, never full workflow fit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rl_xy.aligned_graph import AlignedGraphBuilder
from rl_xy.baseline_contract import BaselinePolicy,audit_policy,policy_fingerprint
from rl_xy.config import load_case
from rl_xy.evidence_io import phase_deadline,EvidencePhaseTimeout
from rl_xy.execution_contract import prompt_for,group_for
from rl_xy.gemm_census import CensusProvider
from rl_xy.gpu_model import effective_gpu_case,load_gpu_profile
from rl_xy.memoized_costs import MemoizedCostProvider
from rl_xy.reporting import write_json
from rl_xy.tensor_plan import preflight_contract
from rl_xy.wafer_run import effective_case

MEASUREMENT_ADMISSION_CEILING=(1<<63)-1
PROCESS_ADDRESS_SPACE_LIMIT=7*1024**3
OVERALL_SECONDS=180
SCOPE=('Exact declared tensor DRAM lifecycle lower bound for one training microbatch '
       'plus all global DO initial resident objects. No rollout/export/logprob/phase '
       'carriers, no other replicas active tape, no mapped SRAM workspace, no '
       'native/GPU execution, no full-workflow fit or performance conclusion.')


def _rss_peak_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform=='darwin' else value*1024


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def source_snapshot():
    paths=sorted((ROOT/'src'/'rl_xy').rglob('*.py'))+[Path(__file__).resolve()]
    rows={p.relative_to(ROOT).as_posix():_sha(p) for p in paths}
    return dict(files=rows,tree_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def measure_training_stage(case,policy,*,allocated,active,tp,microbatch,
        target_capacity_bytes=None,guard_bytes=0):
    """Compile exact one-mb alloc/READY/free with the entire DO state present.

    Only the compiler's admission ceiling is raised so an overflowing candidate
    still yields its exact required DRAM. Actual target capacity is compared
    independently and recorded. This cannot turn overflow into a fit.
    """
    for name,value in [('active',active),('allocated',allocated),('tp',tp),('microbatch',microbatch)]:
        if type(value) is not int or value<1:raise ValueError(name+' must be a positive integer')
    if type(guard_bytes) is not int or guard_bytes<0:raise ValueError('guard bytes must be a nonnegative integer')
    target=case.hardware.dram_bytes_per_die if target_capacity_bytes is None else target_capacity_bytes
    if type(target) is not int or not 0<target<MEASUREMENT_ADMISSION_CEILING:
        raise ValueError('physical target DRAM capacity must be a positive integer below measurement ceiling')
    candidate=replace(policy,train_tp=tp,train_micro_batch_trajectories=microbatch)
    admission=audit_policy(case.model,candidate,world_size=active)
    if not admission['official_batch_constraints_pass'] or not admission['selected_layout_constraints_pass']:
        raise ValueError('candidate policy invalid: '+str(admission['blockers']))
    if len(case.workload.trajectories)!=candidate.trajectory_batch_size:
        raise ValueError('full case trajectory count differs from global policy')
    shapes={(len(prompt_for(case,t)),len(t.response_token_ids)) for t in case.workload.trajectories}
    if len(shapes)!=1:raise ValueError('this sampled-capacity probe requires homogeneous global trajectory shapes')
    local_count=len(case.workload.trajectories)//(active//tp)
    for offset in range(0,len(case.workload.trajectories),local_count):
        counts=Counter(group_for(case,t) for t in case.workload.trajectories[offset:offset+local_count])
        if set(counts.values())!={candidate.responses_per_prompt}:
            raise ValueError('global contiguous dispatch does not contain complete response groups per DP')
    total_tokens=sum(len(t.response_token_ids) for t in case.workload.trajectories)
    if microbatch>len(case.workload.trajectories)//(active//tp):raise ValueError('microbatch exceeds local full batch')
    compiler_case=replace(case,hardware=replace(case.hardware,dram_bytes_per_die=MEASUREMENT_ADMISSION_CEILING))
    builder=None;provider=None;part=None;start=time.monotonic()
    try:
        provider=MemoizedCostProvider(CensusProvider())
        builder=AlignedGraphBuilder(compiler_case,allocated,candidate,provider,
            '/CAPACITY_ONLY_NOT_NATIVE',active=active,compact_components=False)
        initial={int(r):sum(c['initial_objects'].values()) for r,c in builder.contract['ranks'].items()}
        # Exercise the most resident DP replica, while every other replica's
        # persistent allocation remains present in the global contract.
        replica=max(range(active//tp),key=lambda i:(max(initial[r] for r in range(i*tp,(i+1)*tp)),
            sum(initial[r] for r in range(i*tp,(i+1)*tp)),-i))
        physical=tuple(range(replica*tp,(replica+1)*tp));local_batch=len(case.workload.trajectories)//(active//tp)
        chosen=case.workload.trajectories[replica*local_batch:replica*local_batch+microbatch]
        part=replace(builder.local_case,rounds=1,workload=replace(case.workload,trajectories=chosen),
            strategy=replace(builder.local_case.strategy,training_microbatch_trajectories=microbatch))
        bridges={rank:[] for rank in range(allocated)}
        builder.stage(part,physical,{'train'},'capacity.training.replica'+str(replica),
            training=True,reset_gradient=True,bridges=bridges)
        if any(bridges.values()):raise ValueError('training-only probe unexpectedly created external phase carriers')
        peaks=preflight_contract(builder.contract)
        zero={r:0 for r in physical};rmw={r:0 for r in physical};denominators=set();gradient_frees=[]
        for op in builder.ops:
            for rank in op.participant_ranks:
                if op.name=='zero_gradient_buffer':zero[rank]+=1
                if op.name.endswith('.gradient_accumulate_rmw') or op.metadata.get('training_batch_rmw'):rmw[rank]+=1
                n=builder.contract['ranks'][str(rank)]['nodes'][str(op.node_id)]
                gradient_frees.extend(s for s in n['frees'] if '.gradient.' in s)
            for key in ('global_denominator','global_loss_denominator'):
                if key in op.metadata:denominators.add(op.metadata[key])
        if set(zero.values())!={1} or set(rmw.values())!={1} or denominators!={total_tokens} or gradient_frees:
            raise ValueError('training-only zero/RMW/global-denominator/persistent-gradient invariant failed')
        if not all(c.passed for c in builder.checks):raise ValueError('training component checks failed')
        maximum=max(peaks[f'd{r}.dram'] for r in range(allocated));headroom=target-maximum
        return dict(status='TRAINING_STAGE_CAPACITY_ONLY_COMPLETE',scope=SCOPE,
            train_tp=tp,data_parallel=active//tp,microbatch_trajectories=microbatch,
            active=active,allocated=allocated,full_global_trajectory_count=len(case.workload.trajectories),
            full_global_token_denominator=total_tokens,global_policy=asdict(candidate),policy_sha256=policy_fingerprint(candidate),
            sampled_DP_replica=replica,sampled_physical_ranks=list(physical),
            sampled_source_trajectory_ids=[t.trajectory_id for t in chosen],
            initial_resident_bytes_by_rank={str(r):v for r,v in initial.items()},
            stage_peak_bytes_by_rank={str(r):peaks[f'd{r}.dram'] for r in range(allocated)},
            peak_required_bytes=maximum,physical_target_capacity_bytes=target,
            physical_target_headroom_bytes=headroom,guard_bytes=guard_bytes,
            headroom_after_guard_bytes=headroom-guard_bytes,
            sampled_stage_exceeds_physical_capacity=maximum>target,
            sampled_stage_within_guarded_capacity=maximum+guard_bytes<=target,
            measurement_compiler_admission_ceiling_bytes=MEASUREMENT_ADMISSION_CEILING,
            measurement_ceiling_is_not_hardware_capacity=True,
            gradient_zero_per_sampled_rank=zero,gradient_RMW_per_sampled_rank=rmw,
            gradient_storage_frees=gradient_frees,component_checks_pass=True,
            declared_tensor_lifetimes_exact=True,external_carriers_omitted=True,
            omitted_other_replicas_active_tape=True,SRAM_workspace_validated=False,
            provider='CensusProvider zero workspace; placeholder duration1 never exported or used as a timing result',
            full_workflow_fit=False,native_executed=False,GPU_executed=False,
            modeled_performance_result=False,elapsed_wall_seconds=time.monotonic()-start,
            process_highwater_RSS_bytes=_rss_peak_bytes())
    finally:
        # No TensorPlan/operations/cost-cache escapes the candidate result.
        part=None;builder=None;provider=None;gc.collect()


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['case','policy','scenario','gpu-profile','out']:
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--active',type=int,default=48)
    parser.add_argument('--train-tps',type=int,nargs='+',default=[1,2])
    parser.add_argument('--microbatches',type=int,nargs='+',default=[1,2,4,8,16])
    parser.add_argument('--platforms',choices=['GPU','WAFER'],nargs='+',default=['GPU','WAFER'])
    parser.add_argument('--guard-bytes',type=int,default=0,help='explicit capacity headroom requirement; no hidden reserve by default')
    return parser


def _apply_process_memory_guard():
    if not sys.platform.startswith('linux'):
        raise ValueError('bounded probe CLI requires Linux; portable unit API does not change process limits')
    soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    ceiling=PROCESS_ADDRESS_SPACE_LIMIT
    if hard!=resource.RLIM_INFINITY:ceiling=min(ceiling,hard)
    if soft!=resource.RLIM_INFINITY:ceiling=min(ceiling,soft)
    resource.setrlimit(resource.RLIMIT_AS,(ceiling,hard))
    return ceiling


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.out.exists():raise ValueError('output exists; preserve previous capacity evidence')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    limit=_apply_process_memory_guard();start=time.monotonic()
    report=dict(schema_version='rlxy.training_stage_capacity_probe/1',status='STARTING',scope=SCOPE,
        overall_timeout_seconds=OVERALL_SECONDS,process_address_space_soft_limit_bytes=limit,
        guard_bytes=args.guard_bytes,full_workflow_fit=False,modeled_performance_result=False,
        native_executed=False,GPU_executed=False,candidates=[])
    write_json(args.out,report)
    current=None;snapshot=None
    try:
        with phase_deadline('entire training capacity probe',OVERALL_SECONDS,enabled=True):
            snapshot=source_snapshot();report['source']=snapshot
            original=load_case(args.case);wafer,scenario,source=effective_case(original,args.scenario)
            gpu=load_gpu_profile(args.gpu_profile);gpu_case=effective_gpu_case(wafer,gpu)
            raw=json.loads(args.policy.read_text());policy=BaselinePolicy(**raw.get('policy',raw))
            report.update(model=asdict(original.model),full_global_policy=asdict(policy),
                full_global_case_id=original.case_id,full_global_trajectory_count=len(original.workload.trajectories),
                full_global_token_denominator=sum(len(t.response_token_ids) for t in original.workload.trajectories),
                shape_counts={str(k):v for k,v in Counter((len(prompt_for(original,t)),len(t.response_token_ids))
                    for t in original.workload.trajectories).items()})
            report['inputs']={name:dict(path=str(path.resolve()),sha256=_sha(path)) for name,path in
                [('case',args.case),('policy',args.policy),('scenario',args.scenario),('GPU_profile',args.gpu_profile),('wafer_hardware',source)]}
            platforms={'GPU':(gpu_case,gpu['gpu_count']),'WAFER':(wafer,scenario['mesh'][0]*scenario['mesh'][1])}
            for mb in args.microbatches:
                for tp in args.train_tps:
                    for platform in args.platforms:
                        current=dict(platform=platform,train_tp=tp,microbatch_trajectories=mb)
                        report['current_candidate']=current;write_json(args.out,report)
                        case,allocated=platforms[platform]
                        try:
                            row=measure_training_stage(case,policy,allocated=allocated,active=args.active,tp=tp,
                                microbatch=mb,guard_bytes=args.guard_bytes)
                        except ValueError as exc:
                            row=dict(status='INVALID_OR_FAILED_STAGE_CONTRACT',error=str(exc),scope=SCOPE,
                                full_workflow_fit=False,modeled_performance_result=False)
                        report['candidates'].append({**current,**row});gc.collect();write_json(args.out,report)
            if source_snapshot()!=snapshot:raise ValueError('source changed during probe')
            if any(_sha(Path(row['path']))!=row['sha256'] for row in report['inputs'].values()):
                raise ValueError('input changed during probe')
            report.pop('current_candidate',None)
            invalid=sum(r['status']!='TRAINING_STAGE_CAPACITY_ONLY_COMPLETE' for r in report['candidates'])
            report['candidate_counts']=dict(total=len(report['candidates']),invalid_or_failed_contracts=invalid,
                sampled_stage_capacity_overflows=sum(r.get('sampled_stage_exceeds_physical_capacity',False) for r in report['candidates']),
                sampled_stage_guard_failures=sum(not r.get('sampled_stage_within_guarded_capacity',True) for r in report['candidates']))
            report['status']='CAPACITY_MATRIX_COMPLETE_WITH_INVALID_CANDIDATES' if invalid else 'CAPACITY_MATRIX_COMPLETE_NO_PERFORMANCE'
            report['all_candidates_processed']=True
    except EvidencePhaseTimeout as exc:
        report.update(status='TIMEOUT_PARTIAL',error=str(exc),all_candidates_processed=False,failed_candidate=current)
    except MemoryError:
        gc.collect();report.update(status='PROCESS_MEMORY_GUARD_PARTIAL',all_candidates_processed=False,failed_candidate=current)
    except BaseException as exc:
        report.update(status='FAILED',error=type(exc).__name__+': '+str(exc),all_candidates_processed=False,failed_candidate=current)
        raise
    finally:
        report['elapsed_wall_seconds']=time.monotonic()-start;report['process_highwater_RSS_bytes']=_rss_peak_bytes()
        write_json(args.out,report)
    print(json.dumps({k:report.get(k) for k in ['status','elapsed_wall_seconds','process_highwater_RSS_bytes','all_candidates_processed']},indent=2))
    return 0 if report['status']=='CAPACITY_MATRIX_COMPLETE_NO_PERFORMANCE' else 2


if __name__=='__main__':raise SystemExit(main())
