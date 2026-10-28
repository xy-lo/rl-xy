#!/usr/bin/env python3
"""Bounded preflight counts/RSS; emits no native performance result."""
import argparse,json,time,resource,hashlib
from pathlib import Path
from dataclasses import asdict,replace
from rl_xy.config import load_case
from rl_xy.baseline_contract import BaselinePolicy
from rl_xy.aligned_graph import build_aligned_graph
from rl_xy.wafer_run import effective_case
from rl_xy.gpu_model import load_gpu_profile,effective_gpu_case,GPUCostProvider
from rl_xy.cost_model import AnalyticalCostProvider,CostEstimate
from rl_xy.capacity_mapping import CapacityAwareCostProvider
class ShapeOnly:
 def estimate(self,spec,h):
  reads=spec.read_bytes+sum(m.batches*(m.m*m.k*m.a_bytes+m.k*m.n*m.b_bytes) for m in spec.matmuls)
  writes=spec.write_bytes+sum(m.batches*m.m*m.n*m.output_bytes for m in spec.matmuls)
  return CostEstimate(1,0,1,reads,writes,0,(),source='census_only_not_performance',core_workspace_bytes=(0,)*h.cores_per_die,core_compute_ns=(0,)*h.cores_per_die)
def main():
 a=argparse.ArgumentParser();a.add_argument('--case',type=Path,required=True);a.add_argument('--policy',type=Path,required=True);a.add_argument('--scenario',type=Path,required=True);a.add_argument('--gpu-profile',type=Path);a.add_argument('--shape-only',action='store_true');a.add_argument('--out',type=Path,required=True);x=a.parse_args()
 c,s,_=effective_case(load_case(x.case),x.scenario);p=BaselinePolicy(**json.loads(x.policy.read_text()));world=54
 provider=CapacityAwareCostProvider()
 if x.gpu_profile:
  gpu=load_gpu_profile(x.gpu_profile);c=effective_gpu_case(c,gpu);provider=GPUCostProvider(gpu);world=gpu['gpu_count']
 if x.shape_only:provider=ShapeOnly()
 from rl_xy.memoized_costs import MemoizedCostProvider
 provider=MemoizedCostProvider(provider)
 start=time.monotonic();g=build_aligned_graph(c,world,p,provider,'/probe',active=48)
 r=dict(status='PASS',kind='budget_preflight_no_native_timestamps',shape_only=x.shape_only,build_wall_s=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  native_ops=len(g.plan.operations),native_rank_events=sum(len(o.participant_ranks) for o in g.plan.operations),component_templates=g.audit['component_template_count'],
  component_invocations=g.audit['component_invocation_count'],logical_ops=sum(e['contraction'].certificate['logical_operation_count'] for e in g.component_expansions),
  preflight_peaks=g.plan.preflight_peaks,component_checks_pass=all(ch.passed for ch in g.component_checks),policy_sha256=g.audit['policy_fingerprint'],
  modeled_performance_result=False)
 x.out.parent.mkdir(parents=True,exist_ok=True)
 if x.shape_only:
  from rl_xy.gemm_census import census_graph
  census=census_graph(g,active=48);census.update(status='SHAPE_ONLY_BUDGET_CENSUS_NO_PERFORMANCE',policy_sha256=g.audit['policy_fingerprint'])
  x.out.with_name(x.out.stem+'-census.json').write_text(json.dumps(census,indent=2)+'\n')
 x.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','build_wall_s','peak_RSS_KiB','native_ops','logical_ops','component_templates']}))
if __name__=='__main__':main()
