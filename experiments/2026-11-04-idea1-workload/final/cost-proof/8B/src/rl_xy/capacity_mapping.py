"""Prespecified legal output-grid/tile search; no outcome-dependent constants."""
from dataclasses import replace
from functools import lru_cache
from itertools import product
from collections import Counter
from .mapping import ceildiv,choose_os_mapping
from .multicore import spans,estimate_multicore,partition_matmul
from .cost_model import CostEstimate,AnalyticalCostProvider


@lru_cache(maxsize=8192)
def local_mapping(problem,capacity):
    return choose_os_mapping(problem,capacity,max_tile=512)


@lru_cache(maxsize=4096)
def capacity_partition(problem,hardware):
    cores=hardware.cores_per_die;capacity=hardware.sram_bytes_per_core
    def choices(dim):return sorted({min(dim,x) for x in [1,2,4,8,16,32,64,128,256,cores] if x<=cores})
    candidates=[]
    rate=(hardware.gemm_fp32_flops_per_ns or hardware.gemm_flops_per_ns) if max(problem.a_bytes,problem.b_bytes)>=4 else hardware.gemm_flops_per_ns
    for pn,pm,pb in product(choices(problem.n),choices(problem.m),choices(problem.batches)):
        if pn*pm*pb>cores:continue
        counts=Counter((bc,mc,nc) for _,bc in spans(problem.batches,pb) for _,mc in spans(problem.m,pm) for _,nc in spans(problem.n,pn))
        reads=writes=work=private=0
        for (bc,mc,nc),count in counts.items():
            local=replace(problem,batches=bc,m=mc,n=nc);mapping=local_mapping(local,capacity)
            reads+=mapping.dram_read_bytes*count;writes+=mapping.dram_write_bytes*count
            work=max(work,ceildiv(mapping.flops,rate));private=max(private,ceildiv(mapping.dram_read_bytes+mapping.dram_write_bytes,hardware.sram_bytes_per_ns))
        memory=max(ceildiv(reads+writes,hardware.dram_bytes_per_ns),ceildiv(reads+writes,hardware.module_path_bytes_per_ns),private)
        candidates.append((work+memory,reads+writes,-pn*pm*pb,pn,pm,pb))
    # The legacy partition is a legal candidate as well, so this search cannot
    # intentionally worsen declared local cost to favor one final platform.
    best,_,_,pn,pm,pb=min(candidates)
    baseline=partition_matmul(problem,cores,capacity)
    traffic=sum(p.dram_read_bytes+p.dram_write_bytes for _,_,p in baseline)
    baseline_cost=max(ceildiv(p.flops,rate) for _,_,p in baseline)+max(
        ceildiv(traffic,hardware.dram_bytes_per_ns),ceildiv(traffic,hardware.module_path_bytes_per_ns),
        max(ceildiv(p.dram_read_bytes+p.dram_write_bytes,hardware.sram_bytes_per_ns) for _,_,p in baseline))
    if baseline_cost<best:return baseline
    parts=[]
    for b,bc in spans(problem.batches,pb):
        for m,mc in spans(problem.m,pm):
            for n,nc in spans(problem.n,pn):
                parts.append((len(parts),(b,bc,m,mc,n,nc),local_mapping(replace(problem,m=mc,n=nc,batches=bc),capacity)))
    if sum(x[2].flops for x in parts)!=problem.flops:raise ValueError('capacity-aware FLOP conservation')
    return tuple(parts)


class CapacityAwareCostProvider:
    def estimate(self,spec,hardware):
        if hardware.cores_per_die==1:return AnalyticalCostProvider().estimate(spec,hardware)
        e=estimate_multicore(spec,hardware,CostEstimate,partitioner=capacity_partition)
        return replace(e,source='analytic_capacity_aware_multicore_v1_uncalibrated',
            partitions=tuple({**p,'policy':'legal_output_grid_and_private_OS_tile_search','tile_upper_bound':512} for p in e.partitions))
