"""Evaluate EXACTLY the existing OS candidate set with less Python work.

Simulation wall-clock acceleration only: K does not change input traffic for
an output-stationary (M,N) tile; choose the largest legal candidate K, matching
the original tie-break. No candidate, hardware parameter or modeled ns changes.
"""
from bisect import bisect_right
from functools import lru_cache
from rl_xy.mapping import MappingEstimate,ceildiv

@lru_cache(maxsize=8192)
def choose_os_mapping_exact(problem,capacity,max_tile=32):
    if max_tile<=0:raise ValueError('max_tile must be positive')
    if type(capacity) is not int or capacity<=0:raise ValueError('invalid SRAM capacity/tile')
    choices=[sorted({min(dim,x) for x in (1,2,4,8,16,32,64,128,256,max_tile) if x<=max_tile}) for dim in (problem.m,problem.n,problem.k)]
    best=None;tile=None;workspace=None
    writes=problem.batches*problem.m*problem.n*problem.output_bytes
    for m in choices[0]:
        for n in choices[1]:
            room=capacity-m*n*problem.accumulator_bytes
            limit=room//(m*problem.a_bytes+n*problem.b_bytes)
            j=bisect_right(choices[2],limit)-1
            if j<0:continue
            k=choices[2][j]
            reads=problem.batches*(problem.m*problem.k*ceildiv(problem.n,n)*problem.a_bytes+problem.k*problem.n*ceildiv(problem.m,m)*problem.b_bytes)
            key=(reads+writes,-k,-m,-n)
            if best is None or key<best:
                best=key;tile=(m,n,k);workspace=m*k*problem.a_bytes+k*n*problem.b_bytes+m*n*problem.accumulator_bytes
    if best is None:raise ValueError(f'no legal OS mapping in SRAM capacity={capacity} B')
    return MappingEstimate(tile,workspace,best[0]-writes,writes,problem.flops)

def install_exact_mapper():
    import rl_xy.mapping as mapping
    import rl_xy.capacity_mapping as capacity
    import rl_xy.multicore as multicore
    original=mapping.choose_os_mapping
    mapping.choose_os_mapping=choose_os_mapping_exact
    capacity.choose_os_mapping=choose_os_mapping_exact
    multicore.choose_os_mapping=choose_os_mapping_exact
    return original
