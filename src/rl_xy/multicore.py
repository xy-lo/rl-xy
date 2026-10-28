"""Static rectangular output partition, private per-core SRAM, shared module path.

No split-K: outputs are disjoint; no hidden inter-core reduction or cache reuse.
This is a conservative legal baseline, not an optimal mapping or a cycle model.
"""
from dataclasses import replace
from functools import lru_cache
from .mapping import choose_os_mapping, ceildiv


def spans(size, parts):
    q, r = divmod(size, parts)
    start = 0
    for i in range(parts):
        count = q + (i < r)
        yield start, count
        start += count


@lru_cache(maxsize=4096)
def partition_matmul(problem, cores, capacity):
    if type(cores) is not int or cores <= 0:
        raise ValueError('positive core count required')
    # Fixed N -> M -> batch policy. Small outputs leave cores idle; no ideal
    # division of total FLOPs by the configured number of cores is assumed.
    pn = min(problem.n, cores)
    pm = min(problem.m, cores // pn)
    pb = min(problem.batches, cores // (pn * pm))
    result = []
    for b, bc in spans(problem.batches, pb):
        for m, mc in spans(problem.m, pm):
            for n, nc in spans(problem.n, pn):
                local = replace(problem, m=mc, n=nc, batches=bc)
                result.append((len(result), (b,bc,m,mc,n,nc), choose_os_mapping(local,capacity)))
    if sum(mapping.flops for _,_,mapping in result) != problem.flops:
        raise ValueError('multicore output/FLOP conservation failed')
    return tuple(result)


def estimate_multicore(spec, hardware, estimate_type, *, partitioner=None):
    cores=hardware.cores_per_die
    workspace=[0]*cores
    compute=[0]*cores
    reads=writes=memory_ns=0
    mappings=[]
    partitions=[]
    critical_ns=0
    for problem in spec.matmuls:
        parts=(partitioner(problem,hardware) if partitioner else partition_matmul(problem,cores,hardware.sram_bytes_per_core))
        per_compute=[0]*cores
        per_memory=[0]*cores
        local_reads=local_writes=0
        for core,extent,mapping in parts:
            workspace[core]=max(workspace[core],mapping.workspace_bytes)
            rate=hardware.gemm_fp32_flops_per_ns if max(problem.a_bytes,problem.b_bytes)>=4 else None
            per_compute[core]=ceildiv(mapping.flops,rate or hardware.gemm_flops_per_ns)
            per_memory[core]=ceildiv(mapping.dram_read_bytes+mapping.dram_write_bytes,hardware.sram_bytes_per_ns)
            compute[core]+=per_compute[core]
            local_reads+=mapping.dram_read_bytes;local_writes+=mapping.dram_write_bytes
        # Shared DRAM and module/NoC path are not multiplied by the core count.
        service=max(ceildiv(local_reads+local_writes,hardware.dram_bytes_per_ns),
                    ceildiv(local_reads+local_writes,hardware.module_path_bytes_per_ns),max(per_memory))
        memory_ns+=service;critical_ns+=max(per_compute)
        reads+=local_reads;writes+=local_writes
        mappings.append(max((p[2] for p in parts),key=lambda m:m.workspace_bytes))
        partitions.append({'policy':'N_then_M_then_batch_no_split_K','active_cores':len(parts),
            'grid_n':len({p[1][4] for p in parts}),'grid_m':len({p[1][2] for p in parts}),
            'flops_sum':sum(p[2].flops for p in parts),'read_bytes_sum':local_reads,'write_bytes_sum':local_writes})
    active=min(cores,max(spec.vector_ops,spec.read_bytes+spec.write_bytes))
    if active:
        vwork=[q for _,q in spans(spec.vector_ops,active)]
        traffic=[q for _,q in spans(spec.read_bytes+spec.write_bytes,active)]
        vector_ns=max(ceildiv(v,hardware.vector_ops_per_ns) for v in vwork)
        for core in range(active):
            workspace[core]=max(workspace[core],16)
            compute[core]+=ceildiv(vwork[core],hardware.vector_ops_per_ns)
        memory_ns+=max(ceildiv(sum(traffic),hardware.dram_bytes_per_ns),
            ceildiv(sum(traffic),hardware.module_path_bytes_per_ns),
            max(ceildiv(t,hardware.sram_bytes_per_ns) for t in traffic))
        critical_ns+=vector_ns
        reads+=spec.read_bytes;writes+=spec.write_bytes
    if any(w>hardware.sram_bytes_per_core for w in workspace):
        raise ValueError('private SRAM capacity exceeded')
    return estimate_type(max(1,critical_ns+memory_ns),critical_ns,memory_ns,reads,writes,
        sum(workspace),tuple(mappings),source='analytic_private_sram_multicore_v1_not_hardware_calibrated',
        core_workspace_bytes=tuple(workspace),core_compute_ns=tuple(compute),partitions=tuple(partitions))
