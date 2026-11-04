"""Explicit DRAM endpoint service around native AstraSim communication.

No separate network timer: COMP nodes own endpoint DRAM service and native
COMM owns network transit. Rank-local dependencies enforce this serial baseline.
"""
from dataclasses import replace
from .operator_ir import KernelSpec
from .types import CaseConfig, Operation


def lower_communication_endpoints(case: CaseConfig, operations: list[Operation]) -> list[Operation]:
    expanded=[]
    for op in operations:
        if op.kind=="compute":
            expanded.append(op)
            for norm,addition in (("rmsnorm_ffn_backward","residual_ffn_grad_add"),
                                   ("rmsnorm_attn_backward","residual_attn_grad_add")):
                if op.name.endswith('.'+norm):
                    elements=op.metadata['tokens']*case.model.hidden_size
                    spec=KernelSpec(addition,vector_ops=elements,read_bytes=elements*8,write_bytes=elements*4)
                    expanded.append(replace(op,op_id=op.op_id+'.residual_add',
                        name=op.name.rsplit('.',1)[0]+'.'+addition,kernel_spec=spec,
                        vector_ops=elements,flops=0,tensor_bytes=elements*4,
                        metadata={**op.metadata,'semantic':'add_residual_branch_gradient'}))
            continue
        for direction in ("read","network","write"):
            if direction=="network":
                expanded.append(replace(op,metadata={**op.metadata,"endpoints_lowered":True}))
                continue
            participants=op.participant_ranks if op.kind=="collective" else (
                (op.src_rank,) if direction=="read" else (op.dst_rank,))
            spec=KernelSpec("network_endpoint_"+direction,
                            read_bytes=op.payload_bytes if direction=="read" else 0,
                            write_bytes=op.payload_bytes if direction=="write" else 0)
            expanded.append(Operation(
                node_id=0,op_id=op.op_id+".endpoint_"+direction,round_id=op.round_id,phase=op.phase,
                name=op.name+".endpoint_"+direction,kind="compute",duration_ns=1,
                tensor_bytes=op.payload_bytes,participant_ranks=participants,kernel_spec=spec,
                metadata={**op.metadata,"local_transfer":direction,"parent_op":op.op_id,
                          "parent_name":op.name,"execution_domain":"device"}))
    last={r:None for r in range(case.hardware.dies)}
    result=[]
    for i,op in enumerate(expanded,1):
        deps={r:() if last[r] is None else (last[r],) for r in op.participant_ranks}
        result.append(replace(op,node_id=i,dependencies_by_rank=deps,
                              dependencies=tuple(sorted({x for ds in deps.values() for x in ds}))))
        for r in op.participant_ranks:last[r]=i
    return result
