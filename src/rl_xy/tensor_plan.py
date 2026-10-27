"""Conservative no-recompute tensor plan, with executable Astra resource contracts.

Forward intermediates are out-of-place and pinned through their backward
microbatch (or through the forward invocation for inference). This is an
explicit conservative retention policy, not an allocator's optimal peak.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re

from .model import enumerate_parameters, kv_bytes_per_processed_token_per_die
from .operator_ir import KernelSpec
from .types import CaseConfig, Operation
from .execution_contract import prompt_for, local_qkv_width


@dataclass
class TensorPlan:
    contract: dict
    tensors: list[dict]
    operations: list[Operation]
    preflight_peaks: dict[str, int]


def build_tensor_plan(case: CaseConfig, operations: list[Operation], log_prefix: str) -> TensorPlan:
    m, h = case.model, case.hardware
    tensors: dict[str, dict] = {}
    rank_contract = {}
    alive = {r: set() for r in range(h.dies)}
    groups: dict[tuple, list[str]] = {}
    kv: dict[tuple, list[str]] = {}
    payloads: dict[tuple, list[str]] = {}
    probabilities = {}
    backward = {}
    reserve_started = set()
    nodes = {}
    pending = {}
    roles = {}
    trajectories = {t.trajectory_id: t for t in case.workload.trajectories}
    parameter_rows = enumerate_parameters(case)
    kv_producer = ".rope_apply" if m.rotary_embedding else ".qkv_bias_add" if m.qkv_bias else ".qkv"
    kv_width = m.kv_heads * m.head_dim // case.strategy.tensor_parallel

    for rank in range(h.dies):
        initial = {}
        for p in parameter_rows:
            for kind, width in (("actor",m.actor_dtype_bytes),("reference",m.reference_dtype_bytes),
                                ("master",m.master_dtype_bytes),("adam_m",m.optimizer_dtype_bytes),
                                ("adam_v",m.optimizer_dtype_bytes),("gradient",m.gradient_dtype_bytes)):
                sid=f"d{rank}.{kind}.{p.parameter_id}"
                size=p.local_elements_per_die*width
                tensors[sid]={"storage_id":sid,"tensor_id":sid,"shape":[p.local_elements_per_die],
                    "dtype_bytes":width,"nbytes":size,"rank":rank,"kind":kind,"round_id":None,
                    "created_node":None,"released_node":None,"persistent":True}
                initial[sid]=size;alive[rank].add(sid)
        rank_contract[str(rank)]={"initial_objects":initial,"dram_capacity":h.dram_bytes_per_die,"nodes":{}}
        if h.cores_per_die > 1:
            rank_contract[str(rank)].update(cores_per_die=h.cores_per_die,sram_bytes_per_core=h.sram_bytes_per_core)
        for role in ("actor","reference","gradient","master","adam_m","adam_v"):
            roles[rank,role]=[sid for sid in initial if sid.startswith(f"d{rank}.{role}.")]

    result=[]
    for op in operations:
        meta=dict(op.metadata)
        if op.name.endswith('_loss_backward'):
            trajectory=trajectories[meta.get('trajectory_id', op.phase.removeprefix('train_'))]
            response=len(trajectory.response_token_ids)
            tokens=len(prompt_for(case, trajectory))+response-1
            elements=tokens*(m.vocab_size//case.strategy.tensor_parallel)
            meta['masked_prompt_logits_gradient']=True
            op=replace(op,tensor_bytes=elements*4,vector_ops=op.vector_ops+elements,metadata=meta)
        prefix=meta.get("forward_prefix")
        is_backward=bool(meta.get("backward_prefix"))
        if prefix and op.name.endswith(kv_producer) and op.phase=="rollout":
            copy_bytes=2*meta["tokens"]*kv_width*m.actor_dtype_bytes
            original = op.kernel_spec or KernelSpec(op.name.rsplit('.', 1)[-1], vector_ops=op.vector_ops,
                                                     read_bytes=2*op.tensor_bytes, write_bytes=op.tensor_bytes)
            spec=replace(original,read_bytes=original.read_bytes+copy_bytes,
                         write_bytes=original.write_bytes+copy_bytes)
            op=replace(op,kernel_spec=spec)
        participants=op.participant_ranks or tuple(range(h.dies))
        for rank in participants:
            key=(rank,op.round_id,prefix)
            tape=groups.setdefault(key,[]) if prefix else []
            read=set(tape)
            outputs=[]
            allocations={}
            frees=[]
            before=after=0

            def add(label: str, size: int, kind="activation", width=4, shape=None):
                if size<=0 or size%width:raise ValueError(f"invalid {kind} byte/dtype contract: {op.name}")
                sid=f"r{op.round_id}.d{rank}.{op.op_id}.{label}"
                if sid in tensors:raise ValueError("duplicate tensor storage identity")
                shape=shape or [size//width]
                product=1
                for dim in shape:product*=dim
                if product*width!=size:raise ValueError("shape/byte mismatch")
                tensors[sid]={"storage_id":sid,"tensor_id":sid,"shape":list(shape),"dtype_bytes":width,
                    "nbytes":size,"rank":rank,"kind":kind,"round_id":op.round_id,
                    "created_node":op.node_id,"released_node":None,"persistent":False}
                alive[rank].add(sid);allocations[sid]=size;outputs.append(sid)
                return sid

            def release(ids):
                for sid in dict.fromkeys(ids):
                    if sid in alive[rank] and not tensors[sid]["persistent"]:
                        alive[rank].remove(sid);frees.append(sid);tensors[sid]["released_node"]=op.node_id

            round_payload=[sid for sid in alive[rank] if tensors[sid]["round_id"]==op.round_id
                           and tensors[sid]["kind"] in ("token","trajectory","reward","advantage","log_prob")]
            if op.kind=="compute" and meta.get("local_transfer"):
                if is_backward:read.update(backward.get((rank,op.round_id,op.phase),[]))
                if meta["local_transfer"]=="write":
                    outputs.extend(pending.pop((rank,meta["parent_op"])))
                    read.difference_update(outputs)
                    for sid in outputs:tensors[sid]["ready_node"]=op.node_id
                elif not prefix:
                    read.update(payloads.get((rank,op.round_id,op.phase),[]))
            elif op.kind=="compute":
                if prefix:
                    read.update(roles[rank,meta.get("model_role","actor")])
                    read.update(round_payload)
                    if op.phase=="rollout":
                        trajectory=next(t for t in trajectories if prefix.startswith(f"rollout_{t}_"))
                        read.update(kv.get((rank,op.round_id,trajectory),[]))
                        if (rank,op.round_id) not in reserve_started:
                            before=sum(len(prompt_for(case, t))+case.workload.max_response_length-1
                                       for t in trajectories.values())*kv_bytes_per_processed_token_per_die(case)
                            reserve_started.add((rank,op.round_id))
                    if op.name.endswith(".sample_on_rank0"):
                        add("sampled_token",4,"token")
                    elif op.kernel_spec and op.kernel_spec.matmuls:
                        for i,p in enumerate(op.kernel_spec.matmuls):
                            add(f"product{i}",p.batches*p.m*p.n*p.output_bytes,width=p.output_bytes,
                                shape=(p.batches,p.m,p.n))
                    else:
                        width=2 if meta.get("dtype")=="bf16" or op.name.endswith(".embedding") else 4
                        add("output",op.tensor_bytes,width=width)
                    if ".rmsnorm" in op.name or op.name.endswith(".final_rmsnorm"):
                        add("norm_statistics",meta["tokens"]*4,"norm_statistics")
                    if op.name.endswith(kv_producer) and op.phase=="rollout":
                        size=2*meta["tokens"]*kv_width*2
                        sid=add("kv_increment",size,"kv",2,
                                (2,meta["tokens"],m.kv_heads//case.strategy.tensor_parallel,m.head_dim))
                        kv.setdefault((rank,op.round_id,trajectory),[]).append(sid);before-=size
                elif is_backward:
                    fkey=(rank,op.round_id,f"{op.phase}_forward")
                    read.update(groups.get(fkey,[]))
                    if not groups.get(fkey):raise ValueError("backward lacks forward tape")
                    read.update(roles[rank,"actor"])
                    read.update(backward.get((rank,op.round_id,op.phase),[]))
                    if op.name.endswith(".gradient_accumulate_rmw"):
                        read.update(roles[rank,"gradient"]);outputs.extend(roles[rank,"gradient"])
                        release(groups.get(fkey,[]));release(backward.get((rank,op.round_id,op.phase),[]))
                        groups[fkey]=[];backward[rank,op.round_id,op.phase]=[]
                    else:
                        if op.kernel_spec and op.kernel_spec.matmuls:
                            for i,p in enumerate(op.kernel_spec.matmuls):
                                add(f"gradient_product{i}",p.batches*p.m*p.n*p.output_bytes,
                                    "parameter_gradient" if p.direction=="dweight" else "activation_gradient",
                                    p.output_bytes,(p.batches,p.m,p.n))
                        elif op.name.endswith(".embedding_backward"):
                            embedding=parameter_rows[0].local_elements_per_die
                            add("embedding_parameter_gradient",embedding*4,"parameter_gradient")
                        elif op.name.endswith(".qkv_bias_backward"):
                            add("qkv_bias_parameter_gradient",local_qkv_width(case)*4,"parameter_gradient")
                        else:
                            add("activation_gradient",op.tensor_bytes,"activation_gradient")
                        if "rmsnorm" in op.name:
                            add("norm_parameter_gradient",m.hidden_size*4,"parameter_gradient")
                        backward.setdefault((rank,op.round_id,op.phase),[]).extend(outputs)
                elif op.name in ("zero_gradient_buffer","apply_gradient_clip"):
                    read.update(roles[rank,"gradient"]);outputs.extend(roles[rank,"gradient"])
                elif op.name=="adamw_read_modify_write":
                    for role in ("gradient","master","adam_m","adam_v"):read.update(roles[rank,role])
                    for role in ("actor","master","adam_m","adam_v"):outputs.extend(roles[rank,role])
                elif op.name=="release_round_temporaries":
                    release([sid for sid in alive[rank] if tensors[sid]["round_id"]==op.round_id])
                elif op.name=="publish_actor_version":
                    read.update(roles[rank,"actor"])
                else:
                    read.update(round_payload)
                    if op.phase.startswith("train_"):
                        read.update(groups.get((rank,op.round_id,f"{op.phase}_forward"),[]))
                        read.update(roles[rank,"gradient"])
                    if op.name=="local_gradient_square_sum":
                        read.update(roles[rank,"gradient"]);add("local_norm",4,"scalar")
                    else:
                        kind={"input":"trajectory","reward":"reward","advantage":"advantage"}.get(op.phase,"temporary")
                        size=op.tensor_bytes
                        if op.name.endswith("_loss_backward"):
                            t=trajectories[meta.get('trajectory_id', op.phase.removeprefix("train_"))]
                            length=len(prompt_for(case,t))+len(t.response_token_ids)-1
                            size=length*(m.vocab_size//case.strategy.tensor_parallel)*4
                        add("output",size,kind)
                        payloads[rank,op.round_id,op.phase]=list(outputs)
                        if op.name.endswith("_loss_backward"):
                            backward.setdefault((rank,op.round_id,op.phase),[]).extend(outputs)
            elif op.kind=="collective":
                if prefix:read.update(tape)
                elif is_backward:read.update(backward.get((rank,op.round_id,op.phase),[]))
                else:
                    read.update(payloads.get((rank,op.round_id,op.phase),[]))
                    if op.phase=="publish":read.update(roles[rank,"actor"])
                kind="log_prob" if op.name.endswith(".logsoftmax_target_logit") else "communication_output"
                sid=add("collective_result",op.payload_bytes,kind,2 if meta.get("dtype")=="bf16" else 4)
                if kind=="log_prob":probabilities[rank,op.round_id,prefix]=sid
                if is_backward:backward.setdefault((rank,op.round_id,op.phase),[]).append(sid)
            elif op.kind=="p2p":
                if rank==op.src_rank:
                    read.update(tape if prefix else payloads.get((rank,op.round_id,op.phase),[]))
                    if not read:read.update(round_payload)
                else:
                    kind="token" if "sample_token" in op.name else "communication_output"
                    if op.phase in ("input","reward","advantage"):kind=op.phase if op.phase!="input" else "trajectory"
                    add("received_payload",op.payload_bytes,kind,4)
                    payloads[rank,op.round_id,op.phase]=list(outputs)
            if prefix:
                tape.extend(sid for sid in outputs if sid in allocations and tensors[sid]["kind"] not in ("kv","token"))
                terminal_name=meta.get("parent_name",op.name) if meta.get("local_transfer")=="write" else op.name
                eligible=(not meta.get("endpoints_lowered") or (op.kind=="p2p" and rank==op.src_rank))
                terminal=eligible and (terminal_name.endswith(".sample_token_rank0_to_rank1") or terminal_name.endswith(".logsoftmax_target_logit"))
                if terminal and not meta.get("save_tape"):
                    release([sid for sid in tape if tensors[sid]["kind"]!="log_prob"])
                    groups[key]=[]
                if op.phase=="rollout" and terminal and terminal_name.endswith(".sample_token_rank0_to_rank1"):
                    trajectory=next(t for t in trajectories if prefix.startswith(f"rollout_{t}_"))
                    response_len=len(trajectories[trajectory].response_token_ids)
                    final=prefix.endswith("_prefill") if response_len==1 else prefix.endswith(f"_decode{response_len-1}")
                    if final:
                        release(kv.get((rank,op.round_id,trajectory),[]))
                        prompt_length=len(prompt_for(case,trajectories[trajectory]))
                        processed=prompt_length+response_len-1
                        after=-(case.workload.max_response_length+prompt_length-1-processed)*kv_bytes_per_processed_token_per_die(case)
            # Requiring all pinned tensors live is conservative. It is not a
            # claim that every kernel physically rereads every byte of the tape.
            read.difference_update(allocations)
            if op.kind!="compute" and meta.get("endpoints_lowered") and outputs:
                pending[rank,op.op_id]=list(outputs)
                outputs=[]  # network completion is not destination DRAM READY
            c={"reads":sorted(read),"allocations":allocations,"ready_outputs":list(dict.fromkeys(outputs)),
               "frees":frees,"workspace_bytes":0,"sram_capacity":h.sram_bytes_per_core,
               "local_memory":op.kind=="compute" and meta.get("execution_domain")!="host",
               "reserve_before":before,"reserve_after":after}
            rank_contract[str(rank)]["nodes"][str(op.node_id)]=c
        result.append(op)
    contract={"schema_version":"1.0","policy":"conservative_out_of_place_no_recompute",
              "log_prefix":log_prefix,"ranks":rank_contract}
    for tensor in tensors.values():tensor.setdefault('ready_node',tensor['created_node'])
    return TensorPlan(contract,list(tensors.values()),result,{})


def finalize_tensor_costs(plan: TensorPlan, case: CaseConfig, *, provider=None) -> None:
    from .mapped_costs import apply_mapped_costs
    plan.operations=apply_mapped_costs(case,plan.operations,provider=provider)
    for op in plan.operations:
        workspace=op.metadata.get("cost_estimate",{}).get("sram_workspace_bytes",0)
        for rank in op.participant_ranks:
            node=plan.contract["ranks"][str(rank)]["nodes"][str(op.node_id)]
            node["workspace_bytes"]=workspace
            if case.hardware.cores_per_die>1:
                node['core_workspace_bytes']=list(op.metadata.get('cost_estimate',{}).get(
                    'core_workspace_bytes',[0]*case.hardware.cores_per_die))
            node["local_memory"]=(
                op.kind=='compute' and op.metadata.get('execution_domain')!='host')
    plan.preflight_peaks=preflight_contract(plan.contract)


def preflight_contract(contract: dict) -> dict[str,int]:
    peaks={}
    for rank,r in contract["ranks"].items():
        live=dict(r["initial_objects"]);ready=set(live);reserved=0
        peak=sum(live.values())
        if peak>r["dram_capacity"]:raise ValueError("initial DRAM overflow")
        for node,n in sorted(r["nodes"].items(),key=lambda row:int(row[0])):
            if set(n["reads"])-ready:raise ValueError(f"rank{rank}/node{node}: read absent/not ready")
            if set(n["allocations"]) & set(live):raise ValueError("duplicate allocation")
            reserved+=n["reserve_before"]
            if reserved<0:raise ValueError("KV reservation consumed twice")
            live.update(n["allocations"]);used=sum(live.values())
            if used+reserved>r["dram_capacity"]:raise ValueError(f"DRAM admission overflow rank{rank}/node{node}")
            if 'core_workspace_bytes' in n:
                values=n['core_workspace_bytes']
                if (len(values)!=r['cores_per_die'] or sum(values)!=n['workspace_bytes'] or
                    any(type(w) is not int or w<0 or w>r['sram_bytes_per_core'] for w in values)):
                    raise ValueError('private SRAM overflow/invalid mapping')
            elif n["workspace_bytes"]>n["sram_capacity"]:raise ValueError("SRAM overflow")
            peak=max(peak,used)
            if set(n["ready_outputs"])-set(live):raise ValueError("READY without storage")
            if any('.reference.' in sid for sid in n['ready_outputs']):raise ValueError('immutable reference write')
            ready.update(n["ready_outputs"])
            for sid in n["frees"]:
                if sid not in live:raise ValueError("double free")
                del live[sid];ready.discard(sid)
            reserved+=n["reserve_after"]
            if reserved<0:raise ValueError("invalid reservation release")
        if live!=r["initial_objects"] or reserved:raise ValueError("temporary/reservation leak")
        peaks[f"d{rank}.dram"]=peak
    return peaks
