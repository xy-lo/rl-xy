"""Qwen3 operator costs for iteration replay; no numeric weights or kernels.

Dense GEMMs retain existing platform mappers. Attention is a fused exact
causal/block operator with private SRAM admission and explicit recomputation.
All local compute and memory service are additive as in the frozen baseline.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
from functools import lru_cache
import json,math
from pathlib import Path
from .types import ModelConfig
from .operator_ir import Matmul,KernelSpec
from .model import enumerate_parameters
from .model_import import inspect_hf_model
from .config import load_case
from .wafer_run import effective_case
from .gpu_model import GPUCostProvider,load_gpu_profile,effective_gpu_case
from .capacity_mapping import CapacityAwareCostProvider
from .distributed_optimizer import layout_for_case
from .mapping import ceildiv


@dataclass(frozen=True)
class Cost:
    compute:float=0
    memory:float=0
    flops:int=0
    read:int=0
    write:int=0
    workspace:int=0
    @property
    def ns(self):return self.compute+self.memory
    def __add__(self,b):
        return Cost(self.compute+b.compute,self.memory+b.memory,self.flops+b.flops,
                    self.read+b.read,self.write+b.write,max(self.workspace,b.workspace))
    def __mul__(self,n):return Cost(self.compute*n,self.memory*n,self.flops*n,self.read*n,self.write*n,self.workspace)
    __rmul__=__mul__


@dataclass(frozen=True)
class Block:
    """Local work followed by an optional collective, one copy per TP rank."""
    name:str
    cost:Cost
    collective:str=''
    payload:int=0


def model_case(root,config):
    base=load_case(Path(root)/'configs/aligned_fixture/case.json')
    d=inspect_hf_model(config);r=d['config']
    m=ModelConfig(layers=r['num_hidden_layers'],hidden_size=r['hidden_size'],query_heads=r['num_attention_heads'],
        kv_heads=r['num_key_value_heads'],head_dim=d['head_dim'],ffn_size=r['intermediate_size'],vocab_size=r['vocab_size'],
        actor_dtype_bytes=2,reference_dtype_bytes=2,master_dtype_bytes=4,gradient_dtype_bytes=4,optimizer_dtype_bytes=4,
        qkv_bias=False,rotary_embedding=True,tie_word_embeddings=r['tie_word_embeddings'],qk_norm=True)
    case=replace(base,model=m)
    assert sum(x.unique_elements for x in enumerate_parameters(case))==d['unique_parameter_elements']
    return case,d


class Operators:
    def __init__(self,root,config,platform,*,compute_scale=1.0,network_scale=1.0):
        self.root=Path(root);self.config_path=Path(config).resolve();base,self.info=model_case(root,config);self.model=base.model
        wafer,self.scenario,_=effective_case(base,self.root/'configs/wafer54_fixture/wafer.json')
        self.platform=platform;self.compute_scale=compute_scale;self.network_scale=network_scale
        if platform=='gpu':
            self.profile=load_gpu_profile(self.root/'configs/gpu/a100_48_reference.json')
            self.case=effective_gpu_case(base,self.profile);self.provider=GPUCostProvider(self.profile)
        elif platform=='wafer':self.case=wafer;self.provider=CapacityAwareCostProvider();self.profile=None
        else:raise ValueError(platform)
        self.hw=self.case.hardware;self.units=self.hw.cores_per_die
        self.capacity=self.hw.dram_bytes_per_die
        self.rows=enumerate_parameters(self.case)
        prompt_file=Path(config).parent/"prompt_lengths.json"
        self.prompt_cap=max((x["prompt_length"] for x in json.loads(prompt_file.read_text()).values()),default=0) if prompt_file.exists() else 0
    def case_tp(self,tp):return replace(self.case,strategy=replace(self.case.strategy,tensor_parallel=tp))
    @lru_cache(maxsize=65536)
    def kernel(self,spec):
        e=self.provider.estimate(spec,self.hw)
        return Cost(e.compute_ns/self.compute_scale,e.local_memory_ns,sum(m.flops for m in spec.matmuls),e.dram_read_bytes,e.dram_write_bytes,e.sram_workspace_bytes)
    def vec(self,ops,reads,writes,name='vector'):
        return self.kernel(KernelSpec(name,vector_ops=int(ops),read_bytes=int(reads),write_bytes=int(writes)))
    @lru_cache(maxsize=65536)
    def mm(self,m,n,k,direction='forward'):
        return self.kernel(KernelSpec('qwen3_'+direction,matmuls=(Matmul(int(m),int(n),int(k),a_bytes=2,b_bytes=2,
            output_bytes=4 if direction in ('dinput','dweight') else 2,accumulator_bytes=4,direction=direction),)))
    def linear_backward(self,t,inputs,outputs):
        # Stored FP32 dY is converted to BF16 input; dW/accumulation stay FP32.
        cast=self.vec(t*outputs,t*outputs*4,t*outputs*2,'gradient_cast_bf16')
        return cast+self.mm(t,inputs,outputs,'dinput')+self.mm(inputs,outputs,t,'dweight')
    @lru_cache(maxsize=65536)
    def attention(self,queries:tuple,contexts:tuple,tp:int,backward=False):
        m=self.model;heads=m.query_heads//tp;hd=m.head_dim
        cap=self.hw.sram_bytes_per_core
        candidates=[]
        for q in (16,32,64,128,256):
            for k in (16,32,64,128,256):
                qq=min(q,max(queries));kk=min(k,max(contexts))
                work=(6*(qq+2*kk)*hd if backward else 2*(qq+2*kk)*hd+4*qq*hd)+4*qq*kk+16*qq
                if work<=cap:candidates.append((qq,kk,work))
        if not candidates:raise ValueError('attention has no legal private SRAM tile')
        qtile,ktile,workspace=max(candidates,key=lambda x:(x[0],x[1]))
        pairs=kv_reads=tasks=0
        for q,c in zip(queries,contexts):
            if q==1:
                pairs+=c;kv_reads+=c;tasks+=heads
            else:
                # Causal masked block matmuls include work within diagonal tiles.
                if q!=c:raise ValueError('prefill/teacher query must equal causal context')
                for start in range(0,q,qtile):
                    count=min(qtile,q-start);end=start+count
                    pairs+=count*end;kv_reads+=end;tasks+=heads
        flops=pairs*heads*hd*(10 if backward else 4)
        vector_ops=pairs*heads*(20 if backward else 5)
        units=min(self.units,tasks)
        if self.profile:
            unit_rate=self.profile['bf16_flops_per_ns']/self.units
            vector_rate=self.profile['vector_ops_per_ns']/self.units
            launch=self.profile['kernel_launch_ns']
        else:unit_rate=self.hw.gemm_flops_per_ns;vector_rate=self.hw.vector_ops_per_ns;launch=0
        compute=(flops/(units*unit_rate)+vector_ops/(units*vector_rate))/self.compute_scale+launch
        tokens=sum(queries)
        reads=(2*kv_reads*heads*hd*2+tokens*heads*hd*2)
        writes=tokens*heads*hd*2+tokens*heads*8
        if backward:
            reads=reads*2+tokens*heads*hd*6
            writes=tokens*(heads+2*m.kv_heads//tp)*hd*4
        traffic=reads+writes
        memory=max(traffic/self.hw.dram_bytes_per_ns,traffic/self.hw.module_path_bytes_per_ns,
                   traffic/(units*self.hw.sram_bytes_per_ns))
        return Cost(compute,memory,flops,int(reads),int(writes),workspace*units)
    @lru_cache(maxsize=65536)
    def layer_forward(self,queries,contexts,tp):
        m=self.model;t=sum(queries);qw=m.query_width//tp;kvw=m.kv_heads*m.head_dim//tp;h=m.hidden_size;f=m.ffn_size//tp
        norm=self.vec(t*h*6,t*h*2,t*h*2,'rmsnorm')
        qknorm=self.vec(t*(qw+kvw)*6,t*(qw+kvw)*2,t*(qw+kvw)*2,'qk_norm')
        rope=self.vec(t*(qw+kvw)*6,t*(qw+kvw)*2,t*(qw+kvw)*2,'rope')
        attn=norm+self.mm(t,qw+2*kvw,h)+qknorm+rope+self.attention(queries,contexts,tp)+self.mm(t,h,qw)
        ffn=norm+self.mm(t,2*f,h)+self.vec(t*f*5,t*f*4,t*f*2,'swiglu')+self.mm(t,h,f)
        residual=self.vec(t*h,t*h*4,t*h*2,'residual_add')
        return (Block('attention_out',attn,'ALL_REDUCE' if tp>1 else '',t*h*2),
                Block('attention_residual',residual),
                Block('ffn_out',ffn,'ALL_REDUCE' if tp>1 else '',t*h*2),
                Block('ffn_residual',residual))
    @lru_cache(maxsize=65536)
    def layer_backward(self,lengths,tp):
        m=self.model;t=sum(lengths);h=m.hidden_size;qw=m.query_width//tp;kvw=m.kv_heads*m.head_dim//tp;f=m.ffn_size//tp
        norm=self.vec(t*h*12,t*h*8,t*h*4,'rmsnorm_backward')
        ffn=self.linear_backward(t,f,h)+self.vec(t*f*10,t*f*10,t*f*8,'swiglu_backward')+self.linear_backward(t,h,2*f)
        attn=self.linear_backward(t,qw,h)+self.attention(lengths,lengths,tp,True)+self.vec(t*(qw+kvw)*18,t*(qw+kvw)*12,t*(qw+kvw)*8,'qk_norm_rope_backward')+self.linear_backward(t,h,qw+2*kvw)
        # Each TP rank has partial dX from column-parallel FFN and QKV projections.
        post=norm+self.vec(t*h,t*h*8,t*h*4,'residual_gradient_add')
        return (Block('ffn_dinput',ffn,'ALL_REDUCE' if tp>1 else '',t*h*4),Block('ffn_norm_residual_backward',post),
                Block('qkv_dinput',attn,'ALL_REDUCE' if tp>1 else '',t*h*4),Block('attention_norm_residual_backward',post))
    def forward(self,lengths,tp,mode='teacher',contexts=None):
        m=self.model;queries=tuple(lengths);contexts=tuple(contexts or lengths);t=sum(queries)
        embedding=self.vec(t*m.hidden_size,t*m.hidden_size*2,t*m.hidden_size*2,'embedding')
        blocks=[Block('embedding',embedding,'ALL_REDUCE' if tp>1 else '',t*m.hidden_size*2)]
        layer=self.layer_forward(queries,contexts,tp)
        for _ in range(m.layers):blocks.extend(layer)
        rows=len(lengths) if mode=='generation' else t
        head=self.vec(t*m.hidden_size*6,t*m.hidden_size*2,t*m.hidden_size*2,'final_norm')+self.mm(rows,m.vocab_size//tp,m.hidden_size)
        head+=self.vec(rows*m.vocab_size//tp*5,rows*m.vocab_size//tp*4,rows*m.vocab_size//tp*4,'logsoftmax_or_sampling')
        if mode=='generation':
            blocks.append(Block('head',head,'GATHER' if tp>1 else '',rows*(m.vocab_size//tp)*4))
            if tp>1:blocks.append(Block('sampled_token',Cost(),'BCAST',rows*4))
        else:
            blocks.append(Block('head',head,'ALL_REDUCE' if tp>1 else '',rows*4))
            if tp>1:blocks.append(Block('logsoftmax_sum',Cost(),'ALL_REDUCE',rows*4))
        return blocks
    def training(self,lengths,tp,target_positions=None):
        m=self.model;t=sum(lengths);blocks=self.forward(lengths,tp)
        head=self.linear_backward(t,m.hidden_size,m.vocab_size//tp)
        targets=t if target_positions is None else target_positions
        head+=self.vec(targets*25,targets*17,targets*4,'grpo_kl_loss')
        blocks.append(Block('head_dinput',head,'ALL_REDUCE' if tp>1 else '',t*m.hidden_size*4))
        params=enumerate_parameters(self.case_tp(tp))
        head_params=m.hidden_size*m.vocab_size//tp
        blocks.append(Block('head_gradient_accumulate',self.vec(head_params*2,head_params*8,head_params*4,'gradient_accumulate')))
        layer_params=sum(x.local_elements_per_die for x in params if x.parameter_id.startswith('layer0.'))
        blocks.append(Block('final_norm_backward',self.vec(t*m.hidden_size*12,t*m.hidden_size*8,t*m.hidden_size*4,'final_rmsnorm_backward')))
        forward=self.layer_forward(tuple(lengths),tuple(lengths),tp)
        backward=self.layer_backward(tuple(lengths),tp)
        for _ in range(m.layers):
            # Whole-layer activation checkpoint: re-execute forward, including TP reductions.
            blocks.extend(Block('checkpoint_'+x.name,x.cost,x.collective,x.payload) for x in forward)
            blocks.extend(backward)
            blocks.append(Block("layer_gradient_accumulate",self.vec(layer_params*2,layer_params*8,layer_params*4,"gradient_accumulate")))
        # Final norm and embedding gradients; tied head and embedding merge into one buffer.
        remaining=sum(x.local_elements_per_die for x in params)-head_params-m.layers*layer_params
        tail=self.vec(t*m.hidden_size*2,t*m.hidden_size*4,t*m.hidden_size*4,'embedding_backward')
        if remaining:tail+=self.vec(remaining*2,remaining*8,remaining*4,'gradient_accumulate')
        blocks.append(Block('embedding_final_norm_gradient',tail))
        return blocks
    @lru_cache(maxsize=128)
    def state(self,train_tp,rollout_tp):
        train=self.case_tp(train_tp);rows=enumerate_parameters(train);layout=layout_for_case(train,dp_size=48//train_tp)
        states=[layout.state_bytes(i)['total_allocated_bytes'] for i in range(48//train_tp)]
        roll=sum(x.local_elements_per_die for x in enumerate_parameters(self.case_tp(rollout_tp)))*2
        export=max(x.unique_elements*2 for x in rows if not x.replicated) if train_tp>1 else 0
        return {'training_base_per_rank':states,'rollout_weights_per_rank':roll,'export_scratch_per_rank':export,
                'layout':layout,'training_base_max':max(states)}
    def admission(self,trajectories,train_tp,rollout_tp,mb,logmb,cap):
        m=self.model;state=self.state(train_tp,rollout_tp);base=state['training_base_max'];reasons=[]
        kv_per_token=2*m.layers*(m.kv_heads//rollout_tp)*m.head_dim*2
        # Conservative upper-bound admission uses known prompt and response cap, never future R.
        max_prompt=max(self.prompt_cap,max(t['prompt_length'] for t in trajectories))
        gen=base+state['rollout_weights_per_rank']+cap*(max_prompt+8192)*kv_per_token
        # Export is per-tensor, before KV starts; distinct rollout buffers are allocated.
        export=base+state['rollout_weights_per_rank']+state['export_scratch_per_rank']
        # Training starts after generation: its actual lengths are known then.
        # Check the declared fixed-order microbatches, not an invented batch of
        # the longest requests from unrelated replicas.
        dp=48//train_tp;per=max(1,len(trajectories)//dp)
        def batch_tokens(size):
            return max(sum(t['prompt_length']+t['response_length']-1 for t in trajectories[g+j:g+min(j+size,per)])
                       for g in range(0,len(trajectories),per) for j in range(0,per,size))
        rows=enumerate_parameters(self.case_tp(train_tp))
        max_param=max(x.local_elements_per_die for x in rows)
        # Every-layer BF16 input checkpoints + one layer's FP32/BF16 working tensors.
        tokens=batch_tokens(mb);checkpoint=tokens*m.hidden_size*2*m.layers
        layer_work=tokens*(m.hidden_size*12+2*m.ffn_size//train_tp*6+(m.query_width+2*m.kv_heads*m.head_dim)//train_tp*6)
        head_work=tokens*(m.vocab_size//train_tp)*8
        train_peak=base+checkpoint+max(layer_work,head_work)+max_param*8
        logtokens=batch_tokens(logmb);log_peak=base+logtokens*(m.hidden_size*4+(m.vocab_size//train_tp)*8)
        # Existing collective staging reserves one transport buffer, endpoint reads/writes explicit.
        staging=max(4*max_param,4*tokens*m.hidden_size)
        peaks={'generation':gen+1024**2,'weight_export':export+4*max_param,'training':train_peak+staging,'logprob':log_peak+staging}
        for phase,value in peaks.items():
            if value>self.capacity:reasons.append(f'{phase}:{value}>{self.capacity}')
        return {'passed':not reasons,'peak_bytes':max(peaks.values()),'phase_peak_bytes':peaks,'reasons':reasons,
                'capacity_bytes':self.capacity,'admission':'conservative per-phase lifetime upper bounds; actual trace peaks validated in replay'}
