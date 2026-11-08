"""Phase/lifetime capacity for compact regional gradients and adaptive batches.

Reference/actor/optimizer layouts retain the frozen physical TP2 ownership.
The gradient arena is compact while activations live, then grows in place after
all activation readers retire. Non-root TP2 gradients are created as zeros only
after their compact shards have been delivered. No optimizer state is dropped.
"""
from dataclasses import dataclass
from rl_xy.model import enumerate_parameters

@dataclass(frozen=True)
class PhaseAllocation:
    phase: str
    persistent_bytes: int
    gradient_bytes: int
    activation_bytes: int
    transport_bytes: int
    peak_bytes: int

class RegionalMemory:
    def __init__(self,op,record_reserve=200_000_000):
        self.op=op;self.state=op.state(2,2);self.record_reserve=record_reserve
        self.alloc=self.state['layout'].state_bytes(0)['allocations']
        self.actor=self.alloc['actor_BF16_parameter_buffer'];self.grad2=self.alloc['full_FP32_gradient_working_buffer']
        self.reference=self.alloc['reference_BF16_valid_parameters'];self.owner=max(sum(self.alloc[k] for k in ('master_FP32_shard','adam_m_FP32_shard','adam_v_FP32_shard')),self.state['training_base_max']-self.actor-self.grad2-self.reference)
        self.staged='persistent_reference_shard_bytes' in self.state
        self.shard=self.state.get('persistent_reference_shard_bytes',0)
        # Working gradient storage is released before teacher in all models.
        self.teacher_base=self.actor+self.owner+self.reference+(self.shard if self.staged else 0)
        self.training_base=self.actor+self.owner+(self.shard if self.staged else self.reference)
        self.restore_base=self.training_base+self.grad2
    def compact_gradient(self,tp):
        return sum(x.local_elements_per_die for x in enumerate_parameters(self.op.case_tp(tp)))*4
    def phase(self,stage,tp,batch):
        m=self.op.model;tokens=sum(x['prompt_length']+x['response_length']-1 for x in batch)
        rows=enumerate_parameters(self.op.case_tp(tp));maxparam=max(x.local_elements_per_die for x in rows)
        staging=max(4*maxparam,4*tokens*m.hidden_size)+2*1024**2+self.record_reserve
        if stage=='training':
            checkpoint=tokens*m.hidden_size*2*m.layers
            layer=tokens*(m.hidden_size*12+2*m.ffn_size//tp*6+(m.query_width+2*m.kv_heads*m.head_dim)//tp*6)
            head=tokens*(m.vocab_size//tp)*8
            activation=checkpoint+max(layer,head)+maxparam*8
            gradient=self.compact_gradient(tp);base=self.training_base
        else:
            activation=tokens*(m.hidden_size*4+(m.vocab_size//tp)*8);gradient=0;base=self.teacher_base
        return PhaseAllocation(stage,base,gradient,activation,staging,base+gradient+activation+staging)
    def batches(self,items,stage,tp,maximum):
        cursor=0
        while cursor<len(items):
            n=min(maximum,len(items)-cursor)
            while self.phase(stage,tp,items[cursor:cursor+n]).peak_bytes>self.op.capacity:
                if n==1:raise ValueError('observed single-sample allocation exceeds per-die DRAM')
                n=max(1,n//2)
            batch=items[cursor:cursor+n];yield batch;cursor+=n
    def hard_bounds(self,tp,maximum,stage,prompt_cap):
        batch=[dict(prompt_length=prompt_cap,response_length=8192) for _ in range(maximum)]
        return self.phase(stage,tp,batch)
