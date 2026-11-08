"""Experimental owner-TP memory contract; not wired into native execution."""
from rl_xy.model import enumerate_parameters
from idea1_controller.lifecycle_memory import LifecycleMemory
class WideOwnerMemory(LifecycleMemory):
    def __init__(self,op,owner_tp,record_reserve=200_000_000):
        if owner_tp not in (2,4,8,16) or owner_tp>op.model.kv_heads: raise ValueError('unsupported owner TP or KV replication')
        if op.active_dies%owner_tp: raise ValueError('active dies must form complete owners')
        self.op=op;self.owner_tp=owner_tp;self.record_reserve=record_reserve
        self.state=op.state(owner_tp,owner_tp);self.alloc=self.state['layout'].state_bytes(0)['allocations'];a=self.alloc
        self.actor=a['actor_BF16_parameter_buffer'];self.grad2=a['full_FP32_gradient_working_buffer'];self.reference=a['reference_BF16_valid_parameters']
        self.owner=max(sum(a[k] for k in ('master_FP32_shard','adam_m_FP32_shard','adam_v_FP32_shard')),self.state['training_base_max']-self.actor-self.grad2-self.reference)
        self.staged='persistent_reference_shard_bytes' in self.state;self.shard=self.state.get('persistent_reference_shard_bytes',0)
        self.teacher_base=self.actor+self.owner+self.reference+(self.shard if self.staged else 0);self.training_base=self.actor+self.owner+(self.shard if self.staged else self.reference);self.restore_base=self.training_base+self.grad2
        self.param_shapes={tp:(max(x.local_elements_per_die for x in enumerate_parameters(op.case_tp(tp))),sum(x.local_elements_per_die for x in enumerate_parameters(op.case_tp(tp)))*4) for tp in (2,4,8,16) if owner_tp<=tp<=op.model.kv_heads}
        if self.staged:self.teacher_base+=max(0,self.shard*(op.active_dies//owner_tp)-self.reference)
    def compact_gradient(self,tp):
        if tp not in self.param_shapes or tp%self.owner_tp:raise ValueError('invalid training TP for owner')
        return self.grad2 if tp==self.owner_tp else self.param_shapes[tp][1]
    def phase(self,stage,tp,batch):
        if stage not in ('teacher','training') or tp not in self.param_shapes or tp%self.owner_tp:raise ValueError('invalid phase or training TP for owner')

        alloc=super().phase(stage,tp,batch)
        if stage=='training':
            from dataclasses import replace
            tokens=sum(x['prompt_length']+x['response_length']-1 for x in batch)
            saved=tokens*self.op.model.hidden_size*2*(self.op.model.layers-16)
            alloc=replace(alloc,activation_bytes=alloc.activation_bytes-saved,peak_bytes=alloc.peak_bytes-saved)
        return alloc
