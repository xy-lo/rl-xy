"""Phase-lifetime memory sizing independent of the native execution backend."""
from .search_v2 import PowerOfTwoMemory

class LifecycleMemory(PowerOfTwoMemory):
    def __init__(self,op,record_reserve=200_000_000):
        super().__init__(op,record_reserve)
        from rl_xy.model import enumerate_parameters
        self.param_shapes={tp:(max(x.local_elements_per_die for x in enumerate_parameters(op.case_tp(tp))),sum(x.local_elements_per_die for x in enumerate_parameters(op.case_tp(tp)))*4) for tp in (2,4,8)}
        if self.staged:self.teacher_base+=max(0,self.shard*27-self.reference)
    def compact_gradient(self,tp):return self.grad2 if tp==2 else self.param_shapes[tp][1]
    def phase(self,stage,tp,batch):
        from .memory import PhaseAllocation
        m=self.op.model;tokens=sum(x['prompt_length']+x['response_length']-1 for x in batch)
        maxparam=self.param_shapes[tp][0]
        # Only activation-gradient TP traffic coexists with backward. Parameter
        # restore packing is later, after activations retire, and has its own
        # explicit pair of parameter-sized transport buffers.
        transport=4*tokens*m.hidden_size+2*1024**2+self.record_reserve+self.op.hw.nic_tx_bytes_per_die+self.op.hw.nic_rx_bytes_per_die
        if stage=='training':
            checkpoint=tokens*m.hidden_size*2*m.layers
            layer=tokens*(m.hidden_size*12+2*m.ffn_size//tp*6+(m.query_width+2*m.kv_heads*m.head_dim)//tp*6)
            head=tokens*(m.vocab_size//tp)*8
            activation=checkpoint+max(layer,head)+maxparam*8;gradient=self.compact_gradient(tp);base=self.training_base
        else:activation=tokens*(m.hidden_size*4+(m.vocab_size//tp)*8);gradient=0;base=self.teacher_base
        return PhaseAllocation(stage,base,gradient,activation,transport,base+gradient+activation+transport)


# Compatibility import for existing simulator adapters.
SystemMemory=LifecycleMemory
