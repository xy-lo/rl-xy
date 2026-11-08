"""Immutable BF16 TP2 actor views shared by inference and regional training.

The native model starts with resident TP2 actor shards, as do the frozen
baselines. No second rollout materialization is needed for an identical TP2
view. Gradient arenas are absent during generation. Adam master/m/v remain on
the original owners; reference storage is independent and protected.
"""
from dataclasses import asdict
from .coordinates import ParameterCoordinates
from .history import digest
from .lifecycle_memory import SystemMemory

class ActorViewCatalog:
    def __init__(self,op,pair_mapping=None):
        self.schema=ParameterCoordinates(op.model);self.version='actor_v0';self.generation=0
        self.views={h:[asdict(v) for pid in sorted({x.parameter_id for x in self.schema.components}) for v in self.schema.views(pid,2,h)] for h in (0,1)}
        lanes=({d:d%2 for d in range(54)} if pair_mapping is None else
               {die:lane for pair in pair_mapping.pairs for lane,die in enumerate(pair)})
        self.allocations={d:dict(actor_allocation=f'actor-die-{d}',training_actor=f'actor-die-{d}',inference_actor=f'actor-die-{d}',
            dtype='BF16',logical_tp_rank=lane,version=self.version,allocation_generation=0,shared_read_only=True) for d,lane in lanes.items()}
    def validate(self):
        return all(v['training_actor']==v['inference_actor'] and v['dtype']=='BF16' and v['shared_read_only'] for v in self.allocations.values())
    def publish_after_native_update(self):
        assert self.version=='actor_v0' and self.validate();self.version='actor_v1';self.generation+=1
        for v in self.allocations.values():v.update(version=self.version,allocation_generation=self.generation)
    def manifest(self):
        return dict(schema_digest=self.schema.schema_digest,allocations=self.allocations,logical_views_by_tp_rank=self.views,
            version=self.version,generation=self.generation,proof='same allocation identity, dtype and canonical TP2 coordinate view; writes only after global old-reader drain')

class ViewAdmission:
    def __init__(self,op,record_reserve=200_000_000):self.op=op;self.memory=SystemMemory(op,record_reserve);self.reserve=record_reserve
    def evaluate(self,cap,public_groups):
        op=self.op;m=op.model;mem=self.memory;maxp=max(op.prompt_cap,max(g['prompt_length'] for g in public_groups))
        # Initial actor+reference+owner, with no gradient or duplicate rollout
        # allocation. Staged models also retain their persistent ref shard.
        base=mem.teacher_base
        kv=2*m.layers*(m.kv_heads//2)*m.head_dim*m.actor_dtype_bytes
        scratch=cap*(m.hidden_size*32+m.ffn_size*16)+1024**2
        nic=op.hw.nic_tx_bytes_per_die+op.hw.nic_rx_bytes_per_die
        generation=base+cap*(maxp+8192)*kv+scratch+nic+self.reserve
        # During refill the vacated slots contain only new prompt KV. A linear
        # envelope reaches its maximum at all-decode or all-prefill, so checking
        # both endpoints bounds mixed old-decode/new-prefill occupancy.
        tokens=cap*maxp
        layer=tokens*(m.hidden_size*12+2*m.ffn_size//2*6+(m.query_width+2*m.kv_heads*m.head_dim)//2*6)
        head=cap*(m.vocab_size//2)*8
        prefill=base+cap*maxp*kv+max(layer,head)+4*tokens*m.hidden_size+scratch+nic+self.reserve
        # Helper attention weights are local immutable subviews. The required
        # KV half still moves and remains live through the explicit lease.
        helper=base+cap*(maxp+8192)*(kv//2)+scratch+nic+self.reserve
        from rl_xy.model import enumerate_parameters
        maxparam=max(x.local_elements_per_die for x in enumerate_parameters(op.case_tp(2)))
        restore=mem.restore_base+nic+self.reserve+2*1024**2+maxparam*8
        minimum_tp=getattr(op,'rlxy_min_training_tp',2)
        single_train=mem.hard_bounds(minimum_tp,1,'training',maxp).peak_bytes;single_teacher=mem.hard_bounds(minimum_tp,1,'teacher',maxp).peak_bytes
        peaks=dict(generation=generation,prefill=prefill,helper=helper,gradient_restore=restore,teacher=single_teacher,training=single_train,weight_export=base+self.reserve)
        good=max(peaks.values())<=op.capacity
        return dict(passed=good,capacity_bytes=op.capacity,phase_peak_bytes=peaks,rollout_cap=cap,record_and_staging_bound_bytes=self.reserve,
            actor_storage='shared immutable BF16 TP2 allocation; regional TP uses canonical local subviews',gradient_storage='absent during generation, compact TPp during backward, TP2 only after activation release',
            reference_storage='protected independent full copy plus persistent shard' if mem.staged else 'protected independent full copy',
            prefill_capacity='max of all-decode and all-prefill linear envelopes; no future response lengths',adaptive_phase_upper_bound_bytes=op.capacity,
            fallback_microbatch=1,minimum_training_tp=minimum_tp,limitations=['formula/live-allocation simulation, not physical tensor allocation or hardware calibration'])
    def candidates(self,public_groups,values=(16,24,32,40,48,56,64)):
        return [self.evaluate(c,public_groups) for c in values]
