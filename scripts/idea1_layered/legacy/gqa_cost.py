"""Original Idea1 generation-only GQA wrapper, extracted without formula changes.

Canonical source: outputs/idea1-t106/14B/idea-code/locality_gqa_4b_v3.py
The optimistic grouped-core reuse assumption is inherited, not calibrated.
"""
import functools
import types


def apply_generation_gqa(op):
    from timeline_provider import attention_memory_ns, attention_geometry
    original = op.attention
    @functools.lru_cache(maxsize=65536)
    def ideal(self, queries, contexts, tp, backward=False):
        old = original(queries, contexts, tp, backward)
        if backward or max(queries) != 1:
            return old
        heads = self.model.query_heads//tp
        kvheads = self.model.kv_heads//tp
        saving = 2*sum(contexts)*(heads-kvheads)*self.model.head_dim*2
        reads = old.read-saving
        assert 0 <= reads <= old.read
        move_bytes = sum(queries)*(2*(self.model.query_width//tp)+2*(self.model.kv_heads*self.model.head_dim//tp))*self.model.actor_dtype_bytes
        reads += move_bytes
        writes = old.write+move_bytes
        old_units = attention_geometry(self, queries, contexts, tp, backward)
        units = min(self.units, len(queries)*kvheads)
        workspace_per_core = (old.workspace+old_units-1)//old_units*(heads//kvheads)
        if workspace_per_core > self.hw.sram_bytes_per_core:
            raise ValueError('grouped GQA core SRAM capacity exceeded')
        from dataclasses import replace
        memory = attention_memory_ns(reads, writes, units, self.hw,
                                     load_bytes_per_ns=self.provider.load, store_bytes_per_ns=self.provider.store)
        return replace(old, read=int(reads), write=int(writes), memory=memory,
                       compute=old.compute*old_units/units, workspace=workspace_per_core*units)
    op.attention = types.MethodType(ideal, op)
    return op
