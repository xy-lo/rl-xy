#!/usr/bin/env python3
"""Real-trace sensitivity experiments; ideal GQA is an optimistic bound only."""
import argparse, functools, json, sys, types
from collections import deque
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['queue','ideal-gqa'],required=True)
p.add_argument('--cap',type=int,default=64)
known, rest=p.parse_known_args()
sys.argv=[sys.argv[0], "--cap", str(known.cap)]+rest
import locality_full4b_v3 as base
from rl_xy.f2_runtime import INPUTS

if known.mode=='queue':
    # One global FIFO, all requests initially available. This intentionally omits
    # prompt distribution and is an optimistic screening, not a complete system.
    trace=json.loads((INPUTS/'Qwen3-4B'/'test1.json').read_text())['trajectories']
    shared=deque(x for i,x in enumerate(trace) if i%12<4)
    Original=base.Replica
    def factory(op, requests, ranks, cap, **kwargs):
        r=Original(op,[],ranks,known.cap,**kwargs)
        r.requests=shared
        return r
    base.Replica=factory
else:
    original_make=base.make_operator
    def make(*args,**kwargs):
        op=original_make(*args,**kwargs)
        from timeline_provider import attention_memory_ns, attention_geometry
        original=op.attention
        @functools.lru_cache(maxsize=65536)
        def ideal(self,queries,contexts,tp,backward=False):
            old=original(queries,contexts,tp,backward)
            if backward or max(queries)!=1:return old
            heads=self.model.query_heads//tp
            kvheads=self.model.kv_heads//tp
            # Preserves arithmetic and query traffic. Each K/V element is read
            # once per KV head; assumes free same-core GQA reuse, not measured.
            saving=2*sum(contexts)*(heads-kvheads)*self.model.head_dim*2
            reads=old.read-saving
            assert 0 <= reads <= old.read
            # Conservative materialization via DRAM: reorder Q/K/V into the
            # grouped-core layout and reorder attention output for O projection.
            move_bytes=sum(queries)*(2*(self.model.query_width//tp)+2*(self.model.kv_heads*self.model.head_dim//tp))*self.model.actor_dtype_bytes
            reads+=move_bytes
            writes=old.write+move_bytes
            old_units=attention_geometry(self,queries,contexts,tp,backward)
            units=min(self.units,len(queries)*kvheads)
            # Co-locate one KV head and its query heads in each private core;
            # conservative fourfold per-core workspace and reduced parallelism.
            workspace_per_core=(old.workspace + old_units - 1)//old_units * (heads//kvheads)
            if workspace_per_core > self.hw.sram_bytes_per_core:
                raise ValueError('grouped GQA core SRAM capacity exceeded')
            from dataclasses import replace
            memory=attention_memory_ns(reads,writes,units,self.hw,
                         load_bytes_per_ns=self.provider.load,store_bytes_per_ns=self.provider.store)
            return replace(old,read=int(reads),write=int(writes),memory=memory,compute=old.compute*old_units/units,workspace=workspace_per_core*units)
        op.attention=types.MethodType(ideal,op)
        return op
    base.make_operator=make
base.main()
