"""Canonical logical parameter coordinates for uniform TP2/4/8 native views.

Q/K/V and gate/up/down remain distinct components even when the cost model
stores aggregate parameter rows. Row-parallel matrices retain strided columns;
a flat interval is never substituted for their physical coordinate set.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math,re
from .history import digest

@dataclass(frozen=True)
class ParameterComponent:
    parameter_id: str
    component_id: str
    shape: tuple
    shard_axis: int | None
    gradient_semantics: str

@dataclass(frozen=True)
class CoordinateView:
    component_id: str
    shape: tuple
    axis: int | None
    start: int
    end: int
    elements: int
    gradient_semantics: str

class ParameterCoordinates:
    def __init__(self,model):
        self.model=model;h=model.hidden_size;f=model.ffn_size;q=model.query_width;k=model.kv_heads*model.head_dim
        self.components=[]
        def add(pid,cid,shape,axis,kind='SHARDED'):
            self.components.append(ParameterComponent(pid,pid+'/'+cid,tuple(shape),axis,kind))
        add('token_embedding','weight',(model.vocab_size,h),0)
        for layer in range(model.layers):
            p=f'layer{layer}'
            for name,n in [('q',q),('k',k),('v',k)]:add(p+'.qkv',name,(n,h),0)
            add(p+'.o_proj','weight',(h,q),1)
            for name,shape,axis in [('gate',(f,h),0),('up',(f,h),0),('down',(h,f),1)]:add(p+'.ffn',name,shape,axis)
            add(p+'.rmsnorm','attention_and_ffn',(2,h),None,'REPLICATED_EQUAL')
            if model.qk_norm:add(p+'.qk_norm','q_and_k',(2,model.head_dim),None,'REPLICATED_PARTIAL')
            if model.qkv_bias:
                for name,n in [('q',q),('k',k),('v',k)]:add(p+'.qkv_bias',name,(n,),0)
        add('final_rmsnorm','weight',(h,),None,'REPLICATED_EQUAL')
        if not model.tie_word_embeddings:add('lm_head','weight',(model.vocab_size,h),0)
        self.schema_digest=digest([asdict(x) for x in self.components])
    def views(self,parameter_id,tp,rank):
        if tp not in (1,2,4,8) or not 0<=rank<tp:raise ValueError('unsupported uniform TP coordinate view')
        out=[]
        for x in self.components:
            if x.parameter_id!=parameter_id:continue
            if x.shard_axis is None:start,end=0,1;elements=math.prod(x.shape)
            else:
                width=x.shape[x.shard_axis]
                if width%tp:raise ValueError('nonuniform component requires an unavailable kernel')
                start=rank*(width//tp);end=start+width//tp;elements=math.prod(x.shape)//tp
            out.append(CoordinateView(x.component_id,x.shape,x.shard_axis,start,end,elements,x.gradient_semantics))
        if not out:raise ValueError('unknown canonical parameter')
        return tuple(out)
    def restore_descriptor(self,parameter_id,tp,old_lane,subshard):
        if tp not in (2,4,8):raise ValueError('training gradients retain TP2/4/8 contract')
        m=tp//2;rank=old_lane*m+subshard
        source=self.views(parameter_id,tp,rank);target=self.views(parameter_id,2,old_lane)
        for s,t in zip(source,target):
            assert s.component_id==t.component_id and s.axis==t.axis
            if s.axis is not None:assert t.start<=s.start<s.end<=t.end
        return dict(schema_digest=self.schema_digest,source=[asdict(v) for v in source],target=[asdict(v) for v in target],
            payload_elements=sum(v.elements for v in source),local_expansion_order='reverse parameter/component order; reverse rows for strided columns')
