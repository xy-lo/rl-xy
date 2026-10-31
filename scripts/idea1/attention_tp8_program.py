"""Experimental exact-shape decode program: TP8 attention, TP4 FFN.

All source and helper ranks participate in the attention all-reduce. Q/K/V
and output-projection shards use interleaved TP8 lane order (src0,dst0,...).
KV and attention weights must be transferred and admitted before execution.
This is a cost/event program, not an implementation of numerical tensors.
"""
from rl_xy.qwen3_cost import Block, Cost
from rl_xy.qwen3_sequence import compile_blocks, local


def attention_weight_bytes_per_helper(op):
    m = op.model
    if m.query_heads % 8 or m.kv_heads % 8:
        raise ValueError('TP8 head division not exact')
    qkv = m.hidden_size * (m.query_width + 2*m.kv_heads*m.head_dim)//8
    out = m.hidden_size * m.query_width//8
    # Conservatively copy the complete input RMS norm and replicated Q/K norm.
    norm = m.hidden_size + (2*m.head_dim if m.qk_norm else 0)
    bias = ((m.query_width + 2*m.kv_heads*m.head_dim)//8 if m.qkv_bias else 0)
    return m.layers*(qkv + out + norm + bias)*m.actor_dtype_bytes


def kv_bytes_per_helper(op, contexts):
    m = op.model
    if m.kv_heads % 8 or any(type(x) is not int or x <= 0 for x in contexts):
        raise ValueError('invalid TP8 KV geometry')
    return sum(contexts)*m.layers*2*(m.kv_heads//8)*m.head_dim*m.actor_dtype_bytes


def head_layout(op, src, dst):
    """Explicit logical TP4-to-TP8 Q/KV group ownership; not tensor storage."""
    m=op.model
    if m.query_heads % m.kv_heads or m.kv_heads % 8 or m.query_heads % 8:raise ValueError('invalid GQA head geometry')
    kv_per=m.kv_heads//8;queries_per_kv=m.query_heads//m.kv_heads
    lanes=[]
    for i in range(4):
        for side,rank,logical in (('source',src[i],2*i),('helper',dst[i],2*i+1)):
            heads=list(range(logical*kv_per,(logical+1)*kv_per))
            queries=[q for h in heads for q in range(h*queries_per_kv,(h+1)*queries_per_kv)]
            lanes.append({'rank':rank,'side':side,'logical_tp8_lane':logical,'kv_heads':heads,'query_heads':queries})
    assert sorted(h for x in lanes for h in x['kv_heads'])==list(range(m.kv_heads))
    assert sorted(q for x in lanes for q in x['query_heads'])==list(range(m.query_heads))
    return lanes

def migration(op, src, dst, contexts, *, tag_start):
    if len(src)!=4 or len(dst)!=4 or set(src)&set(dst): raise ValueError('two disjoint TP4 groups required')
    weights = attention_weight_bytes_per_helper(op)
    kv = kv_bytes_per_helper(op, contexts)
    ss, hs = [], []
    for i,(s,d) in enumerate(zip(src,dst)):
        for offset,(size,name) in enumerate(((weights,'attention_weight'),(kv,'attention_kv'))):
            msg={'kind':'p2p_direct','src':s,'dst':d,'bytes':size,'tag':tag_start+2*i+offset}
            if name=='attention_kv':
                # Streaming pack/unpack via bounded staging; this adds a full
                # extra read/write each side, not a single giant SRAM buffer.
                ss.append(local(op.vec(0,size,size,'kv_stream_pack'),ranks=[s]))
            ss.extend([local(op.vec(0,size,0,name+'_read'),ranks=[s]),msg])
            hs.extend([msg,local(op.vec(0,0,size,name+'_write'),ranks=[d])])
            if name=='attention_kv':
                hs.append(local(op.vec(0,size,size,'kv_stream_unpack'),ranks=[d]))
    return {'source':ss,'helper':hs,'weight_bytes_per_lane':weights,'kv_bytes_per_lane':kv,'head_layout':head_layout(op,src,dst),'pack_extra_read_bytes_per_lane':kv,'pack_extra_write_bytes_per_lane':kv}


def compile_attention_tp8(op, src, dst, contexts, *, group_id, tag_start):
    contexts=tuple(contexts); batch=len(contexts)
    if not batch or len(src)!=4 or len(dst)!=4 or set(src)&set(dst): raise ValueError('invalid groups')
    if op.model.query_heads%8 or op.model.kv_heads%8: raise ValueError('TP8 incompatible')
    original=op.forward((1,)*batch,4,'generation',contexts)
    original_next=op.forward((1,)*batch,4,'generation',tuple(x+1 for x in contexts))
    wider=op.forward((1,)*batch,8,'generation',contexts)
    wider_next=op.forward((1,)*batch,8,'generation',tuple(x+1 for x in contexts))
    ranks8=tuple(x for pair in zip(src,dst) for x in pair)
    h=op.model.hidden_size; inp=batch*h*op.model.actor_dtype_bytes
    ss,hs=[],[]; layer=0
    for old,old_next,new,new_next in zip(original,original_next,wider,wider_next):
        if old.name != new.name or old.name != old_next.name or new.name != new_next.name:
            raise ValueError('layer/block structure differs')
        if old.name!='attention_out':
            ss.extend(compile_blocks(op,src,[old],[old_next])); continue
        ss.append(local(op.vec(0,inp,0,'attn_helper_input_read')))
        for i,(s,d) in enumerate(zip(src,dst)):
            msg={'kind':'p2p_direct','src':s,'dst':d,'bytes':inp,'tag':tag_start+layer*8+i}
            ss.append(msg); hs.append(msg)
        hs.append(local(op.vec(0,0,inp,'attn_helper_input_write')))
        delta=Cost(new_next.cost.compute-new.cost.compute,new_next.cost.memory-new.cost.memory)
        ss.append(local(new.cost,delta)); hs.append(local(new.cost,delta))
        # Conservatively materialize the newly generated K/V in each lane's
        # resident store; adds a write beyond the existing QKV output service.
        append_bytes=batch*2*(op.model.kv_heads//8)*op.model.head_dim*op.model.actor_dtype_bytes
        ss.append(local(op.vec(0,0,append_bytes,'kv_persistent_append')))
        hs.append(local(op.vec(0,0,append_bytes,'kv_persistent_append')))
        ar=compile_blocks(op,ranks8,[Block('attention_tp8_collective',Cost(),new.collective,new.payload)])
        for x in ar:
            if x['kind']=='AR': x['group']=group_id
        ss.extend(ar); hs.extend(ar); layer+=1
    if layer != op.model.layers: raise ValueError('attention layer coverage')
    return {'source':ss,'helper':hs,'layers':layer,'input_bytes_per_step':4*layer*inp,
            'ranks8':ranks8,'batch':batch}
