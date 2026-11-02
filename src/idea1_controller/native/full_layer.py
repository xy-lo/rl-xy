"""Extend the admitted attention lease to split FFN work as well.

Matching TP2 replicas already contain the same actor FFN shards. Each exposes
one half of its intermediate channels, forming logical TP4 views. Attention
KV/weight migration remains exactly the charged legacy protocol. Both sides
materialize the residual and use native TP4 SUM for FFN; sampling/head stay TP2.
"""
from rl_xy.qwen3_cost import Block,Cost
from rl_xy.qwen3_sequence import compile_blocks,local
from attention_tp8_program import attention_weight_bytes_per_helper,kv_bytes_per_helper
from .pending import PendingMixin
from .adapter import NativeKernel
from bisect import bisect_right


def compile_full_layer(op,src,dst,contexts,*,group_id,tag_start):
    contexts=tuple(contexts);batch=len(contexts)
    if not batch or len(src)!=2 or len(dst)!=2 or set(src)&set(dst):raise ValueError('invalid admitted TP2+TP2 lease')
    original=op.forward((1,)*batch,2,'generation',contexts)
    original_next=op.forward((1,)*batch,2,'generation',tuple(x+1 for x in contexts))
    wider=op.forward((1,)*batch,4,'generation',contexts)
    wider_next=op.forward((1,)*batch,4,'generation',tuple(x+1 for x in contexts))
    ranks4=tuple(x for pair in zip(src,dst) for x in pair);h=op.model.hidden_size;inp=batch*h*2;ss=[];hs=[];layer=0;ffns=0
    for old,on,new,nn in zip(original,original_next,wider,wider_next):
        if old.name!=new.name or old.name!=on.name or new.name!=nn.name:raise ValueError('layer structure differs')
        if old.name=='attention_out':
            ss.append(local(op.vec(0,inp,0,'attn_helper_input_read')))
            for i,(s,d) in enumerate(zip(src,dst)):
                msg=dict(kind='p2p_direct',src=s,dst=d,bytes=inp,tag=tag_start+layer*8+i);ss.append(msg);hs.append(msg)
            hs.append(local(op.vec(0,0,inp,'attn_helper_input_write')))
            delta=Cost(nn.cost.compute-new.cost.compute,nn.cost.memory-new.cost.memory)
            ss.append(local(new.cost,delta));hs.append(local(new.cost,delta))
            append=batch*2*(op.model.kv_heads//4)*op.model.head_dim*2
            ss.append(local(op.vec(0,0,append,'kv_persistent_append')));hs.append(local(op.vec(0,0,append,'kv_persistent_append')))
            ar=compile_blocks(op,ranks4,[Block('attention_tp4_collective',Cost(),new.collective,new.payload)])
            for x in ar:
                if x['kind']=='AR':x['group']=group_id
            ss.extend(ar);hs.extend(ar);layer+=1
        elif old.name=='ffn_out':
            steps=compile_blocks(op,ranks4,[new],[nn])
            for x in steps:
                if x['kind']=='AR':x['group']=group_id
            ss.extend(steps);hs.extend(steps);ffns+=1
        elif old.name in ('attention_residual','ffn_residual'):
            steps=compile_blocks(op,src,[old],[on]);ss.extend(steps);hs.extend(steps)
        else:ss.extend(compile_blocks(op,src,[old],[on]))
    if layer!=op.model.layers or ffns!=layer:raise ValueError('incomplete full-layer helper')
    return dict(source=ss,helper=hs,layers=layer,input_bytes_per_step=2*layer*inp,ranks8=ranks4,batch=batch)


class FullLayerAttention:
    def estimate_locality(self,w,helper):
        from locality_runtime_v3 import AttentionRuntime
        estimate=AttentionRuntime.estimate_locality(self,w,helper)
        if estimate is None:return None
        rep=self.rep[w];batch=len(rep.active);q,e=divmod(sum(x[1] for x in rep.active),batch);ctx=tuple(q+(i<e) for i in range(batch))
        old=self.op.forward((1,)*batch,2,'generation',ctx);new=self.op.forward((1,)*batch,4,'generation',ctx)
        extra=sum(a.cost.ns-b.cost.ns for a,b in zip(old,new) if a.name=='ffn_out')
        activation=batch*self.op.model.hidden_size*2;bw=self.op.hw.d2d_bytes_per_ns_per_direction;lat=self.op.hw.d2d_latency_ns
        ar=self.op.model.layers*(3*lat+3*activation/(4*bw))
        estimate['predicted_local_gain_ns_per_step']+=extra;estimate['predicted_network_ns_per_step']+=ar
        estimate['predicted_net_ns']+=estimate['predicted_steps']*(extra-ar)
        estimate['accepted']=estimate['predicted_net_ns']>max(1000,estimate['predicted_migration_ns']*.05)
        estimate['helper_kernel']='attention-and-FFN-TP4';return estimate

    def _prepare_segment(self,w,it,tx,*,already_pending=False):
        r=self.rep[w];chosen=r.active[:r.decode_batch_limit] if r.decode_batch_limit else r.active
        if not chosen or r.requests:raise ValueError('lease expects active decode with no queued prefill')
        batch=len(chosen);q,extra=divmod(sum(x[1] for x in chosen),batch)
        ctx=tuple(q+(j<extra) for j in range(batch))
        if already_pending:
            repeat=tx['initial_repeat']
            if r.pending is None or r.pending[0]!='decode':raise ValueError('initial decode pending absent')
        else:
            if r.pending is not None:raise ValueError('decode already pending')
            r.safe_interval=self.interval;repeat=r.next(it)['repeat']
        self.segment_counter+=1
        tag=1000000+self.segment_counter*1024
        if tag+self.op.model.layers*8>=199000000:raise ValueError('attention tag space exhausted')
        compiled=compile_full_layer(self.op,r.ranks,self.rep[tx['helper']].ranks,ctx,
            group_id=tx['group_id'],tag_start=tag)
        tx.update(stage='running',ready=True,helper_started=False,repeat=repeat,
                  source_steps=compiled['source'],helper_steps=compiled['helper'])
        self.runs.append({'source':w,'helper':tx['helper'],'repeat':repeat,'batch':batch,
                          'contexts':ctx,'tx':tx['id'],'segment':self.segment_counter,
                          'active_ids':[x[0]['trajectory_id'] for x in chosen],
                          'actual_contexts':[x[1] for x in chosen],
                          'kv_append_bytes_per_rank_per_step':batch*2*(self.op.model.kv_heads//4)*self.op.model.head_dim*self.op.model.actor_dtype_bytes*self.op.model.layers})
        prefix=[local(Cost(compute=1000))]
        if self.waiting[tx['helper']]:prefix.append(self.wake(tx['helper'],r.ranks[0]))
        return {'steps':prefix+[{'kind':'controller_rpc','event':{'worker':w,'phase':'attention_source'}}]}



class FullLayerKernel(PendingMixin,FullLayerAttention,NativeKernel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.init_pending()
