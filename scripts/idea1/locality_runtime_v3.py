"""Generation-only persistent TP8-attention lease with TP4 FFN.

Uses the existing TP4 FIFO replicas and observable history. The helper first
receives one TP8 attention-weight shard and the active half-KV shard; both
groups synchronize before decode. The lease persists until source completion,
so a growing KV cache is not retransmitted on each scheduling interval.
"""
from rl_xy.qwen3_cost import Cost
from rl_xy.qwen3_sequence import local
from generation_reclaim_priority_v4 import Reclaimer
from reclaim_priority import score as priority_score
from attention_tp8_program import migration,compile_attention_tp8,attention_weight_bytes_per_helper,kv_bytes_per_helper
from bisect import bisect_right


class AttentionRuntime(Reclaimer):
    def __init__(self,*args,group_ids,**kwargs):
        super().__init__(*args,**kwargs)
        self.group_ids=group_ids;self.migrations=[];self.releases=[];self.segment_counter=0;self.locality_decisions=[]

    def estimate_locality(self,w,helper):
        # Online heuristic. It cannot inspect the live response_length field or
        # active tuple's remaining length; completed history is right-censored.
        rep=self.rep[w]; batch=len(rep.active)
        if not batch:return None
        ordered=sorted(self.completed_history); residual=[]
        for item,context,_ in rep.active:
            elapsed=context-item['prompt_length'];i=bisect_right(ordered,elapsed);n=len(ordered)-i
            residual.append(max(1,ordered[i+(9*(n-1))//10]-elapsed) if n>=8 else max(64,min(2048,elapsed//4)))
        horizon=max(residual);q,extra=divmod(sum(x[1] for x in rep.active),batch)
        contexts=tuple(q+(j<extra) for j in range(batch))
        old=self.op.forward((1,)*batch,4,'generation',contexts)
        new=self.op.forward((1,)*batch,8,'generation',contexts)
        local_gain=sum(a.cost.ns-b.cost.ns for a,b in zip(old,new) if a.name=='attention_out')
        src=rep.ranks;dst=self.rep[helper].ranks
        hops=max(abs(s//6-d//6)+abs(s%6-d%6) for s,d in zip(src,dst))
        bw=self.op.hw.d2d_bytes_per_ns_per_direction
        lat=self.op.hw.d2d_latency_ns
        weights=attention_weight_bytes_per_helper(self.op);kv=kv_bytes_per_helper(self.op,contexts)
        transfer=(weights+kv)/bw*max(1,hops)+2*lat*max(1,hops)
        transfer+=self.op.vec(0,weights+kv,0,'estimate_migration_read').ns+self.op.vec(0,0,weights+kv,'estimate_migration_write').ns
        transfer+=2*self.op.vec(0,kv,kv,'estimate_stream_pack_unpack').ns
        activation=batch*self.op.model.hidden_size*self.op.model.actor_dtype_bytes
        append_bytes=batch*2*(self.op.model.kv_heads//8)*self.op.model.head_dim*self.op.model.actor_dtype_bytes
        network_step=self.op.model.layers*(activation/bw*max(1,hops)+lat*max(1,hops)+self.op.vec(0,0,append_bytes,'estimate_persistent_kv_append').ns)
        # Additional 8-rank collective cost approximated using ring service;
        # actual traffic and contention are evaluated in the native replay.
        collective_step=self.op.model.layers*(7*lat+7*activation/(8*bw))
        benefit=horizon*(local_gain-network_step-collective_step)-transfer
        result={'source':w,'helper':helper,'hops':hops,'batch':batch,'context_sum':sum(contexts),
                'history_samples':len(ordered),'predicted_steps':horizon,'predicted_local_gain_ns_per_step':local_gain,
                'predicted_network_ns_per_step':network_step+collective_step,'predicted_migration_ns':transfer,
                'predicted_net_ns':benefit,'accepted':benefit>max(1000,transfer*0.05)}
        return result

    def _observe(self,w):
        r=self.rep[w];r.observe()
        for tid in r.done[self.cursor[w]:]:
            if tid in self.done or tid not in self.ids:raise ValueError('duplicate or unknown completion')
            self.done.add(tid);rec=self.items[tid]
            if rec.get('finish_reason') in ('stop','length'):self.completed_history.append(rec['response_length'])
        self.cursor[w]=len(r.done)

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
        if tag+op_layers_bound(self.op)>=199000000:raise ValueError('attention tag space exhausted')
        compiled=compile_attention_tp8(self.op,r.ranks,self.rep[tx['helper']].ranks,ctx,
            group_id=tx['group_id'],tag_start=tag)
        tx.update(stage='running',ready=True,helper_started=False,repeat=repeat,
                  source_steps=compiled['source'],helper_steps=compiled['helper'])
        self.runs.append({'source':w,'helper':tx['helper'],'repeat':repeat,'batch':batch,
                          'contexts':ctx,'tx':tx['id'],'segment':self.segment_counter,
                          'active_ids':[x[0]['trajectory_id'] for x in chosen],
                          'actual_contexts':[x[1] for x in chosen],
                          'kv_append_bytes_per_rank_per_step':batch*2*(self.op.model.kv_heads//8)*self.op.model.head_dim*self.op.model.actor_dtype_bytes*self.op.model.layers})
        prefix=[local(Cost(compute=1000))]
        if self.waiting[tx['helper']]:prefix.append(self.wake(tx['helper'],r.ranks[0]))
        return {'steps':prefix+[{'kind':'controller_rpc','event':{'worker':w,'phase':'attention_source'}}]}

    def step(self,w,it):
        if not self.enabled:return super().step(w,it)
        r=self.rep[w];self._observe(w)
        tx=next((v for v in self.tx.values() if w in (v['source'],v['helper'])),None)
        if tx and tx['helper']==w:
            if tx['stage']=='migrating' and not tx['helper_migration_started']:
                tx['helper_migration_started']=True;key=f'attn-migrate-{tx["id"]}'
                return {'steps':tx['migration_helper']+[{'kind':'gate_signal','key':key},
                    {'kind':'gate_wait','key':key,'threshold':8},
                    {'kind':'controller_rpc','event':{'worker':w,'phase':'attention_helper_ready'}}]}
            if tx['stage']=='running' and tx['ready'] and not tx['helper_started']:
                tx['helper_started']=True;return {'steps':tx['helper_steps'],'repeat':tx['repeat']}
            return self.wait(w)
        if tx and tx['source']==w:
            if tx['stage']!='running' or not tx['ready']:
                raise ValueError('source returned with invalid lease state')
            tx['ready']=False
            if r.active:
                return self._prepare_segment(w,it,tx)
            if r.requests or r.pending is not None:raise ValueError('lease ended with unfinished local work')
            del self.tx[w];self.releases.append({'source':w,'helper':tx['helper'],'tx':tx['id']})
            prefix=[local(Cost(compute=1000))]
            if self.waiting[tx['helper']]:prefix.append(self.wake(tx['helper'],r.ranks[0]))
            if len(self.done)==len(self.ids) and not self.tx:
                for j in sorted(self.rep):
                    if self.waiting[j]:prefix.append(self.wake(j,r.ranks[0]))
                return {'steps':prefix+[{'kind':'controller_rpc','event':{'worker':w,'phase':'finish'}}]}
            pending=self.wait(w);pending['steps']=prefix+pending['steps'];return pending
        if len(self.done)==len(self.ids) and not self.tx:
            prefix=[self.wake(j,r.ranks[0]) for j in sorted(self.rep) if self.waiting[j]]
            if prefix:return {'steps':prefix+[{'kind':'controller_rpc','event':{'worker':w,'phase':'finish'}}]}
            self.returned.add(w);return {'steps':[],'finish':True}
        if not r.requests and not r.active and r.pending is None:return self.wait(w)
        busy={x for v in self.tx.values() for x in (v['source'],v['helper'])}
        free=[j for j in self.rep if j!=w and self.waiting[j] and j not in busy and
              not self.rep[j].requests and not self.rep[j].active and self.rep[j].pending is None]
        free.sort(key=lambda j:(max(abs(s//6-d//6)+abs(s%6-d%6) for s,d in zip(r.ranks,self.rep[j].ranks)),j))
        can_decode=bool(r.active) and not r.requests and r.pending is None
        # Jointly score every ready source with every truly idle helper.
        # Do not read active requests' remaining/EOS values. Pending decode
        # cannot be remapped until its next safe-point callback.
        choices=[]
        if self.priority and free:
            for source,v in self.rep.items():
                if source in busy or not v.active or v.requests or v.pending is not None:continue
                for helper,h in self.rep.items():
                    if helper==source or helper in busy or not self.waiting[helper]:continue
                    if h.requests or h.active or h.pending is not None:continue
                    estimate=self.estimate_locality(source,helper)
                    if estimate['accepted']:choices.append((estimate['predicted_net_ns'],-source,-helper,source))
        chosen=max(choices)[3] if choices else None
        prefix=[local(Cost(compute=1000))] if self.priority and free else []
        if w not in busy and free and can_decode and (not self.priority or w==chosen):
            estimates=[self.estimate_locality(w,j) for j in free]
            choice=max(estimates,key=lambda x:x['predicted_net_ns'])
            self.locality_decisions.append(choice)
            if not choice['accepted']:
                r.safe_interval=self.interval
                result=r.next(it)
                if prefix:
                    self.deferred[w]=result
                    return {'steps':prefix+[{'kind':'controller_rpc','event':{'worker':w,'phase':'deferred'}}]}
                return result
            helper=choice['helper'];batch=len(r.active);q,extra=divmod(sum(x[1] for x in r.active),batch)
            ctx=tuple(q+(j<extra) for j in range(batch));r.safe_interval=self.interval
            normal=r.next(it);self.next_tx+=1
            tag=280000000+self.next_tx*16
            if tag+7>=400000000:raise ValueError('migration tag space exhausted')
            move=migration(self.op,r.ranks,self.rep[helper].ranks,ctx,tag_start=tag)
            t={'id':self.next_tx,'source':w,'helper':helper,'stage':'migrating',
               'helper_migration_started':False,'initial_repeat':normal['repeat'],
               'migration_source':move['source'],'migration_helper':move['helper'],
               'group_id':self.group_ids[(w,helper)]}
            self.tx[w]=t
            self.migrations.append({'tx':t['id'],'source':w,'helper':helper,'batch':batch,
                'context_sum':sum(ctx),'weight_bytes_per_lane':move['weight_bytes_per_lane'],
                'kv_bytes_per_lane':move['kv_bytes_per_lane'],
                'active_ids':[x[0]['trajectory_id'] for x in r.active],
                'initial_actual_contexts':[x[1] for x in r.active],
                'head_layout':move['head_layout'], 'pack_extra_read_bytes_per_lane':move['pack_extra_read_bytes_per_lane'],
                'pack_extra_write_bytes_per_lane':move['pack_extra_write_bytes_per_lane']})
            prefix.extend([local(Cost(compute=1000)),self.wake(helper,r.ranks[0])])
            return {'steps':prefix+[{'kind':'controller_rpc','event':{'worker':w,'phase':'attention_migrate_source'}}]}
        r.safe_interval=self.interval;result=r.next(it)
        if prefix:
            self.deferred[w]=result
            return {'steps':prefix+[{'kind':'controller_rpc','event':{'worker':w,'phase':'deferred'}}]}
        return result

    def migration_source(self,w):
        tx=self.tx[w];key=f'attn-migrate-{tx["id"]}'
        return {'steps':tx['migration_source']+[{'kind':'gate_signal','key':key},
            {'kind':'gate_wait','key':key,'threshold':8},
            {'kind':'controller_rpc','event':{'worker':w,'phase':'attention_source_ready'}}]}

    def source_ready(self,w,it):
        tx=self.tx[w]
        if tx['stage']!='migrating':raise ValueError('migration ready twice')
        return self._prepare_segment(w,it,tx,already_pending=True)

    def attention_source(self,w):
        tx=self.tx[w]
        if tx['stage']!='running' or not tx['ready']:raise ValueError('invalid attention source')
        return {'steps':tx['source_steps'],'repeat':tx['repeat']}


def op_layers_bound(op):
    return op.model.layers*8
