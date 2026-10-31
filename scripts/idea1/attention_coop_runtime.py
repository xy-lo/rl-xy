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
from attention_tp8_program import migration,compile_attention_tp8


class AttentionRuntime(Reclaimer):
    def __init__(self,*args,group_ids,**kwargs):
        super().__init__(*args,**kwargs)
        self.group_ids=group_ids;self.migrations=[];self.releases=[];self.segment_counter=0

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
                          'contexts':ctx,'tx':tx['id'],'segment':self.segment_counter})
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
        free.sort(key=lambda j:(abs(self.rep[j].ranks[0]-r.ranks[0]),j))
        can_decode=bool(r.active) and not r.requests and r.pending is None
        eligible=[(priority_score(v,self.completed_history),-j,j) for j,v in self.rep.items()
                  if j not in busy and v.active and not v.requests and
                  (v.pending is None or v.pending[0]=='decode')] if self.priority and free else []
        chosen=max(eligible)[2] if eligible else None
        prefix=[local(Cost(compute=1000))] if self.priority and free else []
        if w not in busy and free and can_decode and (not self.priority or w==chosen):
            helper=free[0];batch=len(r.active);q,extra=divmod(sum(x[1] for x in r.active),batch)
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
                'active_ids':[x[0]['trajectory_id'] for x in r.active]})
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
