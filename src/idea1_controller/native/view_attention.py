"""Published attention kernel, restricted to planned training-region members."""
from locality_runtime_v3 import AttentionRuntime
from rl_xy.qwen3_sequence import local
from rl_xy.qwen3_cost import Cost
from .view_migration import migrate_actor_view as migration

class ViewAttention(AttentionRuntime):
    def step(self,w,it):
        if not self.enabled:return super().step(w,it)
        r=self.rep[w];self._observe(w)
        tx=next((v for v in self.tx.values() if w in (v['source'],v['helper'])),None)
        if tx and tx['helper']==w:
            if tx['stage']=='migrating' and not tx['helper_migration_started']:
                tx['helper_migration_started']=True;key=f'attn-migrate-{tx["id"]}'
                return {'steps':tx['migration_helper']+[{'kind':'gate_signal','key':key},
                    {'kind':'gate_wait','key':key,'threshold':4},
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
        free=[j for j in self.rep if self.same_domain(j,w) and j!=w and self.waiting[j] and j not in busy and
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
                    if not self.same_domain(helper,source) or helper==source or helper in busy or not self.waiting[helper]:continue
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

