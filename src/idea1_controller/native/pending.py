"""Atomic region-local handoff of unstarted recorded answers, with native cost."""
from rl_xy.qwen3_sequence import local


class PendingMixin:
    def init_pending(self):
        self.pending_transfers=[];self.pending_serial=0;self.initial_owners=dict(self.owners)
    def pending_domain(self,a,b):
        return self.same_domain(a,b) if hasattr(self,'same_domain') else a//3==b//3
    def pending_choice(self,w):
        if self.leased(w) or w in self.services:return None
        candidates=[]
        # A source must be at a native safe point (current or already waiting).
        for src,rep in self.rep.items():
            if not rep.requests or src in self.services or self.leased(src) or (src!=w and not self.waiting[src]):continue
            # Own admission has priority. Donate only queued overflow behind a
            # fully occupied source, so a received request cannot ping-pong.
            if len(rep.active)<rep.cap:continue
            for dst in self.rep:
                if dst==src or not self.pending_domain(src,dst) or not self.empty(dst) or dst in self.services or self.leased(dst):continue
                if dst!=w and not self.waiting[dst]:continue
                if w not in (src,dst):continue
                first=rep.requests[0];gid=first['group_id']
                # FIFO group identity, then stable sample order; no future R.
                items=[x for x in rep.requests if x['group_id']==gid][:min(4,self.rep[dst].cap)]
                if items:candidates.append((str(gid),src,dst,items))
        if not candidates:return None
        _,src,dst,items=min(candidates,key=lambda x:(x[0],x[1],x[2]))
        return src,dst,items
    def schedule_pending(self,w,choice):
        assert self.pending_choice(w)==choice
        src,dst,items=choice;ids={x['trajectory_id'] for x in items}
        assert ids and not ids&self.done
        assert all(self.owners[t]==src for t in ids)
        self.rep[src].requests=type(self.rep[src].requests)(x for x in self.rep[src].requests if x['trajectory_id'] not in ids)
        self.move_id+=1;self.pending_serial+=1
        tx=dict(id=self.move_id,kind='pending',src=src,dst=dst,items=items,tids=sorted(ids),consumed=set(),start_tick=self.tick,
            bytes_per_lane=sum(x['prompt_length']*4+64 for x in items),serial=self.pending_serial)
        self.services[src]=self.services[dst]=tx;self.waiting[w]=False
        other=dst if src==w else src
        return {'steps':[self.wake(other,self.rep[w].ranks[0]),self.rpc(w,'sample_run')]}
    def sample_run(self,w):
        tx=self.services[w]
        if tx.get('kind')!='pending':return super().sample_run(w)
        size=tx['bytes_per_lane'];steps=[]
        for lane,(src,dst) in enumerate(zip(self.rep[tx['src']].ranks,self.rep[tx['dst']].ranks)):
            steps += [local(self.op.vec(0,size,0,'pending_prompt_read'),ranks=[src]),
                dict(kind='p2p_direct',src=src,dst=dst,bytes=size,tag=480000000+tx['serial']*2+lane),
                local(self.op.vec(0,0,size,'pending_prompt_write'),ranks=[dst])]
        key=f'pending-{tx["serial"]}'
        return {'steps':steps+[dict(kind='gate_signal',key=key),dict(kind='gate_wait',key=key,threshold=4),self.rpc(w,'sample_done',transfer=tx['id'])]}
    def sample_done(self,w,transfer):
        tx=self.services[w]
        if tx.get('kind')!='pending':return super().sample_done(w,transfer)
        self.services.pop(w);assert tx['id']==transfer and w not in tx['consumed']
        if not tx['consumed']:
            assert self.empty(tx['dst'])
            for item in tx['items']:
                tid=item['trajectory_id'];assert self.owners[tid]==tx['src'] and tid not in self.done
                self.owners[tid]=tx['dst'];self.rep[tx['dst']].requests.append(item)
            self.pending_transfers.append({k:v for k,v in tx.items() if k not in ('items','consumed')}|dict(end_tick=self.tick))
        tx['consumed'].add(w)
        return {'steps':[]}
