"""Two real TP1 request engines per TP2 storage/control pair.

Each engine executes the full TP1 operator on its single die. The shared
control safe point is the slower of two <=16-token chunks, explicitly paid by
native gates. EOS records are copied to both canonical TP2 owners before any
training consumer sees completion. This is a declared TP1 policy, not ideal
fully asynchronous die control.
"""
from online_rollout import Replica
from rl_xy.qwen3_sequence import local


class PairedTP1Replica:
    inference_tp=1
    def __init__(self,op,items,ranks,cap,new_tag):
        self.op=op;self.ranks=tuple(ranks);self.execution_ranks=self.ranks;self.cap=cap
        self.items=tuple(items)
        groups=[]
        for x in items:
            if not groups or groups[-1][0]['group_id']!=x['group_id']:groups.append([])
            groups[-1].append(x)
        queues=[[],[]]
        for i,g in enumerate(groups):queues[i%2].extend(g)
        self.engines={r:Replica(op,q,(r,),cap,safe_interval=16) for r,q in zip(ranks,queues)}
        self.request_rank={x['trajectory_id']:r for r,e in self.engines.items() for x in e.requests}
        self.cursor={r:0 for r in ranks};self.done=[];self.pending=None;self.cache={};self.new_tag=new_tag
        self.record_queue=[];self.event={}
    @property
    def requests(self):return [x for e in self.engines.values() for x in e.requests]
    @property
    def active(self):return [x for e in self.engines.values() for x in e.active]
    def observe(self):
        if not self.pending:return
        kind,data=self.pending
        if kind=='paired_body':
            for r,e in self.engines.items():
                e.observe()
                self.record_queue.extend((t,r) for t in e.done[self.cursor[r]:]);self.cursor[r]=len(e.done)
            self.pending=('copy_needed',None) if self.record_queue else None
        elif kind=='paired_copy':
            self.done.extend(t for t,r in self.record_queue);self.record_queue=[];self.pending=None
    def next(self,iteration):
        if iteration in self.cache:return self.cache[iteration]
        self.observe()
        if self.pending and self.pending[0]=='copy_needed':
            steps=[];copied=[]
            for src in self.ranks:
                tids=[t for t,r in self.record_queue if r==src]
                if not tids:continue
                dst=next(r for r in self.ranks if r!=src)
                allitems={x['trajectory_id']:x for x in self.items}
                size=sum(4*(allitems[t]['prompt_length']+allitems[t]['response_length']) for t in tids)
                tag=self.new_tag()
                steps.extend([local(self.op.vec(0,size,0,'TP1_EOS_record_read'),ranks=[src]),
                    dict(kind='p2p_direct',src=src,dst=dst,bytes=size,tag=tag),
                    local(self.op.vec(0,0,size,'TP1_EOS_record_write'),ranks=[dst])])
                copied.append(dict(src=src,dst=dst,tids=tids,bytes=size,tag=tag))
            self.event=dict(kind='paired_copy',copies=copied,lanes=[]);self.pending=('paired_copy',None)
            result=dict(steps=steps,repeat=1)
        else:
            replies={};lanes=[]
            for r,e in self.engines.items():
                reply=e.next(iteration);e.cache.pop(iteration,None);replies[r]=reply
                lanes.append(dict(rank=r,kind=e.pending[0] if e.pending else 'empty',active_ids=[x[0]['trajectory_id'] for x in e.active],
                    actual_contexts=[x[1] for x in e.active],pending_ids=[x['trajectory_id'] for x in e.pending[1]] if e.pending and e.pending[0]=='prefill' else []))
            repeated=[v.get('repeat',1) for v in replies.values() if not v.get('finish')]
            repeat=min(repeated) if repeated else 1;steps=[]
            for r,reply in replies.items():
                engine=self.engines[r]
                if engine.pending and engine.pending[0]=='decode':engine.pending=('decode',(repeat,engine.pending[1][1]))
                for s in reply['steps']:
                    assert s['kind']=='local','TP1 unexpectedly issued an inter-die operator collective'
                    steps.append(dict(s,ranks=[r]))
            self.event=dict(kind='paired_body',lanes=lanes,copies=[]);self.pending=('paired_body',None)
            result=dict(steps=steps,repeat=repeat)
        self.cache[iteration]=result;return result
