"""Online FIFO rollout workload engine, independent of scheduling policy.

Response lengths are read solely to inject recorded EOS; they are never given
to the scheduler. Actions commit at the following native safe-point callback.
"""
from collections import deque
from rl_xy.qwen3_sequence import compile_blocks


class Replica:
    def __init__(self,op,requests,ranks,cap,*,safe_interval=None,decode_batch_limit=None):
        self.op=op;self.requests=deque(requests);self.ranks=tuple(ranks);self.cap=cap;self.safe_interval=safe_interval;self.decode_batch_limit=decode_batch_limit
        if safe_interval is not None and (type(safe_interval) is not int or safe_interval<1):raise ValueError("invalid safe interval")
        if decode_batch_limit is not None and (type(decode_batch_limit) is not int or decode_batch_limit<1):raise ValueError("invalid decode batch limit")
        self.active=[];self.pending=None;self.done=[];self.cache={}

    def observe(self):
        if self.pending:
            kind,data=self.pending
            if kind=='prefill':
                for r in data:
                    if r['response_length']==1:self.done.append(r['trajectory_id'])
                    else:self.active.append([r,r['prompt_length']+1,r['response_length']-1])
            else:
                steps,selected=data;selected=set(selected);untouched=[];processed=[]
                for x in self.active:
                    if x[0]['trajectory_id'] in selected:
                        x[1]+=steps;x[2]-=steps
                        if x[2]==0:self.done.append(x[0]['trajectory_id'])
                        else:processed.append(x)
                    else:untouched.append(x)
                # Round-robin among batches; no future length is used to order.
                self.active=untouched+processed
            self.pending=None

    def next(self,iteration):
        if iteration in self.cache:return self.cache[iteration]
        self.observe()
        tp=len(self.ranks)
        if self.requests and len(self.active)<self.cap:
            add=[]
            while self.requests and len(self.active)+len(add)<self.cap:add.append(self.requests.popleft())
            blocks=self.op.forward(tuple(r['prompt_length'] for r in add),tp,'generation')
            result={'steps':compile_blocks(self.op,self.ranks,blocks),'repeat':1}
            self.pending=('prefill',add)
        elif self.active:
            chosen=self.active[:self.decode_batch_limit] if self.decode_batch_limit else self.active
            batch=len(chosen);q,extra=divmod(sum(x[1] for x in chosen),batch)
            ctx=tuple(q+(j<extra) for j in range(batch))
            blocks=self.op.forward((1,)*batch,tp,'generation',ctx)
            next_blocks=self.op.forward((1,)*batch,tp,'generation',tuple(x+1 for x in ctx))
            steps=min(x[2] for x in chosen)
            if self.safe_interval:steps=min(steps,self.safe_interval)
            result={'steps':compile_blocks(self.op,self.ranks,blocks,next_blocks),'repeat':steps}
            self.pending=('decode',(steps,[x[0]['trajectory_id'] for x in chosen]))
        else:result={'steps':[],'finish':True}
        self.cache[iteration]=result
        return result
