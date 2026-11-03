"""Small placement-only service estimates, distinct from scenario validation.

Raw conditional K-vectors are independent history when available. Otherwise a
stable conditional point length avoids inventing per-task information from
independent Monte Carlo draws. Native replay remains the performance test.
"""
import math
from idea1_controller.lifecycle_memory import SystemMemory


class PlacementServices:
    def __init__(self,env,forecasts):
        self.env=env;self.forecasts=forecasts;self.cache={}
        self.memory=SystemMemory(env.op)

    def group(self,g,tp):
        key=(g,tp)
        if key not in self.cache:
            f=self.forecasts[g];k=8;p=self.env.public[g]['prompt_length']
            raw=f.get('conditional_lengths')
            if raw:
                if len(raw)%k or any(type(v) is not int or not 0<v<=8192 for v in raw):raise ValueError('invalid conditional K-vector pool')
                vectors=[raw[i:i+k] for i in range(0,len(raw),k)]
            else:
                # Output score includes sum(K outputs)+one maximum. This is
                # a stable placement approximation, not E[R] or E[R^2].
                length=max(1,min(8192,int(math.ceil(f['score']/(k+1)))))
                vectors=[(length,)*k]
            total=0.;members=tp//2
            for vector in vectors:
                items=[dict(prompt_length=p,response_length=v) for v in vector]
                for batch in self.memory.batches(items,'teacher',tp,self.env.policy.logprob_mb*members):
                    total+=2*self.env.catalog.phase('teacher',tp,[x['prompt_length']+x['response_length']-1 for x in batch])
                maximum=self.env.policy.train_mb if tp==2 else min(8,self.env.policy.train_mb*tp)
                for batch in self.memory.batches(items,'training',tp,maximum):
                    total+=self.env.catalog.phase('training',tp,[x['prompt_length']+x['response_length']-1 for x in batch])
            self.cache[key]=total/len(vectors)
        return self.cache[key]

    def costs(self,training,inference,paired=False):
        home={r:{g:self.group(g,2*len(ws)) for g in self.env.public} for r,ws in training.items()}
        generation={}
        for r,ws in inference.items():
            tp=1 if paired else 2*len(ws)
            cap=self.env.catalog_data['inference_capacity_frontier'][str(tp)]
            generation[r]=self.env.catalog.decode(tp,cap,4096)*8/(9*cap)
            if paired:generation[r]/=2
        return home,generation
