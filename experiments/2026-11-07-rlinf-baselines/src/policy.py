"""Paper §4.2 feedback; baseline offline profiles use TUNE, never test EOS."""
from dataclasses import replace
from rl_xy.qwen3_rollout import Rollout
from sim import Backend

def admitted_cap(be):
    cap=be.p.cap; maxprompt=be.op.prompt_cap
    while cap>0:
        reserved=be.state['training_base_max']+be.state['rollout_weights_per_rank']+cap*(maxprompt+8192)*be.kv+1024**2
        if reserved<=be.capacity: return cap,reserved
        cap//=2
    raise ValueError('no rollout slot fits')

def offline(root,model,base,tune):
    # Replay a frozen TUNE rollout as an offline profile. It is an analytic
    # profile, not a newly measured GPU benchmark, and is explicitly labelled.
    candidates=[]; backends={}
    for tp in (1,2,4,8):
        be=Backend(root,model,replace(base,rollout_tp=tp)); cap,reserve=admitted_cap(be)
        be.p=replace(be.p,cap=cap)
        seconds=Rollout(be.op,be.net).run(tune,tp,cap)['model_ns']/1e9
        row={'tp':tp,'cap':cap,'seconds':seconds,'reserved_bytes':reserve}
        candidates.append(row); backends[tp]=be
    winner=min(candidates,key=lambda row:(row['seconds'],row['tp']))
    be=backends[winner['tp']]
    responses=sorted(r['response_length'] for r in tune)
    be.history_response_bound=responses[min(len(responses)-1,int(.95*len(responses)))]
    return be,{'scope':'analytic offline TUNE profiling; no test-length decisions','candidates':candidates,'selected':winner,
               'history_response_p95':be.history_response_bound}

class Feedback:
    def __init__(self,tp): self.tp=tp; self.previous=None; self.zeros=0
    def observe(self,preemptions):
        before=self.tp; reason='retain'
        if preemptions>0 and self.previous is not None and preemptions>1.05*self.previous:
            self.tp=min(8,self.tp*2); self.zeros=0; reason='preemption-rise'
        elif preemptions==0:
            self.zeros+=1
            if self.zeros>=4:
                self.tp=max(1,self.tp//2); self.zeros=0; reason='four-zero-preemption-steps'
        else: self.zeros=0
        self.previous=preemptions
        return {'old_tp':before,'new_tp':self.tp,'preemptions':preemptions,'reason':reason}
