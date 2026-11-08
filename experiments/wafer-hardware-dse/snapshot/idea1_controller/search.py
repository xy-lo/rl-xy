"""Resource-first enumeration, quota assignment, then paired scenario selection.

The planner accepts public requests + frozen history, not live response/EOS.
The native manifest is the only plan handed to execution. Every approximate
score is marked; exact native validation can reject a poor chosen candidate.
"""
from collections import deque
from dataclasses import asdict
import statistics,time,math
from .placement import guillotine_layouts,assign_groups,largest_remainder
from .history import predict_groups,digest,quantile


def risk(times):return .75*statistics.mean(times)+.25*statistics.mean(sorted(times)[-2:])

class RiskSelector:
    def select(self,baseline,candidates,planning_s,uncertainty=0):
        b=baseline['scenario_ns'];eligible=[];audit=[]
        for c in candidates:
            delta=[x-y for x,y in zip(b,c['scenario_ns'])]
            low=quantile(delta,.1);margin=(.02+uncertainty)*statistics.median(b)+planning_s*1e9
            good=risk(c['scenario_ns'])<risk(b) and c['pressure_ns']<=1.05*baseline['pressure_ns'] and low>margin
            audit.append(dict(candidate=c['id'],accepted=good,q10_gain_ns=low,required_gain_ns=margin,risk_ns=risk(c['scenario_ns']),pressure_ratio=c['pressure_ns']/baseline['pressure_ns']))
            if good:eligible.append(c)
        return min(eligible,key=lambda c:(risk(c['scenario_ns']),c['id'])) if eligible else baseline,audit


class NativePlanSearch:
    def __init__(self,catalog,op,policy,*,helper=True,max_finalists=10):
        self.catalog=catalog;self.op=op;self.policy=policy;self.helper=helper;self.max_finalists=max_finalists;self.decode_cache={}
        from rl_xy.qwen3_experiment import simple_plan_time,weight_plan,optimizer_plan
        from rl_xy.qwen3_native import NetworkProfiles
        net=NetworkProfiles(op,'unused',native=False)
        self.common_ns=simple_plan_time(weight_plan(op,policy),net)+simple_plan_time(optimizer_plan(op,policy),net)
        if 'persistent_reference_shard_bytes' in op.state(2,2):
            self.common_ns+=net.time(getattr(op,'active_dies',54)//2,'ALL_GATHER',op.state(2,2)['persistent_reference_shard_bytes'])
    def _replica(self,ids,public,forecast,scenario,pressure=False):
        queue=deque((public[g]['prompt_length'],8192 if pressure else int(length))
            for g in ids for length in ((8192,)*8 if pressure else forecast[g].response_scenarios[scenario%16]))
        # Each tuple contains only a historical scenario output. Actual live R
        # never crosses the planner interface.
        active=[];elapsed=0.;cap=self.policy.cap
        while queue or active:
            if queue and len(active)<cap:
                add=[]
                while queue and len(active)+len(add)<cap:add.append(queue.popleft())
                elapsed+=self.catalog.phase('prefill',2,[p for p,r in add])
                active.extend([p+1,r-1] for p,r in add if r>1)
            else:
                steps=min(r for c,r in active);mean=sum(c for c,r in active)/len(active)
                a=self.catalog.decode(2,len(active),mean);b=self.catalog.decode(2,len(active),mean+1)
                elapsed+=steps*a+steps*(steps-1)/2*(b-a)
                active=[[c+steps,r-steps] for c,r in active if r>steps]
        return elapsed
    def _score(self,regions,queues,public,forecast,scenario,pressure=False):
        op=self.op;cat=self.catalog;max_end=0.;region_rows=[]
        for root,ws in sorted(regions.items()):
            own={w:self._replica(queues[w],public,forecast,scenario,pressure) for w in ws};end=own.copy();tp=2*len(ws)
            # Explicit conservative one-pair lease estimate; final native replay
            # executes all real decisions, migration, KV append and contention.
            helper_credit=0.
            if self.helper and len(ws)>1:
                helper=min(ws,key=lambda w:(end[w],w));src=max(ws,key=lambda w:(end[w],-w));ready=end[helper]
                if src!=helper and end[src]>ready:
                    base=cat.decode(2,1,4096);wide=cat.decode(4,1,4096)
                    bw=op.hw.d2d_bytes_per_ns_per_direction;hops=max(abs(s//6-d//6)+abs(s%6-d%6) for s,d in zip((2*src,2*src+1),(2*helper,2*helper+1)))
                    from attention_tp8_program import attention_weight_bytes_per_helper,kv_bytes_per_helper
                    size=attention_weight_bytes_per_helper(op)+kv_bytes_per_helper(op,(4096,))
                    migration=size/bw*max(1,hops)+2*size/op.hw.dram_bytes_per_ns
                    # Extra TP4 collective/input and persistent KV are not free.
                    ratio=min(1,(wide+op.model.layers*(op.model.hidden_size*2/bw*max(1,hops)+op.hw.d2d_latency_ns))/base)
                    saved=max(0,(end[src]-ready)*(1-ratio)-migration)
                    end[src]-=saved;end[helper]=max(ready,end[src]);helper_credit=saved
            rollout=max(end.values(),default=0.);gids=[g for w in ws for g in queues[w]]
            teacher=training=record_bytes=0.
            for g in gids:
                lengths=(8192,)*8 if pressure else forecast[g].response_scenarios[scenario%16]
                teacher_lengths=[public[g]['prompt_length']+r-1 for r in lengths]
                for i in range(0,8,self.policy.logprob_mb):teacher+=2*cat.phase('teacher',tp,teacher_lengths[i:i+self.policy.logprob_mb])
                for i in range(0,8,self.policy.train_mb):training+=cat.phase('training',tp,teacher_lengths[i:i+self.policy.train_mb])
                record_bytes+=sum((public[g]['prompt_length']+r)*4 for r in lengths)
            # All samples are copied to the root then to physical TP members.
            data=record_bytes/max(1,op.hw.d2d_bytes_per_ns_per_direction)*(len(ws)-1)*2
            # Restore to existing immutable TP2 optimizer owners. Extra region
            # ranks become zero DP contributors; no owner relocation assumed.
            from rl_xy.model import enumerate_parameters
            p=sum(x.unique_elements for x in enumerate_parameters(op.case_tp(2)))
            restore=0. if len(ws)==1 else (p*4*(1-1/len(ws))/op.hw.d2d_bytes_per_ns_per_direction+p*16/op.hw.dram_bytes_per_ns)
            finish=rollout+data+teacher+training+restore
            region_rows.append(dict(root=root,tp=tp,rollout_ns=rollout,teacher_ns=teacher,training_ns=training,restore_ns=restore,helper_credit_ns=helper_credit,finish_ns=finish))
            max_end=max(max_end,finish)
        return max_end+self.common_ns,region_rows
    def build(self,public_groups,history,*,deadline_s=5):
        start=time.perf_counter();public={str(g['group_id']):dict(g) for g in public_groups};ids=list(public);forecast=predict_groups(public_groups,history)
        # Freeze integer origin quotas once; changing TP/layout never alters G/K.
        quotas=dict(enumerate(largest_remainder(len(ids),[1]*27)));layouts=guillotine_layouts(2);coarse=[]
        for d,variants in layouts.items():
            for rects in variants:
                regions={min(x//2 for x in r.dies(2)):sorted(set(x//2 for x in r.dies(2))) for r in rects}
                for adaptive in (False,True):
                    queues,longq=assign_groups(ids,forecast,regions,quotas,adaptive)
                    value,rows=self._score(regions,queues,public,forecast,0)
                    cid=digest(dict(regions=regions,queues=queues))
                    coarse.append(dict(id=cid,dp=d,regions=regions,queues=queues,long_quotas=longq,adaptive=adaptive,coarse_ns=value,coarse_regions=rows))
        # Up to four DP-count strata, two distinct best joint candidates each.
        counts=sorted(layouts);bins=min(4,len(counts));selected=[]
        for b in range(bins):
            ds={d for i,d in enumerate(counts) if bins*i//len(counts)==b}
            selected+=sorted((c for c in coarse if c['dp'] in ds),key=lambda c:(c['coarse_ns'],c['id']))[:2]
        baseline=min((c for c in coarse if c['dp']==27 and not c['adaptive']),key=lambda c:c['id'])
        selected=[baseline]+[c for c in selected if c['id']!=baseline['id']]
        final=[];truncated=False
        for c in selected[:self.max_finalists]:
            if final and time.perf_counter()-start>deadline_s:truncated=True;break
            c=dict(c);c['scenario_ns']=[self._score(c['regions'],c['queues'],public,forecast,s)[0] for s in range(16)]
            c['pressure_ns']=self._score(c['regions'],c['queues'],public,forecast,0,True)[0];final.append(c)
        elapsed=time.perf_counter()-start
        winner,audit=RiskSelector().select(final[0],final[1:],elapsed,uncertainty=min(.25,self.catalog.data['p95_relative_validation_error']))
        manifest=dict(schema='idea1.native-selected-plan/1',region_members=winner['regions'],rollout_tp=2,queues=winner['queues'],
            source='public requests + independent same-task history; no live answer lengths',history_snapshot_id=history['snapshot_id'],
            forecasts={g:asdict(f) for g,f in forecast.items()},selected_candidate=winner['id'],selected_dp=winner['dp'],long_quota_by_region=winner['long_quotas'],
            all_coarse_candidates=coarse,fully_scored_candidates=final,selection_audit=audit,planning_wall_s=elapsed,budget_s=deadline_s,budget_truncated=truncated,
            scope='approximate search-only cycle scores; actual native replay is required',optimizer_owners='fixed original TP2 distributed owner layout',
            unsupported=['nonuniform TP6','optimizer-owner migration','hardware-validated timing'])
        manifest['plan_id']=digest({k:manifest[k] for k in ('region_members','queues','history_snapshot_id')});return manifest
