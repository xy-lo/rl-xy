"""Unstarted answers move between real inference cohorts, with paid input fanout."""
from .pending import PendingMixin
from rl_xy.qwen3_sequence import local
from rl_xy.qwen3_cost import Cost


class CohortPending:
    def pending_choice(self,w):
        if not getattr(self,'pending_enabled',False) or getattr(self,'paired_tp1',False):return None
        if self.inference_owner[w]!=w or w in self.services or self.pending_lease(w):return None
        src=self.rep[w]
        if not src.requests or len(src.active)<src.cap or src.pending is not None:return None
        roles=self.logical_roles;choices=[];source_members=self.inference_members[w]
        for dst,members in self.initial_inference_members.items():
            if dst==w or set(source_members)&set(members) or len(members)<len(source_members):continue
            if any(self.inference_owner[j]!=dst for j in members):continue
            if any(self.inference_busy(j) or self.leased(j) or j in self.services or j in self.trained or not self.waiting[j] for j in members):continue
            def allowed(j):return roles[j]=='ROLLOUT' or (getattr(self,'pending_data_wait',False) and roles[j]=='TRAINING' and self.stages[j]=='data_wait' and self.region_of[j] not in self.region_started)
            if not all(allowed(j) for j in members):continue
            hops=max(abs(a//6-b//6)+abs(a%6-b%6) for a,b in zip(src.ranks,self.rep[dst].ranks))
            choices.append((not self.same_domain(w,dst),hops,-len(members),dst))
        if not choices:return None
        dst=min(choices)[-1];items=list(src.requests)[:self.rep[dst].cap]
        return w,dst,items

    def pending_lease(self,w):return any(w in members for members in getattr(self,'pending_cohort_leases',{}).values())

    def schedule_pending(self,w,choice):
        src,dst,items=choice;members=self.initial_inference_members[dst]
        for j in members:
            if self.logical_roles[j]=='TRAINING':
                assert self.stages[j]=='data_wait' and self.region_of[j] not in self.region_started
                self.inference_saved_roles[j]=dict(role='TRAINING',stage='data_wait');self.logical_roles[j]='ROLLOUT'
                self.inference_role_events.append(dict(worker=j,tick=self.tick,old='TRAINING',new='ROLLOUT',reason='unstarted-answer admission during DATA_WAIT'))
        # Re-evaluation sees the same safe peers after explicit role transition.
        reply=PendingMixin.schedule_pending(self,w,choice);tx=self.services[w]
        tx.update(cohort_pending=True,dst_members=members)
        self.inference_tag+=2;tx['explicit_tags']=[self.inference_tag-1,self.inference_tag]
        self.pending_cohort_leases[tx['id']]=tuple(sorted(set(self.inference_members[src])|set(members)))
        self.cohort_drained=getattr(self,'cohort_drained',set());self.cohort_drained.discard(dst)
        return reply

    def sample_run(self,w):
        tx=self.services[w];reply=super().sample_run(w)
        if tx.get('cohort_pending'):
            for i,step in enumerate(s for s in reply['steps'] if s['kind']=='p2p_direct'):step['tag']=tx['explicit_tags'][i]
        return reply

    def sample_done(self,w,transfer):
        tx=self.services[w];pending=tx.get('cohort_pending');first=not tx['consumed']
        reply=super().sample_done(w,transfer)
        if pending and first:
            self.inference_input_broadcasts.setdefault(tx['dst'],[]).extend(tx['items'])
        if pending and len(tx['consumed'])==2:
            members=self.pending_cohort_leases.pop(tx['id'])
            wake=[self.wake(j,self.rep[w].ranks[0]) for j in members if self.waiting[j]]
            if wake:return dict(steps=wake+[local(Cost(compute=1000)),self.rpc(w,'step')])
        return reply

    def pending_input_fanout(self,r):
        items=self.inference_input_broadcasts.pop(r,[])
        if not items:return [],[]
        size=sum(x['prompt_length']*4+64 for x in items);steps=[]
        for w in self.inference_members[r]:
            if w==r:continue
            for src,dst in zip(self.rep[r].ranks,self.rep[w].ranks):
                self.inference_tag+=1
                steps.extend([local(self.op.vec(0,size,0,'pending_cohort_prompt_read'),ranks=[src]),
                    dict(kind='p2p_direct',src=src,dst=dst,bytes=size,tag=self.inference_tag),
                    local(self.op.vec(0,0,size,'pending_cohort_prompt_write'),ranks=[dst])])
        return steps,[x['trajectory_id'] for x in items]
