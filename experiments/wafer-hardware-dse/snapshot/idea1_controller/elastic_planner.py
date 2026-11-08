"""TUNE-bound candidate bank -> public-history TP choice -> native manifest."""
from copy import deepcopy
import math,time,json
from .system import SystemPlanner
from .inference_groups import InferenceAdmission,validate_partition
from .inference_search import InferenceScenarioScorer,balanced_execution,InferenceHomePlacement
from .history import digest,validate_history,quantile


class ElasticSystemPlanner:
    def __init__(self,op,policy,catalog,bank):
        self.op=op;self.policy=policy;self.catalog=catalog;self.bank=bank
        if bank.get('selection_window')!='tune':raise ValueError('TP bank must be selected on TUNE')
        if bank.get('service_contract')!=catalog.data.get('service_contract'):raise ValueError('TP bank mathematical/hardware binding mismatch')
        if bank.get('model_config_sha256')!=catalog.data.get('model_config_sha256'):raise ValueError('TP bank model identity mismatch')
    def build(self,public,history,budget_s=15,force_profile=None):
        if any(set(g)-{'group_id','prompt_length','prompt_sha256','library'} for g in public):raise ValueError('planner forbids live response/EOS fields')
        validate_history(public,history,self.bank['model'],self.bank['model_config_sha256'])
        start=time.perf_counter();homes={};prepared={}
        public_by_id={str(g['group_id']):g for g in public}
        maxp=max(self.op.prompt_cap,max(g['prompt_length'] for g in public))
        samples=len(public)*8;scorer=InferenceScenarioScorer(self.op,self.policy,self.catalog)
        layout_count=len({digest(p.get('training_regions',self.bank['training_regions'])) for p in self.bank['profiles']})
        candidates=[];rejected=[]
        for profile in self.bank['profiles']:
            regions=profile.get('training_regions',self.bank['training_regions']);strategy=profile.get('home_assignment_strategy');layout_id=digest(dict(regions=regions,strategy=strategy))
            if layout_id not in homes:
                home=(InferenceHomePlacement(self.op,self.policy,self.catalog).build(public,history,regions,strategy) if strategy is not None else
                    SystemPlanner(self.op,self.policy,self.catalog).build(public,history,regions,budget_s/max(1,layout_count)))
                home=json.loads(json.dumps(home)) # same key/types as installed native manifest
                forecast=home['forecasts'];cold=not forecast
                if cold:forecast={str(g['group_id']):dict(label='unknown',score=9*8192,response_scenarios=[(8192,)*8]*16) for g in public}
                maximum=max(sum(len(home['queues'][str(w)])*8 for w in ws) for ws in regions.values())
                reserve=2*(samples+max(64,maximum))*(maxp+8192)*4
                training=[scorer.training_services(home,public_by_id,forecast,s) for s in range(16)]
                homes[layout_id]=home;prepared[layout_id]=(forecast,cold,InferenceAdmission(self.op,reserve),training)
            home=homes[layout_id];forecast,cold,memory,training=prepared[layout_id]
            inf=deepcopy(profile['inference']);groups={int(r):tuple(ws) for r,ws in inf['groups'].items()};validate_partition(groups)
            certs={str(r):memory.evaluate(1 if inf.get('paired_tp1') else len(ws)*2,int(inf['caps'][str(r)]),public) for r,ws in groups.items()}
            if not all(c['passed'] for c in certs.values()):rejected.append(dict(profile_id=profile['profile_id'],reason='current public P capacity'));continue
            if profile.get('assignment_mode')=='throughput-balanced' and not cold:
                inf['execution_queues'],inf['execution_assignment_proof']=balanced_execution(public,forecast,inf,self.catalog)
            raw=scorer.score(home,inf,public,forecast,training);ratio=profile.get('TUNE_calibration_ratio',1.)
            if not .25<=ratio<=4 or not math.isfinite(ratio):raise ValueError('invalid TUNE discrepancy factor')
            scenarios=[s*ratio for s in raw['scenario_ns']]
            candidates.append(dict(profile_id=profile['profile_id'],risk_ns=raw['risk_ns']*ratio,scenario_ns=scenarios,inference=inf,
                training_layout_id=layout_id,
                source_TUNE_audit_sha256=profile['source_TUNE_audit_sha256'],TUNE_calibration_ratio=ratio,
                raw_surrogate=raw,capacity_proof_sha256=digest(certs)))
        if not candidates:raise ValueError('no capacity-feasible inference TP profile for public tasks')
        anchor=next((p for p in candidates if p['profile_id']==self.bank['anchor_profile']),min(candidates,key=lambda p:p['risk_ns']))
        best=min(candidates,key=lambda p:(p['risk_ns'],p['profile_id']));chosen=anchor;reason='TUNE anchor / uncertainty fallback'
        margin=self.bank.get('switch_margin_fraction',.03)
        gains=[a-b for a,b in zip(anchor['scenario_ns'],best['scenario_ns'])]
        if best['profile_id']!=anchor['profile_id'] and quantile(gains,.1)>margin*anchor['risk_ns']:
            chosen=best;reason='public-history Q10 benefit exceeds calibrated uncertainty margin'
        if force_profile is not None:
            chosen=next(c for c in candidates if c['profile_id']==force_profile);reason='explicit controlled profile replay'
        elapsed=time.perf_counter()-start
        home=deepcopy(homes[chosen['training_layout_id']]);regions=home['region_members']
        actual=chosen['inference']
        home.update(inference_layout=chosen['inference'],selected_inference_profile=chosen['profile_id'],
            rollout_tp=1 if actual.get('paired_tp1') else max(2*len(ws) for ws in actual['groups'].values()),
            rollout_cap=max(map(int,actual['caps'].values())),rollout_replica_count=54 if actual.get('paired_tp1') else len(actual['groups']),
            storage_owner_tp=2,optimizer_owner_dp=27,
            inference_selection=dict(reason=reason,anchor_profile=anchor['profile_id'],switch_margin_fraction=margin,
                candidates=[{k:v for k,v in c.items() if k not in ('inference','scenario_ns','raw_surrogate')} for c in candidates],rejected=rejected,
                surrogate_scope='TUNE-only discrepancy factors affect selection, never reported native duration; exact queues/stealing not predicted'),
            supported_inference_tp=[1,2,4,8],parallelism_feasibility=[dict(profile_id=c['profile_id'],feasible=True,capacity_proof_sha256=c['capacity_proof_sha256']) for c in candidates],
            total_planner_wall_s=elapsed,planning_wall_s=elapsed,planner_source='ElasticSystemPlanner: independent inference TP partition over canonical TP2 optimizer owners')
        home['plan_id']=digest(dict(home_assignment=home['queues'],training_regions=regions,inference=chosen['inference'],history=history['snapshot_id']))
        return home
