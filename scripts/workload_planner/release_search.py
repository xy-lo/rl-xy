"""Offline joint evaluation: every inference/training pair gets its own PB.

A training-home assignment cannot be cached solely by the training partition:
its producer locality depends on the inference partition as well.
"""
import time
from dataclasses import asdict
from idea1_controller.region_release_placement import balanced_release_placement
from idea1_controller.quotas import layout_quotas
from idea1_controller.inference_groups import InferenceAdmission,validate_partition
from idea1_controller.lifecycle_memory import SystemMemory
from idea1_controller.history import predict_groups
from idea1_controller.workload import digest
from idea1_controller.search import risk
from fast_replica import HeapScenarioScorer


def resource_candidates(values,layouts):
    """Explicit offline candidate subset; reject typos instead of silently falling back."""
    if values is None:return None
    if not values:raise ValueError('empty resource candidate bank')
    allowed=set()
    for value in values:
        try:
            inference,training=value.split(':');name,fraction=inference.split('@');fraction=float(fraction)
        except (ValueError,TypeError):raise ValueError('resource candidate must be inference@fraction:training')
        if name not in {'paired1',*layouts} or training not in layouts or fraction not in (.5,.75,1.):
            raise ValueError('unknown resource template or capacity fraction')
        allowed.add((name,training,fraction))
    return allowed

def search(env,mode):
    from search import templates
    start=time.perf_counter();clock=lambda:time.perf_counter()-start
    forecast=env.injected_forecasts if env.injected_forecasts is not None else {g:asdict(x) for g,x in predict_groups(env.pub,env.history).items()}
    if set(forecast)!=set(env.public) or any(len(f['response_scenarios'])!=16 or any(len(v)!=8 for v in f['response_scenarios']) for f in forecast.values()):
        raise ValueError('current execution contract requires complete K8/16 forecasts')
    from placement_services import PlacementServices
    services_model=PlacementServices(env,forecast) if getattr(env,'home_policy','score')=='service' else None
    layouts=templates();allowed=resource_candidates(getattr(env,'resource_candidates',None),layouts);placement_cache={};homes={};variants={};reserves={};certs={};admissions={};candidates=[]
    # The same fixed integer home quotas are used for each physical layout.
    base_quotas={name:layout_quotas(len(env.pub),layout) for name,layout in layouts.items()}
    for inf_name,layout in [('paired1',layouts['row1']),*layouts.items()]:
        if allowed is not None and not any(x[0]==inf_name for x in allowed):continue
        paired=inf_name=='paired1'
        for train_name,training in layouts.items():
            if allowed is not None and not any(x[:2]==(inf_name,train_name) for x in allowed):continue
            pkey=digest([layout,training,base_quotas[train_name],paired if services_model else None])
            if pkey not in placement_cache:
                options={}
                if services_model:
                    costs,producer=services_model.costs(training,layout,paired)
                    options=dict(service_costs=costs,inference_ns_per_score=producer)
                placement_cache[pkey]=balanced_release_placement(env.pub,forecast,training,base_quotas[train_name],layout,**options)
            pb=placement_cache[pkey]
            home=dict(region_members=training,queues=pb['queues'],placement_proof=pb['proof'],long_quota_by_region=pb['proof']['long_quota_by_region'])
            hkey=digest(home);homes[hkey]=home
            maximum=max(sum(len(home['queues'][str(w)])*8 for w in ws) for ws in training.values())
            reserves[hkey]=2*(len(env.pub)*8+max(64,maximum))*(max(env.op.prompt_cap,max(g['prompt_length'] for g in env.pub))+8192)*4
            for fraction in (.5,.75,1.):
                if allowed is not None and (inf_name,train_name,fraction) not in allowed:continue
                tp={r:1 if paired else len(ws)*2 for r,ws in layout.items()}
                caps={str(r):max(1,int(fraction*env.catalog_data['inference_capacity_frontier'][str(tp[r])])) for r in layout}
                inference=dict(groups=layout,caps=caps,paired_tp1=paired,execution_queues=pb['execution_queues'],quota=pb['proof'])
                ikey=digest(inference);variants[ikey]=inference
                for r in layout:
                    key=(reserves[hkey],tp[r],caps[str(r)])
                    if key not in certs:
                        if key[0] not in admissions:admissions[key[0]]=InferenceAdmission(env.op,key[0])
                        admission=admissions[key[0]]
                        certs[key]=admission.evaluate(key[1],key[2],env.pub)
                passed=all(certs[(reserves[hkey],tp[r],caps[str(r)])]['passed'] for r in layout)
                owner={w:r for r,ws in layout.items() for w in ws};origin={g:int(r) for r,gs in pb['execution_queues'].items() for g in gs}
                dependencies={root:sorted({owner[w] for w in ws}|{origin[g] for w in ws for g in home['queues'][str(w)]}) for root,ws in training.items()}
                candidates.append(dict(inf=ikey,training=hkey,inf_name=f'{inf_name}@{fraction}',training_name=train_name,dp=len(training),signature=digest([inference,home]),generated_s=clock(),passed=passed,dependencies=dependencies))
    generated=clock();scorer=HeapScenarioScorer(env.op,env.policy,env.catalog);ends={};services={};memories={};events=[]
    def evaluate(candidate,scenes,stage):
        before=clock();inf=variants[candidate['inf']];times=[]
        for scene in scenes:
            ek=(candidate['inf'],scene)
            if ek not in ends:
                values={}
                for root,ws in inf['groups'].items():
                    gids=inf['execution_queues'][str(root)];tp=1 if inf['paired_tp1'] else len(ws)*2
                    parts=[gids[::2],gids[1::2]] if inf['paired_tp1'] else [gids]
                    values[root]=max(scorer.replica(part,env.public,forecast,scene,tp,inf['caps'][str(root)]) for part in parts)
                ends[ek]=values
            sk=(candidate['training'],scene)
            if sk not in services:
                hkey=candidate['training']
                if hkey not in memories:memories[hkey]=SystemMemory(env.op,reserves[hkey])
                scorer.memory=memories[hkey];services[sk]=scorer.training_services(homes[hkey],env.public,forecast,scene)
            finish=max(max(ends[ek][r] for r in deps)+services[sk][root] for root,deps in candidate['dependencies'].items())
            times.append((finish+env.prefix+env.suffix+(env.tp1_extra if inf['paired_tp1'] else 0))/1e9)
        result={k:v for k,v in candidate.items() if k!='dependencies'}
        result.update(stage=stage,score_start_s=before,scored_s=clock(),risk_s=risk(times),scenarios_s=times,mean_region_dependencies=sum(len(v) for v in candidate['dependencies'].values())/len(candidate['dependencies']))
        events.append(result);return result
    legal=[c for c in candidates if c['passed']]
    if not legal:raise ValueError('no currently capacity-feasible joint candidate')
    coarse_end=generated
    if mode=='capacity_fallback':
        def key(c):
            inf=variants[c['inf']];tp=1 if inf['paired_tp1'] else max(2*len(ws) for ws in inf['groups'].values())
            return tp,-c['dp'],c['signature']
        fine=[evaluate(min(legal,key=key),[0],'capacity-only-fallback-diagnostic')]
    elif mode=='full':fine=[evaluate(c,range(16),'full') for c in legal]
    else:
        coarse=sorted((evaluate(c,[0],'coarse') for c in legal),key=lambda c:(c['risk_s'],c['signature']));coarse_end=clock()
        if mode=='one_scene':fine=coarse
        else:
            keep=12 if mode=='shortlist12' else 6
            representatives={}
            for c in coarse:representatives.setdefault(c['inf_name'].split('@')[0],c)
            promoted=list(representatives.values())
            for c in coarse:
                if len(promoted)>=keep:break
                if c not in promoted:promoted.append(c)
            ids={c['signature'] for c in promoted};fine=[evaluate(c,range(16),'fine') for c in legal if c['signature'] in ids]
    fine_end=clock();winner=min(fine,key=lambda c:(c['risk_s'],c['signature']));inf=variants[winner['inf']];home=homes[winner['training']]
    validate_partition(inf['groups']);validate_partition(home['region_members'])
    if sorted(g for q in inf['execution_queues'].values() for g in q)!=sorted(env.public):raise AssertionError('inference group loss')
    if sorted(g for q in home['queues'].values() for g in q)!=sorted(env.public):raise AssertionError('training group loss')
    elapsed=clock()
    return dict(candidate_constraint=None if allowed is None else sorted(f'{n}@{v}:{t}' for n,t,v in allowed),mode=mode,placement_mode='release',home_policy=getattr(env,'home_policy','score'),winner=winner,inference=inf,training=home,
        timings=dict(generate_admit_s=generated,coarse_s=coarse_end-generated,fine_s=fine_end-coarse_end,finalize_s=elapsed-fine_end,search_s=elapsed),
        counts=dict(inference_variants=len(variants),training_covers=len(layouts),joint_home_assignments=len(homes),candidate_pairs=len(candidates),legal_pairs=len(legal),fine_pairs=len(fine),scenario_replays=sum(len(e['scenarios_s']) for e in events),unique_inference_scenarios=len(ends),fresh_replica_cache_entries=len(scorer.replica_cache)),
        events=events,input_hashes=env.input_hashes,
        score_scope='release-aware fixed-service scenario surrogate; shared mesh queues/runtime collaboration/stealing still require native validation')
