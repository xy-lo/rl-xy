"""Independent conservation, lifecycle and timing audit of a selected native run.

This audits simulator evidence. It does not establish Transformer numerical
equivalence, physical memory allocation, calibrated hardware or learning quality.
"""
import argparse
import csv
import hashlib
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load(p):
    return json.loads(p.read_text())


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit(folder):
    result = load(folder/'result.json')
    cmd = load(folder/'command.json')
    trace=Path(cmd['trace_path'])
    if not trace.is_absolute():trace=Path(__file__).resolve().parents[1]/trace
    items = load(trace)['trajectories']
    if result['subset_prompts']:
        items = items[:result['subset_prompts']*8]
    byid = {x['trajectory_id']: x for x in items}
    ids = set(byid)
    assert digest(items) == cmd['canonical_requests_sha256']
    assert len(items) == len(ids) == result['requests'] == result['completed'] == result['trained_samples']
    assert set(Counter(x['group_id'] for x in items).values()) == {8}
    plan = load(folder/'resource-layout.json')
    regions = {int(r): tuple(ws) for r, ws in plan['region_members'].items()}
    assert sorted(w for ws in regions.values() for w in ws) == list(range(27))
    assert all(r == min(ws) and len(ws)*2 in (2, 4, 8) for r, ws in regions.items())
    queues = {int(w): q for w, q in plan['queues'].items()}
    assert Counter(g for q in queues.values() for g in q) == Counter(str(x['group_id']) for x in items[::8])
    owner_region = {w: r for r, ws in regions.items() for w in ws}
    group_region = {g: owner_region[w] for w, q in queues.items() for g in q}
    if plan.get('forecasts'):
        for r, quota in plan['long_quota_by_region'].items():
            quota_regions={int(k):v for k,v in plan.get('helper_domains',plan['region_members']).items()}
            assert sum(plan['forecasts'][g]['label'] == 'proxy-long' for w in quota_regions[int(r)] for g in queues[w]) == quota
    # Verify copied source bytes, independent of the live development tree.
    archive = load(folder/'source-archive-map.json')
    for name, sha in load(folder/'source_hashes.json').items():
        source=Path(archive[name])
        if not source.is_absolute():source=folder/source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == sha, name
    capacity = load(folder/'capacity.json')
    assert capacity['passed'] and max(capacity['phase_peak_bytes'].values()) <= capacity['capacity_bytes']
    alloc = load(folder/'allocation-events.json')
    assert all(x['peak_bytes'] <= capacity['capacity_bytes'] and x['start_tick'] <= x['end_tick'] for x in alloc)
    before, after = (load(folder/f'actor-views-{s}.json') for s in ('before', 'after'))
    assert before['version'] == 'actor_v0' and after['version'] == 'actor_v1' and after['generation'] == 1
    assert len(after['allocations']) == 54 and before['schema_digest'] == after['schema_digest']
    for d, a in after['allocations'].items():
        b = before['allocations'][d]
        assert a['training_actor'] == a['inference_actor'] == b['training_actor'] == a['actor_allocation']
    rows = [json.loads(line) for line in (folder/'occupancy.jsonl').open()]
    eos = {}
    for i, row in enumerate(rows):
        assert len(row['roles']) == 27 and (i == 0 or rows[i-1]['tick'] <= row['tick'])
        for t in row['new_done']:
            assert t in ids and t not in eos
            eos[t] = row['tick']
    assert set(eos) == ids
    domains={int(r):tuple(ws) for r,ws in plan.get('helper_domains',plan['region_members']).items()}
    helper_owner={w:r for r,ws in domains.items() for w in ws}
    assert sorted(helper_owner)==list(range(27))
    producer={x['trajectory_id']:w for w,q in queues.items() for g in q for x in items if str(x['group_id'])==g}
    for move in load(folder/'pending-transfers.json'):
        assert helper_owner[move['src']]==helper_owner[move['dst']]
        assert move['bytes_per_lane']==sum(byid[t]['prompt_length']*4+64 for t in move['tids'])
        for t in move['tids']:
            assert producer[t]==move['src'] and move['start_tick']<=move['end_tick']<eos[t]
            producer[t]=move['dst']
    delivered={}
    for move in load(folder/'sample-transfers.json'):
        assert move['bytes_per_lane']==sum(4*(byid[t]['prompt_length']+byid[t]['response_length']) for t in move['tids'])
        for t in move['tids']:
            assert t not in delivered and producer[t]==move['src'] and group_region[str(byid[t]['group_id'])]==move['dst']
            assert eos[t]<=move['start_tick']<=move['end_tick']
            delivered[t]=move['end_tick']
    assert set(delivered)=={t for t in ids if producer[t]!=group_region[str(byid[t]['group_id'])]}
    events = load(folder/'training-events.json')
    for phase in ('old_policy', 'reference', 'training'):
        selected = [e for e in events if e['phase'] == phase and e['tids']]
        assert Counter(t for e in selected for t in e['tids']) == Counter(ids), phase
        for e in selected:
            assert e['start_tick'] <= e['end_tick']
            assert all(eos[t] <= e['start_tick'] for t in e['tids'])
            if phase in ('old_policy','reference'):
                assert all(delivered.get(t,eos[t])<=e['start_tick'] for t in e['tids'])
    regional = []
    for r, ws in regions.items():
        local = [t for t in ids if group_region[str(byid[t]['group_id'])] == r]
        own = [e for e in events if e['worker'] == r]
        first = min(e['start_tick'] for e in own)
        assert all(eos[t] <= first for t in local)
        if plan.get('helper_domains'):
            assert all(eos[t]<=first for t in ids if helper_owner[group_region[str(byid[t]['group_id'])]]==helper_owner[r])
        teacher = [e for e in own if e['phase'] in ('old_policy', 'reference')]
        advantage = next(e for e in own if e['phase'] == 'advantage')
        zero = next(e for e in own if e['phase'] == 'gradient_zero')
        assert all(e['end_tick'] <= advantage['start_tick'] for e in teacher)
        assert advantage['end_tick'] <= zero['start_tick']
        localtrain = [e for e in own if e['phase'] == 'training']
        assert all(zero['end_tick'] <= e['start_tick'] for e in localtrain)
        regional.append(dict(region=r,training_tp=2*len(ws),samples=len(local),
            first_prepare_s=first/1e12,teacher_s=sum(e['end_tick']-e['start_tick'] for e in teacher)/1e12,
            training_s=sum(e['end_tick']-e['start_tick'] for e in localtrain)/1e12,
            gradient_ready_s=max(e['end_tick'] for e in own)/1e12))
    batches, executions = load(folder/'training-batches.json'), load(folder/'training-batch-executions.json')
    assert Counter(e['batch_id'] for e in executions) == Counter(batches.keys())
    assert Counter(t for e in executions for t in e['tids']) == Counter(ids)
    ticketfile=folder/'training-ticket-events.json'
    if ticketfile.exists():
        tickets=load(ticketfile);assert Counter(x['batch_id'] for x in tickets)==Counter(batches.keys())
        executed_byid={x['batch_id']:x for x in executions}
        for ticket in tickets:
            e=executed_byid[ticket['batch_id']]
            assert ticket['region']==e['home'] and ticket['run_here']==(e['home']==e['worker'])
            if ticket['run_here']:assert ticket['tick']<=e['start_tick']
    moves = load(folder/'training-transfers.json')
    delivered = {m['batch_id']: m for m in moves}
    for e in executions:
        assert len(e['physical_ranks']) == e['training_tp']
        if e['worker'] != e['home']:
            move = delivered[e['batch_id']]
            assert move['end_tick'] <= e['start_tick'] and move['dst'] == e['worker']
            source_events=[x for x in events if x['worker']==e['home']]
            assert next(x for x in source_events if x['phase']=='advantage')['end_tick']<=move['start_tick']
            assert next(x for x in source_events if x['phase']=='gradient_zero')['end_tick']<=move['start_tick']
            assert move['bytes_per_lane']==sum(5*(byid[t]['prompt_length']+byid[t]['response_length'])+8*byid[t]['response_length']+68 for t in e['tids'])
            if plan.get('training_help_admission','global-eos')=='global-eos':assert max(eos.values())<=move['start_tick']
            assert e['training_tp'] == 2
            receiver_region=owner_region[e['worker']]
            restore=[x for x in events if x['worker'] in regions[receiver_region] and x['phase']=='gradient_restore']
            assert len(restore)==len(regions[receiver_region]) and max(x['end_tick'] for x in restore)<=e['start_tick']
    generation=load(folder/'generation-events.json')
    assert {x['tx'] for x in generation['migrations']}=={x['tx'] for x in generation['releases']}
    config=load(trace.parent/'config.json')
    for move in generation['migrations']:
        assert helper_owner[move['source']]==helper_owner[move['helper']] and move['weight_bytes_per_lane']==0
        assert sorted(h for lane in move['head_layout'] for h in lane['kv_heads'])==list(range(config['num_key_value_heads']))
        assert sorted(h for lane in move['head_layout'] for h in lane['query_heads'])==list(range(config['num_attention_heads']))
        assert move['context_sum']==sum(move['initial_actual_contexts'])
    # Reconstruct every teacher/training issued-program rolling hash from the
    # immutable operator bodies and actual event order, including remote jobs.
    compiled={int(f.stem.split('.')[0]):json.load(gzip.open(f,'rt')) for f in (folder/'compiled-regions').glob('*.json.gz')}
    jobsteps={j[3]['batch_id']:j for jobs in compiled.values() for j in jobs if j[0]=='training'}
    portable_path=folder/'portable-training-programs.json.gz'
    portable=json.load(gzip.open(portable_path,'rt')) if portable_path.exists() else {}
    for bid,p in portable.items():
        assert p['sha256']==digest(dict(repeat=1,steps=p['steps']))
        assert [t for sub in p['subbatches'] for t in sub]==batches[bid]['tids']
        assert p['peak_allocation']['peak_bytes']<=capacity['capacity_bytes']
        assert all(s['kind'] in ('local','AR') and not set(s)&{'ranks','group','src','dst'} for s in p['steps'])
    nativegroups=load(folder/'native/groups.json')
    execution_by_event={(e['worker'],tuple(e['tids']),e['start_tick']):e for e in executions}
    issued=load(folder/'issued-programs.json')
    for w in range(27):
        r=owner_region[w];jobs=compiled[r]
        for phase in ('old_policy','reference','training'):
            relevant=[e for e in events if e['worker']==w and e['phase'] in (('training','gradient_zero') if phase=='training' else (phase,))]
            bodies=[];cursor=0
            for e in sorted(relevant,key=lambda e:e['start_tick']):
                if e['phase']=='gradient_zero':body=next(j[2] for j in jobs if j[0]=='gradient_zero')
                elif phase=='training' and e['tids']:
                    ex=execution_by_event[(w,tuple(e['tids']),e['start_tick'])];job=jobsteps[ex['batch_id']]
                    assert digest(dict(repeat=1,steps=job[3]['steps']))==batches[ex['batch_id']]['program_sha256']
                    if ex['worker']==ex['home']:body=job[2]
                    elif ex['batch_id'] in portable:
                        body=portable[ex['batch_id']]['steps']
                        assert digest(dict(repeat=1,steps=body))==ex['program_sha256']
                    else:
                        body=[dict(s) for s in job[3]['steps']]
                        for s in body:
                            if s['kind']=='AR':s.pop('group',None)
                        if digest(dict(repeat=1,steps=body))!=ex['program_sha256']:
                            gid=next(int(g) for g,rs in nativegroups.items() if rs==[2*w,2*w+1])
                            for s in body:
                                if s['kind']=='AR':s['group']=gid
                        assert digest(dict(repeat=1,steps=body))==ex['program_sha256']
                else:
                    choices=[j for j in jobs if j[0]==phase]
                    if phase=='training':
                        local_ids={e['batch_id'] for e in executions if e['worker']==r and e['home']==r}
                        choices=[j for j in choices if j[3]['batch_id'] in local_ids]
                    body=choices[cursor][2];cursor+=1
                bodies.append(dict(repeat=1,steps=body))
            if bodies:
                assert issued[str(w)][phase]==dict(sha256=digest(bodies),segments=len(bodies)),(w,phase)
    proofs = load(folder/'gradient-restore-proof.json')
    for p in proofs:
        c = p['coordinate_descriptor']
        assert c['schema_digest'] == after['schema_digest'] and c['payload_elements']*4 == p['bytes']
        assert p['semantics'] in ('partial_sum', 'canonical_shard_concat')
        for s, t in zip(c['source'], c['target']):
            assert s['component_id'] == t['component_id'] and s['axis'] == t['axis']
            if s['axis'] is not None:
                assert t['start'] <= s['start'] < s['end'] <= t['end']
    native = load(folder/'native/result.json')
    assert result['native_exit_status'] == native['native_exit_status'] == 0
    assert native['native_coverage_and_dependencies_checked'] and len(native['native_sequence_summaries']) == 54
    ops = load(folder/'native/operations.json')
    rawevents={}
    pattern=re.compile(r'(issue|callback),sys->id=(\d+), tick=(\d+), node->id=(\d+),')
    for logfile in (folder/'native/astra_logs').glob('log*.log'):
        for action,rank,tick,node in pattern.findall(logfile.read_text()):
            key=(action,int(rank),int(node));assert key not in rawevents
            rawevents[key]=int(tick)
    expected={(action,r,o['id']) for o in ops for r in o['ranks'] for action in ('issue','callback')}
    assert set(rawevents)==expected
    for o in ops:
        for r in o['ranks']:
            assert rawevents['issue',r,o['id']]<=rawevents['callback',r,o['id']]
    for r in range(54):
        chain=[o['id'] for o in ops if r in o['ranks']]
        assert all(rawevents['callback',r,a]<=rawevents['issue',r,b] for a,b in zip(chain,chain[1:]))
    assert abs(max(rawevents.values())/1e12-result['total_s'])<1e-9
    assert result['binary_sha256']==native['binary_sha256']==cmd['native_binary_sha256']
    assert len([o for o in ops if 'publish_actor_v1' in o['name']]) == 1
    assert len([o for o in ops if o['kind'] == 'collective' and 'dp_gradient_RS' in o['name']]) == 2
    # Preserve owner ordinal for RS/AG. Physical AR rings may be reordered.
    for o in ops:
        if o['collective'] in ('REDUCE_SCATTER', 'ALL_GATHER') and 'dp_' in o['name']:
            assert o['ranks'] == list(range(o['ranks'][0], 54, 2))
    # Integrate dynamic roles only on the dynamic interval. Prefix/suffix are
    # separately labelled; final optimizer service must never count as idle.
    integrals = defaultdict(float)
    genend = max(eos.values())
    post_idle = 0
    for a, b in zip(rows, rows[1:]):
        dt = (b['tick']-a['tick'])/1e12
        counts = Counter(a['roles'])
        for role, n in counts.items():
            integrals[role] += 2*n*dt
        post_idle += 2*counts['idle']*max(0, b['tick']-max(a['tick'], genend))/1e12
    dynamic_end = max(x['end_tick'] for x in native['native_sequence_summaries'].values())/1e12
    report = dict(status='PASS_SIMULATION_EVIDENCE_AUDIT',model=result['model'],window=result['window'],
        samples=len(ids),regions=len(regions),total_s=result['total_s'],planning_proxy_s=plan.get('planning_wall_s',0),
        dynamic_origin_s=rows[0]['tick']/1e12,generation_end_s=genend/1e12,dynamic_end_s=dynamic_end,
        native_suffix_s=result['total_s']-dynamic_end,
        first_training_s=min(e['start_tick'] for e in events if e['phase']=='training' and e['tids'])/1e12,
        role_integral_die_s=dict(integrals),post_generation_idle_die_s=post_idle,
        observed_peak_bytes=max(x['peak_bytes'] for x in alloc),regional=regional,
        native_training_transfers=len(moves),remote_samples=sum(len(e['tids']) for e in executions if e['worker']!=e['home']),
        early_training_transfers=sum(m['start_tick']<max(eos.values()) for m in moves),
        remote_sequential_microbatches=sum(len(e.get('sequential_subbatches',[e['tids']])) for e in executions if e['worker']!=e['home']),
        remote_source_tp_counts=dict(Counter(e.get('source_training_tp',2) for e in executions if e['worker']!=e['home'])),
        reconstructed_issued_program_hashes=True,independent_raw_native_events=len(rawevents),
        limitations=['cost/native-event simulation; no Transformer tensor or learning-quality proof',
                     'role occupancy includes memory, communication and dependency waiting, not arithmetic utilization',
                     'prefix and suffix excluded from dynamic role integrals'])
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.folder.resolve());a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('regional','limitations','role_integral_die_s')}))
