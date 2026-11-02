"""Post-run audit independent of scheduling, using immutable native evidence."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
digest = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit(folder):
    result = load(folder/'result.json'); command = load(folder/'command.json')
    full = result['subset_prompts'] == 0
    model = result['model'].removeprefix('Qwen3-')
    source = Path(command.get('trace_path',ROOT/'experiments/2026-10-29-qwen3/inputs'/('Qwen3-'+model)/'test1.json'))
    config = load(source.parent/'config.json')
    layers = config['num_hidden_layers']; kv_heads = config['num_key_value_heads']; q_heads = config['num_attention_heads']; head_dim = config.get('head_dim', config['hidden_size']//q_heads)
    kv_factor = layers*2*(kv_heads//4)*head_dim*2
    requests = load(source)['trajectories']
    if not full:
        requests = requests[:result['subset_prompts']*8]
    items = {x['trajectory_id']: x for x in requests}; ids = set(items)
    assert set(Counter(x['group_id'] for x in requests).values()) == {8}
    assert len(ids) == len(requests) == result['requests'] == result['completed'] == result['trained_samples']
    assert result['native_exit_status'] == 0 and result['trained_workers'] == 27
    assert sha(source) == command['trace_sha256']
    assert digest(requests) == command['canonical_requests_sha256']
    archive = load(folder/'source-archive-map.json')
    assert all(sha(Path(archive[p])) == h for p, h in load(folder/'source_hashes.json').items())
    capacity = load(folder/'capacity.json')
    assert capacity['passed'] and max(capacity['phase_peak_bytes'].values()) <= 64000000000
    stages = load(folder/'training-events.json'); moves = load(folder/'sample-transfers.json')
    first_eos = {}; rows = []; last_tick = -1
    for line in (folder/'occupancy.jsonl').open():
        row = json.loads(line)
        assert row['tick'] >= last_tick
        last_tick = row['tick']
        assert len(row['roles']) == 27
        assert set(row['roles']) <= {'own', 'helper', 'prepare', 'training', 'idle', 'sample_transfer'}
        for tid in row['new_done']:
            assert tid in ids and tid not in first_eos
            first_eos[tid] = row['tick']
        rows.append(row)
    assert set(first_eos) == ids
    delivered = {}; transfer_tids = []
    owners = {x['trajectory_id']: i % 27 for i, x in enumerate(requests)}
    for move in moves:
        assert move['start_tick'] <= move['end_tick']
        assert move['bytes_per_lane'] == sum((items[t]['prompt_length']+items[t]['response_length'])*4 for t in move['tids'])
        for tid in move['tids']:
            assert owners[tid] == move['src'] and first_eos[tid] <= move['start_tick']
            assert tid not in delivered
            delivered[tid] = move['end_tick']; transfer_tids.append(tid)
    for phase in ('old_policy', 'reference', 'training'):
        selected = [e for e in stages if e['phase'] == phase]
        tids = [t for e in selected for t in e['tids']]
        assert len(tids) == len(set(tids)) and set(tids) == ids
        for event in selected:
            assert event['end_tick'] >= event['start_tick']
            for tid in event['tids']:
                assert first_eos[tid] <= event['start_tick']
                if owners[tid] != event['worker']:
                    assert delivered[tid] <= event['start_tick']
    # Reference must complete before FP32 gradient allocation on each worker.
    for w in range(27):
        own = [e for e in stages if e['worker'] == w]
        zero = next(e for e in own if e['phase'] == 'gradient_zero')
        assert all(e['end_tick'] <= zero['start_tick'] for e in own if e['phase'] == 'reference')
        teacher = [e for e in own if e['phase'] == 'old_policy']
        if teacher:
            begin = min(e['start_tick'] for e in teacher)
            assigned = {t for e in teacher for t in e['tids']}
            assert all(first_eos[t] <= begin for t in assigned)
            assert all(delivered[t] <= begin for t in assigned if owners[t] != w)
            domain_tids = {t for t in ids if owners[t]//3 == w//3}
            assert all(first_eos[t] <= begin for t in domain_tids)
    native = load(folder/'native/result.json')
    assert native['native_coverage_and_dependencies_checked'] and native['native_exit_status'] == 0
    assert len(native['native_sequence_summaries']) == 54
    operations = load(folder/'native/operations.json')
    restore = [o for o in operations if o['kind'] == 'collective' and 'early_restore_reference' in o['name']]
    assert len(restore) == (2 if model=='14B' else 0)
    assert len([o for o in operations if 'publish_actor_v1' in o['name']]) == 1
    assert len([o for o in operations if o['kind'] == 'collective' and 'dp_gradient_RS' in o['name']]) == 2
    generation = load(folder/'generation-events.json')
    assert len(generation['migrations']) == len(generation['releases'])
    assert {x['tx'] for x in generation['migrations']} == {x['tx'] for x in generation['releases']}
    for migration in generation['migrations']:
        assert migration['source']//3 == migration['helper']//3
        assert sorted(h for lane in migration['head_layout'] for h in lane['kv_heads']) == list(range(kv_heads))
        assert sorted(h for lane in migration['head_layout'] for h in lane['query_heads']) == list(range(q_heads))
    prior = {}
    for migration in generation['migrations']:
        tids = migration['active_ids']; contexts = migration['initial_actual_contexts']
        assert len(tids) == len(set(tids)) == len(contexts) == migration['batch']
        assert all(owners[t] == migration['source'] and c >= items[t]['prompt_length'] for t, c in zip(tids, contexts))
        assert migration['context_sum'] == sum(contexts)
        # 40 layers, two KV heads/lane at TP4, head dim128, BF16.
        assert migration['kv_bytes_per_lane'] == sum(contexts)*kv_factor
        prior[migration['tx']] = dict(zip(tids, contexts))
    for segment in generation['helper_runs']:
        before = prior[segment['tx']]
        tids = segment['active_ids']; contexts = segment['actual_contexts']
        assert len(tids) == len(contexts) == segment['batch']
        assert set(tids) <= set(before) and all(before[t] == c for t, c in zip(tids, contexts))
        assert segment['kv_append_bytes_per_rank_per_step'] == len(tids)*kv_factor
        prior[segment['tx']] = {t:c+segment['repeat'] for t,c in zip(tids,contexts)}
    if full:
        baseline = ROOT/'experiments/2026-11-02-idea1-joint'/model
        import gzip
        programs = load(folder/'issued-programs.json')
        for phase in ('old_policy','reference','training'):
            with gzip.open(baseline/(phase+'-program.json.gz'),'rt') as stream: expected=json.load(stream)
            for w in range(27):
                key=expected['ranks'][str(2*w)]; frozen=expected['programs'][key]['segments']; ours=programs[str(w)][phase]
                assert len(frozen)==ours['segments'] and digest(frozen)==ours['sha256'],(phase,w)
        assert command['policy']==load(baseline/'baseline-metadata.json')['result_fields']['policy']
    origin = min(r['tick'] for r in rows)
    gen_end = max(first_eos.values())
    integrals = defaultdict(float)
    samples = []
    for i, row in enumerate(rows):
        end = rows[i+1]['tick'] if i+1 < len(rows) else row['tick']
        counts = Counter(row['roles'])
        assert sum(2*n for n in counts.values()) == 54
        seconds = (end-row['tick'])/1e12
        for role, n in counts.items():
            integrals[role] += 2*n*seconds
        samples.append(dict(step_s=row['tick']/1e12, generation_clock_s=(row['tick']-origin)/1e12,
            **{r+'_dies': 2*counts[r] for r in ('own', 'helper', 'prepare', 'training', 'sample_transfer', 'idle')}))
    with (folder/'die-timeline.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0])); writer.writeheader(); writer.writerows(samples)
    report = dict(status='PASS_EVIDENCE_AUDIT', model=model, full_test1=full, requests=len(ids),
        exact_frozen_teacher_training_program_parity=full, peak_bytes=max(capacity['phase_peak_bytes'].values()),
        prefix_until_generation_s=origin/1e12, generation_s=(gen_end-origin)/1e12,
        first_prepare_s=min(e['start_tick'] for e in stages if e['phase'] == 'old_policy' and e['tids'])/1e12,
        first_training_s=min(e['start_tick'] for e in stages if e['phase'] == 'training')/1e12,
        generation_end_s=gen_end/1e12,
        training_end_s=max(e['end_tick'] for e in stages if e['phase'] == 'training')/1e12,
        training_overlap_generation_die_s=2*sum(max(0, min(e['end_tick'], gen_end)-max(e['start_tick'], origin)) for e in stages if e['phase'] == 'training')/1e12,
        total_s=result['total_s'], role_integral_die_s=dict(integrals),
        sample_transferred_bytes_all_lanes=2*sum(m['bytes_per_lane'] for m in moves),
        note='Role occupancy includes local memory/network/waiting; not core arithmetic utilization. Integrals cover dynamic sequence interval, excluding prefix/suffix.')
    (folder/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    audit(Path(sys.argv[1]).resolve())
