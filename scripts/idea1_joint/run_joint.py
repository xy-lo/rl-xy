#!/usr/bin/env python3
"""Reproducible H0 4B/8B native joint experiment, complete fixed test1 by default; frozen per-model microbatches."""
import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import shutil
import gzip
from pathlib import Path

from rl_xy.f2_runtime import INPUTS, ROOT as COST_ROOT, configure_native, make_operator, sha, verify_assets
from rl_xy.qwen3_experiment import Policy, weight_plan, optimizer_plan
from rl_xy.qwen3_cost import Cost
from rl_xy.qwen3_sequence import attach
from dynamic_native import execute
from online_rollout import Replica
from joint_runtime import JointRuntime
from attention_tp8_program import attention_weight_bytes_per_helper
from gqa_cost import apply_generation_gqa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('4B','8B','14B'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--subset-prompts', type=int, default=0)
    parser.add_argument('--no-early', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    native = args.native_root.resolve()
    os.environ['ASTRA_SIM_ROOT'] = str(native)
    os.environ['ASTRA_SIM_BIN'] = str(native/'build-dynarl/bin/AstraSim_Analytical_Congestion_Aware')
    configure_native()
    manifest = verify_assets(); trace = INPUTS/('Qwen3-'+args.model)/'test1.json'
    assert sha(trace) == manifest['input_hashes'][args.model]['test1.json']
    original = json.loads(trace.read_text())['trajectories']
    requests = original[:args.subset_prompts*8] if args.subset_prompts else original
    if args.subset_prompts and not 1 <= args.subset_prompts <= 192:
        raise ValueError('invalid original subset')
    p = {'4B': Policy(2,2,4,8,32), '8B': Policy(2,2,2,1,16), '14B': Policy(2,2,1,1,16)}[args.model]
    op = make_operator(args.model, 'wafer', cache_dir=out.parent/'mapper-cache')
    apply_generation_gqa(op)
    admission = op.admission(original, p.train_tp, p.rollout_tp, p.train_mb, p.logprob_mb, p.cap)
    assert admission['passed']
    state = op.state(2, 2)
    staged = args.model == '14B'
    extra_ref = state['persistent_reference_shard_bytes']*26 if staged else 0
    scratch = p.cap*(op.model.hidden_size*32+op.model.ffn_size*16)+1024**2
    nic = op.hw.nic_tx_bytes_per_die+op.hw.nic_rx_bytes_per_die
    # Each producer retains token records until every receiver has a copy.
    # A destination has its own producer records plus the assigned training set.
    # Double this sum to conservatively reserve transfer staging as well.
    token_cap = max(op.prompt_cap, max(x['prompt_length'] for x in original))+8192
    record_bound = 2*(64+57)*token_cap*4
    factor = op.model.layers*2*(op.model.kv_heads//4)*op.model.head_dim*2
    kv_bound = max(sum(sorted((x['prompt_length']+8192 for x in original[i::27]), reverse=True)[:p.cap])*factor for i in range(27))
    peaks = dict(
        weight_export=admission['phase_peak_bytes']['weight_export']+record_bound,
        generation=admission['phase_peak_bytes']['generation']+extra_ref+scratch+nic+record_bound,
        helper=state.get('generation_base_max',state['training_base_max'])+state['rollout_weights_per_rank']+kv_bound+attention_weight_bytes_per_helper(op)+extra_ref+scratch+nic+record_bound,
        teacher=admission['phase_peak_bytes']['logprob']+record_bound+nic,
        training=admission['phase_peak_bytes']['training']+record_bound+nic)
    capacity = dict(passed=max(peaks.values()) <= op.capacity, capacity_bytes=op.capacity,
                    phase_peak_bytes=peaks, record_and_staging_bound_bytes=record_bound,
                    reference_layout="staged reference reconstruction" if staged else "full resident actor/reference/gradient", original_admission=admission,
                    lifecycle=['full actor/reference/gradient state resident as frozen baseline',
                               'weight export with original admitted staging',
                               'domain-drained: release KV/helper and rollout copies',
                               'old/reference with frozen model-specific microbatch sizes',
                               'gradient zero; all sample gradients summed',
                               'global sample denominator; DP reduction; one optimizer update'],
                    limitations=['formula upper bounds, no physical tensor allocation'])
    if staged:
        capacity['phase_peak_bytes']['reference_reconstruction'] = admission['phase_peak_bytes']['logprob']+record_bound+nic
        capacity['lifecycle'] = ['reference shards resident initially', 'two native AG before generation, full reference retained', 'domain drained, data delivered, old/reference evaluation', 'release full reference before gradient allocation', 'all sample gradients summed, one global update']
    capacity['passed'] = max(capacity['phase_peak_bytes'].values()) <= op.capacity
    (out/'capacity.json').write_text(json.dumps(capacity, indent=2)+'\n')
    if not capacity['passed']:
        raise ValueError('joint lifecycle capacity failed')
    replicas = {w: Replica(op, requests[w::27], (2*w, 2*w+1), p.cap) for w in range(27)}
    # Prefix and suffix are native operation DAGs around the dynamic sequences.
    plan = weight_plan(op, p)
    if staged:
        for lane in range(2):
            plan.comm(tuple(range(lane,54,2)), 'ALL_GATHER', state['persistent_reference_shard_bytes'], 'early_restore_reference')
    # Establish one common generation origin.
    plan.comm(tuple(range(54)), 'ALL_REDUCE', 54*4, 'prefix_global_ready')
    groups = {}
    for src in range(27):
        for dst in range(src//3*3, src//3*3+3):
            if src == dst:
                continue
            ranks = tuple(r for pair in zip(replicas[src].ranks, replicas[dst].ranks) for r in pair)
            gid = str(len(plan.groups)+1); plan.groups[gid] = list(ranks); plan.groupids[ranks] = gid
            groups[src, dst] = int(gid)
    frozen = None
    if not args.subset_prompts:
        baseline = Path(__file__).resolve().parents[2]/'experiments/2026-11-02-idea1-joint'/args.model
        metadata = json.loads((baseline/'baseline-metadata.json').read_text())
        baseline_result = metadata['result_fields']
        canonical = hashlib.sha256(json.dumps(requests, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert baseline_result['trajectory_sha256'] == canonical
        assert baseline_result['policy'] == p.__dict__
        source_manifest = {'files':metadata['cost_source_files']}
        source_root = COST_ROOT
        checked = {}
        for name, expected in source_manifest['files'].items():
            if name.split('/')[0] in ('src', 'configs', 'docker') and name != 'src/rl_xy/qwen3_source.py':
                assert sha(source_root/name) == expected, name
                checked[name] = expected
        frozen = {}; evidence = {}
        for stage in ('old_policy', 'reference', 'training'):
            path = baseline/(stage+'-program.json.gz')
            with gzip.open(path,'rt') as stream: value = json.load(stream)
            frozen[stage] = {w: value['programs'][value['ranks'][str(2*w)]]['segments'] for w in range(27)}
            evidence[stage] = dict(path=str(path), sha256=sha(path))
        # Independently recompile two original samples as a serialization and
        # cost-contract check, in addition to complete source/input identities.
        from rl_xy.qwen3_sequence import compile_blocks
        from rl_xy.qwen3_experiment import teacher_lengths
        from rl_xy.qwen3_experiment import chunks
        for i in (0, 1):
            logbatch = chunks(requests[:64], p.logprob_mb)[i]
            trainbatch = chunks(requests[:64], p.train_mb)[i]
            assert compile_blocks(op, (0,1), op.forward(teacher_lengths(logbatch), 2)) == frozen['old_policy'][0][i]['steps']
            assert compile_blocks(op, (0,1), op.training(teacher_lengths(trainbatch), 2, sum(x['response_length'] for x in trainbatch))) == frozen['training'][0][i+1]['steps']
        (out/'frozen-program-reuse.json').write_text(json.dumps(dict(programs=evidence, checked_source_files=checked,
            independently_recompiled_samples=[requests[i]['trajectory_id'] for i in (0, 1)],
            scope='reuse compiled per-batch operators only; all native computation and network events execute again'), indent=2)+'\n')
    ctrl = JointRuntime(op, replicas, requests, True, 16, True, True, group_ids=groups, early=not args.no_early, frozen=frozen, train_mb=p.train_mb, logprob_mb=p.logprob_mb)
    seq = attach(plan, [(r.ranks, [{'repeat': 10000000, 'steps': [ctrl.rpc(w, 'arrive')]}]) for w, r in replicas.items()])
    suffix = optimizer_plan(op, p)
    for operation in suffix.operations[1:]:
        if operation.kind == 'collective':
            ranks = operation.participant_ranks
            if ranks not in plan.groupids:
                gid = str(len(plan.groups)+1); plan.groups[gid] = list(ranks); plan.groupids[ranks] = gid
            operation = __import__('dataclasses').replace(operation, metadata={'comm_group': plan.groupids[ranks]})
        added = plan.append(operation)
        if operation.kind == 'compute':
            plan.stats[added.node_id] = suffix.stats[operation.node_id]
    plan.local(tuple(range(54)), Cost(compute=1), 'publish_actor_v1_reference_v0')
    (out/'command.json').write_text(json.dumps(dict(argv=sys.argv, policy=p.__dict__,
        native_binary_sha256=sha(Path(os.environ['ASTRA_SIM_BIN'])),
        trace_path=str(trace), model=args.model, trace_sha256=sha(trace), canonical_requests_sha256=hashlib.sha256(json.dumps(requests, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        subset_prompts=args.subset_prompts, early=not args.no_early,
        gradient_normalization=dict(accumulation='sum of all assigned sample contributions', global_denominator=len(requests), worker_samples=[len(x) for x in ctrl.groups]),
        generation_cost='original Idea1 grouped-core GQA wrapper; unchanged arithmetic/reorder and reduced-parallelism assumptions',
        python=sys.executable), indent=2)+'\n')
    from rl_xy.communication_service import operator_cost_identity
    (out/'operator_cost_identity.json').write_text(json.dumps(operator_cost_identity(op), indent=2)+'\n')
    source_paths = sorted((Path(__file__).parent).glob('*.py'))
    source_paths += sorted((COST_ROOT/'src').rglob('*.py'))
    hashes = {str(f): sha(f) for f in source_paths}
    (out/'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    archive_map = {}
    for name in hashes:
        source = Path(name)
        rel = Path('cost-workspace')/source.relative_to(COST_ROOT) if COST_ROOT in source.parents else Path('controller')/source.relative_to(Path(__file__).parent)
        target = out/'source-snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        archive_map[name] = str(target)
    (out/'source-archive-map.json').write_text(json.dumps(archive_map, indent=2)+'\n')
    cache = {}; consumers = {}; errors = []; stop = threading.Event()
    log = (out/'controller.jsonl').open('w'); occupancy = (out/'occupancy.jsonl').open('w')
    observed = set()
    def decide(req):
        e = req['event']; w = e['worker']; phase = e['phase']; it = req['iteration']
        key = (w, phase, it)
        if key in cache:
            return cache[key]
        ctrl.tick = req['tick']
        if phase == 'arrive':
            gate = f'joint-worker-{w}-{it}'
            result = {'steps': [{'kind': 'gate_signal', 'key': gate},
                               {'kind': 'gate_wait', 'key': gate, 'threshold': 2}, ctrl.rpc(w, 'step')]}
        elif phase in ('step', 'resume', 'attention_helper_ready'):
            result = ctrl.step(w, it)
        elif phase == 'sample_run':
            result = ctrl.sample_run(w)
        elif phase == 'sample_done':
            result = ctrl.sample_done(w, e['transfer'])
        elif phase == 'attention_migrate_source':
            result = ctrl.migration_source(w)
        elif phase == 'attention_source_ready':
            result = ctrl.source_ready(w, it)
        elif phase == 'attention_source':
            result = ctrl.attention_source(w)
        elif phase == 'deferred':
            result = ctrl.deferred.pop(w)
        elif phase == 'joint_finish':
            ctrl.returned.add(w); result = {'steps': [], 'finish': True}
        else:
            raise ValueError('unknown phase '+phase)
        cache[key] = result
        log.write(json.dumps(dict(tick=req['tick'], worker=w, phase=phase, iteration=it,
            completed=len(ctrl.done), trained=len(ctrl.trained), transfers=len(ctrl.moves)))+'\n'); log.flush()
        roles = ctrl.roles(); new_done = sorted(ctrl.done-observed); observed.update(new_done)
        occupancy.write(json.dumps(dict(tick=req['tick'], worker=w, phase=phase, roles=roles, new_done=new_done), separators=(',', ':'))+'\n'); occupancy.flush()
        return result
    temp = tempfile.TemporaryDirectory(prefix='t125-')
    os.environ['RLXY_CONTROLLER_SOCKET'] = str(Path(temp.name)/'s')
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(os.environ['RLXY_CONTROLLER_SOCKET']); server.listen(54); server.settimeout(.2)
    def serve():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    data = b''
                    while not data.endswith(b'\n'):
                        part = conn.recv(65536)
                        if not part:
                            raise ValueError('short RPC')
                        data += part
                    req = json.loads(data); reply = decide(req)
                    conn.sendall((json.dumps(reply, separators=(',', ':'))+'\n').encode())
                    e = req['event']; key = (e['worker'], e['phase'], req['iteration'])
                    consumers.setdefault(key, set()).add(req['rank'])
                    if len(consumers[key]) == 2:
                        consumers.pop(key); cache.pop(key, None)
                        if e['phase'] in ('step', 'resume'):
                            replicas[e['worker']].cache.pop(req['iteration'], None)
                except Exception:
                    errors.append(traceback.format_exc()); (out/'controller-error.log').write_text(errors[-1]); break
    thread = threading.Thread(target=serve, daemon=True); thread.start(); start = time.monotonic()
    try:
        native_result = execute(plan, out/'native', timeout=7200, sequence=seq)
    finally:
        stop.set(); thread.join(2); server.close(); temp.cleanup(); log.close(); occupancy.close()
    if errors:
        raise RuntimeError(errors[0])
    ctrl.audit()
    assert ctrl.returned == set(replicas)
    assert all(sha(Path(f)) == h for f, h in hashes.items())
    (out/'training-events.json').write_text(json.dumps(ctrl.events, separators=(',', ':'))+'\n')
    (out/'sample-transfers.json').write_text(json.dumps(ctrl.moves, separators=(',', ':'))+'\n')
    (out/'issued-programs.json').write_text(json.dumps(ctrl.program_proof(), indent=2)+'\n')
    (out/'generation-events.json').write_text(json.dumps(dict(migrations=ctrl.migrations, helper_runs=ctrl.runs, releases=ctrl.releases), separators=(',', ':'))+'\n')
    result = dict(status='PASS_NATIVE_JOINT_CONDITIONAL_SIMULATION', model='Qwen3-'+args.model, window='test1',
                  requests=len(requests), completed=len(ctrl.done), trained_samples=sum(map(len, ctrl.train_done.values())),
                  trained_workers=len(ctrl.trained), sample_transfers=len(ctrl.moves),
                  helper_segments=len(ctrl.runs), helper_migrations=len(ctrl.migrations),
                  generation_end_tick=ctrl.generation_end_tick, total_s=native_result['model_ns']/1e9,
                  binary_sha256=native_result['binary_sha256'], native_exit_status=native_result['native_exit_status'],
                  capacity_passed=capacity['passed'], wall_s=time.monotonic()-start,
                  subset_prompts=args.subset_prompts, early=not args.no_early,
                  limitations=['trace-driven computation/communication costs, no numerical tensors',
                               'reference lifetime recorded per model; no hidden reconstruction cost',
                               'training waits for own six-die domain generation to drain',
                               'GPU local calibration/extrapolation; no physical wafer validation'])
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
