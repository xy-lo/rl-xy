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

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_ROOT = ROOT/'src'
sys.path.insert(0,str(Path(__file__).resolve().parent/'legacy'))
sys.path.append(str(FRAMEWORK_ROOT))

from rl_xy.f2_runtime import INPUTS, ROOT as COST_ROOT, configure_native, make_operator, sha, verify_assets
from rl_xy.qwen3_experiment import Policy, weight_plan, optimizer_plan
from rl_xy.qwen3_cost import Cost
from rl_xy.qwen3_sequence import attach
from dynamic_native import execute
from online_rollout import Replica
from idea1_controller.native.adapter import NativeKernel as JointRuntime, LayeredNativeAdapter
from attention_tp8_program import attention_weight_bytes_per_helper
from gqa_cost import apply_generation_gqa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-root',type=Path,default=ROOT/'experiments/2026-11-02-idea1-joint')
    parser.add_argument('--training-scope',choices=('domain','wafer'),default='domain')
    parser.add_argument('--inference-layout',type=Path,required=True)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--resource-layout',type=Path,required=True)
    parser.add_argument('--history',type=Path)
    parser.add_argument('--workload-forecast',type=Path)
    parser.add_argument('--rollout-cap',type=int,default=0)
    parser.add_argument('--fast-cost-validation',type=Path)
    parser.add_argument('--window',choices=('tune','test1','test2','test3','test4'),default='test1')
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
    manifest = verify_assets(); trace = INPUTS/('Qwen3-'+args.model)/(args.window+'.json')
    assert sha(trace) == manifest['input_hashes'][args.model][args.window+'.json']
    original = json.loads(trace.read_text())['trajectories']
    requests = original[:args.subset_prompts*8] if args.subset_prompts else original
    if args.subset_prompts and not 1 <= args.subset_prompts <= 192:
        raise ValueError('invalid original subset')
    assignment=None
    if args.assignment:
        pass # subset is validated by exact canonical request digest
        from idea1_controller.history import digest
        assignment=json.loads(args.assignment.read_text())
        assert assignment['source_requests_digest']==digest(requests)
        assert assignment['model']==args.model and assignment['window']==args.window
        groups_by_id={str(x['group_id']):[] for x in requests}
        for x in requests:groups_by_id[str(x['group_id'])].append(x)
        assigned={int(w):[x for gid in gids for x in groups_by_id[gid]] for w,gids in assignment['queues'].items()}
        assert sorted(x['trajectory_id'] for xs in assigned.values() for x in xs)==sorted(x['trajectory_id'] for x in requests)
        assert max(map(len,assigned.values()))<=64
        (out/'assignment.json').write_text(json.dumps(assignment,indent=2)+'\n')
    resource_layout=json.loads(args.resource_layout.read_text())
    regions={int(r):tuple(ws) for r,ws in resource_layout['region_members'].items()}
    assert sorted(w for ws in regions.values() for w in ws)==list(range(27))
    assert all(r==min(ws) and len(ws)*2 in (2,4,8) for r,ws in regions.items())
    (out/'resource-layout.json').write_text(json.dumps(resource_layout,indent=2)+'\n')
    p = {'4B': Policy(2,2,4,8,32), '8B': Policy(2,2,2,1,16), '14B': Policy(2,2,1,1,16)}[args.model]
    from dataclasses import replace
    p=replace(p,cap=args.rollout_cap or p.cap)
    op = make_operator(args.model, 'wafer', cache_dir=out.parent/('mapper-cache-fast' if args.fast_cost_validation else 'mapper-cache'))
    if args.fast_cost_validation:
        from idea1_controller.cost_backend import install
        acceleration=install(op,out.parent/'cost-accelerator',json.loads(args.fast_cost_validation.read_text()))
        (out/'cost-accelerator-identity.json').write_text(json.dumps(acceleration,indent=2)+'\n')
    apply_generation_gqa(op)
    from idea1_controller.actor_views import ViewAdmission
    public_groups=[dict(group_id=str(x['group_id']),prompt_length=x['prompt_length']) for x in original[::8]]
    state=op.state(2,2);staged=args.model=='14B'
    token_cap=max(op.prompt_cap,max(x['prompt_length'] for x in original))+8192
    region_samples=max(sum(len(assigned[w]) for w in ws) for ws in regions.values())
    record_bound=2*(1536+max(64,region_samples))*token_cap*4
    inference_layout=json.loads(args.inference_layout.read_text())
    from idea1_controller.inference_groups import InferenceAdmission,validate_partition,inference_ranks
    infer_groups={int(r):tuple(ws) for r,ws in inference_layout['groups'].items()};validate_partition(infer_groups)
    infer_caps=inference_layout['caps']
    from idea1_controller.inference_groups import parameter_view_proof
    view_proof=parameter_view_proof(op.model,infer_groups,inference_layout.get('paired_tp1',False))
    (out/'inference-parameter-views.json').write_text(json.dumps(view_proof,indent=2)+'\n')
    certificates={str(r):InferenceAdmission(op,record_bound).evaluate(1 if inference_layout.get('paired_tp1') else len(ws)*2,int(infer_caps[str(r)]),public_groups) for r,ws in infer_groups.items()}
    capacity=ViewAdmission(op,record_bound).evaluate(p.cap,public_groups)
    capacity['passed']=all(c['passed'] for c in certificates.values())
    for key in ('generation','prefill','helper'):capacity['phase_peak_bytes'][key]=max(c['phase_peak_bytes'][key] for c in certificates.values())
    capacity['inference_group_certificates']=certificates
    (out/'inference-layout.json').write_text(json.dumps(inference_layout,indent=2)+'\n')
    (out/'capacity.json').write_text(json.dumps(capacity,indent=2)+'\n')
    if not capacity['passed']:raise ValueError('actor-view lifecycle/cap capacity failed')
    replicas = {w: Replica(op, (assigned[w] if assignment else requests[w::27]), (2*w, 2*w+1), p.cap) for w in range(27)}
    # Prefix and suffix are native operation DAGs around the dynamic sequences.
    from rl_xy.qwen3_native import Plan
    plan = Plan(op)
    plan.local(tuple(range(54)),Cost(compute=1000),'acquire_shared_actor_v0_TP2_views')
    if staged:
        for lane in range(2):
            plan.comm(tuple(range(lane,54,2)), 'ALL_GATHER', state['persistent_reference_shard_bytes'], 'early_restore_reference')
    if inference_layout.get('paired_tp1'):
        # Canonical TP2 actor halves remain resident; each die receives the
        # missing half into an extra read-only allocation before TP1 starts.
        actor_half=ViewAdmission(op,record_bound).memory.actor
        for w in range(27):
            ranks=replicas[w].ranks
            plan.local(ranks,op.vec(0,actor_half,0,'TP1_missing_actor_half_read'),'TP1_actor_read_'+str(w))
            plan.comm(ranks,'ALL_GATHER',actor_half,'TP1_restore_full_actor_'+str(w))
            plan.local(ranks,op.vec(0,0,actor_half,'TP1_missing_actor_half_write'),'TP1_actor_write_'+str(w))
        (out/'inference-storage.json').write_text(json.dumps(dict(tp=1,canonical_actor_bytes_per_die=actor_half,extra_actor_bytes_per_die=actor_half,
            lifecycle='extra half allocated before GEN and retired before local teacher/training; canonical TP2 owner view persists',
            control='two independent TP1 engines rendezvous at <=16-token pair safe points; all EOS records copied to both owners'),indent=2)+'\n')
    # The observed hot planner CPU pause is kept separate in metadata and
    # optionally injected once, on the common clock, never once per replica.
    if resource_layout.get('planning_wall_s',0):
        plan.local(tuple(range(54)),Cost(compute=resource_layout['planning_wall_s']*1e9),'exposed_Mac_CPU_planning_proxy')
    # Establish one common generation origin.
    plan.comm(tuple(range(54)), 'ALL_REDUCE', 54*4, 'prefix_global_ready')
    groups = {}
    for root,ws in {int(r):tuple(ws) for r,ws in resource_layout.get('helper_domains',resource_layout['region_members']).items()}.items():
        ranks=tuple(replicas[w].ranks[h] for h in range(2) for w in ws)
        if ranks not in plan.groupids:
            gid=str(len(plan.groups)+1);plan.groups[gid]=list(ranks);plan.groupids[ranks]=gid
        for src in ws:
            for dst in ws:
                if src==dst:continue
                ranks4=tuple(r for pair in zip(replicas[src].ranks,replicas[dst].ranks) for r in pair)
                if ranks4 not in plan.groupids:
                    gid=str(len(plan.groups)+1);plan.groups[gid]=list(ranks4);plan.groupids[ranks4]=gid
                groups[src,dst]=int(plan.groupids[ranks4])
    for w in range(27):
        ranks=(2*w,2*w+1)
        if ranks not in plan.groupids:
            gid=str(len(plan.groups)+1);plan.groups[gid]=list(ranks);plan.groupids[ranks]=gid
    frozen = None
    if not args.subset_prompts and not assignment:
        if args.window!='test1':raise ValueError('non-test1 requires a compiled assignment')
        baseline = args.evidence_root.resolve()/args.model
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
    from idea1_controller.native.inference_cohorts import CohortKernel as RegionKernel
    from idea1_controller.native.inference_cohorts import CohortAdapter as RegionAdapter
    ctrl = RegionKernel(op, replicas, requests, True, 16, True, True, group_ids=groups, early=not args.no_early, frozen=frozen, train_mb=p.train_mb, logprob_mb=p.logprob_mb, training_scope=args.training_scope)
    ctrl.configure_regions(regions,assigned,resource_layout)
    ctrl.region_communicators={r:int(plan.groupids[ranks]) for r,ranks in ctrl.region_ranks.items()}
    from idea1_controller.native.batch_catalog import FrozenBatchCatalog,OperatorBatchCatalog
    from rl_xy.communication_service import operator_cost_identity
    frozen_catalog=FrozenBatchCatalog(args.evidence_root.resolve()/args.model,original,replace(p,cap={'4B':32,'8B':16,'14B':16}[args.model])) if args.window=='test1' else None
    ctrl.batch_catalog=OperatorBatchCatalog(op,operator_cost_identity(op),out.parent/('batch-program-cache-'+args.model),frozen_catalog)
    ctrl.init_actor_views()
    ctrl.evidence_output=out
    ctrl.init_adaptive(record_bound)
    if bool(args.history)==bool(args.workload_forecast):
        raise ValueError('select exactly one helper information source')
    if args.workload_forecast:
        predicted=json.loads(args.workload_forecast.read_text())
        if digest(predicted)!=assignment.get('workload_forecast_digest'):
            raise ValueError('runtime forecast identity mismatch')
        if set(predicted)!={str(x['group_id']) for x in requests}:
            raise ValueError('runtime forecast must cover fixed batch')
        ctrl.init_workload_help(predicted)
        ctrl.resource_layout['helper_history_snapshot_id']=assignment['workload_distribution_version']
        ctrl.resource_layout['helper_prediction_kind']='calibration-distribution-forecast'
    else:
        frozen_history=json.loads(args.history.read_text())
        assert frozen_history['model']==args.model
        ctrl.init_history_help(frozen_history)
        ctrl.resource_layout['helper_history_snapshot_id']=frozen_history['snapshot_id']
    ctrl.resource_layout=dict(ctrl.resource_layout,execution_backend='complete-system-native-v1',gradient_arena='compact then restore after activation release',training_microbatch_max_by_region={str(r):(p.train_mb if len(ws)==1 else min(8,p.train_mb*len(ws)*2)) for r,ws in regions.items()})
    (out/'resource-layout.json').write_text(json.dumps(ctrl.resource_layout,indent=2)+'\n')
    ctrl.resource_layout['parameter_schema_sha256']=ctrl.coordinates.schema_digest
    ctrl.resource_layout['actor_storage']='immutable shared TP2 actor view'
    (out/'actor-views-before.json').write_text(json.dumps(ctrl.actor_views.manifest(),indent=2)+'\n')
    ctrl.resource_layout['assignment_forecasts']=assignment.get('forecasts',{})
    ctrl.resource_layout['assignment_long_quotas']=assignment.get('long_quota_by_region',{})
    ctrl.resource_layout['assignment_id']=assignment.get('plan_id',assignment.get('selected_candidate','manual-experiment-placement'))
    ctrl.resource_layout['fallback_group_ids']=assignment.get('fallback_group_ids',[])
    capacity['single_sample_phase_bounds']={k:capacity['phase_peak_bytes'][k] for k in ('teacher','training')}
    capacity['phase_peak_bytes'].update(teacher=capacity['capacity_bytes'],training=capacity['capacity_bytes'])
    capacity['training_arena_policy']='compact TPp working gradients; in-place TP2 expansion after activation release'
    capacity['observed_allocation_audit_required']=True
    inference_communicators={}
    for r,ws in infer_groups.items():
        ranks=inference_ranks(ws)
        if ranks not in plan.groupids:
            gid=str(len(plan.groups)+1);plan.groups[gid]=list(ranks);plan.groupids[ranks]=gid
        inference_communicators[r]=int(plan.groupids[ranks])
    from idea1_controller.inference_groups import elastic_rectangles
    ctrl.elastic_groups=elastic_rectangles();ctrl.elastic_communicators={}
    ctrl.elastic_enabled=inference_layout.get('elastic',False)
    ctrl.paired_tp1=inference_layout.get('paired_tp1',False)
    ctrl.elastic_data_wait=inference_layout.get('elastic_data_wait',False)
    ctrl.pending_enabled=inference_layout.get('pending_enabled',False)
    ctrl.pending_data_wait=inference_layout.get('pending_data_wait',False)
    if ctrl.paired_tp1 and ctrl.pending_enabled:raise ValueError('TP1 pending transfer is not supported')
    ctrl.inference_queue_override=inference_layout.get('execution_queues')
    if ctrl.paired_tp1 and ctrl.elastic_enabled:raise ValueError('TP1 policy has explicit pair-safe-point control, no mid-GEN TP1 resizing')
    for ws in ctrl.elastic_groups:
        ranks=inference_ranks(ws)
        if ranks not in plan.groupids:
            gid=str(len(plan.groups)+1);plan.groups[gid]=list(ranks);plan.groupids[ranks]=gid
        ctrl.elastic_communicators[ws]=int(plan.groupids[ranks])
    ctrl.init_inference(infer_groups,infer_caps,inference_communicators)
    p=replace(p,cap=max(map(int,infer_caps.values())))
    ctrl = RegionAdapter(ctrl,p,capacity,args.model,out)
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
    from idea1_controller.topology import optimize_ar_groups
    ar_groups=set(groups.values())|set(ctrl.region_communicators.values())|set(inference_communicators.values())|set(ctrl.elastic_communicators.values())
    owner_collectives={str(o.metadata.get('comm_group')) for o in plan.operations if o.kind=='collective' and o.collective!='ALL_REDUCE'}
    ar_groups={g for g in ar_groups if str(g) not in owner_collectives}
    ring_audit=optimize_ar_groups(plan,ar_groups)
    (out/'ring-placement.json').write_text(json.dumps(ring_audit,indent=2)+'\n')
    ctrl.bind_native_plan(plan)
    (out/'command.json').write_text(json.dumps(dict(argv=sys.argv, policy=p.__dict__,
        native_binary_sha256=sha(Path(os.environ['ASTRA_SIM_BIN'])),
        trace_path=str(trace), model=args.model, trace_sha256=sha(trace), canonical_requests_sha256=hashlib.sha256(json.dumps(requests, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        inference_tp_by_group={str(r):2*len(ws) for r,ws in infer_groups.items()},inference_cap_by_group=infer_caps,
        subset_prompts=args.subset_prompts, early=not args.no_early,
        gradient_normalization=dict(accumulation='sum of all assigned sample contributions', global_denominator=len(requests), home_samples=[len(x) for x in ctrl.groups]),
        generation_cost='original Idea1 grouped-core GQA wrapper; unchanged arithmetic/reorder and reduced-parallelism assumptions',
        python=sys.executable), indent=2)+'\n')
    from rl_xy.communication_service import operator_cost_identity
    (out/'operator_cost_identity.json').write_text(json.dumps(operator_cost_identity(op), indent=2)+'\n')
    source_paths = sorted((Path(__file__).parent).rglob('*.py'))
    source_paths += sorted((FRAMEWORK_ROOT/'idea1_controller').rglob('*.py'))
    source_paths += sorted((FRAMEWORK_ROOT/'idea1_controller').rglob('*.cpp'))
    source_paths += sorted((COST_ROOT/'src').rglob('*.py'))
    hashes = {str(f): sha(f) for f in source_paths}
    (out/'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    archive_map = {}
    for name in hashes:
        source = Path(name)
        if COST_ROOT in source.parents:rel = Path('cost-workspace')/source.relative_to(COST_ROOT)
        elif FRAMEWORK_ROOT in source.parents:rel = Path('framework')/source.relative_to(FRAMEWORK_ROOT)
        else:rel = Path('controller')/source.relative_to(Path(__file__).parent)
        target = out/'source-snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        archive_map[name] = str(target)
    (out/'source-archive-map.json').write_text(json.dumps(archive_map, indent=2)+'\n')
    cache = {}; consumers = {}; errors = []; stop = threading.Event()
    log = (out/'controller.jsonl').open('w'); occupancy = (out/'occupancy.jsonl').open('w')
    reply_log=gzip.open(out/'native-replies.jsonl.gz','wt')
    observed = set()
    progress_guard=[None,0,0]
    def decide(req):
        e = req['event']; w = e['worker']; phase = e['phase']; it = req['iteration']
        key = (w, phase, it)
        if key in cache:
            if req['rank'] in consumers.get(key,set()):
                raise RuntimeError('duplicate same-rank RPC before TP quorum: zero-time recursive injection')
            return cache[key]
        signature=(len(ctrl.done),len(ctrl.batch_executions),len(ctrl.moves),len(ctrl.trained),sum(len(s) for s in ctrl.received),len(ctrl.tx))
        if signature!=progress_guard[0] or req['tick']-progress_guard[1]>1000000:
            progress_guard[:]=[signature,req['tick'],0]
        else:progress_guard[2]+=1
        if progress_guard[2]>10000:
            state=dict(signature=signature,tick=req['tick'],waiting=ctrl.waiting,stages=ctrl.stages,
                regions={str(r):dict(drained=ctrl.domain_drained(r),expected=len(ctrl.groups[r]),received=len(ctrl.received[r])) for r in ctrl.region_members},
                leases=[dict(source=t['source'],helper=t['helper'],stage=t['stage'],issued=t.get('helper_issued_epoch'),acked=t.get('helper_acked_epoch')) for t in ctrl.tx.values()])
            (out/'livelock-state.json').write_text(json.dumps(state,indent=2)+'\n')
            raise RuntimeError('native progress watchdog: repeated callbacks without device/data/training progress')
        result = ctrl.dispatch(req)
        reply_log.write(json.dumps(dict(worker=w,phase=phase,iteration=it,tick=req['tick'],reply=result),separators=(',',':'))+'\n');reply_log.flush()
        cache[key] = result
        log.write(json.dumps(dict(tick=req['tick'], worker=w, phase=phase, iteration=it,
            completed=len(ctrl.done), trained=len(ctrl.trained), transfers=len(ctrl.moves)))+'\n'); log.flush()
        roles = ctrl.roles(); new_done = sorted(ctrl.done-observed); observed.update(new_done)
        occupancy.write(json.dumps(dict(tick=req['tick'], worker=w, phase=phase, roles=roles, stages=ctrl.stages, new_done=new_done), separators=(',', ':'))+'\n'); occupancy.flush()
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
        stop.set(); thread.join(2); server.close(); temp.cleanup(); log.close(); occupancy.close();reply_log.close()
    if errors:
        raise RuntimeError(errors[0])
    ctrl.on_native_complete(native_result)
    ctrl.audit()
    assert ctrl.returned == set(replicas)
    assert all(sha(Path(f)) == h for f, h in hashes.items())
    (out/'inference-resizes.json').write_text(json.dumps(ctrl.inference_resizes,indent=2)+'\n')
    (out/'inference-role-events.json').write_text(json.dumps(ctrl.inference_role_events,indent=2)+'\n')
    (out/'inference-releases.json').write_text(json.dumps(ctrl.inference_releases,indent=2)+'\n')
    (out/'inference-program-events.json').write_text(json.dumps(ctrl.inference_program_events,indent=2)+'\n')
    (out/'training-events.json').write_text(json.dumps(ctrl.events, separators=(',', ':'))+'\n')
    (out/'sample-transfers.json').write_text(json.dumps(ctrl.moves, separators=(',', ':'))+'\n')
    (out/'gradient-zero-programs.json').write_text(json.dumps(ctrl.gradient_zero_programs,indent=2)+'\n')
    (out/'training-batches.json').write_text(json.dumps(ctrl.batch_manifest,indent=2)+'\n')
    (out/'training-ticket-events.json').write_text(json.dumps(getattr(ctrl,'training_ticket_events',[]),separators=(',',':'))+'\n')
    with gzip.open(out/'portable-training-programs.json.gz','wt') as stream:
        json.dump(getattr(ctrl,'portable_jobs',{}),stream,separators=(',',':'))
    (out/'training-batch-executions.json').write_text(json.dumps(ctrl.batch_executions,indent=2)+'\n')
    (out/'training-transfers.json').write_text(json.dumps(ctrl.train_moves,indent=2)+'\n')
    (out/'issued-programs.json').write_text(json.dumps(ctrl.program_proof(), indent=2)+'\n')
    (out/'generation-events.json').write_text(json.dumps(dict(migrations=ctrl.migrations, helper_runs=ctrl.runs, releases=ctrl.releases), separators=(',', ':'))+'\n')
    result = dict(status='PASS_NATIVE_JOINT_CONDITIONAL_SIMULATION', model='Qwen3-'+args.model, window=args.window,
                  implementation='actual-inference-cohorts-TP2-4-8',requests=len(requests), completed=len(ctrl.done), trained_samples=sum(map(len, ctrl.train_done.values())),
                  training_scope=args.training_scope,training_batches=len(ctrl.batch_manifest),remotely_executed_batches=sum(e['home']!=e['worker'] for e in ctrl.batch_executions),training_transfers=len(ctrl.train_moves),worker_trained_samples={str(w):len(ctrl.train_done[w]) for w in range(27)},trained_workers=len(ctrl.trained), sample_transfers=len(ctrl.moves),
                  helper_segments=len(ctrl.runs), helper_migrations=len(ctrl.migrations),
                  generation_end_tick=ctrl.generation_end_tick, total_s=native_result['model_ns']/1e9,
                  binary_sha256=native_result['binary_sha256'], native_exit_status=native_result['native_exit_status'],
                  capacity_passed=capacity['passed'], wall_s=time.monotonic()-start,
                  subset_prompts=args.subset_prompts, early=not args.no_early,
                  limitations=['trace-driven computation/communication costs, no numerical tensors',
                               'reference lifetime recorded per model; no hidden reconstruction cost',
                               'unstarted frozen training batches may move within configured scope after generation drains',
                               'GPU local calibration/extrapolation; no physical wafer validation'])
    (out/'allocation-events.json').write_text(json.dumps(ctrl.actual_allocation_events,indent=2)+'\n')
    (out/'microbatch-choices.json').write_text(json.dumps(ctrl.mb_choices,indent=2)+'\n')
    result['observed_training_allocation_peak_bytes']=max(x['peak_bytes'] for x in ctrl.actual_allocation_events)
    capacity['observed_teacher_training_peak_bytes']=result['observed_training_allocation_peak_bytes']
    capacity['allocation_certificate_count']=len(ctrl.actual_allocation_events)
    (out/'capacity.json').write_text(json.dumps(capacity,indent=2)+'\n')
    (out/'operator-reuse.json').write_text(json.dumps(dict(hits=ctrl.batch_catalog.hits,misses=ctrl.batch_catalog.misses,scope='native operators only; no durations reused'),indent=2)+'\n')
    (out/'actor-views-after.json').write_text(json.dumps(ctrl.actor_views.manifest(),indent=2)+'\n')
    (out/'helper-completion-quorums.json').write_text(json.dumps(ctrl.helper_completion_quorums,indent=2)+'\n')
    (out/'pending-transfers.json').write_text(json.dumps(ctrl.pending_transfers,indent=2)+'\n')
    result['pending_transfers']=len(ctrl.pending_transfers)
    (out/'gradient-restore-proof.json').write_text(json.dumps(ctrl.restore_proofs,indent=2)+'\n')
    result['inference_instance_count']=54 if inference_layout.get('paired_tp1') else len(infer_groups)
    result['inference_tp_by_group']={str(r):(1 if inference_layout.get('paired_tp1') else 2*len(ws)) for r,ws in infer_groups.items()}
    result['training_region_count']=len(regions)
    result['training_tp_by_region']={str(r):len(ws)*2 for r,ws in regions.items()}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
