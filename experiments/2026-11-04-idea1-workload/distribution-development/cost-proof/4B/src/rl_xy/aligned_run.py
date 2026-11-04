"""Native conditional evidence for the selected PP1/DO veRL protocol.

Global hooks validate the submitted macro/arena graph. Component logical
objects are separately replayed at clocks expanded from observed native events;
those logical events are derived, never mislabeled original native callbacks.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import sys

from .aligned_graph import build_aligned_graph
from .astra_adapter import generate_traces, parse_astra_output, run_astrasim
from .baseline_contract import BaselinePolicy, audit_policy, policy_fingerprint
from .capacity_mapping import CapacityAwareCostProvider
from .config import load_case
from .component_proof_cache import ComponentProofCache
from .evidence_io import write_minified_json, write_gzip_json, phase_deadline, write_aligned_tables
from .gpu_model import load_gpu_profile, effective_gpu_case, GPUCostProvider
from .model import enumerate_parameters
from .reporting import copy_inputs, write_json, write_manifest, write_summary_markdown
from .resource_encoding import compact_rank
from .resource_validation import validate_resources
from .run import _prepare_output, _fail, _success, _check_or_fail, _transfers, _write_summary_json
from .types import AstraEvent, CheckResult, ObjectRecord
from .validation import _event_checks, _dependency_checks
from .wafer_run import effective_case


ALIGNED_CAPABILITIES = frozenset({
    'equal_size_trajectory_dispatch', 'global_GRPO_uid_advantages', 'PPO_ratio_clipping',
    'global_token_loss_reduction', 'static_microbatch_execution', 'conditional_reference_stage',
    'TP_owner_gradient_norm', 'actor_version_fence', 'explicit_naive_weight_export',
    'BF16_GEMM_FP32_accumulation', 'multi_trajectory_train_microbatch',
    'multi_trajectory_logprob_microbatch', 'GRPO_sample_std_normalization', 'configured_KL_term',
    'distributed_optimizer_state_and_collectives'})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_policy(path):
    data = json.loads(Path(path).read_text())
    if set(data) == {'schema_version', 'policy'}:
        if data['schema_version'] != 'rlxy.aligned_policy/1': raise ValueError('unsupported aligned policy schema')
        data = data['policy']
    if not isinstance(data, dict): raise ValueError('aligned policy must be a JSON object')
    return BaselinePolicy(**data)


def validate_aligned_clocks(case, graph, events):
    """Use actual native clocks, without legacy per-trajectory assumptions."""
    ops = {o.node_id: o for o in graph.plan.operations};stages = {};errors = [];versions = []
    for event in events:
        op = ops[event.node_id];stage = op.metadata.get('colocation_stage', op.phase)
        span = stages.setdefault((op.round_id, stage), [event.issue_tick, event.callback_tick])
        span[0] = min(span[0], event.issue_tick);span[1] = max(span[1], event.callback_tick)
        if op.name.startswith('Adam_owned_shard_') or op.name == 'publish_actor_version':
            expected = case.workload.behavior_version+op.round_id-case.workload.round_id
            if (op.metadata.get('actor_version_in'), op.metadata.get('actor_version_out')) != (expected, expected+1):
                versions.append(op.op_id)
    for off in range(case.rounds):
        rid = case.workload.round_id+off
        for stage in ('weight_reshard', 'generation', 'reward', 'old_policy', 'reference', 'advantage',
                      'train', 'dp_sync', 'clip', 'optimizer', 'optimizer_sync', 'publish'):
            if (rid, stage) not in stages: errors.append([rid, 'missing', stage])
        for before, after in [('weight_reshard', 'generation'), ('switch_to_rollout', 'generation'),
            ('generation', 'reward'), ('switch_to_train', 'reward'), ('reward', 'old_policy'),
            ('old_policy', 'reference'), ('reference', 'advantage'), ('advantage', 'train'),
            ('train', 'dp_sync'), ('dp_sync', 'clip'), ('optimizer_sync', 'publish')]:
            if (rid, before) in stages and (rid, after) in stages and stages[rid, before][1] > stages[rid, after][0]:
                errors.append([rid, before, after, stages[rid, before], stages[rid, after]])
        published = [o for o in graph.plan.operations if o.round_id == rid and o.name == 'publish_actor_version']
        if len(published) != 1 or set(published[0].participant_ranks) != set(range(graph.audit['active_devices'])):
            versions.append([rid, 'one active-world publication required'])
        if off and (rid, 'generation') in stages and (rid-1, 'publish') in stages:
            if stages[rid, 'generation'][0] < stages[rid-1, 'publish'][1]: errors.append([rid, 'stale next-round rollout'])
    checks = [CheckResult('aligned.global_phase_barriers', not errors,
        'global preparation/train/DO boundaries from submitted native issue/callback ticks', errors,
        'actual event intervals; local clip/Adam/collective arrivals may overlap across different ranks', 'run'),
        CheckResult('aligned.actor_reference_publication', not versions,
        'one active-world publication after all parameter gathers; Actor advances and Reference fixed', versions,
        'submitted update/publication metadata plus actual global parameter-gather fence', 'run')]
    return checks, {f'r{rid}.{stage}': dict(start_tick=s[0], end_tick=s[1], span_model_ns=(s[1]-s[0])/1000)
                    for (rid, stage), s in stages.items()}


def expand_component_proofs(graph, events, *, proof_cache=None):
    """Validate every observed invocation; replay each immutable template once.

    The native graph has already been parsed against global names/kinds. Local
    remapping changes only identity fields, retaining the actual ticks and kind.
    Reused proofs explicitly report no additional materialized logical events.
    """
    cache = proof_cache if proof_cache is not None else ComponentProofCache()
    by_node = {}
    for e in events: by_node.setdefault(e.node_id, []).append(e)
    proofs = {};checks = [];assigned_nodes = set()
    for component in getattr(graph, 'component_expansions', ()):
        contraction = component['contraction'];mapping = component['global_to_local'];physical = component['physical']
        if component['label'] in proofs: raise ValueError('duplicate component invocation label')
        if assigned_nodes.intersection(mapping): raise ValueError('global native macro belongs to multiple components')
        assigned_nodes.update(mapping)
        if len(set(physical)) != len(physical): raise ValueError('duplicate component physical rank')
        local_rank = {r: i for i, r in enumerate(physical)}
        local_ops = {o.node_id: o for o in contraction.macro_plan.operations};observed = []
        if len(set(mapping.values())) != len(mapping) or set(mapping.values()) != set(local_ops):
            raise ValueError('incomplete/duplicate component native node mapping')
        if set(local_rank.values()) != {int(r) for r in contraction.macro_plan.contract['ranks']}:
            raise ValueError('component physical/local rank coverage differs')
        for global_node, local_node in mapping.items():
            for e in by_node.get(global_node, ()):
                if e.rank not in local_rank: raise ValueError('component event rank outside placement')
                op = local_ops[local_node]
                observed.append(AstraEvent(local_rank[e.rank], local_node, op.op_id, op.round_id, op.phase,
                    e.node_type, e.issue_tick, e.callback_tick, e.duration_ticks, e.issue_ns, e.callback_ns, e.duration_ns))
        proof = cache.verify(contraction, observed)
        proof.update(physical_ranks=list(physical),
            logical_operations_sha256=contraction.certificate['logical_operations_sha256'])
        proofs[component['label']] = proof
        checks.append(CheckResult('aligned.component_expansion.'+component['label'], True,
            'every native macro clock checked; first template ledger replay or explicit serial proof reuse',
            dict(mode=proof['mode'], logical_events_represented=proof['logical_events_represented'],
                 expanded_logical_events=proof['expanded_logical_events']),
            'network clocks preserved; derived/template resource proof, not additional native callbacks', 'run'))
        del observed
    guard = cache.finalize()
    checks.append(CheckResult('aligned.component_template_immutability', True,
        'all unique source/contract/macro/block/certificate signatures rechecked before success', guard,
        'run-scoped read-only templates; every invocation independently clock-checked', 'run'))
    return checks, proofs


def run_aligned(case_path, scenario_path, policy_path, out_path, *, gpu_profile_path=None, active=48,
                contract_local_chains=False, compact_components=True, native_timeout_seconds=300,conservative_bridge_reads=False,
                intern_resource_read_ids=True):
    if contract_local_chains:
        raise ValueError('whole-graph --contract-local-chains is unsupported; component contraction is available and independently replayed')
    if type(active) is not int or active < 1: raise ValueError('positive integer active world required')
    if type(native_timeout_seconds) is not int or native_timeout_seconds < 1: raise ValueError('positive native timeout required')
    case = load_case(case_path);case, scenario, hardware_source = effective_case(case, scenario_path)
    policy = load_policy(policy_path);gpu = None;provider = CapacityAwareCostProvider();topology = 'mesh2d'
    if gpu_profile_path is not None:
        gpu = load_gpu_profile(gpu_profile_path);case = effective_gpu_case(case, gpu);provider = GPUCostProvider(gpu);topology = 'gpu_cluster'
        scenario = {**scenario, 'mesh': [gpu['gpus_per_node'], gpu['gpu_count']//gpu['gpus_per_node']]}
    from .memoized_costs import MemoizedCostProvider
    provider = MemoizedCostProvider(provider)
    allocated = scenario['mesh'][0]*scenario['mesh'][1]
    out = _prepare_output(out_path);timings = {};start = time.perf_counter()
    enforce_phase_deadlines = sys.platform.startswith('linux') and Path('/.dockerenv').is_file()
    try:
        copied = copy_inputs(case.source_path.parent, out, ('case.json',));inputs = out/'inputs'
        write_json(out/'phase_deadlines.json', dict(schema_version='rlxy.aligned_phase_deadlines/1',
            enforced=enforce_phase_deadlines, enforcement_scope='Linux Docker main-thread new aligned runner only',
            planner_including_unique_signatures_seconds=180, encode_and_trace_seconds=180,
            native_subprocess_seconds=native_timeout_seconds, parse_through_manifest_seconds=180,
            timeout_policy='FAIL with preserved partial evidence; no budget increase or automatic retry'))
        optional = [n for n in ('model_config.json', 'workload_provenance.json', 'compilation.json', 'gpu_provenance.json')
                    if (case.source_path.parent/n).is_file()]
        if optional: copied.update(copy_inputs(case.source_path.parent, out, optional))
        import shutil
        for name, source in [('wafer.json', scenario_path), ('source_hardware.json', hardware_source), ('policy.json', policy_path)]:
            shutil.copyfile(source, inputs/name);copied[name] = inputs/name
        if gpu is not None:
            shutil.copyfile(gpu_profile_path, inputs/'gpu_profile.json');copied['gpu_profile'] = inputs/'gpu_profile.json'
            if gpu.get('gemm_profile'):
                source = (Path(gpu_profile_path).resolve().parent/gpu['gemm_profile']).resolve()
                shutil.copyfile(source, inputs/'gemm_measurements.json');copied['gemm_measurements'] = inputs/'gemm_measurements.json'
        audit = audit_policy(case.model, policy, world_size=active, runtime_capabilities=ALIGNED_CAPABILITIES)
        scope_errors = []
        if not policy.needs_reference or not policy.use_kl_loss or policy.use_kl_in_reward:
            scope_errors.append('selected builder currently requires KL-in-loss and an explicit Reference')
        if policy.logprob_micro_batch_trajectories > 1:
            from .execution_contract import prompt_for
            lengths = {(len(prompt_for(case, t)), len(t.response_token_ids)) for t in case.workload.trajectories}
            if len(lengths) > 1: scope_errors.append('multi-trajectory logprob packing currently requires homogeneous lengths')
        audit['selected_runtime_subset_blockers'] = scope_errors
        write_json(out/'policy_preflight.json', audit)
        if not audit['runtime_alignment_ready'] or scope_errors:
            raise ValueError('aligned policy preflight failed: '+str(audit['blockers']+audit['runtime_capabilities_missing']+scope_errors))
        planner_start = time.perf_counter()
        with phase_deadline('planner', 180, enabled=enforce_phase_deadlines):
            t = time.perf_counter();graph = build_aligned_graph(case, allocated, policy, provider,
                str(out/'astra_logs/resources'), active=active, compact_components=compact_components,conservative_bridge_reads=conservative_bridge_reads)
            timings['build_graph_seconds'] = time.perf_counter()-t
            # Reuse this exact built graph, including original contracted kernels;
            # never rebuild a separate duration1 graph for the actual cost census.
            from rl_xy.gemm_census import census_graph, measured_domain
            t = time.perf_counter();census = census_graph(graph, active=active)
            census.update(schema_version='rlxy.aligned_gemm_census/1', status='ACTUAL_SUBMITTED_GRAPH_CENSUS',
                platform='GPU' if gpu else 'WAFER', policy_sha256=policy_fingerprint(policy),
                source='same built graph submitted to native; component originals weighted by every invocation',
                scope='shape/call/FLOP coverage; does not itself calibrate kernel or full-system time',
                measured_domain=measured_domain(census, gpu.get('gemm_measurements') if gpu else None))
            write_json(out/'actual_gemm_census.json', census)
            timings['actual_graph_gemm_census_seconds'] = time.perf_counter()-t
            write_json(out/'resolved_config.json', dict(case=case.resolved, effective_hardware=asdict(case.hardware),
                scenario=scenario, gpu_profile=gpu, policy=asdict(policy), active_devices=active, allocated_devices=allocated))
            write_json(out/'placement.json', graph.placements)
            certificates = {};invocations = [];certificate_keys = {};proof_cache = ComponentProofCache()
            t = time.perf_counter()
            for component in getattr(graph, 'component_expansions', ()):
                contraction = component['contraction'];cert = contraction.certificate
                proof_cache.prepare(contraction)
                key = certificate_keys.get(id(contraction))
                if key is None: key = _digest(cert);certificate_keys[id(contraction)] = key
                certificates.setdefault(key, cert)
                invocations.append(dict(label=component['label'], certificate_sha256=key, physical=list(component['physical']),
                    global_to_local=component['global_to_local'],source_to_canonical_trajectories=component['source_to_canonical_trajectories']))
            timings['component_unique_signature_prepare_seconds'] = time.perf_counter()-t
        timings['planner_including_unique_signatures_seconds'] = time.perf_counter()-planner_start
        graph_audit = {k: v for k, v in graph.audit.items() if k != 'component_contraction_certificates'}
        t = time.perf_counter()
        with phase_deadline('encode', 180, enabled=enforce_phase_deadlines):
            write_json(out/'aligned_contract.json', dict(audit=graph_audit, policy_preflight=audit,
                component_templates=certificates, component_invocations=invocations,
                native_resource_scope='macro/arena hooks; component objects independently replayed at expanded observed clocks'))
            compact = {**graph.plan.contract, 'schema_version': '1.0', 'ranks': {r: compact_rank(c, intern_read_ids=intern_resource_read_ids) for r, c in graph.plan.contract['ranks'].items()}}
            audit_encoding = write_gzip_json(out/'resource_contract.json.gz', compact)
            write_json(out/'tensor_specs.json', graph.plan.tensors)
            files = {};native_encodings = {}
            for rank, c in compact['ranks'].items():
                path = inputs/f'resources.{rank}.json';record = write_minified_json(path, c)
                files[rank] = str(path);copied['resources.'+rank] = path
                native_encodings[rank] = dict(path=path.relative_to(out).as_posix(), **record)
            write_json(out/'resource_contract_format.json', dict(schema_version='rlxy.resource_contract_format/1',
                audit_contract=dict(path='resource_contract.json.gz', **audit_encoding),
                native_rank_contracts=native_encodings, native_index_path='inputs/resource_index.json',
                canonical_semantic_definition='SHA256 of compact sorted-key UTF8 JSON without trailing newline',
                reconstruction='load_json_auto(audit_path); each ranks[rank] equals load_json_auto(native_rank_path)',
                resource_scope='complete submitted macro compact contract; expand_rank reconstructs read sets/private workspace',
                read_ID_encoding='interned_read_id_symbols' if intern_resource_read_ids else 'legacy_string_delta',
                encoding_only_change=True, native_resource_logs_preserved=True,
                legacy_plain_audit_path_present=False))
            write_json(inputs/'resource_index.json', dict(schema_version='1.0', rank_files=files, log_prefix=graph.plan.contract['log_prefix']))
            copied['resource_index'] = inputs/'resource_index.json';write_json(inputs/'groups.json', graph.groups);copied['groups'] = inputs/'groups.json'
            system = json.loads((case.source_path.parent/'astra_system.json').read_text())
            for key in ('all-reduce-implementation', 'all-gather-implementation', 'reduce-scatter-implementation', 'all-to-all-implementation'):
                system[key] = ['ring', 'ring']
            system.update({'local-mem-bw': 1e12, 'endpoint-delay': 0, 'local-reduction-delay': 0})
            write_json(inputs/'astra_system.json', system);copied['system'] = inputs/'astra_system.json'
            rates = [gpu['nvlink_bytes_per_ns_per_direction'], gpu['node_uplink_bytes_per_ns_per_direction']] if gpu else [case.hardware.d2d_bytes_per_ns_per_direction]*2
            lats = [gpu['nvlink_latency_ns'], gpu['uplink_latency_ns']] if gpu else [case.hardware.d2d_latency_ns]*2
            bws = [r/1000*1e9/(2**30) for r in rates]
            (inputs/'astra_network.yml').write_text(f'topology: [ Ring, Ring ]\nnpus_count: [ {scenario["mesh"][0]}, {scenario["mesh"][1]} ]\nbandwidth: [ {bws[0]}, {bws[1]} ]\nlatency: [ {lats[0]*1000}, {lats[1]*1000} ]\n')
            copied['network'] = inputs/'astra_network.yml';write_json(inputs/'astra_remote_memory.json', {'memory-type': 'NO_MEMORY_EXPANSION'});copied['remote'] = inputs/'astra_remote_memory.json'
            write_json(out/'astra_config_contract.json', dict(backend='congestion_aware', topology=topology, dimensions=scenario['mesh'],
                native_packetbundle_local_mem_bw_GBps=1e12,
                local_mem_bw_scope='disable duplicate native implicit service proxy; NOT hardware bandwidth',
                physical_memory_cost_source='explicit local kernel/endpoint/reduction KernelSpecs with platform provider',
                GROUP_ALL_GATHER_comm_size='per-rank input shard bytes, not full gathered output bytes',
                REDUCE_SCATTER_comm_size='full per-rank gradient input buffer bytes', hardware_calibrated=False))
            traces = generate_traces(graph.plan.operations, inputs/'workload', ranks=allocated, astra_time_scale=1000, comm_groups=graph.groups)
            copied.update({f'workload_rank{r}':p for r, p in enumerate(traces)})
            timings['serialization_and_trace_seconds'] = time.perf_counter()-t
        t = time.perf_counter();execution = run_astrasim(trace_prefix=inputs/'workload', system_config=copied['system'], network_config=copied['network'],
            remote_memory_config=copied['remote'], output_dir=out, resource_contract=copied['resource_index'],
            binary='/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware',
            network_model=topology, comm_groups=copied['groups'], timeout_seconds=native_timeout_seconds)
        timings['native_seconds'] = time.perf_counter()-t;t = time.perf_counter()
        with phase_deadline('post', 180, enabled=enforce_phase_deadlines):
            events, summaries = parse_astra_output(execution, graph.plan.operations, ranks=allocated, astra_time_scale=1000,
                inactive_ranks=graph.audit['idle_devices'])
            global_case = replace(case, hardware=replace(case.hardware, dies=allocated), strategy=replace(case.strategy, tensor_parallel=policy.train_tp))
            checks = list(graph.component_checks)+_event_checks(global_case, graph.plan.operations, events)+_dependency_checks(global_case, graph.plan.operations)
            extra, stages = validate_aligned_clocks(case, graph, events);checks.extend(extra)
            resources, snapshots, peaks = validate_resources(graph.plan, global_case, events, gradient_visits=[]);checks.extend(resources)
            expanded, proofs = expand_component_proofs(graph, events, proof_cache=proof_cache);checks.extend(expanded)
            signature_guard = proof_cache.finalize();write_json(out/'component_template_signature_guard.json', signature_guard)
            write_json(out/'component_logical_proofs.json', proofs);write_json(out/'global_stage_times.json', stages)
            timings['parse_and_validation_seconds'] = time.perf_counter()-t
            objects = [ObjectRecord(x['storage_id'], x['tensor_id'], x['kind'], f'd{x["rank"]}.dram', x['nbytes'], str(x['dtype_bytes'])+'B',
                None, str(x['created_node']), str(x['ready_node']), str(x['released_node']), None if x['released_node'] is None else str(x['released_node']),
                x['persistent'], x['rank']) for x in graph.plan.tensors]
            profiles = {};exported = []
            for op in graph.plan.operations:
                meta = dict(op.metadata)
                if 'cost_estimate' in meta:
                    value = meta.pop('cost_estimate');key = _digest(value);profiles[key] = value;meta['cost_profile_sha256'] = key
                exported.append(replace(op, metadata=meta))
            write_json(out/'cost_profiles.json', profiles)
            write_aligned_tables(out, operations=exported, events=events, summaries=summaries, objects=objects, snapshots=snapshots,
                         transfers=_transfers(graph.plan.operations), checks=checks, parameters=enumerate_parameters(global_case))
            _check_or_fail(out, checks)
            metrics = dict(metric_scope='conditional_selected_verl_PP1_DO_simulation_not_measured_hardware',
                total_simulated_time_ns=max(e.callback_tick for e in events)/1000, rounds=case.rounds, dies=allocated,
                active_devices=active, inactive_devices=graph.audit['idle_devices'], cores_per_die=case.hardware.cores_per_die,
                tensor_parallel=policy.train_tp, data_parallel=active//policy.train_tp, rollout_tensor_parallel=policy.rollout_tp,
                rollout_data_parallel=active//policy.rollout_tp, trajectory_count=len(case.workload.trajectories),
                response_tokens=sum(len(t.response_token_ids) for t in case.workload.trajectories), operation_count=len(graph.plan.operations),
                operation_rank_event_count=len(events), optimizer_updates=case.rounds, actor_version_end=case.workload.behavior_version+case.rounds,
                reference_version_end=case.workload.behavior_version, memory_peak_bytes_by_location=peaks,
                memory_peak_scope='native conservative component-arena reservations; exact component logical peaks in proof file',
                policy=asdict(policy), policy_fingerprint=policy_fingerprint(policy), aligned_protocol=graph_audit,
                component_logical_replay_count=signature_guard['full_template_replays'],
                component_invocation_proof_count=len(proofs),
                component_reused_serial_proof_count=signature_guard['reused_serial_template_invocations'],
                logical_component_event_count=sum(p['logical_events_represented'] for p in proofs.values()),
                materialized_logical_component_event_count=sum(p['expanded_logical_events'] for p in proofs.values()),
                compact_components=compact_components, whole_graph_contraction=False, hardware_calibrated=False,
                resource_read_ids_interned=intern_resource_read_ids,
                numerical_training=False, full_verl_runtime_reproduced=False, paper_performance_ready=False,
                physical_geometry_validated=False, assumptions=scenario['assumptions'], platform='GPU' if gpu else 'WAFER')
            metrics['cost_cache'] = dict(hits=provider.hits, misses=provider.misses, exact_spec_hardware_keys=True)
            if gpu: metrics.update(gpu_profile=gpu, device_count=allocated, sms_per_gpu=gpu['sms_per_gpu'])
            write_json(out/'metrics.json', metrics);_write_summary_json(out, metrics, checks, evidence_kind=case.evidence_kind)
            write_summary_markdown(out, title='固定veRL PP1/DO协议的条件性原生仿真', metrics=metrics, checks=checks)
            timings['total_before_manifest_seconds'] = time.perf_counter()-start;write_json(out/'wallclock_timings.json', timings)
            _success(out);manifest=write_manifest(out, case_id=case.case_id, evidence_kind=case.evidence_kind,
                run_kind='gpu_verl_aligned' if gpu else 'wafer_verl_aligned', workflow_scope='selected PP1 DO protocol; native macro resources plus clock-bound component logical replay; uncalibrated',
                execution=execution, metrics=metrics, checks=checks, input_paths=copied)
            # The shared historical writer carries an unaware-backend text label.
            # State this runner's actual binary, without changing frozen bundles.
            manifest['scope']['simulator']='pinned AstraSim analytical congestion-aware backend with rl-xy mesh/GPU/resource patches'
            write_json(out/'manifest.json',manifest)
            return out
    except BaseException as exc:
        timings['total_before_failure_seconds'] = time.perf_counter()-start;write_json(out/'wallclock_timings.json', timings)
        _fail(out, exc);raise
