"""Native mesh execution and fail-closed evidence for DP x TP2 wafer graphs."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

from .astra_adapter import generate_traces, parse_astra_output, run_astrasim
from .config import load_case
from .execution_contract import demean_advantages
from .execution_validation import shape_checks
from .hardware_profile import load_hardware_profile
from .model import enumerate_parameters
from .reporting import (copy_inputs,write_json,write_csv,write_tables,write_manifest,write_summary_markdown)
from .resource_validation import validate_resources
from .run import _prepare_output,_fail,_success,_check_or_fail,_transfers,_write_summary_json
from .tensor_plan import preflight_contract
from .training_schedule import training_visits
from .types import CheckResult,ObjectRecord
from .validation import _event_checks,_dependency_checks,_training_checks,_trajectory_checks
from .wafer_graph import build_wafer_graph
from .resource_encoding import compact_rank


def effective_case(case,scenario_path):
    scenario_path=Path(scenario_path).resolve()
    s=json.loads(scenario_path.read_text())
    if set(s)!={'schema_version','mesh','data_parallel','source_profile','assumptions'} or s['schema_version']!='rlxy.wafer/1':
        raise ValueError('invalid explicit wafer scenario')
    source=(scenario_path.parent/s['source_profile']).resolve()
    profile=load_hardware_profile(source);p={k:v['value'] for k,v in profile['parameters'].items()}
    mesh=s['mesh']
    if mesh!=[p['modules_x'],p['modules_y']]:raise ValueError('scenario topology differs from source H')
    required={'gemm_flops_per_ns','gemm_fp32_flops_per_ns','vector_ops_per_ns','module_path_bytes_per_ns',
        'sram_bytes_per_ns','d2d_latency_ns','d2d_packet_bytes','nic_tx_bytes_per_die',
        'nic_rx_bytes_per_die','host_bytes_per_ns','host_latency_ns','d2d_caliber'}
    if set(s['assumptions'])!=required:raise ValueError('all unresolved execution fields must be explicit assumptions')
    a={}
    for key,row in s['assumptions'].items():
        if set(row)!={'value','status','reason'} or row['status']!='test_assumption' or not row['reason']:
            raise ValueError('invalid assumption provenance')
        value=row['value']
        if key=='d2d_caliber':
            if value!='per_edge_per_direction':raise ValueError('explicit per-edge/direction assumption required')
        elif type(value) is not int or value<=0:raise ValueError('positive integer assumption required')
        else:a[key]=value
    a.update(cores_per_die=p['cores_per_module'],dram_bytes_per_die=p['dram_bytes_per_module'],
        sram_bytes_per_core=p['sram_bytes_per_core'],dram_bytes_per_ns=p['dram_bytes_per_ns_per_module'],
        d2d_bytes_per_ns_per_direction=p['d2d_bytes_per_ns'])
    return replace(case,hardware=replace(case.hardware,**a)),s,source


def verify_wafer_graph(case,graph,events):
    world=len(graph.plan.contract['ranks'])
    global_case=replace(case,hardware=replace(case.hardware,dies=world))
    checks=_event_checks(global_case,graph.plan.operations,events)+_dependency_checks(global_case,graph.plan.operations)
    denom=sum(len(t.response_token_ids) for t in case.workload.trajectories)
    visits=[]
    for i,(local,operations) in enumerate(zip(graph.cases,graph.local_operations)):
        # Use actual submitted operation metadata, not the unmodified template.
        submitted=[]
        for o in graph.plan.operations:
            if o.metadata.get('replica_id')==i:
                submitted.append(replace(o,op_id=o.op_id.removeprefix(f'dp{i}.'),
                    participant_ranks=tuple(r-2*i for r in o.participant_ranks),
                    src_rank=None if o.src_rank is None else o.src_rank-2*i,
                    dst_rank=None if o.dst_rank is None else o.dst_rank-2*i))
        # Shape/coverage normalization: one optimizer update per epoch; all
        # collected groups retain original advantages throughout all epochs.
        checks.extend(replace(c,check_id=f'dp{i}.'+c.check_id) for c in shape_checks(local,submitted))
        checks.extend(replace(c,check_id=f'dp{i}.'+c.check_id) for c in _trajectory_checks(local,submitted))
        for offset in range(case.rounds):
            rid=case.workload.round_id+offset
            original_adv=[o for o in submitted if o.round_id==rid and o.name=='group_advantage_demean']
            for epoch in range(case.strategy.training_epochs):
                update=offset*case.strategy.training_epochs+epoch
                selected=[o for o in submitted if o.round_id==rid and
                    (o.metadata.get('update_id')==update or (case.strategy.training_epochs==1 and
                     o.phase.startswith(('train_','clip','optimizer','publish'))))]
                normalized=[]
                for o in selected:
                    tid=o.metadata.get('trajectory_id')
                    old=f'train_u{update}_{tid}' if tid else ''
                    normalized.append(replace(o,name=o.name.replace(old,f'train_{tid}',1) if old else o.name))
                virtual=replace(local,rounds=1,workload=replace(local.workload,round_id=rid))
                for c in _training_checks(virtual,normalized+original_adv,advantages_override=demean_advantages(local),denominator_override=denom):
                    checks.append(replace(c,check_id=f'dp{i}.u{update}.'+c.check_id))
        for rid,_,_,_,_,phase in training_visits(local):
            visits.extend((2*i+rank,rid,phase+'_backward') for rank in (0,1))
    # DP nodes must precede local clip/Adam for every update, all 54 ranks
    # are connected by actual collective dependencies and round fences.
    errors=[]
    for offset in range(case.rounds):
        rid=case.workload.round_id+offset
        for epoch in range(case.strategy.training_epochs):
            u=offset*case.strategy.training_epochs+epoch
            for shard in (0,1):
                sync=[o for o in graph.plan.operations if o.name=='dp_gradient_sum' and o.round_id==rid
                      and o.metadata.get('dp_update')==u and o.participant_ranks==tuple(range(shard,world,2))]
                if len(sync)!=1 or not sync[0].metadata.get('sum_not_mean') or sync[0].metadata.get('global_loss_denominator')!=denom:
                    errors.append([rid,u,shard,'missing/invalid DP gradient SUM'])
            for i in range(world//2):
                candidates=[o for o in graph.plan.operations if o.round_id==rid and o.metadata.get('replica_id')==i
                            and (o.metadata.get('update_id',u)==u)]
                adam=[o for o in candidates if o.name=='adamw_read_modify_write']
                if len(adam)!=1 or (adam[0].metadata.get('actor_version_in'),adam[0].metadata.get('actor_version_out'))!=(case.workload.behavior_version+u,case.workload.behavior_version+u+1):
                    errors.append([rid,u,i,'version chain'])
    checks.append(CheckResult('wafer.dp_gradient_sum_versions',not errors,'global token mean then DP SUM; consistent actor versions',errors,
        'native TP/DP communicator groups and actual graph','run'))
    extra,snapshots,peaks=validate_resources(graph.plan,global_case,events,gradient_visits=visits)
    checks.extend(extra)
    return checks,snapshots,peaks


def run_wafer(case_path,scenario_path,out_path,*,gpu_profile_path=None,colocation_path=None):
    case=load_case(case_path)
    case,scenario,source=effective_case(case,scenario_path)
    gpu=None;provider=None;network_model='mesh2d'
    if gpu_profile_path is not None:
        from .gpu_model import load_gpu_profile,effective_gpu_case,GPUCostProvider
        gpu=load_gpu_profile(gpu_profile_path);case=effective_gpu_case(case,gpu)
        provider=GPUCostProvider(gpu);network_model='gpu_cluster'
        scenario={**scenario,'mesh':[gpu['gpus_per_node'],gpu['gpu_count']//gpu['gpus_per_node']],
                  'data_parallel':gpu['gpu_count']//2}
    out=_prepare_output(out_path)
    try:
        copied=copy_inputs(case.source_path.parent,out,('case.json',))
        optional=[name for name in ('model_config.json','compilation.json','gpu_provenance.json','H_SCOPE.md','workload_provenance.json') if (case.source_path.parent/name).is_file()]
        if optional:copied.update(copy_inputs(case.source_path.parent,out,optional))
        inputs=out/'inputs'
        for name,path in [('wafer.json',Path(scenario_path)),('source_hardware.json',source)]:
            import shutil
            shutil.copyfile(path,inputs/name);copied[name]=inputs/name
        if gpu is not None:
            import shutil
            shutil.copyfile(gpu_profile_path,inputs/'gpu_profile.json');copied['gpu_profile']=inputs/'gpu_profile.json'
            if gpu.get('gemm_profile'):
                gemm_source=(Path(gpu_profile_path).resolve().parent/gpu['gemm_profile']).resolve()
                shutil.copyfile(gemm_source,inputs/'gemm_measurements.json');copied['gemm_measurements']=inputs/'gemm_measurements.json'
        world=scenario['mesh'][0]*scenario['mesh'][1]
        colocation=None
        if colocation_path is not None:
            from .colocation import load_colocation,build_colocated_graph
            colocation=load_colocation(colocation_path,case,world)
            import shutil
            shutil.copyfile(colocation_path,inputs/'colocation.json');copied['colocation']=inputs/'colocation.json'
            if gpu is None and colocation['mapping']=='capacity_aware':
                from .capacity_mapping import CapacityAwareCostProvider
                provider=CapacityAwareCostProvider()
            graph=build_colocated_graph(case,world,colocation,str(out/'astra_logs/resources'),cost_provider=provider)
            write_json(out/'colocation_contract.json',{'config':colocation,**graph.audit})
        else:
            graph=build_wafer_graph(case,scenario['mesh'],scenario['data_parallel'],str(out/'astra_logs/resources'),cost_provider=provider)
        write_json(out/'resolved_config.json',{'case':case.resolved,'effective_hardware':asdict(case.hardware),'scenario':scenario,
            'gpu_profile':gpu,'gpu_rates_authoritative_from_profile_not_rounded_per_sm_fields':gpu is not None})
        write_json(out/'placement.json',graph.placements)
        compact={**graph.plan.contract,'ranks':{r:compact_rank(c) for r,c in graph.plan.contract['ranks'].items()}}
        write_json(out/'resource_contract.json',compact)
        write_json(out/'tensor_specs.json',graph.plan.tensors)
        # Rank-split native input avoids parsing the entire 54-rank contract
        # once per rank; the unified audit contract remains independently replayable.
        files={}
        for rank,contract in compact['ranks'].items():
            path=inputs/f'resources.{rank}.json';write_json(path,contract);files[rank]=str(path)
            copied[f'resources.{rank}']=path
        write_json(inputs/'resource_index.json',{'schema_version':'1.0','rank_files':files,'log_prefix':graph.plan.contract['log_prefix']})
        copied['resource_index']=inputs/'resource_index.json'
        write_json(inputs/'groups.json',graph.groups);copied['groups']=inputs/'groups.json'
        system=json.loads((case.source_path.parent/'astra_system.json').read_text())
        for key in ('all-reduce-implementation','all-gather-implementation','reduce-scatter-implementation','all-to-all-implementation'):
            system[key]=['ring','ring']
        write_json(inputs/'astra_system.json',system);copied['system']=inputs/'astra_system.json'
        rates=[case.hardware.d2d_bytes_per_ns_per_direction]*2
        lats=[case.hardware.d2d_latency_ns]*2
        if gpu is not None:
            rates=[gpu['nvlink_bytes_per_ns_per_direction'],gpu['node_uplink_bytes_per_ns_per_direction']]
            lats=[gpu['nvlink_latency_ns'],gpu['uplink_latency_ns']]
        bws=[r/1000*1e9/(2**30) for r in rates];lats=[x*1000 for x in lats]
        (inputs/'astra_network.yml').write_text(f'topology: [ Ring, Ring ]\nnpus_count: [ {scenario["mesh"][0]}, {scenario["mesh"][1]} ]\nbandwidth: [ {bws[0]}, {bws[1]} ]\nlatency: [ {lats[0]}, {lats[1]} ]\n')
        copied['network']=inputs/'astra_network.yml'
        write_json(inputs/'astra_remote_memory.json',{'memory-type':'NO_MEMORY_EXPANSION'});copied['remote']=inputs/'astra_remote_memory.json'
        write_json(out/'astra_config_contract.json',{'backend':'congestion_aware','topology':network_model,
            'dimensions':scenario['mesh'],'placement':'selected train TP groups; strided rollout TP; explicit standby ranks' if colocation else 'DP replicas, TP2 adjacent linear ranks',
            'logical_vs_transport_padding':'explicit per-operation transport_payload_bytes',
            'physical_geometry_validated':False,'assumptions':scenario['assumptions']})
        traces=generate_traces(graph.plan.operations,inputs/'workload',ranks=world,astra_time_scale=1000,comm_groups=graph.groups)
        copied.update({f'workload_rank{r}':path for r,path in enumerate(traces)})
        execution=run_astrasim(trace_prefix=inputs/'workload',system_config=copied['system'],network_config=copied['network'],
            remote_memory_config=copied['remote'],output_dir=out,resource_contract=copied['resource_index'],
            binary='/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware',
            network_model=network_model,comm_groups=copied['groups'],
            timeout_seconds=int(os.environ.get('RLXY_ASTRA_TIMEOUT_SECONDS','1200')))
        events,summaries=parse_astra_output(execution,graph.plan.operations,ranks=world,astra_time_scale=1000)
        if colocation is None:checks,snapshots,peaks=verify_wafer_graph(case,graph,events)
        else:
            from .colocation_validation import validate_colocation
            checks,snapshots,peaks,stage_times=validate_colocation(case,graph,events)
            write_json(out/'global_stage_times.json',stage_times)
        objects=[ObjectRecord(t['storage_id'],t['tensor_id'],t['kind'],f'd{t["rank"]}.dram',t['nbytes'],
            str(t['dtype_bytes'])+'B',None,str(t['created_node']),str(t['ready_node']),str(t['released_node']),
            None if t['released_node'] is None else str(t['released_node']),t['persistent'],t['rank']) for t in graph.plan.tensors]
        profiles={};exported=[]
        for op in graph.plan.operations:
            meta=dict(op.metadata)
            if 'cost_estimate' in meta:
                estimate=meta.pop('cost_estimate')
                key=hashlib.sha256(json.dumps(estimate,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                profiles[key]=estimate;meta['cost_profile_sha256']=key
            exported.append(replace(op,metadata=meta))
        write_json(out/'cost_profiles.json',profiles)
        display_case=replace(case,strategy=replace(case.strategy,tensor_parallel=colocation['train_tp'])) if colocation else case
        write_tables(out,operations=exported,events=events,summaries=summaries,objects=objects,
            snapshots=snapshots,transfers=_transfers(graph.plan.operations),checks=checks,parameters=enumerate_parameters(display_case))
        _check_or_fail(out,checks)
        metrics={'metric_scope':'uncalibrated_wafer_scenario_not_numeric_training','total_simulated_time_ns':max(e.callback_tick for e in events)/1000,
            'rounds':case.rounds,'dies':world,'cores_per_die':case.hardware.cores_per_die,'tensor_parallel':2,
            'data_parallel':scenario['data_parallel'],'trajectory_count':len(case.workload.trajectories),
            'response_tokens':sum(len(t.response_token_ids) for t in case.workload.trajectories),'operation_count':len(graph.plan.operations),
            'operation_rank_event_count':len(events),'optimizer_updates':case.rounds*case.strategy.training_epochs,
            'actor_version_end':case.workload.behavior_version+case.rounds*case.strategy.training_epochs,
            'reference_version_end':case.workload.behavior_version,'memory_peak_bytes_by_location':peaks,
            'hardware_calibrated':False,'physical_geometry_validated':False,'assumptions':scenario['assumptions'],
            'native_collective_scratch':'explicit full transport buffer; native internal fragmentation not exposed'}
        if gpu is not None:
            metrics.update(metric_scope='uncalibrated_gpu_analytic_scenario_not_measured_runtime',platform='GPU',
                device_count=world,sms_per_gpu=gpu['sms_per_gpu'],
                gpu_profile=gpu,assumptions={'gpu':gpu['provenance'],'shared_host':scenario['assumptions']},
                native_topology='gpu_node_switch_shared_uplink_spine')
        if colocation is not None:
            metrics.update(tensor_parallel=colocation['train_tp'],data_parallel=graph.audit['train_dp'],
                rollout_tensor_parallel=colocation['rollout_tp'],rollout_data_parallel=graph.audit['rollout_dp'],
                active_devices=graph.audit['active_devices'],colocation=graph.audit,
                metric_scope='pinned_verl_sync_colocation_protocol_analytic_kernels_uncalibrated')
        write_json(out/'metrics.json',metrics)
        _write_summary_json(out,metrics,checks,evidence_kind=case.evidence_kind)
        write_summary_markdown(out,title='GPU SM/CTA与节点交换网络执行' if gpu else 'DP×TP2多核晶圆mesh执行',metrics=metrics,checks=checks)
        _success(out)
        write_manifest(out,case_id=case.case_id,evidence_kind=case.evidence_kind,
            run_kind=('gpu_verl_sync' if gpu else 'wafer_verl_sync') if colocation else ('gpu_dp_tp2' if gpu else 'wafer_dp_tp2'),
            workflow_scope='native GPU node-switch-spine; separate CTA/SM cost; uncalibrated' if gpu else 'native 2D mesh; DP gradient SUM after global token normalization; private multicore SRAM; uncalibrated scenario',
            execution=execution,metrics=metrics,checks=checks,input_paths=copied)
        return out
    except BaseException as exc:
        _fail(out,exc);raise
