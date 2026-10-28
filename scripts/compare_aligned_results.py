#!/usr/bin/env python3
"""Pair two MVP-verified aligned gzip result directories, without simulation.

The runtime bundle provenance and this later analysis tool version are distinct.
Failure releases no ratio. Success releases a conditional simulated time ratio,
never hardware speedup, cost calibration, or paper readiness.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

OPERATOR_CONTRACT='qwen2_PP1_GRPO_DO_last_row_sampling_full_teacher_logits_v1'
SHA_RE=re.compile(r'^[0-9a-f]{64}$')


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def value(text):
    def pairs(rows):
        result={}
        for k,v in rows:
            require(k not in result, 'duplicate JSON key: '+k);result[k]=v
        return result
    return json.loads(text, object_pairs_hook=pairs,
        parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def validate_census(census):
    shapes,signatures,totals=census['shapes'],census['signatures'],census['totals']
    require(shapes and set(shapes)==set(signatures), 'empty or inconsistent GEMM domain')
    for key,problem in shapes.items():
        row=signatures[key]
        require(hashlib.sha256(canonical(problem).encode()).hexdigest()==key==row['signature_sha256'] and row['problem']==problem, 'GEMM signature content SHA mismatch')
        require(type(row['occurrences'])is int and row['occurrences']>0, 'positive rank-local occurrences required')
        require(row['weighted_flops']==2*problem['batches']*problem['m']*problem['n']*problem['k']*row['occurrences'], 'GEMM weighted FLOPs mismatch')
    require(totals['unique_signatures']==len(shapes), 'GEMM unique signature count mismatch')
    for total,field in [('rank_local_matmul_occurrences','occurrences'),('logical_matmul_occurrences','logical_matmul_occurrences'),('weighted_flops','weighted_flops')]:
        require(totals[total]==sum(r[field] for r in signatures.values()), 'GEMM count/FLOP conservation: '+total)


class Bundle:
    def __init__(self, root, verification):
        self.root=Path(root).resolve();self.verification_path=Path(verification).resolve()
        self.manifest=value((self.root/'manifest.json').read_text())
        self.manifest_sha=sha(self.root/'manifest.json')
        self.verified=value(self.verification_path.read_text())
        report=self.verified
        require(report.get('schema_version')=='rlxy.aligned_result_check/1' and report.get('status')=='PASS', 'MVP verification missing/PASS required')
        require(report.get('manifest_sha256')==self.manifest_sha, 'MVP verification is not bound to this manifest')
        require(SHA_RE.fullmatch(report.get('checker_source_sha256', '')), 'MVP checker source SHA missing')
        require(report.get('checks') and all(r.get('status')=='PASS' for r in report['checks']), 'MVP checks not all PASS')
        require(report.get('paper_readiness_promoted_by_checker') is False, 'MVP must not promote paper readiness')
        require(self.manifest.get('status')=='success' and self.manifest.get('exit_status')==0 and self.manifest.get('checks_failed')==0, 'bundle status/native exit failed')
        self.metrics=self.load('metrics.json');self.resolved=self.load('resolved_config.json')
        self.contract=self.load('aligned_contract.json');self.census=self.load('actual_gemm_census.json')
        self.placement=self.load('placement.json')
        self.stages=self.load('global_stage_times.json');self.table_format=self.load('table_format.json')
        self.idle_markers=[];self.operations_scanned=False;self._idle_profiles=None
        software=self.manifest['software'];version=re.fullmatch(r'(\d+)\.(\d+)\.(\d+)',software.get('rl_xy_version',''))
        require(version and tuple(map(int,version.groups())) >= (0,9,2), 'pre-0.9.2 engineering results cannot enter this baseline')
        require(self.metrics['aligned_protocol'].get('operator_contract_id')==OPERATOR_CONTRACT, 'selected operator contract missing or obsolete')
        require(self.contract['audit']==self.metrics['aligned_protocol'], 'metrics/aligned operator audit differs')
        for flag in ('paper_performance_ready','hardware_calibrated','numerical_training','full_verl_runtime_reproduced'):
            require(self.metrics.get(flag) is False, 'unverified scope flag must remain false: '+flag)
        require(self.manifest.get('metrics_sha256')==sha(self.root/'metrics.json'), 'metrics SHA mismatch')

    def artifact(self, name):
        path=(self.root/name).resolve()
        require(path.is_relative_to(self.root) and path.is_file(), 'unsafe or absent bundle artifact: '+name)
        require(self.manifest['artifact_sha256'].get(name)==sha(path), 'used artifact SHA mismatch: '+name)
        return path

    def load(self, name):
        path=self.artifact(name)
        require(path.stat().st_size<=64*1024**2, 'small JSON artifact budget exceeded')
        return value(path.read_text())

    def input_sha(self, basename):
        records=[r for r in self.manifest['inputs'].values() if Path(r['path']).name==basename]
        require(len(records)==1, 'one input required: '+basename)
        row=records[0];path=self.artifact(row['path'])
        require(row['sha256']==sha(path), 'input SHA mismatch: '+basename)
        return row['sha256']

    def table_rows(self,logical):
        require(self.table_format['schema_version']=='rlxy.aligned_tables/1','unsupported aligned table format')
        row=self.table_format['tables'][logical];path=self.artifact(row['path'])
        require(row['encoding']=='gzip_jsonl' and path.suffix=='.gz','gzip JSONL required: '+logical)
        require(row['stored_sha256']==sha(path) and row['stored_bytes']==path.stat().st_size,'stored SHA/bytes mismatch: '+logical)
        digest=hashlib.sha256();byte_count=count=0
        with gzip.open(path,'rb') as f:
            for line in f:
                digest.update(line);byte_count+=len(line);count+=1;yield value(line)
        require(count==row['row_count'] and digest.hexdigest()==row['uncompressed_sha256']
            and byte_count==row['uncompressed_bytes'],'expanded SHA/bytes/count mismatch: '+logical)

    def _idle_operation(self,op):
        meta=op.get('metadata',{});idle=self.metrics['inactive_devices']
        require(self.metrics['platform']=='WAFER' and self.metrics['active_devices']==48
            and self.metrics['dies']==54 and idle==list(range(48,54)),'standby exception only permits six declared wafer idle ranks')
        require(op['kind']=='compute' and op.get('name')=='inactive_rank_marker' and meta.get('idle_rank_marker') is True,
            'idle namespace contains an unverified marker or hidden work')
        require(op.get('phase')=='standby' and meta.get('colocation_stage')=='standby'
            and meta.get('execution_domain')=='device','idle marker stage/domain differs')
        require(op.get('participant_ranks')==idle,'idle marker must cover exactly ranks48..53')
        require(type(op.get('duration_ns'))is int and op['duration_ns']==1,'idle marker duration must be exactly1ns')
        for field in ('flops','vector_ops','tensor_bytes','payload_bytes'):
            require(type(op.get(field))is int and op[field]==0,'idle marker has nonzero/missing work: '+field)
        require(op.get('kernel_spec') is None,'idle marker must not contain a kernel/GEMM specification')
        for source in (op,meta):
            for field in ('read_bytes','write_bytes','dram_read_bytes','dram_write_bytes','traffic_bytes','transport_payload_bytes','sram_workspace_bytes'):
                if field in source:require(type(source[field])is int and source[field]==0,'idle marker has hidden traffic: '+field)
        require(op.get('src_rank') is None and op.get('dst_rank') is None and op.get('collective') is None,'idle marker has communication endpoints')
        require(positive_integer(op.get('node_id')) and type(op.get('round_id'))is int,'idle marker identity missing')
        require(op.get('op_id')==f"r{op['round_id']}.inactive_rank_marker",'idle marker op_id differs')
        if self._idle_profiles is None:self._idle_profiles=self.load('cost_profiles.json')
        key=meta.get('cost_profile_sha256');require(isinstance(key,str) and SHA_RE.fullmatch(key) and key in self._idle_profiles,'idle cost profile missing')
        profile=self._idle_profiles[key]
        require(hashlib.sha256(canonical(profile).encode()).hexdigest()==key and type(profile.get('duration_ns'))is int and profile['duration_ns']==1,'idle cost profile SHA/duration differs')
        for field in ('flops','vector_ops','read_bytes','write_bytes','dram_read_bytes','dram_write_bytes','traffic_bytes','sram_workspace_bytes','local_memory_ns'):
            if field in profile:require(type(profile[field])is int and profile[field]==0,'idle profile has nonzero work: '+field)
        require(not profile.get('partitions'),'idle profile contains compute partitions')
        for field in ('core_workspace_bytes','core_compute_ns'):
            if field in profile:require(isinstance(profile[field],list) and all(type(x)is int and x==0 for x in profile[field]),'idle profile has nonzero per-core work')
        self.idle_markers.append(op)

    def network(self):
        self.idle_markers=[];self.operations_scanned=False
        for op in self.table_rows('operations.jsonl'):
            meta=op.get('metadata',{});ps=op.get('participant_ranks')
            require(isinstance(ps,list) and ps and len(set(ps))==len(ps)
                and all(type(r)is int and 0<=r<self.metrics['dies'] for r in ps),'operation participant namespace invalid')
            stage=meta.get('colocation_stage',op.get('phase'))
            if (any(r>=48 for r in ps) or meta.get('idle_rank_marker') or stage=='standby'
                    or op.get('name')=='inactive_rank_marker'):
                self._idle_operation(op);continue
            require(all(r<48 for r in ps),'non-marker work on idle rank')
            if op['kind']=='compute':continue
            require(op['kind'] in ('p2p','collective'),'unknown network operation kind')
            logical=op['payload_bytes'];transport=meta.get('transport_payload_bytes',logical)
            require(type(logical)is int and type(transport)is int and min(logical,transport)>=0,'noninteger/negative network payload')
            yield dict(round_id=op['round_id'],phase=stage,kind=op['kind'],participant_ranks=ps,
                src_rank=op.get('src_rank'),dst_rank=op.get('dst_rank'),collective=op.get('collective'),
                payload_bytes=logical,transport_payload_bytes=transport)
        self.operations_scanned=True


# These are software layout/dispatch fields. Hardware annotation is explicitly
# separate and cannot replace any required software field.
PLACEMENT_FIELDS=('replica','tp_ranks','groups','trajectories','rollout_tp_groups',
    'global_loss_denominator','distributed_optimizer_shard')
HARDWARE_PLACEMENT_FIELDS=frozenset({'hardware','hardware_metadata','hardware_profile',
    'physical_coordinates','cores_per_die','core_count','core_area_mm2','die_area_mm2','chip_area_mm2'})


def positive_integer(x):return type(x)is int and x>0


def _response_length(trajectory):
    ids=trajectory.get('response_token_ids');length=trajectory.get('response_length')
    if ids is not None:
        require(isinstance(ids,list) and ids,'nonempty response token list required')
        if length is not None:require(length==len(ids),'response length/token list differs')
        return len(ids)
    require(positive_integer(length),'explicit positive response length required')
    return length


def validate_actual_mapping(bundle):
    """Bind ordered actual mappings to full source UIDs, not just counts.

    Global node numbers may differ because an idle marker precedes wafer work.
    Each invocation nevertheless must cover the same ordered local native blocks.
    """
    case=bundle.resolved['case'];policy=bundle.metrics['policy'];workload=case['workload']
    active=bundle.metrics['active_devices'];tr=policy['train_tp'];ge=policy['rollout_tp']
    require(all(positive_integer(x) for x in (active,tr,ge)) and active%tr==0 and tr%ge==0,
        'unsupported actual TP/rollout namespace')
    require(policy.get('ppo_epochs')==1,'selected mapping contract requires one explicit PPO epoch')
    train_mb=policy['train_micro_batch_trajectories'];lp_mb=policy['logprob_micro_batch_trajectories']
    require(positive_integer(train_mb) and positive_integer(lp_mb),'explicit positive mapping microbatches required')
    dp=active//tr;rounds=case['rounds'];first_round=workload['round_id']
    require(positive_integer(rounds) and type(first_round)is int and first_round>=0,'explicit rounds/round ID required')
    trajectories=workload['trajectories'];require(isinstance(trajectories,list) and trajectories,'full source trajectory list required')
    ids=[t['trajectory_id'] for t in trajectories]
    require(all(isinstance(x,str) and x for x in ids) and len(set(ids))==len(ids),'full source UIDs must be nonempty and unique')
    require(len(ids)==bundle.metrics['trajectory_count']==policy['prompt_batch_size']*policy['responses_per_prompt'],
        'source UID/global trajectory count mismatch')
    total_tokens=sum(_response_length(t) for t in trajectories)
    require(total_tokens==bundle.metrics['response_tokens'],'source response/global token count mismatch')
    require(bundle.metrics['rounds']==rounds and bundle.metrics['optimizer_updates']==rounds,
        'round/epoch update reuse count mismatch')
    require((bundle.metrics['tensor_parallel'],bundle.metrics['data_parallel'],bundle.metrics['rollout_tensor_parallel'],
             bundle.metrics['rollout_data_parallel'])==(tr,dp,ge,active//ge),'actual software layout differs from policy')
    require(len(ids)%dp==0,'source trajectories cannot be equally dispatched')
    count=len(ids)//dp;require(count%train_mb==0 and count%lp_mb==0,'source local count/static microbatch mismatch')
    source_group={t['trajectory_id']:t.get('group_id') or workload.get('group_id') for t in trajectories}
    require(all(isinstance(g,str) and g for g in source_group.values()),'source group identity missing')
    groups=Counter(source_group.values())
    require(len(groups)==policy['prompt_batch_size'] and set(groups.values())=={policy['responses_per_prompt']},
        'full source prompt-group cardinality mismatch')
    layout=bundle.contract['audit']['distributed_optimizer']
    require(layout['dp_size']==dp and layout['tp_size']==tr and positive_integer(layout['shard_elements'])
        and layout['padded_elements']==dp*layout['shard_elements'],'DO actual shard layout missing/inconsistent')
    placements=bundle.placement;require(isinstance(placements,list) and len(placements)==dp,'full nonempty placement rows required')
    projected=[];ignored=set()
    for replica,row in enumerate(placements):
        require(isinstance(row,dict) and set(PLACEMENT_FIELDS)<=set(row),'required software placement fields missing')
        extra=set(row)-set(PLACEMENT_FIELDS)
        require(extra<=HARDWARE_PLACEMENT_FIELDS,'unknown placement fields cannot be silently ignored')
        ignored.update(extra);actual={k:row[k] for k in PLACEMENT_FIELDS}
        local=ids[replica*count:(replica+1)*count];local_groups=list(dict.fromkeys(source_group[t] for t in local))
        require(all(sum(source_group[t]==g for t in local)==policy['responses_per_prompt'] for g in local_groups),
            'placement splits a source prompt group across DP')
        expected=dict(replica=replica,tp_ranks=list(range(replica*tr,(replica+1)*tr)),groups=local_groups,trajectories=local,
            rollout_tp_groups=[list(range(replica*tr+j*ge,replica*tr+(j+1)*ge)) for j in range(tr//ge)],
            global_loss_denominator=total_tokens,distributed_optimizer_shard=[replica*layout['shard_elements'],(replica+1)*layout['shard_elements']])
        require(actual==expected,'actual software placement differs from declared source dispatch: replica '+str(replica))
        projected.append(actual)
    expected=[];phase_visits=Counter()
    def add(label,physical,source,phase,rid):
        expected.append((label,physical,{uid:'T'+str(i) for i,uid in enumerate(source)}))
        for uid in source:phase_visits[rid,phase,uid]+=1
    for rid in range(first_round,first_round+rounds):
        for replica,row in enumerate(projected):
            for j,physical in enumerate(row['rollout_tp_groups']):
                chosen=row['trajectories'][j::tr//ge]
                if chosen:add(f'r{rid}.generation{replica}.{j}',physical,chosen,'generation',rid)
        for phase in ('reward','old_policy','reference','advantage','training'):
            for replica,row in enumerate(projected):
                mb=lp_mb if phase in ('old_policy','reference') else train_mb if phase=='training' else count
                for start in range(0,count,mb):
                    add(f'r{rid}.{phase}{replica}.{start}',row['tp_ranks'],row['trajectories'][start:start+mb],phase,rid)
    require(set(phase_visits.values())=={1} and len(phase_visits)==rounds*6*len(ids),
        'declared per-round/epoch source UID reuse is not exactly one per phase')
    invocations=bundle.contract['component_invocations'];templates=bundle.contract['component_templates']
    require(isinstance(invocations,list) and invocations and isinstance(templates,dict) and templates,
        'nonempty actual component invocations/templates required; no legacy fallback')
    actual=[];used_labels=set();used_global=set();used_templates=set();structures={};local_structure=[]
    source_ids=set(ids)
    for entry in invocations:
        require(isinstance(entry,dict) and all(k in entry for k in ('label','physical','source_to_canonical_trajectories',
            'certificate_sha256','global_to_local')),'actual invocation mapping fields missing')
        label=entry['label'];physical=entry['physical'];mapping=entry['source_to_canonical_trajectories']
        require(isinstance(label,str) and label and label not in used_labels,'duplicate/empty actual component label');used_labels.add(label)
        require(isinstance(physical,list) and physical and len(set(physical))==len(physical)
            and all(type(r)is int and 0<=r<active for r in physical),'actual component physical ranks invalid')
        require(isinstance(mapping,dict) and mapping and set(mapping)<=source_ids,'missing/unknown actual source UID mapping')
        require(set(mapping.values())=={'T'+str(i) for i in range(len(mapping))},'canonical UID coverage/uniqueness differs')
        key=entry['certificate_sha256'];require(isinstance(key,str) and SHA_RE.fullmatch(key) and key in templates,'invocation certificate binding missing')
        used_templates.add(key)
        if key not in structures:
            cert=templates[key]
            require(hashlib.sha256(canonical(cert).encode()).hexdigest()==key,'component certificate content SHA mismatch')
            require(cert['schema_version']=='rlxy.local_chain_contraction/1' and cert['strict_per_rank_serialization_proved'] is True,
                'strict local contraction certificate required')
            blocks=cert['blocks'];require(isinstance(blocks,list) and blocks,'local native blocks missing')
            structure=[];logical_ids=[];previous=0
            for block in blocks:
                native=block['native_node_id'];logical=block['logical_node_ids'];ps=block['participants']
                require(positive_integer(native) and native>previous and isinstance(logical,list) and logical
                    and all(positive_integer(n) for n in logical) and logical==sorted(set(logical)) and logical[0]==native,
                    'local native/logical block IDs invalid')
                require(isinstance(ps,list) and ps and len(set(ps))==len(ps)
                    and all(type(r)is int and 0<=r<len(physical) for r in ps),'local block participants invalid')
                structure.append((native,tuple(logical),tuple(ps)));logical_ids.extend(logical);previous=native
            require(len(blocks)==cert['native_operation_count'] and len(logical_ids)==cert['logical_operation_count']
                and len(set(logical_ids))==len(logical_ids),'local block operation coverage mismatch')
            structures[key]=tuple(structure)
        structure=structures[key]
        require(all(all(0<=r<len(physical) for r in ps) for _,_,ps in structure)
            and {r for _,_,ps in structure for r in ps}==set(range(len(physical))),
            'template/local physical rank coverage mismatch')
        global_map=entry['global_to_local'];require(isinstance(global_map,dict) and global_map,'global-to-local native node mapping missing')
        require(all(isinstance(k,str) and re.fullmatch(r'[1-9][0-9]*',k) for k in global_map),'canonical positive global node IDs required')
        ordered=sorted((int(k),v) for k,v in global_map.items())
        require(all(positive_integer(v) for _,v in ordered) and [v for _,v in ordered]==[n for n,_,_ in structure],
            'local native block ID/value coverage or order differs')
        require(not used_global.intersection(k for k,_ in ordered),'global native node assigned to duplicate component')
        used_global.update(k for k,_ in ordered)
        actual.append((label,physical,mapping));local_structure.append(structure)
    require(used_templates==set(templates),'unused/unbound component templates present')
    require(actual==expected,'ordered actual source UID/physical dispatch differs from declared case/placement')
    return dict(dispatch=actual,local_structure=local_structure,placement=projected,
        summary=dict(component_invocations=len(actual),full_source_UIDs=len(ids),rounds=rounds,training_epochs=policy['ppo_epochs'],
            each_source_visits_per_phase_per_round=1,phases=['generation','reward','old_policy','reference','advantage','training'],
            ordered_dispatch_sha256=hashlib.sha256(canonical(actual).encode()).hexdigest(),
            software_placement_sha256=hashlib.sha256(canonical(projected).encode()).hexdigest(),
            local_block_coverage_sha256=hashlib.sha256(canonical(local_structure).encode()).hexdigest(),
            global_node_keys_cross_platform_compared=False,local_block_IDs_and_order_compared=True,
            ignored_hardware_placement_fields=sorted(ignored)))


def validate_standby(bundle):
    """Prove the one permitted extra wafer stage from actual zero-work evidence."""
    require(bundle.operations_scanned,'complete operation scan required before standby validation')
    rounds=bundle.resolved['case']['rounds'];first=bundle.resolved['case']['workload']['round_id']
    expected_rounds=list(range(first,first+rounds));idle=bundle.metrics['inactive_devices']
    stage_keys={key for key in bundle.stages if re.fullmatch(r'r[0-9]+\.standby',key)}
    if bundle.metrics['platform']=='GPU':
        require(not idle and not bundle.idle_markers and not stage_keys,'GPU may not use wafer standby exception')
        markers={};envelopes={}
    else:
        require(idle==list(range(48,54)) and len(bundle.idle_markers)==rounds,'one wafer idle marker per round required')
        require([op['round_id'] for op in bundle.idle_markers]==expected_rounds,'idle marker round coverage/order differs')
        require(len({op['node_id'] for op in bundle.idle_markers})==rounds,'duplicate idle marker node ID')
        markers={op['round_id']:op for op in bundle.idle_markers};envelopes={}
        previous=None
        for offset,rid in enumerate(expected_rounds):
            op=markers[rid];dependencies=[] if previous is None else [previous]
            require(op.get('dependencies')==dependencies and op.get('dependencies_by_rank')=={str(r):dependencies for r in idle},'idle marker dependency chain differs')
            previous=op['node_id'];key=f'r{rid}.standby'
            expected=dict(start_tick=offset*1000,end_tick=(offset+1)*1000,span_model_ns=1.0)
            actual=bundle.stages.get(key)
            require(isinstance(actual,dict) and set(actual)==set(expected) and actual==expected
                and type(actual['start_tick'])is int and type(actual['end_tick'])is int
                and type(actual['span_model_ns'])in (int,float),'standby envelope differs from exact idle marker clock')
            envelopes[key]=actual
        require(stage_keys==set(envelopes),'unexpected/missing standby stage vocabulary')
    observed={}
    for event in bundle.table_rows('events.jsonl'):
        rank=event.get('rank');require(type(rank)is int and 0<=rank<bundle.metrics['dies'],'event rank namespace invalid')
        if rank not in idle:
            require(event.get('phase')!='standby' and not str(event.get('op_id','')).endswith('.inactive_rank_marker'),
                'active event cannot use standby exception')
            continue
        rid=event.get('round_id');require(type(rid)is int and rid in markers,'idle event round has no marker')
        marker=markers[rid];key=(rid,rank)
        require(key not in observed,'duplicate idle event')
        offset=rid-first;start=offset*1000;end=(offset+1)*1000
        require(type(event.get('node_id'))is int and event['node_id']==marker['node_id'] and event.get('op_id')==marker['op_id']
            and event.get('phase')=='standby' and type(event.get('node_type'))is int and event['node_type']==4,'idle event identity/type differs')
        require(all(type(event.get(k))is int for k in ('issue_tick','callback_tick','duration_ticks'))
            and (event['issue_tick'],event['callback_tick'],event['duration_ticks'])==(start,end,1000),'idle event is not exact1ns serial clock')
        require(all(type(event.get(k))in (int,float) for k in ('issue_ns','callback_ns','duration_ns'))
            and (event['issue_ns'],event['callback_ns'],event['duration_ns'])==(start/1000,end/1000,1),
            'idle displayed clocks differ from exact ticks')
        require(event['callback_ns']<=bundle.metrics['total_simulated_time_ns'],'idle callback exceeds reported total MAX')
        observed[key]=event
    require(set(observed)=={(rid,r) for rid in markers for r in idle},'idle event rank/round coverage differs')
    resources={}
    if idle:
        formats=bundle.load('resource_contract_format.json');index=bundle.load('inputs/resource_index.json')
        require(formats['schema_version']=='rlxy.resource_contract_format/1' and index['schema_version']=='1.0','idle resource format/index unsupported')
        for rank in idle:
            record=formats['native_rank_contracts'][str(rank)];path=bundle.artifact(record['path'])
            require(record['encoding']=='minified_json' and record['stored_sha256']==sha(path)
                and record['stored_bytes']==path.stat().st_size,'idle native rank input SHA/encoding differs')
            inputs=[r for r in bundle.manifest['inputs'].values() if Path(r['path']).name==f'resources.{rank}.json']
            require(len(inputs)==1 and inputs[0]['path']==record['path'] and inputs[0]['sha256']==sha(path)
                and Path(index['rank_files'][str(rank)]).name==Path(record['path']).name,'idle resource input/index binding differs')
            data=bundle.load(record['path']);encoded=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
            require(record['canonical_semantic_sha256']==hashlib.sha256(encoded).hexdigest()
                and record['uncompressed_canonical_sha256']==record['canonical_semantic_sha256']
                and record['uncompressed_canonical_bytes']==len(encoded),'idle resource canonical SHA/bytes differs')
            require(data['initial_objects']=={} and data.get('read_id_symbols',[])==[],'idle rank has resident objects/read symbols')
            require(set(data['nodes'])=={str(op['node_id']) for op in markers.values()},'idle rank resource nodes contain hidden work')
            cores=data.get('cores_per_die',1);require(positive_integer(cores),'idle core count invalid')
            profiles=data.get('core_workspace_profiles',{})
            for profile_key,vector in profiles.items():
                require(isinstance(vector,list) and len(vector)==cores and all(type(x)is int and x==0 for x in vector)
                    and hashlib.sha256(canonical(vector).encode()).hexdigest()==profile_key,'idle private workspace profile invalid/nonzero')
            base=0
            for op in markers.values():
                node=data['nodes'][str(op['node_id'])]
                require(node['allocations']=={} and node['ready_outputs']==[] and node['frees']==[],'idle rank alloc/READY/free not empty')
                if 'reads' in node:
                    require(node['reads']==[] and 'read_delta' not in node,'idle reads must be empty/unambiguous')
                else:
                    require(isinstance(node.get('read_delta'),dict) and type(node['read_delta'].get('base_node'))is int
                        and node['read_delta']==dict(base_node=base,add=[],remove=[]),'idle read delta not empty/ordered')
                base=op['node_id']
                for field in ('workspace_bytes','reserve_before','reserve_after'):
                    require(type(node.get(field))is int and node[field]==0,'idle resource has nonzero '+field)
                require(node['local_memory'] is False,'idle marker owns local memory')
                if 'core_workspace_bytes' in node:
                    require(isinstance(node['core_workspace_bytes'],list) and len(node['core_workspace_bytes'])==cores
                        and all(type(x)is int and x==0 for x in node['core_workspace_bytes']),'idle private workspace bytes nonzero')
                if 'core_workspace_profile' in node:
                    require(node['core_workspace_profile'] in profiles,'idle private workspace profile missing')
                require(cores==1 or 'core_workspace_bytes' in node or 'core_workspace_profile' in node,'idle private workspace representation missing')
            log=bundle.artifact(f'astra_logs/resources.{rank}.jsonl')
            expected=[('initial',0,0)]+[(kind,markers[rid]['node_id'],tick) for offset,rid in enumerate(expected_rounds)
                for kind,tick in [('issue',offset*1000),('callback',(offset+1)*1000)]]
            actual=[]
            with log.open('rb') as stream:
                for line in stream:
                    row=value(line)
                    require(type(row.get('node_id'))is int and type(row.get('tick'))is int,'idle raw resource node/tick type invalid')
                    actual.append((row['event'],row['node_id'],row['tick']))
                    require(all(type(row.get(k))is int and row[k]==0 for k in
                        ('dram_bytes','sram_bytes','reserved_bytes','local_memory_owner')),'idle raw resource snapshot is not zero')
                    if 'core_sram_bytes' in row:require(isinstance(row['core_sram_bytes'],list) and len(row['core_sram_bytes'])==cores
                        and all(type(x)is int and x==0 for x in row['core_sram_bytes']),'idle raw core SRAM nonzero')
                    if 'core_sram_nonzero' in row:require(row['core_sram_nonzero']==[],'idle raw sparse core SRAM nonzero')
                    require(cores==1 or 'core_sram_bytes' in row or 'core_sram_nonzero' in row,'idle raw core snapshot missing')
            require(actual==expected,'idle raw resource log node/clock coverage differs')
            resources[str(rank)]=dict(input_sha256=sha(path),raw_resource_sha256=sha(log),resource_rows=len(actual))
    return dict(envelopes=envelopes,idle_ranks=idle,marker_count=len(markers),idle_event_count=len(observed),
        zero_work_operations_verified=True,zero_resource_contracts_and_actual_logs_verified=True,
        resources=resources,total_model_time_MAX_unchanged=True)


def peak_summary(metrics):
    peaks=metrics['memory_peak_bytes_by_location'];active=metrics['active_devices']
    dram=[v for k,v in peaks.items() if k.endswith('.dram') and int(k.split('.')[0][1:])<active]
    idle=[v for k,v in peaks.items() if k.endswith('.dram') and int(k.split('.')[0][1:])>=active]
    core=[v for k,v in peaks.items() if re.fullmatch(r'd\d+\.c\d+\.sram',k)]
    return dict(max_active_rank_DRAM_bytes=max(dram,default=0),max_idle_rank_DRAM_bytes=max(idle,default=0),
        max_core_SRAM_bytes=max(core,default=0),scope=metrics['memory_peak_scope'])


def compare(gpu_root, wafer_root, gpu_verification, wafer_verification):
    result=dict(schema_version='rlxy.strict_transfer_pair/1',status='REJECTED_PAIR',
        created_at_utc=datetime.now(timezone.utc).isoformat(),paper_performance_ready=False,
        paper_readiness_promoted_by_checker=False,conditional_ratio_released=False,
        conditional_gpu_time_over_wafer_time=None,checks=[],
        scope='strict48-active transfer-control conditional simulation comparison; not hardware accuracy or strategy optimality',
        analysis_version={'pair_checker_source_sha256':sha(Path(__file__))})
    def done(name,detail):result['checks'].append(dict(check=name,status='PASS',detail=detail))
    try:
        g=Bundle(gpu_root,gpu_verification);w=Bundle(wafer_root,wafer_verification)
        result['raw_bundles']={label:dict(manifest_sha256=b.manifest_sha,source_tree_sha256=b.manifest['software']['rl_xy_source_tree_sha256'],
            image_id=b.manifest['software']['container_image_id'],rl_xy_version=b.manifest['software']['rl_xy_version'],
            MVP_verification_sha256=sha(b.verification_path),MVP_checker_source_sha256=b.verified['checker_source_sha256']) for label,b in [('GPU',g),('WAFER',w)]}
        done('MVP_PASS_bound_to_each_manifest_and_new_operator_contract',OPERATOR_CONTRACT)
        for key in ('rl_xy_source_tree_sha256','container_image_id','astrasim_binary_sha256','astrasim_commit','rl_xy_version'):
            require(g.manifest['software'].get(key) and g.manifest['software'][key]==w.manifest['software'].get(key), 'runtime software/image mismatch: '+key)
        require(SHA_RE.fullmatch(g.manifest['software']['rl_xy_source_tree_sha256']), 'source SHA malformed')
        require(re.fullmatch(r'sha256:[0-9a-f]{64}',g.manifest['software']['container_image_id']), 'image SHA malformed')
        done('same_frozen_runtime_source_image_binary',g.manifest['software']['rl_xy_source_tree_sha256'])
        input_hashes={}
        for basename in ('case.json','model_config.json','policy.json','workload_provenance.json'):
            x,y=g.input_sha(basename),w.input_sha(basename);require(x==y,'shared input SHA mismatch: '+basename);input_hashes[basename]=x
        require(g.manifest['case_id']==w.manifest['case_id'],'case ID mismatch')
        for key in ('case_id','rounds','model','workload','strategy'):
            require(g.resolved['case'][key]==w.resolved['case'][key], 'resolved case/workload differs: '+key)
        done('identical_case_model_workload_policy_input_SHA',input_hashes)
        gp,wp=g.metrics['policy'],w.metrics['policy'];fingerprint=hashlib.sha256(canonical(gp).encode()).hexdigest()
        require(gp==wp and fingerprint==g.metrics['policy_fingerprint']==w.metrics['policy_fingerprint']==g.census['policy_sha256']==w.census['policy_sha256'], 'policy fingerprint/content mismatch')
        for label,b,allocated in [('GPU',g,48),('WAFER',w,54)]:
            m=b.metrics;require(m['platform']==label and m['active_devices']==48 and m['dies']==allocated, '48-active/allocated platform mismatch: '+label)
            require(m['inactive_devices']==list(range(48,allocated)), 'idle namespace mismatch: '+label)
            require(b.resolved['active_devices']==48 and b.resolved['allocated_devices']==allocated, 'resolved active/allocated mismatch')
        for key in ('trajectory_count','response_tokens','rounds','optimizer_updates','actor_version_end','reference_version_end','tensor_parallel','data_parallel','rollout_tensor_parallel','rollout_data_parallel'):
            require(g.metrics[key]==w.metrics[key], 'counts/layout/version mismatch: '+key)
        done('same_policy_versions_counts_48active_48vs54allocated',fingerprint)
        gm,wm=validate_actual_mapping(g),validate_actual_mapping(w)
        require(gm['placement']==wm['placement'],'actual software placement differs across hardware')
        require(gm['dispatch']==wm['dispatch'],'ordered actual invocation dispatch differs across hardware')
        require(gm['local_structure']==wm['local_structure'],'actual local block IDs/coverage differ across hardware')
        done('identical_actual_software_placement',dict(sequence_sha256=gm['summary']['software_placement_sha256'],
            replicas=len(gm['placement']),software_fields=list(PLACEMENT_FIELDS),
            ignored_hardware_fields=dict(GPU=gm['summary']['ignored_hardware_placement_fields'],WAFER=wm['summary']['ignored_hardware_placement_fields'])))
        done('identical_actual_ordered_source_dispatch_and_local_block_coverage',gm['summary'])
        require(g.census['shapes']==w.census['shapes'] and g.census['signatures']==w.census['signatures'], 'GEMM signature/domain/frequency/FLOP mismatch')
        validate_census(g.census)
        for key in ('unique_signatures','rank_local_matmul_occurrences','logical_matmul_occurrences','weighted_flops'):
            require(g.census['totals'][key]==w.census['totals'][key], 'GEMM totals mismatch: '+key)
        done('identical_actual_GEMM_domain_counts_and_FLOPs',g.census['totals'])
        network_hash=hashlib.sha256();network_count=logical_bytes=transport_bytes=0
        for index,(left,right) in enumerate(itertools.zip_longest(g.network(),w.network()),1):
            require(left==right, 'network semantic sequence differs at position '+str(index))
            network_hash.update((canonical(left)+'\n').encode());network_count+=1
            logical_bytes+=left['payload_bytes'];transport_bytes+=left['transport_payload_bytes']
        done('same_ordered_active_network_semantics',dict(operations=network_count,sequence_sha256=network_hash.hexdigest(),
            summed_logical_payload_bytes=logical_bytes,summed_transport_payload_bytes=transport_bytes,
            byte_scope='one count per operation, not ring wire traffic; p2p transport defaults to logical payload if undeclared'))
        gs,ws=validate_standby(g),validate_standby(w)
        shared_g=set(g.stages)-set(gs['envelopes']);shared_w=set(w.stages)-set(ws['envelopes'])
        require(shared_g==shared_w,'shared software stage vocabulary differs outside verified standby exception')
        done('only_declared_six_idle_wafer_standby_exception',dict(GPU=gs,WAFER=ws,
            shared_software_stages=sorted(shared_g),unknown_stage_skipped=False))
        times=[b.metrics['total_simulated_time_ns'] for b in (g,w)]
        require(all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in times),'positive finite model times required')
        result.update(status='PASS_CONDITIONAL_PAIR',conditional_ratio_released=True,
            conditional_gpu_time_over_wafer_time=times[0]/times[1],GPU_minus_WAFER_model_time_ns=times[0]-times[1],
            model_time_ns=dict(GPU=times[0],WAFER=times[1]),peaks=dict(GPU=peak_summary(g.metrics),WAFER=peak_summary(w.metrics)),
            stage_spans={k:dict(GPU_model_ns=g.stages[k]['span_model_ns'],WAFER_model_ns=w.stages[k]['span_model_ns'],
                GPU_minus_WAFER_model_ns=g.stages[k]['span_model_ns']-w.stages[k]['span_model_ns']) for k in sorted(shared_g)},
            verified_standby_envelopes=dict(GPU=gs['envelopes'],WAFER=ws['envelopes']),
            total_model_time_includes_original_allocated_rank_MAX=True,
            stage_scope='phase envelopes may overlap and must not be summed into step time',
            not_established=['physical GPU/wafer performance accuracy','numerical veRL or learning quality','GPU strategy optimum','paper readiness'])
    except Exception as exc:
        result['checks'].append(dict(check='strict_pair_gate',status='FAIL',error_type=type(exc).__name__,error=str(exc)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('gpu','wafer','gpu-verification','wafer-verification','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():p.error('preserve existing pair analysis')
    if any(a.out.resolve().is_relative_to(x.resolve()) for x in (a.gpu,a.wafer)):p.error('analysis must remain outside immutable bundles')
    result=compare(a.gpu,a.wafer,a.gpu_verification,a.wafer_verification)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','conditional_ratio_released','conditional_gpu_time_over_wafer_time','paper_performance_ready')}))
    return 0 if result['status']=='PASS_CONDITIONAL_PAIR' else 2


if __name__=='__main__':raise SystemExit(main())
