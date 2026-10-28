#!/usr/bin/env python3
"""Independent stdlib MVP checker for the aligned gzip evidence format.

Reads tables in a stream and retains only operation/event dependency indexes.
Does not reconstruct model graphs, refit costs, or promote paper readiness.
Raw debug event re-parsing and independent object-ledger replay are explicitly
outside this MVP; raw native artifacts are required and SHA-checked.
"""
import argparse
import codecs
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

PINNED_ASTRA='4ce9ecfe746a3ee5fe47559a11d91644d641fa7f'
PINNED_INPUT_POLICY_DEFAULTS={'sequence_parallel':False,
    'overlap_grad_reduce':False,'overlap_param_gather':False}
TABLES={'operations.jsonl','events.jsonl','objects.csv','memory.csv','transfers.csv',
    'checks.csv','parameters.csv','astra_rank_summary.csv'}
REQUIRED={'manifest.json','status.json','summary.json','checks.json','metrics.json',
    'astra_command.json','astra_stdout.log','astra_stderr.log','astra_logs/log.log',
    'resolved_config.json','aligned_contract.json','policy_preflight.json',
    'component_logical_proofs.json','component_template_signature_guard.json','actual_gemm_census.json',
    'resource_contract_format.json','resource_contract.json.gz','table_format.json',
    'inputs/resource_index.json'}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def semantic(value,ensure_ascii=False):
    h=hashlib.sha256();enc=json.JSONEncoder(ensure_ascii=ensure_ascii,sort_keys=True,separators=(',',':'),allow_nan=False)
    for chunk in enc.iterencode(value):h.update(chunk.encode())
    return h.hexdigest()


def update_semantic(h,value):
    """Encode the actual decoded value; never substitute an input byte SHA."""
    enc=json.JSONEncoder(ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    # One bounded value at a time, using the same stdlib canonical encoder.
    # This C path avoids millions of Python token callbacks on each rank.
    for chunk in enc.iterencode(value,_one_shot=True):h.update(chunk.encode())


def no_duplicates(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,'duplicate JSON key: '+k);out[k]=v
    return out


def json_value(text):
    return json.loads(text,object_pairs_hook=no_duplicates,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))


class StreamingJSON:
    """Incremental strict values with at most one decoded rank in the caller.

    An incremental UTF-8 decoder handles byte boundaries. The one-MiB read window may retain
    following values, but only the current value is decoded. Every individual
    value has the same byte cap formerly applied to one whole JSON document.
    Larger decode retries are geometrically spaced to avoid decoding a large
    partial rank on every read. The cap itself is never increased.
    """
    def __init__(self,stream,value_limit,chunk_chars=1024*1024):
        require(type(value_limit)is int and value_limit>0,'positive streamed JSON value limit')
        require(type(chunk_chars)is int and chunk_chars>0,'positive streamed read window')
        self.stream=stream;self.limit=value_limit;self.chunk=chunk_chars
        self.buffer='';self.buffer_bytes=0;self.pos=0;self.eof=False;self.max_value_bytes=0
        self.utf8=codecs.getincrementaldecoder('utf-8')()
        self.decoder=json.JSONDecoder(object_pairs_hook=no_duplicates,
            parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))

    def append(self):
        if self.eof:return
        part=self.stream.read(self.chunk)
        if part:
            require(isinstance(part,bytes),'streamed JSON requires a binary input')
            text=self.utf8.decode(part,final=False)
        else:text=self.utf8.decode(b'',final=True);self.eof=True
        self.buffer+=text;self.buffer_bytes+=len(text.encode('utf-8'))

    def peek(self):
        while True:
            while self.pos<len(self.buffer)and self.buffer[self.pos]in ' \t\r\n':self.pos+=1
            if self.pos<len(self.buffer):return self.buffer[self.pos]
            if self.eof:return ''
            self.buffer='';self.buffer_bytes=0;self.pos=0;self.append()

    def expect(self,char):
        require(self.peek()==char,'streamed JSON expected '+repr(char));self.pos+=1

    def value(self):
        require(bool(self.peek()),'truncated streamed JSON value')
        self.buffer=self.buffer[self.pos:];self.buffer_bytes=len(self.buffer.encode('utf-8'));self.pos=0
        primitive=self.buffer[0]not in '{["';next_attempt=0
        while True:
            if not self.eof and len(self.buffer)<next_attempt and self.buffer_bytes<=self.limit:
                self.append();continue
            try:
                obj,end=self.decoder.raw_decode(self.buffer)
                # A split numeric token such as1e+3 may first decode as1.
                complete=(not primitive or end<len(self.buffer)and self.buffer[end]in ',]} \t\r\n' or self.eof and end==len(self.buffer))
                if complete:
                    n=len(self.buffer[:end].encode('utf-8'))
                    require(n<=self.limit,'streamed JSON value over checker limit')
                    self.max_value_bytes=max(self.max_value_bytes,n);self.pos=end
                    return obj
                if self.eof:raise ValueError('invalid or truncated streamed JSON primitive')
            except json.JSONDecodeError:
                if self.eof:raise
            # With no completed value every buffered byte is a token prefix.
            require(self.buffer_bytes<=self.limit,'streamed JSON value over checker limit')
            next_attempt=min(max(len(self.buffer)*2,self.chunk),self.limit+1)
            self.append()

    def keys(self):
        self.expect('{');seen=set();previous=None
        if self.peek()=='}':self.expect('}');return
        while True:
            key=self.value();require(isinstance(key,str),'streamed object key must be a string')
            require(key not in seen,'duplicate streamed JSON key: '+key)
            require(previous is None or previous<key,'noncanonical streamed object key order')
            seen.add(key);previous=key;self.expect(':');yield key
            if self.peek()=='}':self.expect('}');return
            self.expect(',')

    def finish(self):require(self.peek()=='','trailing streamed JSON content')


class Checker:
    def __init__(self,root,max_json_bytes):
        self.root=Path(root).resolve();self.limit=max_json_bytes;self.results=[];self.remaps=[];self.stage='required_files';self.paper=None;self.policy_defaults={}
    def path(self,name,remap=False):
        p=Path(name)
        candidate=self.root/p if not p.is_absolute() else None
        if candidate is not None:
            candidate=candidate.resolve()
            if candidate.is_relative_to(self.root) and candidate.is_file():return candidate
        if remap:
            candidate=(self.root/'inputs'/p.name).resolve()
            if candidate.is_relative_to(self.root) and candidate.is_file():
                self.remaps.append({'declared':str(name),'local':str(candidate.relative_to(self.root))});return candidate
        raise ValueError('missing or unsafe artifact path: '+str(name))
    def load(self,name):
        p=self.path(name);require(p.stat().st_size<=self.limit,'JSON stored bytes over checker limit: '+name)
        opener=gzip.open if p.suffix=='.gz'else open
        with opener(p,'rt',encoding='utf-8')as f:return json_value(f.read(self.limit+1))
    def done(self,name,detail):self.results.append({'check':name,'status':'PASS','detail':detail})
    def record(self,row,json_contract=False,streamed_contract=False):
        p=self.path(row['path']);require(row['stored_sha256']==sha(p),'stored SHA: '+row['path'])
        require(row['stored_bytes']==p.stat().st_size,'stored bytes: '+row['path'])
        compressed=row['encoding'] in ('gzip_json','gzip_jsonl','gzip_csv')
        require(compressed==(p.suffix=='.gz'),'encoding/suffix mismatch: '+row['path'])
        h=hashlib.sha256();n=0
        with (gzip.open(p,'rb')if compressed else p.open('rb'))as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
        key='uncompressed_canonical_'if json_contract else'uncompressed_'
        require(row[key+'sha256']==h.hexdigest() and row[key+'bytes']==n,'uncompressed SHA/bytes: '+row['path'])
        if json_contract:
            if streamed_contract:
                # Semantic equality is proved by stream_resource_contract,
                # not by returning this already-checked byte digest.
                return p,h.hexdigest(),n
            require(n<=self.limit,'expanded JSON over checker limit: '+row['path'])
            obj=self.load(row['path']);digest=semantic(obj)
            require(digest==row['canonical_semantic_sha256']==h.hexdigest(),'canonical semantic SHA: '+row['path'])
            return obj
        return p
    def stream_resource_contract(self,rf,index,allocated,*,chunk_chars=1024*1024):
        row=rf['audit_contract'];path,byte_digest,expanded_bytes=self.record(row,True,True)
        native=rf['native_rank_contracts'];expected={str(r)for r in range(allocated)}
        require(set(native)==set(index['rank_files'])==expected,'resource rank coverage')
        semantic_hash=hashlib.sha256();semantic_hash.update(b'{')
        resource_nodes={};seen_top=set();seen_ranks=set();log_prefix=None
        with (gzip.open(path,'rb')if path.suffix=='.gz'else path.open('rb'))as f:
            parser=StreamingJSON(f,self.limit,chunk_chars)
            for key in parser.keys():
                if seen_top:semantic_hash.update(b',')
                update_semantic(semantic_hash,key);semantic_hash.update(b':');seen_top.add(key)
                if key=='ranks':
                    semantic_hash.update(b'{')
                    for rank in parser.keys():
                        require(rank in expected,'unexpected resource rank: '+rank)
                        if seen_ranks:semantic_hash.update(b',')
                        update_semantic(semantic_hash,rank);semantic_hash.update(b':')
                        obj=parser.value();require(isinstance(obj,dict),'resource rank must be an object')
                        update_semantic(semantic_hash,obj)
                        native_obj=self.record(native[rank],True)
                        require(obj==native_obj and semantic(obj)==native[rank]['canonical_semantic_sha256'],
                            'unified/native rank semantic mismatch: '+rank)
                        require(self.path(index['rank_files'][rank],True)==self.path(native[rank]['path']),
                            'native resource index mapping: '+rank)
                        resource_nodes[int(rank)]=set(map(int,obj['nodes']))
                        seen_ranks.add(rank);del obj,native_obj
                    semantic_hash.update(b'}')
                else:
                    obj=parser.value();update_semantic(semantic_hash,obj)
                    if key=='log_prefix':log_prefix=obj
                    del obj
            parser.finish();semantic_hash.update(b'}')
            largest_value=parser.max_value_bytes
        require('ranks'in seen_top and seen_ranks==expected,'resource rank coverage')
        require('log_prefix'in seen_top and log_prefix==index['log_prefix'],'resource log prefix mismatch')
        digest=semantic_hash.hexdigest()
        require(digest==row['canonical_semantic_sha256']==byte_digest,'canonical semantic SHA: '+row['path'])
        return resource_nodes,dict(ranks=len(native),streamed_unified_contract=True,
            unified_expanded_bytes=expanded_bytes,largest_decoded_value_bytes=largest_value,
            max_single_value_bytes=self.limit,canonical_semantic_sha256=digest,
            top_level_keys=sorted(seen_top),semantic_SHA_reencoded_from_all_top_fields=True)
    def rows(self,row):
        p=self.path(row['path']);opener=gzip.open if p.suffix=='.gz'else open
        with opener(p,'rt',encoding='utf-8',newline='')as f:
            if row['encoding']=='gzip_jsonl':
                for line in f:
                    require(bool(line.strip()),'blank JSONL record');yield json_value(line)
            else:
                reader=csv.DictReader(f);require(bool(reader.fieldnames),'missing CSV header')
                require(len(reader.fieldnames)==len(set(reader.fieldnames)),'duplicate CSV columns')
                for value in reader:yield value
    def run(self):
        missing=sorted(x for x in REQUIRED if not(self.root/x).is_file());require(not missing,'missing required files: '+str(missing))
        m=self.load('manifest.json');art=m['artifact_sha256'];require(isinstance(art,dict)and art,'empty artifact inventory')
        actual={p.relative_to(self.root).as_posix()for p in self.root.rglob('*')if p.is_file()and p.name!='manifest.json'}
        require(set(art)==actual,'artifact inventory mismatch: missingSHA='+str(sorted(actual-set(art)))+' missingFile='+str(sorted(set(art)-actual)))
        for name,digest in art.items():require(isinstance(digest,str)and len(digest)==64 and sha(self.path(name))==digest,'artifact SHA: '+name)
        inputs=m['inputs'];require(isinstance(inputs,dict)and inputs,'empty input inventory')
        for name,row in inputs.items():require(isinstance(row.get('sha256'),str)and sha(self.path(row['path'],True))==row['sha256'],'input SHA: '+name)
        self.done('manifest_all_artifact_and_input_SHA',{'artifacts':len(art),'inputs':len(inputs)})
        self.stage='status_exit_checks_metrics';metrics=self.load('metrics.json');self.paper=metrics.get('paper_performance_ready')
        summary=self.load('summary.json');status=self.load('status.json');cmd=self.load('astra_command.json');checks=self.load('checks.json')['checks']
        require(m['status']==status['status']=='success' and summary['status']=='PASS','success status disagreement')
        require(type(m['exit_status'])is int and m['exit_status']==0 and type(cmd['exit_status'])is int and cmd['exit_status']==0,'missing/nonzero actual native exit')
        require(cmd['argv']==m['command'],'native argv mismatch')
        require(bool(cmd.get('started_at_utc'))and bool(cmd.get('finished_at_utc')),'actual native timestamps missing')
        require(cmd['started_at_utc']==m['started_at_utc']and cmd['finished_at_utc']==m['simulator_finished_at_utc'],'native timestamp mismatch')
        require(m['software']['astrasim_commit']==PINNED_ASTRA and m['execution_host']['inside_docker']is True,'pinned Docker/Astra provenance missing')
        require(bool(m['software'].get('container_image_tag'))and bool(m['software'].get('container_image_id')),'Docker image missing')
        require(m['metrics_sha256']==sha(self.path('metrics.json'))and summary['metrics']==metrics,'metrics SHA/summary mismatch')
        require(isinstance(checks,list)and checks and all(x.get('passed')is True for x in checks),'missing/failed checks')
        require(len({x['check_id']for x in checks})==len(checks),'duplicate check IDs')
        for x in [m,summary]:require(x['checks_passed']==len(checks)and x['checks_failed']==0,'check count mismatch')
        for k,v in m['headline_metrics'].items():require(k in metrics and metrics[k]==v,'headline metric mismatch: '+k)
        for flag in ['paper_performance_ready','numerical_training','hardware_calibrated','full_verl_runtime_reproduced']:
            require(metrics.get(flag)is False,'unverified scope flag missing/not false: '+flag)
        self.done('status_native_exit_checks_and_metrics',{'checks':len(checks),'paper_performance_ready':self.paper})
        self.stage='table_formats';fmt=self.load('table_format.json');require(fmt['schema_version']=='rlxy.aligned_tables/1','unsupported table format')
        tables=fmt['tables'];require(set(tables)==TABLES,'table set missing/unknown')
        counts={}
        for name,row in tables.items():
            self.record(row);count=sum(1 for _ in self.rows(row));require(type(row['row_count'])is int and count==row['row_count'],'table row count: '+name);counts[name]=count
        check_csv=list(self.rows(tables['checks.csv']))
        require({x['check_id']:x['passed'].lower()for x in check_csv}=={x['check_id']:'true'for x in checks},'checks CSV/JSON mismatch')
        self.done('table_stored_uncompressed_SHA_bytes_rows',counts)
        self.stage='resource_contract_formats';rf=self.load('resource_contract_format.json');require(rf['schema_version']=='rlxy.resource_contract_format/1','unsupported resource format')
        index=self.load(rf['native_index_path'])
        resource_nodes,resource_detail=self.stream_resource_contract(rf,index,metrics['dies'])
        self.done('gzip_unified_and_per_rank_semantic_SHA',resource_detail)
        self.stage='policy_counts_versions';resolved=self.load('resolved_config.json');ac=self.load('aligned_contract.json');pre=self.load('policy_preflight.json');policy=resolved['policy'];case=resolved['case']
        require(policy==metrics['policy']==pre['policy'],'policy dictionaries differ')
        require(pre.get('runtime_alignment_ready')is True and pre.get('status')=='VALID_CONFIGURATION_RUNTIME_ALIGNED_COST_PENDING','runtime preflight not ready/cost-pending')
        require(ac['policy_preflight']==pre and metrics['aligned_protocol']==ac['audit'],'aligned audit/preflight metadata differ')
        policy_input=[x for x in inputs.values()if Path(x['path']).name=='policy.json'];require(len(policy_input)==1,'one policy input required')
        pi=json_value(self.path(policy_input[0]['path'],True).read_text());pi=pi['policy']if set(pi)=={'schema_version','policy'}else pi
        # These three False values are the explicit pinned producer defaults.
        # Do not fill any other missing field or alter a supplied value.
        self.policy_defaults={k:v for k,v in PINNED_INPUT_POLICY_DEFAULTS.items()if k not in pi}
        pi={**self.policy_defaults,**pi}
        require(pi==policy,'input policy differs from resolved policy')
        fp=semantic(policy,True);require(fp==metrics['policy_fingerprint']==ac['audit']['policy_fingerprint']==pre['policy_sha256']==self.load('actual_gemm_census.json')['policy_sha256'],'policy fingerprint disagreement')
        rounds=case['rounds'];v0=case['workload']['behavior_version'];r0=case['workload']['round_id'];active=metrics['active_devices'];allocated=metrics['dies']
        ts=case['workload']['trajectories'];response=sum(t.get('response_length',len(t.get('response_token_ids',[])))for t in ts)
        require(len(ts)==metrics['trajectory_count']==policy['prompt_batch_size']*policy['responses_per_prompt']and response==metrics['response_tokens'],'trajectory/response count mismatch')
        require(rounds==metrics['rounds']==metrics['optimizer_updates']and metrics['actor_version_end']==v0+rounds and metrics['reference_version_end']==v0,'round/update/version mismatch')
        require(active==resolved['active_devices']==ac['audit']['active_devices']and allocated==resolved['allocated_devices']==ac['audit']['allocated_devices'],'active/allocated mismatch')
        require(metrics['data_parallel']==active//policy['train_tp']and metrics['rollout_data_parallel']==active//policy['rollout_tp'],'TP/DP mismatch')
        require(metrics['tensor_parallel']==policy['train_tp']and metrics['rollout_tensor_parallel']==policy['rollout_tp'],'TP metadata mismatch')
        require(metrics['inactive_devices']==list(range(active,allocated))==ac['audit']['idle_devices'],'idle rank mismatch')
        self.done('policy_FP_counts_and_final_versions',{'policy_sha256':fp,'trajectories':len(ts),'response_tokens':response})
        self.stage='operations_events_dependencies';ops={};expected={};dependencies={};publications={};adam={};last_node=0
        for o in self.rows(tables['operations.jsonl']):
            n=o['node_id'];ps=o['participant_ranks'];meta=o['metadata'];kind=o['kind']
            require(type(n)is int and n>last_node,'duplicate/nonincreasing operation node');last_node=n
            require(ps and len(set(ps))==len(ps)and all(type(r)is int and 0<=r<allocated for r in ps),'bad participants')
            require(kind in ['compute','collective','p2p'],'unknown operation kind')
            if kind=='compute':require(type(o['duration_ns'])is int and o['duration_ns']>0,'bad compute duration')
            if kind=='p2p':require(set(ps)=={o['src_rank'],o['dst_rank']}and o['src_rank']!=o['dst_rank'],'bad p2p endpoints')
            rid=o['round_id'];require(type(rid)is int and r0<=rid<r0+rounds,'operation round outside case')
            if o['name']=='publish_actor_version':
                require(rid not in publications and set(ps)==set(range(active)),'publication coverage');publications[rid]=n
                require(meta['actor_version_in']==v0+rid-r0 and meta['actor_version_out']==v0+rid-r0+1 and meta['reference_version_in']==meta['reference_version_out']==v0,'publication version mismatch')
            if o['name'].startswith('Adam_owned_shard_'):
                require(meta['actor_version_in']==v0+rid-r0 and meta['actor_version_out']==v0+rid-r0+1,'Adam version mismatch')
                adam.setdefault(rid,[]).extend(ps)
            if 'reference_version'in meta:require(meta['reference_version']==v0,'Reference version changed')
            if 'actor_version'in meta:require(meta['actor_version']==v0+rid-r0,'stale Actor version')
            for rank in ps:
                require(rank<active or meta.get('idle_rank_marker')is True,'work on idle rank')
                key=(rank,n);expected[key]=4 if kind=='compute'else 7 if kind=='collective'else 5 if rank==o['src_rank']else 6
                deps=o['dependencies_by_rank'].get(str(rank),o['dependencies']);require(len(set(deps))==len(deps)and all(type(d)is int and 0<d<n for d in deps),'bad dependency IDs');dependencies[key]=deps
            ops[n]=(o['op_id'],rid,o['phase'],kind,o['duration_ns'])
        require(len(ops)==metrics['operation_count']==counts['operations.jsonl'],'operation count mismatch')
        for rid in range(r0,r0+rounds):require(rid in publications and sorted(adam.get(rid,[]))==list(range(active)),'missing/duplicate per-rank Adam or publication')
        events={};last_callback={};maximum=0
        for e in self.rows(tables['events.jsonl']):
            key=(e['rank'],e['node_id']);require(key in expected and key not in events,'missing/duplicate/unknown event key')
            opid,rid,phase,kind,duration=ops[e['node_id']]
            require((e['op_id'],e['round_id'],e['phase'],e['node_type'])==(opid,rid,phase,expected[key]),'event identity/type mismatch')
            start,end=e['issue_tick'],e['callback_tick'];require(type(start)is int and type(end)is int and 0<=start<=end,'negative/noninteger event clocks')
            delta=end-start;require(type(e['duration_ticks'])is int and delta==e['duration_ticks'],'event duration mismatch')
            require((e['issue_ns'],e['callback_ns'],e['duration_ns'])==(start/1000,end/1000,delta/1000),'tick/ns mismatch')
            require(kind!='compute'or delta==duration*1000,'exact compute ticks differ')
            events[key]=(start,end);maximum=max(maximum,end)
        require(set(events)==set(expected)and len(events)==metrics['operation_rank_event_count']==counts['events.jsonl'],'event coverage/count mismatch')
        for key,deps in dependencies.items():
            require(all((key[0],d)in events and events[key[0],d][1]<=events[key][0]for d in deps),'dependency clock violation')
        for rank in range(allocated):
            nodes=sorted(n for r,n in expected if r==rank);require(set(nodes)==resource_nodes[rank],'resource/operation node coverage')
            for n in nodes:
                require(rank not in last_callback or events[rank,n][0]>=last_callback[rank],'overlapping local clocks');last_callback[rank]=events[rank,n][1]
        require(metrics['total_simulated_time_ns']==maximum/1000,'metrics total time differs from actual event clocks')
        self.done('streamed_operation_event_coverage_compute_ticks_dependencies',{'operations':len(ops),'rank_events':len(events),'max_callback_tick':maximum})
        self.stage='component_guard_and_invocations';guard=self.load('component_template_signature_guard.json');proofs=self.load('component_logical_proofs.json');inv=ac['component_invocations'];templates=ac['component_templates'];sigs=guard['template_signatures']
        require(guard['status']=='PASS'and guard['every_invocation_native_macro_clocks_validated']is True,'template signature guard missing/failed')
        # Frozen producer caches by contraction object identity, but its
        # signature/certificate dictionaries deduplicate by semantic SHA.
        # "unique_templates" is therefore a historical instance-count name.
        require(len(sigs)==len(templates)>0,'distinct semantic template counts differ')
        require(type(guard['unique_templates'])is int and type(guard['full_template_replays'])is int
            and guard['unique_templates']==guard['full_template_replays']>=len(sigs),'template instance/full-replay counts differ')
        require({sig['certificate_sha256']for sig in sigs.values()}==set(templates),'semantic signature/certificate coverage differs')
        for key,sig in sigs.items():
            require(key==semantic(sig,True)and sig['certificate_sha256']in templates,'template signature identity')
            cert=templates[sig['certificate_sha256']];require(semantic(cert,True)==sig['certificate_sha256']and semantic(cert['blocks'],True)==sig['blocks_sha256'],'certificate/block signature mismatch')
            require(sig['logical_operations_sha256']==cert['logical_operations_sha256']and sig['logical_contract_sha256']==cert['logical_resources']['original_contract_sha256'],'logical certificate signature mismatch')
        require(len(proofs)==len(inv)==guard['invocations']==metrics['component_invocation_proof_count'],'invocation count mismatch')
        require(len({x['label']for x in inv})==len(inv)and set(proofs)=={x['label']for x in inv},'invocation labels differ')
        assigned=set();proof_templates=set();full=reused=represented=expanded=0
        for invocation in inv:
            p=proofs[invocation['label']];cert=templates[invocation['certificate_sha256']];mapping={int(k):v for k,v in invocation['global_to_local'].items()}
            require(not(assigned&set(mapping))and set(mapping)<=set(ops),'component node mapping missing/duplicate');assigned.update(mapping)
            require(set(mapping.values())=={b['native_node_id']for b in cert['blocks']}and len(set(mapping.values()))==len(mapping),'component block mapping coverage')
            require(p['status']=='PASS'and p['template_sha256']in sigs and sigs[p['template_sha256']]['certificate_sha256']==invocation['certificate_sha256'],'invocation template mismatch')
            proof_templates.add(p['template_sha256'])
            for flag in ['all_native_macro_clocks_validated','network_issue_callback_ticks_preserved','exact_compute_durations_validated','strict_per_rank_serialization_validated','local_dependencies_validated']:require(p.get(flag)is True,'invocation clocks/flag missing: '+flag)
            require(p['physical_ranks']==invocation['physical']and p['logical_operations_sha256']==cert['logical_operations_sha256'],'invocation placement/signature differs')
            represented+=p['logical_events_represented'];expanded+=p['expanded_logical_events']
            if p['mode']=='full_template_replay':full+=1;require(p['clock_bound_to_expanded_native_events']is True,'full replay not clock bound')
            elif p['mode']=='reused_verified_serial_template':reused+=1;require(p['clock_bound_by_verified_serial_template']is True and p['expanded_logical_events']==0,'reused proof scope mismatch')
            else:raise ValueError('unknown proof mode')
        require((full,reused)==(guard['full_template_replays'],guard['reused_serial_template_invocations']),'replay/reuse counts differ')
        require(proof_templates==set(sigs),'semantic template lacks invocation proof')
        require(full==metrics['component_logical_replay_count']and reused==metrics['component_reused_serial_proof_count']and represented==metrics['logical_component_event_count']and expanded==metrics['materialized_logical_component_event_count'],'proof metric totals differ')
        self.done('template_signature_guard_and_every_invocation_clock_flags',{
            'distinct_semantic_signatures':len(sigs),'contraction_instances':guard['unique_templates'],
            'producer_unique_templates_field_scope':'contraction object instance count; historical producer name',
            'invocations':len(inv),'full':full,'reused':reused})
        self.stage='unchanged_artifacts';require(all(sha(self.path(k))==v for k,v in art.items()),'artifact changed during verification')
        self.done('artifacts_unchanged_after_verification',len(art))


def verify_bundle(root,*,max_json_bytes=512*1024**2):
    c=Checker(root,max_json_bytes)
    manifest_path=c.root/'manifest.json';checker_path=Path(__file__).resolve()
    manifest_digest=None;checker_digest=None
    try:
        checker_digest=sha(checker_path)
        if manifest_path.is_file():manifest_digest=sha(manifest_path)
        c.run()
        c.stage='verification_bindings_unchanged'
        require(sha(manifest_path)==manifest_digest,'manifest changed during verification')
        require(sha(checker_path)==checker_digest,'checker source changed during verification')
        c.done('manifest_and_checker_source_bindings_unchanged',True)
        status='PASS'
    except Exception as exc:
        status='FAIL';c.results.append({'check':c.stage,'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)})
    return {'schema_version':'rlxy.aligned_result_check/1','status':status,'run_dir':str(c.root),
        'manifest_sha256':manifest_digest,'checker_source_sha256':checker_digest,
        'checker_scope':'MVP independent artifact/encoding/metadata/native-exit/event-index/dependency checks; not a new simulation or paper release.',
        'checks':c.results,'input_path_remaps':c.remaps,'paper_performance_ready_from_input':c.paper,
        'pinned_input_policy_defaults_applied':c.policy_defaults,
        'paper_readiness_promoted_by_checker':False,'max_single_JSON_bytes':max_json_bytes,
        'not_independently_recomputed':['raw rotated debug issue/callback records vs exported event rows','full object alloc/read/READY/free/private-SRAM ledger','logical component source IR or original native macro event stream SHA; guard records/certificates are cross-checked,not substituted for source reconstruction','cost calibration,model numerical training,learning quality,or paper readiness']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--max-json-bytes',type=int,default=512*1024**2);a=p.parse_args()
    if a.out.exists():p.error('output exists; preserve earlier verification')
    if a.out.resolve().is_relative_to(a.run_dir.resolve()):p.error('write verification outside immutable run bundle')
    if a.max_json_bytes<=0:p.error('positive JSON limit required')
    result=verify_bundle(a.run_dir,max_json_bytes=a.max_json_bytes);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status']=='PASS'else 2


if __name__=='__main__':raise SystemExit(main())
