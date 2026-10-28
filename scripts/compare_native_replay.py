#!/usr/bin/env python3
"""Independent streaming native-replay equivalence; no runtime dependencies.

This proves the observed native event/resource streams for one fixed workload.
It does not turn a failed wrapper/post phase into a successful workflow, decode
Chakra ET semantics, publish GPU/wafer speedup, or calibrate hardware.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

EVENT=re.compile(r'(?P<action>issue|callback),sys->id=(?P<rank>\d+), tick=(?P<tick>\d+), node->id=(?P<node>\d+), node->name=(?P<name>[^\r\n]+?), node->type=(?P<type>\d+)\s*$')
INTENT=re.compile(r'(?:issue|callback),\s*sys->id')
ERROR=re.compile(r'\] \[(?:error|critical|e|c|E|C)\](?: |$)')
FINISHED=re.compile(r'sys\[(\d+)\] finished, (\d+) cycles, exposed communication (\d+) cycles\.')
ROTATION=re.compile(r'log\.([1-9][0-9]*)\.log$')
CONFIG_FLAGS={'--system-configuration':'system','--network-configuration':'network',
    '--remote-memory-configuration':'remote_memory','--comm-group-configuration':'groups'}
JSON_INPUTS=('case.json','model_config.json','policy.json','workload_provenance.json',
    'source_hardware.json','wafer.json','gemm_measurements.json','compilation.json','gpu_provenance.json')


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def semantic(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()


def sha(path):
    before=path.stat();h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    after=path.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'file changed during read: '+str(path))
    return h.hexdigest()


def strict_json(path,limit):
    before=path.stat();require(before.st_size<=limit,'JSON input exceeds single-file cap: '+str(path))
    def pairs(items):
        result={}
        for key,val in items:
            require(key not in result,'duplicate JSON key: '+key);result[key]=val
        return result
    data=path.read_bytes();after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'JSON changed during read: '+str(path))
    obj=json.loads(data,object_pairs_hook=pairs,
        parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))
    digest=semantic(obj)  # Also rejects finite-parser overflow toInfinity.
    return obj,dict(path=str(path),bytes=len(data),raw_sha256=hashlib.sha256(data).hexdigest(),semantic_sha256=digest)


def network(path):
    """The pinned producer's four-key flat inline-list YAML only."""
    values={}
    for line in path.read_text().splitlines():
        line=line.split('#',1)[0].strip()
        if not line:continue
        require(':'in line,'unsupported network YAML line');key,text=line.split(':',1);key=key.strip();text=text.strip()
        require(key not in values and key in ('topology','npus_count','bandwidth','latency'),'unknown/duplicate network key')
        require(text.startswith('[')and text.endswith(']'),'only inline network lists supported')
        tokens=[x.strip()for x in text[1:-1].split(',')];require(all(tokens),'empty network token')
        parsed=[]
        for token in tokens:
            if key=='topology':
                token=token.strip('"\'');require(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',token),'unsupported topology token');parsed.append(token)
            elif key=='npus_count':
                require(re.fullmatch(r'[1-9][0-9]*',token),'positive network dimension required');parsed.append(int(token))
            else:
                try:d=Decimal(token)
                except InvalidOperation:raise ValueError('invalid network number')
                require(d.is_finite()and abs(d.adjusted())<=1000 and (d>0 if key=='bandwidth'else d>=0),'invalid network service value')
                parsed.append(format(d.normalize(),'f'))
        values[key]=parsed
    require(set(values)=={'topology','npus_count','bandwidth','latency'}and len({len(x)for x in values.values()})==1,'incomplete network schema')
    return values,dict(path=str(path),raw_sha256=sha(path),semantic_sha256=semantic(values),semantic_scope='strict pinned flat network schema; no general YAML parser')


class Bundle:
    def __init__(self,root,ranks,limit):
        self.root=Path(root).resolve();self.ranks=ranks;self.limit=limit;self.remaps=[];self.inputs={}
        self.command,self.command_record=strict_json(self.path('astra_command.json'),limit)
        require(type(self.command.get('exit_status'))is int and self.command['exit_status']==0,'actual native exit0 required')
        require(self.command.get('started_at_utc')and self.command.get('finished_at_utc'),'actual native start/finish receipt required')
        argv=self.command['argv'];require(isinstance(argv,list)and argv,'native argv missing');self.binary_name=Path(argv[0]).name
        self.flags={}
        for arg in argv[1:]:
            require('='in arg,'unsupported native argv');key,val=arg.split('=',1);require(key not in self.flags,'duplicate native option');self.flags[key]=val
        require(set(self.flags)==set(CONFIG_FLAGS)|{'--workload-configuration','--logging-folder'},'missing/unknown native options')
        self.manifest_record=None;self.manifest=None
        if(self.root/'manifest.json').is_file():self.manifest,self.manifest_record=strict_json(self.path('manifest.json'),limit)
        self.outer_report_record=None
        if(self.root.parent/'report.json').is_file():
            obj,record=strict_json(self.root.parent/'report.json',limit)
            self.outer_report_record={**record,'status':obj.get('status'),'exit_status':obj.get('exit_status'),'OOMKilled':obj.get('OOMKilled')}

    def path(self,relative):
        path=(self.root/relative).resolve();require(path.is_relative_to(self.root)and path.is_file(),'missing/unsafe bundle file: '+relative);return path

    def declared(self,text):
        p=Path(text);candidate=(self.root/p).resolve()if not p.is_absolute()else p.resolve()
        if not candidate.is_relative_to(self.root)or not candidate.is_file():candidate=self.path('inputs/'+p.name)
        require(candidate.is_relative_to(self.root)and candidate.is_file(),'native input is outside bundle')
        self.remaps.append(dict(declared=text,local=str(candidate),rule='same bundle input basename; original paths retained'))
        return candidate

    def record_input(self,label,path,obj=None):
        if obj is None:obj,record=strict_json(path,self.limit)
        else:record=dict(path=str(path),raw_sha256=sha(path),semantic_sha256=semantic(obj))
        self.inputs[label]=record;return obj

    def input_contract(self):
        result={}
        for flag,label in CONFIG_FLAGS.items():
            path=self.declared(self.flags[flag])
            if label=='network':obj,record=network(path);self.inputs[label]=record
            else:obj=self.record_input(label,path)
            result[label]=obj
        for name in JSON_INPUTS:
            path=self.root/'inputs'/name
            if path.is_file():result[name]=self.record_input(name,self.path('inputs/'+name))
            else:result[name]=None
        require(all(result[n]is not None for n in ('case.json','model_config.json','policy.json','workload_provenance.json')),'required model/case/policy/provenance missing')
        profile=self.record_input('gpu_profile.json',self.path('inputs/gpu_profile.json'))
        if 'gemm_profile'in profile:
            require(isinstance(profile['gemm_profile'],str)and result['gemm_measurements.json']is not None,'profile path requires copied measurement input')
            self.remaps.append(dict(json_file='gpu_profile.json',key='gemm_profile',declared=profile['gemm_profile'],normalized='copied gemm_measurements.json',rule='copied full measurement semantics are compared independently'))
            profile={**profile,'gemm_profile':'<COPIED_GEMM_MEASUREMENTS>'}
        result['gpu_profile.json']=profile
        census=self.record_input('actual_gemm_census.json',self.path('actual_gemm_census.json'));result['actual_gemm_census.json']=census
        index=self.record_input('resource_index.json',self.path('inputs/resource_index.json'))
        require(set(index['rank_files'])=={str(r)for r in range(self.ranks)},'resource index rank coverage')
        require(isinstance(index['log_prefix'],str)and Path(index['log_prefix']).name=='resources','unexpected resource log prefix')
        self.remaps.append(dict(json_file='resource_index.json',key='log_prefix',declared=index['log_prefix'],normalized='<BUNDLE>/astra_logs/resources'))
        normalized={**index,'log_prefix':'<BUNDLE>/astra_logs/resources','rank_files':{str(r):f'<RANK{r}>'for r in range(self.ranks)}}
        result['resource_index.json']=normalized
        resource_semantics={}
        for rank in range(self.ranks):
            path=self.declared(index['rank_files'][str(rank)]);obj=self.record_input('resource_rank'+str(rank),path)
            resource_semantics[str(rank)]=semantic(obj);del obj
        result['resource_rank_semantic_SHA']=resource_semantics
        self.selected_contract_sha256=semantic(result)
        return result

    def ET_bytes(self):
        prefix=Path(self.flags['--workload-configuration']).name
        self.remaps.append(dict(declared=self.flags['--workload-configuration'],normalized='<BUNDLE>/inputs/'+prefix,rule='ET prefix path only;protobuf semantics not decoded'))
        return {str(r):sha(self.path(f'inputs/{prefix}.{r}.et'))for r in range(self.ranks)}


def raw_events(bundle,expected_events):
    folder=bundle.root/'astra_logs';segments={}
    for path in folder.iterdir():
        if path.name=='log.log':continue
        if path.name.startswith('log.')and path.name.endswith('.log'):
            match=ROTATION.fullmatch(path.name);require(match,'invalid rotated log name: '+path.name);segments[int(match[1])]=path
    require(not segments or set(segments)==set(range(1,max(segments)+1)),'rotation gap')
    paths=[segments[k]for k in sorted(segments,reverse=True)]+[bundle.path('astra_logs/log.log')]
    state=[dict(last_node=0,last_tick=0,open=None,count=0,issues=0,callbacks=0,max_callback=0,hash=hashlib.sha256())for _ in range(bundle.ranks)]
    raw_segments=[]
    for path in paths:
        before=sha(path);raw_segments.append(dict(path=str(path),sha256=before,bytes=path.stat().st_size))
        with path.open(encoding='utf-8')as f:
            for line in f:
                require(not ERROR.search(line),'native error/critical log record')
                if not INTENT.search(line):continue
                require(len(INTENT.findall(line))==1,'multiple native records in one line');match=EVENT.search(line);require(match,'invalid native event record')
                d=match.groupdict();rank=int(d['rank']);node=int(d['node']);tick=int(d['tick']);kind=int(d['type']);action=d['action'];name=d['name']
                require(0<=rank<bundle.ranks and node>0 and kind in (4,5,6,7),'invalid native rank/node/type')
                s=state[rank];require(tick>=s['last_tick'],'rank-local tick decreased')
                if action=='issue':
                    require(s['open']is None and node>s['last_node'],'duplicate/reused/out-of-order node or overlapping rank execution')
                    s['open']=(node,name,kind,tick);s['last_node']=node;s['issues']+=1
                else:
                    require(s['open']is not None and s['open'][:3]==(node,name,kind),'missing/duplicate/mismatched callback')
                    require(tick>=s['open'][3],'callback precedes issue');s['open']=None;s['callbacks']+=1;s['max_callback']=max(s['max_callback'],tick)
                s['last_tick']=tick;s['count']+=1;s['hash'].update((canonical([action,rank,node,tick,name,kind])+'\n').encode())
        require(sha(path)==before,'native log changed after parse')
    require(all(s['issues']>0 and s['issues']==s['callbacks']and s['open']is None for s in state),'missing rank or unpaired native events')
    require(sum(s['count']for s in state)==2*expected_events,'native raw record count differs from fixed expected matrix')
    stdout=bundle.path('astra_stdout.log');completions={}
    with stdout.open(encoding='utf-8')as f:
        for line in f:
            for rank,tick,_ in FINISHED.findall(line):
                rank=int(rank);require(rank not in completions,'duplicate native completion');completions[rank]=int(tick)
    require(set(completions)==set(range(bundle.ranks)),'completion rank coverage')
    require(all(completions[r]==state[r]['max_callback']for r in range(bundle.ranks)),'completion/native callback clock mismatch')
    require(bundle.path('astra_stderr.log').stat().st_size==0,'native stderr is not empty')
    return dict(raw_segments=raw_segments,per_rank={str(r):dict(record_count=s['count'],issues=s['issues'],callbacks=s['callbacks'],max_callback_tick=s['max_callback'],ordered_canonical_event_sha256=s['hash'].hexdigest())for r,s in enumerate(state)},
        raw_event_count=sum(s['count']for s in state),model_max_callback_tick=max(s['max_callback']for s in state),stdout_sha256=sha(stdout))


def compare(left,right,*,ranks=48,expected_rank_events=754584,max_json_bytes=512*1024**2):
    result=dict(schema_version='rlxy.observed_native_replay_equivalence/1',status='FAIL_NOT_EQUIVALENT_OR_INCOMPLETE',
        checker_source_sha256=sha(Path(__file__)),checked_at_utc=datetime.now(timezone.utc).isoformat(),ranks=ranks,expected_paired_rank_events=expected_rank_events,
        checks=[],paper_performance_ready=False,performance_ratio_published=False,
        scope='observed same-workload native event/resource equivalence only; no wrapper/fullworkflow/hardware/numerical-quality release')
    def done(name,detail):result['checks'].append(dict(check=name,status='PASS',detail=detail))
    try:
        require(type(ranks)is int and ranks>0 and type(expected_rank_events)is int and expected_rank_events>=ranks,'positive fixed coverage required')
        a,b=Bundle(left,ranks,max_json_bytes),Bundle(right,ranks,max_json_bytes)
        result['provenance']={label:dict(bundle=str(x.root),original_manifest=x.manifest_record,manifest_declared_status=(x.manifest or {}).get('status'),native_command=x.command_record,outer_report=x.outer_report_record)for label,x in [('left',a),('right',b)]}
        require(a.binary_name==b.binary_name,'native binary name differs')
        binaries=[(x.manifest or {}).get('software',{}).get('astrasim_binary_sha256')for x in (a,b)]
        if all(binaries):require(binaries[0]==binaries[1],'declared native binary SHA differs')
        result['declared_native_binary_sha256']=dict(left=binaries[0],right=binaries[1],independently_measured_by_this_tool=False,both_available_and_equal=bool(all(binaries)and binaries[0]==binaries[1]))
        left_contract,right_contract=a.input_contract(),b.input_contract();require(left_contract==right_contract,'selected input semantics differ')
        done('same_selected_input_semantics',a.selected_contract_sha256)
        result['inputs']={label:dict(records=x.inputs,tool_computed_selected_input_manifest_sha256=semantic(x.inputs),normalized_contract_sha256=x.selected_contract_sha256,explicit_path_normalizations=x.remaps)for label,x in [('left',a),('right',b)]}
        expected_resources={f'resources.{r}.jsonl'for r in range(ranks)}
        for x in(a,b):require({p.name for p in(x.root/'astra_logs').glob('resources.*.jsonl')}==expected_resources,'raw resource rank coverage differs')
        resources={str(r):dict(left_sha256=sha(a.path(f'astra_logs/resources.{r}.jsonl')),right_sha256=sha(b.path(f'astra_logs/resources.{r}.jsonl')))for r in range(ranks)}
        require(all(v['left_sha256']==v['right_sha256']for v in resources.values()),'rank resource raw bytes differ');done('all_rank_resource_raw_bytes_SHA_equal',resources)
        events_left,events_right=raw_events(a,expected_rank_events),raw_events(b,expected_rank_events)
        require(events_left['per_rank']==events_right['per_rank'],'ordered rank-local native event/tick/name/type streams differ')
        require(events_left['model_max_callback_tick']==events_right['model_max_callback_tick'],'model max tick differs')
        done('complete_fixed_matrix_and_ordered_per_rank_events_equal',dict(per_rank=events_left['per_rank'],total_raw_records=events_left['raw_event_count']))
        et_left,et_right=a.ET_bytes(),b.ET_bytes();result['ET_bytes']=dict(left=et_left,right=et_right,differing_ranks=[str(r)for r in range(ranks)if et_left[str(r)]!=et_right[str(r)]],
            semantic_equality_independently_checked=False,scope='raw ET bytes only; differences do not establish an operation difference without Chakra protobuf decoding')
        result.update(status='PASS_OBSERVED_NATIVE_REPLAY_EQUIVALENCE',native_logs=dict(left=events_left,right=events_right),
            model_max_tick_equal=True,model_max_callback_tick=events_left['model_max_callback_tick'],wrapper_success_implied=False,
            limitations=['rank-local strict serial protocol and monotonically increasing node IDs required; not a generic overlapping-workload parser',
                'only the declared flat four-key analytical network YAML schema is supported',
                'ET protobuf semantics and full logical model IR not independently decoded',
                'failed wrapper/post remains failed even when native streams match; no GPU/wafer ratio or paper release'])
    except Exception as exc:result['checks'].append(dict(check='native_replay_gate',status='FAIL',error_type=type(exc).__name__,error=str(exc)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('left','right','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--ranks',type=int,default=48);p.add_argument('--expected-rank-events',type=int,default=754584);a=p.parse_args()
    if a.out.exists():p.error('preserve existing analysis')
    if any(a.out.resolve().is_relative_to(x.resolve())for x in (a.left,a.right)):p.error('analysis must be outside original bundles')
    result=compare(a.left,a.right,ranks=a.ranks,expected_rank_events=a.expected_rank_events)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ['status','checker_source_sha256','paper_performance_ready','performance_ratio_published']}))
    return 0 if result['status']=='PASS_OBSERVED_NATIVE_REPLAY_EQUIVALENCE'else 2


if __name__=='__main__':raise SystemExit(main())
