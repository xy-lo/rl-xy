"""Native compact-vs-expanded equivalence with overlapping wafer TP routes."""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.qwen3_cost import Operators,Cost
from rl_xy.qwen3_native import Plan,execute
from rl_xy.qwen3_sequence import generation_program
from rl_xy.types import Operation
root=Path(__file__).resolve().parents[1]
from rl_xy.qwen3_paths import input_root,output_path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs',type=Path,default=input_root(root))
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();out=output_path(root,args.output)
if out.exists():raise ValueError('Choose a new output directory; existing evidence is preserved')
requests=[{'trajectory_id':str(i),'group_id':str(i//8),'prompt_length':41+i%5,'response_length':1+(i%4)*3} for i in range(16)]
checks=[]
for platform in ['gpu','wafer']:
 op=Operators(root,args.inputs/'Qwen3-4B/config.json',platform)
 compact,seq,_=generation_program(op,requests,4,2)
 a=execute(compact,out/(platform+'-compact'),sequence=seq)
 expanded=Plan(op);expanded.groups=dict(compact.groups);expanded.groupids=dict(compact.groupids)
 for key,pg in seq['programs'].items():
  ranks=tuple(compact.groups[key])
  for segment in pg['segments']:
   for iteration in range(segment['repeat']):
    for s in segment['steps']:
     kind=s['kind']
     if kind=='local':
      ps=tuple(s.get('ranks',ranks));cost=Cost(s.get('compute',0)+iteration*s.get('compute_delta',0),s.get('memory',0)+iteration*s.get('memory_delta',0));expanded.local(ps,cost,'expanded_local')
     elif kind=='AR':expanded.append(Operation(0,'expanded_ar',1,'test','expanded_ar','collective',payload_bytes=s['bytes'],collective='ALL_REDUCE',participant_ranks=ranks,metadata={'comm_group':key}))
     else:expanded.append(Operation(0,'expanded_p2p',1,'test','expanded_p2p','p2p',payload_bytes=s['bytes'],src_rank=s['src'],dst_rank=s['dst'],participant_ranks=(s['src'],s['dst'])))
 b=execute(expanded,out/(platform+'-expanded'))
 delta=max(abs(a['per_rank_completion_ns'][r]-b['per_rank_completion_ns'][r]) for r in range(compact.world))
 checks.append({'platform':platform,'compact_ns':a['model_ns'],'expanded_ns':b['model_ns'],'max_rank_delta_ns':delta,'passed':delta<0.01,'expanded_nodes':len(expanded.operations),'compact_nodes':len(compact.operations)})
 print(checks[-1],flush=True)
(out/'verification.json').write_text(json.dumps({'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':checks},indent=2)+'\n')
assert all(c['passed'] for c in checks)
