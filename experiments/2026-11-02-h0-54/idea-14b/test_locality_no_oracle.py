import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from types import SimpleNamespace as NS
from locality_runtime_v3 import AttentionRuntime
from rl_xy.qwen3_cost import Cost
class Op:
 model=NS(query_heads=32,kv_heads=8,head_dim=128,hidden_size=2560,query_width=4096,layers=36,actor_dtype_bytes=2,qkv_bias=False,qk_norm=True)
 hw=NS(d2d_bytes_per_ns_per_direction=1000,d2d_latency_ns=100)
 def vec(self,*args):return Cost(memory=10)
 def forward(self,q,tp,mode,ctx):return [NS(name='attention_out',cost=Cost(compute=10000 if tp==4 else 5000))]
a=AttentionRuntime.__new__(AttentionRuntime);a.op=Op();a.completed_history=list(range(100,3000,20));a.rep={0:NS(active=[[{'prompt_length':50,'response_length':2000},250,1800]],ranks=(0,1,2,3)),1:NS(ranks=(6,7,8,9))}
x=a.estimate_locality(0,1)
a.rep[0].active[0][0]['response_length']=8000;a.rep[0].active[0][2]=99999
y=a.estimate_locality(0,1)
assert x==y,(x,y)
assert x['hops']==1 and x['predicted_steps']>0
print('PASS: estimator ignores future response length and hidden remaining count')
