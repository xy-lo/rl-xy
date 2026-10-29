"""Exact vectorized enumeration of the original OS tile set and min key.

No interpolation, different mapping candidates, throughput change or retiming.
Only immutable tile candidates are cached; all work/byte expressions use int64
after explicit overflow rejection. Original tuple tie ordering is preserved.
The production source files are not edited; installation is runner-local.
"""
from functools import lru_cache
from itertools import product
import numpy as np
from rl_xy.mapping import MappingEstimate

@lru_cache(maxsize=4096)
def tiles(m,n,k,cap,a,b,acc,max_tile):
    dims=(m,n,k);powers=(1,2,4,8,16,32,64,128,256,max_tile)
    choices=[sorted({min(dim,x) for x in powers if x<=max_tile}) for dim in dims]
    raw=np.array(list(product(*choices)),dtype=np.int64)
    work=raw[:,0]*raw[:,2]*a+raw[:,2]*raw[:,1]*b+raw[:,0]*raw[:,1]*acc
    legal=work<=cap
    return raw[legal],work[legal]

def exact_choose_os_mapping(p,capacity,max_tile=32):
    if type(capacity) is not int or capacity<=0 or type(max_tile) is not int or max_tile<=0:raise ValueError('invalid SRAM/tile')
    # Each tensor byte count is far below 2**63 for the declared experiment.
    worst=p.batches*(p.m*p.k*p.n*p.a_bytes+p.k*p.n*p.m*p.b_bytes+p.m*p.n*p.output_bytes)
    if worst>=2**63:raise OverflowError('exact mapper int64 domain exceeded')
    t,w=tiles(min(p.m,max_tile),min(p.n,max_tile),min(p.k,max_tile),capacity,p.a_bytes,p.b_bytes,p.accumulator_bytes,max_tile)
    if not len(t):raise ValueError('no legal mapping')
    reads=p.batches*(p.m*p.k*((p.n+t[:,1]-1)//t[:,1])*p.a_bytes+p.k*p.n*((p.m+t[:,0]-1)//t[:,0])*p.b_bytes)
    write=p.batches*p.m*p.n*p.output_bytes
    # np.lexsort last key is primary: traffic, then -K,-M,-N, exactly original.
    idx=int(np.lexsort((-t[:,1],-t[:,0],-t[:,2],reads+write))[0])
    return MappingEstimate(tuple(int(x) for x in t[idx]),int(w[idx]),int(reads[idx]),write,p.flops)

def install():
    import rl_xy.capacity_mapping as cm
    import rl_xy.multicore as mc
    cm.choose_os_mapping=exact_choose_os_mapping
    mc.choose_os_mapping=exact_choose_os_mapping

def validate():
    import random,time,json
    from rl_xy.mapping import choose_os_mapping
    from rl_xy.operator_ir import Matmul
    rng=random.Random(7603);dims=[1,2,3,7,16,31,32,63,64,127,128,255,256,511,512,513,1023,2560,4352,60625,151936]
    rows=[];slow=fast=0.
    for i in range(512):
        p=Matmul(*[rng.choice(dims) for _ in range(3)],a_bytes=rng.choice([2,4]),b_bytes=rng.choice([2,4]),output_bytes=rng.choice([2,4]),batches=rng.choice([1,2,7]))
        cap=rng.choice([167936,625000,1250000]);mt=rng.choice([32,512])
        start=time.perf_counter();a=choose_os_mapping(p,cap,mt);slow+=time.perf_counter()-start
        start=time.perf_counter();b=exact_choose_os_mapping(p,cap,mt);fast+=time.perf_counter()-start
        assert a==b,(p,cap,mt,a,b)
    return {'status':'PASS','cases':512,'seed':7603,'original_seconds':slow,'vectorized_seconds':fast,
        'scope':'Exact tile tuple/workspace/read/write/FLOP equality for unchanged bounded candidate search; no physical calibration'}

def integrated_validate():
    from pathlib import Path
    import json,time,hashlib
    from dataclasses import asdict
    from independent_checks import OUT,DATA,REPO
    from rl_xy.qwen3_cost import Operators
    from rl_xy.operator_ir import Matmul,KernelSpec
    import rl_xy.capacity_mapping as cm
    import rl_xy.multicore as mc
    rows=json.loads((OUT/'independent-checks.json').read_text())['rows']
    cases=[];expected=[];start=time.perf_counter()
    for r in rows:
        op=Operators(REPO,DATA/'inputs'/('Qwen3-'+r['model'])/'config.json','wafer')
        p=Matmul(r['m'],r['n'],r['k'],output_bytes=4 if r['direction'] in ('dinput','dweight') else 2,direction=r['direction'])
        spec=KernelSpec(r['label'],matmuls=(p,));cases.append((op,spec))
        expected.append(asdict(op.provider.estimate(spec,op.hw)))
    original=time.perf_counter()-start
    cm.local_mapping.cache_clear();cm.capacity_partition.cache_clear();mc.partition_matmul.cache_clear()
    install();start=time.perf_counter()
    for (op,spec),ref in zip(cases,expected):
        actual=asdict(op.provider.estimate(spec,op.hw));assert actual==ref,(spec,actual,ref)
    return {'status':'PASS','cases':len(cases),'compared':'All CostEstimate fields including duration/compute/memory/reads/writes/workspaces/mapping/core allocation/partition metadata',
        'original_seconds':original,'vectorized_seconds':time.perf_counter()-start,
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy_version':np.__version__}
if __name__=='__main__':
    import json
    print(json.dumps({'random_tail_cases':validate(),'integrated_representative_cases':integrated_validate()},indent=2))
