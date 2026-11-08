"""Extend the service directory to every legal power-of-two microbatch."""
from .cost_catalog import ServiceCatalog
import time,json,math
from pathlib import Path

class AdaptiveServiceCatalog(ServiceCatalog):
    @classmethod
    def extend(cls,base,op,policy,output):
        import numpy as np
        from rl_xy.qwen3_native import NetworkProfiles
        net=NetworkProfiles(op,'unused',native=False);start=time.perf_counter();data=dict(base.data);table=dict(data['table']);errors=[]
        for tp in (2,4,8):
            maxima={'teacher':policy.logprob_mb*(tp//2),'training':policy.train_mb if tp==2 else min(8,policy.train_mb*tp)}
            for stage,maximum in maxima.items():
                batches=[2**k for k in range(int(math.log2(maximum))+1)]
                for batch in batches:
                    key=f'{stage}:{tp}:{batch}'
                    if key in table:continue
                    design=[];ys=[]
                    for length in (64,512,1024,2048,4096,8192,11060):
                        xs=(length,)*batch;blocks=op.training(xs,tp,batch*max(1,length-530)) if stage=='training' else op.forward(xs,tp)
                        design.append([1,batch*length,batch*length*length]);ys.append(net.blocks(tp,blocks))
                    a=np.asarray(design,dtype=float);scale=np.maximum(np.max(abs(a),axis=0),1);coef=np.linalg.lstsq(a/scale,np.asarray(ys),rcond=None)[0]/scale;table[key]=coef.tolist()
                    for length in (777,3072,6144):
                        xs=tuple(max(1,length+(i%2)*64-32) for i in range(batch));blocks=op.training(xs,tp,sum(max(1,x-530) for x in xs)) if stage=='training' else op.forward(xs,tp)
                        actual=net.blocks(tp,blocks);pred=float(np.dot(coef,[1,sum(xs),sum(x*x for x in xs)]));errors.append(abs(pred-actual)/max(actual,1))
        data['table']=table;data['adaptive_extension_cpu_s']=time.perf_counter()-start
        if errors:
            data['adaptive_max_relative_error']=max(errors);data['adaptive_p95_relative_error']=sorted(errors)[math.ceil(.95*len(errors))-1]
        Path(output).write_text(json.dumps(data,indent=2)+'\n');return cls(data)
