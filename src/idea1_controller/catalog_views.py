"""Service coverage for the concurrency admitted by shared actor views."""
from .catalog_v2 import AdaptiveServiceCatalog
import time,json,math
from pathlib import Path

class ViewServiceCatalog(AdaptiveServiceCatalog):
    @classmethod
    def expand(cls,base,op,policy,output):
        import numpy as np
        from rl_xy.qwen3_native import NetworkProfiles
        net=NetworkProfiles(op,'unused',native=False);start=time.perf_counter();data=dict(base.data);table=dict(data['table']);errors=[]
        for tp in (2,4):
            for batch in range(1,policy.cap+1):
                key=f'decode:{tp}:{batch}'
                if key not in table:
                    ys=[net.blocks(tp,op.forward((1,)*batch,tp,'generation',(ctx,)*batch)) for ctx in (1,11060)]
                    slope=(ys[1]-ys[0])/11059;intercept=ys[0]-slope;table[key]=[intercept,slope]
                    for ctx in (2048,8192):
                        actual=net.blocks(tp,op.forward((1,)*batch,tp,'generation',(ctx,)*batch));pred=intercept+slope*ctx;errors.append(abs(pred-actual)/max(actual,1))
                key=f'prefill:{tp}:{batch}'
                if tp==2 and key not in table:
                    a=[];ys=[]
                    for length in (41,256,530,1024,2048,2869):
                        xs=(length,)*batch;a.append([1,batch*length,batch*length*length]);ys.append(net.blocks(tp,op.forward(xs,tp,'generation')))
                    a=np.asarray(a,dtype=float);scale=np.maximum(np.max(abs(a),axis=0),1);coef=np.linalg.lstsq(a/scale,np.asarray(ys),rcond=None)[0]/scale;table[key]=coef.tolist()
                    for length in (128,768,1536):
                        xs=(length,)*batch;actual=net.blocks(tp,op.forward(xs,tp,'generation'));pred=float(np.dot(coef,[1,sum(xs),sum(x*x for x in xs)]));errors.append(abs(pred-actual)/max(actual,1))
        data.update(table=table,cap=policy.cap,view_expansion_cpu_s=time.perf_counter()-start)
        if errors:data['view_expansion_max_error']=max(errors);data['view_expansion_p95_error']=sorted(errors)[math.ceil(.95*len(errors))-1]
        Path(output).write_text(json.dumps(data,indent=2)+'\n');return cls(data)
