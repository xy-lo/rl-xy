"""Small calibrated *simulator-service* surrogate for bounded planning.

Source samples are operator queries at declared public shape ranges, never
current answer lengths. Numerical interpolation error is measured separately
from unmeasured hardware-model error. Native replay remains the acceptance test.
"""
import json,time,math
from pathlib import Path

def service_contract(op):
    """Bind predictive services to math/hardware, independent of host compiler."""
    from dataclasses import asdict
    result=dict(hardware=asdict(op.hw),local_timing_identity=getattr(op,'local_timing_identity',None),
        assets_manifest_sha256=op.runtime_cost_identity['assets_manifest_sha256'],
        compute_scale=op.compute_scale,network_scale=op.network_scale,
        generation_cost_contract='frozen grouped-core GQA wrapper; no arithmetic scaling override')
    candidate=op.runtime_cost_identity.get('hardware_candidate_sha256')
    if candidate:
        result['hardware_candidate_sha256']=candidate
        result['physical_mesh_shape']=op.runtime_cost_identity['physical_mesh_shape']
        result['paired_owner_count']=op.runtime_cost_identity['paired_owner_count']
    return result

def validate_catalog(catalog,op,model,config_sha256):
    if catalog.get('model_id')!=model or catalog.get('model_config_sha256')!=config_sha256:
        raise ValueError('service catalog model configuration mismatch')
    if catalog.get('service_contract')!=service_contract(op):
        raise ValueError('service catalog hardware or mathematical source mismatch')
    return True

class ServiceCatalog:
    def __init__(self,data):self.data=data
    @classmethod
    def build(cls,op,cap,train_mb,logprob_mb,output):
        import numpy as np
        from rl_xy.qwen3_native import NetworkProfiles
        net=NetworkProfiles(op,'unused',native=False);start=time.perf_counter();table={};validation=[]
        # Exact two-point decode interpolation, independently checked at two
        # interior context lengths for every supported batch/TP.
        for tp in (2,4,8):
            for batch in range(1,cap+1):
                ys=[net.blocks(tp,op.forward((1,)*batch,tp,'generation',(ctx,)*batch)) for ctx in (1,11060)]
                slope=(ys[1]-ys[0])/11059;intercept=ys[0]-slope
                table[f'decode:{tp}:{batch}']=[intercept,slope]
                for ctx in (2048,8192):
                    actual=net.blocks(tp,op.forward((1,)*batch,tp,'generation',(ctx,)*batch));pred=intercept+slope*ctx
                    validation.append(abs(pred-actual)/max(actual,1))
        for tp in (2,4,8):
            for stage,batches in [('prefill',range(1,cap+1)),('teacher',[logprob_mb]),('training',[train_mb])]:
                for batch in batches:
                    lengths=(64,512,1024,2048,4096,8192,11060) if stage!='prefill' else (41,256,530,1024,2048,2869)
                    points=[];ys=[]
                    for length in lengths:
                        xs=(length,)*batch;blocks=op.training(xs,tp,batch*max(1,length-530)) if stage=='training' else op.forward(xs,tp,'generation' if stage=='prefill' else 'teacher')
                        points.append([1,batch*length,batch*length*length]);ys.append(net.blocks(tp,blocks))
                    # Scale design columns for numerical stability.
                    design=np.asarray(points,dtype=float);scale=np.maximum(np.max(abs(design),axis=0),1);coef=np.linalg.lstsq(design/scale,np.asarray(ys),rcond=None)[0]/scale
                    table[f'{stage}:{tp}:{batch}']=coef.tolist()
                    for length in (777,3072,6144) if stage!='prefill' else (128,768,1536):
                        xs=tuple(max(1,length+(i%2)*64-32) for i in range(batch));blocks=op.training(xs,tp,sum(max(1,x-530) for x in xs)) if stage=='training' else op.forward(xs,tp,'generation' if stage=='prefill' else 'teacher')
                        actual=net.blocks(tp,blocks);pred=float(np.dot(coef,[1,sum(xs),sum(x*x for x in xs)]));validation.append(abs(pred-actual)/max(actual,1))
        data=dict(schema='idea1.operator-service-surrogate/1',model=op.model.name if hasattr(op.model,'name') else str(op.info['config'].get('_name_or_path','')),
            cap=cap,train_mb=train_mb,logprob_mb=logprob_mb,table=table,
            max_relative_validation_error=max(validation),p95_relative_validation_error=sorted(validation)[math.ceil(.95*len(validation))-1],
            startup_cpu_s=time.perf_counter()-start,scope='frozen simulated operator services plus alpha-beta communication; not hardware calibration or exact native mesh timing')
        data['service_contract']=service_contract(op)
        Path(output).write_text(json.dumps(data,indent=2)+'\n');return cls(data)
    def decode(self,tp,batch,mean_context):
        a,b=self.data['table'][f'decode:{tp}:{batch}'];return max(1,a+b*mean_context)
    def phase(self,stage,tp,lengths):
        if not lengths:return 0
        co=self.data['table'][f'{stage}:{tp}:{len(lengths)}'];return max(1,co[0]+co[1]*sum(lengths)+co[2]*sum(x*x for x in lengths))
