"""Measured A100 GEMM services with bounded one-dimensional interpolation.

Validation samples are never interpolation knots. Measured services include
kernel memory access; no analytic HBM or launch is added for the same GEMM.
The reference mapper remains an explicit resource-accounting approximation.
"""
from dataclasses import asdict,replace
from bisect import bisect_left
from pathlib import Path
import hashlib,json,math


def family(problem):
    p=dict(problem);axis='k' if p['direction']=='dweight' else 'm'
    value=p.pop(axis)
    key=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return key,axis,value


def attention_features(queries,contexts,heads,kv_heads,head_dim):
    pairs=sum(q*(q+1)/2 if q==c else q*c for q,c in zip(queries,contexts))
    tasks=max(1,sum((q+127)//128 for q in queries)*heads)
    occupancy=108/min(tasks,108)
    return [1.,sum(queries)*heads*head_dim,sum(contexts)*kv_heads*head_dim,
            pairs*heads*head_dim,len(queries)*heads,
            pairs*heads*head_dim*occupancy,sum(contexts)*kv_heads*head_dim*occupancy]


def attention_geometry(queries,contexts):
    if not queries or len(queries)!=len(contexts) or any(type(q) is not int or type(c) is not int or q<=0 or c<=0 or q>c for q,c in zip(queries,contexts)):
        raise ValueError('invalid attention query/context geometry')
    if any(q!=1 and q!=c for q,c in zip(queries,contexts)):
        raise ValueError('unsupported mixed prefill attention geometry')
    return {'batch':len(queries),'query_tokens':sum(queries),'context_tokens':sum(contexts),
            'max_query':max(queries),'max_context':max(contexts)}


class GEMMCalibration:
    def __init__(self,path):
        path=Path(path).resolve();raw=path.read_bytes();data=json.loads(raw)
        if data.get('schema_version')!='rlxy.gpu_calibration/1' or data.get('status')!='PASS':
            raise ValueError('GPU calibration must be a completed measurement profile')
        if data.get('device')!='NVIDIA A100-SXM4-80GB':raise ValueError('GPU calibration SKU mismatch')
        if not data.get('numeric_fixture_checks') or not all(r['passed'] for r in data['numeric_fixture_checks']):
            raise ValueError('GPU calibration numerical checks missing or failed')
        self.identity={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
                       'scope':data['scope'],'device':data['device']}
        self.attention_models=data.get('attention_models',{})
        self.attention_domains=data.get('attention_domains',{})
        self.require_attention_domain=data.get('require_attention_domain',False)
        self.knots={}
        for row in data['measurements'].values():
            if row.get('split')!='calibration':continue
            key,axis,x=family(row['problem']);y=row['duration_ns']
            if not math.isfinite(y) or y<=0 or not row.get('numeric_sample_passed'):
                raise ValueError('invalid measured GEMM service')
            self.knots.setdefault(key,[]).append((x,float(y)))
        for key in self.knots:
            self.knots[key]=sorted(self.knots[key])
            if len({x for x,y in self.knots[key]})!=len(self.knots[key]):raise ValueError('duplicate interpolation knot')
        if not self.knots:raise ValueError('empty GPU calibration domain')

    def predict(self,problem):
        key,axis,x=family(asdict(problem) if not isinstance(problem,dict) else problem)
        if key not in self.knots:raise ValueError('GEMM family outside measured calibration domain')
        knots=self.knots[key];xs=[a for a,b in knots];i=bisect_left(xs,x)
        if i<len(xs) and xs[i]==x:return knots[i][1]
        if i==0 or i==len(xs):raise ValueError('GEMM interpolation outside measured range')
        a,ya=knots[i-1];b,yb=knots[i]
        return max(1.,ya+(yb-ya)*(x-a)/(b-a))

    def predict_attention(self,queries,contexts,heads,kv_heads,head_dim,backward=False):
        geometry=attention_geometry(queries,contexts)
        mode='backward' if backward else 'decode' if max(queries)==1 else 'prefill'
        key=f'{heads}/{kv_heads}/{head_dim}/{mode}'
        if key not in self.attention_models:raise ValueError('attention outside measured head/mode domain')
        domain=self.attention_domains.get(key)
        if self.require_attention_domain and not domain:
            raise ValueError('attention calibration requires explicit geometry domain')
        if domain:
            for name,value in geometry.items():
                limits=domain.get(name)
                if not limits or len(limits)!=2 or value<limits[0] or value>limits[1]:
                    raise ValueError('attention geometry outside calibration domain: '+name)
        features=attention_features(queries,contexts,heads,kv_heads,head_dim)
        coefficients=self.attention_models[key]['coefficients']
        return max(1.,sum(a*b for a,b in zip(features,coefficients)))


class CalibratedGPUCostProvider:
    def __init__(self,reference,calibration):
        self.reference=reference;self.calibration=calibration

    def estimate(self,spec,hardware):
        if not spec.matmuls:return self.reference.estimate(spec,hardware)
        quota=self.reference.estimate(spec,hardware)
        inclusive=sum(self.calibration.predict(p) for p in spec.matmuls)
        extra=self.reference.estimate(replace(spec,matmuls=()),hardware) if (spec.vector_ops or spec.read_bytes or spec.write_bytes) else None
        compute=inclusive+(extra.compute_ns if extra else 0)
        memory=extra.local_memory_ns if extra else 0
        return replace(quota,duration_ns=max(1,compute+memory),compute_ns=compute,local_memory_ns=memory,
                       core_compute_ns=tuple(0 for _ in quota.core_compute_ns),
                       source='measured_A100_layout_GEMM_interpolation_v1_plus_analytic_nonGEMM',
                       timing_contract='inclusive_measured_GEMM_no_second_HBM_or_launch; reference_mapper_resource_bounds_only')
