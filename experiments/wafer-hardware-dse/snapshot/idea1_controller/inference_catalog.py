"""Sparse bound service curves for real inference TP1/2/4/8 planning."""
from bisect import bisect_left
from .cost_catalog import ServiceCatalog


class InferenceServiceCatalog(ServiceCatalog):
    def __init__(self,data):
        super().__init__(data);self._nodes={};self._coefs={}
        for key in data['table']:
            stage,tp,batch=key.split(':');self._nodes.setdefault((stage,int(tp)),[]).append(int(batch))
        for key in self._nodes:self._nodes[key].sort()
    def coefficients(self,stage,tp,batch):
        key=(stage,tp,batch)
        if key in self._coefs:return self._coefs[key]
        text=f'{stage}:{tp}:{batch}'
        if text in self.data['table']:co=tuple(self.data['table'][text])
        else:
            sizes=self._nodes[stage,tp];i=bisect_left(sizes,batch)
            if i==len(sizes):raise ValueError('inference shape exceeds service coverage')
            a,b=(sizes[0],sizes[0]) if i==0 else sizes[i-1:i+1]
            lo=self.data['table'][f'{stage}:{tp}:{a}'];hi=self.data['table'][f'{stage}:{tp}:{b}'];t=0 if a==b else (batch-a)/(b-a)
            co=tuple((1-t)*x+t*y for x,y in zip(lo,hi)) if stage=='decode' else (lo[0]*(1-t)+hi[0]*t,*((lo[j]*a*(1-t)+hi[j]*b*t)/batch for j in (1,2)))
        self._coefs[key]=co;return co
    def decode(self,tp,batch,mean_context):
        co=self.coefficients('decode',tp,batch);return max(1,co[0]+co[1]*mean_context)
    def phase(self,stage,tp,lengths):
        if not lengths:return 0
        batch=len(lengths);s=sum(lengths);sq=sum(x*x for x in lengths)
        co=self.coefficients(stage,tp,batch)
        return max(1,co[0]+co[1]*s+co[2]*sq)
