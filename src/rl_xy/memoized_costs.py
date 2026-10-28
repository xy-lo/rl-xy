"""Exact immutable-by-contract estimate reuse; changes no kernel cost.

Per-operation metadata references a shared read-only dictionary. Exporters may
copy the metadata wrapper but must not mutate the estimate payload. Canonical
KernelSpec/hardware values form the complete key; provider is per-run.
"""
from dataclasses import asdict
class MemoizedCostProvider:
 def __init__(self,provider):self.provider=provider;self.estimates={};self.metadata={};self.hits=0;self.misses=0
 def estimate(self,spec,hardware):
  key=(spec,hardware)
  if key not in self.estimates:self.estimates[key]=self.provider.estimate(spec,hardware);self.misses+=1
  else:self.hits+=1
  return self.estimates[key]
 def metadata_for(self,spec,hardware):
  key=(spec,hardware)
  if key not in self.metadata:self.metadata[key]=asdict(self.estimate(spec,hardware))
  return self.metadata[key]
