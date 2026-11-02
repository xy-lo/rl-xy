"""Attach exact QKV/FFN/strided coordinate proof to every gradient payload."""
from .adaptive import AdaptiveKernel
from ..coordinates import ParameterCoordinates

class CoordinateKernel(AdaptiveKernel):
    def init_adaptive(self,record_reserve):
        super().init_adaptive(record_reserve);self.coordinates=ParameterCoordinates(self.op.model)
        for ranks in self.region_ranks.values():
            tp=len(ranks)
            for pid in sorted({x.parameter_id for x in self.coordinates.components}):
                self.coordinates.views(pid,tp,0)
    def _restore_gradient(self,r):
        begin=len(self.restore_proofs);steps=super()._restore_gradient(r);tp=len(self.region_ranks[r])
        for proof in self.restore_proofs[begin:]:
            descriptor=self.coordinates.restore_descriptor(proof['parameter'],tp,proof['logical_lane'],proof['subshard'])
            assert descriptor['payload_elements']*4==proof['bytes']
            proof['coordinate_descriptor']=descriptor
        return steps
