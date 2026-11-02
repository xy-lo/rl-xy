"""Batch-aware PENDING handoff fills the admitted receiver's normal queue.

Every request is unstarted and has the same home domain. Filling up to normal
cap avoids repeatedly creating underfilled decode batches of four long answers.
No extra KV allocation beyond the already admitted normal cap is introduced.
"""
from .pending import PendingMixin
from .full_layer import FullLayerKernel
from .coordinate_kernel import CoordinateKernel

class PackedPending:
    def pending_choice(self,w):
        choice=PendingMixin.pending_choice(self,w)
        if choice is None:return None
        src,dst,_=choice
        return src,dst,list(self.rep[src].requests)[:self.rep[dst].cap]

class PackedFullLayerKernel(PackedPending,FullLayerKernel):pass
class PackedCoordinateKernel(PackedPending,CoordinateKernel):pass
