"""Execution kernel for the complete planner-selected system."""
from dataclasses import replace
from .gradient_protocol import CanonicalGradientProtocol
from .history_help import HistoricalAdaptiveKernel
from ..lifecycle_memory import SystemMemory


class SystemKernel(CanonicalGradientProtocol,HistoricalAdaptiveKernel):
    def init_adaptive(self,record_reserve):
        super().init_adaptive(record_reserve);self.memory=SystemMemory(self.op,record_reserve)
    def _build_region_program(self,r):
        super()._build_region_program(r)
        # Preserve independently auditable operator bodies, not only hashes.
        import gzip,json
        folder=self.evidence_output/'compiled-regions';folder.mkdir(exist_ok=True)
        with gzip.open(folder/f'{r}.json.gz','wt') as f:json.dump(self.region_programs[r],f,separators=(',',':'))

    def _prepare_segment(self,w,it,tx,**kwargs):
        reply=super()._prepare_segment(w,it,tx,**kwargs)
        tx['segment_epoch']=self.segment_counter
        return reply

    def roles(self):
        labels=super().roles()
        for w,stage in self.stages.items():
            if stage in ('region_data','gradient_restore'):labels[w]='sample_transfer'
        return labels
