"""Task-keyed conditional helper horizon from a frozen independent history."""
from bisect import bisect_right
from .packed_pending import PackedPending
from .coordinate_kernel import CoordinateKernel
from .full_layer import FullLayerKernel
from ..search_v2 import PowerOfTwoMemory

class HistoricalHelper:
    def estimate_locality(self,w,helper):
        estimate=super().estimate_locality(w,helper)
        if estimate is None:return None
        residual=[];supported=0
        for item,context,_remaining in self.rep[w].active:
            pool=self.helper_history.get(str(item['group_id']),())
            elapsed=context-item['prompt_length'];i=bisect_right(pool,elapsed);n=len(pool)-i
            if n>=32:
                residual.append(max(1,pool[i+(9*(n-1))//10]-elapsed));supported+=1
            else:residual.append(max(64,min(2048,elapsed//4)))
        horizon=max(residual);estimate['predicted_steps']=horizon
        estimate['predicted_net_ns']=horizon*(estimate['predicted_local_gain_ns_per_step']-estimate['predicted_network_ns_per_step'])-estimate['predicted_migration_ns']
        estimate['accepted']=estimate['predicted_net_ns']>max(1000,estimate['predicted_migration_ns']*.05)
        estimate['frozen_task_history_supported_answers']=supported;return estimate

class HistoricalAdaptiveKernel(PackedPending,HistoricalHelper,CoordinateKernel):
    def init_history_help(self,history):
        self.helper_history={str(g):tuple(sorted(s['response_length'] for r in records for s in r['samples'])) for g,records in history['records'].items()}
    def init_adaptive(self,record_reserve):
        super().init_adaptive(record_reserve);self.memory=PowerOfTwoMemory(self.op,record_reserve)

class HistoricalFullKernel(PackedPending,HistoricalHelper,FullLayerKernel):
    init_history_help=HistoricalAdaptiveKernel.init_history_help
