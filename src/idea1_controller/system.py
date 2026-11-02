"""Top-level plan/build boundary for the complete single-round cost simulator."""
from dataclasses import replace
import time
from .actor_views import ViewAdmission
from .coordinates import ParameterCoordinates
from .search_views import ViewPlanSearch
from .history import digest
from .placement import largest_remainder
from .quotas import layout_quotas

class SystemPlanner:
    def __init__(self,op,policy,catalog):self.op=op;self.policy=policy;self.catalog=catalog
    def build(self,public_groups,history,layout=None,budget_s=15,helper_domains=None):
        if any(set(g)-{'group_id','prompt_length','prompt_sha256','library'} for g in public_groups):
            raise ValueError('planner requires public task views; live response/EOS fields are forbidden')
        start=time.perf_counter();memory=ViewAdmission(self.op);capacity_frontier=memory.candidates(public_groups)
        selected_capacity=memory.evaluate(self.policy.cap,public_groups)
        if self.policy.cap not in {x['rollout_cap'] for x in capacity_frontier}:
            capacity_frontier=sorted(capacity_frontier+[selected_capacity],key=lambda x:x['rollout_cap'])
        if not selected_capacity['passed']:raise ValueError('selected concurrency is not capacity-feasible')
        # Compare R1 at the same total in-flight quota, optimistically reusing
        # existing TP2 data. Extra full-model replication already exceeds DRAM
        # in the current three selected policies, before any extra R1 staging.
        actor2=memory.memory.actor
        r1cap=(self.policy.cap+1)//2
        r1lower=selected_capacity['phase_peak_bytes']['generation']+actor2
        parallelism=[dict(rollout_tp=1,replicas=54,per_instance_cap=r1cap,global_inflight=54*r1cap,
            feasible=False,target_global_inflight=27*self.policy.cap,optimistic_generation_peak_bytes=r1lower,reasons=['DRAM lower bound exceeds capacity'] if r1lower>self.op.capacity else ['TP1 native execution adapter not implemented']),
            dict(rollout_tp=2,replicas=27,per_instance_cap=self.policy.cap,global_inflight=27*self.policy.cap,feasible=True)]
        coordinates=ParameterCoordinates(self.op.model)
        for tp in (2,4,8):
            for pid in sorted({c.parameter_id for c in coordinates.components}):coordinates.views(pid,tp,0)
        missing=[str(g['group_id']) for g in public_groups if len(history.get('records',{}).get(str(g['group_id']),()))<2]
        if missing:
            regions={int(r):tuple(ws) for r,ws in layout.items()} if layout else {w:(w,) for w in range(27)}
            quotas=layout_quotas(len(public_groups),regions);ids=[str(g['group_id']) for g in public_groups];cursor=0;queues={}
            for w,q in quotas.items():queues[w]=ids[cursor:cursor+q];cursor+=q
            plan=dict(schema='idea1.native-selected-plan/2',region_members=regions,queues=queues,rollout_tp=2,rollout_cap=self.policy.cap,
                selected_dp=len(regions),selected_candidate='legal-cold-start-baseline',long_quota_by_region={},forecasts={},history_snapshot_id=history['snapshot_id'],
                fallback_group_ids=ids,planning_wall_s=time.perf_counter()-start,budget_truncated=False,plan_id=digest(dict(regions=regions,queues=queues,cap=self.policy.cap)))
        else:
            from .pool_search import PoolPlanSearch
            planner=(PoolPlanSearch if helper_domains is not None else ViewPlanSearch)(self.catalog,self.op,self.policy)
            planner.control_overhead_s=time.perf_counter()-start+0.05
            fixed_domains=helper_domains if isinstance(helper_domains,dict) else layout
            plan=planner.build_fixed(public_groups,history,fixed_domains,budget_s) if fixed_domains else planner.build(public_groups,history,budget_s)
            plan['fallback_group_ids']=[]
        plan.update(capacity_frontier=capacity_frontier,parallelism_feasibility=parallelism,parameter_schema_sha256=coordinates.schema_digest,
            public_input_digest=digest(public_groups),history_kind=history.get('kind','unknown'),supported_training_tp=[2,4,8],
            optimizer_contract='original TP2 distributed optimizer, global sample sum; canonical region gradients explicitly restored',
            actor_contract='immutable shared TP2 BF16 storage, no duplicate rollout weights; gradient absent during generation')
        plan['total_planner_wall_s']=time.perf_counter()-start
        return plan
