"""Bind executed planning evidence to the immutable runtime StepPlan."""
from .adaptive_adapter import AdaptiveContext
from .context import FixedPlanner,digest
from ..contracts import Readiness
from dataclasses import replace

class SelectedContext(AdaptiveContext):
    def _fixed_plan(self):
        plan=super()._fixed_plan();k=self.kernel;manifest=k.resource_layout
        forecast=manifest.get('assignment_forecasts',{})
        assignment=replace(plan.assignment,plan_id=manifest.get('assignment_id',digest(dict(plan.assignment.initial_instance_by_request))),
            long_group_quota_by_region={f'R{r}':n for r,n in manifest.get('assignment_long_quotas',{}).items()},
            fallback_group_ids=frozenset(manifest.get('fallback_group_ids',[])),
            prediction_digest=digest(forecast) if forecast else 'explicit-baseline-or-manual-placement')
        state=replace(plan.placement.state_placement,migration_plan_digest=digest(k.actor_views.manifest()))
        placement=replace(plan.placement,state_placement=state)
        pools=manifest.get('helper_domains')
        if pools:
            owner={w:ws for ws in pools.values() for w in ws}
            placement=replace(placement,
                admission_domains={f'C{r}':tuple(d for w in ws for d in k.rep[w].ranks) for r,ws in pools.items()},
                drain_dependencies={f'R{r}':tuple(f'I{w}' for w in owner[r]) for r in k.region_members})
        return replace(plan,placement=placement,assignment=assignment,step_plan_id=manifest.get('plan_id',plan.step_plan_id))
    def snapshot(self):
        snapshot=super().snapshot();dies=dict(snapshot.dies);k=self.kernel
        # Role and activity are separate: a borrowed training pair keeps the
        # TRAINING role while its temporary execution group changes.
        for w,job in k.helper_jobs.items():
            state=k.batch_state[job['batch_id']]
            for die in k.rep[w].ranks:
                old=dies[die];lease='training:'+job['batch_id']
                dies[die]=replace(old,activity='HELP' if state=='running' else 'TRANSFER',
                    phase='BACKWARD' if state=='running' else old.phase,
                    execution_group_id='training-help-I'+str(w),lease_ids=old.lease_ids|frozenset((lease,)))
        return replace(snapshot,dies=dies,backend_capability_id='shared-actor-native-regions-TP2-4-8')

class SelectedManifestPlanner(FixedPlanner):
    def build(self,requests,model,mesh,algorithm,services):
        c=self.c;k=c.kernel
        assert {r.request_id for r in requests}==set(k.ids)
        assert model.parameter_schema_digest==k.coordinates.schema_digest
        assert c.plan.algorithm==algorithm
        assert c.plan.placement.candidate.rollout_cap==c.policy.cap
        assert c.plan.assignment.prediction_digest!='disabled-history-frozen-baseline'
        return c.plan
