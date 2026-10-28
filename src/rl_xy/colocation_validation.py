"""Actual-clock stage ordering and pinned-layout protocol validation."""
from dataclasses import replace
from .types import CheckResult
from .validation import _event_checks,_dependency_checks
from .resource_validation import validate_resources


def validate_colocation(case,graph,events):
    ops={o.node_id:o for o in graph.plan.operations};stages={};versions=[]
    for e in events:
        op=ops[e.node_id];stage=op.metadata.get('colocation_stage',op.phase)
        if stage in ('input','rollout'):stage='generation'
        s=stages.setdefault((op.round_id,stage),[e.issue_tick,e.callback_tick])
        s[0]=min(s[0],e.issue_tick);s[1]=max(s[1],e.callback_tick)
        if op.name=='adamw_read_modify_write':
            expected=case.workload.behavior_version+op.round_id-case.workload.round_id
            if (op.metadata['actor_version_in'],op.metadata['actor_version_out'])!=(expected,expected+1):versions.append(op.op_id)
    errors=[]
    for offset in range(case.rounds):
        rid=case.workload.round_id+offset
        required=['generation','reward','reference','old_policy','advantage','train','dp_sync','clip','optimizer','publish']
        for stage in required:
            if (rid,stage) not in stages:errors.append([rid,'missing',stage])
        for before,after in [('weight_reshard','generation'),('switch_to_rollout','generation'),
            ('generation','reward'),('switch_to_train','reward'),('reward','old_policy'),('old_policy','reference'),
            ('reference','advantage'),('advantage','train'),('train','dp_sync'),('dp_sync','clip')]:
            if (rid,before) in stages and (rid,after) in stages and stages[rid,before][1]>stages[rid,after][0]:
                errors.append([rid,before,after,stages[rid,before],stages[rid,after]])
        if offset and stages[rid,'generation'][0]<stages[rid-1,'publish'][1]:errors.append([rid,'stale inter-round rollout'])
    global_case=replace(case,hardware=replace(case.hardware,dies=graph.audit['allocated_devices']),
        strategy=replace(case.strategy,tensor_parallel=graph.audit['train_tp']))
    checks=list(graph.component_checks)+_event_checks(global_case,graph.plan.operations,events)+_dependency_checks(global_case,graph.plan.operations)
    checks.append(CheckResult('colocation.global_stage_barriers',not errors,'all selected preparation stages finish globally before train',errors,
        'native issue/callback timestamps across every physical rank; not Python time estimates','run'))
    checks.append(CheckResult('colocation.actor_reference_versions',not versions,'actor advances once per synchronized update; reference immutable',versions,
        'actual submitted update metadata plus resource reference-write prohibition','run'))
    extra,snapshots,peaks=validate_resources(graph.plan,global_case,events,gradient_visits=[])
    checks.extend(extra)
    audit={f'r{rid}.{stage}':{'start_tick':s[0],'end_tick':s[1],'span_model_ns':(s[1]-s[0])/1000}
        for (rid,stage),s in stages.items()}
    return checks,snapshots,peaks,audit
