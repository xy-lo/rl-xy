"""Public-cap capacity and backend checks before any history performance search."""
from dataclasses import dataclass,asdict
from .memory import RegionalMemory
from .coordinates import ParameterCoordinates

class InferenceFeasibility:
    def __init__(self,op,policy):self.op=op;self.policy=policy
    def evaluate(self,public_groups):
        op=self.op;p=self.policy;rows=[];memory=RegionalMemory(op);coords=ParameterCoordinates(op.model)
        # All answers use hard R cap; current observed R is absent.
        xs=[dict(prompt_length=g['prompt_length'],response_length=8192) for g in public_groups for _ in range(8)]
        for r in (1,2):
            a=op.admission(xs,2,r,1,1,p.cap);peaks={k:v for k,v in a['phase_peak_bytes'].items() if k in ('generation','weight_export')}
            if 'persistent_reference_shard_bytes' in memory.state:
                # Preserve the explicitly charged full reference during rollout.
                peaks['generation']+=memory.reference-memory.shard
            reasons=[f'{phase} exceeds per-die DRAM' for phase,b in peaks.items() if b>op.capacity]
            if r!=2:reasons.append('native TP1 helper/region adapter unavailable; candidate cannot execute even if it fits')
            for pid in sorted({c.parameter_id for c in coords.components}):coords.views(pid,2,0)
            single_train=memory.hard_bounds(2,1,'training',max(op.prompt_cap,max(g['prompt_length'] for g in public_groups)))
            single_teacher=memory.hard_bounds(2,1,'teacher',max(op.prompt_cap,max(g['prompt_length'] for g in public_groups)))
            if max(single_train.peak_bytes,single_teacher.peak_bytes)>op.capacity:reasons.append('no bounded single-sample training fallback')
            rows.append(dict(rollout_tp=r,replicas=54//r,normal_cap=p.cap,feasible=not reasons,reasons=reasons,phase_peak_bytes=peaks,
                minimum_training_capacity=asdict(single_train),minimum_teacher_capacity=asdict(single_teacher),parameter_schema=coords.schema_digest))
        if not any(x['feasible'] for x in rows):raise ValueError('NO_SUPPORTED_INFERENCE_LAYOUT: '+str(rows))
        return rows
