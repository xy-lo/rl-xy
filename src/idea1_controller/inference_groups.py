"""Physical inference partitions over canonical TP2 storage owners.

TP2 owners are a storage/control implementation detail, not inference replicas.
Mixed TP2/4/8 partitions retain every one of the 54 physical dies.
"""
from dataclasses import dataclass
from .actor_views import ViewAdmission


def inference_ranks(atoms):
    return tuple(2*w+h for h in range(2) for w in atoms)


def validate_partition(groups):
    assert sorted(w for ws in groups.values() for w in ws)==list(range(27))
    assert all(int(r)==min(ws) and len(ws) in (1,2,4) for r,ws in groups.items())
    for ws in groups.values():
        assert list(ws)==sorted(ws),'inference ranks must preserve canonical TP2 half order'
        reached={ws[0]}
        while True:
            expanded=reached|{b for a in reached for b in ws if abs(a//3-b//3)+abs(a%3-b%3)==1}
            if expanded==reached:break
            reached=expanded
        assert reached==set(ws),'inference cohort must be physically connected'
    return True


def stripe_partition(tp):
    """Deterministic nearby vertical strips; preserve TP2 remainder per column."""
    if tp not in (2,4,8):raise ValueError('unsupported inference TP')
    size=tp//2;groups={}
    for col in range(3):
        atoms=[3*row+col for row in range(9)]
        while atoms:
            n=size if len(atoms)>=size else (2 if len(atoms)>=2 else 1)
            ws=tuple(atoms[:n]);groups[ws[0]]=ws;atoms=atoms[n:]
    validate_partition(groups)
    return groups


def elastic_rectangles():
    """Supported connected TP4/8 physical cohorts, independent of training."""
    result=set()
    for h,w in ((1,2),(2,1),(2,2),(4,1)):
        for row in range(10-h):
            for col in range(4-w):result.add(tuple((row+i)*3+col+j for i in range(h) for j in range(w)))
    return tuple(sorted(result))


def kv_reshard_moves(old,new,contexts,model):
    """Conserve actual GQA head intervals when a cohort changes TP.

    Byte counts include both K/V, all layers, and every observed context token.
    Local retained intervals still incur an explicit in-place repack service.
    """
    if model.kv_heads%len(old) or model.kv_heads%len(new):raise ValueError('KV heads do not divide actual TP')
    unit=2*model.layers*sum(contexts)*model.head_dim*model.actor_dtype_bytes
    moves=[]
    for i,src in enumerate(old):
        a,b=i*model.kv_heads//len(old),(i+1)*model.kv_heads//len(old)
        for j,dst in enumerate(new):
            lo=max(a,j*model.kv_heads//len(new));hi=min(b,(j+1)*model.kv_heads//len(new))
            if lo<hi:moves.append(dict(src=src,dst=dst,head_start=lo,head_end=hi,bytes=unit*(hi-lo)))
    assert sum(x['bytes'] for x in moves)==unit*model.kv_heads
    # Retire every old destination's outgoing KV before another source writes
    # its replacement there. Local compaction is last, after remote sends, so
    # it cannot overwrite a later token/head still needed by a sender.
    dependencies={r:{m['dst'] for m in moves if m['src']==r and m['dst'] in old and m['dst']!=r} for r in old}
    retired=set();ordered=[]
    while len(retired)<len(old):
        ready=[r for r in old if r not in retired and dependencies[r]<=retired]
        if not ready:raise ValueError('KV repartition needs an additional preserved arena; no bounded in-place schedule')
        r=ready[0];ordered.extend(sorted((m for m in moves if m['src']==r),key=lambda m:(m['src']==m['dst'],m['head_start'])))
        retired.add(r)
    return ordered


def parameter_view_proof(model,groups,paired_tp1=False):
    """Verify all actor components against the canonical storage owner views."""
    from dataclasses import asdict
    from .coordinates import ParameterCoordinates
    from .history import digest
    schema=ParameterCoordinates(model);pids=sorted({c.parameter_id for c in schema.components});views={}
    for root,ws in groups.items():
        ranks=inference_ranks(ws);tp=1 if paired_tp1 else len(ranks)
        for logical,physical in enumerate(ranks):
            target=[]
            for pid in pids:
                selected=schema.views(pid,tp,0 if tp==1 else logical)
                if tp==1:
                    halves=[schema.views(pid,2,h) for h in (0,1)]
                    for full,a,b in zip(selected,*halves):
                        assert full.component_id==a.component_id==b.component_id
                        if full.axis is not None:
                            assert full.start==a.start and a.end==b.start and b.end==full.end and a.elements+b.elements==full.elements
                        else:assert full.elements==a.elements==b.elements # select equal, never add weights
                else:
                    for v,owner in zip(selected,schema.views(pid,2,physical%2)):
                        assert v.component_id==owner.component_id and v.axis==owner.axis
                        if v.axis is not None:assert owner.start<=v.start<v.end<=owner.end
                target.extend(asdict(x) for x in selected)
            views[str(physical)]=dict(tp=tp,logical_rank=0 if tp==1 else logical,canonical_owner_lane=physical%2,
                coordinate_sha256=digest(target),elements=sum(x['elements'] for x in target),
                storage='local canonical superset' if tp>1 else 'canonical local half plus explicitly copied missing half; equal norms selected once')
    assert sorted(map(int,views))==list(range(54))
    return dict(schema_digest=schema.schema_digest,views=views,weight_semantics='coordinate subviews or TP1 concatenation/select; no norm summation')


class InferenceAdmission:
    """Generation certificate with TP2 actor/reference owners kept resident."""
    def __init__(self,op,reserve=200_000_000):self.op=op;self.base=ViewAdmission(op,reserve);self.reserve=reserve
    def evaluate(self,tp,cap,public_groups):
        if tp not in (1,2,4,8) or cap<1:raise ValueError('unsupported TP/cap')
        result=self.base.evaluate(1,public_groups);op=self.op;m=op.model
        maxp=max(op.prompt_cap,max(x['prompt_length'] for x in public_groups))
        base=self.base.memory.teacher_base+(self.base.memory.actor if tp==1 else 0);nic=op.hw.nic_tx_bytes_per_die+op.hw.nic_rx_bytes_per_die
        kv=2*m.layers*(m.kv_heads//tp)*m.head_dim*m.actor_dtype_bytes
        scratch=cap*(m.hidden_size*32+m.ffn_size*16)+1024**2
        generation=base+cap*(maxp+8192)*kv+scratch+nic+self.reserve
        tokens=cap*maxp
        layer=tokens*(m.hidden_size*12+2*m.ffn_size//tp*6+(m.query_width+2*m.kv_heads*m.head_dim)//tp*6)
        head=cap*(m.vocab_size//tp)*8
        prefill=base+cap*maxp*kv+max(layer,head)+4*tokens*m.hidden_size+scratch+nic+self.reserve
        result['phase_peak_bytes'].update(generation=generation,prefill=prefill,helper=generation)
        result.update(inference_tp=tp,rollout_cap=cap,passed=max(result['phase_peak_bytes'].values())<=op.capacity,
            inference_storage='canonical TP2 plus explicitly copied missing BF16 half for TP1' if tp==1 else 'canonical TP2 BF16 superset; parameter-coordinate TPp subviews; KV sharded at actual TP')
        return result

    def frontier(self,tp,public_groups,limit=512):
        lo,hi=0,limit
        while lo<hi:
            mid=(lo+hi+1)//2
            if self.evaluate(tp,mid,public_groups)['passed']:lo=mid
            else:hi=mid-1
        return lo
