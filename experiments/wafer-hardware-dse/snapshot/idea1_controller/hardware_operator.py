"""Construct a fresh operator for an explicit candidate hardware profile.

This is a cost-provider adapter, not an end-to-end scheduler integration.
The operator's hw.dies represents its operator/TP case, not the physical mesh;
the physical die count is carried separately in the returned identity.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
import os


def install_reference_staging(op):
    """Enable the existing early-reference gather lifecycle on this operator.

    Owners retain one DP shard, gather the full reference before rollout, and
    release that materialization after reference scoring.  The native runner
    is responsible for emitting the gather.  This function changes no tensor
    parallel ownership or arithmetic costs.
    """
    if not hasattr(op,'active_dies') or op.active_dies <= 0:
        raise ValueError('reference staging requires active physical owners')
    original=op.state
    if 'persistent_reference_shard_bytes' in original(2,2):
        raise ValueError('reference staging already provided by cost source')
    def state(train_tp,rollout_tp):
        if train_tp<=0 or op.active_dies%train_tp:
            raise ValueError('reference DP shards require complete TP owners')
        result=dict(original(train_tp,rollout_tp))
        alloc=result['layout'].state_bytes(0)['allocations']
        ref=alloc['reference_BF16_valid_parameters']
        grad=alloc['full_FP32_gradient_working_buffer']
        dp=op.active_dies//train_tp
        alignment=4*dp
        shard=(((ref+dp-1)//dp+alignment-1)//alignment)*alignment
        result.update(reference_bytes_per_rank=ref,gradient_bytes_per_rank=grad,
                      persistent_reference_shard_bytes=shard,
                      generation_base_max=result['training_base_max']-grad-ref+shard)
        return result
    op.state=state


def make_candidate_operator(factory, model, candidate, *, cache_dir=None, require_active_dies_support=False):
    required=('id','columns','rows','dies','dram_GB_per_die','dram_TB_s_per_die',
              'd2d_TB_s_per_edge_direction','module_path_TB_s')
    if any(k not in candidate for k in required):
        raise ValueError('incomplete candidate')
    if any(type(candidate[k]) is not int or candidate[k]<=0 for k in ('columns','rows','dies')):
        raise ValueError('invalid mesh dimensions')
    if candidate['columns']*candidate['rows'] != candidate['dies']:
        raise ValueError('mesh/die count mismatch')
    for key in ('dram_GB_per_die','dram_TB_s_per_die','d2d_TB_s_per_edge_direction','module_path_TB_s'):
        value=candidate[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('invalid '+key)
    # Fresh instance is essential because kernel/attention methods have lru caches.
    op=factory(model,'wafer',cache_dir=cache_dir)
    supports_active_dies=hasattr(op,'active_dies')
    if require_active_dies_support and not supports_active_dies:
        raise ValueError('cost provider cannot parameterize optimizer DP; refusing cross-hardware execution')
    hw=replace(op.hw,dram_bytes_per_die=int(candidate['dram_GB_per_die']*1_000_000_000),
               dram_bytes_per_ns=int(candidate['dram_TB_s_per_die']*1000),
               module_path_bytes_per_ns=int(candidate['module_path_TB_s']*1000),
               d2d_bytes_per_ns_per_direction=int(candidate['d2d_TB_s_per_edge_direction']*1000))
    op.hw=hw
    op.case=replace(op.case,hardware=hw)
    op.capacity=hw.dram_bytes_per_die
    # Keep wafer topology distinct from hw.dies, which is an operator TP case.
    op.rlxy_mesh_shape=(candidate['columns'],candidate['rows'])
    from .mesh_geometry import MeshGeometry,PairMapping
    idle_pairs=int(os.environ.get('RLXY_IDLE_OWNER_PAIRS','0'))
    minimum_tp=int(os.environ.get('RLXY_MIN_TRAINING_TP','2'))
    if minimum_tp not in (2,4,8):raise ValueError('invalid minimum training TP')
    op.rlxy_pair_mapping=PairMapping.with_idle_tail_pairs(MeshGeometry(*op.rlxy_mesh_shape),idle_pairs)
    op.rlxy_min_training_tp=minimum_tp
    if supports_active_dies:
        # This first implementation uses TP2 optimizer owners; an odd die is
        # present in the mesh but is not an optimizer owner.
        op.active_dies=2*len(op.rlxy_pair_mapping.pairs)
    staging=os.environ.get('RLXY_REFERENCE_STAGING','0')
    if staging not in ('0','1'):raise ValueError('invalid RLXY_REFERENCE_STAGING')
    if staging=='1':install_reference_staging(op)
    payload={'profile':candidate,'operator_hardware':asdict(hw),'model':model,
             'active_dies_supported':supports_active_dies,
             'active_dies':getattr(op,'active_dies',None),
             'scope':'local operator only; native network/placement not configured'}
    if staging=='1':payload['reference_staging']=True
    if idle_pairs:payload['idle_owner_pairs']=idle_pairs
    if minimum_tp!=2:payload['minimum_training_tp']=minimum_tp
    payload['sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    # Existing batch/catalog identities include runtime_cost_identity but not
    # active_dies or geometry directly. Include them to prevent invalid reuse.
    original_identity=getattr(op,'runtime_cost_identity',None)
    if original_identity is not None and not isinstance(original_identity,dict):
        raise ValueError('unexpected runtime cost identity')
    op.runtime_cost_identity={**(original_identity or {}),'hardware_candidate_sha256':payload['sha256'],
                              'physical_mesh_shape':list(op.rlxy_mesh_shape),
                              'paired_owner_count':getattr(op,'active_dies',None)}
    return op,payload
