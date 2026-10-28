"""Layer-scope no-backward inference lifetimes for the declared dense graph.

Drops only provably unrelated conservative read permissions: earlier layers'
nonresidual activations and other layers' KV. Kernel work/traffic is unchanged.
Output tokens/logprobs remain live through the explicit export/cleanup boundary.
"""
import copy,re
from .tensor_plan import TensorPlan,preflight_contract
_LAYER=re.compile(r'\.layer(\d+)\.')

def trim_inference_lifetimes(plan,case):
    if any(o.metadata.get('backward_prefix') or o.metadata.get('save_tape') for o in plan.operations):
        raise ValueError('inference liveness pass cannot remove a training tape')
    by_id={o.node_id:o for o in plan.operations};tensors={t['storage_id']:dict(t) for t in plan.tensors}
    contract={**plan.contract,'ranks':{r:{**c,'initial_objects':dict(c['initial_objects']),
        'nodes':{nid:{**n,'reads':list(n['reads']),'allocations':dict(n['allocations']),'ready_outputs':list(n['ready_outputs']),'frees':[]} for nid,n in c['nodes'].items()}}
        for r,c in plan.contract['ranks'].items()}}
    removed={'earlier_layer_activation':0,'other_layer_KV':0,'inference_norm_statistics':0}
    for r,c in contract['ranks'].items():
        local=[o for o in plan.operations if int(r) in o.participant_ranks]
        cleanup=[o.node_id for o in local if o.name=='release_round_temporaries']
        if len(cleanup)!=1:raise ValueError('one explicit complete inference cleanup required')
        last={s:t['created_node'] for s,t in tensors.items() if t['rank']==int(r) and not t['persistent']}
        for op in local:
            node=c['nodes'][str(op.node_id)];meta=op.metadata
            prefix=meta.get('forward_prefix');match=_LAYER.search(op.name);layer=int(match[1]) if match else None
            reads=[]
            for sid in node['reads']:
                t=tensors[sid];drop=None
                if prefix and not t['persistent']:
                    producer=by_id.get(t['created_node']);name=producer.name if producer else ''
                    own_prefix=producer.metadata.get('forward_prefix') if producer else None
                    m=_LAYER.search(name);oldlayer=int(m[1]) if m else None
                    if t['kind']=='norm_statistics':drop='inference_norm_statistics'
                    elif t['kind']=='kv' and oldlayer is not None and oldlayer!=layer:drop='other_layer_KV'
                    elif t['kind']=='kv' and layer is None:drop='other_layer_KV'
                    elif own_prefix==prefix and t['kind'] not in ('kv','token','trajectory','log_prob'):
                        if layer is not None:
                            keep=(oldlayer==layer or (oldlayer==layer-1 and '.residual_ffn' in name)
                                or '.rope_table' in name or (layer==0 and '.embedding' in name))
                        else:
                            keep=(oldlayer is None or (oldlayer==case.model.layers-1 and '.residual_ffn' in name))
                        if not keep:drop='earlier_layer_activation'
                if drop:removed[drop]+=1
                else:reads.append(sid)
            node['reads']=reads
            for sid in reads+node['ready_outputs']:
                if sid in last:last[sid]=max(last[sid],op.node_id)
        # Request outputs are exported by the enclosing stage immediately
        # before cleanup; their IDs stay ready until this boundary.
        for sid in last:
            if tensors[sid]['kind'] in ('token','trajectory','log_prob'):last[sid]=cleanup[0]
        for sid,nid in last.items():
            contract['ranks'][r]['nodes'][str(nid)]['frees'].append(sid);tensors[sid]['released_node']=nid
        for node in c['nodes'].values():node['frees'].sort()
    contract['inference_liveness']={'policy':'layer_scope_out_of_place_no_backward',
        'removed_conservative_read_permissions':removed,'kernel_cost_or_FLOP_changes':False,
        'completed_request_outputs_retained_to_cleanup':True}
    return TensorPlan(contract,list(tensors.values()),list(plan.operations),preflight_contract(contract))
