"""Lossless read-set delta encoding for strictly ordered per-rank graphs.

Only serialization changes. The independent replay uses the original complete
contract; the native admission reconstructs the same set at every issue.
"""
import copy
import hashlib
import json


UINT64_MAX=(1<<64)-1


def compact_rank(rank, *, intern_read_ids=False):
    """Optionally intern ONLY read_delta IDs; old string output is default.

    Symbols are deterministic first-use order over the existing sorted deltas.
    Allocation/initial/READY/free identities remain their original strings.
    """
    if type(intern_read_ids)is not bool:raise ValueError('intern_read_ids must be boolean')
    if 'read_id_symbols'in rank:raise ValueError('compact_rank expects an expanded rank, not read_id_symbols')
    result={k:copy.deepcopy(v) for k,v in rank.items() if k!='nodes'}
    result['nodes']={};previous=set();base=0
    profiles={};profile_keys={};symbols=[];symbol_index={}
    def encode(ids):
        if not intern_read_ids:return ids
        encoded=[]
        for value in ids:
            if type(value)is not str or not value:raise ValueError('read ID symbol must be a nonempty string')
            index=symbol_index.get(value)
            if index is None:
                index=len(symbols)
                if index>UINT64_MAX:raise ValueError('read ID symbol index exceeds uint64')
                symbol_index[value]=index;symbols.append(value)
            encoded.append(index)
        return encoded
    for node,n in sorted(rank['nodes'].items(),key=lambda row:int(row[0])):
        # The full retained tape can contain thousands of read IDs per node.
        # Never duplicate that list merely to discard it for delta encoding.
        row=copy.deepcopy({k:v for k,v in n.items() if k not in ('reads','core_workspace_bytes')})
        reads=set(n['reads'])
        row['read_delta']={'base_node':base,'add':encode(sorted(reads-previous)),'remove':encode(sorted(previous-reads))}
        if 'core_workspace_bytes' in n:
            vector=n['core_workspace_bytes'];signature=tuple(vector)
            key=profile_keys.get(signature)
            if key is None:
                key=hashlib.sha256(json.dumps(vector,separators=(',',':')).encode()).hexdigest()
                profile_keys[signature]=key;profiles[key]=copy.deepcopy(vector)
            row['core_workspace_profile']=key
        result['nodes'][node]=row;previous=reads;base=int(node)
    if profiles:result['core_workspace_profiles']=profiles
    if intern_read_ids:result['read_id_symbols']=symbols
    return result


def expand_rank(rank):
    result={k:copy.deepcopy(v) for k,v in rank.items() if k not in ('nodes','core_workspace_profiles','read_id_symbols')}
    result['nodes']={};previous=set();base=0
    profiles=rank.get('core_workspace_profiles',{});verified_profiles=set()
    symbol_mode='read_id_symbols'in rank;symbols=rank.get('read_id_symbols')
    if symbol_mode:
        if type(symbols)is not list or any(type(s)is not str or not s for s in symbols):
            raise ValueError('read_id_symbols must be an array of nonempty strings')
        if len(set(symbols))!=len(symbols):raise ValueError('duplicate read ID symbol')
    def decode(items):
        if type(items)is not list:raise ValueError('read delta IDs must be an array')
        if symbol_mode:
            if any(type(i)is not int or not 0<=i<=UINT64_MAX or i>=len(symbols) for i in items):
                raise ValueError('read ID symbol index must be uint64 and in range')
            values=[symbols[i] for i in items]
        else:
            if any(type(s)is not str for s in items):raise ValueError('legacy read delta IDs must be strings')
            values=items
        if len(set(values))!=len(values):raise ValueError('duplicate read delta ID')
        return set(values)
    for node,encoded in sorted(rank['nodes'].items(),key=lambda item:int(item[0])):
        row=copy.deepcopy({k:v for k,v in encoded.items() if k not in ('read_delta','core_workspace_profile')})
        delta=encoded['read_delta']
        removed=decode(delta['remove']);added=decode(delta['add'])
        if type(delta['base_node'])is not int or not 0<=delta['base_node']<=UINT64_MAX or delta['base_node']!=base or not removed<=previous or added&previous:
            raise ValueError('invalid read-set delta chain')
        previous=(previous-removed)|added
        row['reads']=sorted(previous);base=int(node)
        if 'core_workspace_profile' in encoded:
            key=encoded['core_workspace_profile']
            vector=profiles[key]
            if key not in verified_profiles:
                if hashlib.sha256(json.dumps(vector,separators=(',',':')).encode()).hexdigest()!=key:
                    raise ValueError('private workspace profile hash mismatch')
                verified_profiles.add(key)
            row['core_workspace_bytes']=list(vector)
        result['nodes'][node]=row
    return result
