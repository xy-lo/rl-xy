"""Lossless read-set delta encoding for strictly ordered per-rank graphs.

Only serialization changes. The independent replay uses the original complete
contract; the native admission reconstructs the same set at every issue.
"""
import copy
import hashlib
import json


def compact_rank(rank):
    result={k:copy.deepcopy(v) for k,v in rank.items() if k!='nodes'}
    result['nodes']={};previous=set();base=0
    profiles={}
    for node,n in sorted(rank['nodes'].items(),key=lambda row:int(row[0])):
        row=copy.deepcopy(n);reads=set(row.pop('reads'))
        row['read_delta']={'base_node':base,'add':sorted(reads-previous),'remove':sorted(previous-reads)}
        if 'core_workspace_bytes' in row:
            vector=row.pop('core_workspace_bytes')
            key=hashlib.sha256(json.dumps(vector,separators=(',',':')).encode()).hexdigest()
            profiles[key]=vector;row['core_workspace_profile']=key
        result['nodes'][node]=row;previous=reads;base=int(node)
    if profiles:result['core_workspace_profiles']=profiles
    return result


def expand_rank(rank):
    result=copy.deepcopy(rank);previous=set();base=0
    profiles=result.pop('core_workspace_profiles',{})
    for node,row in sorted(result['nodes'].items(),key=lambda item:int(item[0])):
        delta=row.pop('read_delta')
        if delta['base_node']!=base or not set(delta['remove'])<=previous or set(delta['add'])&previous:
            raise ValueError('invalid read-set delta chain')
        previous=(previous-set(delta['remove']))|set(delta['add'])
        row['reads']=sorted(previous);base=int(node)
        if 'core_workspace_profile' in row:
            key=row.pop('core_workspace_profile')
            vector=profiles[key]
            if hashlib.sha256(json.dumps(vector,separators=(',',':')).encode()).hexdigest()!=key:
                raise ValueError('private workspace profile hash mismatch')
            row['core_workspace_bytes']=list(vector)
    return result
