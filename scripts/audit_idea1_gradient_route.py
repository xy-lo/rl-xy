"""Independent sparse RS graph coverage audit, retaining canonical owners."""
import json,math
from pathlib import Path


def audit_gradient_route(folder,ops,regions,capacity):
    p=json.loads((Path(folder)/'gradient-route.json').read_text())
    if p['mode']=='original':
        assert all(o['ranks']==list(range(o['ranks'][0],54,2)) for o in ops if o['kind']=='collective' and 'dp_gradient_RS' in o['name'])
        return
    assert p['mode']=='sparse-root-rs'
    roots=sorted(regions);D=len(roots);N=p['elements'];M=N//27
    assert 2<=D<27 and N%27==0 and p['roots']==roots and p['owners']==27
    assert p['regions']=={str(r):list(ws) for r,ws in regions.items()}
    P=((N+math.lcm(D,27)-1)//math.lcm(D,27))*math.lcm(D,27);S=P//D
    assert p['rs_padded_elements']==P and p['source_extent']==S and p['owner_extent']==M and p['staging_start']==N-M
    assert S<=N-M and p['extra_allocation_bytes_per_contributor']==4*(P-N)
    assert p['capacity']['passed'] and p['capacity']['required_bytes']==capacity['phase_peak_bytes']['gradient_restore']+4*(P-N)
    assert capacity['phase_peak_bytes']['optimizer_rs']==p['capacity']['required_bytes']<=capacity['capacity_bytes']
    expected=[]
    for i,root in enumerate(roots):
        for owner in range(27):
            a=max(i*S,owner*M);b=min((i+1)*S,(owner+1)*M,N)
            if b>a:expected.append(dict(index=len(expected),source_root=root,owner=owner,global_start=a,global_end=b,source_start=a-i*S,destination_stage_start=N-M+a-owner*M,owner_view_start=a,elements=b-a))
    assert p['moves']==expected
    for lane in (0,1):
        rs=[o for o in ops if o['kind']=='collective' and 'dp_gradient_RS' in o['name'] and o['ranks'][0]%2==lane]
        assert len(rs)==1 and rs[0]['ranks']==[2*r+lane for r in roots] and rs[0]['payload_bytes']==P*4
        expected_network=[]
        reads_by_rank={}
        for m in expected:
            src=2*m['source_root']+lane;dst=2*m['owner']+lane
            name='sparse_owner_route.lane%d.move%d'%(lane,m['index'])
            rows=[o for o in ops if name+'.' in o['name'] or o['name'].endswith('.'+name)]
            assert all(o['id']>rs[0]['id'] for o in rows)
            if src==dst:
                assert len(rows)==1 and rows[0]['kind']=='compute' and rows[0]['ranks']==[src]
                reads_by_rank.setdefault(src,[]).append(rows[0]['id'])
            else:
                assert len(rows)==3 and [o['kind'] for o in rows]==['compute','p2p','compute']
                move=rows[1];assert (move['src'],move['dst'],move['payload_bytes'])==(src,dst,m['elements']*4)
                assert rows[0]['ranks']==[src] and rows[-1]['ranks']==[dst]
                reads_by_rank.setdefault(src,[]).append(move['id']);expected_network.append(move['id'])
        actual=[o['id'] for o in ops if o['kind']=='p2p' and ('sparse_owner_route.lane%d.'%lane) in o['name']]
        assert actual==expected_network
        for owner in range(26):
            rank=2*owner+lane
            copies=[o for o in ops if o['name'].endswith('.'+'sparse_owner_view.lane%d.owner%d'%(lane,owner)) and o['kind']=='compute']
            assert len(copies)==1 and copies[0]['ranks']==[rank]
            assert all(i<copies[0]['id'] for i in reads_by_rank.get(rank,[]))
        # Owner ordinal and every subsequent optimizer stage remain canonical.
        ag=[o for o in ops if o['kind']=='collective' and 'dp_parameter_AG' in o['name'] and o['ranks'][0]%2==lane]
        assert len(ag)==1 and ag[0]['ranks']==list(range(lane,54,2))
    assert p['compiled_p2p_messages']==2*sum(m['source_root']!=m['owner'] for m in expected)
    assert p['logical_payload_bytes']==2*sum(4*m['elements'] for m in expected if m['source_root']!=m['owner'])
