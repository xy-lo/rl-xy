"""Migrate KV only; matching attention weights are already resident actor views."""
from attention_tp8_program import kv_bytes_per_helper,head_layout
from rl_xy.qwen3_sequence import local

def migrate_actor_view(op,src,dst,contexts,*,tag_start):
    if len(src)!=2 or len(dst)!=2 or set(src)&set(dst):raise ValueError('invalid TP2 pair')
    kv=kv_bytes_per_helper(op,contexts);ss=[];hs=[]
    for i,(s,d) in enumerate(zip(src,dst)):
        msg=dict(kind='p2p_direct',src=s,dst=d,bytes=kv,tag=tag_start+2*i+1)
        ss.extend([local(op.vec(0,kv,kv,'kv_stream_pack'),ranks=[s]),local(op.vec(0,kv,0,'attention_kv_read'),ranks=[s]),msg])
        hs.extend([msg,local(op.vec(0,0,kv,'attention_kv_write'),ranks=[d]),local(op.vec(0,kv,kv,'kv_stream_unpack'),ranks=[d])])
    return dict(source=ss,helper=hs,weight_bytes_per_lane=0,kv_bytes_per_lane=kv,head_layout=head_layout(op,src,dst),pack_extra_read_bytes_per_lane=kv,pack_extra_write_bytes_per_lane=kv)
