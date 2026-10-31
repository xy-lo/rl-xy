"""Compile one source/helper decode-token template; experimental draft.

Every invocation has actual P2P send/receive and the original source TP2
collectives. The caller must keep both groups in the lease until all in-flight
messages finish, provide valid resident weights/capacity and add control cost.
"""
from rl_xy.qwen3_cost import Block,Cost
from rl_xy.qwen3_sequence import local,compile_blocks
from rl_xy.operator_ir import KernelSpec,Matmul

def compile_helper_batch(op,src,dst,contexts,*,tag_start=1000):
 contexts=tuple(contexts)
 if not contexts:raise ValueError("empty decode batch")
 batch=len(contexts)
 if len(src)!=4 or len(dst)!=4 or set(src)&set(dst):raise ValueError('two disjoint TP4 groups required')
 if any(type(x) is not int or x<=0 for x in contexts):raise ValueError('positive contexts required')
 h=op.model.hidden_size;f=op.model.ffn_size//8
 quarter=op.mm(batch,2*f,h)+op.vec(batch*f*5,batch*f*4,batch*f*2,'idea1_swiglu')+op.kernel(KernelSpec('idea1_down',matmuls=(Matmul(batch,h,f,a_bytes=2,b_bytes=2,output_bytes=4,accumulator_bytes=4),)))
 c={'quarter':quarter,'input_bytes_per_rank':batch*h*2,'returned_bytes_per_rank':batch*h*4}
 blocks=op.forward((1,)*batch,4,'generation',contexts);next_blocks=op.forward((1,)*batch,4,'generation',tuple(x+1 for x in contexts));ss=[];hs=[];layer=0
 for b,nb in zip(blocks,next_blocks):
  if b.name!=nb.name:raise ValueError('unstable block structure')
  if b.name!='ffn_out':ss.extend(compile_blocks(op,src,[b],[nb]));continue
  ss.extend([local(op.vec(batch*6*h,batch*2*h,batch*2*h,'idea1_ffn_norm')),local(op.vec(0,c['input_bytes_per_rank'],0,'helper_input_read'))])
  for i,(s,d) in enumerate(zip(src,dst)):
   msg={'kind':'p2p_direct','src':s,'dst':d,'bytes':c['input_bytes_per_rank'],'tag':tag_start+layer*8+i};ss.append(msg);hs.append(msg)
  hs.extend([local(op.vec(0,0,c['input_bytes_per_rank'],'helper_input_write')),local(c['quarter']),local(op.vec(0,c['returned_bytes_per_rank'],0,'helper_output_read'))]);ss.append(local(c['quarter']))
  for i,(s,d) in enumerate(zip(src,dst)):
   msg={'kind':'p2p_direct','src':d,'dst':s,'bytes':c['returned_bytes_per_rank'],'tag':tag_start+layer*8+4+i};ss.append(msg);hs.append(msg)
  ss.extend([local(op.vec(0,0,c['returned_bytes_per_rank'],'helper_output_write')),local(op.vec(batch*h,batch*8*h,batch*2*h,'helper_merge_cast'))])
  ss.extend(compile_blocks(op,src,[Block('ffn_tp4_collective',Cost(),'ALL_REDUCE',batch*h*2)]));layer+=1
 if layer!=op.model.layers:raise ValueError('layer coverage mismatch')
 return {'source':ss,'helper':hs,'layers':layer,'input_bytes':4*layer*c['input_bytes_per_rank'],'output_bytes':4*layer*c['returned_bytes_per_rank'],'contexts':contexts,'batch':batch}
