"""Canonical gradient expansion includes replicated fields and DO padding."""
from rl_xy.qwen3_sequence import local
from rl_xy.model import enumerate_parameters

class CanonicalGradientProtocol:
    def _restore_gradient(self,r):
        """Reverse-order streaming expansion into the original TP2 buffers.

        Logical shards: t=h*m+k -> old lane h, designated replica root r.
        Root local pieces expand backward, then other k pieces arrive; other
        original DP replicas become zero contributors. FP32 Q/K norm partials
        are summed across k; original suffix merges the two h contributions.
        """
        begin=len(self.restore_proofs)
        ranks=self.region_ranks[r];ws=self.region_members[r];m=len(ws);tp=2*m
        steps=[];tag=460000000+r*100000
        for row_index,row in reversed(list(enumerate(enumerate_parameters(self.op.case_tp(tp))))):
            partial=row.layout=='replicated_per_head_dimension' if hasattr(row,'layout') else row.sharding=='replicated_per_head_dimension'
            if row.replicated and not partial:
                # Equal values need no network, but compact-to-expanded offsets
                # still change. Repack them in the same reverse traversal.
                if m>1:
                    for dst in self.rep[r].ranks:
                        steps.append(local(self.op.vec(0,row.local_elements_per_die*4,row.local_elements_per_die*4,'gradient_repack_equal'),ranks=[dst]))
                continue
            size=row.local_elements_per_die*4
            for h in range(2):
                dst=self.rep[r].ranks[h]
                # Local repack is explicit; source shard has a valid alias only
                # for TP2. Reverse row traversal prevents in-place expansion
                # from overwriting a later row's unread compact producer data.
                if m>1:steps.append(local(self.op.vec(0,size,size,'gradient_local_expand'),ranks=[dst]))
                for k in range(1,m):
                    src=self.rep[ws[k]].ranks[h]
                    # Bounded pack/unpack, all endpoint service explicitly paid.
                    steps += [local(self.op.vec(0,size,size,'gradient_pack'),ranks=[src]),
                              local(self.op.vec(0,size,0,'gradient_send_read'),ranks=[src]),
                              dict(kind='p2p_direct',src=src,dst=dst,bytes=size,tag=tag),
                              local(self.op.vec(0,0,size,'gradient_receive_write'),ranks=[dst]),
                              local(self.op.vec(size//4 if partial else 0,size,size,'gradient_unpack_or_partial_sum'),ranks=[dst])]
                    self.restore_proofs.append(dict(region=r,parameter=row.parameter_id,logical_lane=h,subshard=k,subshards=m,
                        src=src,dst=dst,bytes=size,semantics='partial_sum' if partial else 'canonical_shard_concat',tag=tag))
                    tag+=1
        # All data has arrived before any non-root producer clears its buffer.
        key=f'gradient-restore-{r}';steps += [dict(kind='gate_signal',key=key),dict(kind='gate_wait',key=key,threshold=tp)]
        padded=self.op.state(2,2)['layout'].padded_elements*4
        for w in ws[1:]:
            steps.append(local(self.op.vec(0,0,padded,'zero_noncontributing_dp_gradient'),ranks=list(self.rep[w].ranks)))
        # The original DO suffix reads padded buffers. Initialize the small
        # tail even when no physical model element occupies it.
        valid=sum(x.local_elements_per_die for x in enumerate_parameters(self.op.case_tp(2)))*4
        tail=padded-valid
        if tail:
            assert tail>0
            steps.append(local(self.op.vec(0,0,tail,'gradient_padding_zero'),ranks=list(self.rep[r].ranks)))
        for proof in self.restore_proofs[begin:]:
            descriptor=self.coordinates.restore_descriptor(proof['parameter'],tp,proof['logical_lane'],proof['subshard'])
            assert descriptor['payload_elements']*4==proof['bytes']
            proof['coordinate_descriptor']=descriptor
        return steps

