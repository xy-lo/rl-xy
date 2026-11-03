import unittest
from types import SimpleNamespace
from idea1_controller.inference_groups import stripe_partition,inference_ranks,validate_partition,kv_reshard_moves,elastic_rectangles,parameter_view_proof


class InferenceGroups(unittest.TestCase):
    def test_tp4_tp8_keep_remainder_and_cover_every_die(self):
        for tp,n in ((2,27),(4,15),(8,9)):
            g=stripe_partition(tp);self.assertTrue(validate_partition(g));self.assertEqual(len(g),n)
            self.assertEqual(sorted(d for ws in g.values() for d in inference_ranks(ws)),list(range(54)))
            self.assertTrue(all(2*len(ws)<=tp for ws in g.values()))

    def test_kv_resize_conserves_every_logical_head_with_reordered_ranks(self):
        model=SimpleNamespace(kv_heads=8,layers=2,head_dim=4,actor_dtype_bytes=2)
        for a,b in (((2,3),(0,2,1,3)),((0,6,1,7),(0,2,6,8,1,3,7,9))):
            moves=kv_reshard_moves(a,b,[11,19],model)
            self.assertEqual(sum(x['bytes'] for x in moves),2*2*30*4*2*8)
            for h in range(8):self.assertEqual(sum(x['head_start']<=h<x['head_end'] for x in moves),1)
            self.assertTrue(any(x['src']!=x['dst'] for x in moves))

    def test_kv_repartition_never_overwrites_an_unretired_source_or_local_tail(self):
        m=SimpleNamespace(kv_heads=8,layers=1,head_dim=1,actor_dtype_bytes=2)
        for old,new in (((0,6,1,7),(0,6,12,18,1,7,13,19)),((12,18,13,19),(0,6,12,18,1,7,13,19)),((0,1),(0,2,1,3))):
            moves=kv_reshard_moves(old,new,[3],m)
            for i,x in enumerate(moves):
                later=moves[i+1:]
                if x['src']==x['dst']:self.assertFalse(any(y['src']==x['src'] and y['dst']!=y['src'] for y in later))
                elif x['dst'] in old:self.assertFalse(any(y['src']==x['dst'] for y in later))

    def test_supported_growth_shapes_are_connected_and_have_no_duplicate_dies(self):
        for ws in elastic_rectangles():
            self.assertIn(len(ws),(2,4));self.assertEqual(len(ws),len(set(ws)))
            reached={ws[0]}
            while True:
                nxt=reached|{b for a in reached for b in ws if abs(a//3-b//3)+abs(a%3-b%3)==1}
                if nxt==reached:break
                reached=nxt
            self.assertEqual(reached,set(ws))

    def test_tp1_weight_assembly_and_wide_views_cover_canonical_components(self):
        m=SimpleNamespace(hidden_size=64,ffn_size=128,query_width=64,kv_heads=8,head_dim=8,vocab_size=128,layers=1,tie_word_embeddings=False,qk_norm=True,qkv_bias=True)
        p1=parameter_view_proof(m,stripe_partition(2),True)
        p4=parameter_view_proof(m,stripe_partition(4))
        self.assertEqual(len(p1['views']),54);self.assertEqual(len(p4['views']),54)
        self.assertTrue(all(v['tp']==1 for v in p1['views'].values()))
        self.assertGreater(p1['views']['0']['elements'],p4['views']['0']['elements'])


if __name__=='__main__':unittest.main()
