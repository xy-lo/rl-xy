import math,unittest
from rl_xy.grpo_math import group_advantages,policy_objective,aggregate

class GrpoMathTests(unittest.TestCase):
    def test_group_sample_std_and_zero_variance(self):
        a=group_advantages([0,1,2,2],['a','a','b','b'])
        self.assertAlmostEqual(a[0],-.5/(math.sqrt(.5)+1e-6));self.assertEqual(a[2:],[0,0])
        self.assertEqual(group_advantages([0,1],['x','x'],normalize_std=False),[-.5,.5])
    def test_positive_negative_ratio_and_dual_clipping(self):
        a=policy_objective([[math.log(2),math.log(5)]],[[0,0]],[[1,-1]],[[1,1]],kl_coefficient=0)
        self.assertAlmostEqual(a['policy_loss'],(-1.2+3)/2)
    def test_kl_and_mask(self):
        r=policy_objective([[-1,-100]], [[-1,-100]], [[0,100]], [[1,0]],reference=[[-2,0]])
        self.assertAlmostEqual(r['kl_loss'],math.exp(-1));self.assertAlmostEqual(r['loss'],.001*math.exp(-1))
    def test_dp_average_matches_unsplit_token_mean(self):
        values=[[1,2],[9,0]];mask=[[1,1],[1,0]]
        full=aggregate(values,mask)
        local=[aggregate([v],[m],dp_size=2,global_tokens=3) for v,m in zip(values,mask)]
        self.assertAlmostEqual(sum(local)/2,full)
        with self.assertRaises(ValueError):aggregate([[1]],[[1]],dp_size=2)
    def test_sequence_vs_token_aggregation_is_explicit(self):
        a=aggregate([[1,1],[9,0]],[[1,1],[1,0]])
        b=aggregate([[1,1],[9,0]],[[1,1],[1,0]],'seq-mean-token-mean')
        self.assertAlmostEqual(a,11/3);self.assertAlmostEqual(b,5,places=6)
    def test_bad_values_do_not_become_synthetic_rewards(self):
        with self.assertRaises((ValueError,TypeError)):group_advantages([None],['x'])
        with self.assertRaises(ValueError):policy_objective([[0]],[[0]],[[1]],[[1]],reference=[[0]],dual_clip=1)
if __name__=='__main__':unittest.main()
