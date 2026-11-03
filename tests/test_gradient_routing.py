import unittest
from idea1_controller.gradient_routing import range_plan

class GradientRoutingTests(unittest.TestCase):
    def regions(self,n):
        # n roots, canonical distinct partition of original 27 owners.
        return {str(i):[i]+list(range(n+i,27,n)) for i in range(n)}
    def test_exact_coverage_and_owner_identity(self):
        for n in (2,7,8,15,18,26):
            p=range_plan(27*7,self.regions(n))
            seen=[]
            for m in p['moves']:
                seen.extend(range(m['global_start'],m['global_end']))
                self.assertGreaterEqual(m['source_start'],0)
                self.assertLessEqual(m['source_start']+m['elements'],p['source_extent'])
                self.assertGreaterEqual(m['destination_stage_start'],p['source_extent'])
                self.assertLessEqual(m['destination_stage_start']+m['elements'],p['elements'])
                self.assertEqual(m['owner'],m['global_start']//p['owner_extent'])
            self.assertEqual(seen,list(range(p['elements'])))
    def test_numerical_reduction_and_inplace_liveness(self):
        for n in (2,7,15,18,26):
            p=range_plan(27*7,self.regions(n));N=p['elements'];P=p['rs_padded_elements'];S=p['source_extent']
            contributions={r:[float((r+1)*(i%7-3)) for i in range(N)]+[0.]*(P-N) for r in p['roots']}
            total=[sum(x[i] for x in contributions.values()) for i in range(P)]
            # Native RS leaves each root's summed interval at its compact prefix.
            buffers={r:[float('nan')]*P for r in range(27)}
            for i,r in enumerate(p['roots']):buffers[r][:S]=total[i*S:(i+1)*S]
            for m in p['moves']:
                value=buffers[m['source_root']][m['source_start']:m['source_start']+m['elements']]
                a=m['destination_stage_start'];buffers[m['owner']][a:a+m['elements']]=value
            # Only after ALL reads finish may original owner aliases be written.
            for owner in range(27):
                a=owner*p['owner_extent'];s=p['staging_start']
                buffers[owner][a:a+p['owner_extent']]=buffers[owner][s:s+p['owner_extent']]
                self.assertEqual(buffers[owner][a:a+p['owner_extent']],total[a:a+p['owner_extent']])
    def test_original_for_no_sparse_contributors(self):
        self.assertEqual(range_plan(270,{str(i):[i] for i in range(27)})['mode'],'original')
        self.assertEqual(range_plan(270,{'0':list(range(27))})['mode'],'original')
    def test_bad_partition_and_alignment(self):
        for regions in ({'0':list(range(26))},{'0':list(range(27)),'1':[1]}):
            with self.assertRaises(ValueError):range_plan(270,regions)
        with self.assertRaises(ValueError):range_plan(271,self.regions(15))
if __name__=='__main__':unittest.main()
