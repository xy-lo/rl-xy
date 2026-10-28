"""Lossless read-only ID dictionary tests; no graph/native simulation runs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import unittest

from rl_xy.resource_encoding import compact_rank,expand_rank


def fixture():
    rank={'initial_objects':{'d0.actor.weight':8,'x':4},'dram_capacity':1024,'nodes':{}}
    for i,reads in enumerate([['d0.actor.weight'],['d0.actor.weight','x'],['x'],[]],1):
        rank['nodes'][str(i)]={'reads':reads,'allocations':{},'ready_outputs':[],
            'frees':[],'workspace_bytes':0,'core_workspace_bytes':[0,0]}
    return rank


class ReadSymbolTests(unittest.TestCase):
    def test_optin_roundtrip_old_default_and_unchanged_other_fields(self):
        rank=fixture();before=copy.deepcopy(rank);old=compact_rank(rank);new=compact_rank(rank,intern_read_ids=True)
        self.assertNotIn('read_id_symbols',old)
        self.assertEqual(new['read_id_symbols'],['d0.actor.weight','x'])
        self.assertEqual(expand_rank(new),expand_rank(old));self.assertEqual(expand_rank(new),rank)
        for key in ['initial_objects','dram_capacity']:self.assertEqual(new[key],old[key])
        for n,node in new['nodes'].items():
            self.assertEqual({k:v for k,v in node.items()if k!='read_delta'},
                {k:v for k,v in old['nodes'][n].items()if k!='read_delta'})
            self.assertTrue(all(type(x)is int and x>=0 for k in ['add','remove']for x in node['read_delta'][k]))
        self.assertEqual(rank,before)
    def test_roundtrip_with_no_read_symbols(self):
        rank=fixture()
        for n in rank['nodes'].values():n['reads']=[]
        encoded=compact_rank(rank,intern_read_ids=True)
        self.assertEqual(encoded['read_id_symbols'],[]);self.assertEqual(expand_rank(encoded),rank)
    def test_rejects_bad_indices_and_symbol_tables(self):
        encoded=compact_rank(fixture(),intern_read_ids=True)
        for value in [-1,True,0.0,'0',None,2,2**64]:
            with self.subTest(value=value):
                bad=copy.deepcopy(encoded);bad['nodes']['1']['read_delta']['add']=[value]
                with self.assertRaises(ValueError):expand_rank(bad)
        for symbols in [{},['x','x'],[''],[0],None]:
            with self.subTest(symbols=symbols):
                bad=copy.deepcopy(encoded);bad['read_id_symbols']=symbols
                with self.assertRaises(ValueError):expand_rank(bad)
    def test_duplicate_indices_and_bad_delta_chain_rejected(self):
        encoded=compact_rank(fixture(),intern_read_ids=True)
        for mutate in [lambda x:x['nodes']['1']['read_delta'].update(add=[0,0]),
                lambda x:x['nodes']['2']['read_delta'].update(remove=[1]),
                lambda x:x['nodes']['2']['read_delta'].update(base_node=0)]:
            bad=copy.deepcopy(encoded);mutate(bad)
            with self.assertRaises(ValueError):expand_rank(bad)
    def test_long_repeated_IDs_reduce_plain_native_bytes(self):
        rank=fixture();ids=['persistent.'+str(i)+'.'+'x'*90 for i in range(30)];rank['nodes']={}
        for i in range(200):
            rank['nodes'][str(i+1)]={'reads':sorted(ids[(i%2)*15:(i%2+1)*15]),'allocations':{},'ready_outputs':[],'frees':[]}
        old=compact_rank(rank);new=compact_rank(rank,intern_read_ids=True)
        size=lambda x:len(json.dumps(x,separators=(',',':')).encode())
        self.assertLess(size(new),size(old)//2);self.assertEqual(expand_rank(new),expand_rank(old))


@unittest.skipUnless(os.environ.get('RLXY_READ_ID_HARNESS'),'set precompiled local C++ header-test harness path; no network/build is started by tests')
class NativeReadSymbolHeaderTests(unittest.TestCase):
    def call(self,rank):
        return subprocess.run([os.environ['RLXY_READ_ID_HARNESS']],input=json.dumps(rank),capture_output=True,text=True,timeout=10)
    def test_cpp_old_and_optin_restore_identical_IDs(self):
        rank=fixture();expected={n:x['reads']for n,x in rank['nodes'].items()}
        for encoded in [compact_rank(rank),compact_rank(rank,intern_read_ids=True)]:
            r=self.call(encoded);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),expected)
    def test_cpp_rejects_bad_uint_indices_and_symbol_tables(self):
        encoded=compact_rank(fixture(),intern_read_ids=True)
        for value in [-1,True,0.0,'0',None,2,2**64]:
            with self.subTest(value=value):
                bad=copy.deepcopy(encoded);bad['nodes']['1']['read_delta']['add']=[value]
                self.assertNotEqual(self.call(bad).returncode,0)
        for symbols in [{},['x','x'],[''],[0],None]:
            bad=copy.deepcopy(encoded);bad['read_id_symbols']=symbols
            self.assertNotEqual(self.call(bad).returncode,0)


if __name__=='__main__':unittest.main()
