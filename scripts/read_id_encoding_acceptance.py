#!/usr/bin/env python3
"""Native A/B for lossless read-ID interning; no kernel/network cost changes."""
import argparse,gzip,json,os
from pathlib import Path
from rl_xy.aligned_run import run_aligned
from rl_xy.resource_encoding import expand_rank
ROOT=Path(__file__).resolve().parents[1]
def read_lines(p):
    with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]
def read_json(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    records=[];case=ROOT/'configs/aligned_fixture'
    for platform in ('gpu','wafer'):
        outputs=[]
        for intern in (False,True):
            out=a.out/(platform+('-interned' if intern else '-strings'))
            previous=os.environ.get('RLXY_RESOURCE_EAGER_FLUSH')
            os.environ['RLXY_RESOURCE_EAGER_FLUSH']='0' if intern else '1'
            try:
                run_aligned(case/'case.json',case/'wafer.json',case/'policy.json',out,
                    gpu_profile_path=case/'gpu.json' if platform=='gpu' else None,active=4,intern_resource_read_ids=intern)
            finally:
                if previous is None:os.environ.pop('RLXY_RESOURCE_EAGER_FLUSH',None)
                else:os.environ['RLXY_RESOURCE_EAGER_FLUSH']=previous
            outputs.append(out)
        old,new=outputs
        for name in ['operations.jsonl.gz','events.jsonl.gz']:
            assert read_lines(old/name)==read_lines(new/name),(platform,name)
        first,second=[read_json(d/'resource_contract.json.gz') for d in outputs]
        assert set(first['ranks'])==set(second['ranks'])
        for rank in first['ranks']:
            assert expand_rank(first['ranks'][rank])==expand_rank(second['ranks'][rank]),(platform,rank,'contract')
            name='resources.'+rank+'.jsonl'
            assert (old/'astra_logs'/name).read_bytes()==(new/'astra_logs'/name).read_bytes(),(platform,rank,'raw logs')
        m,n=[json.loads((d/'metrics.json').read_text()) for d in outputs]
        assert m['total_simulated_time_ns']==n['total_simulated_time_ns']
        assert m['memory_peak_bytes_by_location']==n['memory_peak_bytes_by_location']
        sizes=[sum(f.stat().st_size for f in (d/'inputs').glob('resources.*.json')) for d in outputs]
        records.append(dict(platform=platform,status='PASS',all_operation_event_rows_equal=True,
            all_expanded_rank_contracts_equal=True,all_raw_rank_resource_log_bytes_equal=True,
            eager_strings_vs_buffered_interned=True,
            model_ns=n['total_simulated_time_ns'],rank_events=n['operation_rank_event_count'],
            native_rank_JSON_bytes=dict(strings=sizes[0],interned=sizes[1])))
    report=dict(status='PASS',scope='tiny native lossless storage encoding proof, not 3B performance or calibration',comparisons=records)
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
