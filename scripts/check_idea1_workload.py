#!/usr/bin/env python3
"""Independent portable workload-system evidence audit, no simulator needed."""
import argparse,json,hashlib,math,tarfile,tempfile
from pathlib import Path
from audit_idea1_inference import check
ROOT=Path(__file__).resolve().parents[1]
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_bundle(root,require_final=False):
    root=Path(root);m=load(root/'bundle-manifest.json')
    assert m['schema']=='idea1.portable-workload-evidence/1'
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='bundle-manifest.json'}
    assert actual==set(m['files'])
    for name,value in m['files'].items():
        path=Path(name);assert not path.is_absolute() and '..' not in path.parts
        assert sha(root/path)==value,name
    for model,proof in m['cost_proofs'].items():
        prefix=Path(proof['prefix']);assert not prefix.is_absolute() and '..' not in prefix.parts
        expected=load(ROOT/f'experiments/2026-11-02-idea1-joint/{model}/baseline-metadata.json')['cost_source_files']
        assert proof['files']==expected and proof['checked_files']==len(expected)==187
        for name,value in expected.items():assert sha(root/proof['prefix']/name)==value
    reports=[]
    with tempfile.TemporaryDirectory() as tmp:
        for case in m['cases']:
            ident=Path(case['id']);archive=Path(case['archive'])
            assert len(ident.parts)==1 and not ident.is_absolute() and case['id'] not in ('.','..')
            assert not archive.is_absolute() and '..' not in archive.parts
            dst=Path(tmp)/case['id'];dst.mkdir()
            with tarfile.open(root/case['archive'],'r:gz') as tar:
                members=tar.getmembers();assert len(members)==len(case['files'])
                assert {x.name for x in members}==set(case['files'])
                for item in members:
                    name=Path(item.name);assert item.isfile() and not name.is_absolute() and '..' not in name.parts
                    content=tar.extractfile(item).read();assert hashlib.sha256(content).hexdigest()==case['files'][item.name]
                    path=dst/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(content)
            run=dst/'native-run'
            for name,identity in case['relocation'].items():
                target=Path(name);original=Path(identity['original_copy'])
                assert not target.is_absolute() and '..' not in target.parts
                assert not original.is_absolute() and '..' not in original.parts
                assert sha(run/name)==identity['published_sha256']
                assert sha(run/identity['original_copy'])==identity['original_sha256']
            source_map=load(run/'source-archive-map.json');source_hashes=load(run/'source_hashes.json')
            assert set(source_map)==set(source_hashes)
            for key,path in source_map.items():
                path=Path(path);assert not path.is_absolute() and '..' not in path.parts
                assert sha(run/path)==source_hashes[key]
            cmd=load(run/'command.json')
            trace=Path(cmd['trace_path'])
            assert not trace.is_absolute() and '..' not in trace.parts
            assert sha(ROOT/trace)==cmd['trace_sha256']
            report=check(run);result=load(run/'result.json')
            assert report['status']==case['audit_status']=='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT'
            assert result['requests']==result['completed']==result['trained_samples']==case['expected_answers']
            assert result['total_s']==case['total_device_s'] and result['binary_sha256']==case['binary_sha256']
            reports.append(dict(id=case['id'],model=case['model'],window=case['window'],answers=case['expected_answers'],total_device_s=result['total_s']))
    if require_final or m['final']:
        assert m['final']
        final=[c for c in m['cases'] if c['kind']=='final']
        assert {(c['model'],c['window']) for c in final}=={(x,'test1') for x in ('4B','8B','14B')} and len(final)==3
        assert all(c['expected_answers']==1536 for c in final)
        assert set(m['cost_proofs'])=={'4B','8B','14B'}
        comparison=load(root/'comparison.json')
        assert comparison['status']=='FINAL_NATIVE_RESULTS_READY_FOR_REVIEW'
        assert len(comparison['comparison'])==3
        import csv
        old=list(csv.DictReader((ROOT/'experiments/2026-11-03-idea1-elastic/comparison.csv').open()))
        for row in comparison['comparison']:
            case=next(c for c in final if c['model']==row['model'])
            baseline=next(r for r in old if r['model']==row['model'] and r['window']==row['window'])
            assert row['device_s']==case['total_device_s']
            assert row['gpu_s']==float(baseline['gpu_s']) and row['old_complete_system_device_s']==float(baseline['device_s'])
            assert math.isclose(row['speedup_gpu_device'],row['gpu_s']/row['device_s'],rel_tol=1e-12)
            assert math.isclose(row['reduction_old_complete_device'],1-row['device_s']/row['old_complete_system_device_s'],rel_tol=1e-12)
    return dict(status='PASS_PORTABLE_WORKLOAD_SYSTEM_EVIDENCE',final=m['final'],cases=reports,cost_models=sorted(m['cost_proofs']),scope='trace/cost timing simulation; not Transformer numerical training, learning quality or hardware calibration')

def main():
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('--require-final',action='store_true');a=p.parse_args()
    print(json.dumps(verify_bundle(a.bundle,a.require_final),indent=2))
if __name__=='__main__':main()
