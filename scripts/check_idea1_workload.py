#!/usr/bin/env python3
"""Independent portable workload-system evidence audit, no simulator needed."""
import argparse,json,hashlib,math,tarfile,tempfile
from pathlib import Path
from audit_idea1_inference import check
ROOT=Path(__file__).resolve().parents[1]
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_comparison(comparison,final,baseline_path):
    """Bind every published baseline column and unique model row, not just speedup."""
    import csv
    assert comparison['status']=='FINAL_NATIVE_RESULTS_READY_FOR_REVIEW'
    rows=comparison['comparison']
    keys={(c['model'],c['window']) for c in final}
    assert len(rows)==len(keys)==3
    assert {(x['model'],x['window']) for x in rows}==keys
    assert comparison['baseline_csv_sha256']==sha(baseline_path)
    with Path(baseline_path).open() as stream:old=list(csv.DictReader(stream))
    fields={'gpu_s':'gpu_s','h0_s':'h0_s','original_generation_s':'original_generation_s',
            'previous_innovation3_s':'previous_innovation3_s','old_complete_system_device_s':'device_s'}
    for row in rows:
        case=next(c for c in final if c['model']==row['model'] and c['window']==row['window'])
        baseline=next(x for x in old if x['model']==row['model'] and x['window']==row['window'])
        assert math.isfinite(row['device_s']) and row['device_s']>0
        assert row['device_s']==case['total_device_s']
        for output,source in fields.items():assert row[output]==float(baseline[source]),output
        assert math.isclose(row['speedup_gpu_device'],row['gpu_s']/row['device_s'],rel_tol=1e-12)
        assert math.isclose(row['reduction_old_complete_device'],1-row['device_s']/row['old_complete_system_device_s'],rel_tol=1e-12)

def verify_deployment(deployment,case,assignment_bytes,layout,inference_bytes,constraint=None):
    """Bind published planner options and original input hashes to executed data."""
    assert deployment['model']==case['model'] and deployment['window']==case['window']
    assignment=json.loads(assignment_bytes)
    fields={'adaptive_training_batch_max':('adaptive_training_batch_max',0),
            'training_tail_fine_samples':('training_tail_fine_samples',0),
            'late_region_batch_cap':('late_region_batch_cap_per_pair',0),
            'training_ready_order':('training_ready_order','input')}
    for published,(native,default) in fields.items():
        value=deployment.get(published,default);actual=layout.get(native,default)
        assert type(value) is type(default) and value==actual,published
        assert assignment.get(native,default)==actual,'executed option differs from original input'
    unscored=deployment.get('runtime_training_packing_unscored',False)
    assert type(unscored) is bool and unscored==bool(layout.get('adaptive_training_batch_max',0))
    for key,data in [('selected-plan.json',assignment_bytes),('inference-layout.json',inference_bytes)]:
        # Native recordings add one trailing LF to the input JSON. Accept that
        # exact serialization difference, without replacing byte hashes by a
        # loose semantic comparison.
        candidates={hashlib.sha256(data).hexdigest(),hashlib.sha256(data.removesuffix(b'\n')).hexdigest()}
        assert deployment['hashes'][key] in candidates,key
    # PB writes standard JSON; the optional PS stage writes indented JSON.
    forecast_hashes={hashlib.sha256(json.dumps(assignment['forecasts'],**formatting).encode()).hexdigest()
        for formatting in ({},{'indent':2})}
    assert deployment['hashes']['forecast.json'] in forecast_hashes
    declared=deployment.get('resource_candidates')
    if declared is None:assert constraint is None
    else:
        assert isinstance(declared,list) and declared
        normalized=set()
        for value in declared:
            inference,training=value.split(':');name,fraction=inference.split('@')
            normalized.add(name+'@'+str(float(fraction))+':'+training)
        assert normalized==set(constraint or []),'resource candidate constraint'


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
    assert len({c['id'] for c in m['cases']})==len(m['cases'])
    for case in m['cases']:
        with tempfile.TemporaryDirectory() as tmp:
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
            assert case['model']==cmd['model']==load(run/'result.json')['model'].removeprefix('Qwen3-')
            assert case['window']==load(run/'result.json')['window']
            if (dst/'planning/offline').exists():
                from audit_idea1_planning import audit_planning
                planning_report=audit_planning(run,dst/'planning/offline')
                assert planning_report==load(dst/'planning/audit.json')
            else:
                assert case['kind']!='final','final planning provenance missing'
            trace=Path(cmd['trace_path'])
            assert not trace.is_absolute() and '..' not in trace.parts
            assert sha(ROOT/trace)==cmd['trace_sha256']
            report=check(run);result=load(run/'result.json')
            assert report['status']==case['audit_status']=='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT'
            assert result['requests']==result['completed']==result['trained_samples']==case['expected_answers']
            assert result['total_s']==case['total_device_s'] and result['binary_sha256']==case['binary_sha256']
            if m['final'] and case['kind']=='final':
                deployment=load(root/'deployment'/(case['model']+'.json'))
                offline=load(dst/'planning/offline/result.json')
                verify_deployment(deployment,case,(run/'assignment.json').read_bytes(),load(run/'resource-layout.json'),
                    (run/'inference-layout.json').read_bytes(),offline.get('candidate_constraint'))
            reports.append(dict(id=case['id'],model=case['model'],window=case['window'],answers=case['expected_answers'],total_device_s=result['total_s']))
    if require_final or m['final']:
        assert m['final']
        final=[c for c in m['cases'] if c['kind']=='final']
        assert {(c['model'],c['window']) for c in final}=={(x,'test1') for x in ('4B','8B','14B')} and len(final)==3
        assert all(c['expected_answers']==1536 for c in final)
        assert set(m['cost_proofs'])=={'4B','8B','14B'}
        comparison=load(root/'comparison.json')
        verify_comparison(comparison,final,ROOT/'experiments/2026-11-03-idea1-elastic/comparison.csv')
        for row in comparison['comparison']:
            deployment=load(root/'deployment'/ (row['model']+'.json'))
            assert deployment['model']==row['model'] and deployment['window']==row['window']
            assert row['planning_host_s']==deployment['host_wall_s']
            assert math.isfinite(row['planning_host_s']) and row['planning_host_s']>0
            assert all(x['returncode']==0 and x['wall_s']>=0 for x in deployment['phases'])
            assert deployment['host_wall_s']>=sum(x['wall_s'] for x in deployment['phases'])
    return dict(status='PASS_PORTABLE_WORKLOAD_SYSTEM_EVIDENCE',final=m['final'],cases=reports,cost_models=sorted(m['cost_proofs']),scope='trace/cost timing simulation; not Transformer numerical training, learning quality or hardware calibration')

def main():
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('--require-final',action='store_true');a=p.parse_args()
    print(json.dumps(verify_bundle(a.bundle,a.require_final),indent=2))
if __name__=='__main__':main()
