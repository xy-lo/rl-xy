#!/usr/bin/env python3
"""Package audited runs without altering their archived source bytes.

Relocation changes only command.trace_path and source archive destinations.
Original bytes of changed documents remain in the archive for identity proof.
This never publishes, deletes evidence, or labels a fixture as a final result.
"""
import argparse,json,hashlib,shutil,tarfile,tempfile,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def relocate(run,original_root):
    changes={}
    p=run/'command.json';data=load(p);model=data['model'];window=Path(data['trace_path']).stem
    trace=Path('experiments/2026-10-29-qwen3/inputs')/('Qwen3-'+model)/(window+'.json')
    if model not in ('4B','8B','14B') or window not in ('tune','test1','test2','test3','test4'):
        raise ValueError('unrecognized canonical trace')
    assert sha(ROOT/trace)==data['trace_sha256']
    for name in ('command.json','source-archive-map.json'):
        path=run/name;original=path.read_bytes();value=load(path)
        if name=='command.json':value['trace_path']=str(trace)
        else:
            value={k:str(Path(v).resolve().relative_to(original_root.resolve())) if Path(v).is_absolute() else v for k,v in value.items()}
        updated=(json.dumps(value,indent=2)+'\n').encode()
        if updated!=original:
            historical=run/'historical'/name;historical.parent.mkdir(exist_ok=True);historical.write_bytes(original)
            path.write_bytes(updated)
            changes[name]=dict(original_sha256=hashlib.sha256(original).hexdigest(),
                published_sha256=hashlib.sha256(updated).hexdigest(),original_copy='historical/'+name)
    return changes

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();spec=load(a.plan)
    if spec.get('schema')!='idea1.evidence-bundle-plan/1':raise ValueError('unknown packaging plan')
    ids=[x['id'] for x in spec['cases']]
    if len(set(ids))!=len(ids) or any(not re.fullmatch('[A-Za-z0-9_-]+',x) for x in ids):
        raise ValueError('unique safe case IDs required')
    for case in spec['cases']:
        source=Path(case['run']).resolve();output=a.output.resolve()
        if output==source or source in output.parents:raise ValueError('bundle cannot be inside its source run')
    a.output.mkdir(exist_ok=False,parents=True);(a.output/'evidence').mkdir()
    from audit_idea1_inference import check
    manifest=dict(schema='idea1.portable-workload-evidence/1',final=spec.get('final',False),
        description=spec['description'],cases=[],cost_proofs={},extras={},scope=spec.get('scope',{}))
    for model,cost_root in spec.get('cost_roots',{}).items():
        expected=load(ROOT/f'experiments/2026-11-02-idea1-joint/{model}/baseline-metadata.json')['cost_source_files']
        prefix=Path('cost-proof')/model
        for name,digest in expected.items():
            source=Path(cost_root)/name
            if sha(source)!=digest:raise ValueError('cost source changed '+name)
            target=a.output/prefix/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source,target)
        manifest['cost_proofs'][model]=dict(prefix=str(prefix),files=expected,checked_files=len(expected))
    for case in spec['cases']:
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(case['run']);result=load(original/'result.json')
            if result['native_exit_status']!=0 or result['completed']!=result['trained_samples']:
                raise ValueError('incomplete native case '+case['id'])
            # Validate before relocation as well as independently after packaging.
            before=check(original)
            assert before['status']=='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT'
            assert result['requests']==case['expected_answers']
            dst=Path(tmp)/case['id'];dst.mkdir();run=dst/'native-run'
            for source in original.rglob('*'):
                if source.is_symlink():raise ValueError('symlink in evidence '+str(source))
            shutil.copytree(original,run)
            changes=relocate(run,original)
            if case.get('offline_plan'):
                planning=dst/'planning/offline';planning.mkdir(parents=True)
                for name in ('public.json','global-distribution.json','distribution.json','forecast.json','split-manifest.json','offline-candidates.json','result.json'):
                    shutil.copy2(Path(case['offline_plan'])/name,planning/name)
                from audit_idea1_planning import audit_planning
                planning_audit=audit_planning(run,planning)
                (dst/'planning/audit.json').write_text(json.dumps(planning_audit,indent=2))
            elif case['kind']=='final':
                raise ValueError('final case requires complete offline planning provenance')
            after=check(run);assert after['status']==before['status']
            identities={}
            for path in sorted(dst.rglob('*')):
                if path.is_file():identities[str(path.relative_to(dst))]=sha(path)
            archive=a.output/'evidence'/(case['id']+'.tar.gz')
            with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
                for name in identities:tar.add(dst/name,arcname=name,recursive=False)
            meta={k:v for k,v in case.items() if k not in ('run','offline_plan')}
            meta.update(model=result['model'].removeprefix('Qwen3-'),window=result['window'],
                total_device_s=result['total_s'],archive=str(archive.relative_to(a.output)),files=identities,
                relocation=changes,audit_status=before['status'],binary_sha256=result['binary_sha256'])
            manifest['cases'].append(meta)
    for name,source in spec.get('extras',{}).items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] in ('evidence','cost-proof'):
            raise ValueError('unsafe extra destination')
        target=a.output/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);manifest['extras'][name]=sha(target)
    manifest['files']={str(path.relative_to(a.output)):sha(path) for path in sorted(a.output.rglob('*')) if path.is_file()}
    (a.output/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    from check_idea1_workload import verify_bundle
    print(json.dumps(verify_bundle(a.output),indent=2))
if __name__=='__main__':main()
