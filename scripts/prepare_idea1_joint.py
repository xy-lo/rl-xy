#!/usr/bin/env python3
"""Prepare an isolated H0 cost workspace without touching the release checkout."""
import argparse
import os
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(model, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('src', 'configs', 'docker'):
        shutil.copytree(ROOT/name, output/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', '.gitignore'):
        shutil.copy2(ROOT/name, output/name)
    for name in ('2026-10-29-qwen3-f2', '2026-10-29-qwen3/inputs'):
        shutil.copytree(ROOT/'experiments'/name, output/'experiments'/name)
    patch = ROOT/'experiments/2026-11-02-h0-54'/('h0-14b-source.patch' if model=='14B' else 'h0-source.patch')
    # The destination can sit inside a different, user-owned Git repository.
    # Stop discovery before its parent, so git apply cannot silently skip paths.
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(output.parent))
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=output, env=env, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=output, env=env, check=True)
    # Only provenance loading changes: no arithmetic, capacity or network model.
    source = output/'src/rl_xy/qwen3_source.py'
    text = source.read_text().replace("subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()",
        "json.loads((root/'snapshot-base.json').read_text())['base_commit']")
    source.write_text(text)
    base = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (output/'snapshot-base.json').write_text(json.dumps(dict(base_commit=base,
        scope='prepared release workspace; never the historical run commit', model=model),indent=2)+'\n')
    metadata = json.loads((ROOT/'experiments/2026-11-02-idea1-joint'/model/'baseline-metadata.json').read_text())
    import hashlib
    for name, expected in metadata['cost_source_files'].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Prepared cost source differs: '+name)
    return dict(status='PASS', model=model, output=str(output), checked_cost_files=len(metadata['cost_source_files']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('4B','8B','14B'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.model,a.output),indent=2))
