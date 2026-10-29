#!/usr/bin/env python3
"""Check the canonical document set, relative links, shell syntax and file map."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CURRENT = {'README.md', 'docs/ARCHITECTURE.md', 'docs/REPRODUCING.md', 'docs/EXPERIMENT_DESIGN.md', 'docs/RESULTS.md', 'docs/FILE_MAP.md', 'docs/F2_HANDOFF.md'}
HISTORICAL = {'experiments/2026-10-27-v0.6/实验结果.md', 'experiments/2026-10-27-v0.6/模型内部瓶颈分解.md'}


def source_files():
    skip = {'__pycache__', '.pytest_cache', '.git'}
    return {str(p.relative_to(ROOT)) for folder in ['src', 'scripts', 'tests', 'configs', 'docker', 'calibration'] for p in (ROOT / folder).rglob('*') if p.is_file() and not any(x in skip or x.endswith(('.egg-info', '.dist-info')) for x in p.parts) and p.suffix not in {'.pyc', '.pyo'} and p.name != '.DS_Store'}


def main():
    # Only the source tree is scanned: untracked run outputs do not become repository documents.
    actual = {str(p.relative_to(ROOT)) for folder in ['docs', 'experiments', 'examples'] if (ROOT / folder).exists() for p in (ROOT / folder).rglob('*.md')}
    actual.add('README.md')
    if actual != CURRENT | HISTORICAL:
        raise ValueError('Noncanonical/missing Markdown: ' + repr(sorted(actual ^ (CURRENT | HISTORICAL))))
    frozen = json.loads((ROOT / 'experiments/2026-10-28-v0.9.5/source_freeze.json').read_text())['files']
    for name in HISTORICAL:
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != frozen[name]:
            raise ValueError('Frozen historical document changed: ' + name)
    architecture = (ROOT / 'docs/ARCHITECTURE.md').read_text()
    modules = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*\.py', architecture))
    known = {p.name for folder in ['src', 'scripts'] for p in (ROOT / folder).rglob('*.py')}
    if modules - known:
        raise ValueError('Architecture mentions absent local modules: ' + repr(sorted(modules - known)))
    links, shell_blocks = 0, 0
    for name in sorted(CURRENT):
        path = ROOT / name
        text = path.read_text()
        targets = re.findall(r'\]\(([^\s)]+)\)', text) + re.findall(r'^\[[^\]]+\]:\s+(\S+)', text, re.M)
        for target in targets:
            url = urlsplit(target.strip('<>'))
            if url.scheme or not url.path:
                continue
            if not (path.parent / unquote(url.path)).exists():
                raise ValueError('Broken local link: ' + name + ' → ' + target)
            links += 1
        for block in re.findall(r'```(?:bash|sh)\n(.*?)```', text, re.S):
            process = subprocess.run(['bash', '-n'], input=block, text=True, capture_output=True)
            if process.returncode:
                raise ValueError(name + ': ' + process.stderr)
            shell_blocks += 1
    text = (ROOT / 'docs/FILE_MAP.md').read_text()
    rows = re.findall(r'^\| `([^`]+)` \|', text, re.M)
    if len(rows) != len(set(rows)):
        raise ValueError('Duplicate file-map entries')
    missing = source_files() - set(rows)
    if missing:
        raise ValueError('Source files missing from FILE_MAP: ' + repr(sorted(missing)))
    nonexistent = [name for name in rows if not (ROOT / name).is_file()]
    if nonexistent:
        raise ValueError('FILE_MAP paths missing: ' + repr(nonexistent))
    print(json.dumps(dict(status='PASS', current_documents=len(CURRENT), frozen_historical_documents=len(HISTORICAL), local_links=links, shell_blocks=shell_blocks, indexed_files=len(rows), covered_source_files=len(source_files()), raw_native_replay=False)))


if __name__ == '__main__':
    main()
