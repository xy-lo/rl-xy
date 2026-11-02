#!/usr/bin/env python3
"""Prepare bound history/service inputs without modifying the cost workspace."""
import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'src'))
from idea1_controller.history import import_history,validate_history


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=('4B','8B','14B'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--history-source',type=Path,help='optional original lengths.csv.gz; otherwise use archived disjoint proxy')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assets=ROOT/'experiments/2026-11-03-idea1-complete/assets'
    inputs=ROOT/'experiments/2026-10-29-qwen3/inputs'/('Qwen3-'+a.model)
    if a.history_source:history=import_history(a.history_source,inputs,a.model,out/'history.json')
    else:
        with gzip.open(assets/f'history-{a.model}.json.gz','rt') as f:history=json.load(f)
        (out/'history.json').write_text(json.dumps(history,separators=(',',':'))+'\n')
    prompts=json.loads((inputs/'prompt_lengths.json').read_text())
    public=[dict(group_id=g,prompt_length=x['prompt_length'],prompt_sha256=x['rendered_sha256'],library=x['library']) for g,x in prompts.items()]
    validate_history(public,history,a.model,hashlib.sha256((inputs/'config.json').read_bytes()).hexdigest())
    shutil.copy2(assets/f'catalog-{a.model}.json',out/'catalog.json')
    shutil.copy2(assets/'cost-backend-validation.json',out/'cost-backend-validation.json')
    receipt=dict(model=a.model,history_snapshot_id=history['snapshot_id'],history_kind=history['kind'],
        chronological=history['chronological'],files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},
        scope='offline independent same-prompt proxy; no actor-version confidence or physical timing calibration')
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='PASS_BOUND_INPUTS',model=a.model,output=str(out))))


if __name__=='__main__':main()
