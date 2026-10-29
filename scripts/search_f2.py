#!/usr/bin/env python3
"""Run the finite analytical f2 GPU screen on the tune window only."""
import argparse
import json
from pathlib import Path
from rl_xy.f2_runtime import INPUTS, ROOT, make_operator, sha, verify_assets
from rl_xy.qwen3_experiment import Screen
from rl_xy.qwen3_paths import output_path

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=('4B', '8B', '14B'), required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
manifest = verify_assets()
input_path = INPUTS / ('Qwen3-' + args.model) / 'tune.json'
if sha(input_path) != manifest['input_hashes'][args.model]['tune.json']:
    raise ValueError('tune input differs from the frozen public trace')
out = output_path(ROOT, args.output)
out.mkdir(parents=True, exist_ok=False)
requests = json.loads(input_path.read_text())['trajectories']
op = make_operator(args.model, 'gpu')
results = Screen(op, requests).run(out)
print(json.dumps({'status':'PASS', 'model':args.model, 'candidates':len(results),
                  'scope':'finite analytical screen; not native or physical global optimum',
                  'best':results[0] if results else None}, indent=2))
