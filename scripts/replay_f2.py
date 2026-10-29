#!/usr/bin/env python3
"""Replay a fixed f2 policy from the public length trace on a built backend."""
import argparse
import json
from pathlib import Path
from rl_xy.f2_runtime import ASSETS, INPUTS, ROOT, configure_native, make_operator, sha, verify_assets
from rl_xy.qwen3_paths import output_path

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=('4B', '8B', '14B'), required=True)
parser.add_argument('--platform', choices=('gpu', 'wafer'), required=True)
parser.add_argument('--window', choices=('tune', 'test1', 'test2', 'test3', 'test4'), default='test1')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
native = configure_native()
from rl_xy.qwen3_experiment import Policy, run_candidate
manifest = verify_assets()
input_path = INPUTS / ('Qwen3-' + args.model) / (args.window + '.json')
if sha(input_path) != manifest['input_hashes'][args.model][args.window + '.json']:
    raise ValueError('input differs from the frozen public trace')
requests = json.loads(input_path.read_text())['trajectories']
policy = Policy(**json.loads((ASSETS / 'policies' / (args.model + '-' + args.platform + '.json')).read_text()))
out = output_path(ROOT, args.output)
op = make_operator(args.model, args.platform, cache_dir=out.parent / 'f2-wafer-local-cache')
result = run_candidate(op, requests, policy, out)
print(json.dumps({'status':result['status'],'model':args.model,'platform':args.platform,
                  'window':args.window,'step_s':result['step_model_ns']/1e9,
                  'native_binary_sha256':sha(native),'result':str(out/'result.json')}, indent=2))
