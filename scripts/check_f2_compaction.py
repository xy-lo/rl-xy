#!/usr/bin/env python3
"""Run a small f2 native integration check on both platforms."""
import argparse
import json
from pathlib import Path
from rl_xy.f2_runtime import configure_native, make_operator, sha
from rl_xy.qwen3_paths import output_path
from rl_xy.f2_runtime import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = output_path(ROOT, args.output)
out.mkdir(parents=True, exist_ok=False)
binary = configure_native()
from rl_xy.qwen3_cost import Block, Cost
from rl_xy.qwen3_native import Plan, execute
from rl_xy.qwen3_sequence import attach, compile_blocks

rows = []
for platform in ('gpu', 'wafer'):
    op = make_operator('14B', platform, cache_dir=out/'wafer-local-cache')
    cost = op.mm(128, op.model.vocab_size//4, op.model.hidden_size) + op.attention((1,), (1000,), 4)
    blocks = [Block('local', cost), Block('AR', Cost(), 'ALL_REDUCE', 4096),
              Block('GATHER', Cost(), 'GATHER', 4096), Block('BCAST', Cost(), 'BCAST', 16)]
    ranks = (0, 1, 2, 3)
    expanded = Plan(op); expanded.blocks(ranks, blocks, 'smoke')
    first = execute(expanded, out/platform/'expanded')
    compact = Plan(op)
    sequence = attach(compact, [(ranks, [{'repeat':1, 'steps':compile_blocks(op, ranks, blocks)}])])
    second = execute(compact, out/platform/'compact', sequence=sequence)
    difference = abs(first['model_ns'] - second['model_ns'])
    if first['status'] != 'PASS' or second['status'] != 'PASS' or difference > 30:
        raise ValueError('f2 native compact/expanded consistency failed')
    rows.append({'platform':platform, 'expanded_ns':first['model_ns'], 'compact_ns':second['model_ns'],
                 'difference_ns':difference, 'native_binary_sha256':sha(binary)})
report = {'status':'PASS', 'scope':'small integration check, not hardware or full-step calibration', 'rows':rows}
(out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
