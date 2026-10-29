"""Replay published Qwen3 policies with verified checkpoints and new identities.

This controller replays the recorded strategies; it does not claim to repeat the
historical tuning search or establish an optimal GPU baseline.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rl_xy.qwen3_cost import Operators
from rl_xy.qwen3_experiment import Policy, run_candidate, write
from rl_xy.qwen3_paths import input_root, native_binary, output_path, campaign_lock
from rl_xy.qwen3_source import manifest, freeze, sha


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=input_root(root))
    parser.add_argument('--policies', type=Path, default=root / 'experiments/2026-10-29-qwen3/policies')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['4B', '8B', '14B'], default=['4B', '8B', '14B'])
    parser.add_argument('--windows', nargs='+', choices=['test1', 'test2', 'test3', 'test4'], default=['test1', 'test2', 'test3', 'test4'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = output_path(root, args.output)
    if out.exists() and not args.resume: parser.error('Output exists; choose a new directory or explicitly use --resume')
    if not native_binary(root).is_file(): parser.error('Native binary missing; configure ASTRA_SIM_ROOT and ASTRA_SIM_BIN for the pinned Qwen3 backend')
    out.mkdir(parents=True, exist_ok=True)
    with campaign_lock(out / 'campaign.lock'):
        source = manifest(root)
        if (out / 'source/manifest.json').exists():
            if json.loads((out / 'source/manifest.json').read_text())['tree_sha256'] != source['tree_sha256']:
                raise RuntimeError('Source differs from checkpoint identity')
        else: freeze(root, out / 'source')
        completed = []
        for model in args.models:
            for platform in ['gpu', 'wafer']:
                op = Operators(root, args.inputs / ('Qwen3-' + model) / 'config.json', platform)
                policy = Policy(**json.loads((args.policies / (model + '-' + platform + '.json')).read_text()))
                for window in args.windows:
                    requests = json.loads((args.inputs / ('Qwen3-' + model) / (window + '.json')).read_text())['trajectories']
                    identity = hashlib.sha256(json.dumps(requests, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    folder = out / ('Qwen3-' + model) / window / platform
                    folder.mkdir(parents=True, exist_ok=True)
                    attempts = sorted(folder.glob('attempt-*')); result = None
                    for attempt in attempts:
                        path = attempt / 'result.json'
                        if not path.exists(): continue
                        item = json.loads(path.read_text())
                        actual = (item.get('status'), item.get('source_tree_sha256'), item.get('trajectory_sha256'), item.get('platform'), item.get('policy'), item.get('compute_scale'), item.get('network_scale'))
                        if actual != ('PASS', source['tree_sha256'], identity, platform, policy.__dict__, 1.0, 1.0):
                            raise RuntimeError('Checkpoint identity mismatch: ' + str(attempt))
                        if {p['binary_sha256'] for p in item['native_proof'].values()} != {sha(native_binary(root))}:
                            raise RuntimeError('Checkpoint native binary mismatch')
                        result = item; break
                    if result is None:
                        attempt = folder / ('attempt-' + str(len(attempts) + 1))
                        try: result = run_candidate(op, requests, policy, attempt)
                        except Exception:
                            write(folder / 'failure.json', {'attempt': attempt.name, 'traceback': traceback.format_exc()})
                            raise
                    completed.append({'model': model, 'platform': platform, 'window': window, 'step_model_ns': result['step_model_ns']})
                    write(out / 'status.json', {'status': 'RUNNING', 'source_tree_sha256': source['tree_sha256'], 'completed': completed})
                    print(model, platform, window, result['step_model_ns'] / 1e9, flush=True)
        write(out / 'status.json', {'status': 'PASS', 'source_tree_sha256': source['tree_sha256'], 'completed': completed, 'hardware_calibrated': False})


if __name__ == '__main__': main()
