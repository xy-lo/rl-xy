#!/usr/bin/env python3
"""Verify published Qwen3 inputs, recorded phase ledgers and pooled metrics.

Checks stored evidence and arithmetic only; does not replay raw native logs or
establish hardware calibration or strategy optimality.
"""
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['4B', '8B', '14B']
PHASES = ['weight_sync', 'generation', 'samples', 'old_policy', 'reference', 'advantage', 'training', 'optimizer', 'publish']
SOURCE_TREE = 'd7530f67e57de75cbcbfd4fc518bfb5cb6f738bdc3f895eb0014cee18bfb45ad'


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def rows(path):
    with path.open() as f: return list(csv.DictReader(f))
def close(a, b): return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-7)


def check(data=None, root=ROOT):
    data = Path(data) if data else root / 'experiments/2026-10-29-qwen3'
    provenance = read(data / 'provenance.json'); reliability = read(data / 'reliability.json')
    require(provenance['hardware_calibrated'] is False and reliability['hardware_calibrated'] is False, 'hardware_calibrated must remain false for these recorded results')
    require(provenance['paper_performance_ready'] is False and reliability['paper_performance_ready'] is False, 'Paper scope promoted')
    require(provenance['gpu_global_optimum_proven'] is False and provenance['raw_native_logs_distributed'] is False, 'Evidence scope promoted')
    inventory = read(data / 'publication_manifest.json')['files']
    actual = {str(p.relative_to(data)) for p in data.rglob('*') if p.is_file() and p.name != 'publication_manifest.json'}
    require(set(inventory) == actual, 'Published file inventory differs')
    for name, expected in inventory.items():
        path = data / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink(), 'Unsafe evidence path')
        require(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'Evidence checksum differs: ' + name)
    original = read(data / 'inputs/manifest.json')
    require(sha(data / 'inputs/windows.json') == original['windows_sha256'], 'Window selection identity differs')
    selected = read(data / 'inputs/windows.json')['windows']
    require(len(selected) == 5 and len({i for w in selected for i in w['prompt_ids']}) == 960, 'Window prompts overlap or are incomplete')
    formal, phases, summary, window_metrics, comparisons = [rows(data / name) for name in ['formal_windows.csv', 'formal_phases.csv', 'summary.csv', 'window_metrics.csv', 'paired_comparison.csv']]
    require((len(formal), len(phases), len(summary), len(window_metrics), len(comparisons)) == (24, 216, 6, 24, 3), 'Table counts differ')
    require(len({(x['model'], x['window'], x['platform']) for x in formal}) == 24, 'Duplicate formal run')
    work = {}; trajectories = {}
    for model in MODELS:
        config = next(m for m in original['models'] if m['model'] == 'Qwen3-' + model)
        folder = data / 'inputs' / ('Qwen3-' + model)
        for name in ['config.json', 'prompt_lengths.json', 'tune.json', 'test1.json', 'test2.json', 'test3.json', 'test4.json']:
            require(sha(folder / name) == config['files'][name], 'Pinned input checksum differs: ' + model + '/' + name)
        for window in ['test1', 'test2', 'test3', 'test4']:
            item = read(folder / (window + '.json')); xs = item['trajectories']
            require((item['prompt_count'], item['responses_per_prompt'], len(xs)) == (192, 8, 1536), 'Trajectory counts differ')
            require(len({x['trajectory_id'] for x in xs}) == 1536, 'Duplicate trajectory')
            selection = next(w for w in selected if w['name'] == window)
            require({x['group_id'] for x in xs} == {str(i) for i in selection['prompt_ids']}, 'Selected prompt IDs differ')
            require(all(0 < x['response_length'] <= 8192 for x in xs), 'Response lengths outside recorded domain')
            identity = hashlib.sha256(json.dumps(xs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            trajectories[model, window] = identity
            work[model, window] = sum(x['prompt_length'] + x['response_length'] for x in xs)
    for item in formal:
        model, window, platform = item['model'], item['window'], item['platform']
        require(item['source_tree_sha256'] == SOURCE_TREE and item['trajectory_sha256'] == trajectories[model, window], 'Run source/input identity differs')
        result = read(data / item['evidence_path'] / 'result.json')
        require(result['status'] == 'PASS' and result['request_count'] == 1536, 'Recorded run failed or incomplete')
        require(result['source_tree_sha256'] == SOURCE_TREE and result['trajectory_sha256'] == trajectories[model, window], 'Result identity differs')
        require(result['platform'] == platform and result['policy'] == read(data / 'policies' / (model + '-' + platform + '.json')), 'Policy/platform differs')
        require(set(result['stage_ns']) == set(PHASES), 'Stage inventory differs')
        require(close(sum(result['stage_ns'].values()), result['step_model_ns']), 'Result stage sum differs')
        require(close(result['step_model_ns'] / 1e9, item['step_s']), 'Recorded step time differs from table')
        require({p['binary_sha256'] for p in result['native_proof'].values()} == {provenance['binary_sha256']}, 'Native binary identity differs')
        ps = [p for p in phases if (p['model'], p['window'], p['platform']) == (model, window, platform)]
        require(len(ps) == 9 and {p['phase'] for p in ps} == set(PHASES), 'Phase table incomplete')
        require(close(sum(float(p['seconds']) for p in ps), item['step_s']), 'Phase table step sum differs')
        for p in ps:
            require(close(float(p['seconds']) * 1e9, result['stage_ns'][p['phase']]), 'Phase seconds differ')
            ledger = sum(float(p[k]) for k in ['compute_s', 'memory_s', 'communication_s', 'rounding_and_markers_s'])
            require(close(ledger, p['seconds']), 'Phase ledger does not close')
        metric = next(x for x in window_metrics if (x['model'], x['window'], x['platform']) == ('Qwen3-' + model, window, platform))
        require(int(metric['effective_tokens']) == work[model, window] and close(metric['device_step_s'], item['step_s']), 'Metric work/time differs')
        require(close(metric['effective_tokens_per_s'], work[model, window] / float(item['step_s'])), 'Window throughput differs')
    for model in MODELS:
        total = sum(work[model, w] for w in ['test1', 'test2', 'test3', 'test4'])
        gpu = next(x for x in summary if (x['model'], x['platform']) == ('Qwen3-' + model, 'gpu'))
        wafer = next(x for x in summary if (x['model'], x['platform']) == ('Qwen3-' + model, 'wafer'))
        for platform, s in [('gpu', gpu), ('wafer', wafer)]:
            times = [float(x['step_s']) for x in formal if (x['model'], x['platform']) == (model, platform)]
            require(close(statistics.mean(times), s['step_mean_s']) and close(statistics.stdev(times), s['step_sample_sd_s']), 'Mean/SD differs')
            require(int(s['total_effective_tokens']) == total and close(total / sum(times), s['pooled_effective_tokens_per_s']), 'Pooled throughput differs')
            require(close(float(s['time_ratio_to_gpu']) * float(s['throughput_ratio_to_gpu']), 1), 'Normalization reciprocity differs')
        c = next(x for x in comparisons if x['model'] == 'Qwen3-' + model)
        ratio = float(wafer['step_mean_s']) / float(gpu['step_mean_s'])
        require(close(c['time_reduction_percent'], 100 * (1 - ratio)), 'Time reduction differs')
        require(close(c['throughput_gain_percent'], 100 * (1 / ratio - 1)), 'Throughput gain differs')
    historical = read(data / 'source/manifest.json')
    canonical = json.dumps(historical['files'], sort_keys=True, separators=(',', ':')).encode()
    require(hashlib.sha256(canonical).hexdigest() == historical['tree_sha256'] == SOURCE_TREE, 'Historical source tree identity differs')
    core = ['qwen3_cost.py', 'qwen3_rollout.py', 'qwen3_sequence.py', 'gpu_model.py', 'capacity_mapping.py', 'mapping.py', 'multicore.py', 'cost_model.py', 'model.py', 'model_import.py', 'config.py', 'types.py', 'workflow.py']
    changes = [name for name in core if sha(root / 'src/rl_xy' / name) != historical['files']['src/rl_xy/' + name]]
    return {'status': 'PASS', 'formal_runs': 24, 'paired_windows': 12, 'phase_rows': 216,
            'published_files': len(inventory), 'effective_tokens_by_model': {m: sum(work[m,w] for w in ['test1','test2','test3','test4']) for m in MODELS},
            'current_vs_historical_cost_file_changes': changes, 'hardware_calibrated': False, 'paper_performance_ready': False,
            'scope': 'Published stored inputs, summaries and ledgers only; no raw-log replay or hardware calibration'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--data', type=Path)
    args = parser.parse_args(); print(json.dumps(check(args.data), ensure_ascii=False, indent=2))
