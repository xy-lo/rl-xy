#!/usr/bin/env python3
"""Validate distributed summary evidence; never replay absent native logs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT = 'experiments/2026-10-28-v0.9.5'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def check_current(root=ROOT):
    data = root / CURRENT
    publication = read(data / 'publication_manifest.json')
    require(publication['raw_native_logs_distributed'] is False, 'Raw-log distribution scope changed')
    wanted = set(publication['files'])
    actual = {str(p.relative_to(data)) for p in data.rglob('*') if p.is_file() and p.name != 'publication_manifest.json'}
    require(wanted == actual, 'Public evidence inventory differs')
    for name, expected in publication['files'].items():
        path = data / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink(), 'Unsafe public evidence path')
        require(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'Public evidence SHA/size differs: ' + name)
    pair, facts = read(data / 'pair_verification.json'), read(data / 'results.json')
    require(pair['status'] == 'PASS_CONDITIONAL_PAIR' and pair['conditional_ratio_released'] is True, 'Pair not accepted')
    require(pair['checks'] and all(r['status'] == 'PASS' for r in pair['checks']), 'Pair gate failed')
    require(facts['strict_pair_report_sha256'] == sha(data / 'pair_verification.json'), 'Facts not bound to pair')
    require(pair['paper_performance_ready'] is False and pair['paper_readiness_promoted_by_checker'] is False and facts['paper_main_ready'] is False, 'Paper scope promoted')
    times, peaks = {}, {}
    for label, key, allocated in [('gpu', 'GPU', 48), ('wafer', 'WAFER', 54)]:
        base = data / label
        manifest, metrics, run, verification = [read(base / name) for name in ['manifest.json', 'metrics.json', 'run.json', 'verification.json']]
        require(manifest['status'] == 'success' and manifest['exit_status'] == 0 and manifest['checks_failed'] == 0, label + ': native status')
        require(run['status'] == 'PASS' and run['exit_status'] == 0 and run['OOMKilled'] is False, label + ': wrapper status')
        require(run['elapsed_wall_s'] < 900 and run['artifact_bytes'] <= run['output_limit_bytes'], label + ': budget')
        require(verification['status'] == 'PASS' and all(x['status'] == 'PASS' for x in verification['checks']), label + ': MVP status')
        require(verification['manifest_sha256'] == sha(base / 'manifest.json'), label + ': MVP manifest binding')
        binding = pair['raw_bundles'][key]
        require(binding['manifest_sha256'] == sha(base / 'manifest.json') and binding['MVP_verification_sha256'] == sha(base / 'verification.json'), label + ': pair binding')
        require(binding['MVP_checker_source_sha256'] == verification['checker_source_sha256'], label + ': checker source binding')
        for name in ['metrics.json', 'actual_gemm_census.json', 'global_stage_times.json', 'wallclock_timings.json']:
            require(sha(base / name) == manifest['artifact_sha256'][name], label + ': original artifact SHA ' + name)
        require(sha(base / 'metrics.json') == manifest['metrics_sha256'], label + ': original metric binding')
        for name in ['case.json', 'model_config.json', 'policy.json', 'workload_provenance.json']:
            require(sha(data / 'inputs' / name) == manifest['inputs'][name]['sha256'], label + ': original input ' + name)
        require(metrics['active_devices'] == 48 and metrics['dies'] == allocated and metrics['trajectory_count'] == 3072 and metrics['response_tokens'] == 393216, label + ': counts')
        require(metrics['optimizer_updates'] == 1 and metrics['actor_version_end'] == 1 and metrics['reference_version_end'] == 0, label + ': versions')
        for flag in ['paper_performance_ready', 'hardware_calibrated', 'numerical_training', 'full_verl_runtime_reproduced']:
            require(metrics[flag] is False, label + ': unsupported claim ' + flag)
        require(metrics['policy_fingerprint'] == read(data / 'gpu/metrics.json')['policy_fingerprint'], 'Policy differs')
        require(run['image_id'] == binding['image_id'] == manifest['software']['container_image_id'], label + ': image binding')
        require(manifest['software']['rl_xy_source_tree_sha256'] == binding['source_tree_sha256'], label + ': runtime source binding')
        times[key] = metrics['total_simulated_time_ns'] / 1e9
        require(times[key] == facts['model_seconds'][key] and metrics['total_simulated_time_ns'] == pair['model_time_ns'][key], label + ': time')
        peaks[key] = max(v for k, v in metrics['memory_peak_bytes_by_location'].items() if k.endswith('.dram'))
        require(peaks[key] == pair['peaks'][key]['max_active_rank_DRAM_bytes'], label + ': peak')
    ratio = times['GPU'] / times['WAFER']
    require(close(ratio, pair['conditional_gpu_time_over_wafer_time']) and close(ratio, facts['GPU_over_WAFER_time']), 'Ratio differs')
    require(close((times['WAFER'] / times['GPU'] - 1) * 100, facts['WAFER_time_penalty_percent']), 'Penalty differs')
    gc, wc = read(data / 'gpu/actual_gemm_census.json'), read(data / 'wafer/actual_gemm_census.json')
    require(gc['shapes'] == wc['shapes'] and gc['signatures'] == wc['signatures'], 'Published GEMM domain differs')
    history = read(data / 'execution_history.json')
    require(len(history['entries']) == 42 and len(history['execution_identity_groups']) == 29, 'Execution history inventory differs')
    require(history['coverage']['unique_execution_states'] == {'completed_wrapper_pass': 20, 'completed_wrapper_fail': 9} and history['paper_performance_ready'] is False, 'Execution history scope differs')
    require(sha(root / 'calibration/20261028/W1-mb4-a100-hot.json') == read(data / 'gpu/manifest.json')['inputs']['gemm_measurements']['sha256'], 'Calibration differs from original run input')
    frozen = read(data / 'source_freeze.json')['files']
    core_changes = [name for name, value in frozen.items() if name.startswith(('src/', 'configs/', 'docker/astrasim/', 'calibration/')) and (not (root / name).is_file() or sha(root / name) != value)]
    return dict(status='PASS', scope='published small evidence, original manifest/MVP bindings and arithmetic; not raw-log replay or independent re-execution of recorded gates', files=len(wanted), model_seconds=times, conditional_GPU_over_WAFER_time=ratio, max_rank_DRAM_bytes=peaks, paper_main_ready=False, current_vs_frozen_runtime_core_differences=core_changes)


def check_legacy(root=ROOT, require_original_core=False):
    data = root / 'experiments/2026-10-27-v0.6'
    for name, wanted in read(data / 'input_hashes.json').items():
        require(sha(data / 'inputs' / name) == wanted, 'Legacy input differs: ' + name)
    changed = [name for name, wanted in read(data / 'core_runtime_hashes.json').items() if not (root / name).is_file() or sha(root / name) != wanted]
    if require_original_core:
        require(not changed, 'Current code is not the frozen0.6 core')
    comparisons, metrics = read(data / 'comparisons.json'), read(data / 'metrics_summary.json')
    for name, (w, g) in {'primary': ('wafer_original', 'reference'), 'gpu_favoring': ('wafer_original', 'optimistic'), 'wafer_rate_sensitivity': ('wafer_rates', 'reference'), 'joint_sensitivity': ('wafer_rates', 'optimistic')}.items():
        row = comparisons[name]
        ws, gs = metrics[w]['total_simulated_time_ns'] / 1e9, metrics[g]['total_simulated_time_ns'] / 1e9
        require(row['status'] == 'NEAR_BUDGET_SIMULATION_COMPARISON' and not row['blockers'], 'Legacy comparison rejected')
        require(row['wafer_predicted_s'] == ws and row['gpu_predicted_s'] == gs and row['gpu_over_wafer_time_ratio_SCENARIO_ONLY'] == gs / ws, 'Legacy time arithmetic')
        require(row['wafer_faster_in_this_scenario'] == (gs > ws) and row['observed_GPU_step_seconds_used_for_prediction'] is False and row['hardware_calibrated'] is False and row['feasibility_proven'] is False, 'Legacy claim scope')
    accepted, replay = read(data / 'acceptance_summary.json'), read(data / 'local_verification_summary.json')
    require(accepted['status'] == replay['status'] == 'PASS', 'Legacy acceptance')
    require(len(accepted['commands']) == 20 and all(x['exit_status'] == x['expected_exit'] for x in accepted['commands']), 'Legacy command status')
    require(all(x['passed'] for x in accepted['checks']) and sum(accepted['native_fixture_check_counts'].values()) == 21, 'Legacy checks')
    require(sum(x['rank_events'] for x in replay['independent_native_replay'].values()) == 1645238, 'Legacy stored replay count')
    trajectories = read(data / 'inputs/case.json')['workload']['trajectories']
    require(len(trajectories) == 64 and sum(x['response_length'] for x in trajectories) == 256 and len({x['group_id'] for x in trajectories}) == 32, 'Legacy workload')
    require(all(metrics[k]['optimizer_updates'] == 1 for k in ['reference', 'optimistic', 'wafer_rates', 'wafer_original']), 'Legacy updates')
    return dict(status='PASS', comparisons=len(comparisons), current_vs_historical_core_difference_count=len(changed), scope='frozen0.6 summaries and inputs only; not raw-log replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-original-core', action='store_true', help='strict0.6 identity; expected to reject later versions')
    args = parser.parse_args()
    print(json.dumps(dict(status='PASS', current=check_current(), historical=check_legacy(require_original_core=args.require_original_core), raw_log_replay_performed_by_this_check=False), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
