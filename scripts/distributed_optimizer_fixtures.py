#!/usr/bin/env python3
"""Tiny bounded native GROUP RS/AG size-convention checks, not a model run.

Four ranks, one-dimensional analytical Ring, 8 model B/ns/link and1ns latency.
Two payload sizes let the serialization slope be checked without attributing
startup latency to bytes. --plan-only performs no native or GPU work.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rl_xy.distributed_optimizer import build_layout
from rl_xy.reporting import deterministic_tree_hash
from rl_xy.types import Operation

ROOT = Path(__file__).resolve().parents[1]


def make_plan(dp_size, elements, collective):
    if dp_size not in (2, 4) or collective not in ('REDUCE_SCATTER', 'ALL_GATHER'):
        raise ValueError('tiny fixture only supports DP2/4 and RS/AG')
    layout = build_layout([SimpleNamespace(parameter_id='tiny_weight',
        local_elements_per_die=elements, replicated=False)], dp_size=dp_size)
    groups = {'1': [0, 1, 2, 3]} if dp_size == 4 else {'1': [0, 2], '2': [1, 3]}
    ops = [Operation(1, 'start', 0, 'fixture', 'start', 'compute', duration_ns=1,
        participant_ranks=(0, 1, 2, 3))]
    last = {r: 1 for r in range(4)}
    for gid, participants in groups.items():
        pair = layout.native_collectives(participants, group_id=int(gid), round_id=0, actor_version=0)
        op = pair[0] if collective == 'REDUCE_SCATTER' else pair[1]
        node = len(ops) + 1
        deps = {r: (last[r],) for r in participants}
        ops.append(replace(op, node_id=node, dependencies_by_rank=deps,
            dependencies=tuple(sorted({x for xs in deps.values() for x in xs}))))
        for r in participants:
            last[r] = node
    deps = {r: (last[r],) for r in range(4)}
    ops.append(Operation(len(ops) + 1, 'finish', 0, 'fixture', 'finish', 'compute', duration_ns=1,
        participant_ranks=(0, 1, 2, 3), dependencies_by_rank=deps,
        dependencies=tuple(sorted(set(last.values())))))
    return layout, groups, ops


def execute(folder, dp_size, elements, collective, native_local_mem_gbps=16):
    from rl_xy.astra_adapter import generate_traces, run_astrasim, parse_astra_output
    layout, groups, ops = make_plan(dp_size, elements, collective)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'operations.json').write_text(json.dumps([asdict(o) for o in ops], indent=2) + '\n')
    (folder / 'groups.json').write_text(json.dumps(groups, indent=2) + '\n')
    generate_traces(ops, folder / 'workload', ranks=4, astra_time_scale=1000, comm_groups=groups)
    system = json.loads((ROOT / 'configs/wafer4_fixture/astra_system.json').read_text())
    for key in ('all-reduce-implementation', 'all-gather-implementation',
                'reduce-scatter-implementation', 'all-to-all-implementation'):
        system[key] = ['ring']
    system['local-mem-bw']=native_local_mem_gbps
    (folder / 'system.json').write_text(json.dumps(system) + '\n')
    bandwidth = 8 / 1000 * 1e9 / 2**30
    (folder / 'network.yml').write_text(f'topology: [Ring]\nnpus_count: [4]\nbandwidth: [{bandwidth}]\nlatency: [1000]\n')
    (folder / 'remote.json').write_text('{"memory-type":"NO_MEMORY_EXPANSION"}\n')
    result = run_astrasim(trace_prefix=folder / 'workload', system_config=folder / 'system.json',
        network_config=folder / 'network.yml', remote_memory_config=folder / 'remote.json', output_dir=folder,
        binary='/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware',
        comm_groups=folder / 'groups.json', timeout_seconds=60)
    events, _ = parse_astra_output(result, ops, ranks=4, astra_time_scale=1000)
    selected = [e for e in events if e.node_id not in (1, len(ops))]
    (folder / 'events.json').write_text(json.dumps([asdict(e) for e in events], indent=2) + '\n')
    return {'collective': collective, 'dp_size': dp_size, 'padded_elements': layout.padded_elements,
        'payload_bytes': layout.padded_elements * 4 if collective == 'REDUCE_SCATTER' else layout.shard_elements * 2,
        'native_local_mem_gbps':native_local_mem_gbps,
        'span_ticks': max(e.callback_tick for e in selected) - min(e.issue_tick for e in selected),
        'native_exit': result.return_code, 'binary_sha256': result.binary_sha256,
        'layout': layout.contract()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--native-local-mem-gbps',type=float,default=16,
        help='native protocol proxy in GB/s;1e12 suppresses its separately floored staging delay, not a hardware bandwidth')
    p.add_argument('--small-elements',type=int,default=1)
    p.add_argument('--large-elements',type=int,default=129)
    a = p.parse_args()
    if not math.isfinite(a.native_local_mem_gbps) or not 0<a.native_local_mem_gbps<=1e12:
        p.error('native local memory must be positive and no more than1e12GB/s')
    if not 0<a.small_elements<a.large_elements<=2**40:
        p.error('require0<small<large<=2**40; message counts remain int64-safe')
    out = a.out.resolve();out.mkdir(parents=True, exist_ok=False)
    results = {};checks = []
    for dp in (2, 4):
        for elements, label in [(a.small_elements, 'small'), (a.large_elements, 'large')]:
            for collective, code in [('REDUCE_SCATTER', 'RS'), ('ALL_GATHER', 'AG')]:
                name = f'dp{dp}_{label}_{code}'
                if a.plan_only:
                    layout, groups, ops = make_plan(dp, elements, collective)
                    results[name] = {'layout': layout.contract(), 'groups': groups,
                        'operations': [asdict(o) for o in ops]}
                else:
                    results[name] = execute(out / name, dp, elements, collective,a.native_local_mem_gbps)
        if not a.plan_only:
            small = results[f'dp{dp}_small_RS']['padded_elements']
            large = results[f'dp{dp}_large_RS']['padded_elements']
            rs = results[f'dp{dp}_large_RS']['span_ticks'] - results[f'dp{dp}_small_RS']['span_ticks']
            ag = results[f'dp{dp}_large_AG']['span_ticks'] - results[f'dp{dp}_small_AG']['span_ticks']
            expected_rs = (large - small) * 4 * (dp - 1) / dp / 8 * 1000
            expected_ag = (large - small) * 2 * (dp - 1) / dp / 8 * 1000
            # PacketBundle.cc: processed RS packets pay three separately
            # floored local memory visits in RAW ns, not model ns.
            local_delta=3*(dp-1)*(int((large*4//dp)/a.native_local_mem_gbps)-int((small*4//dp)/a.native_local_mem_gbps))
            expected_rs_total=expected_rs+local_delta
            checks.extend([
                {'check': f'dp{dp}_full_buffer_RS_serialization_slope', 'passed': abs(rs - expected_rs_total) <= 2 * dp,
                    'observed_delta_ticks': rs, 'expected_wire_delta_ticks': expected_rs,
                    'expected_native_processing_memory_delta_ticks':local_delta,
                    'expected_total_delta_ticks':expected_rs_total},
                {'check': f'dp{dp}_full_buffer_AG_serialization_slope', 'passed': abs(ag - expected_ag) <= 2 * dp,
                    'observed_delta_ticks': ag, 'expected_delta_ticks': expected_ag},
                {'check': f'dp{dp}_FP32_RS_vs_BF16_AG_slope_ratio', 'passed': ag > 0 and abs(rs-local_delta - 2 * ag) <= 2 * dp}])
    report = {'status': 'PLAN_ONLY' if a.plan_only else ('PASS' if all(c['passed'] for c in checks) else 'FAIL'),
        'checks': checks, 'results': results, 'source_tree_sha256': deterministic_tree_hash(ROOT),
        'scope': 'native message size/PG serialization convention, not NCCL values or full DO training validation',
        'numeric_reduction_values_not_simulated_by_Astra': True,
        'expected_ring_formula': '(padded_bytes_large-padded_bytes_small)*(DP-1)/DP/(8 B/model_ns); latency cancels',
        'AG_native_payload_convention':'per-rank input shard, unlike RS full input',
        'native_local_mem_gbps':a.native_local_mem_gbps,
        'requested_parameter_elements':{'small':a.small_elements,'large':a.large_elements},
        'native_processing_formula_raw_ticks':'3*floor(packet_bytes/native_local_mem_gbps), one write and two reads; RS/AR-reduction processed packets only',
        'native_local_mem_effective_model_bytes_per_ns':a.native_local_mem_gbps*1000,
        'native_local_mem_parameter_is_not_target_hardware_SRAM_NoC_or_HBM_bandwidth':True}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'])
    return 2 if report['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
