"""Validated candidate mesh contract and conservative H0 placement screening.

This module is a preflight/configuration adapter, not the native RL simulator.
One decimal TB/s equals 1000 B/ns; D2D is assumed per edge per direction.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_profiles(path=HERE / 'profiles.json'):
    data = json.loads(path.read_text())
    if data.get('schema') != 'wafer.hardware.candidates/1':
        raise ValueError('unsupported hardware profile schema')
    shared = data['shared']
    if shared['train_tp'] != 2:
        raise ValueError('this preflight supports TP2 pairing only')
    profiles = {}
    for c in data['candidates']:
        if c['id'] in profiles:
            raise ValueError('duplicate candidate id')
        if any(type(c[k]) is not int or c[k] <= 0 for k in ('columns', 'rows', 'dies')):
            raise ValueError('invalid dimensions')
        if c['dies'] != c['columns'] * c['rows']:
            raise ValueError('die count must equal full rectangle')
        for k in ('dram_GB_per_die', 'dram_TB_s_per_die', 'd2d_TB_s_per_edge_direction'):
            if not isinstance(c[k], (float, int)) or isinstance(c[k], bool) or not math.isfinite(c[k]) or c[k] <= 0:
                raise ValueError('nonpositive or nonfinite hardware parameter')
        profiles[c['id']] = {**c, **shared}
    if set(profiles) != {'H0', 'H1', 'H2', 'H3'}:
        raise ValueError('expected exactly H0, H1, H2, H3')
    return profiles


def tp2_pairs(c):
    """Pair adjacent horizontal dies, then join the odd-column leftovers vertically."""
    w, h = c['columns'], c['rows']
    pairs = []
    for y in range(h):
        for x in range(0, w - 1, 2):
            pairs.append((y*w+x, y*w+x+1))
    if w % 2:
        for y in range(0, h - 1, 2):
            pairs.append((y*w+w-1, (y+1)*w+w-1))
    used = [die for pair in pairs for die in pair]
    if len(used) != len(set(used)) or len(used) != c['dies'] - c['dies'] % 2:
        raise AssertionError('pair coverage error')
    for a, b in pairs:
        if abs(a//w-b//w) + abs(a%w-b%w) != 1:
            raise AssertionError('nonadjacent TP2')
    return pairs, sorted(set(range(c['dies'])) - set(used))


def validate_existing_native_runner(c):
    """Refuse to send an unsupported profile to the fixed-54 H0 runner."""
    expected={'id':'H0','columns':6,'rows':9,'dies':54,'dram_GB_per_die':64,
              'dram_TB_s_per_die':2,'d2d_TB_s_per_edge_direction':2,
              'module_path_TB_s':2,'d2d_latency_ns':100,'packet_bytes':4096}
    mismatches={k:(c.get(k),value) for k,value in expected.items() if c.get(k)!=value}
    if mismatches:
        raise ValueError('existing native runner only accepts frozen H0; port topology, allocation and cost first: '+str(mismatches))
    return True


def contract(c):
    pairs, idle = tp2_pairs(c)
    w, h = c['columns'], c['rows']
    # These are sums of declared nominal device bandwidths, not achieved throughput.
    return {'id': c['id'], 'case': c['case'], 'columns': w, 'rows': h,
            'physical_die_count': c['dies'], 'tp2_pairs': [list(x) for x in pairs],
            'tp2_worker_count': len(pairs), 'tp2_unpaired_dies': idle,
            'capacity_bytes_per_die': int(c['dram_GB_per_die'] * 1_000_000_000),
            'dram_bytes_per_ns': c['dram_TB_s_per_die'] * 1000,
            'module_path_bytes_per_ns': c['module_path_TB_s'] * 1000,
            'assumed_upper_bound_dram_path_TB_s': min(c['dram_TB_s_per_die'], c['module_path_TB_s']),
            'd2d_bytes_per_ns_per_direction': c['d2d_TB_s_per_edge_direction'] * 1000,
            'd2d_latency_ns': c['d2d_latency_ns'], 'd2d_packet_bytes': c['packet_bytes'],
            'cores_per_die': c['cores_per_die'], 'sram_bytes_per_core': c['sram_bytes_per_core'],
            'nominal_aggregate_dram_TB_s_all_dies': c['dies'] * c['dram_TB_s_per_die'],
            'nominal_aggregate_capacity_GB_all_dies': c['dies'] * c['dram_GB_per_die'],
            'mesh_undirected_edges': h*(w-1)+w*(h-1),
            'shorter_straight_cut_edges': min(w, h),
            'shorter_straight_cut_TB_s_per_direction': min(w, h)*c['d2d_TB_s_per_edge_direction'],
            'native_final_method_supported': all(c.get(k)==v for k,v in
                {'id':'H0','columns':6,'rows':9,'dies':54,'dram_GB_per_die':64,
                 'dram_TB_s_per_die':2,'d2d_TB_s_per_edge_direction':2,
                 'module_path_TB_s':2,'d2d_latency_ns':100,'packet_bytes':4096}.items()),
            'scope': 'static architecture preflight, not full RL simulation or performance prediction'}


def mesh_view_for(c):
    """Translate into the existing controller's immutable MeshView contract.

    This validates the adapter boundary only. It does not change the native
    runner, workload placement, operator-cost provider or native executable.
    """
    source = ROOT/'src'
    if str(source) not in sys.path: sys.path.insert(0, str(source))
    from idea1_controller.contracts import MeshView
    w,h=c['columns'],c['rows'];nodes=tuple(range(c['dies']));coordinates={d:(d//w,d%w) for d in nodes}
    edges=[]
    for die,(row,col) in coordinates.items():
        for dy,dx in ((0,1),(1,0)):
            y,x=row+dy,col+dx
            if y<h and x<w:
                other=y*w+x;edges.extend(((die,other),(other,die)))
    bandwidth=c['d2d_TB_s_per_edge_direction']*1000
    return MeshView(nodes,coordinates,{d:int(c['dram_GB_per_die']*1e9) for d in nodes},
                    {e:bandwidth for e in edges},{e:c['d2d_latency_ns'] for e in edges})


def h0_memory_check(profiles):
    evidence = json.loads((HERE / 'h0-summary.json').read_text())
    result = {}
    for model, row in evidence.items():
        peak = row['h0_observed_training_peak_bytes']
        result[model] = {**row, 'unchanged_peak_vs_capacity': {name: peak <= int(c['dram_GB_per_die'] * 1e9) for name, c in profiles.items()}, 'scope': 'fixed H0 observed training peak only; new placements may change peak'}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'preflight.json');args=ap.parse_args()
    profiles=load_profiles();results={name:contract(c) for name,c in profiles.items()}
    result={'status':'PASS_STATIC_PREFLIGHT','profile_sha256':hashlib.sha256((HERE/'profiles.json').read_bytes()).hexdigest(),
            'assumptions':['decimal GB and TB/s','array is columns x rows, H0 native layout matches 6 columns x 9 rows',
                           'D2D per edge per direction is unverified','other hardware parameters held at H0',
                           'one H1 die unpaired under all-TP2 policy, not necessarily physically unusable'],
            'candidates':results,'h0_training_peak_screen':h0_memory_check(profiles)}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'output':str(args.output)}))


if __name__=='__main__': main()
