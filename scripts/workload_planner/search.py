"""Input-isolated 4B TP/DP template search. No bank or native outcome reads."""
import time
import os
PROCESS_START_NS = time.perf_counter_ns()
import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT
COST_ROOT = Path(os.environ.get('RLXY_COST_ROOT', str(ROOT))).resolve()
sys.path[:0] = [str(COST_ROOT / 'src'),
               str(FREEZE / 'scripts/idea1_layered/legacy')]
sys.path.append(str(FREEZE / 'src'))
from rl_xy.f2_runtime import make_operator, sha, INPUTS
from rl_xy.qwen3_experiment import Policy, simple_plan_time, optimizer_plan
from rl_xy.qwen3_native import NetworkProfiles, Plan
from gqa_cost import apply_generation_gqa
from idea1_controller.inference_catalog import InferenceServiceCatalog
from idea1_controller.cost_catalog import validate_catalog
from idea1_controller.history import validate_history, predict_groups, digest
from idea1_controller.inference_search import balanced_execution
from idea1_controller.inference_groups import validate_partition, InferenceAdmission
from idea1_controller.quotas import layout_quotas
from idea1_controller.lifecycle_memory import SystemMemory
from idea1_controller.search import risk
from fast_replica import HeapScenarioScorer
from placement import HistoryPlacer


def templates():
    layouts = {}
    def add(name, groups):
        normalized = sorted(sorted(g) for g in groups)
        layout = {g[0]: g for g in normalized}
        validate_partition(layout)
        if layout not in layouts.values():
            layouts[name] = layout
    def stripes(lines, size):
        groups = []
        for line in lines:
            while line:
                n = max(x for x in (1, 2, 4) if x <= min(size, len(line)))
                groups.append(line[:n])
                line = line[n:]
        return groups
    for n in (1, 2, 4):
        for axis, lines in [('row', [list(range(i, i+3)) for i in range(0, 27, 3)]),
                            ('col', [list(range(i, 27, 3)) for i in range(3)])]:
            add(f'{axis}{n}', stripes(lines, n))
    boxes = [[r*3, r*3+1, r*3+3, r*3+4] for r in (0, 2, 4, 6)]
    add('box4', boxes + stripes([list(range(2, 27, 3)), [24, 25]], 4))
    return layouts


class Environment:
    def __init__(self, public_path, cache_dir, *, model='4B', injected_forecasts=None, placement_mode='independent'):
        start = time.perf_counter_ns()
        if placement_mode not in ('independent','release'):raise ValueError('invalid placement mode')
        self.placement_mode=placement_mode
        self.public_path = Path(public_path).resolve()
        self.pub = json.loads(self.public_path.read_text())
        allowed = {'group_id', 'prompt_length', 'prompt_sha256', 'library'}
        assert self.pub and all(set(g) == allowed for g in self.pub)
        assert len({g['group_id'] for g in self.pub}) == len(self.pub)
        self.public = {x['group_id']: x for x in self.pub}
        hp = ROOT / f'experiments/2026-11-03-idea1-elastic/assets/history-{model}.json.gz'
        cp = ROOT / f'experiments/2026-11-03-idea1-elastic/assets/inference-catalog-{model}.json'
        import gzip
        self.history = json.loads(gzip.decompress(hp.read_bytes())) if injected_forecasts is None else None
        self.injected_forecasts = injected_forecasts
        self.catalog_data = json.loads(cp.read_text())
        self.catalog = InferenceServiceCatalog(self.catalog_data)
        self.op = make_operator(model, 'wafer', cache_dir=Path(cache_dir))
        apply_generation_gqa(self.op)
        assert self.op.active_dies == 54
        cfg = sha(INPUTS / f'Qwen3-{model}/config.json')
        validate_catalog(self.catalog_data, self.op, model, cfg)
        if self.history is not None:
            validate_history(self.pub, self.history, model, cfg)
        self.policy = {'4B':Policy(2,2,4,8,64),'8B':Policy(2,2,2,1,16),'14B':Policy(2,2,1,1,16)}[model]
        net = NetworkProfiles(self.op, 'unused', native=False)
        self.suffix = simple_plan_time(optimizer_plan(self.op, self.policy), net) + 1
        self.prefix = 1000 + net.time(54, 'ALL_REDUCE', 54*4)
        extra = Plan(self.op)
        half = SystemMemory(self.op).actor
        extra.local((0, 1), self.op.vec(0, half, 0, 'TP1_missing_actor_half_read'), 'read')
        extra.comm((0, 1), 'ALL_GATHER', half, 'comm')
        extra.local((0, 1), self.op.vec(0, 0, half, 'TP1_missing_actor_half_write'), 'write')
        self.tp1_extra = simple_plan_time(extra, net)
        self.input_hashes = {str(p): sha(p) for p in (hp, cp, self.public_path)}
        self.setup_s = (time.perf_counter_ns() - start) / 1e9

    def search(self, mode='shortlist6'):
        if mode not in ('full', 'shortlist12', 'shortlist6', 'one_scene', 'capacity_fallback'):
            raise ValueError('unknown search mode')
        if self.placement_mode=='release':
            from release_search import search
            return search(self,mode)
        start = time.perf_counter_ns()
        clock = lambda: (time.perf_counter_ns() - start) / 1e9
        forecast = self.injected_forecasts if self.injected_forecasts is not None else {g: asdict(x) for g, x in predict_groups(self.pub, self.history).items()}
        if set(forecast) != set(self.public) or any(len(f['response_scenarios']) != 16 or any(len(v) != 8 for v in f['response_scenarios']) for f in forecast.values()):
            raise ValueError('current executor supports K8 and 16 scenarios; complete batch required')
        placer = HistoryPlacer(self.pub, forecast, self.catalog)
        layouts = templates()
        work = {g: statistics.mean(sum(v) for v in f['response_scenarios'])
                for g, f in forecast.items()}
        homes, reserves = {}, {}
        for name, layout in layouts.items():
            quota = layout_quotas(len(self.pub), layout)
            queues = {str(w): [] for w in quota}
            loads = {w: 0.0 for w in quota}
            for gid in sorted(self.public, key=lambda g: -forecast[g]['score']):
                w = min((w for w in quota if len(queues[str(w)]) < quota[w]),
                        key=lambda w: (loads[w], w))
                queues[str(w)].append(gid)
                loads[w] += work[gid]
            homes[name] = dict(region_members=layout, queues=queues)
            maximum = max(sum(len(queues[str(w)])*8 for w in ws) for ws in layout.values())
            reserves[name] = 2*(len(self.pub)*8 + max(64, maximum))*(max(self.op.prompt_cap,
                              max(g['prompt_length'] for g in self.pub)) + 8192)*4
        variants, candidates, certs, admissions = {}, [], {}, {}
        for inf, layout in [('paired1', layouts['row1']), *layouts.items()]:
            paired = inf == 'paired1'
            for fraction in (0.5, 0.75, 1.0):
                tp = {r: 1 if paired else len(ws)*2 for r, ws in layout.items()}
                caps = {str(r): max(1, int(fraction*self.catalog_data[
                    'inference_capacity_frontier'][str(tp[r])])) for r in layout}
                profile = dict(groups=layout, caps=caps, paired_tp1=paired)
                queues, quota = placer.assign(profile)
                key = f'{inf}@{fraction}'
                variants[key] = dict(**profile, execution_queues=queues, quota=quota)
                owner = {w: r for r, ws in layout.items() for w in ws}
                origin = {g: int(r) for r, gs in queues.items() for g in gs}
                for train_name, home in homes.items():
                    for r in layout:
                        k = (reserves[train_name], tp[r], caps[str(r)])
                        if k not in certs:
                            if k[0] not in admissions:
                                admissions[k[0]] = InferenceAdmission(self.op, k[0])
                            certs[k] = admissions[k[0]].evaluate(k[1], k[2], self.pub)
                    passed = all(certs[(reserves[train_name], tp[r], caps[str(r)])]['passed']
                                 for r in layout)
                    deps = {root: sorted({owner[w] for w in ws} |
                            {origin[g] for w in ws for g in home['queues'][str(w)]})
                            for root, ws in home['region_members'].items()}
                    candidates.append(dict(inf=key, training=train_name, dp=len(home['region_members']),
                        signature=digest([variants[key], home]), generated_s=clock(), passed=passed,
                        dependencies=deps))
        generation_s = clock()
        scorer = HeapScenarioScorer(self.op, self.policy, self.catalog)
        ends_cache, service_cache, memories = {}, {}, {}
        events = []
        def evaluate(candidate, scenes, stage):
            before = clock()
            inf = variants[candidate['inf']]
            times = []
            for scene in scenes:
                end_key = (candidate['inf'], scene)
                if end_key not in ends_cache:
                    end = {}
                    for r, ws in inf['groups'].items():
                        gids = inf['execution_queues'][str(r)]
                        tp = 1 if inf['paired_tp1'] else len(ws)*2
                        parts = [gids[::2], gids[1::2]] if inf['paired_tp1'] else [gids]
                        end[r] = max(scorer.replica(part, self.public, forecast, scene,
                                     tp, inf['caps'][str(r)]) for part in parts)
                    ends_cache[end_key] = end
                sk = (candidate['training'], scene)
                if sk not in service_cache:
                    if candidate['training'] not in memories:
                        memories[candidate['training']] = SystemMemory(self.op, reserves[candidate['training']])
                    scorer.memory = memories[candidate['training']]
                    service_cache[sk] = scorer.training_services(homes[candidate['training']],
                                                        self.public, forecast, scene)
                end, services = ends_cache[end_key], service_cache[sk]
                finish = max(max(end[r] for r in deps)+services[root]
                             for root, deps in candidate['dependencies'].items())
                times.append(finish+self.prefix+self.suffix+(self.tp1_extra if inf['paired_tp1'] else 0))
            result = {k: v for k, v in candidate.items() if k != 'dependencies'}
            result.update(stage=stage, score_start_s=before, scored_s=clock(),
                          risk_s=risk(times)/1e9, scenarios_s=[v/1e9 for v in times])
            events.append(result)
            return result
        legal = [c for c in candidates if c['passed']]
        assert legal
        coarse_end = generation_s
        if mode == 'capacity_fallback':
            def fallback_key(c):
                inf = variants[c['inf']]
                max_tp = 1 if inf['paired_tp1'] else max(2*len(ws) for ws in inf['groups'].values())
                return (max_tp, -c['dp'], c['signature'])
            chosen = min(legal, key=fallback_key)
            fine = [evaluate(chosen, [0], 'capacity-only-fallback-diagnostic')]
        elif mode == 'full':
            fine = [evaluate(c, range(16), 'full') for c in legal]
        else:
            coarse = [evaluate(c, [0], 'coarse') for c in legal]
            coarse.sort(key=lambda c: (c['risk_s'], c['signature']))
            coarse_end = clock()
            if mode == 'one_scene':
                fine = coarse
            else:
                count = 12 if mode == 'shortlist12' else 6
                representatives = {}
                for c in coarse:
                    representatives.setdefault(c['inf'].split('@')[0], c)
                promoted = list(representatives.values())
                for c in coarse:
                    if len(promoted) >= count:
                        break
                    if c not in promoted:
                        promoted.append(c)
                ids = {c['signature'] for c in promoted}
                fine = [evaluate(c, range(16), 'fine') for c in legal if c['signature'] in ids]
        fine_end = clock()
        winner = min(fine, key=lambda c: (c['risk_s'], c['signature']))
        inference = variants[winner['inf']]
        home = homes[winner['training']]
        validate_partition(inference['groups'])
        validate_partition(home['region_members'])
        assert sorted(g for gs in home['queues'].values() for g in gs) == sorted(self.public)
        assert sorted(g for gs in inference['execution_queues'].values() for g in gs) == sorted(self.public)
        elapsed = clock()
        return dict(mode=mode, winner=winner, inference=inference, training=home,
            timings=dict(generate_admit_s=generation_s, coarse_s=coarse_end-generation_s,
                fine_s=fine_end-coarse_end, finalize_s=elapsed-fine_end, search_s=elapsed),
            counts=dict(inference_variants=len(variants), training_covers=len(homes),
                candidate_pairs=len(candidates), legal_pairs=len(legal), fine_pairs=len(fine),
                scenario_replays=sum(len(e['scenarios_s']) for e in events),
                unique_inference_scenarios=len(ends_cache), fresh_replica_cache_entries=len(scorer.replica_cache)),
            events=events, input_hashes=self.input_hashes,
            score_scope='static historical scenario surrogate, no native queue/helper replay or hardware guarantee')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, default=Path(__file__).with_name('public.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('full', 'shortlist12', 'shortlist6', 'one_scene', 'capacity_fallback'), default='shortlist6')
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = Environment(a.public, a.output/'model-cache')
    setup_s = (time.perf_counter_ns()-PROCESS_START_NS)/1e9
    result = env.search(a.mode)
    result['timings']['setup_including_import_s'] = setup_s
    result['timings']['process_to_decision_s'] = (time.perf_counter_ns()-PROCESS_START_NS)/1e9
    before = time.perf_counter_ns()
    (a.output/'result.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    serialization_s = (time.perf_counter_ns()-before)/1e9
    (a.output/'timing.json').write_text(json.dumps(dict(**result['timings'], serialization_s=serialization_s), indent=2)+'\n')
    print(json.dumps(dict(mode=a.mode, winner=result['winner']['inf'], training_dp=result['winner']['dp'],
                         proxy_score_s=result['winner']['risk_s'], timings=result['timings'], counts=result['counts'])))


if __name__ == '__main__':
    main()
