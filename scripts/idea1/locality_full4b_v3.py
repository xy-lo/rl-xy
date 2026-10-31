#!/usr/bin/env python3
"""4B generation-only TP4 reclaim experiment; no training or optimizer.

The full run preserves f2's request[i::12] assignment, TP4 and cap128.
Idle complete TP4 replicas can lend their matching resident FFN shards to
another active replica on its next safe point. This is NOT die-level reclaim.
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

from rl_xy.f2_runtime import INPUTS, configure_native, make_operator, sha, verify_assets
from rl_xy.qwen3_cost import Cost
from rl_xy.qwen3_sequence import attach, local

HERE = Path(__file__).resolve().parent
# Runtime helpers are shipped alongside this driver.
sys.path.insert(0, str(HERE))
from dynamic_native import Plan, execute
from online_rollout import Replica
from helper_program_v10 import compile_helper_batch
from reclaim_priority import score as priority_score
if os.environ.get("IDEA1_POLICY") == "ordinary":
    from attention_coop_runtime import AttentionRuntime
else:
    from locality_runtime_v3 import AttentionRuntime
from attention_tp8_program import attention_weight_bytes_per_helper


class Reclaimer:
    def __init__(self, op, replicas, items, enabled, interval, safe_points, priority):
        self.op, self.rep = op, replicas
        self.items = {x['trajectory_id']: x for x in items}
        self.ids = set(self.items)
        self.enabled, self.interval, self.safe_points = enabled, interval, safe_points
        self.priority = priority
        self.completed_history = []
        self.done, self.returned = set(), set()
        self.cursor = {w: 0 for w in replicas}
        self.waiting = {w: False for w in replicas}
        self.notify = defaultdict(int)
        self.tx, self.next_tx, self.runs = {}, 0, []
        self.deferred = {}

    def wake(self, target, rank):
        self.notify[target] += 1
        self.waiting[target] = False
        return {'kind': 'gate_signal', 'key': f'reclaim-notify-{target}', 'rank': rank}

    def wait(self, w):
        self.waiting[w] = True
        return {'steps': [{'kind': 'gate_wait', 'key': f'reclaim-notify-{w}',
                           'threshold': self.notify[w] + 1},
                          {'kind': 'controller_rpc', 'event': {'worker': w, 'phase': 'resume'}}]}

    def step(self, w, iteration):
        r = self.rep[w]
        r.observe()
        for tid in r.done[self.cursor[w]:]:
            if tid in self.done or tid not in self.ids:
                raise ValueError('duplicate or unknown completion')
            self.done.add(tid)
            record = self.items[tid]
            if record.get('finish_reason') in ('stop', 'length'):
                self.completed_history.append(record['response_length'])
        self.cursor[w] = len(r.done)
        tx = next((t for t in self.tx.values() if w in (t['source'], t['helper'])), None)
        prefix = []
        if tx and tx['source'] == w and tx['started']:
            del self.tx[w]
            prefix.append(local(Cost(compute=1000)))
            if self.waiting[tx['helper']]:
                prefix.append(self.wake(tx['helper'], r.ranks[0]))
            tx = None
        if tx and tx['helper'] == w:
            if not tx['started']:
                tx['started'] = True
                return {'steps': tx['helper_steps'], 'repeat': tx['repeat']}
            return self.wait(w)
        if len(self.done) == len(self.ids) and not self.tx:
            for other in sorted(self.rep):
                if self.waiting[other]:
                    prefix.append(self.wake(other, r.ranks[0]))
            if prefix:
                return {'steps': prefix + [{'kind': 'controller_rpc',
                                            'event': {'worker': w, 'phase': 'finish'}}]}
            self.returned.add(w)
            return {'steps': [], 'finish': True}
        if not r.requests and not r.active and r.pending is None:
            if not self.enabled:
                self.returned.add(w)
                return {'steps': prefix, 'finish': True}
            pending = self.wait(w)
            pending['steps'] = prefix + pending['steps']
            return pending
        busy = {x for t in self.tx.values() for x in (t['source'], t['helper'])}
        free = [j for j in self.rep if j != w and self.waiting[j] and j not in busy
                and not self.rep[j].requests and not self.rep[j].active
                and self.rep[j].pending is None]
        # Stable physical-rank distance; never inspect uncompleted EOS lengths.
        free.sort(key=lambda j: (abs(self.rep[j].ranks[0] - r.ranks[0]), j))
        can_decode = bool(r.active) and (not r.requests or len(r.active) >= r.cap)
        eligible = [(priority_score(v, self.completed_history), -j, j) for j, v in self.rep.items()
                    if j not in busy and v.active and (not v.requests or len(v.active) >= v.cap)
                    and (v.pending is None or v.pending[0] == 'decode')] if self.priority and free else []
        if self.priority and free:
            prefix.append(local(Cost(compute=1000)))  # assumed 1 us, not hardware calibrated
        chosen_priority = max(eligible)[2] if eligible else None
        allow_priority = not self.priority or w == chosen_priority
        if self.enabled and w not in busy and free and can_decode and r.pending is None and allow_priority:
            helper = free[0]
            r.safe_interval = self.interval
            chosen = r.active[:r.decode_batch_limit] if r.decode_batch_limit else r.active
            batch = len(chosen)
            q, extra = divmod(sum(x[1] for x in chosen), batch)
            contexts = tuple(q + (j < extra) for j in range(batch))
            normal = r.next(iteration)
            repeat = normal['repeat']
            self.next_tx += 1
            if self.next_tx * 1024 >= 199000000:
                raise ValueError('helper tag space exhausted')
            compiled = compile_helper_batch(self.op, r.ranks, self.rep[helper].ranks,
                                            contexts, tag_start=1000000+self.next_tx*1024)
            self.tx[w] = {'source': w, 'helper': helper, 'started': False,
                          'repeat': repeat, 'helper_steps': compiled['helper'],
                          'source_steps': compiled['source']}
            self.runs.append({'source': w, 'helper': helper, 'repeat': repeat,
                              'batch': batch, 'contexts': contexts, 'tx': self.next_tx,
                              'priority_score': priority_score(r, self.completed_history) if self.priority else None})
            prefix.extend([local(Cost(compute=1000)), self.wake(helper, r.ranks[0])])
            return {'steps': prefix + [{'kind': 'controller_rpc',
                                        'event': {'worker': w, 'phase': 'helper_source'}}]}
        r.safe_interval = self.interval if (self.enabled or self.safe_points) else None
        result = r.next(iteration)
        if prefix:
            self.deferred[w] = result
            return {'steps': prefix + [{'kind': 'controller_rpc',
                                        'event': {'worker': w, 'phase': 'deferred'}}]}
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=('4B','8B','14B'), required=True)
    p.add_argument('--cap', type=int, required=True)
    p.add_argument('--train-tp', type=int, required=True)
    p.add_argument('--train-mb', type=int, required=True)
    p.add_argument('--window', choices=('tune', 'test1', 'test2', 'test3', 'test4'), default='tune')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--helper', action='store_true')
    p.add_argument('--safe-points', action='store_true')
    p.add_argument('--priority', action='store_true')
    p.add_argument('--small-original', action='store_true')
    p.add_argument('--per-group', type=int, default=128)
    p.add_argument('--interval', type=int, default=64)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.priority and not a.helper:
        raise ValueError('priority policy requires --helper')
    if a.small_original or a.smoke:
        raise ValueError('full original trace only')
    if not 1 <= a.per_group <= 128:
        raise ValueError('invalid original trace per-group limit')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    native = a.native_root.resolve()
    os.environ['ASTRA_SIM_ROOT'] = str(native)
    os.environ['ASTRA_SIM_BIN'] = str(native / 'build-dynarl/bin/AstraSim_Analytical_Congestion_Aware')
    configure_native()
    manifest = verify_assets()
    trace = INPUTS / ('Qwen3-' + a.model) / (a.window + '.json')
    if sha(trace) != manifest['input_hashes'][a.model][a.window + '.json']:
        raise ValueError('trace identity mismatch')
    requests = json.loads(trace.read_text())['trajectories']
    original = requests
    num_workers = 12
    if a.smoke:
        num_workers = 2
        requests = [dict(x, prompt_length=16, response_length=(300 if i % 2 == 0 else 8))
                    for i, x in enumerate(requests[:16])]
    if a.smoke and a.small_original:
        raise ValueError('synthetic smoke and original subset are incompatible')
    if a.small_original:
        num_workers = 4
        requests = [x for i, x in enumerate(original) if i % 12 < num_workers and i//12 < a.per_group]
    op = make_operator(a.model, 'wafer', cache_dir=a.cache_root.resolve())
    # Preserve the f2 f2 admission contract; helper scratch is additive.
    admission = op.admission(requests, a.train_tp, 4, a.train_mb, 16, a.cap)
    scratch = a.cap * (op.model.hidden_size * 32 + op.model.ffn_size * 16) + 1024**2
    peak = admission['phase_peak_bytes']['generation'] + (scratch if a.helper else 0)
    if peak > op.capacity:
        raise ValueError('generation plus helper scratch exceeds capacity')
    (out / 'capacity.json').write_text(json.dumps({'f2_admission': admission,
        'helper_scratch_bound_bytes': scratch if a.helper else 0,
        'generation_upper_bound_bytes': peak, 'capacity_bytes': op.capacity}, indent=2) + '\n')
    replicas = {i: Replica(op, (original[i::12][:a.per_group] if a.small_original else requests[i::num_workers]), tuple(range(i*4, (i+1)*4)),
                           a.cap, safe_interval=None) for i in range(num_workers)}
    plan = Plan(op)
    group_ids={}
    for i,source_rep in replicas.items():
        for j,helper_rep in replicas.items():
            if i==j:continue
            combined=tuple(x for pair in zip(source_rep.ranks,helper_rep.ranks) for x in pair)
            gid=str(len(plan.groups)+1);plan.groups[gid]=list(combined);plan.groupids[combined]=gid
            group_ids[(i,j)]=int(gid)
    if a.helper:
        state=op.state(a.train_tp,4)
        factor=op.model.layers*2*(op.model.kv_heads//8)*op.model.head_dim*op.model.actor_dtype_bytes
        kv_bound=max(sum(sorted((x['prompt_length']+8192 for x in original[i::12]), reverse=True)[:a.cap])*factor for i in range(num_workers))
        weight_bound=attention_weight_bytes_per_helper(op)
        nic_bound=op.hw.nic_tx_bytes_per_die+op.hw.nic_rx_bytes_per_die
        source_bound=admission['phase_peak_bytes']['generation']+scratch+nic_bound
        helper_bound=state['training_base_max']+state['rollout_weights_per_rank']+kv_bound+weight_bound+scratch+nic_bound
        if max(source_bound,helper_bound)>op.capacity:raise ValueError('attention lease capacity rejected')
        (out/'attention-capacity.json').write_text(json.dumps({'source_bound_bytes':source_bound,
            'helper_bound_bytes':helper_bound,'helper_kv_bound_bytes':kv_bound,
            'helper_extra_weight_bytes':weight_bound,'scratch_bytes':scratch,
            'additional_nic_bound_bytes':nic_bound,'capacity_bytes':op.capacity,
            'assumptions':['inactive helper KV released; original actor/reference state remains resident',
              'source retains old KV until source completes; KV transfer streamed in NIC-sized chunks',
              'bound uses prompt+8192 per original source request, not future lengths',
              'DRAM scratch and fragment overhead are formula upper bounds, not measured lifecycle']},indent=2)+'\n')
    ctrl = AttentionRuntime(op, replicas, requests, a.helper, a.interval, a.safe_points, a.priority,group_ids=group_ids)
    seq = attach(plan, [(r.ranks, [{'repeat': 10000000, 'steps': [
        {'kind': 'controller_rpc', 'event': {'worker': i, 'phase': 'arrive'}}]}])
        for i, r in replicas.items()])
    sources = [Path(__file__).resolve(), Path(__import__('online_rollout').__file__).resolve(),
               Path(__import__('dynamic_native').__file__).resolve(), HERE / 'helper_program_v10.py', HERE / 'reclaim_priority.py',
               HERE / 'generation_reclaim_priority_v4.py',HERE / 'locality_runtime_v3.py',HERE / 'attention_coop_runtime.py',HERE / 'attention_tp8_program.py']
    for name in ('rl_xy.f2_runtime', 'rl_xy.qwen3_cost', 'rl_xy.qwen3_sequence',
                 'rl_xy.qwen3_native', 'rl_xy.qwen3_rollout', 'rl_xy.operator_ir'):
        sources.append(Path(__import__(name, fromlist=['x']).__file__).resolve())
    hashes = {str(f): sha(f) for f in sources}
    (out / 'source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    keys, consumers, errors = {}, {}, []
    stop = threading.Event()
    log = (out / 'controller.jsonl').open('w')

    def decide(req):
        e = req['event']; w = e['worker']; phase = e['phase']; it = req['iteration']
        key = (w, phase, it)
        if key in keys:
            return keys[key]
        if phase == 'arrive':
            gate = f'reclaim-worker-{w}-{it}'
            result = {'steps': [{'kind': 'gate_signal', 'key': gate},
                {'kind': 'gate_wait', 'key': gate, 'threshold': 4},
                {'kind': 'controller_rpc', 'event': {'worker': w, 'phase': 'step'}}]}
        elif phase in ('step', 'resume'):
            result = ctrl.step(w, it)
        elif phase == 'helper_source':
            tx = ctrl.tx[w]
            result = {'steps': tx['source_steps'], 'repeat': tx['repeat']}
        elif phase == 'attention_migrate_source':
            result = ctrl.migration_source(w)
        elif phase == 'attention_source_ready':
            result = ctrl.source_ready(w,it)
        elif phase == 'attention_helper_ready':
            result = ctrl.step(w,it)
        elif phase == 'attention_source':
            result = ctrl.attention_source(w)
        elif phase == 'deferred':
            result = ctrl.deferred.pop(w)
        elif phase == 'finish':
            ctrl.returned.add(w)
            result = {'steps': [], 'finish': True}
        else:
            raise ValueError('unknown phase ' + phase)
        keys[key] = result
        log.write(json.dumps({'tick': req['tick'], 'worker': w, 'iteration': it,
                              'phase': phase, 'completed': len(ctrl.done),
                              'helper_runs': len(ctrl.runs)}) + '\n')
        log.flush()
        return result

    tmp = tempfile.TemporaryDirectory(prefix='reclaim-')
    os.environ['RLXY_CONTROLLER_SOCKET'] = str(Path(tmp.name) / 's')
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(os.environ['RLXY_CONTROLLER_SOCKET']); server.listen(54); server.settimeout(.2)
    def serve():
        while not stop.is_set():
            try: conn = server.accept()[0]
            except socket.timeout: continue
            except OSError: break
            with conn:
                try:
                    data = b''
                    while not data.endswith(b'\n'):
                        part = conn.recv(65536)
                        if not part: raise ValueError('short controller request')
                        data += part
                    req = json.loads(data); result = decide(req)
                    conn.sendall((json.dumps(result, separators=(',', ':')) + '\n').encode())
                    e = req['event']; key = (e['worker'], e['phase'], req['iteration'])
                    consumers.setdefault(key, set()).add(req['rank'])
                    if len(consumers[key]) == 4:
                        consumers.pop(key); keys.pop(key, None)
                        if e['phase'] in ('step', 'resume'):
                            replicas[e['worker']].cache.pop(req['iteration'], None)
                except Exception:
                    errors.append(traceback.format_exc())
                    (out / 'controller-error.log').write_text(errors[-1])
                    break
    thread = threading.Thread(target=serve, daemon=True); thread.start()
    start = time.monotonic()
    try: native_result = execute(plan, out / 'native', timeout=7200, sequence=seq)
    finally:
        stop.set(); thread.join(2); server.close(); tmp.cleanup(); log.close()
    if errors: raise RuntimeError(errors[0])
    if ctrl.done != ctrl.ids or ctrl.tx or ctrl.returned != set(replicas):
        raise ValueError('request or helper lifecycle incomplete')
    if any(sha(Path(f)) != h for f, h in hashes.items()):
        raise ValueError('source changed during replay')
    canonical = hashlib.sha256(json.dumps(requests, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = {'status': 'PASS_GENERATION_ONLY', 'model': a.model, 'window': a.window,
              'smoke': a.smoke, 'small_original': a.small_original, 'per_group':a.per_group, 'helper': a.helper, 'priority': a.priority, 'safe_points': a.safe_points, 'interval': a.interval,
              'trace_sha256': sha(trace), 'canonical_requests_sha256': canonical,
              'binary_sha256': native_result['binary_sha256'], 'native_exit_status': native_result['native_exit_status'],
              'requests': len(requests), 'completed': len(ctrl.done), 'tp': 4, 'cap': a.cap,
              'ranks': 4*num_workers, 'generation_s': native_result['model_ns']/1e9,
              'locality_decisions':getattr(ctrl, 'locality_decisions', []), 'helper_runs': ctrl.runs, 'migrations':ctrl.migrations,'releases':ctrl.releases,'wall_s': time.monotonic()-start,
              'limitations': ['TP8 attention / TP4 FFN cost replay; no tensor execution',
                              'whole TP4 helper only; no single-die reclaim',
                              'no training competition or end-to-end time',
                              'no physical wafer or BF16 numerical validation']}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'helper_runs'}, indent=2))


if __name__ == '__main__':
    main()
