"""54-die online generation + early teacher/training on one native clock.

Policy: preserve domain-local attention leases; start teacher/training only
after the entire six-die domain has drained generation. No idle reservation.
Inputs move at safe points, only after recorded EOS and before teacher use.
Model-specific reference and gradient lifetimes are admitted by the driver. Actor v0 remains immutable until global update.
"""
from collections import defaultdict
import hashlib
import json
from rl_xy.qwen3_cost import Cost
from rl_xy.qwen3_sequence import local, compile_blocks
from rl_xy.qwen3_experiment import training_groups, teacher_lengths, chunks
from locality_runtime_v3 import AttentionRuntime


class JointRuntime(AttentionRuntime):
    def __init__(self, *args, early=True, frozen=None, train_mb=1, logprob_mb=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.early = early
        self.groups = training_groups(list(self.items.values()), 2, 54)
        self.targets = {x['trajectory_id']: w for w, xs in enumerate(self.groups) for x in xs}
        self.owners = {x['trajectory_id']: w for w, rep in self.rep.items() for x in rep.requests}
        self.received = [set() for _ in self.rep]
        self.services = {}
        self.moves = []
        self.move_id = 0
        self.stages = {w: 'generation' for w in self.rep}
        self.pending_stage = {}
        self.jobs = {}
        self.trained = set()
        self.teacher_done = defaultdict(list)
        self.train_done = defaultdict(list)
        self.events = []
        self.generation_end_tick = None
        self.tick = 0
        self.frozen = frozen
        self.train_mb, self.logprob_mb = train_mb, logprob_mb
        self.issued = defaultdict(dict)

    def _observe(self, w):
        before = set(self.done)
        super()._observe(w)
        for tid in self.done - before:
            target = self.targets[tid]
            if target == w:
                self.received[w].add(tid)
        if len(self.done) == len(self.ids) and self.generation_end_tick is None:
            self.generation_end_tick = self.tick

    def empty(self, w):
        r = self.rep[w]
        return not r.requests and not r.active and r.pending is None

    def leased(self, w):
        return any(w in (t['source'], t['helper']) for t in self.tx.values())

    def domain_drained(self, w):
        return all(self.empty(j) and not self.leased(j) for j in range(w//3*3, w//3*3+3))

    def sample_candidates(self, src, dst):
        return sorted(tid for tid in self.done if self.owners[tid] == src
                      and self.targets[tid] == dst and tid not in self.received[dst])

    def schedule_transfer(self, w):
        # The current worker is at a native safe point. The peer must already
        # be waiting, so no running generation/helper/teacher is overwritten.
        # Prefer receiving needed inputs, then serving other workers.
        candidates = []
        if self.empty(w):
            for src in self.rep:
                if src != w and self.waiting[src] and src not in self.services and not self.leased(src):
                    tids = self.sample_candidates(src, w)
                    if tids:
                        candidates.append((src, w, tids))
        for dst in self.rep:
            if dst != w and self.waiting[dst] and dst not in self.services and self.empty(dst) and not self.leased(dst):
                tids = self.sample_candidates(w, dst)
                if tids:
                    candidates.append((w, dst, tids))
        if not candidates:
            return None
        src, dst, tids = candidates[0]
        self.move_id += 1
        tx = dict(id=self.move_id, src=src, dst=dst, tids=tids, consumed=set(), start_tick=self.tick)
        tx['bytes_per_lane'] = sum((self.items[t]['prompt_length']+self.items[t]['response_length'])*4 for t in tids)
        self.services[src] = self.services[dst] = tx
        other = dst if src == w else src
        self.waiting[w] = False
        signal = self.wake(other, self.rep[w].ranks[0])
        return {'steps': [signal, self.rpc(w, 'sample_run')]}

    @staticmethod
    def rpc(w, phase, **kw):
        return {'kind': 'controller_rpc', 'event': dict(worker=w, phase=phase, **kw)}

    def sample_run(self, w):
        tx = self.services[w]
        size = tx['bytes_per_lane']; steps = []
        for lane, (src, dst) in enumerate(zip(self.rep[tx['src']].ranks, self.rep[tx['dst']].ranks)):
            steps += [local(self.op.vec(0, size, 0, 'sample_read'), ranks=[src]),
                      {'kind': 'p2p_direct', 'src': src, 'dst': dst, 'bytes': size,
                       'tag': 410000000+tx['id']*2+lane},
                      local(self.op.vec(0, 0, size, 'sample_write'), ranks=[dst])]
        key = f'sample-done-{tx["id"]}'
        steps += [{'kind': 'gate_signal', 'key': key},
                  {'kind': 'gate_wait', 'key': key, 'threshold': 4},
                  self.rpc(w, 'sample_done', transfer=tx['id'])]
        return {'steps': steps}

    def sample_done(self, w, transfer):
        tx = self.services.pop(w)
        if tx['id'] != transfer or w in tx['consumed']:
            raise ValueError('sample service identity mismatch')
        if not tx['consumed']:
            for tid in tx['tids']:
                if tid not in self.done or tid in self.received[tx['dst']]:
                    raise ValueError('sample transfer before EOS or duplicate delivery')
                self.received[tx['dst']].add(tid)
            self.moves.append({k: v for k, v in tx.items() if k != 'consumed'} | {'end_tick': self.tick})
        tx['consumed'].add(w)
        return {'steps': []}

    def training_jobs(self, w):
        items = self.groups[w]; ranks = self.rep[w].ranks
        for stage in ('old_policy', 'reference'):
            for i, batch in enumerate(chunks(items, self.logprob_mb)):
                steps = self.frozen[stage][w][i]['steps'] if self.frozen else compile_blocks(self.op, ranks, self.op.forward(teacher_lengths(batch), 2))
                yield stage, [item['trajectory_id'] for item in batch], steps
        n = len(items)
        yield 'advantage', [], [local(self.op.vec(5*n+4*(n//8), 8*n, 4*n, 'grpo_group_advantage'))]
        # 4B/8B retain the admitted full actor/reference/gradient layout.
        # KV/helper/rollout phase buffers are no longer needed for training.
        zero = self.frozen['training'][w][0]['steps'] if self.frozen else [local(self.op.vec(0, 0, self.op.state(2, 2)['layout'].padded_elements*4, 'gradient_zero'))]
        yield 'gradient_zero', [], zero
        for i, batch in enumerate(chunks(items, self.train_mb)):
            steps = self.frozen['training'][w][i+1]['steps'] if self.frozen else compile_blocks(self.op, ranks, self.op.training(teacher_lengths(batch), 2, sum(item['response_length'] for item in batch)))
            yield 'training', [item['trajectory_id'] for item in batch], steps

    def record_program(self, w, stage, steps):
        if stage == 'gradient_zero':
            stage = 'training'
        if stage not in ('old_policy', 'reference', 'training'):
            return
        if stage not in self.issued[w]:
            self.issued[w][stage] = [hashlib.sha256(b'['), 0]
        h, n = self.issued[w][stage]
        if n:
            h.update(b',')
        h.update(json.dumps(dict(repeat=1, steps=steps), sort_keys=True, separators=(',', ':')).encode())
        self.issued[w][stage][1] += 1

    def program_proof(self):
        out = {}
        for w, phases in self.issued.items():
            out[w] = {}
            for stage, (h, n) in phases.items():
                final = h.copy(); final.update(b']')
                out[w][stage] = dict(sha256=final.hexdigest(), segments=n)
        return out

    def commit_stage(self, w):
        pending = self.pending_stage.pop(w, None)
        if pending is None:
            return
        stage, tids, start = pending
        self.events.append(dict(worker=w, phase=stage, tids=tids, start_tick=start, end_tick=self.tick))
        if stage in ('old_policy', 'reference'):
            self.teacher_done[stage].extend(tids)
        elif stage == 'training':
            self.train_done[w].extend(tids)

    def step(self, w, it):
        self.waiting[w] = False
        self.commit_stage(w)
        if w in self.services:
            return self.sample_run(w)
        self._observe(w)
        if not self.leased(w):
            transfer = self.schedule_transfer(w)
            if transfer is not None:
                return transfer
        if self.domain_drained(w) and (self.early or len(self.done) == len(self.ids)):
            expected = {x['trajectory_id'] for x in self.groups[w]}
            if not expected <= self.received[w]:
                self.stages[w] = 'data_wait'
                return self.wait(w)
            if w not in self.jobs:
                self.jobs[w] = iter(self.training_jobs(w))
            if w not in self.trained:
                try:
                    stage, tids, steps = next(self.jobs[w])
                except StopIteration:
                    self.trained.add(w)
                    self.stages[w] = 'gradient_ready'
                else:
                    self.stages[w] = stage
                    self.pending_stage[w] = (stage, tids, self.tick)
                    self.record_program(w, stage, steps)
                    return {'steps': steps}
            if len(self.trained) == len(self.rep) and not self.services:
                prefix = [self.wake(j, self.rep[w].ranks[0]) for j in self.rep if self.waiting[j]]
                if prefix:
                    return {'steps': prefix+[self.rpc(w, 'joint_finish')]}
                self.returned.add(w)
                return {'steps': [], 'finish': True}
            return self.wait(w)
        self.stages[w] = 'generation'
        result = super().step(w, it)
        # Generation-only finish becomes a safe point for teacher/training.
        if result.get('finish'):
            return self.wait(w)
        for v in result['steps']:
            if v.get('kind') == 'controller_rpc' and v['event']['phase'] == 'finish':
                v['event']['phase'] = 'step'
        return result

    def roles(self):
        roles = []
        for w in self.rep:
            phase = self.stages[w]
            if w in self.services:
                role = 'sample_transfer'
            elif phase in ('training', 'gradient_zero'):
                role = 'training'
            elif phase in ('old_policy', 'reference', 'advantage'):
                role = 'prepare'
            elif phase in ('gradient_ready', 'data_wait') or self.empty(w):
                role = 'idle'
            else:
                role = 'own'
            roles.append(role)
        for src, tx in self.tx.items():
            roles[src] = 'own'; roles[tx['helper']] = 'helper'
        return roles

    def audit(self):
        if self.done != self.ids or self.tx or self.services or self.trained != set(self.rep):
            raise ValueError('unfinished work or lease')
        for stage in ('old_policy', 'reference'):
            xs = self.teacher_done[stage]
            if len(xs) != len(set(xs)) or set(xs) != self.ids:
                raise ValueError('teacher work conservation failure')
        xs = [t for ts in self.train_done.values() for t in ts]
        if len(xs) != len(set(xs)) or set(xs) != self.ids:
            raise ValueError('gradient work conservation failure')
        moved = [t for m in self.moves for t in m['tids']]
        expected = {t for t in self.ids if self.owners[t] != self.targets[t]}
        if len(moved) != len(set(moved)) or set(moved) != expected:
            raise ValueError('sample transfer conservation failure')
