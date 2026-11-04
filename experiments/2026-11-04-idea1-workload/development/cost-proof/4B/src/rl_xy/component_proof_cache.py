"""Reuse an immutable, fully replayed component's serial resource proof.

This caches a proof, never observed clocks. Every invocation still validates
all native macro events, including exact compute ticks, untouched communication
intervals, and the serial/dependency constraints on every local rank. Because
contraction proves a strict rank-local chain, those constraints imply the same
logical alloc/read/READY/free order and the same resource peaks. The first
invocation materializes logical events and independently replays that ledger.

A cache is scoped to one run and keys by object identity, retaining a strong
reference. ``prepare`` and ``finalize`` compare each unique source, contract,
block mapping and native macro signature. The caller must finalize before
claiming success; this is not a cache for concurrently mutated templates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json

from .local_chain_contraction import (
    _deps, _digest, _ops_digest, _participants, _validate_serial,
    expand_native_events,
)


@dataclass(frozen=True)
class _Rule:
    rank: int
    node_id: int
    kind: str
    expected_type: int
    duration_ticks: int | None
    dependencies: tuple[int, ...]


@dataclass
class _Template:
    contraction: object
    signatures: dict
    template_sha256: str
    rules_by_rank: dict[int, tuple[_Rule, ...]]
    expected: frozenset[tuple[int, int]]
    scale: int
    logical_events: int
    replay: dict | None = None
    invocations: int = 0


def _signatures(result):
    """Hash semantic source once per unique template at each run boundary."""
    return {
        'logical_operations_sha256': _ops_digest(result.original_plan.operations),
        'logical_contract_sha256': _digest(result.original_plan.contract),
        'native_operations_sha256': _ops_digest(result.macro_plan.operations),
        'native_contract_sha256': _digest(result.macro_plan.contract),
        'blocks_sha256': _digest([asdict(b) for b in result.blocks]),
        'certificate_sha256': _digest(result.certificate),
    }


def _event_digest(events):
    h = hashlib.sha256()
    for event in sorted(events, key=lambda e: (e.rank, e.node_id)):
        h.update(json.dumps(asdict(event), sort_keys=True, separators=(',', ':')).encode())
    return h.hexdigest()


class ComponentProofCache:
    """One full ledger replay per identity; strict clocks for every invocation."""

    def __init__(self):
        self._templates: dict[int, _Template] = {}
        self._closed = False
        self._final_report = None

    def prepare(self, result):
        """Freeze fingerprints before native execution; return template ID."""
        if self._closed:
            raise ValueError('component proof cache already finalized')
        existing = self._templates.get(id(result))
        if existing is not None:
            if existing.contraction is not result:
                raise ValueError('component template identity collision')
            return existing.template_sha256
        cert = result.certificate
        if cert.get('schema_version') != 'rlxy.local_chain_contraction/1':
            raise ValueError('unsupported component contraction certificate')
        if cert.get('strict_per_rank_serialization_proved') is not True:
            raise ValueError('component proof reuse requires strict per-rank serialization')
        scale = cert.get('astra_time_scale')
        if type(scale) is not int or scale <= 0:
            raise ValueError('component proof requires positive integer time scale')
        signatures = _signatures(result)
        if signatures['logical_operations_sha256'] != cert.get('logical_operations_sha256'):
            raise ValueError('logical operations changed after contraction')
        if signatures['logical_contract_sha256'] != cert.get('logical_resources', {}).get('original_contract_sha256'):
            raise ValueError('logical resource contract changed after contraction')
        if [asdict(b) for b in result.blocks] != cert.get('blocks'):
            raise ValueError('contraction block mapping changed after certification')
        ranks = _validate_serial(result.macro_plan)
        blocks = {b.native_node_id: b for b in result.blocks}
        if len(blocks) != len(result.blocks) or set(blocks) != {o.node_id for o in result.macro_plan.operations}:
            raise ValueError('native operation/block coverage mismatch')
        rules = {rank: [] for rank in ranks}
        for op in result.macro_plan.operations:
            participants = _participants(op, ranks)
            block = blocks[op.node_id]
            if participants != block.participants:
                raise ValueError('native operation/block rank coverage mismatch')
            if op.kind not in ('compute', 'collective', 'p2p'):
                raise ValueError('unsupported native macro kind')
            if op.kind == 'compute' and (type(op.duration_ns) is not int or op.duration_ns <= 0):
                raise ValueError('native compute duration must be a positive integer')
            if len(block.logical_node_ids) > 1:
                if op.kind != 'compute' or op.duration_ns != block.duration_prefix_model_ns[-1]:
                    raise ValueError('native compute duration differs from contracted prefix')
            for rank in participants:
                expected_type = 4 if op.kind == 'compute' else 7
                if op.kind == 'p2p':
                    if rank not in (op.src_rank, op.dst_rank):
                        raise ValueError('native p2p rank outside endpoints')
                    expected_type = 5 if rank == op.src_rank else 6
                rules[rank].append(_Rule(rank, op.node_id, op.kind, expected_type,
                    op.duration_ns*scale if op.kind == 'compute' else None, _deps(op, rank)))
        expected = frozenset((r.rank, r.node_id) for rows in rules.values() for r in rows)
        logical_events = sum(row['logical_resource_nodes_replayed']
            for row in cert['logical_resources']['ranks'].values())
        template = _Template(result, signatures, _digest(signatures),
            {r: tuple(rows) for r, rows in rules.items()}, expected, scale, logical_events)
        self._templates[id(result)] = template
        return template.template_sha256

    @staticmethod
    def _check_clocks(template, events):
        observed = {}
        scale = template.scale
        for event in events:
            if type(event.rank) is not int or type(event.node_id) is not int:
                raise ValueError('native macro rank/node must be integers')
            key = (event.rank, event.node_id)
            if key in observed or key not in template.expected:
                raise ValueError('duplicate/unknown native macro event')
            if any(type(t) is not int or t < 0 for t in (event.issue_tick, event.callback_tick)):
                raise ValueError('native ticks must be nonnegative integers')
            duration = event.callback_tick-event.issue_tick
            if duration < 0:
                raise ValueError('native macro callback precedes issue')
            if type(event.duration_ticks) is not int or event.duration_ticks != duration:
                raise ValueError('native macro duration field differs from observed ticks')
            if (event.issue_ns, event.callback_ns, event.duration_ns) != (
                    event.issue_tick/scale, event.callback_tick/scale, duration/scale):
                raise ValueError('native macro model clocks differ from observed ticks')
            observed[key] = event
        if set(observed) != template.expected:
            raise ValueError('incomplete native macro event coverage')
        for rank, rows in template.rules_by_rank.items():
            last_callback = None
            for rule in rows:
                event = observed[rank, rule.node_id]
                if type(event.node_type) is not int or event.node_type != rule.expected_type:
                    raise ValueError('native macro kind differs from certified operation')
                if rule.duration_ticks is not None and event.duration_ticks != rule.duration_ticks:
                    raise ValueError('native macro duration differs from exact finalized compute sum')
                if last_callback is not None and event.issue_tick < last_callback:
                    raise ValueError('native macro clocks overlap on a rank')
                for dependency in rule.dependencies:
                    if observed[rank, dependency].callback_tick > event.issue_tick:
                        raise ValueError('native macro clocks violate local dependency')
                last_callback = event.callback_tick
        return observed

    def verify(self, result, native_events):
        """Validate an invocation; reuse only a previously fully replayed object."""
        self.prepare(result)
        template = self._templates[id(result)]
        observed = self._check_clocks(template, native_events)
        native_hash = _event_digest(observed.values())
        first = template.replay is None
        if first:
            logical, replay = expand_native_events(result, list(observed.values()))
            if len(logical) != template.logical_events:
                raise ValueError('full component replay logical coverage mismatch')
            template.replay = {**deepcopy(replay), 'expanded_event_stream_sha256': _event_digest(logical)}
            del logical
        template.invocations += 1
        proof = deepcopy(template.replay)
        proof.update({
            'mode': 'full_template_replay' if first else 'reused_verified_serial_template',
            'template_sha256': template.template_sha256,
            'template_invocation': template.invocations,
            'observed_global_macro_events': len(observed),
            'native_macro_event_stream_sha256': native_hash,
            'all_native_macro_clocks_validated': True,
            'network_issue_callback_ticks_preserved': True,
            'exact_compute_durations_validated': True,
            'strict_per_rank_serialization_validated': True,
            'local_dependencies_validated': True,
            'logical_events_represented': template.logical_events,
            'expanded_logical_events': template.logical_events if first else 0,
            'logical_events_materialized': first,
            'logical_events_derived_from_native_macros': first,
            'clock_bound_to_expanded_native_events': first,
            'clock_bound_by_verified_serial_template': not first,
            'logical_resource_nodes_replayed_this_invocation': template.logical_events if first else 0,
            'logical_resource_proof_reused_from_template': None if first else template.template_sha256,
            'ranks_scope': 'this invocation full replay' if first else 'immutable template ledger; invocation clocks verified separately',
            'native_per_object_hook_claimed': False,
        })
        if not first:
            # An absolute logical stream hash from the first invocation must not
            # be mislabeled the hash of later, differently scheduled invocations.
            proof['template_expanded_event_stream_sha256'] = proof.pop('expanded_event_stream_sha256')
        return proof

    def finalize(self):
        """Reject source/certificate mutations before publishing any success."""
        if self._closed:
            return deepcopy(self._final_report)
        for template in self._templates.values():
            if not template.invocations:
                raise ValueError('prepared component template has no verified invocation')
            current = _signatures(template.contraction)
            if current != template.signatures:
                changed = sorted(k for k in current if current[k] != template.signatures[k])
                raise ValueError('component template changed during run: '+', '.join(changed))
        report = {
            'status': 'PASS',
            'mode': 'unique_template_source_and_contract_signatures_rechecked',
            'unique_templates': len(self._templates),
            'full_template_replays': len(self._templates),
            'invocations': sum(t.invocations for t in self._templates.values()),
            'reused_serial_template_invocations': sum(t.invocations-1 for t in self._templates.values()),
            'every_invocation_native_macro_clocks_validated': True,
            'template_signatures': {t.template_sha256: dict(t.signatures) for t in self._templates.values()},
            'scope': 'read-only templates within one run; semantic signatures exclude only derived cost_estimate metadata',
        }
        self._closed = True;self._final_report = report
        return deepcopy(report)
