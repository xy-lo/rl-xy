"""Strict parsing of AstraSim completion, statistics, and node trace logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ..types import AstraEvent, AstraRankSummary, AstraRunResult, Operation


class AstraParseError(RuntimeError):
    """The simulator output is incomplete, duplicated, or internally inconsistent."""


FINISHED_RE = re.compile(
    r"sys\[(?P<rank>\d+)\] finished, (?P<cycles>\d+) cycles, "
    r"exposed communication (?P<exposed>\d+) cycles\."
)
STAT_RE = re.compile(
    r"sys\[(?P<rank>\d+)\], (?P<kind>Wall|GPU|Comm) time: (?P<cycles>\d+)"
)
EVENT_RE = re.compile(
    r"(?P<action>issue|callback),sys->id=(?P<rank>\d+), tick=(?P<tick>\d+), "
    r"node->id=(?P<node>\d+), node->name=(?P<name>.*?), node->type=(?P<type>\d+)"
)
ERROR_LEVEL_RE = re.compile(r"\] \[(?:error|critical|e|c|E|C)\](?: |$)")


def read_debug_segments(path: Path) -> str:
    """Read spdlog rotation oldest first; coverage remains strictly required.

    log.2.log precedes log.1.log precedes log.log. Never accept a successful
    process with lost/overwritten early events. All segments enter manifest.
    """
    pattern=re.compile(re.escape(path.stem)+r"\.([1-9][0-9]*)"+re.escape(path.suffix)+r"$")
    segments={}
    for candidate in path.parent.iterdir():
        match=pattern.fullmatch(candidate.name)
        if match:segments[int(match[1])]=candidate
    if segments and set(segments)!=set(range(1,max(segments)+1)):
        raise AstraParseError('noncontiguous rotated debug log segments')
    return '\n'.join(p.read_text(encoding='utf-8') for p in
                     [segments[n] for n in sorted(segments,reverse=True)]+[path])


def _one_per_rank(matches: Iterable[re.Match[str]], label: str) -> dict[int, re.Match[str]]:
    result: dict[int, re.Match[str]] = {}
    for match in matches:
        rank = int(match.group("rank"))
        if rank in result:
            raise AstraParseError(f"duplicate {label} record for rank {rank}")
        result[rank] = match
    return result


def parse_astra_output(
    result: AstraRunResult,
    operations: Iterable[Operation],
    *,
    ranks: int,
    astra_time_scale: int,
    inactive_ranks: Iterable[int] = (),
) -> tuple[list[AstraEvent], list[AstraRankSummary]]:
    """Return paired events and rank summaries, rejecting partial success."""

    if result.return_code != 0:
        raise AstraParseError(f"cannot parse failed AstraSim run rc={result.return_code}")
    if astra_time_scale <= 0:
        raise AstraParseError("astra_time_scale must be positive")
    stdout = result.stdout_path.read_text(encoding="utf-8")
    stderr = result.stderr_path.read_text(encoding="utf-8")
    debug = read_debug_segments(result.debug_log_path)
    if stderr.strip():
        # Warnings belong in the debug/info streams in the pinned build.  Any
        # stderr is therefore evidence that the run is not the frozen path.
        raise AstraParseError(f"AstraSim wrote to stderr: {stderr.strip()[:300]}")
    if ERROR_LEVEL_RE.search(debug) or ERROR_LEVEL_RE.search(stdout):
        raise AstraParseError("AstraSim log contains error/critical records")

    finished = _one_per_rank(FINISHED_RE.finditer(stdout), "completion")
    expected_ranks = set(range(ranks))
    if set(finished) != expected_ranks:
        raise AstraParseError(
            f"completion ranks mismatch: expected {sorted(expected_ranks)}, "
            f"got {sorted(finished)}"
        )
    stats: dict[tuple[int, str], int] = {}
    for match in STAT_RE.finditer(stdout):
        key = (int(match.group("rank")), match.group("kind"))
        if key in stats:
            raise AstraParseError(f"duplicate statistic {key}")
        stats[key] = int(match.group("cycles"))
    ops = list(operations)
    inactive=set(inactive_ranks)
    if not inactive <= expected_ranks:
        raise AstraParseError('inactive rank outside world')
    for rank in inactive:
        local=[o for o in ops if rank in (o.participant_ranks or tuple(range(ranks)))]
        if not local or any(o.kind!='compute' or not o.metadata.get('idle_rank_marker') for o in local):
            raise AstraParseError(f'inactive rank {rank} has non-marker work')
        # The pinned binary omits Comm statistics when no communication was
        # submitted on an explicit standby rank. Active-rank coverage is strict.
        stats.setdefault((rank,'Comm'),0)
        if stats[rank,'Comm']!=0:
            raise AstraParseError(f'inactive rank {rank} has communication time')
    required_stats = {(rank, kind) for rank in expected_ranks for kind in ("Wall", "GPU", "Comm")}
    missing_stats = sorted(required_stats - set(stats))
    if missing_stats:
        raise AstraParseError(f"missing AstraSim statistics: {missing_stats}")

    op_by_id = {op.node_id: op for op in ops}
    if len(op_by_id) != len(ops):
        raise AstraParseError("operation node IDs are not unique")
    raw_events: dict[tuple[int, int, str], tuple[int, str, int]] = {}
    for match in EVENT_RE.finditer(debug):
        key = (
            int(match.group("rank")),
            int(match.group("node")),
            match.group("action"),
        )
        if key in raw_events:
            raise AstraParseError(f"duplicate Astra event {key}")
        raw_events[key] = (
            int(match.group("tick")),
            match.group("name"),
            int(match.group("type")),
        )

    expected_keys = {
        (rank, op.node_id, action)
        for op in ops
        for rank in (op.participant_ranks or tuple(range(ranks)))
        for action in ("issue", "callback")
    }
    missing_events = sorted(expected_keys - set(raw_events))
    extra_events = sorted(set(raw_events) - expected_keys)
    if missing_events or extra_events:
        raise AstraParseError(
            f"node event coverage mismatch: missing={missing_events[:12]}, "
            f"extra={extra_events[:12]}"
        )

    events: list[AstraEvent] = []
    for rank in sorted(expected_ranks):
        for op in ops:
            if rank not in (op.participant_ranks or tuple(range(ranks))):
                continue
            issue_tick, issue_name, issue_type = raw_events[(rank, op.node_id, "issue")]
            callback_tick, callback_name, callback_type = raw_events[
                (rank, op.node_id, "callback")
            ]
            if issue_name != op.op_id or callback_name != op.op_id:
                raise AstraParseError(
                    f"node {op.node_id}/rank {rank} name mismatch: "
                    f"{issue_name!r}, {callback_name!r}, expected {op.op_id!r}"
                )
            if issue_type != callback_type:
                raise AstraParseError(
                    f"node {op.node_id}/rank {rank} changed type across callback"
                )
            expected_type = 4 if op.kind == "compute" else 7
            if op.kind == "p2p":
                expected_type = 5 if rank == op.src_rank else 6
            if issue_type != expected_type:
                raise AstraParseError(
                    f"node {op.node_id}/rank {rank} has type {issue_type}, "
                    f"expected {expected_type}"
                )
            if callback_tick < issue_tick:
                raise AstraParseError(
                    f"node {op.node_id}/rank {rank} callback precedes issue"
                )
            duration_ticks = callback_tick - issue_tick
            events.append(
                AstraEvent(
                    rank=rank,
                    node_id=op.node_id,
                    op_id=op.op_id,
                    round_id=op.round_id,
                    phase=op.phase,
                    node_type=issue_type,
                    issue_tick=issue_tick,
                    callback_tick=callback_tick,
                    duration_ticks=duration_ticks,
                    issue_ns=issue_tick / astra_time_scale,
                    callback_ns=callback_tick / astra_time_scale,
                    duration_ns=duration_ticks / astra_time_scale,
                )
            )

    summaries: list[AstraRankSummary] = []
    for rank in sorted(expected_ranks):
        completion = finished[rank]
        cycles = int(completion.group("cycles"))
        wall = stats[(rank, "Wall")]
        if cycles != wall:
            raise AstraParseError(
                f"rank {rank}: completion cycles {cycles} != wall statistic {wall}"
            )
        max_callback = max(event.callback_tick for event in events if event.rank == rank)
        if cycles != max_callback:
            raise AstraParseError(
                f"rank {rank}: wall cycles {cycles} != final callback {max_callback}"
            )
        summaries.append(
            AstraRankSummary(
                rank=rank,
                cycles=cycles,
                exposed_communication_cycles=int(completion.group("exposed")),
                wall_cycles=wall,
                gpu_cycles=stats[(rank, "GPU")],
                communication_cycles=stats[(rank, "Comm")],
            )
        )
    return events, summaries
