"""Generate one Chakra ET file per simulated rank.

The official Chakra schema stores compute duration in microseconds while the
AstraSim analytical backend advances in nanoseconds.  The toy configuration
therefore encodes one model nanosecond as one Chakra microsecond and scales the
network by the same ``astra_time_scale=1000`` factor.  Parsed ticks are divided
by that factor before they enter any rl-xy metric.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import BinaryIO, Iterable

from ..types import Operation


class TraceGenerationError(RuntimeError):
    """Raised when the pinned Chakra schema is unavailable or a trace is invalid."""


def _encode_varint(stream: BinaryIO, value: int) -> None:
    if value < 0:
        raise TraceGenerationError("protobuf message length cannot be negative")
    while value > 0x7F:
        stream.write(bytes(((value & 0x7F) | 0x80,)))
        value >>= 7
    stream.write(bytes((value,)))


def _write_delimited(stream: BinaryIO, message: object) -> None:
    payload = message.SerializeToString()  # type: ignore[attr-defined]
    _encode_varint(stream, len(payload))
    stream.write(payload)


def _schema() -> object:
    try:
        return importlib.import_module(
            "extern.graph_frontend.chakra.schema.protobuf.et_def_pb2"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise TraceGenerationError(
            "official Chakra et_def_pb2 is unavailable; run inside the rl-xy "
            "Docker image built from the pinned AstraSim commit"
        ) from exc


def participant_ranks(op: Operation, ranks: int) -> tuple[int, ...]:
    """Resolve explicit participation, retaining compatibility with old fixtures."""

    return op.participant_ranks or tuple(range(ranks))


def dependencies_for_rank(op: Operation, rank: int) -> tuple[int, ...]:
    return op.dependencies_by_rank.get(rank, op.dependencies)


def _validate_operations(operations: list[Operation], ranks: int, comm_groups=None) -> None:
    if ranks < 2:
        raise TraceGenerationError("the MVP requires at least two ranks")
    if not operations:
        raise TraceGenerationError("cannot generate an empty workload")
    ids = [op.node_id for op in operations]
    if len(ids) != len(set(ids)):
        raise TraceGenerationError("operation node IDs must be unique")
    seen: set[int] = set()
    seen_by_rank: dict[int, set[int]] = {rank: set() for rank in range(ranks)}
    for op in operations:
        group=op.metadata.get('comm_group')
        if group is not None:
            if (op.kind!='collective' or str(group)=='0' or not comm_groups or
                str(group) not in comm_groups or tuple(comm_groups[str(group)])!=tuple(op.participant_ranks)):
                raise TraceGenerationError('collective communicator group membership mismatch')
        if op.node_id <= 0:
            raise TraceGenerationError(f"{op.op_id}: node ID must be positive")
        if "," in op.op_id:
            raise TraceGenerationError(f"{op.op_id}: node name must not contain commas")
        if op.node_id >= 500_000_000:
            raise TraceGenerationError(
                f"{op.op_id}: node ID cannot be used as an unmodified AstraSim p2p tag"
            )
        participants = participant_ranks(op, ranks)
        if len(participants) != len(set(participants)) or any(
            rank < 0 or rank >= ranks for rank in participants
        ):
            raise TraceGenerationError(f"{op.op_id}: invalid participant ranks")
        if not participants:
            raise TraceGenerationError(f"{op.op_id}: empty participant ranks")
        if set(op.dependencies_by_rank) - set(participants):
            raise TraceGenerationError(
                f"{op.op_id}: dependency map contains a non-participant rank"
            )
        for rank in participants:
            dependencies = dependencies_for_rank(op, rank)
            missing = set(dependencies) - seen_by_rank[rank]
            if missing:
                raise TraceGenerationError(
                    f"{op.op_id}/rank{rank}: dependencies are absent or do not "
                    f"precede the node: {sorted(missing)}"
                )
        if op.kind == "compute" and op.duration_ns <= 0:
            raise TraceGenerationError(f"{op.op_id}: invalid compute duration")
        if op.kind != "compute" and op.payload_bytes <= 0:
            raise TraceGenerationError(f"{op.op_id}: invalid communication payload")
        if op.kind not in {"compute", "collective", "p2p"}:
            raise TraceGenerationError(f"{op.op_id}: unknown operation kind {op.kind!r}")
        if op.kind == "p2p":
            if op.src_rank is None or op.dst_rank is None:
                raise TraceGenerationError(f"{op.op_id}: missing p2p endpoints")
            if not (0 <= op.src_rank < ranks and 0 <= op.dst_rank < ranks):
                raise TraceGenerationError(f"{op.op_id}: p2p endpoint outside rank set")
            if set(participants) != {op.src_rank, op.dst_rank}:
                raise TraceGenerationError(
                    f"{op.op_id}: p2p participants must equal its endpoints"
                )
        if op.kind == "collective" and set(participants) != set(range(ranks)) and not op.metadata.get('comm_group'):
            raise TraceGenerationError(
                f"{op.op_id}: MVP collectives must include every TP rank"
            )
        if op.kind == "collective" and op.collective not in {
            "ALL_REDUCE",
            "ALL_GATHER",
            "REDUCE_SCATTER",
            "ALL_TO_ALL",
        }:
            raise TraceGenerationError(
                f"{op.op_id}: unsupported collective {op.collective!r}"
            )
        seen.add(op.node_id)
        for rank in participants:
            seen_by_rank[rank].add(op.node_id)


def _node_for_rank(schema: object, op: Operation, rank: int) -> object:
    node = schema.Node()  # type: ignore[attr-defined]
    node.id = op.node_id
    node.name = op.op_id
    node.ctrl_deps.extend(dependencies_for_rank(op, rank))
    attr = schema.AttributeProto  # type: ignore[attr-defined]

    if op.kind == "compute":
        node.type = schema.COMP_NODE  # type: ignore[attr-defined]
        node.duration_micros = op.duration_ns
        node.attr.append(attr(name="is_cpu_op", bool_val=op.metadata.get("execution_domain")=="host"))
        if op.flops or op.vector_ops:
            node.attr.append(
                attr(name="num_ops", uint64_val=max(1, op.flops + op.vector_ops))
            )
        node.attr.append(attr(name="tensor_size", uint64_val=max(1, op.tensor_bytes)))
        return node

    if op.kind == "collective":
        comm_types = {
            "ALL_REDUCE": schema.ALL_REDUCE,  # type: ignore[attr-defined]
            "ALL_GATHER": schema.ALL_GATHER,  # type: ignore[attr-defined]
            "REDUCE_SCATTER": schema.REDUCE_SCATTER,  # type: ignore[attr-defined]
            "ALL_TO_ALL": schema.ALL_TO_ALL,  # type: ignore[attr-defined]
        }
        try:
            comm_type = comm_types[op.collective]
        except KeyError as exc:
            raise TraceGenerationError(
                f"{op.op_id}: unsupported collective {op.collective!r}"
            ) from exc
        node.type = schema.COMM_COLL_NODE  # type: ignore[attr-defined]
        node.attr.extend(
            [
                attr(name="is_cpu_op", bool_val=False),
                attr(name="comm_type", int64_val=comm_type),
                attr(name="comm_size", int64_val=op.payload_bytes),
                attr(name="involved_dim", bool_list=schema.BoolList(values=[True])),  # type: ignore[attr-defined]
            ]
        )
        if op.metadata.get('comm_group'):
            node.attr.append(attr(name='pg_name',string_val=str(op.metadata['comm_group'])))
        if op.metadata.get('transport_payload_bytes'):
            for entry in node.attr:
                if entry.name=='comm_size':entry.int64_val=op.metadata['transport_payload_bytes']
        return node

    if op.kind != "p2p":
        raise TraceGenerationError(f"{op.op_id}: unknown operation kind {op.kind!r}")
    if rank == op.src_rank:
        node.type = schema.COMM_SEND_NODE  # type: ignore[attr-defined]
    elif rank == op.dst_rank:
        node.type = schema.COMM_RECV_NODE  # type: ignore[attr-defined]
    else:
        # The current two-rank MVP never takes this branch.  A timed placeholder
        # would alter global semantics, so reject wider topologies explicitly.
        raise TraceGenerationError(
            f"{op.op_id}: p2p does not involve rank {rank}; multi-rank placeholders "
            "are intentionally unsupported"
        )
    node.attr.extend(
        [
            attr(name="is_cpu_op", bool_val=False),
            attr(name="comm_src", uint32_val=op.src_rank),
            attr(name="comm_dst", uint32_val=op.dst_rank),
            attr(name="comm_tag", uint32_val=op.node_id),
            attr(name="comm_size", uint64_val=op.payload_bytes),
        ]
    )
    return node


def generate_traces(
    operations: Iterable[Operation],
    trace_prefix: str | Path,
    *,
    ranks: int,
    astra_time_scale: int,
    comm_groups: dict | None = None,
) -> list[Path]:
    """Write ``<prefix>.<rank>.et`` files and return their paths."""

    ops = list(operations)
    _validate_operations(ops, ranks, comm_groups)
    if astra_time_scale != 1000:
        raise TraceGenerationError(
            "this MVP requires astra_time_scale=1000 to encode ns as Chakra us"
        )
    schema = _schema()
    prefix = Path(trace_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for rank in range(ranks):
        path = Path(f"{prefix}.{rank}.et")
        with path.open("wb") as stream:
            _write_delimited(stream, schema.GlobalMetadata(version="0.0.4"))  # type: ignore[attr-defined]
            for op in ops:
                if rank in participant_ranks(op, ranks):
                    _write_delimited(stream, _node_for_rank(schema, op, rank))
        if path.stat().st_size <= 0:
            raise TraceGenerationError(f"empty Chakra trace: {path}")
        paths.append(path)
    return paths
