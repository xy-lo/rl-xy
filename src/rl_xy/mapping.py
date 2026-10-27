"""Deterministic single-core OS reference mapper; no hidden cross-kernel cache."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .operator_ir import Matmul


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class MappingEstimate:
    tile: tuple[int, int, int]
    workspace_bytes: int
    dram_read_bytes: int
    dram_write_bytes: int
    flops: int
    dataflow: str = "output_stationary_no_cross_output_tile_cache"


def os_mapping(problem: Matmul, tile: tuple[int, int, int], capacity: int) -> MappingEstimate:
    if type(capacity) is not int or capacity <= 0 or len(tile) != 3 or any(type(x) is not int or x <= 0 for x in tile):
        raise ValueError("invalid SRAM capacity/tile")
    m, n, k = (min(x, dim) for x, dim in zip(tile, (problem.m, problem.n, problem.k)))
    workspace = m * k * problem.a_bytes + k * n * problem.b_bytes + m * n * problem.accumulator_bytes
    if workspace > capacity:
        raise ValueError(f"SRAM mapping requires {workspace} B, capacity={capacity} B")
    # Each output tile retains its accumulator through all K tiles. A/B are
    # reloaded for every output tile. Ragged edges use their actual extents.
    reads = problem.batches * (
        problem.m * problem.k * ceildiv(problem.n, n) * problem.a_bytes
        + problem.k * problem.n * ceildiv(problem.m, m) * problem.b_bytes
    )
    writes = problem.batches * problem.m * problem.n * problem.output_bytes
    return MappingEstimate((m, n, k), workspace, reads, writes, problem.flops)


def choose_os_mapping(problem: Matmul, capacity: int, max_tile: int = 32) -> MappingEstimate:
    if max_tile <= 0:
        raise ValueError("max_tile must be positive")
    # Bounded deterministic candidates include ragged/full small dimensions.
    choices = [sorted({min(dim, x) for x in (1, 2, 4, 8, 16, max_tile) if x <= max_tile})
               for dim in (problem.m, problem.n, problem.k)]
    legal = []
    for tile in product(*choices):
        try:
            legal.append(os_mapping(problem, tile, capacity))
        except ValueError:
            continue
    if not legal:
        raise ValueError(f"no legal OS mapping in SRAM capacity={capacity} B")
    return min(legal, key=lambda x: (x.dram_read_bytes + x.dram_write_bytes, -x.tile[2], -x.tile[0], -x.tile[1]))

