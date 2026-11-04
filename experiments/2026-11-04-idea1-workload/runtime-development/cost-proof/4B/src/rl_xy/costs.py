"""Transparent synthetic operation costs used to build the toy workload."""

from __future__ import annotations

import math

from .types import CaseConfig


def compute_duration_ns(case: CaseConfig, *, flops: int = 0, vector_ops: int = 0) -> int:
    if flops < 0 or vector_ops < 0:
        raise ValueError("operation counts must be non-negative")
    flop_time = math.ceil(flops / case.hardware.gemm_flops_per_ns) if flops else 0
    vector_time = math.ceil(vector_ops / case.hardware.vector_ops_per_ns) if vector_ops else 0
    return max(1, flop_time + vector_time)


def local_dma_duration_ns(case: CaseConfig, nbytes: int) -> int:
    if nbytes <= 0:
        raise ValueError("DMA size must be positive")
    bottleneck = min(
        case.hardware.dram_bytes_per_ns,
        case.hardware.module_path_bytes_per_ns,
        case.hardware.sram_bytes_per_ns,
    )
    return math.ceil(nbytes / bottleneck)


def dense_linear_flops(tokens: int, in_features: int, out_features: int) -> int:
    return 2 * tokens * in_features * out_features

