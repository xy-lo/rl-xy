"""Auditable inclusive-local reference cost and exact-match profile provider."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .mapping import MappingEstimate, ceildiv, choose_os_mapping
from .operator_ir import KernelSpec
from .types import HardwareConfig


@dataclass(frozen=True)
class CostEstimate:
    duration_ns: int
    compute_ns: int
    local_memory_ns: int
    dram_read_bytes: int
    dram_write_bytes: int
    sram_workspace_bytes: int
    mappings: tuple[MappingEstimate, ...]
    timing_contract: str = "inclusive_local_serial"
    source: str = "analytic_os_reference_v1_not_hardware_calibrated"
    core_workspace_bytes: tuple[int, ...] = ()
    core_compute_ns: tuple[int, ...] = ()
    partitions: tuple[dict, ...] = ()


class AnalyticalCostProvider:
    """No compute/DMA overlap, one core per die; local path is a bandwidth bound.

    DMA traverses DRAM/module/SRAM as one path, charged once at its bottleneck.
    NoC packets, banks, startup and cache effects are NOT inferred from this rate.
    """

    def estimate(self, spec: KernelSpec, hardware: HardwareConfig) -> CostEstimate:
        if spec.execution_domain != "device":
            raise ValueError("host fixture duration must be explicitly provided")
        rates = (hardware.dram_bytes_per_ns, hardware.module_path_bytes_per_ns,
                 hardware.sram_bytes_per_ns, hardware.gemm_flops_per_ns, hardware.vector_ops_per_ns)
        if min(rates) <= 0:
            raise ValueError("rates must be positive")
        if hardware.cores_per_die != 1:
            from .multicore import estimate_multicore
            return estimate_multicore(spec, hardware, CostEstimate)
        mappings = tuple(choose_os_mapping(p, hardware.sram_bytes_per_core) for p in spec.matmuls)
        # Matmuls in a fused cost group run serially and reuse the workspace.
        reads = sum(m.dram_read_bytes for m in mappings) + spec.read_bytes
        writes = sum(m.dram_write_bytes for m in mappings) + spec.write_bytes
        work_ns = sum(ceildiv(p.flops, hardware.gemm_fp32_flops_per_ns
            if hardware.gemm_fp32_flops_per_ns and max(p.a_bytes,p.b_bytes)>=4
            else hardware.gemm_flops_per_ns) for p in spec.matmuls)
        work_ns += ceildiv(spec.vector_ops, hardware.vector_ops_per_ns)
        local_ns = ceildiv(reads + writes, min(rates[:3]))
        # Vector streaming uses one element input/output/FP32 scratch (16B),
        # irrespective of the full tensor length; no whole tensor in SRAM.
        vector_workspace = 16 if (spec.vector_ops or spec.read_bytes or spec.write_bytes) else 0
        workspace = max([m.workspace_bytes for m in mappings] + [vector_workspace])
        if workspace > hardware.sram_bytes_per_core:
            raise ValueError("vector workspace exceeds SRAM capacity")
        return CostEstimate(max(1, work_ns + local_ns), work_ns, local_ns,
                            reads, writes, workspace, mappings)


def profile_key(spec: KernelSpec, hardware: HardwareConfig) -> str:
    data = json.dumps({"kernel": asdict(spec), "hardware": asdict(hardware)}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


class LookupCostProvider:
    """Read-only exact signatures; missing/out-of-domain records fail closed.

    Profiles must carry independently supplied provenance. This class does not
    invent hardware measurements or train a surrogate on synthetic labels.
    """
    def __init__(self, records: dict):
        self.records = records

    def estimate(self, spec: KernelSpec, hardware: HardwareConfig) -> CostEstimate:
        key = profile_key(spec, hardware)
        if key not in self.records:
            raise ValueError("profile miss/out of domain; no silent fallback")
        row = self.records[key]
        if not row.get("label_origin") or not row.get("reference_version"):
            raise ValueError("profile lacks label provenance")
        value = row["estimate"]
        mappings = tuple(MappingEstimate(**{**m, "tile": tuple(m["tile"])}) for m in value["mappings"])
        result = CostEstimate(**{**value, "mappings": mappings})
        if (result.duration_ns <= 0 or min(result.compute_ns, result.local_memory_ns,
                result.dram_read_bytes, result.dram_write_bytes, result.sram_workspace_bytes) < 0
                or result.sram_workspace_bytes > hardware.sram_bytes_per_core
                or result.duration_ns != max(1, result.compute_ns + result.local_memory_ns)
                or any(m.workspace_bytes > result.sram_workspace_bytes for m in result.mappings)
                or result.timing_contract != "inclusive_local_serial"):
            raise ValueError("invalid profile physical/timing contract")
        return result
