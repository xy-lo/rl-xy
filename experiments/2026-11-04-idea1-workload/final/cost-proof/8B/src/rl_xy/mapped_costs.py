"""Attach inspectable analytic local costs to the existing operator graph.

This pass changes costs, never invents AstraSim event timestamps. Tensor life
cycles and resource admission are a separate contract, not inferred here.
"""
from dataclasses import asdict, replace

from .cost_model import AnalyticalCostProvider
from .model import parameter_totals
from .operator_ir import KernelSpec
from .types import CaseConfig, Operation


def apply_mapped_costs(case: CaseConfig, operations: list[Operation], *, provider=None) -> list[Operation]:
    provider = provider or AnalyticalCostProvider()
    result = []
    local_params = parameter_totals(case)["local_parameter_elements_per_die"]
    for op in operations:
        if op.kind != "compute":
            result.append(op)
            continue
        meta = dict(op.metadata)
        logical_executor = meta.get("logical_executor", "")
        if (logical_executor in {"host_controller", "host_link"}
                or "cost_source" in meta):
            meta["execution_domain"] = "host" if logical_executor else "device"
            meta["cost_estimate"] = {
                "duration_ns": op.duration_ns,
                "source": "explicit_synthetic_host_or_control_fixture",
                "timing_contract": "explicit_fixture_no_hidden_dma",
            }
            result.append(replace(op, metadata=meta))
            continue
        if op.flops and op.kernel_spec is None:
            raise ValueError(f"{op.name}: missing shape-bearing matmul contract")
        spec = op.kernel_spec
        if spec is None:
            # Declared conservative streaming contract, not a measured kernel:
            # two full-sized reads and one write, unless known state R/W differs.
            reads, writes = 2 * op.tensor_bytes, op.tensor_bytes
            if op.name == "zero_gradient_buffer":
                reads, writes = 0, local_params * case.model.gradient_dtype_bytes
            elif op.name == "adamw_read_modify_write":
                reads = local_params * (case.model.gradient_dtype_bytes + case.model.master_dtype_bytes + 2 * case.model.optimizer_dtype_bytes)
                writes = local_params * (case.model.master_dtype_bytes + 2 * case.model.optimizer_dtype_bytes + case.model.actor_dtype_bytes)
            elif op.name == "local_gradient_square_sum":
                reads, writes = local_params * case.model.gradient_dtype_bytes, 4
            spec = KernelSpec(op.name.rsplit(".", 1)[-1], vector_ops=op.vector_ops,
                              read_bytes=reads, write_bytes=writes)
            meta["traffic_assumption"] = "streaming_2_reads_1_write_except_explicit_state_kernels"
        estimate = provider.estimate(spec, case.hardware)
        typed_flops = sum(problem.flops for problem in spec.matmuls)
        meta["legacy_flops"] = op.flops
        meta["execution_domain"] = "device"
        meta["cost_estimate"] = provider.metadata_for(spec,case.hardware) if hasattr(provider,"metadata_for") else asdict(estimate)
        result.append(replace(op, duration_ns=estimate.duration_ns,
                              flops=typed_flops, kernel_spec=spec, metadata=meta))
    return result
