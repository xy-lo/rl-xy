"""Shape-bearing contracts for local kernels (not a numeric tensor runtime)."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class TensorSpec:
    tensor_id: str
    shape: tuple[int, ...]
    dtype_bytes: int
    layout: str = "rank_local"

    def __post_init__(self):
        if not self.tensor_id or not self.shape or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ValueError("tensor needs an identity and positive integer shape")
        if self.dtype_bytes not in (1, 2, 4, 8):
            raise ValueError("unsupported tensor dtype width")

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype_bytes


@dataclass(frozen=True)
class Matmul:
    m: int
    n: int
    k: int
    batches: int = 1
    a_bytes: int = 2
    b_bytes: int = 2
    output_bytes: int = 2
    accumulator_bytes: int = 4
    direction: str = "forward"

    def __post_init__(self):
        for field in (self.m, self.n, self.k, self.batches):
            if type(field) is not int or field <= 0:
                raise ValueError("matmul dimensions must be positive integers")
        if any(x not in (1, 2, 4, 8) for x in (self.a_bytes, self.b_bytes, self.output_bytes, self.accumulator_bytes)):
            raise ValueError("unsupported matmul dtype width")
        if self.direction not in ("forward", "dinput", "dweight", "attention_backward"):
            raise ValueError("unknown matmul direction")

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k * self.batches


@dataclass(frozen=True)
class KernelSpec:
    kernel_variant: str
    matmuls: tuple[Matmul, ...] = ()
    vector_ops: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    execution_domain: str = "device"
    timing_contract: str = "inclusive_local_serial"

    def __post_init__(self):
        if not self.kernel_variant or min(self.vector_ops, self.read_bytes, self.write_bytes) < 0:
            raise ValueError("invalid kernel work/traffic")
        if self.execution_domain not in ("device", "host"):
            raise ValueError("unknown execution domain")
        if self.timing_contract != "inclusive_local_serial":
            raise ValueError("mapped kernels use exactly one inclusive-local contract")

