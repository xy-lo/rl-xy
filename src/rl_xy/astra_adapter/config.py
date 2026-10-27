"""Fail-closed validation for the frozen AstraSim configuration bundle.

The case JSON expresses hardware quantities in model nanoseconds while the
official AstraSim analytical backend consumes latency in simulator ticks and
bandwidth in GiB/s.  This module is the single place where that unit contract
is checked before a trace is executed.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


class AstraConfigError(ValueError):
    """The AstraSim files do not implement the frozen rl-xy backend contract."""


SYSTEM_CONTRACT: dict[str, Any] = {
    "scheduling-policy": "FIFO",
    "endpoint-delay": 0,
    "local-reduction-delay": 0,
    "active-chunks-per-dimension": 1,
    "preferred-dataset-splits": 1,
    "all-reduce-implementation": ["ring"],
    "all-gather-implementation": ["ring"],
    "reduce-scatter-implementation": ["ring"],
    "all-to-all-implementation": ["ring"],
    "collective-optimization": "baseline",
    "local-mem-bw": 16,
    "roofline-enabled": 0,
    "replay-only": 0,
    "trace-enabled": 1,
    "track-local-mem": 0,
}

REMOTE_MEMORY_CONTRACT = {"memory-type": "NO_MEMORY_EXPANSION"}
NETWORK_KEYS = {"topology", "npus_count", "bandwidth", "latency"}
NETWORK_LINE = re.compile(
    r"^(?P<key>[A-Za-z_]+)\s*:\s*\[\s*(?P<value>[^\[\]]+?)\s*\]\s*$"
)


def _strict_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AstraConfigError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AstraConfigError(f"{label} must be a JSON object")
    return value


def _network_values(path: Path) -> dict[str, str]:
    """Parse the intentionally tiny one-dimensional network YAML subset.

    Pulling a general YAML loader into the runtime would enlarge the trusted
    surface for four scalar lists.  Unknown syntax, duplicates, and comments
    on value lines are rejected instead of being guessed.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AstraConfigError(f"cannot read network YAML {path}: {exc}") from exc
    values: dict[str, str] = {}
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = NETWORK_LINE.fullmatch(line)
        if match is None:
            raise AstraConfigError(
                f"unsupported network YAML syntax at {path}:{line_number}: {raw!r}"
            )
        key = match.group("key")
        if key in values:
            raise AstraConfigError(f"duplicate network field {key!r}")
        values[key] = match.group("value").strip()
    unknown = sorted(set(values) - NETWORK_KEYS)
    missing = sorted(NETWORK_KEYS - set(values))
    if unknown or missing:
        raise AstraConfigError(
            f"network fields do not match contract: unknown={unknown}, missing={missing}"
        )
    return values


def validate_astra_configs(
    *,
    system_path: str | Path,
    network_path: str | Path,
    remote_memory_path: str | Path,
    ranks: int,
    astra_time_scale: int,
    d2d_bytes_per_model_ns: int,
    d2d_latency_model_ns: int,
) -> dict[str, Any]:
    """Validate and return a JSON-serializable resolved adapter contract."""

    if ranks != 2:
        raise AstraConfigError("the frozen MVP requires exactly two AstraSim ranks")
    if astra_time_scale != 1000:
        raise AstraConfigError("the frozen Chakra ns-to-us contract requires scale=1000")
    if d2d_bytes_per_model_ns <= 0 or d2d_latency_model_ns <= 0:
        raise AstraConfigError("D2D bandwidth and latency must be positive")

    system = _json_object(Path(system_path).resolve(), "system")
    if not _strict_equal(system, SYSTEM_CONTRACT):
        unexpected = sorted(set(system) - set(SYSTEM_CONTRACT))
        missing = sorted(set(SYSTEM_CONTRACT) - set(system))
        mismatched = {
            key: {"expected": SYSTEM_CONTRACT[key], "actual": system.get(key)}
            for key in sorted(set(system) & set(SYSTEM_CONTRACT))
            if not _strict_equal(system[key], SYSTEM_CONTRACT[key])
        }
        raise AstraConfigError(
            "system config does not match frozen contract: "
            f"unexpected={unexpected}, missing={missing}, mismatched={mismatched}"
        )

    remote = _json_object(Path(remote_memory_path).resolve(), "remote-memory")
    if not _strict_equal(remote, REMOTE_MEMORY_CONTRACT):
        raise AstraConfigError(
            "remote-memory config does not match frozen contract: "
            f"expected={REMOTE_MEMORY_CONTRACT}, actual={remote}"
        )

    network = _network_values(Path(network_path).resolve())
    if network["topology"] != "FullyConnected":
        raise AstraConfigError("network topology must be FullyConnected")
    try:
        configured_ranks = int(network["npus_count"])
        configured_bandwidth = float(network["bandwidth"])
        configured_latency = float(network["latency"])
    except ValueError as exc:
        raise AstraConfigError("network rank, bandwidth, and latency must be numeric") from exc
    if configured_ranks != ranks:
        raise AstraConfigError(
            f"network npus_count mismatch: expected {ranks}, got {configured_ranks}"
        )

    # AstraSim converts configured GiB/s to bytes/s and then to bytes/ns.
    # Scaling model time by S therefore requires model_B/ns / S in raw ticks.
    expected_bandwidth = (
        d2d_bytes_per_model_ns / astra_time_scale * 1_000_000_000 / (2**30)
    )
    expected_latency = float(d2d_latency_model_ns * astra_time_scale)
    if not math.isclose(
        configured_bandwidth, expected_bandwidth, rel_tol=0.0, abs_tol=1e-18
    ):
        raise AstraConfigError(
            "network bandwidth does not preserve the case time scale: "
            f"expected {expected_bandwidth:.18g}, got {configured_bandwidth:.18g}"
        )
    if configured_latency != expected_latency:
        raise AstraConfigError(
            "network latency does not preserve the case time scale: "
            f"expected {expected_latency}, got {configured_latency}"
        )

    return {
        "backend": "AstraSim_Analytical_Congestion_Unaware",
        "contention_modelled": False,
        "ranks": ranks,
        "astra_time_scale": astra_time_scale,
        "model_ns_per_raw_tick": 1 / astra_time_scale,
        "topology": network["topology"],
        "network_bandwidth_config_gib_per_s": configured_bandwidth,
        "effective_raw_bytes_per_ns": configured_bandwidth * (2**30) / 1_000_000_000,
        "model_d2d_bytes_per_ns": d2d_bytes_per_model_ns,
        "network_latency_raw_ns": configured_latency,
        "model_d2d_latency_ns": d2d_latency_model_ns,
        "system": system,
        "remote_memory": remote,
    }
