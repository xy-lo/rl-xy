"""Read veRL L1 length captures without inventing tokens, rewards or chip time."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


STAGES = {"rollout", "old_logprob", "ref", "actor", "sync"}


class TraceError(ValueError):
    pass


def _positive(value, label, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise TraceError(f"{label}: expected {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _fields(row, fields, label):
    if not isinstance(row, dict) or set(row) != set(fields):
        raise TraceError(f"{label}: unexpected/missing fields")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise TraceError(f"{label}: nonempty string required")
    return value


def load_verl_trace(path: str | Path, *, origin: str) -> dict:
    """Normalize hook records. Origin is a declaration, not authenticity proof."""
    if origin not in ("synthetic_fixture", "framework_capture", "statistics_constrained_synthetic"):
        raise TraceError("explicit trace origin required")
    source = Path(path)
    records = []
    for number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(TraceError(f"nonfinite {x}")))
        except (json.JSONDecodeError, TraceError) as exc:
            raise TraceError(f"line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise TraceError(f"line {number}: record must be object")
        records.append(row)
    if not records or records[0].get("type") != "run_meta":
        raise TraceError("trace must begin with one run_meta")
    meta = records[0]
    _fields(meta, {"type", "run_id", "model", "data", "algo", "hardware", "environment", "steps_recorded"}, "run_meta")
    _text(meta["run_id"], "run_id")
    for scope, names in {
        "model": {"family", "name", "revision", "variant"},
        "data": {"dataset", "revision", "split"},
        "hardware": {"gpu", "n_gpu"}, "environment": {"enabled", "mode"},
        "algo": {"name", "framework", "framework_version", "framework_commit", "group_n", "batch_prompts",
                 "temperature", "top_p", "max_prompt_tokens", "max_response_tokens"},
    }.items():
        _fields(meta[scope], names, scope)
        for k, value in meta[scope].items():
            if k in {"group_n", "batch_prompts", "max_prompt_tokens", "max_response_tokens", "n_gpu"}:
                _positive(value, k)
            elif k in {"temperature", "top_p"}:
                if value is None:
                    continue  # not available in a historical override-only record
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or (k == "top_p" and value > 1):
                    raise TraceError(f"invalid {k}")
            elif k == "enabled":
                if type(value) is not bool:
                    raise TraceError("environment.enabled must be bool")
            else:
                _text(value, f"{scope}.{k}")
    if meta["algo"]["framework"] != "verl" or meta["algo"]["name"] != "grpo":
        raise TraceError("current adapter accepts veRL GRPO only")
    if meta["environment"]["enabled"]:
        raise TraceError("multi-turn/tool lifecycle is not supported by L1 adapter")
    bounds = meta["steps_recorded"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise TraceError("steps_recorded must be inclusive [first,last]")
    for b in bounds:
        _positive(b, "step bound")
    if bounds[1] < bounds[0]:
        raise TraceError("invalid step interval")
    trajectories = defaultdict(list)
    timings = {}
    seen = set()
    closed_steps = set()
    previous_step = bounds[0]
    for row in records[1:]:
        kind = row.get("type")
        if kind not in ("traj", "stage_times"):
            raise TraceError(f"unsupported/duplicate record type {kind!r}")
        step = _positive(row.get("step"), "step")
        if step < previous_step or not bounds[0] <= step <= bounds[1]:
            raise TraceError("records must have ordered steps within declared interval")
        previous_step = step
        if kind == "traj":
            _fields(row, {"type", "step", "traj_id", "prompt_len", "resp_len", "group_id", "behavior_version", "finish_reason"}, "traj")
            if step in closed_steps:
                raise TraceError("trajectory appears after step timing/commit marker")
            tid = _text(row["traj_id"], "traj_id")
            if tid in seen:
                raise TraceError("duplicate trajectory id")
            seen.add(tid)
            _text(row["group_id"], "group_id")
            for key, maximum in (("prompt_len", "max_prompt_tokens"), ("resp_len", "max_response_tokens")):
                if _positive(row[key], key) > meta["algo"][maximum]:
                    raise TraceError(f"{key} exceeds public bound")
            if _positive(row["behavior_version"], "behavior_version", True) != step - 1:
                raise TraceError("current synchronous adapter requires behavior_version=step-1")
            if row["finish_reason"] not in ("eos", "truncated"):
                raise TraceError("DROP/EMPTY/error trajectories require a richer lifecycle schema")
            trajectories[step].append({k: v for k, v in row.items() if k not in ("type", "step")})
        else:
            _fields(row, {"type", "step", "stage_times_s"}, "stage_times")
            if step in timings:
                raise TraceError("duplicate stage_times")
            _fields(row["stage_times_s"], STAGES, "stage_times_s")
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in row["stage_times_s"].values()):
                raise TraceError("invalid observed GPU stage time")
            timings[step] = row["stage_times_s"]
            closed_steps.add(step)
    observed = sorted(trajectories)
    if not observed or observed != list(range(bounds[0], observed[-1] + 1)) or set(observed) != set(timings):
        raise TraceError("missing/empty/noncontiguous steps or timing coverage")
    steps = []
    warnings = []
    for key in ("temperature", "top_p"):
        if meta["algo"][key] is None:
            warnings.append({"unresolved_sampling_parameter": key})
    for step in observed:
        grouped = defaultdict(list)
        for traj in trajectories[step]:
            grouped[traj["group_id"]].append(traj)
        if len(grouped) != meta["algo"]["batch_prompts"]:
            raise TraceError("observed prompt groups differ from batch_prompts")
        groups = []
        for gid, rows in sorted(grouped.items()):
            if len(rows) != meta["algo"]["group_n"]:
                raise TraceError("partial/oversized GRPO group")
            if len({r["prompt_len"] for r in rows}) != 1:
                raise TraceError("same group has inconsistent prompt lengths")
            groups.append({"group_id": gid, "prompt_length": rows[0]["prompt_len"],
                           "trajectories": sorted(rows, key=lambda x: x["traj_id"])})
        zero = sorted(k for k, v in timings[step].items() if v == 0)
        if zero:
            warnings.append({"step": step, "zero_gpu_stage_times_unverified": zero})
        steps.append({"step": step, "behavior_version": step - 1, "groups": groups,
                      "global_response_tokens": sum(t["resp_len"] for t in trajectories[step]),
                      "gpu_observed_stage_times_s": timings[step]})
    if observed[-1] != bounds[1]:
        warnings.append({"declared_last_step": bounds[1], "observed_last_step": observed[-1], "capture_complete": False})
    return {
        "schema_version": "rlxy.verl-workload/1", "origin_declared": origin,
        "capture_authenticity_verified": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "run_meta": meta, "steps": steps, "warnings": warnings,
        "available": {"lengths": True, "groups": True, "versions": True,
                      "token_ids": False, "reward_values": False, "logprob_values": False},
        "timing_contract": "GPU stage times are observations only; never wafer operation durations",
        "scope": "synchronous complete-group length capture; not a full operator profiler trace",
    }
