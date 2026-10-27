"""Reuse historical veRL step statistics and generate declared synthetic lengths.

Only whitelisted scalar metrics are parsed; log text is never evaluated.
Synthetic trajectories match available marginal statistics, not lost samples.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC = re.compile(rf"(?:^| - )([A-Za-z_][\w/.-]*):({NUMBER})(?= - |$)")
REQUIRED = {
    "prompt_length/mean", "prompt_length/min", "prompt_length/max",
    "response_length/mean", "response_length/min", "response_length/max", "response_length/clip_ratio",
    "timing_s/gen", "timing_s/old_log_prob", "timing_s/ref", "timing_s/update_actor", "timing_s/update_weights", "timing_s/step",
}
OPTIONAL = {"critic/score/mean", "critic/rewards/mean", "actor/pg_loss", "actor/kl_loss", "actor/loss",
            "timing_s/adv", "perf/throughput", "actor/perf/max_memory_allocated_gb"}
OVERRIDES = {
    "data.train_batch_size", "data.max_prompt_length", "data.max_response_length", "data.train_files",
    "actor_rollout_ref.rollout.n", "actor_rollout_ref.model.path", "trainer.experiment_name",
    "trainer.n_gpus_per_node", "trainer.nnodes", "algorithm.adv_estimator",
    "actor_rollout_ref.rollout.temperature", "actor_rollout_ref.rollout.top_p",
    "actor_rollout_ref.actor.ppo_mini_batch_size", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "actor_rollout_ref.model.enable_gradient_checkpointing", "actor_rollout_ref.model.use_remove_padding",
    "actor_rollout_ref.actor.use_kl_loss", "actor_rollout_ref.actor.kl_loss_coef", "actor_rollout_ref.actor.kl_loss_type",
}


def read_legacy_overrides(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("- "):
            raise ValueError("expected Hydra scalar override list")
        text = line[2:].strip()
        if text.startswith(("'", '"')):
            raise ValueError("quoted YAML override requires a YAML adapter; no guessing")
        if "=" not in text:
            raise ValueError("override missing =")
        key, value = text.split("=", 1)
        if key in OVERRIDES:
            if key in values:
                raise ValueError(f"duplicate override {key}")
            values[key] = value
    for k in ("data.train_batch_size", "data.max_prompt_length", "data.max_response_length", "actor_rollout_ref.rollout.n"):
        if k not in values or not values[k].isdigit() or int(values[k]) <= 0:
            raise ValueError(f"missing/invalid workload override {k}")
    if values.get("algorithm.adv_estimator") != "grpo":
        raise ValueError("legacy adapter requires explicit GRPO override")
    for k in ("actor_rollout_ref.model.path", "data.train_files", "trainer.experiment_name"):
        if not values.get(k):
            raise ValueError(f"missing override {k}")
    for k in ("trainer.n_gpus_per_node", "trainer.nnodes"):
        if k not in values or not values[k].isdigit() or int(values[k]) <= 0:
            raise ValueError(f"missing/invalid historical GPU count {k}")
    return values


def import_legacy_run(log_path, overrides_path, run_id):
    log_path, overrides_path = Path(log_path), Path(overrides_path)
    overrides = read_legacy_overrides(overrides_path)
    if overrides["trainer.experiment_name"] != run_id:
        raise ValueError("run_id does not match archived Hydra overrides")
    steps = {}
    for line in log_path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", line)
        match = re.search(r"\bstep:(\d+)\s+-", line)
        if not match or "timing_s/step:" not in line:
            continue
        step = int(match[1])
        values = {k: float(v) for k, v in METRIC.findall(line) if k in REQUIRED | OPTIONAL}
        if REQUIRED - set(values):
            raise ValueError(f"step{step}: missing metrics {sorted(REQUIRED - set(values))}")
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError("nonfinite log metric")
        for kind in ("prompt", "response"):
            a, b, c = (values[f"{kind}_length/{stat}"] for stat in ("min", "mean", "max"))
            if not 0 < a <= b <= c or a != int(a) or c != int(c):
                raise ValueError(f"invalid {kind} length statistics")
            bound = int(overrides[f"data.max_{kind}_length"])
            if c > bound:
                raise ValueError(f"{kind} exceeds configured bound")
        if not 0 <= values["response_length/clip_ratio"] <= 1 or any(v < 0 for k, v in values.items() if k.startswith("timing_s/")):
            raise ValueError("invalid clipping ratio/timing")
        if step in steps and steps[step] != values:
            raise ValueError("conflicting duplicate step")
        steps[step] = values
    if not steps or sorted(steps) != list(range(1, max(steps) + 1)):
        raise ValueError("missing/noncontiguous legacy step coverage")
    inconsistent_clip_steps = [s for s in sorted(steps) if steps[s]["response_length/max"] < int(overrides["data.max_response_length"])
                               and steps[s]["response_length/clip_ratio"] > 0]
    return {
        "schema_version": "rlxy.legacy-statistics/1", "run_id": run_id,
        "evidence_kind": "observed_gpu_step_statistics", "steps": [{"step": s, "metrics": steps[s]} for s in sorted(steps)],
        "overrides": overrides,
        "clip_metric_audit": {"cannot_mean_public_cap_hit_steps": inconsistent_clip_steps,
                              "public_cap_interpretation_consistent": not inconsistent_clip_steps,
                              "observed_batch_max_reference_is_a_modeling_choice": True},
        "sources": {"log": {"path": str(log_path), "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()},
                    "hydra_overrides": {"path": str(overrides_path), "sha256": hashlib.sha256(overrides_path.read_bytes()).hexdigest()}},
        "limitations": ["per-trajectory lengths/group membership/token IDs/rewards/logprobs are not recoverable",
                        "overrides are archived inputs, not the full resolved algorithm contract",
                        "historical framework/model/data revisions need separate provenance verification"],
    }


def _lengths(count, minimum, maximum, mean, rng, *, cap=None, clipped=0):
    minimum, maximum = int(minimum), int(maximum)
    if count < 2 or not 0 < minimum <= mean <= maximum or not 0 <= clipped <= count:
        raise ValueError("infeasible length statistics/count")
    if clipped and (cap is None or maximum != cap):
        raise ValueError("clipping inconsistent with maximum/cap")
    fixed = [cap] * clipped
    remaining = count - clipped
    upper = min(maximum, cap - 1) if cap is not None else maximum
    if remaining:
        if minimum > upper:
            raise ValueError("unclipped lengths incompatible with minimum")
        fixed.append(minimum)
        remaining -= 1
        if not clipped and maximum > minimum:
            if not remaining:
                raise ValueError("cannot represent min/max")
            fixed.append(maximum)
            remaining -= 1
    elif minimum != maximum:
        raise ValueError("all clipped but distinct minimum")
    target = round(mean * count)
    if not sum(fixed) + remaining * minimum <= target <= sum(fixed) + remaining * upper:
        raise ValueError("mean/min/max/clipping statistics infeasible together")
    variable = [rng.randint(minimum, upper) for _ in range(remaining)]
    delta = target - sum(fixed) - sum(variable)
    order = list(range(remaining))
    rng.shuffle(order)
    for i in order:
        change = min(delta, upper - variable[i]) if delta >= 0 else -min(-delta, variable[i] - minimum)
        variable[i] += change
        delta -= change
    if delta:
        raise ValueError("length mean adjustment failed")
    result = fixed + variable
    rng.shuffle(result)
    if len(result) != count or min(result) != minimum or max(result) != maximum or sum(result) != target:
        raise ValueError("length moment preservation failed")
    if cap is not None and sum(x == cap for x in result) != clipped:
        raise ValueError("clip count preservation failed")
    return result


def synthesize_lengths(statistics, *, seed, selected_steps, clip_reference):
    """Build bounded randomized lengths, preserving available integer moments."""
    if type(seed) is not int:
        raise ValueError("integer seed required")
    if clip_reference not in ("public_cap", "observed_max"):
        raise ValueError("explicit legacy clip reference required")
    overrides = statistics["overrides"]
    prompts = int(overrides["data.train_batch_size"])
    group_n = int(overrides["actor_rollout_ref.rollout.n"])
    max_prompt, cap = int(overrides["data.max_prompt_length"]), int(overrides["data.max_response_length"])
    if prompts < 2 or group_n < 2:
        raise ValueError("statistics synthesizer requires at least two groups and two responses/group")
    selected = {s["step"]: s for s in statistics["steps"] if s["step"] in selected_steps}
    if not selected_steps or sorted(selected) != selected_steps or selected_steps != list(range(selected_steps[0], selected_steps[-1] + 1)):
        raise ValueError("selected steps must exist and form an ascending nonempty contiguous range")
    run_id = f"{statistics['run_id']}-statistics-synthetic-seed{seed}"
    model_name = Path(overrides["actor_rollout_ref.model.path"]).name
    data_name = Path(overrides["data.train_files"]).parent.name
    # Sampling knobs are not used to synthesize lengths. Missing historical
    # overrides remain null rather than being replaced by current defaults.
    meta = {"type": "run_meta", "run_id": run_id,
            "model": {"family": "Qwen", "name": model_name, "revision": "historical-unverified", "variant": "Instruct" if "Instruct" in model_name else "Base"},
            "data": {"dataset": data_name, "revision": "historical-unverified", "split": "train"},
            "algo": {"name": "grpo", "framework": "verl", "framework_version": "historical-unverified", "framework_commit": "historical-unverified",
                     "group_n": group_n, "batch_prompts": prompts,
                     "temperature": float(overrides["actor_rollout_ref.rollout.temperature"]) if "actor_rollout_ref.rollout.temperature" in overrides else None,
                     "top_p": float(overrides["actor_rollout_ref.rollout.top_p"]) if "actor_rollout_ref.rollout.top_p" in overrides else None,
                     "max_prompt_tokens": max_prompt, "max_response_tokens": cap},
            "hardware": {"gpu": "historical-GPU-unverified", "n_gpu": int(overrides["trainer.n_gpus_per_node"]) * int(overrides["trainer.nnodes"])},
            "environment": {"enabled": False, "mode": "single-turn-synthetic"}, "steps_recorded": [selected_steps[0], selected_steps[-1]]}
    records = [meta]
    checks = []
    rng = random.Random(seed)
    for step in selected_steps:
        metrics = selected[step]["metrics"]
        p = _lengths(prompts, metrics["prompt_length/min"], metrics["prompt_length/max"], metrics["prompt_length/mean"], rng)
        n = prompts * group_n
        clipped = round(metrics["response_length/clip_ratio"] * n)
        reference_width = cap if clip_reference == "public_cap" else int(metrics["response_length/max"])
        r = _lengths(n, metrics["response_length/min"], metrics["response_length/max"], metrics["response_length/mean"], rng,
                     cap=reference_width, clipped=clipped)
        for i, length in enumerate(r):
            records.append({"type": "traj", "step": step, "traj_id": f"s{step}-generated-t{i}", "group_id": f"s{step}-generated-g{i // group_n}",
                            "prompt_len": p[i // group_n], "resp_len": length, "behavior_version": step - 1,
                            "finish_reason": "truncated" if length == cap else "eos"})
        records.append({"type": "stage_times", "step": step, "stage_times_s": {
            k: metrics[f"timing_s/{key}"] for k, key in (("rollout", "gen"), ("old_logprob", "old_log_prob"),
                                                       ("ref", "ref"), ("actor", "update_actor"), ("sync", "update_weights"))}})
        checks.append({"step": step, "prompt_mean_source": metrics["prompt_length/mean"], "prompt_mean_generated": sum(p) / prompts,
                       "response_mean_source": metrics["response_length/mean"], "response_mean_generated": sum(r) / n,
                       "legacy_clip_ratio_source": metrics["response_length/clip_ratio"], "legacy_clip_ratio_generated": clipped / n,
                       "public_cap_hit_ratio_generated": sum(x == cap for x in r) / n,
                       "trajectory_count": n})
    report = {"schema_version": "rlxy.synthesis/1", "origin": "statistics_constrained_synthetic", "run_id": run_id, "seed": seed,
              "source_statistics_sha256": hashlib.sha256(json.dumps(statistics, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              "checks": checks, "method": "bounded random integers then exact sum adjustment; independent generated group assignment",
              "legacy_clip_reference": clip_reference,
              "assumptions": ["group identities and within/between-group length correlations are generated, not recovered",
                              "stop reasons use the cap rule; actual EOS causes are not recovered",
                              "behavior versions index one synchronous policy publication per outer step; optimizer minibatch count remains unresolved",
                              "missing historical sampling overrides remain null; no defaults inferred",
                              "legacy clip_ratio is matched at the explicitly selected reference width; it is not assumed to be true truncation",
                              "GPU stage times copied only as observations; no wafer time or per-trajectory rewards inferred"],
              "not_available": ["real token IDs", "per-trajectory reward/logprob", "true shape distribution/correlations", "learning quality"]}
    return records, report
