"""Join W, H and model evidence for admission, without substituting toy inputs."""
from __future__ import annotations

from .hardware_profile import inspect_hardware_file, load_hardware_profile
from .model_import import inspect_hf_model
from .verl_trace import load_verl_trace


def preflight_migration(*, trace_path, trace_origin, hardware_path, model_path) -> dict:
    workload = load_verl_trace(trace_path, origin=trace_origin)
    hardware = inspect_hardware_file(hardware_path)
    model = inspect_hf_model(model_path)
    if max(workload["run_meta"]["algo"]["max_prompt_tokens"] + workload["run_meta"]["algo"]["max_response_tokens"],
           max(t["prompt_len"] + t["resp_len"] for s in workload["steps"] for g in s["groups"] for t in g["trajectories"])) > model["config"]["max_position_embeddings"]:
        raise ValueError("trace public maximum exceeds model context support")
    h = load_hardware_profile(hardware_path)
    total = hardware["aggregates"]["dram_bytes"]
    required = model["bf16_adamw_resident_bytes_with_reference_and_gradient"]
    # This is only a necessary condition. A feasible total can still have no
    # legal placement, e.g. indivisible tensors or insufficient local SRAM.
    aggregate_fit = None if total is None else required <= total
    compatible_tp = []
    cfg = model["config"]
    for tp in range(1, min(cfg["num_attention_heads"], hardware["aggregates"]["modules"] or 0) + 1):
        if all(cfg[k] % tp == 0 for k in ("num_attention_heads", "num_key_value_heads", "hidden_size", "intermediate_size", "vocab_size")):
            compatible_tp.append(tp)
    return {
        "schema_version": "rlxy.migration-preflight/1",
        "workload_summary": {"run_id": workload["run_meta"]["run_id"], "origin_declared": workload["origin_declared"],
                             "source_sha256": workload["source_sha256"], "steps": len(workload["steps"]),
                             "trajectories": sum(len(g["trajectories"]) for s in workload["steps"] for g in s["groups"]),
                             "available": workload["available"], "warnings": workload["warnings"]},
        "hardware": hardware, "model": model,
        "state_capacity_necessary_condition": {
            "resident_bytes": required, "total_dram_bytes": total, "aggregate_fit": aggregate_fit,
            "legal_placement_proven": False,
            "scope": "state-only lower bound; no DP replicas, activations, KV, communication buffers or workspace"},
        "uniform_head_sharding_tp_candidates": compatible_tp,
        "tp_candidates_scope": "divisibility only; no topology/PP/DP placement selected",
        "execution_ready": False,
        "blockers": hardware["parameter_blockers"] + hardware["backend_blockers"] + model["backend_blockers"] + [
            "L1 input has no full algorithm/update/minibatch/checkpoint contract",
            "formal H multi-core runtime/placement not implemented; selected groups can be compiled only to an explicit TP2 probe"],
        "model_identity_verified_against_capture": False,
        "scope": "inspectable W/H/model admission input; no simulated wafer time produced",
    }
