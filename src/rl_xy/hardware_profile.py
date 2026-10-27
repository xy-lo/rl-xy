"""Source-bearing wafer design contract and execution-readiness checks.

Design parameters are separate from measured costs and backend capabilities.
Missing values remain null; aggregate capacity never merges per-core SRAM.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "rlxy.hardware/1"
SCHEMA_V2 = "rlxy.hardware/2"
SCHEMA_V3 = "rlxy.hardware/3"
GEOMETRY_UNITS = {
    "layout_width_mm": "mm", "layout_height_mm": "mm",
    "array_width_mm": "mm", "array_height_mm": "mm",
    "compute_die_width_mm": "mm", "compute_die_height_mm": "mm",
}
UNITS = {
    "wafer_diameter_mm": "mm", "module_width_mm": "mm", "module_height_mm": "mm",
    "module_compute_area_mm2": "mm2", "modules_x": "count", "modules_y": "count",
    "cores_per_module": "count", "core_clock_ghz": "GHz",
    "core_fp16_flops_per_ns": "FLOP/ns", "core_bf16_flops_per_ns": "FLOP/ns",
    "core_vector_ops_per_ns": "op/ns", "sram_bytes_per_core": "B",
    "sram_bytes_per_ns_per_core": "B/ns", "dram_bytes_per_module": "B",
    "dram_bytes_per_ns_per_module": "B/ns", "noc_bytes_per_ns_per_module": "B/ns",
    "d2d_bytes_per_ns": "B/ns", "d2d_latency_ns": "ns", "packet_bytes": "B",
    "nic_tx_bytes_per_module": "B", "nic_rx_bytes_per_module": "B",
    "host_bytes_per_ns": "B/ns", "host_latency_ns": "ns",
}
COUNTS = {k for k, u in UNITS.items() if u in ("count", "B")}
STATUSES = {"design_candidate", "test_assumption", "measured", "unresolved"}


class HardwareProfileError(ValueError):
    pass


def _exact(obj, fields, label):
    if not isinstance(obj, dict) or set(obj) != set(fields):
        raise HardwareProfileError(f"{label}: fields must be {sorted(fields)}")


def load_hardware_profile(path: str | Path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text())
    _exact(raw, {"schema_version", "profile_id", "topology", "compute_dtype", "d2d_caliber",
                 "parameters", "provenance", "notes"} |
           ({"declared_geometry"} if raw.get("schema_version") in (SCHEMA_V2,SCHEMA_V3) else set()) |
           ({'geometry_scope'} if raw.get('schema_version')==SCHEMA_V3 else set()), "hardware")
    if raw["schema_version"] not in (SCHEMA, SCHEMA_V2, SCHEMA_V3) or not isinstance(raw["profile_id"], str) or not raw["profile_id"]:
        raise HardwareProfileError("invalid hardware identity/schema")
    if raw["topology"] != "mesh2d_xy" or raw["compute_dtype"] not in ("fp16", "bf16"):
        raise HardwareProfileError("explicit mesh2d_xy and fp16/bf16 required")
    if raw["d2d_caliber"] not in ("per_edge_per_direction", "unspecified"):
        raise HardwareProfileError("unknown D2D caliber")
    _exact(raw["parameters"], UNITS, "parameters")
    for key, field in raw["parameters"].items():
        _exact(field, {"value", "unit", "source", "status"}, key)
        value = field["value"]
        if field["unit"] != UNITS[key] or field["status"] not in STATUSES:
            raise HardwareProfileError(f"{key}: unit/status mismatch")
        if not isinstance(field["source"], str) or not field["source"].strip():
            raise HardwareProfileError(f"{key}: source required")
        if value is None:
            if field["status"] != "unresolved":
                raise HardwareProfileError(f"{key}: null must be unresolved")
        elif (field["status"] == "unresolved" or type(value) not in (int, float)
              or not math.isfinite(value) or value <= 0 or (key in COUNTS and type(value) is not int)):
            raise HardwareProfileError(f"{key}: invalid positive value/status")
    if not isinstance(raw["notes"], list) or not isinstance(raw["provenance"], dict):
        raise HardwareProfileError("notes/provenance required")
    if raw['schema_version']==SCHEMA_V3 and raw['geometry_scope']!='reference_template_not_case3_placement':
        raise HardwareProfileError('hardware/3 requires explicit reference-template geometry scope')
    if raw["schema_version"] in (SCHEMA_V2,SCHEMA_V3):
        _exact(raw["declared_geometry"], GEOMETRY_UNITS, "declared_geometry")
        for key, field in raw["declared_geometry"].items():
            _exact(field, {"value", "unit", "source", "status"}, key)
            if (field["unit"] != GEOMETRY_UNITS[key] or field["status"] not in STATUSES - {"unresolved"}
                    or type(field["value"]) not in (int, float) or not math.isfinite(field["value"])
                    or field["value"] <= 0 or not isinstance(field["source"], str) or not field["source"].strip()):
                raise HardwareProfileError(f"{key}: invalid geometry/source")
    return raw


def inspect_hardware(profile: dict) -> dict:
    p = {k: v["value"] for k, v in profile["parameters"].items()}
    product = lambda *names: None if any(p[n] is None for n in names) else math.prod(p[n] for n in names)
    modules = product("modules_x", "modules_y")
    aggregates = {"modules": modules}
    for name, field in (("cores", "cores_per_module"), ("dram_bytes", "dram_bytes_per_module"),
                        ("dram_bytes_per_ns_sum", "dram_bytes_per_ns_per_module")):
        aggregates[name] = None if modules is None or p[field] is None else modules * p[field]
    cores = aggregates["cores"]
    aggregates["sram_bytes_sum_not_shared_pool"] = (
        None if cores is None or p["sram_bytes_per_core"] is None else cores * p["sram_bytes_per_core"])
    for dtype in ("fp16", "bf16"):
        rate = p[f"core_{dtype}_flops_per_ns"]
        aggregates[f"peak_{dtype}_flops_per_ns"] = None if cores is None or rate is None else cores * rate
    geometric = ("wafer_diameter_mm", "module_width_mm", "module_height_mm", "modules_x", "modules_y")
    geometry_errors = []
    geometry_details = {}
    if profile["schema_version"] in (SCHEMA_V2,SCHEMA_V3):
        g = {k: v["value"] for k, v in profile["declared_geometry"].items()}
        if all(p[n] is not None for n in geometric[1:]):
            width, height = p["modules_x"] * p["module_width_mm"], p["modules_y"] * p["module_height_mm"]
            rotated = (p["modules_x"] * p["module_height_mm"], p["modules_y"] * p["module_width_mm"])
            geometry_details = {"unrotated_grid_mm": [width, height], "rotated_modules_grid_mm": list(rotated),
                "declared_array_mm": [g["array_width_mm"], g["array_height_mm"]],
                "modules_area_mm2": modules * p["module_width_mm"] * p["module_height_mm"],
                "declared_array_area_mm2": g["array_width_mm"] * g["array_height_mm"]}
            if not any(math.isclose(w, g["array_width_mm"], abs_tol=1e-6) and
                       math.isclose(h, g["array_height_mm"], abs_tol=1e-6) for w, h in ((width,height), rotated)):
                geometry_errors.append("declared array footprint does not match fixed-size module grid in either orientation")
            if not any(w <= g["layout_width_mm"] + 1e-6 and h <= g["layout_height_mm"] + 1e-6
                       for w, h in ((width,height), rotated)):
                geometry_errors.append("fixed-size module grid does not fit declared layout in either orientation")
            if geometry_details["modules_area_mm2"] > geometry_details["declared_array_area_mm2"] + 1e-6:
                geometry_errors.append("module area sum exceeds declared module-array area")
            if g["compute_die_width_mm"] * g["compute_die_height_mm"] > p["module_width_mm"] * p["module_height_mm"]:
                geometry_errors.append("compute die area exceeds module area")
        else:
            geometry_errors.append("module grid geometry incomplete")
        # Do not import a 300mm wafer or 26x33mm exposure constraint that is not
        # part of this user's table. V1 keeps its historical audit unchanged.
    elif all(p[n] is not None for n in geometric):
        width, height = p["modules_x"] * p["module_width_mm"], p["modules_y"] * p["module_height_mm"]
        if math.hypot(width, height) > p["wafer_diameter_mm"] + 1e-6:
            geometry_errors.append("centered rectangular module grid exceeds wafer circle")
        if max(p["module_width_mm"], p["module_height_mm"]) > 33 + 1e-6 or min(p["module_width_mm"], p["module_height_mm"]) > 26 + 1e-6:
            geometry_errors.append("module exceeds declared 26x33mm single-exposure field")
        area = p["module_compute_area_mm2"]
        if area is not None and area > p["module_width_mm"] * p["module_height_mm"]:
            geometry_errors.append("compute area exceeds module footprint")
    else:
        geometry_errors.append("module/wafer geometry incomplete")
    geometry_status='checked_fixed_module_grid'
    if profile['schema_version']==SCHEMA_V3:
        geometry_details['template_only_conflicts_not_case3_proof']=geometry_errors
        geometry_errors=[]
        geometry_status='reference_template_only_case3_placement_unknown'
    required = set(UNITS) - {"core_fp16_flops_per_ns", "core_bf16_flops_per_ns", "core_clock_ghz"}
    if profile['schema_version']==SCHEMA_V3:
        required-= {'wafer_diameter_mm','module_width_mm','module_height_mm','module_compute_area_mm2'}
    required.add(f"core_{profile['compute_dtype']}_flops_per_ns")
    missing = sorted(k for k in required if p[k] is None)
    blockers = [f"unresolved:{k}" for k in missing] + geometry_errors
    if profile["d2d_caliber"] != "per_edge_per_direction":
        blockers.append("D2D bandwidth lacks per-edge/per-direction caliber")
    backend = []
    if modules != 2:
        backend.append("direct run requires two TP ranks; hierarchical run-wafer requires a separate explicit DP/mesh scenario")
    if p["cores_per_module"] != 1:
        backend.append("multi-core mapping requires explicit execution rates; detailed NoC packet/bank calibration is unavailable")
    if profile["compute_dtype"] != "bf16":
        backend.append("current complete GRPO kernel template requires BF16 weights/activations")
    return {
        "schema_version": "rlxy.hardware-report/1", "profile_id": profile["profile_id"],
        "aggregates": aggregates, "geometry_errors": geometry_errors,
        "geometry_details": geometry_details,
        'geometry_status':geometry_status,'physical_geometry_validated':False,
        "unresolved_parameters": missing, "parameter_blockers": blockers,
        "backend_blockers": backend, "parameters_ready": not blockers,
        "current_full_grpo_executable": not blockers and not backend,
        "hardware_calibrated": False,
        "scope": "design contract and readiness; no chip execution or speedup claim",
    }


def inspect_hardware_file(path: str | Path) -> dict:
    result = inspect_hardware(load_hardware_profile(path))
    result["source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result
