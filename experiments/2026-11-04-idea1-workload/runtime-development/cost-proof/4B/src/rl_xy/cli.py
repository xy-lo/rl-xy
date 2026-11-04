"""Single command-line entry point for every supported rl-xy run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .run import run_toy_case, run_transfer_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlxy",
        description="AstraSim-backed wafer-scale RL simulation MVP",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    wafer=subparsers.add_parser('run-wafer',help='native DP x TP2 mesh with private multi-core SRAM')
    wafer.add_argument('--case',required=True,type=Path)
    wafer.add_argument('--scenario',required=True,type=Path)
    wafer.add_argument('--out',required=True,type=Path)
    gpu=subparsers.add_parser('run-gpu',help='independent GPU CTA/SM and node-switch-spine simulation')
    gpu.add_argument('--case',required=True,type=Path)
    gpu.add_argument('--scenario',required=True,type=Path,help='shared logical host and original wafer reference scenario')
    gpu.add_argument('--gpu-profile',required=True,type=Path)
    gpu.add_argument('--out',required=True,type=Path)
    for command in (wafer,gpu):
        command.add_argument('--colocation',type=Path,help='pinned global synchronous stage and TP-layout contract')

    aligned=subparsers.add_parser('run-aligned',help='selected PP1 distributed-optimizer veRL protocol; conditional native evidence')
    aligned.add_argument('--case',required=True,type=Path)
    aligned.add_argument('--scenario',required=True,type=Path)
    aligned.add_argument('--policy',required=True,type=Path)
    aligned.add_argument('--gpu-profile',type=Path)
    aligned.add_argument('--active',type=int,default=48)
    aligned.add_argument('--out',required=True,type=Path)
    aligned.add_argument('--contract-local-chains',action='store_true',help='whole-graph contraction is currently rejected; component contraction is enabled by default')
    aligned.add_argument('--conservative-external-reads',action='store_true',help='diagnostic: retain redundant immutable carrier reads for exact native A/B')
    aligned.add_argument('--uncontracted-components',action='store_true',help='diagnostic full component graph for native equivalence checks')
    aligned.add_argument('--legacy-resource-read-ids',action='store_true',help='diagnostic: encode native read deltas with original string IDs for exact A/B')

    hardware = subparsers.add_parser("inspect-hardware", help="validate source-bearing H and report backend readiness")
    hardware.add_argument("--profile", required=True, type=Path)
    hardware.add_argument("--require-executable", action="store_true")
    hardware.add_argument("--out", type=Path)
    trace = subparsers.add_parser("import-verl-trace", help="normalize veRL L1 lengths without inventing missing tensors")
    trace.add_argument("--trace", required=True, type=Path)
    trace.add_argument("--origin", required=True, choices=("synthetic_fixture", "framework_capture", "statistics_constrained_synthetic"))
    trace.add_argument("--out", required=True, type=Path)
    model = subparsers.add_parser("inspect-hf-model", help="inspect local Qwen2 shapes and backend gaps")
    model.add_argument("--config", required=True, type=Path)
    model.add_argument("--out", type=Path)
    migration = subparsers.add_parser("preflight-migration", help="join veRL lengths, actual model metadata and formal H")
    migration.add_argument("--trace", required=True, type=Path)
    migration.add_argument("--origin", required=True, choices=("synthetic_fixture", "framework_capture", "statistics_constrained_synthetic"))
    migration.add_argument("--hardware", required=True, type=Path)
    migration.add_argument("--model-config", required=True, type=Path)
    migration.add_argument("--out", required=True, type=Path)
    migration.add_argument("--require-executable", action="store_true")
    legacy = subparsers.add_parser("import-legacy-run", help="reuse actual historical veRL scalar step metrics")
    legacy.add_argument("--log", required=True, type=Path)
    legacy.add_argument("--overrides", required=True, type=Path)
    legacy.add_argument("--run-id", required=True)
    legacy.add_argument("--out", required=True, type=Path)
    synthetic = subparsers.add_parser("synthesize-workload", help="generate declared lengths constrained by observed GPU statistics")
    synthetic.add_argument("--statistics", required=True, type=Path)
    synthetic.add_argument("--seed", required=True, type=int)
    synthetic.add_argument("--first-step", required=True, type=int)
    synthetic.add_argument("--last-step", required=True, type=int)
    synthetic.add_argument("--clip-reference", required=True, choices=("public_cap", "observed_max"))
    synthetic.add_argument("--out", required=True, type=Path)

    transfer = subparsers.add_parser(
        "fixture-transfer", help="run the independent hand-checkable 21 ns fixture"
    )
    transfer.add_argument("--case", required=True, type=Path)
    transfer.add_argument("--out", required=True, type=Path)

    toy = subparsers.add_parser(
        "run", help="run a complete synthetic synchronous-GRPO round"
    )
    toy.add_argument("--case", required=True, type=Path)
    toy.add_argument("--out", required=True, type=Path)
    toy.add_argument("--rounds", type=int)
    toy.add_argument("--local-cost-model", choices=("legacy", "mapped"), default="legacy",
                     help="mapped adds uncalibrated SRAM/traffic-aware local kernel costs")
    toy.add_argument("--resource-model",action="store_true",help="mapped costs + live tensor and resource admission (patched image)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command=='run-aligned':
            from .aligned_run import run_aligned
            output=run_aligned(args.case,args.scenario,args.policy,args.out,gpu_profile_path=args.gpu_profile,
                active=args.active,contract_local_chains=args.contract_local_chains,compact_components=not args.uncontracted_components,conservative_bridge_reads=args.conservative_external_reads,
                intern_resource_read_ids=not args.legacy_resource_read_ids)
            print(f'rlxy: PASS: {output}')
            return 0
        if args.command in ('run-wafer','run-gpu'):
            from .wafer_run import run_wafer
            output=run_wafer(args.case,args.scenario,args.out,gpu_profile_path=args.gpu_profile if args.command=='run-gpu' else None,
                colocation_path=args.colocation)
            print(f'rlxy: PASS: {output}')
            return 0
        if args.command == "synthesize-workload":
            from .legacy_workload import synthesize_lengths
            from .verl_trace import load_verl_trace
            statistics = json.loads(args.statistics.read_text())
            records, report = synthesize_lengths(statistics, seed=args.seed,
                selected_steps=list(range(args.first_step, args.last_step + 1)), clip_reference=args.clip_reference)
            args.out.mkdir(parents=True, exist_ok=False)
            trace_path = args.out / "lengths.jsonl"
            trace_path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
            normalized = load_verl_trace(trace_path, origin="statistics_constrained_synthetic")
            (args.out / "normalized.json").write_text(json.dumps(normalized, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            (args.out / "synthesis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"rlxy: synthetic workload generated: {args.out}")
            return 0
        if args.command in {"inspect-hardware", "import-verl-trace", "inspect-hf-model", "preflight-migration", "import-legacy-run"}:
            if args.command == "inspect-hardware":
                from .hardware_profile import inspect_hardware_file
                report = inspect_hardware_file(args.profile)
            elif args.command == "import-verl-trace":
                from .verl_trace import load_verl_trace
                report = load_verl_trace(args.trace, origin=args.origin)
            elif args.command == "inspect-hf-model":
                from .model_import import inspect_hf_model
                report = inspect_hf_model(args.config)
            elif args.command == "preflight-migration":
                from .migration import preflight_migration
                report = preflight_migration(trace_path=args.trace, trace_origin=args.origin,
                    hardware_path=args.hardware, model_path=args.model_config)
            else:
                from .legacy_workload import import_legacy_run
                report = import_legacy_run(args.log, args.overrides, args.run_id)
            payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with args.out.open("x", encoding="utf-8") as stream:
                    stream.write(payload)
                print(f"rlxy: inspected: {args.out}")
            else:
                print(payload, end="")
            if args.command == "inspect-hardware" and args.require_executable and not report["current_full_grpo_executable"]:
                print("rlxy: hardware execution blocked; see explicit parameter/backend blockers", file=sys.stderr)
                return 2
            if args.command == "preflight-migration" and args.require_executable and not report["execution_ready"]:
                print("rlxy: migration execution blocked; see explicit blockers", file=sys.stderr)
                return 2
            return 0
        if args.command == "fixture-transfer":
            output = run_transfer_fixture(args.case, args.out)
        else:
            output = run_toy_case(args.case, args.out, rounds_override=args.rounds,
                                  local_cost_model=args.local_cost_model,resource_model=args.resource_model)
    except Exception as exc:
        print(f"rlxy: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"rlxy: PASS: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
