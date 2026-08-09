from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import serve
from .schema import StudySpec, compile_plan, default_study_spec
from .study import atomic_write_json, run_study
from .training import train_trial
from .tuning import (
    MOE_TABLE1_ADAM,
    STANDARD_RESIDUAL_MLP,
    raw_learning_rate_from_normalized_eta,
    summarize_trials,
)


def _read_spec(path: Path) -> StudySpec:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return StudySpec.from_dict(payload)


def _print(payload: object) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and validate fixed-horizon neural scaling laws")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample-spec", help="write a complete example study")
    sample.add_argument("path", type=Path)
    sample.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    sample.add_argument(
        "--architecture",
        choices=("pre_norm_mlp", "pre_norm_moe"),
        default="pre_norm_mlp",
    )
    sample.add_argument("--quick", action="store_true")

    plan = subparsers.add_parser("plan", help="validate and compile a study without training")
    plan.add_argument("spec", type=Path)

    run = subparsers.add_parser("run", help="execute a study")
    run.add_argument("spec", type=Path)
    run.add_argument("--device", default="cpu")
    run.add_argument("--output", type=Path, default=Path("runs/autoscaler/manual"))
    run.add_argument("--summary", action="store_true", help="print a compact result instead of every trial")
    run.add_argument("--progress-jsonl", action="store_true", help="stream machine-readable progress events")

    screen = subparsers.add_parser(
        "screen", help="measure one fixed normalized eta across the full scale ladder"
    )
    screen.add_argument("spec", type=Path)
    screen.add_argument("--normalized-eta", type=float, required=True)
    screen.add_argument("--device", default="cpu")
    screen.add_argument("--steps", type=int)
    screen.add_argument("--seeds", type=int, nargs="+")
    screen.add_argument("--output", type=Path)

    server = subparsers.add_parser("serve", help="start the local product API")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8787)
    server.add_argument("--run-root", type=Path, default=Path("runs/autoscaler"))

    args = parser.parse_args()
    if args.command == "sample-spec":
        atomic_write_json(
            args.path,
            default_study_spec(
                args.optimizer,
                args.quick,
                block_type=args.architecture,
            ).to_dict(),
        )
    elif args.command == "plan":
        _print(compile_plan(_read_spec(args.spec)))
    elif args.command == "run":
        progress = None
        if args.progress_jsonl:
            def progress(event):
                print(json.dumps(event, sort_keys=True), flush=True)
        result = run_study(
            _read_spec(args.spec),
            device=args.device,
            output_dir=args.output,
            progress=progress,
        )
        if args.summary:
            trials = result["trials"]
            _print(
                {
                    "status": result["status"],
                    "forecastable": result["forecastable"],
                    "refusal_reasons": result["refusal_reasons"],
                    "tuning": result["tuning"],
                    "transfer_checks": result["transfer_checks"],
                    "negative_control": result["negative_control"],
                    "routing_quality": result["routing_quality"],
                    "scaling_law": result["scaling_law"],
                    "final_scaling_law": result["final_scaling_law"],
                    "holdout_calibration": result["holdout_calibration"],
                    "warnings": result["warnings"],
                    "next_scale_forecast": result["next_scale_forecast"],
                    "trial_count": len(trials),
                    "total_trial_seconds": sum(trial["duration_seconds"] for trial in trials),
                    "maximum_peak_memory_bytes": max(trial["peak_memory_bytes"] for trial in trials),
                }
            )
        else:
            _print(result)
    elif args.command == "screen":
        with args.spec.open("r", encoding="utf-8") as handle:
            screen_payload = json.load(handle)
        if args.steps is not None:
            screen_payload["horizon"]["steps"] = args.steps
        if args.seeds is not None:
            screen_payload["seeds"] = args.seeds
        screen_spec = StudySpec.from_dict(screen_payload)
        parameterization = (
            MOE_TABLE1_ADAM
            if screen_spec.architecture.block_type == "pre_norm_moe"
            else STANDARD_RESIDUAL_MLP
        )
        rows = []
        for scale in screen_spec.scales:
            raw_rate = raw_learning_rate_from_normalized_eta(
                parameterization,
                screen_spec.optimizer.name,
                args.normalized_eta,
                width=scale.width,
                depth=scale.repeats,
            )
            trials = []
            for seed in screen_spec.seeds:
                print(
                    json.dumps({"scale": scale.name, "seed": seed, "status": "running"}),
                    flush=True,
                )
                trials.append(
                    train_trial(
                        screen_spec,
                        scale,
                        args.normalized_eta,
                        seed,
                        raw_learning_rate=raw_rate,
                        device=args.device,
                    )
                )
            summary = summarize_trials(trials, args.normalized_eta)
            routing_imbalances = [
                trial.max_routing_load_imbalance
                for trial in trials
                if trial.max_routing_load_imbalance is not None
            ]
            rows.append(
                {
                    "scale": scale.name,
                    "width": scale.width,
                    "repeats": scale.repeats,
                    "expert_width": scale.expert_width,
                    "normalized_learning_rate": args.normalized_eta,
                    "raw_learning_rate": raw_rate,
                    "raw_learning_rates": trials[0].raw_learning_rates,
                    "mean_final_validation_loss": summary.mean_final_validation_loss,
                    "sem_final_validation_loss": summary.sem_final_validation_loss,
                    "losses_by_seed": summary.losses_by_seed,
                    "maximum_routing_load_imbalance": (
                        max(routing_imbalances) if routing_imbalances else None
                    ),
                    "mean_routing_load_imbalance": (
                        sum(routing_imbalances) / len(routing_imbalances)
                        if routing_imbalances
                        else None
                    ),
                    "trial_seconds": sum(trial.duration_seconds for trial in trials),
                }
            )
        payload = {
            "study_fingerprint": screen_spec.fingerprint,
            "steps": screen_spec.horizon.steps,
            "normalized_learning_rate": args.normalized_eta,
            "learning_rate_coordinate": "normalized_eta",
            "optimizer_parameterization": parameterization,
            "device": args.device,
            "scale_results": rows,
        }
        if args.output is not None:
            atomic_write_json(args.output, payload)
        _print(payload)
    elif args.command == "serve":
        serve(args.host, args.port, args.run_root)


if __name__ == "__main__":
    main()
