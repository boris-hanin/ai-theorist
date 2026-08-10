from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

from .api import serve
from .batch_campaigns import (
    run_constant_tpp_campaign,
    run_quadratic_calibration,
    run_transformer_batch_census,
)
from .batch_scaling import (
    OptimizerHyperparameters,
    TransferContext,
    apply_transfer_rule,
    transfer_rule_registry,
)
from .critical_batch import CriticalBatchEstimate
from .campaign_jobs import run_campaign_job
from .forecast_campaigns import compile_real_text_scaling_plan
from .forecast_fleet import (
    aggregate_forecast_fleet_cache,
    assign_forecast_fleet_tasks,
    build_forecast_fleet_tasks,
    run_forecast_fleet_shard,
    select_forecast_fleet_learning_rate,
)
from .horizon_campaigns import run_horizon_transfer_campaign
from .joint_transfer_campaigns import run_joint_transfer_campaign
from .pretraining import compile_standard_pretraining_plan
from .public_corpora import PublicCorpusSpec, materialize_public_corpus
from .schema import StudySpec, compile_plan, default_study_spec
from .seesaw import SchedulePoint, compile_seesaw_schedule
from .study import atomic_write_json, run_study
from .training import train_trial
from .tuning import (
    MOE_TABLE1_ADAM,
    NUGPT_MID_ALIGNMENT,
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


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_and_print(payload: object, output: Optional[Path]) -> None:
    if output is not None:
        atomic_write_json(output, payload)
    _print(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and validate explicit-budget neural scaling laws")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample-spec", help="write a complete example study")
    sample.add_argument("path", type=Path)
    sample.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    sample.add_argument(
        "--architecture",
        choices=("pre_norm_mlp", "pre_norm_moe", "normalized_transformer"),
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

    subparsers.add_parser("batch-rules", help="list batch/duration transfer rules")

    transfer = subparsers.add_parser(
        "batch-transfer", help="apply one inspectable optimizer transfer rule"
    )
    transfer.add_argument("config", type=Path)

    quadratic = subparsers.add_parser(
        "batch-quadratic", help="calibrate critical-batch estimators on a noisy quadratic"
    )
    quadratic.add_argument("config", type=Path)
    quadratic.add_argument("--output", type=Path)

    census = subparsers.add_parser(
        "batch-census", help="run an SGD/Adam normalized-Transformer batch census"
    )
    census.add_argument("config", type=Path)
    census.add_argument("--device", default="cpu")
    census.add_argument("--output", type=Path)

    tpp = subparsers.add_parser(
        "batch-tpp", help="run a held-out constant-tokens-per-parameter campaign"
    )
    tpp.add_argument("config", type=Path)
    tpp.add_argument("--device", default="cpu")
    tpp.add_argument("--output", type=Path)

    horizon = subparsers.add_parser(
        "horizon-transfer",
        help="calibrate LR schedules across token horizons with a frozen holdout",
    )
    horizon.add_argument("config", type=Path)
    horizon.add_argument("--device", default="cpu")
    horizon.add_argument("--output", type=Path)
    horizon.add_argument("--progress-jsonl", action="store_true")

    joint = subparsers.add_parser(
        "joint-transfer",
        help="test frozen token-horizon and batch rules at a doubly held-out corner",
    )
    joint.add_argument("config", type=Path)
    joint.add_argument("--device", default="cpu")
    joint.add_argument("--output", type=Path)

    seesaw = subparsers.add_parser(
        "batch-seesaw", help="compile a Seesaw schedule after qualification"
    )
    seesaw.add_argument("config", type=Path)
    seesaw.add_argument("--output", type=Path)

    pretrain_plan = subparsers.add_parser(
        "pretrain-plan", help="validate a real-text Transformer pretraining census"
    )
    pretrain_plan.add_argument("config", type=Path)

    pretrain = subparsers.add_parser(
        "pretrain-census",
        help="run real-text batch scaling with bf16/FlashAttention/FSDP support",
    )
    pretrain.add_argument("config", type=Path)
    pretrain.add_argument("--device", default="cpu")
    pretrain.add_argument(
        "--output", type=Path, default=Path("runs/autoscaler/pretraining-census")
    )
    pretrain.add_argument("--progress-jsonl", action="store_true")

    forecast_plan = subparsers.add_parser(
        "forecast-plan",
        help="validate a pinned real-text Jiang or nGPT scaling ladder",
    )
    forecast_plan.add_argument("config", type=Path)

    forecast = subparsers.add_parser(
        "forecast-ladder",
        help="run a forecast-gated real-text Jiang or nGPT scaling ladder",
    )
    forecast.add_argument("config", type=Path)
    forecast.add_argument("--device", default="cpu")
    forecast.add_argument(
        "--output", type=Path, default=Path("runs/autoscaler/forecast-ladder")
    )
    forecast.add_argument("--progress-jsonl", action="store_true")

    forecast_shard = subparsers.add_parser(
        "forecast-shard",
        help="run one independently schedulable shard of a forecast campaign",
    )
    forecast_shard.add_argument("config", type=Path)
    forecast_shard.add_argument("--phase", choices=("tune", "ladder"), required=True)
    forecast_shard.add_argument("--shard-index", type=int, required=True)
    forecast_shard.add_argument("--shard-count", type=int, required=True)
    forecast_shard.add_argument("--selected-learning-rate", type=float)
    forecast_shard.add_argument("--task-id", action="append")
    forecast_shard.add_argument("--device", default="cuda")
    forecast_shard.add_argument("--output", type=Path, required=True)
    forecast_shard.add_argument("--progress-jsonl", action="store_true")

    forecast_tasks = subparsers.add_parser(
        "forecast-tasks",
        help="compile deterministic FLOP-balanced forecast fleet assignments",
    )
    forecast_tasks.add_argument("config", type=Path)
    forecast_tasks.add_argument("--phase", choices=("tune", "ladder"), required=True)
    forecast_tasks.add_argument("--shard-count", type=int, required=True)
    forecast_tasks.add_argument("--selected-learning-rate", type=float)

    forecast_select = subparsers.add_parser(
        "forecast-select",
        help="select the reference learning rate from complete fleet caches",
    )
    forecast_select.add_argument("config", type=Path)
    forecast_select.add_argument(
        "--cache-directory", type=Path, action="append", required=True
    )
    forecast_select.add_argument("--output", type=Path)

    forecast_aggregate = subparsers.add_parser(
        "forecast-aggregate",
        help="validate and aggregate complete fleet trial caches",
    )
    forecast_aggregate.add_argument("config", type=Path)
    forecast_aggregate.add_argument(
        "--cache-directory", type=Path, action="append", required=True
    )
    forecast_aggregate.add_argument("--output", type=Path, required=True)

    corpus = subparsers.add_parser(
        "corpus-materialize",
        help="freeze an allow-listed public text corpus for reproducible experiments",
    )
    corpus.add_argument("config", type=Path)
    corpus.add_argument(
        "--output-root", type=Path, default=Path("runs/autoscaler/public-corpora")
    )
    corpus.add_argument("--progress-jsonl", action="store_true")

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
                    "parameterization_control": result["parameterization_control"],
                    "routing_quality": result["routing_quality"],
                    "normalization_quality": result["normalization_quality"],
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
            NUGPT_MID_ALIGNMENT
            if screen_spec.architecture.block_type == "normalized_transformer"
            else MOE_TABLE1_ADAM
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
    elif args.command == "batch-rules":
        _print(transfer_rule_registry())
    elif args.command == "batch-transfer":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("batch-transfer config must be an object")
        source = OptimizerHyperparameters(**payload["optimizer"])
        context = TransferContext(**payload["context"])
        _print(
            apply_transfer_rule(
                payload["rule"],
                source,
                context,
                horizon_exponent=float(payload.get("horizon_exponent", 0.32)),
            ).to_dict()
        )
    elif args.command == "batch-quadratic":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("batch-quadratic config must be an object")
        _write_and_print(run_quadratic_calibration(payload), args.output)
    elif args.command == "batch-census":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("batch-census config must be an object")
        _write_and_print(
            run_transformer_batch_census(payload, device=args.device), args.output
        )
    elif args.command == "batch-tpp":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("batch-tpp config must be an object")
        _write_and_print(run_constant_tpp_campaign(payload, device=args.device), args.output)
    elif args.command == "horizon-transfer":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("horizon-transfer config must be an object")
        progress = None
        if args.progress_jsonl:
            def progress(event):
                print(json.dumps(event, sort_keys=True), flush=True)
        _write_and_print(
            run_horizon_transfer_campaign(
                payload, device=args.device, progress=progress
            ),
            args.output,
        )
    elif args.command == "joint-transfer":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("joint-transfer config must be an object")
        _write_and_print(
            run_joint_transfer_campaign(payload, device=args.device), args.output
        )
    elif args.command == "batch-seesaw":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("batch-seesaw config must be an object")
        estimate_payload = dict(payload["critical_batch_consensus"])
        estimate_payload["refusal_reasons"] = tuple(
            estimate_payload.get("refusal_reasons", ())
        )
        estimate = CriticalBatchEstimate(**estimate_payload)
        schedule = [SchedulePoint(**point) for point in payload["baseline_schedule"]]
        result = compile_seesaw_schedule(
            schedule,
            initial_batch_tokens=int(payload["initial_batch_tokens"]),
            critical_batch_consensus=estimate,
            variance_dominated=bool(payload["variance_dominated"]),
            safety_fraction=float(payload.get("safety_fraction", 0.8)),
            maximum_single_cut=float(payload.get("maximum_single_cut", 4.0)),
        )
        _write_and_print(result, args.output)
    elif args.command == "pretrain-plan":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("pretrain-plan config must be an object")
        _print(compile_standard_pretraining_plan(payload))
    elif args.command == "pretrain-census":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("pretrain-census config must be an object")
        progress = None
        if args.progress_jsonl:
            def progress(event):
                print(json.dumps(event, sort_keys=True), flush=True)
        result = run_campaign_job(
            "standard_pretraining_census",
            payload,
            device=args.device,
            output_dir=args.output,
            progress=progress,
        )
        _print(result)
    elif args.command == "forecast-plan":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("forecast-plan config must be an object")
        _print(compile_real_text_scaling_plan(payload))
    elif args.command == "forecast-ladder":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("forecast-ladder config must be an object")
        progress = None
        if args.progress_jsonl:
            def progress(event):
                print(json.dumps(event, sort_keys=True), flush=True)
        result = run_campaign_job(
            "real_text_scaling_ladder",
            payload,
            device=args.device,
            output_dir=args.output,
            progress=progress,
        )
        _print(result)
    elif args.command == "forecast-shard":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("forecast-shard config must be an object")
        progress = None
        if args.progress_jsonl:
            def progress(event):
                print(json.dumps(event, sort_keys=True), flush=True)
        _print(
            run_forecast_fleet_shard(
                payload,
                phase=args.phase,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                selected_learning_rate=args.selected_learning_rate,
                task_ids=args.task_id,
                output_directory=args.output,
                device=args.device,
                progress=progress,
            )
        )
    elif args.command == "forecast-tasks":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("forecast-tasks config must be an object")
        plan = compile_real_text_scaling_plan(payload)
        tasks = build_forecast_fleet_tasks(
            plan,
            phase=args.phase,
            selected_learning_rate=args.selected_learning_rate,
            run_negative_control=bool(payload.get("run_negative_control", True)),
        )
        assignments = assign_forecast_fleet_tasks(tasks, args.shard_count)
        _print(
            {
                "plan_fingerprint": plan["fingerprint"],
                "phase": args.phase,
                "shard_count": args.shard_count,
                "assignments": [
                    {
                        "shard_index": index,
                        "estimated_flops": sum(
                            task.estimated_flops for task in rows
                        ),
                        "tasks": [task.to_dict() for task in rows],
                    }
                    for index, rows in enumerate(assignments)
                ],
            }
        )
    elif args.command == "forecast-select":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("forecast-select config must be an object")
        _write_and_print(
            select_forecast_fleet_learning_rate(
                payload, args.cache_directory
            ),
            args.output,
        )
    elif args.command == "forecast-aggregate":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("forecast-aggregate config must be an object")
        _print(
            aggregate_forecast_fleet_cache(
                payload,
                cache_directories=args.cache_directory,
                output_directory=args.output,
            )
        )
    elif args.command == "corpus-materialize":
        payload = _read_json(args.config)
        if not isinstance(payload, dict):
            raise ValueError("corpus-materialize config must be an object")
        progress = None
        if args.progress_jsonl:
            def progress(event):
                print(json.dumps(event, sort_keys=True), flush=True)
        _print(
            materialize_public_corpus(
                PublicCorpusSpec.from_dict(payload),
                args.output_root,
                progress,
            )
        )


if __name__ == "__main__":
    main()
