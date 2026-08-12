from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .batch_scaling import BatchRunRecord
from .forecast_campaigns import (
    _mean_sem,
    _run_trial,
    _tuning_seeds_for_learning_rate,
    compile_real_text_scaling_plan,
    forecast_tokenized_text_spec,
    forecast_trial_cache_identity,
    run_real_text_scaling_campaign,
)
from .pretraining import (
    PretrainingRuntimeSpec,
    TokenizedTextCorpus,
    close_distributed,
    prepare_distributed,
)
from .study import atomic_write_json


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
FLEET_SCHEMA_VERSION = 1


def _tuning_tau_candidates(plan: Mapping[str, Any]) -> List[Optional[float]]:
    values: List[Optional[float]] = [
        float(value) for value in plan.get("weight_decay_tau_ema_grid", ())
    ]
    optimizer = plan.get("optimizer_contract", {})
    if isinstance(optimizer, Mapping) and bool(
        optimizer.get("include_zero_weight_decay_control", False)
    ):
        values = [None, *values]
    return values or [None]


@dataclass(frozen=True)
class ForecastFleetTask:
    task_id: str
    phase: str
    ordinal: int
    scale_name: str
    eta: float
    weight_decay_tau_ema: Optional[float]
    seed: int
    optimizer_mode: str
    estimated_flops: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "phase": self.phase,
            "ordinal": self.ordinal,
            "scale_name": self.scale_name,
            "eta": self.eta,
            "weight_decay_tau_ema": self.weight_decay_tau_ema,
            "seed": self.seed,
            "optimizer_mode": self.optimizer_mode,
            "estimated_flops": self.estimated_flops,
        }


def build_forecast_fleet_tasks(
    plan: Mapping[str, Any],
    *,
    phase: str,
    selected_learning_rate: Optional[float] = None,
    selected_weight_decay_tau_ema: Optional[float] = None,
    run_negative_control: bool = True,
) -> List[ForecastFleetTask]:
    """Build the physical trial DAG for one dependency-delimited phase."""

    if phase not in {"tune", "ladder"}:
        raise ValueError("forecast fleet phase must be tune or ladder")
    scales = [dict(row) for row in plan["scales"]]
    seeds = [int(value) for value in plan["seeds"]]
    reference_index = int(plan["architecture_contract"]["reference_scale_index"])
    reference = scales[reference_index]
    tasks: List[ForecastFleetTask] = []
    tau_grid = [float(value) for value in plan.get("weight_decay_tau_ema_grid", ())]
    zero_decay_allowed = bool(
        dict(plan.get("optimizer_contract", {})).get(
            "include_zero_weight_decay_control", False
        )
    )

    def append(
        scale: Mapping[str, Any],
        eta: float,
        tau_ema: Optional[float],
        seed: int,
        optimizer_mode: str,
    ) -> None:
        ordinal = len(tasks)
        tasks.append(
            ForecastFleetTask(
                task_id=(
                    f"{phase}-{scale['name']}-{optimizer_mode}-"
                    f"eta{eta:g}"
                    + (f"-tau{tau_ema:g}" if tau_ema is not None else "")
                    + f"-seed{seed}"
                ),
                phase=phase,
                ordinal=ordinal,
                scale_name=str(scale["name"]),
                eta=float(eta),
                weight_decay_tau_ema=tau_ema,
                seed=seed,
                optimizer_mode=optimizer_mode,
                estimated_flops=float(
                    6 * int(scale["parameters"]) * int(scale["presented_tokens"])
                ),
            )
        )

    if plan.get("run_profile") == "extension":
        if phase == "tune":
            raise ValueError("forecast extensions reuse the frozen parent LR and skip tuning")
        if run_negative_control:
            raise ValueError("forecast extensions refuse a wrong-LR control")
        if selected_learning_rate is None:
            raise ValueError("forecast extension requires selected_learning_rate")
        selected = float(selected_learning_rate)
        contract = dict(plan["extension_contract"])
        if selected != float(contract["selected_learning_rate"]):
            raise ValueError("forecast extension LR disagrees with its frozen contract")
        append(scales[-1], selected, None, seeds[0], "theory")
        return tasks

    frozen = plan.get("frozen_optimizer")
    if isinstance(frozen, Mapping):
        if phase == "tune":
            return tasks
        if run_negative_control:
            raise ValueError("frozen adaptive ladders refuse a wrong-LR control")
        expected_eta = float(frozen["selected_learning_rate"])
        expected_tau_raw = frozen.get("selected_weight_decay_tau_ema")
        expected_tau = (
            None if expected_tau_raw is None else float(expected_tau_raw)
        )
        if selected_learning_rate is None or not math.isclose(
            float(selected_learning_rate), expected_eta, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("ladder LR disagrees with the frozen CBS contract")
        if selected_weight_decay_tau_ema != expected_tau:
            raise ValueError("ladder tau_EMA disagrees with the frozen CBS contract")
        selected_scales = (
            [scales[-1]]
            if plan.get("run_profile") == "comparison"
            else scales
        )
        for scale in selected_scales:
            for seed in seeds:
                append(scale, expected_eta, expected_tau, seed, "theory")
        return tasks

    if phase == "tune":
        for eta in plan.get("tuning_task_learning_rates", plan["learning_rates"]):
            for tau_ema in _tuning_tau_candidates(plan):
                for seed in _tuning_seeds_for_learning_rate(plan, float(eta)):
                    append(reference, float(eta), tau_ema, seed, "theory")
        return tasks

    if selected_learning_rate is None:
        raise ValueError("ladder phase requires selected_learning_rate")
    selected = float(selected_learning_rate)
    if selected not in {float(value) for value in plan["learning_rates"]}:
        raise ValueError("selected_learning_rate is not in the preregistered grid")
    if tau_grid:
        if selected_weight_decay_tau_ema is None and not zero_decay_allowed:
            raise ValueError(
                "ladder phase requires selected_weight_decay_tau_ema for a "
                "jointly tuned plan"
            )
        selected_tau = (
            None
            if selected_weight_decay_tau_ema is None
            else float(selected_weight_decay_tau_ema)
        )
        if selected_tau is not None and selected_tau not in set(tau_grid):
            raise ValueError(
                "selected_weight_decay_tau_ema is not in the preregistered grid"
            )
    else:
        if selected_weight_decay_tau_ema is not None:
            raise ValueError(
                "selected_weight_decay_tau_ema requires a preregistered tau grid"
            )
        selected_tau = None
    if plan.get("run_profile") == "comparison":
        if run_negative_control:
            raise ValueError("CompleteP comparison refuses a wrong-LR control")
        for seed in seeds:
            append(scales[-1], selected, selected_tau, seed, "theory")
        return tasks
    # The selected reference trials already exist in the tuning phase. Reusing
    # those exact immutable cache entries avoids three duplicate GPU runs.
    for scale_index, scale in enumerate(scales):
        if scale_index == reference_index:
            completed_reference_seeds = set(
                _tuning_seeds_for_learning_rate(plan, selected)
            )
            for seed in seeds:
                if seed not in completed_reference_seeds:
                    append(scale, selected, selected_tau, seed, "theory")
            continue
        for seed in seeds:
            append(scale, selected, selected_tau, seed, "theory")
    if run_negative_control:
        for seed in seeds:
            append(scales[-1], selected, selected_tau, seed, "wrong_global")
    return tasks


def assign_forecast_fleet_tasks(
    tasks: Sequence[ForecastFleetTask], shard_count: int
) -> List[List[ForecastFleetTask]]:
    """Deterministically balance tasks by estimated training FLOPs."""

    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    assignments: List[List[ForecastFleetTask]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for task in sorted(tasks, key=lambda row: (-row.estimated_flops, row.ordinal)):
        shard = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[shard].append(task)
        loads[shard] += task.estimated_flops
    for rows in assignments:
        rows.sort(key=lambda row: row.ordinal)
    return assignments


def _load_corpus(
    config: Mapping[str, Any], plan: Mapping[str, Any]
) -> TokenizedTextCorpus:
    corpus = TokenizedTextCorpus(
        forecast_tokenized_text_spec(config),
        context_length=int(config["architecture"]["context_length"]),
        vocab_size=int(config["architecture"]["vocab_size"]),
    )
    if not corpus.tokenizer_is_pinned:
        raise ValueError("forecast fleet requires pinned tokenizer provenance")
    if corpus.identity_fingerprint != plan["dataset_identity"]["fingerprint"]:
        raise ValueError("compiled plan and loaded token stream identity disagree")
    return corpus


def run_forecast_fleet_shard(
    config: Mapping[str, Any],
    *,
    phase: str,
    shard_index: int,
    shard_count: int,
    output_directory: Path,
    device: str = "cuda",
    selected_learning_rate: Optional[float] = None,
    selected_weight_decay_tau_ema: Optional[float] = None,
    task_ids: Optional[Sequence[str]] = None,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    """Run one resumable, independently schedulable forecast shard."""

    plan = compile_real_text_scaling_plan(config)
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    if runtime.distributed != "none" or runtime.num_processes != 1:
        raise ValueError(
            "fleet shards are independent single-process workers; set "
            "runtime.distributed=none and num_processes=1"
        )
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    all_tasks = build_forecast_fleet_tasks(
        plan,
        phase=phase,
        selected_learning_rate=selected_learning_rate,
        selected_weight_decay_tau_ema=selected_weight_decay_tau_ema,
        run_negative_control=bool(config.get("run_negative_control", True)),
    )
    tasks = assign_forecast_fleet_tasks(all_tasks, shard_count)[shard_index]
    if task_ids:
        requested = set(task_ids)
        known = {task.task_id for task in tasks}
        unknown = requested - known
        if unknown:
            raise ValueError(
                "requested task IDs are not assigned to this shard: "
                + ", ".join(sorted(unknown))
            )
        tasks = [task for task in tasks if task.task_id in requested]
    output_directory.mkdir(parents=True, exist_ok=True)
    cache_directory = output_directory / "trials"
    cache_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / f"{phase}-shard-{shard_index:03d}.json"
    context = prepare_distributed(runtime, device)
    records: List[Dict[str, Any]] = []
    try:
        corpus = _load_corpus(config, plan)
        scales = {str(row["name"]): dict(row) for row in plan["scales"]}
        for completed, task in enumerate(tasks):
            if progress is not None:
                progress(
                    {
                        "phase": f"fleet-{phase}",
                        "completed": completed,
                        "total": len(tasks),
                        "message": (
                            f"Shard {shard_index + 1}/{shard_count} · "
                            f"{task.scale_name} · seed {task.seed}"
                        ),
                        "task": task.to_dict(),
                    }
                )
            record = _run_trial(
                config=config,
                plan=plan,
                scale=scales[task.scale_name],
                corpus=corpus,
                runtime=runtime,
                context=context,
                eta=task.eta,
                weight_decay_tau_ema=task.weight_decay_tau_ema,
                seed=task.seed,
                optimizer_mode=task.optimizer_mode,
                cache_directory=cache_directory,
            )
            records.append(record.to_dict())
            atomic_write_json(
                manifest_path,
                {
                    "schema_version": FLEET_SCHEMA_VERSION,
                    "status": "running",
                    "phase": phase,
                    "plan_fingerprint": plan["fingerprint"],
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "selected_learning_rate": selected_learning_rate,
                    "selected_weight_decay_tau_ema": selected_weight_decay_tau_ema,
                    "assigned_tasks": [row.to_dict() for row in tasks],
                    "completed_task_ids": [row["run_id"] for row in records],
                    "records": records,
                },
            )
        result = {
            "schema_version": FLEET_SCHEMA_VERSION,
            "status": "completed",
            "phase": phase,
            "plan_fingerprint": plan["fingerprint"],
            "shard_index": shard_index,
            "shard_count": shard_count,
            "selected_learning_rate": selected_learning_rate,
            "selected_weight_decay_tau_ema": selected_weight_decay_tau_ema,
            "assigned_tasks": [row.to_dict() for row in tasks],
            "completed_task_ids": [row["run_id"] for row in records],
            "records": records,
        }
        atomic_write_json(manifest_path, result)
        if progress is not None:
            progress(
                {
                    "phase": f"fleet-{phase}",
                    "completed": len(tasks),
                    "total": len(tasks),
                    "message": f"Shard {shard_index + 1}/{shard_count} complete",
                }
            )
        return result
    finally:
        close_distributed(context)


def _expected_record(
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    corpus: TokenizedTextCorpus,
    runtime: PretrainingRuntimeSpec,
    scale: Mapping[str, Any],
    task: ForecastFleetTask,
    cache_directories: Sequence[Path],
) -> tuple[BatchRunRecord, Path]:
    _, run_id = forecast_trial_cache_identity(
        config=config,
        plan=plan,
        scale=scale,
        dataset_fingerprint=corpus.identity_fingerprint,
        runtime=runtime,
        eta=task.eta,
        weight_decay_tau_ema=task.weight_decay_tau_ema,
        seed=task.seed,
        optimizer_mode=task.optimizer_mode,
    )
    matches = [directory / f"{run_id}.json" for directory in cache_directories]
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise ValueError(f"missing immutable trial cache for {task.task_id}")
    digests = {sha256(path.read_bytes()).hexdigest() for path in matches}
    if len(digests) != 1:
        raise ValueError(f"conflicting duplicate trial caches for {task.task_id}")
    with matches[0].open("r", encoding="utf-8") as handle:
        record = BatchRunRecord.from_dict(json.load(handle))
    if (
        record.run_id != run_id
        or record.seed != task.seed
        or record.metadata.get("optimizer_mode") != task.optimizer_mode
        or record.metadata.get("scale", {}).get("name") != task.scale_name
        or record.metadata.get("weight_decay_tau_ema")
        != task.weight_decay_tau_ema
    ):
        raise ValueError(f"trial cache metadata mismatch for {task.task_id}")
    return record, matches[0]


def select_forecast_fleet_learning_rate(
    config: Mapping[str, Any],
    cache_directories: Sequence[Path],
    *,
    require_interior: bool = False,
) -> Dict[str, Any]:
    """Validate every tuning task and select the preregistered mean-loss optimum."""

    plan = compile_real_text_scaling_plan(config)
    frozen = plan.get("frozen_optimizer")
    if isinstance(frozen, Mapping):
        return {
            "schema_version": FLEET_SCHEMA_VERSION,
            "plan_fingerprint": plan["fingerprint"],
            "selected_learning_rate": float(frozen["selected_learning_rate"]),
            "selected_weight_decay_tau_ema": frozen.get(
                "selected_weight_decay_tau_ema"
            ),
            "learning_rate_optimum_is_interior": True,
            "weight_decay_optimum_is_interior": True,
            "optimum_is_interior": True,
            "selection_mode": "frozen_horizon_safe_critical_batch_source",
            "source_critical_batch_result_sha256": frozen[
                "source_critical_batch_result_sha256"
            ],
            "source_pilot_selection_sha256": frozen[
                "source_pilot_selection_sha256"
            ],
            "grid": [],
        }
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    corpus = _load_corpus(config, plan)
    scales = {str(row["name"]): dict(row) for row in plan["scales"]}
    tasks = build_forecast_fleet_tasks(plan, phase="tune")
    records = [
        _expected_record(
            config=config,
            plan=plan,
            corpus=corpus,
            runtime=runtime,
            scale=scales[task.scale_name],
            task=task,
            cache_directories=cache_directories,
        )[0]
        for task in tasks
    ]
    rows = []
    tau_grid = _tuning_tau_candidates(plan)
    for eta in plan["learning_rates"]:
        for tau_ema in tau_grid:
            expected_seeds = _tuning_seeds_for_learning_rate(plan, float(eta))
            losses = [
                row.final_validation_loss
                for row in records
                if row.optimizer.learning_rate == float(eta)
                and row.metadata.get("weight_decay_tau_ema") == tau_ema
            ]
            if len(losses) != len(expected_seeds):
                raise ValueError(
                    f"incomplete tuning records for learning rate {eta:g}"
                    + (
                        f" and tau_EMA {tau_ema:g}"
                        if tau_ema is not None
                        else ""
                    )
                )
            mean, sem = _mean_sem(losses)
            rows.append(
                {
                    "learning_rate": float(eta),
                    "weight_decay_tau_ema": tau_ema,
                    "mean_validation_loss": mean,
                    "sem_validation_loss": sem,
                    "seed_count": len(expected_seeds),
                    "seeds": expected_seeds,
                    "selection_evidence": (
                        "exploratory_single_seed"
                        if len(expected_seeds) == 1
                        else "matched_multi_seed_mean"
                    ),
                    "seed_losses": losses,
                }
            )
    selected_index = min(
        range(len(rows)), key=lambda index: rows[index]["mean_validation_loss"]
    )
    selected = rows[selected_index]
    learning_rate_index = list(plan["learning_rates"]).index(
        selected["learning_rate"]
    )
    learning_rate_interior = (
        0 < learning_rate_index < len(plan["learning_rates"]) - 1
    )
    weight_decay_interior = True
    if selected["weight_decay_tau_ema"] is not None:
        finite_tau_grid = [value for value in tau_grid if value is not None]
        tau_index = finite_tau_grid.index(selected["weight_decay_tau_ema"])
        weight_decay_interior = 0 < tau_index < len(finite_tau_grid) - 1
    result = {
        "schema_version": FLEET_SCHEMA_VERSION,
        "plan_fingerprint": plan["fingerprint"],
        "selected_learning_rate": selected["learning_rate"],
        "selected_weight_decay_tau_ema": selected["weight_decay_tau_ema"],
        "learning_rate_optimum_is_interior": learning_rate_interior,
        "weight_decay_optimum_is_interior": weight_decay_interior,
        "selected_zero_weight_decay_endpoint": (
            selected["weight_decay_tau_ema"] is None
            and bool(
                dict(plan.get("optimizer_contract", {})).get(
                    "include_zero_weight_decay_control", False
                )
            )
        ),
        "optimum_is_interior": learning_rate_interior and weight_decay_interior,
        "selected_seed_count": selected["seed_count"],
        "selection_has_unequal_seed_counts": len(
            {row["seed_count"] for row in rows}
        )
        > 1,
        "adaptive_exploratory_lr_refinement": bool(
            plan.get("learning_rate_refinement")
        ),
        "grid": rows,
    }
    if require_interior and not result["optimum_is_interior"]:
        raise ValueError(
            "reference eta/tau_EMA optimum is on a grid boundary; expand the "
            "preregistered grid before launching the ladder"
        )
    return result


def aggregate_forecast_fleet_cache(
    config: Mapping[str, Any],
    *,
    cache_directories: Sequence[Path],
    output_directory: Path,
) -> Dict[str, Any]:
    """Refuse incomplete fleets, unify exact caches, and run canonical analysis."""

    if not cache_directories:
        raise ValueError("at least one fleet cache directory is required")
    plan = compile_real_text_scaling_plan(config)
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    if runtime.distributed != "none":
        raise ValueError("fleet aggregation requires a single-process runtime config")
    corpus = _load_corpus(config, plan)
    selection = select_forecast_fleet_learning_rate(config, cache_directories)
    if (
        plan.get("run_profile") == "comparison"
        and not selection["optimum_is_interior"]
    ):
        raise ValueError(
            "CompleteP comparison refuses a boundary reference eta/tau_EMA optimum"
        )
    selected = float(selection["selected_learning_rate"])
    selected_tau = selection["selected_weight_decay_tau_ema"]
    tasks = [
        *build_forecast_fleet_tasks(plan, phase="tune"),
        *build_forecast_fleet_tasks(
            plan,
            phase="ladder",
            selected_learning_rate=selected,
            selected_weight_decay_tau_ema=selected_tau,
            run_negative_control=bool(config.get("run_negative_control", True)),
        ),
    ]
    scales = {str(row["name"]): dict(row) for row in plan["scales"]}
    output_directory.mkdir(parents=True, exist_ok=True)
    unified_cache = output_directory / "trials"
    unified_cache.mkdir(parents=True, exist_ok=True)
    copied = []
    validated_records: List[BatchRunRecord] = []
    for task in tasks:
        record, source = _expected_record(
            config=config,
            plan=plan,
            corpus=corpus,
            runtime=runtime,
            scale=scales[task.scale_name],
            task=task,
            cache_directories=cache_directories,
        )
        destination = unified_cache / f"{record.run_id}.json"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied.append(destination.name)
        validated_records.append(record)
    if plan.get("run_profile") == "comparison":
        target = scales[str(plan["scales"][-1]["name"])]
        target_records = [
            record
            for record in validated_records
            if record.metadata.get("scale", {}).get("name") == target["name"]
            and record.metadata.get("optimizer_mode") == "theory"
            and record.optimizer.learning_rate == selected
            and record.metadata.get("weight_decay_tau_ema") == selected_tau
        ]
        target_records.sort(key=lambda record: record.seed)
        target_losses = [record.final_validation_loss for record in target_records]
        target_mean, target_sem = _mean_sem(target_losses)
        baseline = dict(plan["comparison_contract"])
        baseline_mean = float(baseline["baseline_mean_validation_loss"])
        group_audits = [
            dict(record.metadata.get("optimizer_group_audit", {}))
            for record in target_records
        ]
        gates = {
            "reference_lr_optimum_is_interior": bool(
                selection["learning_rate_optimum_is_interior"]
            ),
            "reference_weight_decay_optimum_is_interior": bool(
                selection["weight_decay_optimum_is_interior"]
            ),
            "target_seed_count_complete": len(target_records) == len(plan["seeds"]),
            "target_losses_finite": bool(target_losses)
            and all(math.isfinite(value) for value in target_losses),
            "dataset_fingerprint_matches_baseline": (
                baseline["baseline_dataset_fingerprint"]
                == plan["dataset_identity"]["fingerprint"]
            ),
            "tokenizer_fingerprint_matches_baseline": (
                baseline["baseline_tokenizer_fingerprint"]
                == plan["dataset_identity"]["tokenizer_fingerprint"]
            ),
            "parameter_count_within_one_percent": (
                abs(
                    int(target["parameters"])
                    / int(baseline["baseline_parameters"])
                    - 1.0
                )
                <= 0.01
            ),
            "constant_20_tpp": abs(float(target["tokens_per_parameter"]) - 20.0)
            <= 0.001,
            "completep_group_contract_complete_and_disjoint": bool(group_audits)
            and all(
                audit.get("complete") is True
                and audit.get("disjoint") is True
                and len(audit.get("groups", [])) == 6
                for audit in group_audits
            ),
            "architecture_contains_no_chizat_component": (
                target["block_type"] == "completep_transformer"
                and "chizat"
                not in json.dumps(plan["architecture_contract"], sort_keys=True).lower()
            ),
            "optimizer_is_adamw": all(
                record.optimizer.name == "adamw" for record in target_records
            ),
        }
        passed = all(gates.values())
        result = {
            "schema_version": FLEET_SCHEMA_VERSION,
            "status": "passed" if passed else "failed",
            "campaign": plan["campaign"],
            "plan_fingerprint": plan["fingerprint"],
            "dataset": plan["dataset_identity"],
            "architecture_contract": plan["architecture_contract"],
            "optimizer_contract": plan["optimizer_contract"],
            "schedule": plan["schedule"],
            "reference_selection": selection,
            "baseline": baseline,
            "target": {
                **target,
                "selected_learning_rate": selected,
                "selected_weight_decay_tau_ema": selected_tau,
                "seeds": [record.seed for record in target_records],
                "seed_losses": target_losses,
                "mean_validation_loss": target_mean,
                "sem_validation_loss": target_sem,
                "perplexity": math.exp(target_mean),
            },
            "comparison": {
                "validation_loss_delta_completep_minus_baseline": (
                    target_mean - baseline_mean
                ),
                "perplexity_ratio_completep_over_baseline": math.exp(
                    target_mean - baseline_mean
                ),
                "parameter_ratio_completep_over_baseline": (
                    int(target["parameters"])
                    / int(baseline["baseline_parameters"])
                ),
            },
            "gates": gates,
            "records": [record.to_dict() for record in target_records],
        }
        atomic_write_json(
            output_directory / "fleet-aggregate.json",
            {
                "schema_version": FLEET_SCHEMA_VERSION,
                "status": "completed" if passed else "failed",
                "plan_fingerprint": plan["fingerprint"],
                "source_cache_directories": [
                    str(path) for path in cache_directories
                ],
                "physical_trial_count": len(set(copied)),
                "logical_trial_count": int(plan["planned_grid_trials"]),
                "reference_selection": selection,
                "result_path": "result.json",
            },
        )
        atomic_write_json(output_directory / "result.json", result)
        if not passed:
            failed_gates = [name for name, value in gates.items() if not value]
            raise ValueError(
                "CompleteP comparison failed scientific gates: "
                + ", ".join(failed_gates)
            )
        return result
    configured = dict(config)
    configured["cache_directory"] = str(unified_cache)
    result = run_real_text_scaling_campaign(configured, device="cpu")
    atomic_write_json(
        output_directory / "fleet-aggregate.json",
        {
            "schema_version": FLEET_SCHEMA_VERSION,
            "status": "completed",
            "plan_fingerprint": plan["fingerprint"],
            "source_cache_directories": [str(path) for path in cache_directories],
            "physical_trial_count": len(set(copied)),
            "logical_trial_count": int(plan["planned_grid_trials"]),
            "reference_selection": selection,
            "result_path": "result.json",
        },
    )
    atomic_write_json(output_directory / "result.json", result)
    return result
