from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .batch_scaling import BatchRunRecord
from .forecast_campaigns import (
    _mean_sem,
    _run_trial,
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


@dataclass(frozen=True)
class ForecastFleetTask:
    task_id: str
    phase: str
    ordinal: int
    scale_name: str
    eta: float
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
            "seed": self.seed,
            "optimizer_mode": self.optimizer_mode,
            "estimated_flops": self.estimated_flops,
        }


def build_forecast_fleet_tasks(
    plan: Mapping[str, Any],
    *,
    phase: str,
    selected_learning_rate: Optional[float] = None,
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

    def append(
        scale: Mapping[str, Any], eta: float, seed: int, optimizer_mode: str
    ) -> None:
        ordinal = len(tasks)
        tasks.append(
            ForecastFleetTask(
                task_id=(
                    f"{phase}-{scale['name']}-{optimizer_mode}-"
                    f"eta{eta:g}-seed{seed}"
                ),
                phase=phase,
                ordinal=ordinal,
                scale_name=str(scale["name"]),
                eta=float(eta),
                seed=seed,
                optimizer_mode=optimizer_mode,
                estimated_flops=float(
                    6 * int(scale["parameters"]) * int(scale["presented_tokens"])
                ),
            )
        )

    if phase == "tune":
        for eta in plan["learning_rates"]:
            for seed in seeds:
                append(reference, float(eta), seed, "theory")
        return tasks

    if selected_learning_rate is None:
        raise ValueError("ladder phase requires selected_learning_rate")
    selected = float(selected_learning_rate)
    if selected not in {float(value) for value in plan["learning_rates"]}:
        raise ValueError("selected_learning_rate is not in the preregistered grid")
    # The selected reference trials already exist in the tuning phase. Reusing
    # those exact immutable cache entries avoids three duplicate GPU runs.
    for scale_index, scale in enumerate(scales):
        if scale_index == reference_index:
            continue
        for seed in seeds:
            append(scale, selected, seed, "theory")
    if run_negative_control:
        for seed in seeds:
            append(scales[-1], selected, seed, "wrong_global")
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
    for eta in plan["learning_rates"]:
        losses = [
            row.final_validation_loss
            for row in records
            if row.optimizer.learning_rate == float(eta)
        ]
        if len(losses) != len(plan["seeds"]):
            raise ValueError(f"incomplete tuning records for learning rate {eta:g}")
        mean, sem = _mean_sem(losses)
        rows.append(
            {
                "learning_rate": float(eta),
                "mean_validation_loss": mean,
                "sem_validation_loss": sem,
                "seed_losses": losses,
            }
        )
    selected_index = min(
        range(len(rows)), key=lambda index: rows[index]["mean_validation_loss"]
    )
    result = {
        "schema_version": FLEET_SCHEMA_VERSION,
        "plan_fingerprint": plan["fingerprint"],
        "selected_learning_rate": rows[selected_index]["learning_rate"],
        "optimum_is_interior": 0 < selected_index < len(rows) - 1,
        "grid": rows,
    }
    if require_interior and not result["optimum_is_interior"]:
        raise ValueError(
            "reference learning-rate optimum is on the grid boundary; "
            "expand the preregistered grid before launching the ladder"
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
    selected = float(selection["selected_learning_rate"])
    tasks = [
        *build_forecast_fleet_tasks(plan, phase="tune"),
        *build_forecast_fleet_tasks(
            plan,
            phase="ladder",
            selected_learning_rate=selected,
            run_negative_control=bool(config.get("run_negative_control", True)),
        ),
    ]
    scales = {str(row["name"]): dict(row) for row in plan["scales"]}
    output_directory.mkdir(parents=True, exist_ok=True)
    unified_cache = output_directory / "trials"
    unified_cache.mkdir(parents=True, exist_ok=True)
    copied = []
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
