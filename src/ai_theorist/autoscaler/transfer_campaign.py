from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .model import make_teacher_dataset
from .schema import DataScalingSpec, HorizonSpec, ScaleLevel, StudySpec
from .study import atomic_write_json
from .training import TrialResult, train_trial


@dataclass(frozen=True)
class CampaignTask:
    phase: str
    scale: ScaleLevel
    n_train: int
    batch_size: int
    steps: int
    normalized_learning_rate: float
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["scale"] = asdict(self.scale)
        return payload


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_floats(values: Any, name: str) -> Tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a list")
    result = tuple(float(value) for value in values)
    if not result or any(value <= 0.0 or not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain positive finite values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def validate_campaign(config: Mapping[str, Any]) -> StudySpec:
    if config.get("schema_version") != 1:
        raise ValueError("campaign schema_version must be 1")
    study_payload = config.get("study")
    if not isinstance(study_payload, Mapping):
        raise ValueError("campaign.study must be an object")
    study = StudySpec.from_dict(study_payload)
    if study.architecture.block_type != "pre_norm_mlp" or study.optimizer.name != "adam":
        raise ValueError("this campaign requires a pre_norm_mlp with Adam")
    scale_names = {scale.name for scale in study.scales}
    for payload in config.get("extra_scales", []):
        scale = ScaleLevel.from_dict(payload, 0)
        if scale.name in scale_names:
            raise ValueError(f"duplicate campaign scale: {scale.name}")
        scale_names.add(scale.name)
    lr_phase = config.get("lr_phase")
    if not isinstance(lr_phase, Mapping):
        raise ValueError("campaign.lr_phase must be an object")
    _positive_floats(lr_phase.get("learning_rates"), "lr_phase.learning_rates")
    seeds = tuple(_positive_int(seed, "lr_phase.seeds") for seed in lr_phase.get("seeds", []))
    if len(seeds) < 12 or len(set(seeds)) != len(seeds):
        raise ValueError("lr_phase.seeds must contain at least 12 unique paired seeds")
    requested = tuple(lr_phase.get("scales", []))
    if not requested or any(name not in scale_names for name in requested):
        raise ValueError("lr_phase.scales contains an unknown scale")
    n_train = lr_phase.get("n_train_by_scale")
    if not isinstance(n_train, Mapping) or any(name not in n_train for name in requested):
        raise ValueError("lr_phase.n_train_by_scale must cover every LR scale")
    for name in requested:
        _positive_int(n_train[name], f"lr_phase.n_train_by_scale.{name}")
    extension = config.get("lr_extension_phase")
    if extension is not None:
        if not isinstance(extension, Mapping):
            raise ValueError("campaign.lr_extension_phase must be an object")
        extension_scales = tuple(extension.get("scales", []))
        if not extension_scales or any(name not in requested for name in extension_scales):
            raise ValueError("lr_extension_phase.scales must be a subset of lr_phase.scales")
        extension_rates = _positive_floats(
            extension.get("learning_rates"),
            "lr_extension_phase.learning_rates",
        )
        if set(extension_rates) & set(_positive_floats(lr_phase["learning_rates"], "lr_phase.learning_rates")):
            raise ValueError("LR extension rates must not duplicate the primary grid")
        extension_seeds = tuple(
            _positive_int(seed, "lr_extension_phase.seeds")
            for seed in extension.get("seeds", [])
        )
        if extension_seeds != seeds:
            raise ValueError("LR extension must use the same ordered paired seeds")
    for phase in ("batch_phase", "horizon_phase"):
        payload = config.get(phase)
        if not isinstance(payload, Mapping):
            raise ValueError(f"campaign.{phase} must be an object")
        phase_scales = tuple(payload.get("scales", []))
        if not phase_scales or any(name not in requested for name in phase_scales):
            raise ValueError(f"{phase}.scales must be a subset of lr_phase.scales")
        phase_seeds = tuple(_positive_int(seed, f"{phase}.seeds") for seed in payload.get("seeds", []))
        if len(phase_seeds) < 4 or len(set(phase_seeds)) != len(phase_seeds):
            raise ValueError(f"{phase}.seeds must contain at least four unique seeds")
        _positive_floats(payload.get("lr_multipliers"), f"{phase}.lr_multipliers")
    _positive_floats(config["batch_phase"].get("batch_sizes"), "batch_phase.batch_sizes")
    _positive_int(config["batch_phase"].get("token_budget"), "batch_phase.token_budget")
    batch_extension = config.get("batch_extension_phase")
    if batch_extension is not None:
        if not isinstance(batch_extension, Mapping):
            raise ValueError("campaign.batch_extension_phase must be an object")
        if tuple(batch_extension.get("scales", [])) != tuple(config["batch_phase"]["scales"]):
            raise ValueError("batch extension must use the same ordered scales")
        if tuple(batch_extension.get("seeds", [])) != tuple(config["batch_phase"]["seeds"]):
            raise ValueError("batch extension must use the same ordered paired seeds")
        if batch_extension.get("token_budget") != config["batch_phase"]["token_budget"]:
            raise ValueError("batch extension must use the same token budget")
        multiplier_map = batch_extension.get("lr_multipliers_by_batch")
        if not isinstance(multiplier_map, Mapping) or not multiplier_map:
            raise ValueError("batch extension requires lr_multipliers_by_batch")
        base_batches = {int(value) for value in config["batch_phase"]["batch_sizes"]}
        base_multipliers = {
            float(value) for value in config["batch_phase"]["lr_multipliers"]
        }
        for batch_size_text, multipliers in multiplier_map.items():
            try:
                batch_size = int(batch_size_text)
            except (TypeError, ValueError) as exc:
                raise ValueError("batch extension keys must be integer batch sizes") from exc
            if batch_size not in base_batches:
                raise ValueError(f"unknown batch extension size: {batch_size}")
            extension_multipliers = _positive_floats(
                multipliers,
                f"batch_extension_phase.lr_multipliers_by_batch.{batch_size}",
            )
            if set(extension_multipliers) & base_multipliers:
                raise ValueError("batch extension multipliers must not duplicate the base grid")
    _positive_floats(
        config["horizon_phase"].get("token_budgets"),
        "horizon_phase.token_budgets",
    )
    _positive_int(config["horizon_phase"].get("batch_size"), "horizon_phase.batch_size")
    return study


def campaign_fingerprint(config: Mapping[str, Any]) -> str:
    validate_campaign(config)
    # The lower-grid extension is a predeclared augmentation of the same
    # factorial. Excluding it keeps already-completed primary-grid trials valid
    # when the edge check activates the extension.
    fingerprint_payload = dict(config)
    fingerprint_payload.pop("lr_extension_phase", None)
    fingerprint_payload.pop("batch_extension_phase", None)
    return _canonical_fingerprint(fingerprint_payload)


def _scale_registry(config: Mapping[str, Any], study: StudySpec) -> Dict[str, ScaleLevel]:
    registry = {scale.name: scale for scale in study.scales}
    for index, payload in enumerate(config.get("extra_scales", [])):
        scale = ScaleLevel.from_dict(payload, index)
        registry[scale.name] = scale
    return registry


def _resolved_rates(
    analysis: Optional[Mapping[str, Any]], scales: Iterable[str]
) -> Dict[str, float]:
    if analysis is None or not analysis.get("gate", {}).get("followups_allowed", False):
        raise ValueError("batch/horizon phases require a resolved LR-transfer gate")
    rates = analysis.get("recommended_eta_by_scale")
    if not isinstance(rates, Mapping):
        raise ValueError("LR analysis is missing recommended_eta_by_scale")
    result = {name: float(rates[name]) for name in scales}
    if any(rate <= 0.0 or not math.isfinite(rate) for rate in result.values()):
        raise ValueError("resolved learning rates must be positive and finite")
    return result


def compile_tasks(
    config: Mapping[str, Any],
    phase: str,
    *,
    analysis: Optional[Mapping[str, Any]] = None,
) -> List[CampaignTask]:
    study = validate_campaign(config)
    registry = _scale_registry(config, study)
    lr_phase = config["lr_phase"]
    n_train_by_scale = {name: int(value) for name, value in lr_phase["n_train_by_scale"].items()}
    tasks: List[CampaignTask] = []
    if phase == "lr":
        rates = _positive_floats(lr_phase["learning_rates"], "lr_phase.learning_rates")
        for scale_name in lr_phase["scales"]:
            for seed in lr_phase["seeds"]:
                for rate in rates:
                    tasks.append(
                        CampaignTask(
                            phase,
                            registry[scale_name],
                            n_train_by_scale[scale_name],
                            study.horizon.batch_size,
                            study.horizon.steps,
                            rate,
                            int(seed),
                        )
                    )
    elif phase == "lr-extension":
        payload = config.get("lr_extension_phase")
        if not isinstance(payload, Mapping):
            raise ValueError("campaign has no lr_extension_phase")
        rates = _positive_floats(
            payload["learning_rates"], "lr_extension_phase.learning_rates"
        )
        for scale_name in payload["scales"]:
            for seed in payload["seeds"]:
                for rate in rates:
                    tasks.append(
                        CampaignTask(
                            phase,
                            registry[scale_name],
                            n_train_by_scale[scale_name],
                            study.horizon.batch_size,
                            study.horizon.steps,
                            rate,
                            int(seed),
                        )
                    )
    elif phase in {"batch", "batch-extension"}:
        payload = (
            config["batch_phase"]
            if phase == "batch"
            else config.get("batch_extension_phase")
        )
        if not isinstance(payload, Mapping):
            raise ValueError("campaign has no batch_extension_phase")
        rates = _resolved_rates(analysis, payload["scales"])
        token_budget = int(payload["token_budget"])
        for scale_name in payload["scales"]:
            for seed in payload["seeds"]:
                batch_sizes = (
                    payload["batch_sizes"]
                    if phase == "batch"
                    else payload["lr_multipliers_by_batch"].keys()
                )
                for batch_size_value in batch_sizes:
                    batch_size = int(batch_size_value)
                    steps = max(1, math.ceil(token_budget / batch_size))
                    multipliers = (
                        payload["lr_multipliers"]
                        if phase == "batch"
                        else payload["lr_multipliers_by_batch"][str(batch_size)]
                    )
                    for multiplier in multipliers:
                        tasks.append(
                            CampaignTask(
                                phase,
                                registry[scale_name],
                                n_train_by_scale[scale_name],
                                batch_size,
                                steps,
                                rates[scale_name] * float(multiplier),
                                int(seed),
                            )
                        )
    elif phase == "horizon":
        payload = config["horizon_phase"]
        rates = _resolved_rates(analysis, payload["scales"])
        batch_size = int(payload["batch_size"])
        for scale_name in payload["scales"]:
            for seed in payload["seeds"]:
                for token_budget_value in payload["token_budgets"]:
                    steps = max(1, math.ceil(int(token_budget_value) / batch_size))
                    for multiplier in payload["lr_multipliers"]:
                        tasks.append(
                            CampaignTask(
                                phase,
                                registry[scale_name],
                                n_train_by_scale[scale_name],
                                batch_size,
                                steps,
                                rates[scale_name] * float(multiplier),
                                int(seed),
                            )
                        )
    else:
        raise ValueError(
            "phase must be lr, lr-extension, batch, batch-extension, or horizon"
        )
    return tasks


def compile_campaign_plan(
    config: Mapping[str, Any],
    phase: str,
    *,
    analysis: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    tasks = compile_tasks(config, phase, analysis=analysis)
    return {
        "schema_version": 1,
        "campaign_fingerprint": campaign_fingerprint(config),
        "phase": phase,
        "trial_count": len(tasks),
        "scale_count": len({task.scale.name for task in tasks}),
        "seed_count": len({task.seed for task in tasks}),
        "resumable": True,
        "paired_by_seed": True,
    }


def _task_key(fingerprint: str, task: CampaignTask) -> str:
    return _canonical_fingerprint(
        {"campaign_fingerprint": fingerprint, "task": task.to_dict()}
    )[:20]


def run_campaign_phase(
    config: Mapping[str, Any],
    phase: str,
    output_dir: Path,
    *,
    device: str,
    shard_index: int = 0,
    shard_count: int = 1,
    analysis: Optional[Mapping[str, Any]] = None,
    progress: Optional[Any] = None,
    only_scales: Optional[Sequence[str]] = None,
    only_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    study = validate_campaign(config)
    fingerprint = campaign_fingerprint(config)
    all_tasks = compile_tasks(config, phase, analysis=analysis)
    if only_scales is not None:
        allowed_scales = set(only_scales)
        unknown_scales = allowed_scales - {task.scale.name for task in all_tasks}
        if unknown_scales:
            raise ValueError(f"unknown filtered scale(s): {', '.join(sorted(unknown_scales))}")
        all_tasks = [task for task in all_tasks if task.scale.name in allowed_scales]
    if only_seeds is not None:
        allowed_seeds = {int(seed) for seed in only_seeds}
        unknown_seeds = allowed_seeds - {task.seed for task in all_tasks}
        if unknown_seeds:
            raise ValueError(
                f"unknown filtered seed(s): {', '.join(str(seed) for seed in sorted(unknown_seeds))}"
            )
        all_tasks = [task for task in all_tasks if task.seed in allowed_seeds]
    seed_order = sorted({task.seed for task in all_tasks})
    seed_shards = {seed: index % shard_count for index, seed in enumerate(seed_order)}
    tasks = [task for task in all_tasks if seed_shards[task.seed] == shard_index]
    tasks.sort(key=lambda task: (task.scale.name, task.seed, task.batch_size, task.steps, task.normalized_learning_rate))
    trials_dir = output_dir / "trials"
    checkpoints_dir = output_dir / "checkpoints"
    trials_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "campaign_fingerprint": fingerprint,
        "phase": phase,
        "device": device,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_seeds": [seed for seed in seed_order if seed_shards[seed] == shard_index],
        "trial_count": len(tasks),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    completed = 0
    current_dataset_key: Optional[Tuple[str, int]] = None
    prepared_dataset = None
    for task in tasks:
        key = _task_key(fingerprint, task)
        trial_path = trials_dir / f"{key}.json"
        if trial_path.is_file():
            with trial_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if (
                cached.get("campaign_fingerprint") != fingerprint
                or cached.get("task") != task.to_dict()
            ):
                raise RuntimeError(f"cached trial metadata mismatch: {trial_path}")
            completed += 1
            continue
        dataset_key = (task.scale.name, task.n_train)
        if dataset_key != current_dataset_key:
            prepared_dataset = None
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            dataset_spec = replace(study.dataset, n_train=task.n_train)
            prepared_dataset = make_teacher_dataset(
                study.architecture,
                dataset_spec,
                device=device,
            )
            current_dataset_key = dataset_key
        concrete = replace(
            study,
            dataset=replace(study.dataset, n_train=task.n_train),
            horizon=HorizonSpec(
                steps=task.steps,
                batch_size=task.batch_size,
                microbatch_size=min(study.horizon.microbatch_size or task.batch_size, task.batch_size),
            ),
            data_scaling=DataScalingSpec(),
        )
        result = train_trial(
            concrete,
            task.scale,
            task.normalized_learning_rate,
            task.seed,
            raw_learning_rate=task.normalized_learning_rate,
            device=device,
            checkpoint_path=checkpoints_dir / f"{key}.pt",
            checkpoint_every=max(1, task.steps // 4),
            prepared_dataset=prepared_dataset,
        )
        atomic_write_json(
            trial_path,
            {
                "schema_version": 1,
                "campaign_fingerprint": fingerprint,
                "task": task.to_dict(),
                "result": result.to_dict(),
            },
        )
        completed += 1
        if progress is not None:
            progress(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": len(tasks),
                    "scale": task.scale.name,
                    "seed": task.seed,
                    "normalized_learning_rate": task.normalized_learning_rate,
                    "batch_size": task.batch_size,
                    "steps": task.steps,
                    "validation_loss": result.final_validation_loss,
                }
            )
    atomic_write_json(output_dir / "complete.json", manifest | {"completed": completed})
    return manifest | {"completed": completed}


def load_trial_rows(paths: Iterable[Path], fingerprint: str) -> List[Dict[str, Any]]:
    rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.glob("**/trials/*.json"))
        for candidate in candidates:
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("campaign_fingerprint") != fingerprint:
                raise ValueError(f"foreign campaign trial: {candidate}")
            task = payload["task"]
            identity = (
                task["phase"],
                task["scale"]["name"],
                task["seed"],
                task["batch_size"],
                task["steps"],
                float(task["normalized_learning_rate"]),
            )
            if identity in rows and rows[identity] != payload:
                raise ValueError(f"conflicting duplicate trial: {identity}")
            rows[identity] = payload
    return list(rows.values())


def _mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("inf"), float("inf")
    return mean(finite), (stdev(finite) / math.sqrt(len(finite)) if len(finite) > 1 else float("inf"))


def _quadratic_optimum(
    rates: Sequence[float],
    losses: np.ndarray,
) -> float:
    means = losses.mean(axis=1)
    best = int(np.argmin(means))
    lo = max(0, best - 2)
    hi = min(len(rates), best + 3)
    if hi - lo < 3:
        return float(rates[best])
    x = np.log(np.asarray(rates[lo:hi], dtype=np.float64))
    coefficients = np.polyfit(x, means[lo:hi], 2)
    if coefficients[0] <= 0.0:
        return float(rates[best])
    optimum = -coefficients[1] / (2.0 * coefficients[0])
    if optimum < x.min() or optimum > x.max():
        return float(rates[best])
    return float(math.exp(optimum))


def _paired_difference(
    left: Mapping[int, float], right: Mapping[int, float]
) -> Tuple[float, float]:
    seeds = sorted(set(left) & set(right))
    if len(seeds) < 2:
        return float("nan"), float("inf")
    differences = [left[seed] - right[seed] for seed in seeds]
    return _mean_sem(differences)


def analyze_lr_trials(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    fingerprint = campaign_fingerprint(config)
    study = validate_campaign(config)
    registry = _scale_registry(config, study)
    phase = config["lr_phase"]
    scales = tuple(phase["scales"])
    primary_rates = tuple(
        sorted(_positive_floats(phase["learning_rates"], "lr_phase.learning_rates"))
    )
    seeds = tuple(int(seed) for seed in phase["seeds"])
    extension = config.get("lr_extension_phase")
    extension_scales = tuple(extension["scales"]) if isinstance(extension, Mapping) else ()
    extension_rates = (
        tuple(
            sorted(
                _positive_floats(
                    extension["learning_rates"],
                    "lr_extension_phase.learning_rates",
                )
            )
        )
        if isinstance(extension, Mapping)
        else ()
    )
    rates_by_scale = {
        scale_name: tuple(
            sorted(
                set(primary_rates)
                | (set(extension_rates) if scale_name in extension_scales else set())
            )
        )
        for scale_name in scales
    }
    expected = sum(len(rates_by_scale[name]) * len(seeds) for name in scales)
    lr_rows = [
        row for row in rows if row["task"]["phase"] in {"lr", "lr-extension"}
    ]
    if len(lr_rows) != expected:
        raise ValueError(f"incomplete LR factorial: expected {expected} trials, found {len(lr_rows)}")
    by_condition: Dict[Tuple[str, float], Dict[int, float]] = {}
    train_by_condition: Dict[Tuple[str, float], Dict[int, float]] = {}
    for row in lr_rows:
        task = row["task"]
        result = row["result"]
        key = (task["scale"]["name"], float(task["normalized_learning_rate"]))
        by_condition.setdefault(key, {})[int(task["seed"])] = float(result["final_validation_loss"])
        trace = result.get("train_loss_trace") or []
        if trace:
            train_by_condition.setdefault(key, {})[int(task["seed"])] = float(trace[-1]["training_loss"])

    rng = np.random.default_rng(20_260_809)
    bootstrap_samples = int(phase.get("bootstrap_samples", 2000))
    scale_results: Dict[str, Any] = {}
    optimum_samples: Dict[str, np.ndarray] = {}
    for scale_name in scales:
        rates = rates_by_scale[scale_name]
        loss_matrix = np.asarray(
            [[by_condition[(scale_name, rate)][seed] for seed in seeds] for rate in rates],
            dtype=np.float64,
        )
        optimum = _quadratic_optimum(rates, loss_matrix)
        samples = []
        for _ in range(bootstrap_samples):
            indices = rng.integers(0, len(seeds), size=len(seeds))
            samples.append(_quadratic_optimum(rates, loss_matrix[:, indices]))
        optimum_samples[scale_name] = np.asarray(samples, dtype=np.float64)
        summaries = []
        for rate_index, rate in enumerate(rates):
            loss_mean, loss_sem = _mean_sem(loss_matrix[rate_index].tolist())
            train_values = list(train_by_condition.get((scale_name, rate), {}).values())
            train_mean, train_sem = _mean_sem(train_values)
            summaries.append(
                {
                    "normalized_learning_rate": rate,
                    "mean_validation_loss": loss_mean,
                    "sem_validation_loss": loss_sem,
                    "mean_final_minibatch_train_loss": train_mean,
                    "sem_final_minibatch_train_loss": train_sem,
                }
            )
        numerical_rate = rates[int(np.argmin(loss_matrix.mean(axis=1)))]
        numerical_index = rates.index(numerical_rate)
        scale_results[scale_name] = {
            "width": registry[scale_name].width,
            "depth": registry[scale_name].repeats,
            "n_train": int(phase["n_train_by_scale"][scale_name]),
            "numerical_best_eta": numerical_rate,
            "grid_bracketed": 0 < numerical_index < len(rates) - 1,
            "evaluated_learning_rates": list(rates),
            "quadratic_best_eta": optimum,
            "quadratic_best_eta_interval_95": [
                float(np.quantile(optimum_samples[scale_name], 0.025)),
                float(np.quantile(optimum_samples[scale_name], 0.975)),
            ],
            "summaries": summaries,
        }

    fit_scales = tuple(phase.get("fit_scales", scales))
    reference = registry[str(phase.get("reference_scale", "S3"))]
    design = np.asarray(
        [
            [
                1.0,
                math.log(registry[name].width / reference.width),
                math.log(registry[name].repeats / reference.repeats),
            ]
            for name in fit_scales
        ],
        dtype=np.float64,
    )
    observed = np.log(np.asarray([scale_results[name]["quadratic_best_eta"] for name in fit_scales]))
    coefficients, _, _, _ = np.linalg.lstsq(design, observed, rcond=None)
    condition_number = float(np.linalg.cond(design))
    coefficient_samples = []
    for sample_index in range(bootstrap_samples):
        response = np.log(
            np.asarray([optimum_samples[name][sample_index] for name in fit_scales])
        )
        sample_coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
        coefficient_samples.append(sample_coefficients)
    coefficient_samples_array = np.asarray(coefficient_samples)
    joint_fit = {
        "formula": "eta = eta_reference * (width/reference_width)^width_exponent * (depth/reference_depth)^depth_exponent",
        "eta_reference": float(math.exp(coefficients[0])),
        "width_exponent": float(coefficients[1]),
        "depth_exponent": float(coefficients[2]),
        "width_exponent_interval_95": np.quantile(coefficient_samples_array[:, 1], (0.025, 0.975)).tolist(),
        "depth_exponent_interval_95": np.quantile(coefficient_samples_array[:, 2], (0.025, 0.975)).tolist(),
        "design_condition_number": condition_number,
        "separately_identifiable": condition_number < 30.0,
        "fit_scales": list(fit_scales),
    }

    base_rate = float(phase["proposed_eta"])
    lower_rate = float(phase["lower_probe_eta"])
    control_rate = float(phase["negative_control_eta"])
    minimum_fraction = float(phase.get("minimum_effect_fraction", 0.01))
    direct_checks: Dict[str, Any] = {}
    all_noninferior = True
    for scale_name in scales:
        numerical = float(scale_results[scale_name]["numerical_best_eta"])
        penalty, penalty_sem = _paired_difference(
            by_condition[(scale_name, base_rate)],
            by_condition[(scale_name, numerical)],
        )
        base_mean = mean(by_condition[(scale_name, base_rate)].values())
        tolerance = max(2.0 * penalty_sem, minimum_fraction * base_mean)
        accepted = penalty <= tolerance
        all_noninferior = all_noninferior and accepted
        direct_checks[scale_name] = {
            "proposed_eta": base_rate,
            "local_numerical_best_eta": numerical,
            "paired_penalty_vs_local_best": penalty,
            "paired_penalty_sem": penalty_sem,
            "noninferiority_tolerance": tolerance,
            "accepted": accepted,
        }

    target = str(phase.get("target_scale", "S6"))
    lower_advantage, lower_sem = _paired_difference(
        by_condition[(target, base_rate)],
        by_condition[(target, lower_rate)],
    )
    control_penalty, control_sem = _paired_difference(
        by_condition[(target, control_rate)],
        by_condition[(target, base_rate)],
    )
    base_mean = mean(by_condition[(target, base_rate)].values())
    lower_threshold = max(2.0 * lower_sem, minimum_fraction * base_mean)
    control_threshold = max(2.0 * control_sem, minimum_fraction * base_mean)
    lower_significantly_better = lower_advantage > lower_threshold
    negative_control_rejected = control_penalty > control_threshold
    target_checks = {
        "lower_probe_vs_proposed": {
            "lower_eta": lower_rate,
            "proposed_eta": base_rate,
            "paired_improvement": lower_advantage,
            "paired_improvement_sem": lower_sem,
            "threshold": lower_threshold,
            "lower_is_significantly_better": lower_significantly_better,
        },
        "negative_control_vs_proposed": {
            "negative_control_eta": control_rate,
            "proposed_eta": base_rate,
            "paired_loss_increase": control_penalty,
            "paired_loss_increase_sem": control_sem,
            "threshold": control_threshold,
            "rejected": negative_control_rejected,
        },
    }

    capacity_x = np.log(
        np.asarray(
            [
                registry[name].width
                * registry[name].repeats
                / (reference.width * reference.repeats)
                for name in scales
            ],
            dtype=np.float64,
        )
    )
    capacity_design = np.column_stack((np.ones_like(capacity_x), capacity_x))
    capacity_y = np.log(np.asarray([scale_results[name]["quadratic_best_eta"] for name in scales]))
    capacity_coefficients, _, _, _ = np.linalg.lstsq(capacity_design, capacity_y, rcond=None)
    capacity_slopes = []
    for sample_index in range(bootstrap_samples):
        response = np.log(np.asarray([optimum_samples[name][sample_index] for name in scales]))
        sample_coefficients, _, _, _ = np.linalg.lstsq(capacity_design, response, rcond=None)
        capacity_slopes.append(float(sample_coefficients[1]))
    capacity_interval = np.quantile(capacity_slopes, (0.025, 0.975)).tolist()
    drift_fit = {
        "capacity_exponent": float(capacity_coefficients[1]),
        "capacity_exponent_interval_95": capacity_interval,
        "formula": "eta proportional to (width * depth)^capacity_exponent",
    }

    gate_scales = tuple(phase.get("gate_scales", ("S3", "S4", "S5", "S6")))
    unbracketed = [name for name in gate_scales if not scale_results[name]["grid_bracketed"]]
    if unbracketed:
        gate_status = "ambiguous"
        reasons = [
            "the LR optimum is not bracketed at: " + ", ".join(unbracketed)
        ]
    elif lower_significantly_better and capacity_interval[1] < 0.0:
        gate_status = "resolved_drift"
        reasons = []
    elif not negative_control_rejected:
        gate_status = "ambiguous"
        reasons = ["the predeclared negative control was not rejected"]
    elif all_noninferior and capacity_interval[0] <= 0.0 <= capacity_interval[1]:
        gate_status = "constant_transfer_supported"
        reasons = []
    else:
        gate_status = "ambiguous"
        reasons = ["the scale-local LR evidence does not resolve constant transfer versus drift"]
    followups_allowed = gate_status != "ambiguous"
    if gate_status == "constant_transfer_supported":
        recommended = {name: base_rate for name in config["lr_phase"]["scales"]}
    elif gate_status == "resolved_drift":
        recommended = {
            name: float(scale_results[name]["quadratic_best_eta"])
            for name in config["lr_phase"]["scales"]
        }
    else:
        recommended = {}
    return {
        "schema_version": 1,
        "campaign_fingerprint": fingerprint,
        "trial_count": len(lr_rows),
        "paired_seed_count": len(seeds),
        "task": {
            "difficulty": study.dataset.difficulty,
            "input_dim": study.architecture.input_dim,
            "teacher_width": study.dataset.teacher_width,
            "teacher_depth": study.dataset.teacher_depth,
            "n_validation": study.dataset.n_validation,
        },
        "scale_results": scale_results,
        "joint_width_depth_fit": joint_fit,
        "capacity_drift_fit": drift_fit,
        "direct_transfer_checks": direct_checks,
        "target_checks": target_checks,
        "gate": {
            "status": gate_status,
            "followups_allowed": followups_allowed,
            "reasons": reasons,
        },
        "recommended_eta_by_scale": recommended,
    }


def analyze_followup_trials(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    phase: str,
) -> Dict[str, Any]:
    if phase not in {"batch", "horizon"}:
        raise ValueError("follow-up phase must be batch or horizon")
    fingerprint = campaign_fingerprint(config)
    accepted_phases = {phase}
    if phase == "batch" and isinstance(config.get("batch_extension_phase"), Mapping):
        accepted_phases.add("batch-extension")
    phase_rows = [row for row in rows if row["task"]["phase"] in accepted_phases]
    if not phase_rows:
        raise ValueError(f"no {phase} trials found")
    phase_config = config[f"{phase}_phase"]
    primary_values = (
        phase_config["batch_sizes"]
        if phase == "batch"
        else phase_config["token_budgets"]
    )
    expected = (
        len(phase_config["scales"])
        * len(phase_config["seeds"])
        * len(primary_values)
        * len(phase_config["lr_multipliers"])
    )
    if phase == "batch" and "batch-extension" in accepted_phases:
        extension = config["batch_extension_phase"]
        expected += (
            len(extension["scales"])
            * len(extension["seeds"])
            * sum(len(values) for values in extension["lr_multipliers_by_batch"].values())
        )
    if len(phase_rows) != expected:
        raise ValueError(
            f"incomplete {phase} factorial: expected {expected} trials, "
            f"found {len(phase_rows)}"
        )
    grouped: Dict[Tuple[str, int, int, float], List[float]] = {}
    train_grouped: Dict[Tuple[str, int, int, float], List[float]] = {}
    paired: Dict[Tuple[str, int, int, float], Dict[int, float]] = {}
    for row in phase_rows:
        task = row["task"]
        result = row["result"]
        key = (
            task["scale"]["name"],
            int(task["batch_size"]),
            int(task["steps"]),
            float(task["normalized_learning_rate"]),
        )
        validation_loss = float(result["final_validation_loss"])
        grouped.setdefault(key, []).append(validation_loss)
        paired.setdefault(key, {})[int(task["seed"])] = validation_loss
        trace = result.get("train_loss_trace") or []
        if trace:
            train_grouped.setdefault(key, []).append(float(trace[-1]["training_loss"]))
    summaries = []
    for key in sorted(grouped):
        validation_mean, validation_sem = _mean_sem(grouped[key])
        train_mean, train_sem = _mean_sem(train_grouped.get(key, []))
        summaries.append(
            {
                "scale": key[0],
                "batch_size": key[1],
                "steps": key[2],
                "token_horizon": key[1] * key[2],
                "normalized_learning_rate": key[3],
                "mean_validation_loss": validation_mean,
                "sem_validation_loss": validation_sem,
                "mean_final_minibatch_train_loss": train_mean,
                "sem_final_minibatch_train_loss": train_sem,
            }
        )
    primary = "batch_size" if phase == "batch" else "token_horizon"
    selected = []
    profiles_by_scale: Dict[str, List[Dict[str, Any]]] = {}
    minimum_fraction = 0.005
    for scale_name in sorted({row["scale"] for row in summaries}):
        candidates = [row for row in summaries if row["scale"] == scale_name]
        local_profiles = []
        for value in sorted({row[primary] for row in candidates}):
            value_candidates = [row for row in candidates if row[primary] == value]
            local_best = min(value_candidates, key=lambda row: row["mean_validation_loss"])
            local_profiles.append(
                dict(local_best)
                | {
                    "selected_normalized_learning_rate": local_best[
                        "normalized_learning_rate"
                    ],
                    "_key": (
                        local_best["scale"],
                        local_best["batch_size"],
                        local_best["steps"],
                        local_best["normalized_learning_rate"],
                    ),
                }
            )
        best = min(local_profiles, key=lambda row: row["mean_validation_loss"])
        best_losses = paired[best["_key"]]
        reported_profiles = []
        for profile in local_profiles:
            penalty, penalty_sem = _paired_difference(paired[profile["_key"]], best_losses)
            tolerance = max(
                2.0 * penalty_sem,
                minimum_fraction * best["mean_validation_loss"],
            )
            reported_profiles.append(
                {key: value for key, value in profile.items() if key != "_key"}
                | {
                    "paired_penalty_vs_best": penalty,
                    "paired_penalty_sem": penalty_sem,
                    "noninferiority_tolerance": tolerance,
                    "noninferior_to_best": penalty <= tolerance,
                }
            )
        noninferior_values = [
            row[primary] for row in reported_profiles if row["noninferior_to_best"]
        ]
        largest_noninferior = max(noninferior_values)
        smallest_profile = min(local_profiles, key=lambda row: row[primary])
        largest_profile = max(local_profiles, key=lambda row: row[primary])
        largest_improvement, largest_improvement_sem = _paired_difference(
            paired[smallest_profile["_key"]],
            paired[largest_profile["_key"]],
        )
        profiles_by_scale[scale_name] = reported_profiles
        selected.append(
            {
                "scale": scale_name,
                "selected_value": best[primary],
                "selected_normalized_learning_rate": best["normalized_learning_rate"],
                "mean_validation_loss": best["mean_validation_loss"],
                "sem_validation_loss": best["sem_validation_loss"],
                "largest_noninferior_value": largest_noninferior,
                "smallest_to_largest_paired_improvement": largest_improvement,
                "smallest_to_largest_paired_improvement_sem": largest_improvement_sem,
            }
        )
    return {
        "schema_version": 1,
        "campaign_fingerprint": fingerprint,
        "phase": phase,
        "trial_count": len(phase_rows),
        "controlled_variable": (
            "token_horizon" if phase == "batch" else "batch_size"
        ),
        "paired_seed_count": len(phase_config["seeds"]),
        "lr_retuned_per_value": True,
        "summaries": summaries,
        "profiles_by_scale": profiles_by_scale,
        "selected_by_scale": selected,
    }
