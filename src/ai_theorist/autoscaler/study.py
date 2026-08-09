from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .scaling import fit_scaling_law
from .schema import ScaleLevel, StudySpec, compile_plan, estimate_training_compute, parameter_count
from .training import TrialResult, train_trial
from .tuning import (
    adaptive_tune,
    paired_mean_and_sem,
    summarize_trials,
    transfer_learning_rate,
    transfer_rule_name,
)


ProgressCallback = Callable[[Dict[str, Any]], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _summary(trials: Sequence[TrialResult], rate: float) -> Dict[str, Any]:
    summary = summarize_trials(trials, rate)
    return summary.to_dict()


def _emit(callback: Optional[ProgressCallback], phase: str, completed: int, total: int, message: str) -> None:
    if callback is not None:
        callback(
            {
                "phase": phase,
                "completed": completed,
                "total": total,
                "message": message,
                "updated_at": _utc_now(),
            }
        )


def run_study(
    spec: StudySpec,
    *,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Run tuning, LR transfer, fixed-horizon scaling, and largest-scale calibration."""
    started_at = _utc_now()
    plan = compile_plan(spec)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / "manifest.json", {"spec": spec.to_dict(), "plan": plan})

    fit_scales = spec.scales[:-spec.holdout_count]
    holdout_scales = spec.scales[-spec.holdout_count:]
    reference_scale = fit_scales[len(fit_scales) // 2]
    trials: List[TrialResult] = []
    trial_cache: Dict[Tuple[str, float, int], TrialResult] = {}
    trial_counter = 0
    estimated_total = int(plan["trial_budget_before_edge_expansion"])

    def run(scale: ScaleLevel, rate: float, seed: int, phase: str) -> TrialResult:
        nonlocal trial_counter
        key = (scale.name, float(rate), int(seed))
        if key not in trial_cache:
            result = train_trial(spec, scale, rate, seed, device=device)
            trial_cache[key] = result
            trials.append(result)
            trial_counter += 1
            _emit(
                progress,
                phase,
                trial_counter,
                estimated_total,
                f"{scale.name} · seed {seed} · lr {rate:.3g}",
            )
        return trial_cache[key]

    _emit(progress, "tuning", 0, estimated_total, f"Tuning {reference_scale.name}")
    tuning, _ = adaptive_tune(
        spec.tuning.learning_rates,
        spec.seeds,
        lambda rate, seed: run(reference_scale, rate, seed, "tuning"),
        max_expansion_rounds=spec.tuning.max_expansion_rounds,
        expansion_factor=spec.tuning.expansion_factor,
    )
    base_learning_rate = tuning.selected_learning_rate

    scale_summaries: List[Dict[str, Any]] = []
    for scale in spec.scales:
        scale_rate = transfer_learning_rate(
            spec.optimizer.name,
            base_learning_rate,
            reference_scale.width,
            scale.width,
        )
        selected_trials = [run(scale, scale_rate, seed, "transfer") for seed in spec.seeds]
        summary = _summary(selected_trials, scale_rate)
        scale_summaries.append(
            {
                "scale": scale.name,
                "width": scale.width,
                "repeats": scale.repeats,
                "parameter_count": parameter_count(spec, scale),
                "estimated_training_compute": estimate_training_compute(spec, scale),
                "learning_rate": scale_rate,
                "mean_final_validation_loss": summary["mean_final_validation_loss"],
                "sem_final_validation_loss": summary["sem_final_validation_loss"],
                "losses_by_seed": summary["losses_by_seed"],
                "role": "holdout" if scale in holdout_scales else "fit",
            }
        )

    transfer_checks = []
    probe_multiplier = 10.0 ** spec.validation.transfer_probe_decades
    for scale in holdout_scales:
        predicted_rate = transfer_learning_rate(
            spec.optimizer.name,
            base_learning_rate,
            reference_scale.width,
            scale.width,
        )
        candidates = (predicted_rate / probe_multiplier, predicted_rate, predicted_rate * probe_multiplier)
        candidate_summaries = []
        candidate_trials: Dict[float, List[TrialResult]] = {}
        for rate in candidates:
            candidate_trials[rate] = [run(scale, rate, seed, "transfer-validation") for seed in spec.seeds]
            candidate_summaries.append(summarize_trials(candidate_trials[rate], rate))
        finite = [item for item in candidate_summaries if math.isfinite(item.mean_final_validation_loss)]
        if not finite:
            raise RuntimeError(f"All transfer probes diverged at {scale.name}")
        local_best = min(finite, key=lambda item: item.mean_final_validation_loss)
        transferred = next(item for item in candidate_summaries if item.learning_rate == predicted_rate)
        if math.isfinite(transferred.mean_final_validation_loss):
            paired_penalty, paired_sem = paired_mean_and_sem(
                transferred.losses_by_seed, local_best.losses_by_seed
            )
        else:
            paired_penalty, paired_sem = float("inf"), float("inf")
        lr_distance = abs(math.log10(local_best.learning_rate / predicted_rate))
        accepted = (
            lr_distance <= spec.validation.transfer_probe_decades + 1e-12
            and paired_penalty <= max(2.0 * paired_sem, 0.02 * local_best.mean_final_validation_loss)
        )
        transfer_checks.append(
            {
                "scale": scale.name,
                "transferred_learning_rate": predicted_rate,
                "local_probe_best_learning_rate": local_best.learning_rate,
                "log10_learning_rate_distance": lr_distance,
                "paired_loss_penalty": paired_penalty,
                "paired_loss_penalty_sem": paired_sem,
                "accepted": accepted,
                "candidates": [item.to_dict() for item in candidate_summaries],
            }
        )

    negative_control: Optional[Dict[str, Any]] = None
    if spec.validation.run_negative_control:
        target = holdout_scales[-1]
        predicted_rate = transfer_learning_rate(
            spec.optimizer.name,
            base_learning_rate,
            reference_scale.width,
            target.width,
        )
        if spec.optimizer.name == "adam":
            wrong_rule = "incorrect_sqrt_width_learning_rate_growth"
            wrong_rate = base_learning_rate * math.sqrt(target.width / reference_scale.width)
        else:
            wrong_rule = "incorrect_constant_learning_rate"
            wrong_rate = base_learning_rate
        control_trials = [run(target, wrong_rate, seed, "negative-control") for seed in spec.seeds]
        control = summarize_trials(control_trials, wrong_rate)
        baseline_trials = [trial_cache[(target.name, float(predicted_rate), seed)] for seed in spec.seeds]
        baseline = summarize_trials(baseline_trials, predicted_rate)
        if math.isfinite(control.mean_final_validation_loss):
            difference, difference_sem = paired_mean_and_sem(control.losses_by_seed, baseline.losses_by_seed)
            rejected = difference > max(2.0 * difference_sem, 0.01 * baseline.mean_final_validation_loss)
        else:
            difference, difference_sem, rejected = float("inf"), float("inf"), True
        negative_control = {
            "rule": wrong_rule,
            "learning_rate": wrong_rate,
            "mean_final_validation_loss": control.mean_final_validation_loss,
            "paired_loss_increase": difference,
            "paired_loss_increase_sem": difference_sem,
            "rejected": rejected,
        }

    fit_rows = [row for row in scale_summaries if row["role"] == "fit"]
    scaling_fit = fit_scaling_law(
        [row["estimated_training_compute"] for row in fit_rows],
        [row["mean_final_validation_loss"] for row in fit_rows],
        [row["sem_final_validation_loss"] for row in fit_rows],
        bootstrap_samples=spec.validation.bootstrap_samples,
    )
    holdout_calibration = []
    for row in (row for row in scale_summaries if row["role"] == "holdout"):
        predicted = scaling_fit.predict(row["estimated_training_compute"])
        lower, upper = scaling_fit.prediction_interval(row["estimated_training_compute"])
        observed = float(row["mean_final_validation_loss"])
        error = abs(predicted - observed)
        relative_error = error / observed
        tolerance = max(0.02 * observed, 3.0 * float(row["sem_final_validation_loss"]), (upper - lower) / 2.0)
        # Uncertainty coverage is necessary but not sufficient: a numerically wide
        # interval must never turn a poor largest-model point prediction into a pass.
        accepted = error <= tolerance and relative_error <= 0.10
        holdout_calibration.append(
            {
                "scale": row["scale"],
                "predicted_final_validation_loss": predicted,
                "prediction_interval_95": [lower, upper],
                "observed_final_validation_loss": observed,
                "absolute_error": error,
                "relative_error": relative_error,
                "tolerance": tolerance,
                "maximum_relative_error": 0.10,
                "accepted": accepted,
            }
        )

    last_compute = float(scale_summaries[-1]["estimated_training_compute"])
    previous_compute = float(scale_summaries[-2]["estimated_training_compute"])
    next_compute = last_compute * (last_compute / previous_compute)
    transfer_ok = all(check["accepted"] for check in transfer_checks)
    calibration_ok = all(check["accepted"] for check in holdout_calibration)
    control_ok = negative_control is None or bool(negative_control["rejected"])
    calibration_law_usable = scaling_fit.forecastable or scaling_fit.short_range_forecastable
    forecastable = (
        tuning.optimum_is_interior
        and transfer_ok
        and calibration_law_usable
        and calibration_ok
        and control_ok
    )
    floor_reason = "estimated loss floor is pinned to the smallest observation"
    refusal_reasons = [
        reason for reason in scaling_fit.refusal_reasons
        if not (scaling_fit.short_range_forecastable and reason == floor_reason)
    ]
    warnings = []
    if scaling_fit.short_range_forecastable and not scaling_fit.forecastable:
        warnings.append(
            "asymptotic loss floor is unresolved; forecast is limited to one adjacent compute step"
        )
    if not tuning.optimum_is_interior:
        refusal_reasons.append("reference learning-rate optimum is on the tested boundary")
    if not transfer_ok:
        refusal_reasons.append("learning-rate transfer failed the largest-scale local probes")
    if not calibration_ok:
        refusal_reasons.append("scaling law missed a held-out largest scale")
    if not control_ok:
        refusal_reasons.append("negative-control transfer could not be distinguished from the proposed rule")
    forecast = None
    final_scaling_fit = None
    if forecastable:
        final_scaling_fit = fit_scaling_law(
            [row["estimated_training_compute"] for row in scale_summaries],
            [row["mean_final_validation_loss"] for row in scale_summaries],
            [row["sem_final_validation_loss"] for row in scale_summaries],
            bootstrap_samples=spec.validation.bootstrap_samples,
            random_seed=2028,
        )
        final_law_usable = final_scaling_fit.forecastable or final_scaling_fit.short_range_forecastable
        if not final_law_usable:
            forecastable = False
            refusal_reasons.extend(final_scaling_fit.refusal_reasons)
            final_scaling_fit = None
        else:
            lower, upper = final_scaling_fit.prediction_interval(next_compute)
            mode = (
                "asymptotic_floor_power_law"
                if final_scaling_fit.forecastable
                else "heldout_calibrated_one_step"
            )
            forecast = {
                "mode": mode,
                "estimated_training_compute": next_compute,
                "compute_ratio_from_largest_observation": next_compute / last_compute,
                "predicted_final_validation_loss": final_scaling_fit.predict(next_compute),
                "prediction_interval_95": [lower, upper],
            }

    result = {
        "schema_version": 1,
        "status": "completed",
        "study_fingerprint": spec.fingerprint,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "device": device,
        "reference_scale": reference_scale.name,
        "tuning": tuning.to_dict(),
        "transfer_rule": transfer_rule_name(spec.optimizer.name),
        "transfer_checks": transfer_checks,
        "negative_control": negative_control,
        "scale_results": scale_summaries,
        "scaling_law": scaling_fit.to_dict(),
        "final_scaling_law": final_scaling_fit.to_dict() if final_scaling_fit else None,
        "holdout_calibration": holdout_calibration,
        "forecastable": forecastable,
        "refusal_reasons": list(dict.fromkeys(refusal_reasons)),
        "warnings": warnings,
        "next_scale_forecast": forecast,
        "trials": [trial.to_dict() for trial in trials],
    }
    if output_dir is not None:
        atomic_write_json(output_dir / "result.json", result)
    _emit(progress, "completed", trial_counter, trial_counter, "Study complete")
    return _json_safe(result)
