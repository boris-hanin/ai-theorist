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
    MOE_TABLE1_ADAM,
    CHIZAT_MEAN_FIELD,
    STANDARD_RESIDUAL_MLP,
    adaptive_tune,
    fixed_eta_transfer_diagnostics,
    fixed_eta_noninferiority,
    optimizer_group_learning_rates_from_normalized_eta,
    parameterization_for_block_type,
    paired_mean_and_sem,
    raw_learning_rate_from_normalized_eta,
    summarize_trials,
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


def _fixed_eta_trajectory_report(
    spec: StudySpec,
    trials: Sequence[TrialResult],
    normalized_eta: float,
) -> Dict[str, Any]:
    selected = [
        trial
        for trial in trials
        if trial.normalized_learning_rate == normalized_eta
        and trial.optimizer_parameterization == "declared"
    ]
    expected = len(spec.scales) * len(spec.seeds)
    trace_maps = {
        (trial.scale, trial.seed): {
            int(row["step"]): float(row["validation_loss"])
            for row in trial.validation_loss_trace
        }
        for trial in selected
    }
    complete = len(trace_maps) == expected and all(
        spec.horizon.steps in trace_maps.get((scale.name, seed), {})
        for scale in spec.scales
        for seed in spec.seeds
    )
    if not trace_maps:
        return {"accepted": False, "complete": False, "checkpoints": []}
    common_steps = sorted(set.intersection(*(set(row) for row in trace_maps.values())))
    dials = [float(parameter_count(spec, scale)) for scale in spec.scales]
    checkpoints = []
    for step in common_steps:
        losses = [
            {seed: trace_maps[(scale.name, seed)][step] for seed in spec.seeds}
            for scale in spec.scales
        ]
        diagnostics = fixed_eta_transfer_diagnostics(dials, losses)
        checkpoints.append({"step": step, **diagnostics.to_dict()})
    eligible = [row for row in checkpoints if row["step"] > 1]
    return {
        "accepted": complete and bool(eligible) and all(row["accepted"] for row in eligible),
        "complete": complete,
        "dial": "parameter_count",
        "normalized_eta": normalized_eta,
        "checkpoints": checkpoints,
    }


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
    parameterization = parameterization_for_block_type(spec.architecture.block_type)

    def raw_group_rates(
        scale: ScaleLevel, eta: float, *, rule: str = "declared"
    ) -> Dict[str, float]:
        return optimizer_group_learning_rates_from_normalized_eta(
            parameterization,
            spec.optimizer.name,
            eta,
            width=scale.width,
            depth=scale.repeats,
            expert_width=scale.expert_width,
            particle_width=scale.particle_width,
            rule=rule,
        )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / "manifest.json", {"spec": spec.to_dict(), "plan": plan})

    fit_scales = spec.scales[:-spec.holdout_count]
    holdout_scales = spec.scales[-spec.holdout_count:]
    reference_scale = fit_scales[len(fit_scales) // 2]
    trials: List[TrialResult] = []
    trial_cache: Dict[Tuple[str, float, int, str], TrialResult] = {}
    trial_counter = 0
    estimated_total = int(plan["trial_budget_before_edge_expansion"])

    def run(
        scale: ScaleLevel,
        normalized_eta: float,
        seed: int,
        phase: str,
        optimizer_parameterization: str = "declared",
    ) -> TrialResult:
        nonlocal trial_counter
        key = (scale.name, float(normalized_eta), int(seed), optimizer_parameterization)
        if key not in trial_cache:
            raw_rate = raw_learning_rate_from_normalized_eta(
                parameterization,
                spec.optimizer.name,
                normalized_eta,
                width=scale.width,
                depth=scale.repeats,
            )
            result = train_trial(
                spec,
                scale,
                normalized_eta,
                seed,
                raw_learning_rate=raw_rate,
                force_global_learning_rate=(
                    normalized_eta / reference_scale.width
                    if optimizer_parameterization == "single_global_control"
                    and parameterization != CHIZAT_MEAN_FIELD
                    else None
                ),
                optimizer_rate_rule=(
                    optimizer_parameterization
                    if parameterization == CHIZAT_MEAN_FIELD
                    else "declared"
                ),
                device=device,
            )
            trial_cache[key] = result
            trials.append(result)
            trial_counter += 1
            _emit(
                progress,
                phase,
                trial_counter,
                estimated_total,
                f"{scale.name} · seed {seed} · eta {normalized_eta:.3g} · raw lr {raw_rate:.3g}",
            )
        return trial_cache[key]

    _emit(progress, "tuning", 0, estimated_total, f"Tuning {reference_scale.name}")
    tuning, _ = adaptive_tune(
        spec.tuning.normalized_learning_rates,
        spec.seeds,
        lambda rate, seed: run(reference_scale, rate, seed, "tuning"),
        max_expansion_rounds=spec.tuning.max_expansion_rounds,
        expansion_factor=spec.tuning.expansion_factor,
    )
    base_normalized_eta = tuning.selected_normalized_learning_rate

    scale_summaries: List[Dict[str, Any]] = []
    for scale in spec.scales:
        raw_rate = raw_learning_rate_from_normalized_eta(
            parameterization,
            spec.optimizer.name,
            base_normalized_eta,
            width=scale.width,
            depth=scale.repeats,
        )
        selected_trials = [
            run(scale, base_normalized_eta, seed, "transfer") for seed in spec.seeds
        ]
        summary = _summary(selected_trials, base_normalized_eta)
        routing_imbalances = [
            trial.max_routing_load_imbalance
            for trial in selected_trials
            if trial.max_routing_load_imbalance is not None
        ]
        scale_summaries.append(
            {
                "scale": scale.name,
                "width": scale.width,
                "repeats": scale.repeats,
                "expert_width": scale.expert_width,
                "particle_width": scale.particle_width,
                "parameter_count": parameter_count(spec, scale),
                "estimated_training_compute": estimate_training_compute(spec, scale),
                "normalized_learning_rate": base_normalized_eta,
                "raw_learning_rate": raw_rate,
                "raw_learning_rates": raw_group_rates(scale, base_normalized_eta),
                "mean_final_validation_loss": summary["mean_final_validation_loss"],
                "sem_final_validation_loss": summary["sem_final_validation_loss"],
                "losses_by_seed": summary["losses_by_seed"],
                "maximum_routing_load_imbalance": (
                    max(routing_imbalances) if routing_imbalances else None
                ),
                "mean_routing_load_imbalance": (
                    mean(routing_imbalances) if routing_imbalances else None
                ),
                "role": "holdout" if scale in holdout_scales else "fit",
            }
        )

    transfer_checks = []
    probe_multiplier = 10.0 ** spec.validation.transfer_probe_decades
    for scale in holdout_scales:
        candidates = (
            base_normalized_eta / probe_multiplier,
            base_normalized_eta,
            base_normalized_eta * probe_multiplier,
        )
        candidate_summaries = []
        candidate_trials: Dict[float, List[TrialResult]] = {}
        for eta in candidates:
            candidate_trials[eta] = [
                run(scale, eta, seed, "transfer-validation") for seed in spec.seeds
            ]
            candidate_summaries.append(summarize_trials(candidate_trials[eta], eta))
        finite = [item for item in candidate_summaries if math.isfinite(item.mean_final_validation_loss)]
        if not finite:
            raise RuntimeError(f"All transfer probes diverged at {scale.name}")
        local_best = min(finite, key=lambda item: item.mean_final_validation_loss)
        conservative, transferred, aggressive = candidate_summaries
        noninferiority = fixed_eta_noninferiority(
            transferred.losses_by_seed,
            conservative.losses_by_seed,
        )
        raw_rate = raw_learning_rate_from_normalized_eta(
            parameterization,
            spec.optimizer.name,
            base_normalized_eta,
            width=scale.width,
            depth=scale.repeats,
        )
        largest_finite_eta = max(item.normalized_learning_rate for item in finite)
        local_best_distance = abs(
            math.log10(local_best.normalized_learning_rate / base_normalized_eta)
        )
        transfer_checks.append(
            {
                "scale": scale.name,
                "normalized_learning_rate": base_normalized_eta,
                "raw_learning_rate": raw_rate,
                "raw_learning_rates": raw_group_rates(scale, base_normalized_eta),
                "acceptance_rule": "fixed_eta_noninferior_to_lower_conservative_probe",
                "paired_loss_penalty": noninferiority["paired_loss_penalty"],
                "paired_loss_penalty_sem": noninferiority["paired_loss_penalty_sem"],
                "noninferiority_tolerance": noninferiority["tolerance"],
                "accepted": noninferiority["accepted"],
                "edge_of_stability": {
                    "purpose": "diagnostic_only_not_a_transfer_gate",
                    "local_probe_best_normalized_eta": local_best.normalized_learning_rate,
                    "local_best_offset_decades": local_best_distance,
                    "largest_finite_probe_normalized_eta": largest_finite_eta,
                    "aggressive_probe_diverged": not math.isfinite(
                        aggressive.mean_final_validation_loss
                    ),
                },
                "candidates": [
                    {
                        "normalized_learning_rate": item.normalized_learning_rate,
                        "raw_learning_rate": raw_learning_rate_from_normalized_eta(
                            parameterization,
                            spec.optimizer.name,
                            item.normalized_learning_rate,
                            width=scale.width,
                            depth=scale.repeats,
                        ),
                        "raw_learning_rates": raw_group_rates(
                            scale, item.normalized_learning_rate
                        ),
                        **item.to_dict(),
                    }
                    for item in candidate_summaries
                ],
            }
        )

    fixed_eta_trajectory = _fixed_eta_trajectory_report(
        spec, trials, base_normalized_eta
    )

    negative_control: Optional[Dict[str, Any]] = None
    if spec.validation.run_negative_control:
        target = holdout_scales[-1]
        if parameterization == CHIZAT_MEAN_FIELD:
            wrong_eta = base_normalized_eta
            wrong_rule = {
                "muon": "wrong_W_D",
                "adam": "wrong_constant_unembed",
                "sgd": "wrong_global_LMD",
            }[spec.optimizer.name]
            control_trials = [
                run(
                    target,
                    wrong_eta,
                    seed,
                    "negative-control",
                    optimizer_parameterization=wrong_rule,
                )
                for seed in spec.seeds
            ]
            wrong_raw_rates = raw_group_rates(target, wrong_eta, rule=wrong_rule)
        elif parameterization == MOE_TABLE1_ADAM:
            wrong_rule = "incorrect_single_global_reference_up_rate"
            wrong_eta = base_normalized_eta
            control_trials = [
                run(
                    target,
                    wrong_eta,
                    seed,
                    "negative-control",
                    optimizer_parameterization="single_global_control",
                )
                for seed in spec.seeds
            ]
            wrong_global_rate = wrong_eta / reference_scale.width
            wrong_raw_rates = {"all": wrong_global_rate}
        elif spec.optimizer.name == "adam":
            wrong_rule = "incorrect_sqrt_width_learning_rate_growth"
            wrong_eta = base_normalized_eta * math.sqrt(
                target.width / reference_scale.width
            )
            control_trials = [
                run(target, wrong_eta, seed, "negative-control") for seed in spec.seeds
            ]
            wrong_raw_rates = raw_group_rates(target, wrong_eta)
        else:
            wrong_rule = "incorrect_constant_learning_rate"
            wrong_eta = base_normalized_eta * math.sqrt(
                target.width / reference_scale.width
            )
            control_trials = [
                run(target, wrong_eta, seed, "negative-control") for seed in spec.seeds
            ]
            wrong_raw_rates = raw_group_rates(target, wrong_eta)
        control = summarize_trials(control_trials, wrong_eta)
        baseline_trials = [
            trial_cache[(target.name, float(base_normalized_eta), seed, "declared")]
            for seed in spec.seeds
        ]
        baseline = summarize_trials(baseline_trials, base_normalized_eta)
        if math.isfinite(control.mean_final_validation_loss):
            difference, difference_sem = paired_mean_and_sem(control.losses_by_seed, baseline.losses_by_seed)
            rejected = difference > max(2.0 * difference_sem, 0.01 * baseline.mean_final_validation_loss)
        else:
            difference, difference_sem, rejected = float("inf"), float("inf"), True
        negative_control = {
            "rule": wrong_rule,
            "normalized_learning_rate": wrong_eta,
            "raw_learning_rate": (
                wrong_global_rate
                if parameterization == MOE_TABLE1_ADAM
                else raw_learning_rate_from_normalized_eta(
                    parameterization,
                    spec.optimizer.name,
                    wrong_eta,
                    width=target.width,
                    depth=target.repeats,
                )
            ),
            "raw_learning_rates": wrong_raw_rates,
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
    trajectory_ok = bool(fixed_eta_trajectory["accepted"])
    calibration_ok = all(check["accepted"] for check in holdout_calibration)
    control_ok = negative_control is None or bool(negative_control["rejected"])
    routing_rows = [
        {
            "scale": row["scale"],
            "maximum_routing_load_imbalance": row["maximum_routing_load_imbalance"],
            "mean_routing_load_imbalance": row["mean_routing_load_imbalance"],
        }
        for row in scale_summaries
        if row["maximum_routing_load_imbalance"] is not None
    ]
    routing_hard_limit = min(1.0, 2.0 * spec.validation.routing_load_tolerance)
    routing_ok = spec.architecture.block_type != "pre_norm_moe" or (
        bool(routing_rows)
        and all(
            float(row["mean_routing_load_imbalance"])
            <= spec.validation.routing_load_tolerance
            and float(row["maximum_routing_load_imbalance"]) <= routing_hard_limit
            for row in routing_rows
        )
    )
    calibration_law_usable = scaling_fit.forecastable or scaling_fit.short_range_forecastable
    forecastable = (
        tuning.optimum_is_interior
        and transfer_ok
        and trajectory_ok
        and calibration_law_usable
        and calibration_ok
        and control_ok
        and routing_ok
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
        refusal_reasons.append(
            "fixed normalized learning rate was inferior to a conservative largest-scale probe"
        )
    if not trajectory_ok:
        refusal_reasons.append(
            "fixed-eta validation trajectories did not settle across the scale ladder"
        )
    if not calibration_ok:
        refusal_reasons.append("scaling law missed a held-out largest scale")
    if not control_ok:
        refusal_reasons.append("negative-control transfer could not be distinguished from the proposed rule")
    if not routing_ok:
        refusal_reasons.append("MoE expert routing exceeded the declared load-imbalance tolerance")
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
        "schema_version": 2,
        "status": "completed",
        "study_fingerprint": spec.fingerprint,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "device": device,
        "dataset_contract": plan["dataset_contract"],
        "reference_scale": reference_scale.name,
        "learning_rate_coordinate": {
            "tuned": "normalized_eta",
            "parameterization": parameterization,
            "optimizer_raw_conversion": transfer_rule_name(
                spec.optimizer.name, parameterization
            ),
            "normalized_eta": base_normalized_eta,
        },
        "tuning": tuning.to_dict(),
        "transfer_rule": transfer_rule_name(spec.optimizer.name, parameterization),
        "transfer_checks": transfer_checks,
        "fixed_eta_trajectory": fixed_eta_trajectory,
        "negative_control": negative_control,
        "routing_quality": {
            "applicable": spec.architecture.block_type == "pre_norm_moe",
            "mean_worst_expert_tolerance": spec.validation.routing_load_tolerance,
            "individual_run_hard_limit": routing_hard_limit,
            "accepted": routing_ok,
            "scales": routing_rows,
        },
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
