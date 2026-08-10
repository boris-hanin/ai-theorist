from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .batch_campaigns import run_transformer_batch_trial
from .batch_scaling import (
    BatchRunRecord,
    OptimizerHyperparameters,
    TransferContext,
    apply_transfer_rule,
)
from .horizon_campaigns import _optimum, _rate_grid
from .lr_schedules import LearningRateSchedule
from .normalized_transformer import NormalizedTransformer
from .schema import ArchitectureTemplate, DatasetSpec, ScaleLevel


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]

JOINT_TRANSFER_RULES: Tuple[str, ...] = (
    "none",
    "horizon_fitted_only",
    "batch_fitted_only",
    "separable_fitted_peak",
    "horizon_fitted_x_adam_sde_batch",
    "one_third_x_adam_sde_batch",
    "complete_dp_joint",
    "exact_token_half_life_joint",
)

PARTIAL_CONTROL_RULES = {
    "none",
    "horizon_fitted_only",
    "batch_fitted_only",
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _optimizer(payload: Mapping[str, Any], learning_rate: float) -> OptimizerHyperparameters:
    return OptimizerHyperparameters(
        name=str(payload["name"]),
        learning_rate=learning_rate,
        momentum=float(payload.get("momentum", 0.0)),
        beta1=float(payload.get("beta1", 0.9)),
        beta2=float(payload.get("beta2", 0.999)),
        epsilon=float(payload.get("epsilon", 1e-8)),
        weight_decay=float(payload.get("weight_decay", 0.0)),
    )


def _progress(
    callback: ProgressCallback,
    phase: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": phase,
                "completed": completed,
                "total": total,
                "message": message,
            }
        )


def _coordinate_fit(
    optima: Sequence[Mapping[str, Any]], coordinate: str
) -> Dict[str, float]:
    x = np.log([float(row[coordinate]) for row in optima])
    y = np.log([float(row["interpolated_learning_rate"]) for row in optima])
    slope, intercept = np.polyfit(x, y, 1)
    predictions = intercept + slope * x
    residual = float(np.sum((y - predictions) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = (
        1.0
        if total <= 1e-20 and residual <= 1e-20
        else 1.0 - residual / max(total, 1e-20)
    )
    return {
        "slope": float(slope),
        "log_coefficient": float(intercept),
        "coefficient": float(math.exp(intercept)),
        "r_squared": r_squared,
    }


def _bootstrap_coordinate_fit(
    optima: Sequence[Mapping[str, Any]],
    coordinate: str,
    *,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    seed_count = len(optima[0]["rate_grid"][0]["seed_losses"])
    if seed_count < 2 or samples <= 0:
        return {"samples": 0, "slope_interval_95": None}
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(samples):
        indices = rng.integers(0, seed_count, size=seed_count)
        sampled_optima = []
        for optimum in optima:
            sampled_grid = []
            for row in optimum["rate_grid"]:
                losses = np.asarray(row["seed_losses"], dtype=np.float64)[indices]
                sampled_grid.append(
                    {
                        "learning_rate": row["learning_rate"],
                        "mean_loss": float(np.mean(losses)),
                    }
                )
            sampled = _optimum(sampled_grid)
            sampled[coordinate] = optimum[coordinate]
            sampled_optima.append(sampled)
        slopes.append(_coordinate_fit(sampled_optima, coordinate)["slope"])
    return {
        "samples": samples,
        "slope_interval_95": [
            float(np.quantile(slopes, 0.025)),
            float(np.quantile(slopes, 0.975)),
        ],
        "bootstrap_slopes": slopes,
    }


def _replace_adam(
    source: OptimizerHyperparameters,
    *,
    learning_rate_multiplier: float,
    epsilon_multiplier: float = 1.0,
    beta_gap_multiplier: Optional[float] = None,
) -> Optional[OptimizerHyperparameters]:
    beta1 = source.beta1
    beta2 = source.beta2
    if beta_gap_multiplier is not None:
        beta1 = 1.0 - beta_gap_multiplier * (1.0 - source.beta1)
        beta2 = 1.0 - beta_gap_multiplier * (1.0 - source.beta2)
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        return None
    return OptimizerHyperparameters(
        name=source.name,
        learning_rate=source.learning_rate * learning_rate_multiplier,
        momentum=source.momentum,
        beta1=beta1,
        beta2=beta2,
        epsilon=source.epsilon * epsilon_multiplier,
        weight_decay=source.weight_decay,
    )


def build_joint_optimizer(
    rule: str,
    source: OptimizerHyperparameters,
    *,
    source_tokens: int,
    target_tokens: int,
    source_batch_tokens: int,
    target_batch_tokens: int,
    parameter_count: int,
    horizon_exponent: float,
    batch_exponent: float,
) -> Dict[str, Any]:
    """Freeze one inspectable optimizer prediction for a target (T, B) cell."""
    if rule not in JOINT_TRANSFER_RULES:
        raise ValueError(f"unknown joint transfer rule: {rule}")
    horizon_multiplier = target_tokens / source_tokens
    batch_multiplier = target_batch_tokens / source_batch_tokens
    valid = True
    refusal_reasons: List[str] = []
    optimizer: Optional[OptimizerHyperparameters]
    multipliers: Dict[str, float]
    assumptions: Tuple[str, ...]

    if rule == "none":
        optimizer = source
        multipliers = {"learning_rate": 1.0, "epsilon": 1.0, "beta_gap": 1.0}
        assumptions = ("All optimizer coordinates remain fixed.",)
    elif rule == "horizon_fitted_only":
        factor = horizon_multiplier ** (-horizon_exponent)
        optimizer = _replace_adam(source, learning_rate_multiplier=factor)
        multipliers = {"learning_rate": factor, "horizon_exponent": horizon_exponent}
        assumptions = ("Only the fitted token-horizon peak-LR effect is applied.",)
    elif rule == "batch_fitted_only":
        factor = batch_multiplier**batch_exponent
        optimizer = _replace_adam(source, learning_rate_multiplier=factor)
        multipliers = {"learning_rate": factor, "batch_exponent": batch_exponent}
        assumptions = ("Only the fitted batch peak-LR effect is applied.",)
    elif rule == "separable_fitted_peak":
        factor = horizon_multiplier ** (-horizon_exponent) * batch_multiplier**batch_exponent
        optimizer = _replace_adam(source, learning_rate_multiplier=factor)
        multipliers = {
            "learning_rate": factor,
            "horizon_exponent": horizon_exponent,
            "batch_exponent": batch_exponent,
        }
        assumptions = (
            "The independently fitted horizon and batch peak-LR effects compose multiplicatively.",
            "Adam moments and epsilon remain fixed.",
        )
    elif rule in {
        "horizon_fitted_x_adam_sde_batch",
        "one_third_x_adam_sde_batch",
    }:
        exponent = horizon_exponent if rule.startswith("horizon_fitted") else 1.0 / 3.0
        factor = horizon_multiplier ** (-exponent) * math.sqrt(batch_multiplier)
        optimizer = _replace_adam(
            source,
            learning_rate_multiplier=factor,
            epsilon_multiplier=1.0 / math.sqrt(batch_multiplier),
            beta_gap_multiplier=batch_multiplier,
        )
        multipliers = {
            "learning_rate": factor,
            "horizon_exponent": exponent,
            "batch_exponent": 0.5,
            "epsilon": 1.0 / math.sqrt(batch_multiplier),
            "beta_gap": batch_multiplier,
        }
        assumptions = (
            "The horizon peak-LR factor composes with the fixed-duration Adam SDE batch rule.",
            "The target remains in the SDE-valid subcritical-batch regime.",
        )
        if optimizer is None:
            valid = False
            refusal_reasons.append("scaled Adam beta gap leaves [0, 1)")
    else:
        context = TransferContext(
            base_parameters=parameter_count,
            target_parameters=parameter_count,
            base_total_tokens=source_tokens,
            target_total_tokens=target_tokens,
            base_batch_tokens=source_batch_tokens,
            target_batch_tokens=target_batch_tokens,
        )
        registry_rule = (
            "complete_dp_joint"
            if rule == "complete_dp_joint"
            else "exact_token_half_life"
        )
        transferred = apply_transfer_rule(registry_rule, source, context)
        optimizer = transferred.target
        multipliers = dict(transferred.multipliers)
        assumptions = transferred.assumptions
        valid = transferred.valid
        refusal_reasons.extend(transferred.refusal_reasons)

    return {
        "rule": rule,
        "joint_rule": rule not in PARTIAL_CONTROL_RULES,
        "valid": valid and optimizer is not None,
        "optimizer": optimizer,
        "multipliers": multipliers,
        "assumptions": list(assumptions),
        "refusal_reasons": refusal_reasons,
    }


def compile_joint_transfer_plan(config: Mapping[str, Any]) -> Dict[str, Any]:
    fit_tokens = tuple(
        _positive_int(int(value), "fit_presented_tokens")
        for value in config["fit_presented_tokens"]
    )
    fit_batches = tuple(
        _positive_int(int(value), "fit_batch_examples")
        for value in config["fit_batch_examples"]
    )
    if len(fit_tokens) < 3 or tuple(sorted(set(fit_tokens))) != fit_tokens:
        raise ValueError("fit_presented_tokens must contain at least three increasing values")
    if len(fit_batches) < 3 or tuple(sorted(set(fit_batches))) != fit_batches:
        raise ValueError("fit_batch_examples must contain at least three increasing values")
    heldout_tokens = _positive_int(
        int(config["heldout_presented_tokens"]), "heldout_presented_tokens"
    )
    heldout_batch = _positive_int(
        int(config["heldout_batch_examples"]), "heldout_batch_examples"
    )
    if heldout_tokens <= fit_tokens[-1] or heldout_batch <= fit_batches[-1]:
        raise ValueError("the held-out horizon and batch must exceed every fit value")
    rates = tuple(
        _positive_float(value, "learning_rate")
        for value in config["optimizer"]["learning_rates"]
    )
    if len(rates) < 3 or tuple(sorted(set(rates))) != rates:
        raise ValueError("optimizer.learning_rates must contain at least three increasing values")
    rules = tuple(str(value) for value in config.get("joint_rules", JOINT_TRANSFER_RULES))
    unknown = [rule for rule in rules if rule not in JOINT_TRANSFER_RULES]
    if unknown:
        raise ValueError(f"unknown joint rule(s): {', '.join(unknown)}")
    missing_controls = PARTIAL_CONTROL_RULES.difference(rules)
    if missing_controls:
        raise ValueError(
            "joint_rules must include the mechanism controls: "
            + ", ".join(sorted(missing_controls))
        )
    schedule = LearningRateSchedule.from_payload(config["schedule"])
    seeds = tuple(int(value) for value in config.get("seeds", [11, 29, 47]))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    architecture = ArchitectureTemplate.from_dict(dict(config["architecture"]))
    context = architecture.context_length
    required_cells = [
        *((tokens, fit_batches[0]) for tokens in fit_tokens),
        *((fit_tokens[0], batch) for batch in fit_batches[1:]),
        (fit_tokens[-1], fit_batches[-1]),
        (heldout_tokens, heldout_batch),
    ]
    for tokens, batch in required_cells:
        if tokens % (batch * context):
            raise ValueError(
                f"presented tokens {tokens} must be divisible by batch tokens {batch * context}"
            )
    expansion_rounds = int(config.get("maximum_grid_expansion_rounds", 2))
    if expansion_rounds < 0:
        raise ValueError("maximum_grid_expansion_rounds cannot be negative")
    calibration_cells = len(fit_tokens) + len(fit_batches) - 1
    calibration_trials = calibration_cells * len(rates) * len(seeds)
    candidate_trials = 2 * len(rules) * len(seeds)
    oracle_trials = 2 * len(rates) * len(seeds)
    maximum_expansion_trials = (
        (calibration_cells + 2) * expansion_rounds * len(seeds)
    )
    return {
        "schema_version": 1,
        "campaign": "joint_horizon_batch_transfer",
        "schedule_name": schedule.name,
        "fit_presented_tokens": list(fit_tokens),
        "fit_batch_examples": list(fit_batches),
        "composition_crosscheck": {
            "presented_tokens": fit_tokens[-1],
            "batch_examples": fit_batches[-1],
        },
        "heldout_corner": {
            "presented_tokens": heldout_tokens,
            "batch_examples": heldout_batch,
        },
        "joint_rules": list(rules),
        "calibration_trials": calibration_trials,
        "candidate_trials": candidate_trials,
        "oracle_trials": oracle_trials,
        "maximum_grid_expansion_trials": maximum_expansion_trials,
        "planned_grid_trials": (
            calibration_trials
            + candidate_trials
            + oracle_trials
            + maximum_expansion_trials
        ),
        "execution_order": [
            "fit_horizon_axis_at_base_batch",
            "fit_batch_axis_at_base_horizon",
            "freeze_joint_rules",
            "crosscheck_frozen_rules_at_unseen_fit_rectangle_corner",
            "reveal_crosscheck_oracle_and_filter_rules",
            "evaluate_frozen_rules_at_doubly_heldout_corner",
            "reveal_heldout_oracle_for_regret_only",
        ],
    }


def run_joint_transfer_campaign(
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    """Fit axis effects, cross-check composition, then test one hidden (T, B) corner."""
    plan = compile_joint_transfer_plan(config)
    architecture = ArchitectureTemplate.from_dict(dict(config["architecture"]))
    if architecture.block_type != "normalized_transformer":
        raise ValueError("joint horizon/batch transfer currently requires normalized_transformer")
    dataset = DatasetSpec.from_dict(dict(config["dataset"]))
    scale = ScaleLevel.from_dict(dict(config["scale"]), 0)
    optimizer_payload = dict(config["optimizer"])
    if optimizer_payload.get("name") != "adam":
        raise ValueError("joint normalized-Transformer transfer currently requires Adam")
    rates = tuple(float(value) for value in optimizer_payload["learning_rates"])
    fit_tokens = tuple(int(value) for value in config["fit_presented_tokens"])
    heldout_tokens = int(config["heldout_presented_tokens"])
    fit_batches = tuple(int(value) for value in config["fit_batch_examples"])
    heldout_batch = int(config["heldout_batch_examples"])
    schedule = LearningRateSchedule.from_payload(config["schedule"])
    rules = tuple(str(value) for value in config.get("joint_rules", JOINT_TRANSFER_RULES))
    seeds = tuple(int(value) for value in config.get("seeds", [11, 29, 47]))
    validation_interval = _positive_int(
        config.get("validation_interval", 8), "validation_interval"
    )
    cache_directory = Path(config["cache_directory"]) if config.get("cache_directory") else None
    expansion_rounds = int(config.get("maximum_grid_expansion_rounds", 2))
    expansion_factor = _positive_float(
        config.get("grid_expansion_factor", 3.0), "grid_expansion_factor"
    )
    if expansion_factor <= 1.0:
        raise ValueError("grid_expansion_factor must exceed one")
    minimum_seeds = _positive_int(config.get("minimum_seeds", 3), "minimum_seeds")
    minimum_horizon_span = _positive_float(
        config.get("minimum_fit_horizon_span", 4.0), "minimum_fit_horizon_span"
    )
    minimum_batch_span = _positive_float(
        config.get("minimum_fit_batch_span", 4.0), "minimum_fit_batch_span"
    )
    minimum_r_squared = float(config.get("minimum_axis_fit_r_squared", 0.8))
    if not 0.0 <= minimum_r_squared <= 1.0:
        raise ValueError("minimum_axis_fit_r_squared must be in [0, 1]")
    bootstrap_samples = int(config.get("bootstrap_samples", 400))
    maximum_crosscheck_regret = _positive_float(
        config.get("maximum_crosscheck_regret", 0.02), "maximum_crosscheck_regret"
    )
    maximum_heldout_regret = _positive_float(
        config.get("maximum_relative_oracle_regret", 0.02),
        "maximum_relative_oracle_regret",
    )
    minimum_recovery = float(config.get("minimum_recovered_improvement", 0.9))
    if not 0.0 <= minimum_recovery <= 1.0:
        raise ValueError("minimum_recovered_improvement must be in [0, 1]")
    identifiability_tolerance = _positive_float(
        config.get("partial_control_relative_tolerance", 0.002),
        "partial_control_relative_tolerance",
    )
    qualified_critical_batch_tokens = (
        _positive_float(
            config["qualified_critical_batch_tokens"],
            "qualified_critical_batch_tokens",
        )
        if config.get("qualified_critical_batch_tokens") is not None
        else None
    )
    maximum_critical_batch_fraction = float(
        config.get("maximum_critical_batch_fraction", 0.8)
    )
    if not 0.0 < maximum_critical_batch_fraction <= 1.0:
        raise ValueError("maximum_critical_batch_fraction must be in (0, 1]")

    probe = NormalizedTransformer(architecture, scale)
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    del probe
    unique_tokens = dataset.n_train * architecture.context_length
    source_tokens = fit_tokens[0]
    source_batch_examples = fit_batches[0]
    source_batch_tokens = source_batch_examples * architecture.context_length
    total = int(plan["planned_grid_trials"])
    completed = 0
    records_by_id: Dict[str, BatchRunRecord] = {}
    trials_by_key: Dict[
        Tuple[int, int, OptimizerHyperparameters, int], BatchRunRecord
    ] = {}

    def run_trial(
        *,
        tokens: int,
        batch_examples: int,
        optimizer: OptimizerHyperparameters,
        seed: int,
        phase: str,
    ) -> BatchRunRecord:
        nonlocal completed
        trial_key = (tokens, batch_examples, optimizer, seed)
        if trial_key in trials_by_key:
            completed += 1
            _progress(
                progress,
                phase,
                completed,
                total,
                f"T={tokens:,}, B={batch_examples * architecture.context_length:,} (reused)",
            )
            return trials_by_key[trial_key]
        record, _ = run_transformer_batch_trial(
            architecture=architecture,
            dataset=dataset,
            scale=scale,
            optimizer=optimizer,
            total_tokens=tokens,
            batch_examples=batch_examples,
            seed=seed,
            validation_interval=validation_interval,
            learning_rate_schedule=asdict(schedule),
            gradient_clip_norm=None,
            device=device,
            cache_directory=cache_directory,
            cache_key_suffix=f"-joint-{schedule.name}",
        )
        records_by_id[record.run_id] = record
        trials_by_key[trial_key] = record
        completed += 1
        _progress(
            progress,
            phase,
            completed,
            total,
            f"T={tokens:,}, B={batch_examples * architecture.context_length:,}",
        )
        return record

    tuned_cells: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def tune_cell(tokens: int, batch_examples: int, phase: str) -> Dict[str, Any]:
        key = (tokens, batch_examples)
        if key in tuned_cells:
            return tuned_cells[key]
        tested_rates = list(rates)
        cell_records: List[BatchRunRecord] = []
        for expansion_round in range(expansion_rounds + 1):
            already_tested = {record.optimizer.learning_rate for record in cell_records}
            for rate in tested_rates:
                if rate in already_tested:
                    continue
                optimizer = _optimizer(optimizer_payload, rate)
                for seed in seeds:
                    cell_records.append(
                        run_trial(
                            tokens=tokens,
                            batch_examples=batch_examples,
                            optimizer=optimizer,
                            seed=seed,
                            phase=phase,
                        )
                    )
            tested_rates.sort()
            optimum = _optimum(_rate_grid(cell_records, tested_rates))
            if optimum["optimum_is_interior"] or expansion_round == expansion_rounds:
                optimum.update(
                    {
                        "presented_tokens": tokens,
                        "batch_examples": batch_examples,
                        "batch_tokens": batch_examples * architecture.context_length,
                        "optimizer_steps": tokens // (batch_examples * architecture.context_length),
                        "grid_expansion_rounds": expansion_round,
                    }
                )
                tuned_cells[key] = optimum
                return optimum
            if math.isclose(optimum["learning_rate"], tested_rates[0], rel_tol=1e-12):
                tested_rates.append(tested_rates[0] / expansion_factor)
            else:
                tested_rates.append(tested_rates[-1] * expansion_factor)
        raise AssertionError("unreachable LR-grid expansion state")

    horizon_optima = [
        tune_cell(tokens, source_batch_examples, "fit-horizon-axis")
        for tokens in fit_tokens
    ]
    batch_optima = [horizon_optima[0]] + [
        tune_cell(source_tokens, batch, "fit-batch-axis")
        for batch in fit_batches[1:]
    ]
    horizon_fit = _coordinate_fit(horizon_optima, "presented_tokens")
    batch_fit = _coordinate_fit(batch_optima, "batch_tokens")
    horizon_exponent = -horizon_fit["slope"]
    batch_exponent = batch_fit["slope"]
    horizon_bootstrap = _bootstrap_coordinate_fit(
        horizon_optima,
        "presented_tokens",
        samples=bootstrap_samples,
        seed=191_071,
    )
    batch_bootstrap = _bootstrap_coordinate_fit(
        batch_optima,
        "batch_tokens",
        samples=bootstrap_samples,
        seed=191_072,
    )
    common_reasons = []
    if len(seeds) < minimum_seeds:
        common_reasons.append(f"requires at least {minimum_seeds} seeds")
    if fit_tokens[-1] / fit_tokens[0] < minimum_horizon_span:
        common_reasons.append("fit horizon span is below the preregistered minimum")
    if fit_batches[-1] / fit_batches[0] < minimum_batch_span:
        common_reasons.append("fit batch span is below the preregistered minimum")
    if not horizon_optima[0]["optimum_is_interior"]:
        common_reasons.append("the shared source LR optimum is on the grid boundary")
    horizon_fit_reasons = list(common_reasons)
    if not all(row["optimum_is_interior"] for row in horizon_optima):
        horizon_fit_reasons.append("at least one horizon-axis LR optimum is on the grid boundary")
    batch_fit_reasons = list(common_reasons)
    if not all(row["optimum_is_interior"] for row in batch_optima):
        batch_fit_reasons.append("at least one batch-axis LR optimum is on the grid boundary")
    if horizon_exponent < 0.0:
        horizon_fit_reasons.append("optimal peak LR increases with token horizon")
    if batch_exponent < 0.0:
        batch_fit_reasons.append("optimal peak LR decreases with batch")
    if horizon_fit["r_squared"] < minimum_r_squared:
        horizon_fit_reasons.append("horizon-axis power fit is below the R-squared gate")
    if batch_fit["r_squared"] < minimum_r_squared:
        batch_fit_reasons.append("batch-axis power fit is below the R-squared gate")
    if horizon_bootstrap["samples"] == 0:
        horizon_fit_reasons.append("horizon paired-seed bootstrap uncertainty is unavailable")
    if batch_bootstrap["samples"] == 0:
        batch_fit_reasons.append("batch paired-seed bootstrap uncertainty is unavailable")
    fit_reasons = list(dict.fromkeys([*horizon_fit_reasons, *batch_fit_reasons]))
    rule_prerequisites = {
        "none": list(common_reasons),
        "horizon_fitted_only": list(horizon_fit_reasons),
        "batch_fitted_only": list(batch_fit_reasons),
        "separable_fitted_peak": list(fit_reasons),
        "horizon_fitted_x_adam_sde_batch": list(horizon_fit_reasons),
        "one_third_x_adam_sde_batch": list(common_reasons),
        "complete_dp_joint": list(common_reasons),
        "exact_token_half_life_joint": list(common_reasons),
    }

    source_optimizer = _optimizer(
        optimizer_payload, float(horizon_optima[0]["interpolated_learning_rate"])
    )

    def evaluate_candidates(
        *,
        tokens: int,
        batch_examples: int,
        phase: str,
    ) -> List[Dict[str, Any]]:
        rows = []
        target_batch_tokens = batch_examples * architecture.context_length
        for rule in rules:
            frozen = build_joint_optimizer(
                rule,
                source_optimizer,
                source_tokens=source_tokens,
                target_tokens=tokens,
                source_batch_tokens=source_batch_tokens,
                target_batch_tokens=target_batch_tokens,
                parameter_count=parameter_count,
                horizon_exponent=horizon_exponent,
                batch_exponent=batch_exponent,
            )
            optimizer = frozen.pop("optimizer")
            frozen["prerequisite_refusal_reasons"] = list(
                rule_prerequisites[rule]
            )
            requires_subcritical_batch = rule in {
                "horizon_fitted_x_adam_sde_batch",
                "one_third_x_adam_sde_batch",
            }
            frozen["theory_assumption_status"] = (
                "not_applicable_empirical_rule"
                if not requires_subcritical_batch
                else "qualified"
                if qualified_critical_batch_tokens is not None
                and target_batch_tokens
                <= maximum_critical_batch_fraction * qualified_critical_batch_tokens
                else "unverified_or_outside_subcritical_gate"
            )
            if not frozen["valid"] or optimizer is None:
                rows.append({**frozen, "evaluated": False})
                continue
            candidate_records = [
                run_trial(
                    tokens=tokens,
                    batch_examples=batch_examples,
                    optimizer=optimizer,
                    seed=seed,
                    phase=phase,
                )
                for seed in seeds
            ]
            rows.append(
                {
                    **frozen,
                    "evaluated": True,
                    "optimizer": optimizer.to_dict(),
                    "mean_loss": float(
                        np.mean([record.final_validation_loss for record in candidate_records])
                    ),
                    "seed_losses": [
                        record.final_validation_loss for record in candidate_records
                    ],
                    "peak_parameter_group_contract": candidate_records[0].metadata[
                        "peak_parameter_group_contract"
                    ],
                }
            )
        return rows

    cross_tokens = fit_tokens[-1]
    cross_batch = fit_batches[-1]
    cross_candidates = evaluate_candidates(
        tokens=cross_tokens,
        batch_examples=cross_batch,
        phase="composition-crosscheck-rules",
    )
    cross_oracle = tune_cell(
        cross_tokens, cross_batch, "composition-crosscheck-oracle"
    )
    cross_scoring_rows = [
        {"source": "lr_grid", "rule": None, "mean_loss": cross_oracle["mean_loss"]},
        *[
            {"source": "frozen_rule", "rule": row["rule"], "mean_loss": row["mean_loss"]}
            for row in cross_candidates
            if row.get("evaluated")
        ],
    ]
    cross_scoring_oracle = min(cross_scoring_rows, key=lambda row: row["mean_loss"])
    cross_oracle_loss = float(cross_scoring_oracle["mean_loss"])
    cross_pass: Dict[str, bool] = {}
    for row in cross_candidates:
        if not row.get("evaluated"):
            row["relative_oracle_regret"] = None
            row["composition_crosscheck_passed"] = False
        else:
            row["relative_oracle_regret"] = row["mean_loss"] / cross_oracle_loss - 1.0
            row["composition_crosscheck_passed"] = (
                not row["prerequisite_refusal_reasons"]
                and row["joint_rule"]
                and cross_oracle["optimum_is_interior"]
                and row["relative_oracle_regret"] <= maximum_crosscheck_regret
            )
        cross_pass[row["rule"]] = bool(row["composition_crosscheck_passed"])

    heldout_candidates = evaluate_candidates(
        tokens=heldout_tokens,
        batch_examples=heldout_batch,
        phase="heldout-frozen-rules",
    )
    heldout_oracle = tune_cell(
        heldout_tokens, heldout_batch, "heldout-oracle"
    )
    heldout_scoring_rows = [
        {"source": "lr_grid", "rule": None, "mean_loss": heldout_oracle["mean_loss"]},
        *[
            {"source": "frozen_rule", "rule": row["rule"], "mean_loss": row["mean_loss"]}
            for row in heldout_candidates
            if row.get("evaluated")
        ],
    ]
    heldout_scoring_oracle = min(
        heldout_scoring_rows, key=lambda row: row["mean_loss"]
    )
    oracle_loss = float(heldout_scoring_oracle["mean_loss"])
    partial_losses = [
        float(row["mean_loss"])
        for row in heldout_candidates
        if row["rule"] in PARTIAL_CONTROL_RULES and row.get("evaluated")
    ]
    best_partial_loss = min(partial_losses)
    available_joint_improvement = best_partial_loss - oracle_loss
    composition_identifiable = (
        available_joint_improvement > identifiability_tolerance * oracle_loss
    )
    for row in heldout_candidates:
        row["composition_crosscheck_passed"] = cross_pass.get(row["rule"], False)
        if not row.get("evaluated"):
            row.update(
                {
                    "relative_oracle_regret": None,
                    "recovered_joint_improvement_fraction": None,
                    "transfer_certified": False,
                    "mechanism_discrimination_certified": False,
                }
            )
            continue
        loss = float(row["mean_loss"])
        row["relative_oracle_regret"] = loss / oracle_loss - 1.0
        row["recovered_joint_improvement_fraction"] = (
            (best_partial_loss - loss) / available_joint_improvement
            if composition_identifiable
            else None
        )
        row["transfer_certified"] = (
            not row["prerequisite_refusal_reasons"]
            and row["joint_rule"]
            and row["composition_crosscheck_passed"]
            and heldout_oracle["optimum_is_interior"]
            and row["relative_oracle_regret"] <= maximum_heldout_regret
        )
        row["mechanism_discrimination_certified"] = (
            row["transfer_certified"]
            and composition_identifiable
            and float(row["recovered_joint_improvement_fraction"]) >= minimum_recovery
        )
        row["theory_transfer_certified"] = (
            row["transfer_certified"]
            and row["theory_assumption_status"] == "qualified"
        )

    certified = [row for row in heldout_candidates if row["transfer_certified"]]
    certified.sort(key=lambda row: row["mean_loss"])
    mechanism_certified = [
        row for row in certified if row["mechanism_discrimination_certified"]
    ]
    recommendation = mechanism_certified[0] if mechanism_certified else (certified[0] if certified else None)
    theory_certified = [
        row
        for row in mechanism_certified
        if row.get("theory_transfer_certified")
    ]
    geometry = []
    for role, tokens, batch_examples in [
        *(("horizon_fit", tokens, source_batch_examples) for tokens in fit_tokens),
        *(("batch_fit", source_tokens, batch) for batch in fit_batches[1:]),
        ("composition_crosscheck", cross_tokens, cross_batch),
        ("doubly_heldout", heldout_tokens, heldout_batch),
    ]:
        batch_tokens = batch_examples * architecture.context_length
        geometry.append(
            {
                "role": role,
                "parameters": parameter_count,
                "unique_tokens": unique_tokens,
                "presented_tokens": tokens,
                "batch_examples": batch_examples,
                "batch_tokens": batch_tokens,
                "optimizer_steps": tokens // batch_tokens,
                "tokens_per_parameter": tokens / parameter_count,
                "presented_to_unique_token_ratio": tokens / unique_tokens,
            }
        )
    _progress(progress, "complete", completed, completed, "Joint transfer campaign complete")
    return {
        "schema_version": 1,
        "status": "completed",
        "campaign": "joint_horizon_batch_transfer",
        "device": device,
        "config": dict(config),
        "plan": plan,
        "schedule": asdict(schedule),
        "schedule_name": schedule.name,
        "coordinates": {
            "N": "trainable parameters",
            "U": "unique training tokens",
            "T": "presented training tokens",
            "B": "tokens per optimizer update",
            "S": "optimizer updates = T / B",
        },
        "fixed_coordinates": {
            "parameters": parameter_count,
            "unique_tokens": unique_tokens,
            "scale": asdict(scale),
        },
        "geometry": geometry,
        "axis_fit_qualification": {
            "qualified": not fit_reasons,
            "refusal_reasons": fit_reasons,
            "horizon_qualified": not horizon_fit_reasons,
            "horizon_refusal_reasons": horizon_fit_reasons,
            "batch_qualified": not batch_fit_reasons,
            "batch_refusal_reasons": batch_fit_reasons,
            "common_refusal_reasons": common_reasons,
            "horizon_exponent": horizon_exponent,
            "batch_exponent": batch_exponent,
            "horizon_fit": horizon_fit,
            "batch_fit": batch_fit,
            "horizon_bootstrap": horizon_bootstrap,
            "batch_bootstrap": batch_bootstrap,
            "horizon_optima": horizon_optima,
            "batch_optima": batch_optima,
        },
        "composition_crosscheck": {
            "presented_tokens": cross_tokens,
            "batch_examples": cross_batch,
            "batch_tokens": cross_batch * architecture.context_length,
            "oracle": cross_oracle,
            "scoring_oracle": cross_scoring_oracle,
            "candidate_results": cross_candidates,
        },
        "heldout_corner": {
            "presented_tokens": heldout_tokens,
            "batch_examples": heldout_batch,
            "batch_tokens": heldout_batch * architecture.context_length,
            "oracle": heldout_oracle,
            "scoring_oracle": heldout_scoring_oracle,
            "best_partial_control_loss": best_partial_loss,
            "composition_identifiable": composition_identifiable,
            "candidate_results": heldout_candidates,
        },
        "certified_joint_rules": certified,
        "mechanism_certified_joint_rules": mechanism_certified,
        "joint_transfer_settled": bool(mechanism_certified),
        "empirical_joint_transfer_settled": bool(mechanism_certified),
        "theory_joint_transfer_settled": bool(theory_certified),
        "critical_batch_contract": {
            "qualified_critical_batch_tokens": qualified_critical_batch_tokens,
            "maximum_critical_batch_fraction": maximum_critical_batch_fraction,
            "heldout_batch_tokens": heldout_batch * architecture.context_length,
            "subcritical_gate_passed": (
                qualified_critical_batch_tokens is not None
                and heldout_batch * architecture.context_length
                <= maximum_critical_batch_fraction * qualified_critical_batch_tokens
            ),
        },
        "recommendation": recommendation,
        "joint_recommendation": recommendation,
        "refusal_reasons": (
            []
            if recommendation is not None
            else ["no joint rule passed every axis, cross-check, and held-out gate"]
        ),
        "execution_order": plan["execution_order"],
        "records": [record.to_dict() for record in records_by_id.values()],
    }
