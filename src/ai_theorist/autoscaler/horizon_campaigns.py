from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .batch_campaigns import run_transformer_batch_trial
from .batch_scaling import BatchRunRecord, OptimizerHyperparameters
from .lr_schedules import LearningRateSchedule
from .normalized_transformer import NormalizedTransformer
from .schema import ArchitectureTemplate, DatasetSpec, ScaleLevel


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
HORIZON_RULES: Dict[str, Optional[float]] = {
    "none": 0.0,
    "nugpt_one_third": 1.0 / 3.0,
    "bjorck_032": 0.32,
    "fitted_power": None,
}


@dataclass(frozen=True)
class BudgetGeometry:
    parameters: int
    unique_tokens: int
    presented_tokens: int
    batch_tokens: int
    optimizer_steps: int

    @property
    def tokens_per_parameter(self) -> float:
        return self.presented_tokens / self.parameters

    @property
    def repetition_ratio(self) -> float:
        return self.presented_tokens / self.unique_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "tokens_per_parameter": self.tokens_per_parameter,
            "presented_to_unique_token_ratio": self.repetition_ratio,
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


def _rate_grid(
    records: Sequence[BatchRunRecord], learning_rates: Sequence[float]
) -> List[Dict[str, Any]]:
    rows = []
    for learning_rate in learning_rates:
        matching = [
            record
            for record in records
            if math.isclose(record.optimizer.learning_rate, learning_rate, rel_tol=1e-12)
        ]
        if not matching:
            raise RuntimeError(f"missing trials for learning rate {learning_rate:g}")
        losses = [record.final_validation_loss for record in matching]
        rows.append(
            {
                "learning_rate": learning_rate,
                "mean_loss": float(np.mean(losses)),
                "sem_loss": float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
                if len(losses) > 1
                else 0.0,
                "seed_losses": losses,
            }
        )
    return rows


def _optimum(rate_grid: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    index = min(range(len(rate_grid)), key=lambda item: float(rate_grid[item]["mean_loss"]))
    selected = rate_grid[index]
    interpolated = float(selected["learning_rate"])
    interpolation_used = False
    if 0 < index < len(rate_grid) - 1:
        neighborhood = rate_grid[index - 1 : index + 2]
        x = np.log([float(row["learning_rate"]) for row in neighborhood])
        y = np.asarray([float(row["mean_loss"]) for row in neighborhood])
        quadratic = np.polyfit(x, y, 2)
        if quadratic[0] > 0.0:
            vertex = -quadratic[1] / (2.0 * quadratic[0])
            if x[0] <= vertex <= x[-1]:
                interpolated = float(math.exp(vertex))
                interpolation_used = True
    return {
        "learning_rate": float(selected["learning_rate"]),
        "interpolated_learning_rate": interpolated,
        "mean_loss": float(selected["mean_loss"]),
        "optimum_is_interior": 0 < index < len(rate_grid) - 1,
        "interpolation_used": interpolation_used,
        "rate_grid": [dict(row) for row in rate_grid],
    }


def _power_fit(optima: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    x = np.log([float(row["presented_tokens"]) for row in optima])
    y = np.log([float(row["interpolated_learning_rate"]) for row in optima])
    slope, intercept = np.polyfit(x, y, 1)
    predictions = intercept + slope * x
    residual = float(np.sum((y - predictions) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total <= 1e-20 and residual <= 1e-20 else 1.0 - residual / max(total, 1e-20)
    return {
        "exponent": float(-slope),
        "log_coefficient": float(intercept),
        "coefficient": float(math.exp(intercept)),
        "r_squared": r_squared,
    }


def _bootstrap_power_fit(
    optima: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    seed_count = len(optima[0]["rate_grid"][0]["seed_losses"])
    if seed_count < 2 or samples <= 0:
        return {
            "samples": 0,
            "exponent_interval_95": None,
            "heldout_learning_rate_interval_95": None,
        }
    rng = np.random.default_rng(seed)
    exponents = []
    intercepts = []
    for _ in range(samples):
        indices = rng.integers(0, seed_count, size=seed_count)
        sampled_optima = []
        for optimum in optima:
            sampled_grid = []
            for row in optimum["rate_grid"]:
                losses = np.asarray(row["seed_losses"], dtype=np.float64)[indices]
                sampled_grid.append(
                    {"learning_rate": row["learning_rate"], "mean_loss": float(np.mean(losses))}
                )
            sampled = _optimum(sampled_grid)
            sampled["presented_tokens"] = optimum["presented_tokens"]
            sampled_optima.append(sampled)
        fitted = _power_fit(sampled_optima)
        exponents.append(fitted["exponent"])
        intercepts.append(fitted["log_coefficient"])
    return {
        "samples": samples,
        "exponent_interval_95": [
            float(np.quantile(exponents, 0.025)),
            float(np.quantile(exponents, 0.975)),
        ],
        "bootstrap_exponents": exponents,
        "bootstrap_log_coefficients": intercepts,
    }


def _predict_rate(
    rule: str,
    *,
    source_tokens: int,
    target_tokens: int,
    source_learning_rate: float,
    fitted: Mapping[str, float],
) -> Tuple[float, float, str]:
    exponent = HORIZON_RULES[rule]
    if exponent is None:
        exponent = float(fitted["exponent"])
        learning_rate = float(
            math.exp(float(fitted["log_coefficient"])) * target_tokens ** (-exponent)
        )
        return learning_rate, exponent, "all_fit_horizons_log_linear"
    learning_rate = source_learning_rate * (target_tokens / source_tokens) ** (-exponent)
    return learning_rate, exponent, "smallest_horizon_anchor"


def compile_horizon_transfer_plan(config: Mapping[str, Any]) -> Dict[str, Any]:
    horizons = tuple(_positive_int(int(value), "presented_tokens") for value in config["presented_tokens"])
    if len(horizons) < 4 or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("presented_tokens must contain at least four unique increasing horizons")
    schedules = tuple(LearningRateSchedule.from_payload(value) for value in config["schedules"])
    if not schedules:
        raise ValueError("schedules must be non-empty")
    if len({schedule.name for schedule in schedules}) != len(schedules):
        raise ValueError("schedules must be unique")
    rates = tuple(_positive_float(value, "learning_rate") for value in config["optimizer"]["learning_rates"])
    if tuple(sorted(set(rates))) != rates or len(rates) < 3:
        raise ValueError("optimizer.learning_rates must contain at least three increasing values")
    seeds = tuple(int(value) for value in config.get("seeds", [11, 29, 47]))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    rules = tuple(str(value) for value in config.get("horizon_rules", HORIZON_RULES))
    unknown = [rule for rule in rules if rule not in HORIZON_RULES]
    if unknown:
        raise ValueError(f"unknown horizon rule(s): {', '.join(unknown)}")
    fit_trials = (len(horizons) - 1) * len(schedules) * len(rates) * len(seeds)
    transfer_trials = len(schedules) * len(rules) * len(seeds)
    oracle_trials = len(schedules) * len(rates) * len(seeds)
    expansion_rounds = int(config.get("maximum_grid_expansion_rounds", 2))
    if expansion_rounds < 0:
        raise ValueError("maximum_grid_expansion_rounds cannot be negative")
    maximum_expansion_trials = (
        len(horizons) * len(schedules) * expansion_rounds * len(seeds)
    )
    return {
        "schema_version": 1,
        "campaign": "horizon_transfer",
        "fit_horizons": list(horizons[:-1]),
        "heldout_horizon": horizons[-1],
        "schedule_names": [schedule.name for schedule in schedules],
        "horizon_rules": list(rules),
        "fit_trials": fit_trials,
        "frozen_transfer_trials": transfer_trials,
        "heldout_oracle_trials": oracle_trials,
        "maximum_grid_expansion_trials": maximum_expansion_trials,
        "planned_grid_trials": (
            fit_trials + transfer_trials + oracle_trials + maximum_expansion_trials
        ),
        "execution_order": [
            "fit_horizon_oracles",
            "freeze_schedule_and_horizon_rules",
            "evaluate_frozen_rules_on_heldout_horizon",
            "reveal_heldout_oracle_for_regret_only",
        ],
    }


def run_horizon_transfer_campaign(
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    """Calibrate schedule/horizon laws at fixed model and batch, then test once."""
    plan = compile_horizon_transfer_plan(config)
    architecture = ArchitectureTemplate.from_dict(dict(config["architecture"]))
    if architecture.block_type != "normalized_transformer":
        raise ValueError("the first horizon-transfer slice requires normalized_transformer")
    dataset = DatasetSpec.from_dict(dict(config["dataset"]))
    scale = ScaleLevel.from_dict(dict(config["scale"]), 0)
    optimizer_payload = dict(config["optimizer"])
    if optimizer_payload.get("name") != "adam":
        raise ValueError(
            "the first normalized-Transformer horizon campaign requires Adam; "
            "AdamW and SGD use separate optimizer contracts"
        )
    learning_rates = tuple(float(value) for value in optimizer_payload["learning_rates"])
    horizons = tuple(int(value) for value in config["presented_tokens"])
    schedules = tuple(LearningRateSchedule.from_payload(value) for value in config["schedules"])
    seeds = tuple(int(value) for value in config.get("seeds", [11, 29, 47]))
    rules = tuple(str(value) for value in config.get("horizon_rules", HORIZON_RULES))
    batch_examples = _positive_int(config["batch_examples"], "batch_examples")
    batch_tokens = batch_examples * architecture.context_length
    if any(horizon % batch_tokens for horizon in horizons):
        raise ValueError("every presented-token horizon must be divisible by batch tokens")
    validation_interval = _positive_int(config.get("validation_interval", 8), "validation_interval")
    cache_directory = Path(config["cache_directory"]) if config.get("cache_directory") else None
    bootstrap_samples = int(config.get("bootstrap_samples", 400))
    maximum_expansion_rounds = int(config.get("maximum_grid_expansion_rounds", 2))
    expansion_factor = _positive_float(
        config.get("grid_expansion_factor", 3.0), "grid_expansion_factor"
    )
    if expansion_factor <= 1.0:
        raise ValueError("grid_expansion_factor must exceed one")
    minimum_seeds = _positive_int(config.get("minimum_seeds", 3), "minimum_seeds")
    minimum_fit_span = _positive_float(
        config.get("minimum_fit_horizon_span", 8.0), "minimum_fit_horizon_span"
    )
    maximum_regret = _positive_float(
        config.get("maximum_relative_oracle_regret", 0.02),
        "maximum_relative_oracle_regret",
    )
    minimum_recovery = float(config.get("minimum_recovered_improvement", 0.90))
    if not 0.0 <= minimum_recovery <= 1.0:
        raise ValueError("minimum_recovered_improvement must be in [0, 1]")
    flat_control_tolerance = _positive_float(
        config.get("flat_control_relative_tolerance", 0.002),
        "flat_control_relative_tolerance",
    )

    probe = NormalizedTransformer(architecture, scale)
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    del probe
    unique_tokens = dataset.n_train * architecture.context_length
    geometry = [
        BudgetGeometry(
            parameters=parameter_count,
            unique_tokens=unique_tokens,
            presented_tokens=horizon,
            batch_tokens=batch_tokens,
            optimizer_steps=horizon // batch_tokens,
        )
        for horizon in horizons
    ]
    fit_span = horizons[-2] / horizons[0]
    total = int(plan["planned_grid_trials"])
    completed = 0
    records: List[BatchRunRecord] = []

    def run_trial(
        *,
        horizon: int,
        learning_rate: float,
        schedule: LearningRateSchedule,
        seed: int,
        role: str,
    ) -> BatchRunRecord:
        nonlocal completed
        record, _ = run_transformer_batch_trial(
            architecture=architecture,
            dataset=dataset,
            scale=scale,
            optimizer=_optimizer(optimizer_payload, learning_rate),
            total_tokens=horizon,
            batch_examples=batch_examples,
            seed=seed,
            validation_interval=validation_interval,
            learning_rate_schedule=asdict(schedule),
            gradient_clip_norm=None,
            device=device,
            cache_directory=cache_directory,
            cache_key_suffix=f"-{role}",
        )
        records.append(record)
        completed += 1
        return record

    def tune_grid(
        *,
        horizon: int,
        schedule: LearningRateSchedule,
        role: str,
        phase: str,
    ) -> Tuple[Dict[str, Any], List[BatchRunRecord]]:
        tested_rates = list(learning_rates)
        tuned_records: List[BatchRunRecord] = []
        for expansion_round in range(maximum_expansion_rounds + 1):
            already_tested = {record.optimizer.learning_rate for record in tuned_records}
            for learning_rate in tested_rates:
                if learning_rate in already_tested:
                    continue
                for seed in seeds:
                    tuned_records.append(
                        run_trial(
                            horizon=horizon,
                            learning_rate=learning_rate,
                            schedule=schedule,
                            seed=seed,
                            role=f"{role}-round{expansion_round}",
                        )
                    )
                    _progress(
                        progress,
                        phase,
                        completed,
                        total,
                        f"{schedule.name}: tuning {horizon:,} presented tokens",
                    )
            tested_rates.sort()
            optimum = _optimum(_rate_grid(tuned_records, tested_rates))
            if optimum["optimum_is_interior"] or expansion_round == maximum_expansion_rounds:
                optimum["grid_expansion_rounds"] = expansion_round
                return optimum, tuned_records
            if math.isclose(optimum["learning_rate"], tested_rates[0], rel_tol=1e-12):
                tested_rates.append(tested_rates[0] / expansion_factor)
            else:
                tested_rates.append(tested_rates[-1] * expansion_factor)
        raise AssertionError("unreachable grid-expansion state")

    schedule_analyses = []
    for schedule in schedules:
        fit_optima = []
        for horizon in horizons[:-1]:
            optimum, _ = tune_grid(
                horizon=horizon,
                schedule=schedule,
                role=f"fit-{schedule.name}-t{horizon}",
                phase="fit-horizons",
            )
            optimum["presented_tokens"] = horizon
            optimum["optimizer_steps"] = horizon // batch_tokens
            fit_optima.append(optimum)
        fitted = _power_fit(fit_optima)
        bootstrap = _bootstrap_power_fit(
            fit_optima,
            samples=bootstrap_samples,
            seed=91_771 + sum(ord(character) for character in schedule.name),
        )
        fit_reasons = []
        if len(seeds) < minimum_seeds:
            fit_reasons.append(f"requires at least {minimum_seeds} seeds")
        if fit_span < minimum_fit_span:
            fit_reasons.append(
                f"fit horizon span {fit_span:.3g}x is below {minimum_fit_span:.3g}x"
            )
        if not all(bool(row["optimum_is_interior"]) for row in fit_optima):
            fit_reasons.append("at least one fit-horizon optimum is on the LR-grid boundary")
        if fitted["exponent"] < 0.0:
            fit_reasons.append("fitted optimal learning rate increases with horizon")
        if bootstrap["samples"] == 0:
            fit_reasons.append("bootstrap uncertainty is unavailable")
        source = fit_optima[0]
        frozen_rules = []
        for rule in rules:
            predicted_rate, exponent, anchor = _predict_rate(
                rule,
                source_tokens=horizons[0],
                target_tokens=horizons[-1],
                source_learning_rate=float(source["interpolated_learning_rate"]),
                fitted=fitted,
            )
            rule_records = [
                run_trial(
                    horizon=horizons[-1],
                    learning_rate=predicted_rate,
                    schedule=schedule,
                    seed=seed,
                    role=f"frozen-{schedule.name}-{rule}",
                )
                for seed in seeds
            ]
            _progress(
                progress,
                "heldout-frozen-rules",
                completed,
                total,
                f"{schedule.name}: tested frozen {rule} rule",
            )
            frozen_rules.append(
                {
                    "rule": rule,
                    "exponent": exponent,
                    "anchor": anchor,
                    "predicted_peak_learning_rate": predicted_rate,
                    "mean_heldout_loss": float(
                        np.mean([record.final_validation_loss for record in rule_records])
                    ),
                    "seed_losses": [record.final_validation_loss for record in rule_records],
                }
            )

        heldout_oracle, _ = tune_grid(
            horizon=horizons[-1],
            schedule=schedule,
            role=f"heldout-oracle-{schedule.name}",
            phase="heldout-oracle",
        )
        no_transfer = next(row for row in frozen_rules if row["rule"] == "none")
        scoring_candidates = [
            {
                "source": "heldout_grid",
                "rule": None,
                "learning_rate": heldout_oracle["learning_rate"],
                "mean_loss": heldout_oracle["mean_loss"],
            },
            *[
                {
                    "source": "frozen_rule",
                    "rule": row["rule"],
                    "learning_rate": row["predicted_peak_learning_rate"],
                    "mean_loss": row["mean_heldout_loss"],
                }
                for row in frozen_rules
            ],
        ]
        scoring_oracle = min(scoring_candidates, key=lambda row: row["mean_loss"])
        oracle_loss = float(scoring_oracle["mean_loss"])
        no_transfer_loss = float(no_transfer["mean_heldout_loss"])
        available_improvement = no_transfer_loss - oracle_loss
        mechanism_identifiable = available_improvement > flat_control_tolerance * oracle_loss
        for row in frozen_rules:
            rule_loss = float(row["mean_heldout_loss"])
            row["oracle_mean_heldout_loss"] = oracle_loss
            row["relative_oracle_regret"] = rule_loss / oracle_loss - 1.0
            row["recovered_improvement_fraction"] = (
                (no_transfer_loss - rule_loss) / available_improvement
                if mechanism_identifiable
                else None
            )
            row["transfer_certified"] = (
                not fit_reasons
                and bool(heldout_oracle["optimum_is_interior"])
                and row["relative_oracle_regret"] <= maximum_regret
            )
            row["mechanism_discrimination_certified"] = (
                row["transfer_certified"]
                and mechanism_identifiable
                and row["rule"] != "none"
                and float(row["recovered_improvement_fraction"]) >= minimum_recovery
            )
        schedule_analyses.append(
            {
                "schedule": asdict(schedule),
                "schedule_name": schedule.name,
                "fit_optima": fit_optima,
                "fitted_power_law": fitted,
                "bootstrap": bootstrap,
                "fit_qualified": not fit_reasons,
                "fit_refusal_reasons": fit_reasons,
                "heldout_oracle": heldout_oracle,
                "scoring_oracle": scoring_oracle,
                "mechanism_identifiable": mechanism_identifiable,
                "frozen_rule_results": frozen_rules,
            }
        )

    certified = [
        {
            "schedule": analysis["schedule_name"],
            **row,
        }
        for analysis in schedule_analyses
        for row in analysis["frozen_rule_results"]
        if row["transfer_certified"]
    ]
    certified.sort(key=lambda row: row["mean_heldout_loss"])
    _progress(progress, "complete", completed, completed, "Horizon-transfer campaign complete")
    return {
        "schema_version": 1,
        "status": "completed",
        "campaign": "horizon_transfer",
        "device": device,
        "config": dict(config),
        "plan": plan,
        "coordinates": {
            "N": "trainable parameters",
            "U": "unique training tokens",
            "T": "presented training tokens",
            "B": "tokens per optimizer update",
            "S": "optimizer updates = T / B",
            "TPP": "presented tokens / parameters",
        },
        "fixed_coordinates": {
            "parameters": parameter_count,
            "unique_tokens": unique_tokens,
            "batch_tokens": batch_tokens,
            "scale": asdict(scale),
        },
        "geometry": [row.to_dict() for row in geometry],
        "fit_horizon_span_ratio": fit_span,
        "heldout_horizon": horizons[-1],
        "execution_order": plan["execution_order"],
        "schedule_analyses": schedule_analyses,
        "certified_schedule_rules": certified,
        "recommendation": certified[0] if certified else None,
        "refusal_reasons": (
            []
            if certified
            else ["no schedule/horizon rule passed the preregistered held-out gates"]
        ),
        "records": [record.to_dict() for record in records],
    }
