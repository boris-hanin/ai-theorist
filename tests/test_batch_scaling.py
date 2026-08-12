import math

import numpy as np
import pytest

from ai_theorist.autoscaler.batch_scaling import (
    BatchRunRecord,
    OptimizerHyperparameters,
    TransferContext,
    apply_transfer_rule,
    transfer_rule_registry,
)
from ai_theorist.autoscaler.critical_batch import (
    LocalBranchObservation,
    CriticalBatchEstimate,
    ContinuationObservation,
    StepsToTargetObservation,
    combine_critical_batch_estimates,
    estimate_direct_checkpoint_critical_batch,
    estimate_gradient_noise_critical_batch,
    estimate_local_branched_critical_batch,
    estimate_steps_to_target_critical_batch,
)
from ai_theorist.autoscaler.seesaw import SchedulePoint, compile_seesaw_schedule


def _context(*, tokens: int = 1000, target_tokens: int = 1000) -> TransferContext:
    return TransferContext(100, 200, tokens, target_tokens, 32, 128)


def test_transfer_registry_is_inspectable_and_joint_rule_is_exact() -> None:
    registry = transfer_rule_registry()
    assert {"none", "adam_sde_sqrt", "complete_dp_joint"} <= set(registry)
    source = OptimizerHyperparameters("adam", 1e-3, beta1=0.9, beta2=0.99, epsilon=1e-8)
    result = apply_transfer_rule("complete_dp_joint", source, _context(target_tokens=2000))
    assert result.valid
    assert result.target is not None
    assert result.multipliers["learning_rate"] == pytest.approx(math.sqrt(2.0))
    assert result.target.beta1 == pytest.approx(0.8)
    assert result.target.beta2 == pytest.approx(0.98)
    assert result.target.epsilon == pytest.approx(1e-8 / math.sqrt(2.0))


def test_invalid_transfer_is_refused_not_clipped() -> None:
    source = OptimizerHyperparameters("adam", 1e-3, beta1=0.8, beta2=0.9)
    context = TransferContext(100, 100, 1000, 1000, 10, 100)
    result = apply_transfer_rule("adam_sde_sqrt", source, context)
    assert not result.valid
    assert result.target is None
    assert "beta" in result.refusal_reasons[0]


def test_batch_record_round_trip_and_timescales() -> None:
    record = BatchRunRecord(
        run_id="tiny",
        model_family="normalized_transformer",
        optimizer=OptimizerHyperparameters("adam", 1e-3, beta1=0.5, beta2=0.75),
        seed=7,
        parameter_count=100,
        width=8,
        depth=1,
        total_tokens=1024,
        batch_tokens=64,
        microbatch_tokens=16,
        accumulation_steps=2,
        data_parallel_replicas=2,
        optimizer_steps=16,
        nonpadding_tokens_seen=1024,
        learning_rate_schedule="constant",
        final_validation_loss=2.0,
    )
    assert record.tokens_per_parameter == pytest.approx(10.24)
    assert record.optimizer_timescales["beta1_half_life_tokens"] == pytest.approx(64.0)
    assert BatchRunRecord.from_dict(record.to_dict()) == record


def test_steps_to_target_recovers_synthetic_transition() -> None:
    rows = []
    for batch in (16, 32, 64, 128, 256, 512):
        for seed, perturbation in enumerate((-1.0, 0.0, 1.0)):
            rows.append(
                StepsToTargetObservation(
                    batch,
                    round(100.0 + 20_000.0 / batch + perturbation),
                    seed,
                )
            )
    estimate = estimate_steps_to_target_critical_batch(rows, bootstrap_samples=50)
    expected = 0.2 * 20_000.0 / 100.0 + 1.2 * 16.0
    assert estimate.qualified
    assert estimate.critical_batch_tokens == pytest.approx(expected, rel=0.08)


def test_steps_to_target_refuses_a_flat_quantized_curve() -> None:
    rows = [
        StepsToTargetObservation(batch, 48, seed)
        for batch in (256, 512, 1024, 2048)
        for seed in (11, 29)
    ]
    estimate = estimate_steps_to_target_critical_batch(rows, bootstrap_samples=20)
    assert not estimate.qualified
    assert "insufficient dynamic range" in " ".join(estimate.refusal_reasons)
    assert estimate.diagnostics["relative_dynamic_range"] == 0.0


def test_direct_checkpoint_requires_and_finds_a_bracket() -> None:
    rows = []
    for batch, progress in ((16, 1.0), (32, 0.95), (64, 0.9), (128, 0.7), (256, 0.5)):
        rows.append(ContinuationObservation(batch, 3.0, 3.0 - progress, 100))
    estimate = estimate_direct_checkpoint_critical_batch(rows)
    assert estimate.qualified
    assert estimate.lower_batch_tokens == 64
    assert estimate.upper_batch_tokens == 128


def test_local_branched_estimator_uses_loss_tolerance_and_brackets() -> None:
    rows = []
    losses = {128: 3.00, 256: 3.004, 512: 3.009, 1024: 3.028, 2048: 3.05}
    for batch, loss in losses.items():
        for seed, offset in enumerate((-0.001, 0.0, 0.001)):
            rows.append(LocalBranchObservation(batch, loss + offset, seed))
    estimate = estimate_local_branched_critical_batch(rows, loss_tolerance=0.01)
    assert estimate.qualified
    assert estimate.lower_batch_tokens == 512
    assert estimate.upper_batch_tokens == 1024
    assert estimate.critical_batch_tokens == pytest.approx(math.sqrt(512 * 1024))


def test_gradient_noise_estimator_and_consensus() -> None:
    rng = np.random.default_rng(4)
    gradients = np.asarray([1.0, 0.0])[None, :] + rng.normal(size=(200, 2))
    noise = estimate_gradient_noise_critical_batch(
        gradients, microbatch_tokens=32, bootstrap_samples=50
    )
    assert noise.qualified
    assert 40.0 < noise.critical_batch_tokens < 90.0

    direct_rows = [
        ContinuationObservation(batch, 3.0, 3.0 - progress, 100)
        for batch, progress in ((16, 1.0), (32, 0.95), (64, 0.9), (128, 0.7))
    ]
    direct = estimate_direct_checkpoint_critical_batch(direct_rows)
    consensus = combine_critical_batch_estimates([noise, direct], maximum_ratio=2.0)
    assert consensus.qualified


def test_batch_record_rejects_inconsistent_global_batch() -> None:
    with pytest.raises(ValueError, match="batch_tokens"):
        BatchRunRecord(
            run_id="bad",
            model_family="toy",
            optimizer=OptimizerHyperparameters("sgd", 0.1),
            seed=0,
            parameter_count=10,
            width=2,
            depth=1,
            total_tokens=100,
            batch_tokens=32,
            microbatch_tokens=8,
            accumulation_steps=2,
            data_parallel_replicas=1,
            optimizer_steps=4,
            nonpadding_tokens_seen=100,
            learning_rate_schedule="constant",
            final_validation_loss=1.0,
        )


def test_seesaw_is_gated_and_compiles_square_root_stages() -> None:
    unqualified = CriticalBatchEstimate("consensus", None, None, None, False, {})
    schedule = [SchedulePoint(0, 1e-3), SchedulePoint(1000, 5e-4)]
    refused = compile_seesaw_schedule(
        schedule,
        initial_batch_tokens=32,
        critical_batch_consensus=unqualified,
        variance_dominated=True,
    )
    assert not refused["qualified"]

    qualified = CriticalBatchEstimate("consensus", 256.0, 200.0, 320.0, True, {})
    result = compile_seesaw_schedule(
        schedule,
        initial_batch_tokens=32,
        critical_batch_consensus=qualified,
        variance_dominated=True,
    )
    assert result["qualified"]
    assert result["stages"][1]["batch_tokens"] == 64
    assert result["stages"][1]["learning_rate"] == pytest.approx(1e-3 / math.sqrt(2))
    assert result["negative_control"][0]["batch_tokens"] == 128
