from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from ai_theorist.autoscaler.model import make_teacher_dataset
from ai_theorist.autoscaler.schema import default_study_spec
from ai_theorist.autoscaler.training import train_trial
from ai_theorist.autoscaler.transfer_campaign import (
    analyze_followup_trials,
    analyze_lr_trials,
    campaign_fingerprint,
    compile_campaign_plan,
    compile_tasks,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "autoscaler"
    / "a100_mlp_adam_hard_transfer.json"
)


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_hard_transfer_plan_is_paired_and_keeps_lm_over_d_constant():
    config = load_config()
    plan = compile_campaign_plan(config, "lr")
    assert plan["trial_count"] == 6 * 9 * 12
    assert plan["paired_by_seed"] is True
    assert compile_campaign_plan(config, "lr-extension")["trial_count"] == 6 * 3 * 12
    resolved = {
        "gate": {"followups_allowed": True},
        "recommended_eta_by_scale": {
            name: 0.002 for name in config["batch_phase"]["scales"]
        },
    }
    assert (
        compile_campaign_plan(config, "batch-extension", analysis=resolved)["trial_count"]
        == 4 * 6 * 6
    )
    tasks = compile_tasks(config, "lr")
    ratios = {
        task.scale.name: task.scale.width * task.scale.repeats / task.n_train
        for task in tasks
    }
    assert len(ratios) == 6
    assert max(ratios.values()) == pytest.approx(min(ratios.values()))


def test_followups_refuse_to_compile_before_lr_gate_resolves():
    config = load_config()
    with pytest.raises(ValueError, match="resolved LR-transfer gate"):
        compile_campaign_plan(config, "batch")
    with pytest.raises(ValueError, match="resolved LR-transfer gate"):
        compile_campaign_plan(
            config,
            "horizon",
            analysis={"gate": {"followups_allowed": False}},
        )


def test_synthetic_constant_transfer_resolves_and_unlocks_followups():
    config = load_config()
    fingerprint = campaign_fingerprint(config)
    rows = []
    seed_offsets = {
        seed: (index - 5.5) * 0.0002
        for index, seed in enumerate(config["lr_phase"]["seeds"])
    }
    for task in compile_tasks(config, "lr") + compile_tasks(config, "lr-extension"):
        rate = task.normalized_learning_rate
        curve = 0.035 * math.log(rate / 0.003) ** 2
        scale_offset = 0.36 - 0.006 * config["lr_phase"]["scales"].index(task.scale.name)
        validation_loss = scale_offset + curve + seed_offsets[task.seed]
        rows.append(
            {
                "schema_version": 1,
                "campaign_fingerprint": fingerprint,
                "task": task.to_dict(),
                "result": {
                    "final_validation_loss": validation_loss,
                    "train_loss_trace": [{"step": 600.0, "training_loss": validation_loss - 0.02}],
                },
            }
        )
    analysis = analyze_lr_trials(config, rows)
    assert analysis["gate"]["status"] == "constant_transfer_supported"
    assert analysis["gate"]["followups_allowed"] is True
    assert analysis["joint_width_depth_fit"]["separately_identifiable"] is True
    assert compile_campaign_plan(config, "batch", analysis=analysis)["trial_count"] == 360
    assert compile_campaign_plan(config, "horizon", analysis=analysis)["trial_count"] == 288


def test_train_trial_accepts_a_reused_prepared_dataset():
    spec = default_study_spec(optimizer="adam", quick=True)
    spec = replace(spec, horizon=replace(spec.horizon, steps=1))
    scale = spec.scales[0]
    prepared = make_teacher_dataset(spec.architecture, spec.dataset)
    result = train_trial(
        spec,
        scale,
        0.001,
        11,
        raw_learning_rate=0.001,
        prepared_dataset=prepared,
    )
    assert result.steps_completed == 1
    bad = (prepared[0][:-1], prepared[1][:-1], prepared[2], prepared[3])
    with pytest.raises(ValueError, match="prepared training data"):
        train_trial(
            spec,
            scale,
            0.001,
            11,
            raw_learning_rate=0.001,
            prepared_dataset=bad,
        )


def test_followup_analysis_retunes_lr_and_preserves_paired_profiles():
    config = load_config()
    base_rates = {name: 0.002 for name in config["batch_phase"]["scales"]}
    analysis = {
        "gate": {"followups_allowed": True},
        "recommended_eta_by_scale": base_rates,
    }
    rows = []
    fingerprint = campaign_fingerprint(config)
    tasks = compile_tasks(config, "batch", analysis=analysis)
    tasks += compile_tasks(config, "batch-extension", analysis=analysis)
    for task in tasks:
        batch_penalty = 0.01 * abs(math.log2(task.batch_size / 256))
        lr_penalty = 0.02 * math.log(task.normalized_learning_rate / 0.002) ** 2
        seed_offset = 0.0001 * config["batch_phase"]["seeds"].index(task.seed)
        validation_loss = 0.5 + batch_penalty + lr_penalty + seed_offset
        rows.append(
            {
                "campaign_fingerprint": fingerprint,
                "task": task.to_dict(),
                "result": {
                    "final_validation_loss": validation_loss,
                    "train_loss_trace": [
                        {"step": float(task.steps), "training_loss": validation_loss - 0.01}
                    ],
                },
            }
        )
    result = analyze_followup_trials(config, rows, "batch")
    assert result["trial_count"] == 504
    assert result["paired_seed_count"] == 6
    assert all(row["selected_value"] == 256 for row in result["selected_by_scale"])
    assert all(len(rows) == 5 for rows in result["profiles_by_scale"].values())
