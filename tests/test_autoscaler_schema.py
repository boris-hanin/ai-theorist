import copy

import pytest

from ai_theorist.autoscaler.model import ResidualMLP
from ai_theorist.autoscaler.schema import (
    SpecError,
    StudySpec,
    compile_plan,
    default_study_spec,
    parameter_count,
)


def test_default_spec_compiles_to_fixed_data_and_horizon():
    spec = default_study_spec("adam", quick=True)
    plan = compile_plan(spec)

    assert len(plan["levels"]) == 5
    assert plan["fixed_data_points"] == 512
    assert plan["fixed_token_horizon"] == 40 * 64
    assert plan["tuned_hyperparameters"] == ["normalized_learning_rate_eta"]
    assert plan["learning_rate_grid_coordinate"] == "normalized_eta"
    assert plan["transfer_rule"] == "raw_lr_equals_normalized_eta"
    assert plan["trial_budget_before_edge_expansion"] == 22
    assert [row["role"] for row in plan["levels"]] == ["fit", "fit", "fit", "fit", "holdout"]


def test_sgd_plan_exposes_inverse_sqrt_width_transfer():
    plan = compile_plan(default_study_spec("sgd", quick=True))
    assert plan["transfer_rule"] == "raw_lr_equals_normalized_eta_over_sqrt_width"


def test_parameter_count_matches_torch_model():
    spec = default_study_spec("sgd", quick=True)
    for scale in spec.scales:
        model = ResidualMLP(spec.architecture, scale)
        assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count(spec, scale)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda data: data["optimizer"].update(name="adamw"), "sgd, adam, or muon"),
        (lambda data: data["architecture"].update(block_type="attention"), "block_type"),
        (lambda data: data.update(extra_graph_edges=[]), "Unknown study"),
        (lambda data: data.update(scales=list(reversed(data["scales"]))), "increasing estimated compute"),
    ],
)
def test_schema_rejects_out_of_scope_or_unsafe_studies(mutation, match):
    data = copy.deepcopy(default_study_spec("adam", quick=True).to_dict())
    mutation(data)
    with pytest.raises(SpecError, match=match):
        StudySpec.from_dict(data)


def test_microbatch_must_divide_batch():
    data = copy.deepcopy(default_study_spec("adam", quick=True).to_dict())
    data["horizon"]["microbatch_size"] = 7
    with pytest.raises(SpecError, match="evenly divide"):
        StudySpec.from_dict(data)


def test_schema_v2_rejects_ambiguous_raw_learning_rate_grid():
    data = copy.deepcopy(default_study_spec("sgd", quick=True).to_dict())
    rates = data["tuning"].pop("normalized_learning_rates")
    data["tuning"]["learning_rates"] = rates
    with pytest.raises(SpecError, match="Unknown tuning field"):
        StudySpec.from_dict(data)


def test_moe_schema_and_parameter_count_match_model():
    spec = default_study_spec("adam", quick=True, block_type="pre_norm_moe")
    plan = compile_plan(spec)
    assert spec.architecture.num_experts == 4
    assert spec.architecture.active_experts == 1
    assert plan["architecture_contract"]["sparsity_policy"] == "fixed_across_scale"
    assert plan["architecture_contract"]["recommended_joint_path"] == "L*M/D constant"
    assert plan["transfer_rule"] == "moe_table1_group_rates_from_normalized_eta"
    assert {
        scale.repeats * scale.expert_width / scale.width
        for scale in spec.scales
    } == {4.0}
    for scale in spec.scales:
        model = ResidualMLP(spec.architecture, scale)
        assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count(
            spec, scale
        )


def test_moe_rejects_uncertified_sgd_and_invalid_sparsity():
    data = copy.deepcopy(
        default_study_spec("adam", quick=True, block_type="pre_norm_moe").to_dict()
    )
    data["optimizer"]["name"] = "sgd"
    with pytest.raises(SpecError, match="SGD MoE transfer is not certified"):
        StudySpec.from_dict(data)
    data["optimizer"]["name"] = "adam"
    data["architecture"]["active_experts"] = data["architecture"]["num_experts"] + 1
    with pytest.raises(SpecError, match="cannot exceed"):
        StudySpec.from_dict(data)
