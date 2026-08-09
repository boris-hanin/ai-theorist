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
    assert plan["tuned_hyperparameters"] == ["global_learning_rate"]
    assert plan["transfer_rule"] == "constant_global_learning_rate"
    assert plan["trial_budget_before_edge_expansion"] == 22
    assert [row["role"] for row in plan["levels"]] == ["fit", "fit", "fit", "fit", "holdout"]


def test_sgd_plan_exposes_inverse_sqrt_width_transfer():
    plan = compile_plan(default_study_spec("sgd", quick=True))
    assert plan["transfer_rule"] == "inverse_sqrt_width_learning_rate"


def test_parameter_count_matches_torch_model():
    spec = default_study_spec("sgd", quick=True)
    for scale in spec.scales:
        model = ResidualMLP(spec.architecture, scale)
        assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count(spec, scale)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda data: data["optimizer"].update(name="adamw"), "sgd or adam"),
        (lambda data: data["architecture"].update(block_type="attention"), "exactly one residual"),
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
