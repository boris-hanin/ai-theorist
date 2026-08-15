import pytest
import torch
from torch import nn

from ai_theorist.autoscaler.lr_contract import (
    LearningRateTheory,
    audit_optimizer_groups,
    theory_group,
)


THEORY = LearningRateTheory(
    contract_id="test",
    architecture="tiny",
    optimizer="adam",
    source_title="Test source",
    source_url="https://example.test",
    source_version="v1",
    base_coordinate="eta",
    applicability="unit test",
)


def make_groups(model):
    return [
        theory_group(
            name="weight",
            params=[model.weight],
            lr=0.1,
            lr_formula="eta",
            theory=THEORY,
            scale_factors={"width_ratio": 1.0},
        ),
        theory_group(
            name="bias",
            params=[model.bias],
            lr=0.2,
            lr_formula="2 eta",
            theory=THEORY,
            scale_factors={"width_ratio": 1.0},
        ),
    ]


def test_group_audit_records_complete_disjoint_assignment():
    model = nn.Linear(3, 2)
    report = audit_optimizer_groups(model, make_groups(model), THEORY)
    assert report["complete"] is True
    assert report["disjoint"] is True
    assert report["trainable_parameter_tensors"] == 2
    assert [row["name"] for row in report["groups"]] == ["weight", "bias"]


def test_group_audit_rejects_missing_duplicate_and_foreign_parameters():
    model = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="missing"):
        audit_optimizer_groups(model, make_groups(model)[:1], THEORY)
    duplicate = make_groups(model)
    duplicate[1]["params"] = [model.weight, model.bias]
    with pytest.raises(ValueError, match="assigned to both"):
        audit_optimizer_groups(model, duplicate, THEORY)
    foreign = nn.Parameter(torch.ones(1))
    bad = make_groups(model)
    bad[1]["params"] = [foreign]
    with pytest.raises(ValueError, match="foreign"):
        audit_optimizer_groups(model, bad, THEORY)
