import copy

from ai_theorist.autoscaler.schema import StudySpec, default_study_spec
from ai_theorist.autoscaler.study import assess_power_law_readiness


def pilot_spec():
    data = copy.deepcopy(default_study_spec("adam", quick=True).to_dict())
    data["run_profile"] = "pilot"
    return StudySpec.from_dict(data)


def rows(losses):
    parameter_counts = [100, 320, 1024, 3277, 10_486]
    return [
        {
            "mean_final_validation_loss": loss,
            "sem_final_validation_loss": 0.001,
            "parameter_count": parameters,
            "estimated_training_compute": parameters * 1000,
            "n_train": 512,
        }
        for loss, parameters in zip(losses, parameter_counts)
    ]


def test_pilot_readiness_accepts_wide_monotone_signal():
    readiness = assess_power_law_readiness(
        pilot_spec(),
        rows([1.0, 0.82, 0.69, 0.59, 0.52]),
    )
    assert readiness["ready"]
    assert readiness["parameter_span_ratio"] > 100
    assert readiness["dynamic_range_to_noise"] > 10
    assert readiness["suggested_next_scale"]["width"] > pilot_spec().scales[-1].width


def test_pilot_readiness_recommends_more_signal_and_budget_for_flat_loss():
    readiness = assess_power_law_readiness(
        pilot_spec(),
        rows([1.0, 1.001, 0.999, 1.001, 1.0]),
    )
    assert not readiness["ready"]
    assert any("task complexity" in item for item in readiness["recommendations"])
    assert any("token budget" in item for item in readiness["recommendations"])
