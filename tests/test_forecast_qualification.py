from copy import deepcopy

from ai_theorist.autoscaler.forecast_qualification import (
    compare_forecast_topologies,
)


def _result(replicas: int, loss: float, wall: float):
    record = {
        "run_id": f"run-{replicas}",
        "parameter_count": 100,
        "optimizer_steps": 4,
        "total_tokens": 32,
        "batch_tokens": 8,
        "accumulation_steps": 1,
        "data_parallel_replicas": replicas,
        "learning_rate_schedule": "schedule",
        "final_validation_loss": loss,
        "wall_time_seconds": wall,
        "seed": 11,
        "optimizer": {"learning_rate": 0.001},
        "validation_checkpoints": [
            {"step": 0.0, "tokens": 0.0, "validation_loss": 10.0},
            {"step": 4.0, "tokens": 32.0, "validation_loss": loss},
        ],
        "metadata": {
            "scale": {"name": "S1"},
            "optimizer_mode": "theory",
            "sampling_contract": "replicated_global_draw_rank_partition_v1",
            "peak_parameter_group_contract": [
                {"name": "jiang_embeddings", "peak_learning_rate": 0.001}
            ],
        },
    }
    return {
        "status": "completed",
        "campaign": "real_text_scaling_ladder",
        "dataset": {"fingerprint": "data"},
        "architecture_contract": {"rho_lm_over_d": 4.0},
        "reference_tuning": {"selected_learning_rate": 0.001},
        "records": [record, deepcopy(record)],
    }


def test_topology_comparison_accepts_definition_preserving_ddp() -> None:
    result = compare_forecast_topologies(
        _result(1, 9.0, 12.0), _result(2, 9.0001, 7.0)
    )
    assert result["status"] == "passed"
    assert result["logical_trials_compared"] == 1
    assert result["comparisons"][0]["speedup"] == 12.0 / 7.0


def test_topology_comparison_accepts_eight_gpu_ddp() -> None:
    result = compare_forecast_topologies(
        _result(1, 9.0, 12.0), _result(8, 9.0001, 2.0)
    )
    assert result["status"] == "passed"
    assert result["ddp_replicas"] == 8


def test_topology_comparison_refuses_loss_or_group_drift() -> None:
    single = _result(1, 9.0, 12.0)
    ddp = _result(2, 9.01, 7.0)
    ddp["records"][0]["metadata"]["peak_parameter_group_contract"][0][
        "peak_learning_rate"
    ] = 0.002
    ddp["records"][1] = deepcopy(ddp["records"][0])
    result = compare_forecast_topologies(single, ddp)
    assert result["status"] == "failed"
    assert any("optimizer contract" in error for error in result["errors"])
    assert any("loss delta" in error for error in result["errors"])
