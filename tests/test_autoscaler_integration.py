import copy
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ai_theorist.autoscaler.api import (
    AutoscalerServer,
    CampaignStore,
    CorpusStore,
    RequestHandler,
    StudyStore,
)
from ai_theorist.autoscaler.schema import StudySpec, compile_plan, default_study_spec
from ai_theorist.autoscaler.study import run_study


def integration_spec():
    data = copy.deepcopy(default_study_spec("adam", quick=True).to_dict())
    data["dataset"] = {"n_train": 32, "n_validation": 24, "noise_std": 0.0, "seed": 17}
    data["horizon"] = {"steps": 4, "batch_size": 8, "microbatch_size": None}
    data["scales"] = [
        {"name": "S1", "width": 4, "repeats": 1},
        {"name": "S2", "width": 6, "repeats": 1},
        {"name": "S3", "width": 8, "repeats": 2},
        {"name": "S4", "width": 10, "repeats": 2},
        {"name": "S5", "width": 12, "repeats": 3},
    ]
    data["tuning"] = {
        "normalized_learning_rates": [0.0001, 0.001, 0.01],
        "max_expansion_rounds": 0,
        "expansion_factor": 3,
    }
    data["validation"] = {
        "transfer_probe_decades": 0.3,
        "run_negative_control": False,
        "bootstrap_samples": 0,
    }
    data["seeds"] = [2, 7]
    return StudySpec.from_dict(data)


def moe_integration_spec():
    data = copy.deepcopy(default_study_spec("adam", quick=True, block_type="pre_norm_moe").to_dict())
    data["dataset"] = {"n_train": 32, "n_validation": 24, "noise_std": 0.0, "seed": 17}
    data["horizon"] = {"steps": 4, "batch_size": 8, "microbatch_size": None}
    data["scales"] = [
        {"name": "S1", "width": 4, "repeats": 1, "expert_width": 8},
        {"name": "S2", "width": 6, "repeats": 1, "expert_width": 12},
        {"name": "S3", "width": 8, "repeats": 2, "expert_width": 16},
        {"name": "S4", "width": 10, "repeats": 2, "expert_width": 20},
        {"name": "S5", "width": 12, "repeats": 3, "expert_width": 24},
    ]
    data["tuning"] = {
        "normalized_learning_rates": [0.01, 0.1, 1.0],
        "max_expansion_rounds": 0,
        "expansion_factor": 3,
    }
    data["validation"] = {
        "transfer_probe_decades": 0.3,
        "run_negative_control": False,
        "bootstrap_samples": 0,
        "routing_load_tolerance": 1e-12,
    }
    data["seeds"] = [2, 7]
    return StudySpec.from_dict(data)


def test_end_to_end_study_writes_strict_manifest_and_result(tmp_path: Path):
    spec = integration_spec()
    output = tmp_path / "study"
    result = run_study(spec, output_dir=output)

    assert result["status"] == "completed"
    assert len(result["scale_results"]) == 5
    assert len(result["holdout_calibration"]) == 1
    assert len(result["trials"]) == compile_plan(spec)["trial_budget_before_edge_expansion"]
    assert result["next_scale_forecast"] is None or result["forecastable"]
    assert result["learning_rate_coordinate"]["tuned"] == "normalized_eta"
    assert all(
        row["normalized_learning_rate"] == result["learning_rate_coordinate"]["normalized_eta"]
        and row["raw_learning_rate"] > 0.0
        for row in result["scale_results"]
    )
    assert result["transfer_checks"][0]["acceptance_rule"].startswith("fixed_eta_")
    assert (
        result["transfer_checks"][0]["edge_of_stability"]["purpose"]
        == "diagnostic_only_not_a_transfer_gate"
    )
    for filename in ("manifest.json", "result.json"):
        parsed = json.loads((output / filename).read_text(encoding="utf-8"))
        assert parsed.get("schema_version", parsed.get("spec", {}).get("schema_version")) == 2


def test_moe_routing_gate_refuses_forecast_when_loads_exceed_tolerance(tmp_path: Path):
    result = run_study(moe_integration_spec(), output_dir=tmp_path / "moe-study")

    assert result["routing_quality"]["applicable"] is True
    assert result["routing_quality"]["accepted"] is False
    assert result["routing_quality"]["scales"]
    assert result["forecastable"] is False
    assert "MoE expert routing exceeded the declared load-imbalance tolerance" in result["refusal_reasons"]


def test_api_health_compile_and_async_study(tmp_path: Path):
    try:
        server = AutoscalerServer(("127.0.0.1", 0), RequestHandler)
    except PermissionError:
        pytest.skip("local socket binding is disabled in this sandbox")
    server.store = StudyStore(tmp_path)
    server.campaign_store = CampaignStore(tmp_path)
    server.corpus_store = CorpusStore(tmp_path)
    server.allowed_origins = set()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health_request = Request(f"{base}/api/health", headers={"Origin": "http://localhost:3000"})
        with urlopen(health_request, timeout=3) as response:
            health = json.load(response)
            assert health["status"] == "ok"
            assert health["schema_version"] == 2
            assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"

        with urlopen(f"{base}/api/corpora/catalog", timeout=3) as response:
            sources = json.load(response)["sources"]
        assert {row["id"] for row in sources} == {"fineweb_edu", "openwebtext"}

        private_origin = "https://autoscaler.example"
        server.allowed_origins.add(private_origin)
        private_request = Request(
            f"{base}/api/health", headers={"Origin": private_origin}
        )
        with urlopen(private_request, timeout=3) as response:
            assert response.headers["Access-Control-Allow-Origin"] == private_origin

        preflight_request = Request(
            f"{base}/api/health",
            headers={
                "Origin": private_origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Private-Network": "true",
            },
            method="OPTIONS",
        )
        with urlopen(preflight_request, timeout=3) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == private_origin
            assert response.headers["Access-Control-Allow-Private-Network"] == "true"

        spec = integration_spec().to_dict()
        compile_request = Request(
            f"{base}/api/compile",
            data=json.dumps({"spec": spec}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(compile_request, timeout=3) as response:
            compiled = json.load(response)
        assert compiled["plan"]["target_metric"] == "final_validation_loss"

        hostile = Request(
            f"{base}/api/compile",
            data=json.dumps({"spec": spec}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": "https://hostile.example"},
            method="POST",
        )
        with pytest.raises(HTTPError) as hostile_error:
            urlopen(hostile, timeout=3)
        assert hostile_error.value.code == 403

        study_request = Request(
            f"{base}/api/studies",
            data=json.dumps({"spec": spec, "device": "cpu"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(study_request, timeout=3) as response:
            job = json.load(response)
        deadline = time.time() + 10
        while job["status"] in {"queued", "running"} and time.time() < deadline:
            time.sleep(0.05)
            with urlopen(f"{base}/api/studies/{job['id']}", timeout=3) as response:
                job = json.load(response)
        assert job["status"] == "completed", job.get("error")
        assert job["result"]["status"] == "completed"
        assert (tmp_path / job["id"] / "job.json").is_file()
        with urlopen(f"{base}/api/studies", timeout=3) as response:
            study_history = json.load(response)["studies"]
        assert study_history[0]["id"] == job["id"]
        assert study_history[0]["architecture"] == "pre_norm_mlp"
        assert study_history[0]["optimizer"] == "adam"
        assert study_history[0]["result_summary"]["trial_count"] == len(
            job["result"]["trials"]
        )
        restored = StudyStore(tmp_path).get(job["id"])
        assert restored is not None
        assert restored["status"] == "completed"
        assert restored["result"]["study_fingerprint"] == job["result"]["study_fingerprint"]

        train_path = tmp_path / "web-train.txt"
        validation_path = tmp_path / "web-validation.txt"
        train_path.write_text("Web batch jobs train on actual tokens. " * 20, encoding="utf-8")
        validation_path.write_text("Validation stays held out. " * 10, encoding="utf-8")
        campaign_config = {
            "model": {
                "vocab_size": 260,
                "context_length": 4,
                "width": 8,
                "depth": 1,
                "num_heads": 2,
                "mlp_multiplier": 2,
            },
            "dataset": {
                "train_path": str(train_path),
                "validation_path": str(validation_path),
            },
            "runtime": {
                "precision": "fp32",
                "attention_backend": "math",
                "distributed": "none",
                "num_processes": 1,
            },
            "scales": [
                {"name": "S1", "width": 8, "depth": 1, "num_heads": 2}
            ],
            "batch_examples": [1, 2, 4, 8],
            "total_tokens": 32,
            "checkpoint_tokens": 8,
            "continuation_tokens": 32,
            "target_validation_loss": 6.0,
            "validation_interval": 1,
            "validation_examples": 4,
            "gradient_noise_samples": 8,
            "seeds": [3],
            "optimizers": [{"name": "adam", "learning_rates": [0.001]}],
        }
        campaign_request_body = json.dumps(
            {
                "campaign": "standard_pretraining_census",
                "config": campaign_config,
                "device": "cpu",
            }
        ).encode("utf-8")
        campaign_request = Request(
            f"{base}/api/batch/jobs",
            data=campaign_request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(campaign_request, timeout=3) as response:
            campaign_job = json.load(response)
        deadline = time.time() + 10
        while (
            campaign_job["status"] in {"queued", "running"}
            and time.time() < deadline
        ):
            time.sleep(0.05)
            with urlopen(
                f"{base}/api/batch/jobs/{campaign_job['id']}", timeout=3
            ) as response:
                campaign_job = json.load(response)
        assert campaign_job["status"] == "completed", campaign_job.get("error")
        assert campaign_job["campaign_job_identity_version"] == 2
        assert campaign_job["result"]["dataset"]["training_tokens"] > 32
        assert len(campaign_job["result"]["scale_optimizer_analyses"]) == 1
        assert campaign_job["config"]["optimizers"][0]["name"] == "adam"
        restored_campaign = CampaignStore(tmp_path).get(campaign_job["id"])
        assert restored_campaign is not None
        assert restored_campaign["status"] == "completed"
        assert restored_campaign["config"] == campaign_job["config"]
        with urlopen(f"{base}/api/batch/jobs", timeout=3) as response:
            campaign_history = json.load(response)["jobs"]
        assert campaign_history[0]["id"] == campaign_job["id"]
        assert campaign_history[0]["result_summary"]["record_count"] == len(
            campaign_job["result"]["records"]
        )

        repeated_campaign_request = Request(
            f"{base}/api/batch/jobs",
            data=campaign_request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(repeated_campaign_request, timeout=3) as response:
            repeated_campaign = json.load(response)
        assert repeated_campaign["id"] == campaign_job["id"]
        assert repeated_campaign["status"] == "completed"

        invalid = Request(
            f"{base}/api/compile",
            data=b'{"schema_version":1}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(invalid, timeout=3)
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("invalid spec should return HTTP 400")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
