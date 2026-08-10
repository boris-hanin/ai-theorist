from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse
from uuid import uuid4

from .batch_scaling import (
    OptimizerHyperparameters,
    TransferContext,
    apply_transfer_rule,
    transfer_rule_registry,
)
from .campaign_jobs import compile_campaign_plan, run_campaign_job
from .public_corpora import (
    PublicCorpusSpec,
    materialize_public_corpus,
    public_corpus_catalog,
)
from .schema import SpecError, StudySpec, compile_plan, default_study_spec
from .study import atomic_write_json, json_safe, run_study
from .tokenization import token_stream_identity, tokenizer_catalog


# Increment when job interpretation changes in a way that makes a persisted
# result unsafe to reuse for an identical request (for example, a new
# estimator qualification gate). Trial-level cache formats version separately.
CAMPAIGN_JOB_IDENTITY_VERSION = 5


def _campaign_data_identity(config: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        return None
    manifest_path = dataset.get("token_stream_manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        return None
    identity = token_stream_identity(Path(manifest_path))
    if dataset.get("tokenizer") != identity["tokenizer_id"]:
        raise ValueError("campaign tokenizer does not match the token stream manifest")
    model = config.get("model")
    architecture = config.get("architecture")
    model_contract = model if isinstance(model, Mapping) else architecture
    if isinstance(model_contract, Mapping) and model_contract.get("vocab_size") != identity["vocab_size"]:
        raise ValueError(
            f"campaign model requires vocab_size {identity['vocab_size']} for its token stream"
        )
    return identity


class StudyStore:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        if self.run_root.is_dir():
            for job_path in self.run_root.glob("*/job.json"):
                try:
                    with job_path.open("r", encoding="utf-8") as handle:
                        job = json.load(handle)
                    if job.get("kind") != "scaling_study":
                        continue
                    observed_at = datetime.fromtimestamp(
                        job_path.stat().st_mtime, timezone.utc
                    ).isoformat()
                    job.setdefault("created_at", observed_at)
                    job.setdefault("updated_at", observed_at)
                    if job.get("status") in {"queued", "running"}:
                        job["status"] = "interrupted"
                        job["error"] = (
                            "The service stopped; launch a fresh study. Completed trials "
                            "remain cached in the original run directory."
                        )
                        job["updated_at"] = self._now()
                        atomic_write_json(job_path, job)
                    self.jobs[str(job["id"])] = job
                except (OSError, ValueError, json.JSONDecodeError, KeyError):
                    continue

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _persist_locked(self, study_id: str) -> None:
        atomic_write_json(self.run_root / study_id / "job.json", self.jobs[study_id])

    @staticmethod
    def _summary(job: Mapping[str, Any]) -> Dict[str, Any]:
        spec = job.get("spec") if isinstance(job.get("spec"), dict) else {}
        architecture = spec.get("architecture") if isinstance(spec.get("architecture"), dict) else {}
        optimizer = spec.get("optimizer") if isinstance(spec.get("optimizer"), dict) else {}
        dataset = spec.get("dataset") if isinstance(spec.get("dataset"), dict) else {}
        result = job.get("result") if isinstance(job.get("result"), dict) else None
        result_summary = None
        if result is not None:
            law = result.get("scaling_law") if isinstance(result.get("scaling_law"), dict) else {}
            tuning = result.get("tuning") if isinstance(result.get("tuning"), dict) else {}
            calibration = result.get("holdout_calibration")
            first_calibration = calibration[0] if isinstance(calibration, list) and calibration else {}
            transfers = result.get("transfer_checks")
            result_summary = {
                "forecastable": bool(result.get("forecastable", False)),
                "selected_normalized_learning_rate": tuning.get("selected_normalized_learning_rate"),
                "scaling_exponent": law.get("exponent"),
                "r_squared": law.get("r_squared"),
                "holdout_relative_error": first_calibration.get("relative_error"),
                "transfer_checks_accepted": (
                    all(bool(item.get("accepted")) for item in transfers)
                    if isinstance(transfers, list) and transfers
                    else None
                ),
                "refusal_reasons": result.get("refusal_reasons", []),
                "trial_count": len(result.get("trials", [])) if isinstance(result.get("trials"), list) else None,
            }
        return {
            "id": job.get("id"),
            "kind": "scaling_study",
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "device": job.get("device"),
            "name": spec.get("name"),
            "run_profile": spec.get("run_profile"),
            "architecture": architecture.get("block_type"),
            "optimizer": optimizer.get("name"),
            "dataset": dataset.get("task_type"),
            "progress": job.get("progress"),
            "result_summary": result_summary,
            "error": job.get("error"),
        }

    def create(self, spec: StudySpec, device: str) -> Dict[str, Any]:
        study_id = uuid4().hex[:12]
        now = self._now()
        job = {
            "id": study_id,
            "kind": "scaling_study",
            "status": "queued",
            "device": device,
            "created_at": now,
            "updated_at": now,
            "spec": spec.to_dict(),
            "plan": compile_plan(spec),
            "progress": {"phase": "queued", "completed": 0, "total": 0, "message": "Waiting"},
            "result": None,
            "error": None,
        }
        with self.lock:
            self.jobs[study_id] = job
            self._persist_locked(study_id)
        thread = threading.Thread(target=self._run, args=(study_id, spec, device), daemon=True)
        thread.start()
        return self.get(study_id)  # type: ignore[return-value]

    def _run(self, study_id: str, spec: StudySpec, device: str) -> None:
        with self.lock:
            self.jobs[study_id]["status"] = "running"
            self.jobs[study_id]["updated_at"] = self._now()
            self._persist_locked(study_id)

        def update(event: Dict[str, Any]) -> None:
            with self.lock:
                self.jobs[study_id]["progress"] = event
                self.jobs[study_id]["updated_at"] = self._now()
                self._persist_locked(study_id)

        try:
            result = run_study(
                spec,
                device=device,
                output_dir=self.run_root / study_id,
                progress=update,
            )
        except Exception as exc:  # The API must retain the failure for UI inspection.
            with self.lock:
                self.jobs[study_id]["status"] = "failed"
                self.jobs[study_id]["error"] = f"{type(exc).__name__}: {exc}"
                self.jobs[study_id]["updated_at"] = self._now()
                self._persist_locked(study_id)
            return
        with self.lock:
            self.jobs[study_id]["status"] = "completed"
            self.jobs[study_id]["result"] = result
            self.jobs[study_id]["updated_at"] = self._now()
            self._persist_locked(study_id)

    def get(self, study_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(study_id)
            return json.loads(json.dumps(job)) if job is not None else None

    def list(self) -> list:
        with self.lock:
            jobs = sorted(
                self.jobs.values(),
                key=lambda item: str(item.get("created_at", "")),
                reverse=True,
            )
            return [self._summary(job) for job in jobs]


class CampaignStore:
    """Persistent, resumable web jobs for batch and pretraining campaigns."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root / "batch-jobs"
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        if self.run_root.is_dir():
            for job_path in self.run_root.glob("*/job.json"):
                try:
                    with job_path.open("r", encoding="utf-8") as handle:
                        job = json.load(handle)
                    if job.get("status") in {"queued", "running"}:
                        job["status"] = "interrupted"
                        job["error"] = "The service stopped; relaunch to resume cached trials."
                        job["updated_at"] = StudyStore._now()
                        atomic_write_json(job_path, job)
                    observed_at = datetime.fromtimestamp(
                        job_path.stat().st_mtime, timezone.utc
                    ).isoformat()
                    job.setdefault("created_at", observed_at)
                    job.setdefault("updated_at", observed_at)
                    self.jobs[str(job["id"])] = job
                except (OSError, ValueError, json.JSONDecodeError, KeyError):
                    continue

    def _persist_locked(self, job_id: str) -> None:
        atomic_write_json(self.run_root / job_id / "job.json", self.jobs[job_id])

    @staticmethod
    def _summary(job: Mapping[str, Any]) -> Dict[str, Any]:
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        analyses = result.get("scale_optimizer_analyses")
        transfers = result.get("transfer_results")
        certified_horizon_rules = result.get("certified_schedule_rules")
        certified_joint_rules = result.get("certified_joint_rules")
        forecasts = result.get("forecasts")
        hidden_backtests = result.get("hidden_scale_backtests")
        dataset = result.get("dataset") if isinstance(result.get("dataset"), dict) else {}
        return {
            "id": job.get("id"),
            "kind": "batch_campaign",
            "campaign": job.get("campaign"),
            "status": job.get("status"),
            "device": job.get("device"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "progress": job.get("progress"),
            "result_summary": {
                "record_count": len(result.get("records", []))
                if isinstance(result.get("records"), list)
                else 0,
                "qualified_analyses": sum(
                    bool(item.get("consensus", {}).get("qualified"))
                    for item in analyses
                )
                if isinstance(analyses, list)
                else 0,
                "analysis_count": len(analyses) if isinstance(analyses, list) else 0,
                "recommendable_rules": sum(bool(item.get("recommendable")) for item in transfers)
                if isinstance(transfers, list)
                else len(certified_horizon_rules)
                if isinstance(certified_horizon_rules, list)
                else len(certified_joint_rules)
                if isinstance(certified_joint_rules, list)
                else 0,
                "corpus_fingerprint": dataset.get("fingerprint"),
                "dataset_identity_fingerprint": dataset.get("identity_fingerprint"),
                "tokenizer_fingerprint": dataset.get("tokenizer_fingerprint"),
                "forecastable": bool(result.get("forecastable", False)),
                "certified_forecasts": sum(
                    bool(item.get("certified")) for item in forecasts
                )
                if isinstance(forecasts, list)
                else 0,
                "forecast_count": len(forecasts)
                if isinstance(forecasts, list)
                else 0,
                "passed_hidden_backtests": sum(
                    bool(item.get("passed")) for item in hidden_backtests
                )
                if isinstance(hidden_backtests, list)
                else 0,
                "hidden_backtest_count": len(hidden_backtests)
                if isinstance(hidden_backtests, list)
                else 0,
            }
            if result
            else None,
            "error": job.get("error"),
        }

    def create(
        self, campaign: str, config: Mapping[str, Any], device: str
    ) -> Dict[str, Any]:
        plan = compile_campaign_plan(campaign, config)
        planned_identity = plan.get("dataset_identity")
        data_identity = (
            dict(planned_identity)
            if isinstance(planned_identity, Mapping)
            else _campaign_data_identity(config)
        )
        identity = json.dumps(
            {
                "campaign_job_identity_version": CAMPAIGN_JOB_IDENTITY_VERSION,
                "campaign": campaign,
                "config": config,
                "device": device,
                "data_identity": data_identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        job_id = sha256(identity.encode("utf-8")).hexdigest()[:12]
        with self.lock:
            existing = self.jobs.get(job_id)
            if existing is not None and existing["status"] in {
                "queued",
                "running",
                "completed",
            }:
                return json.loads(json.dumps(existing))
            now = StudyStore._now()
            job = {
                "id": job_id,
                "kind": "batch_campaign",
                "campaign_job_identity_version": CAMPAIGN_JOB_IDENTITY_VERSION,
                "campaign": campaign,
                "status": "queued",
                "device": device,
                "created_at": now,
                "updated_at": now,
                "config": dict(config),
                "data_identity": data_identity,
                "plan": plan,
                "progress": {
                    "phase": "queued",
                    "completed": 0,
                    "total": int(plan.get("planned_grid_trials", 0)),
                    "message": "Waiting",
                },
                "result": None,
                "error": None,
            }
            self.jobs[job_id] = job
            self._persist_locked(job_id)
        thread = threading.Thread(
            target=self._run,
            args=(job_id, campaign, dict(config), device),
            daemon=True,
        )
        thread.start()
        return self.get(job_id)  # type: ignore[return-value]

    def _run(
        self, job_id: str, campaign: str, config: Dict[str, Any], device: str
    ) -> None:
        with self.lock:
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["error"] = None
            self.jobs[job_id]["updated_at"] = StudyStore._now()
            self._persist_locked(job_id)

        def update(event: Dict[str, Any]) -> None:
            with self.lock:
                self.jobs[job_id]["progress"] = event
                self.jobs[job_id]["updated_at"] = StudyStore._now()
                self._persist_locked(job_id)

        try:
            result = run_campaign_job(
                campaign,
                config,
                device=device,
                output_dir=self.run_root / job_id,
                progress=update,
            )
        except Exception as exc:
            with self.lock:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
                self.jobs[job_id]["updated_at"] = StudyStore._now()
                self._persist_locked(job_id)
            return
        with self.lock:
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["result"] = result
            self.jobs[job_id]["updated_at"] = StudyStore._now()
            self._persist_locked(job_id)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(job_id)
            return json.loads(json.dumps(job)) if job is not None else None

    def list(self) -> list:
        with self.lock:
            jobs = sorted(
                self.jobs.values(),
                key=lambda item: str(item.get("created_at", "")),
                reverse=True,
            )
            return [self._summary(job) for job in jobs]


class CorpusStore:
    """Persistent public-corpus materialization jobs with content verification."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root / "public-corpora"
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        if self.run_root.is_dir():
            for job_path in self.run_root.glob("*/job.json"):
                try:
                    with job_path.open("r", encoding="utf-8") as handle:
                        job = json.load(handle)
                    if job.get("status") in {"queued", "running"}:
                        job["status"] = "interrupted"
                        job["error"] = "The service stopped; prepare the corpus again to resume."
                        job["updated_at"] = StudyStore._now()
                        atomic_write_json(job_path, job)
                    self.jobs[str(job["id"])] = job
                except (OSError, ValueError, json.JSONDecodeError, KeyError):
                    continue

    def _persist_locked(self, corpus_id: str) -> None:
        atomic_write_json(self.run_root / corpus_id / "job.json", self.jobs[corpus_id])

    def create(self, spec: PublicCorpusSpec) -> Dict[str, Any]:
        corpus_id = spec.fingerprint
        with self.lock:
            existing = self.jobs.get(corpus_id)
            if existing is not None and existing["status"] in {
                "queued",
                "running",
                "completed",
            }:
                return json.loads(json.dumps(existing))
            now = StudyStore._now()
            job = {
                "id": corpus_id,
                "kind": "public_corpus",
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "spec": asdict(spec),
                "progress": {
                    "phase": "queued",
                    "completed": 0,
                    "total": spec.train_bytes + spec.validation_bytes,
                    "message": "Waiting",
                },
                "result": None,
                "error": None,
            }
            self.jobs[corpus_id] = job
            self._persist_locked(corpus_id)
        thread = threading.Thread(
            target=self._run,
            args=(corpus_id, spec),
            daemon=True,
        )
        thread.start()
        return self.get(corpus_id)  # type: ignore[return-value]

    def _run(self, corpus_id: str, spec: PublicCorpusSpec) -> None:
        with self.lock:
            self.jobs[corpus_id]["status"] = "running"
            self.jobs[corpus_id]["error"] = None
            self.jobs[corpus_id]["updated_at"] = StudyStore._now()
            self._persist_locked(corpus_id)

        def update(event: Dict[str, Any]) -> None:
            with self.lock:
                self.jobs[corpus_id]["progress"] = event
                self.jobs[corpus_id]["updated_at"] = StudyStore._now()
                self._persist_locked(corpus_id)

        try:
            result = materialize_public_corpus(spec, self.run_root, update)
        except Exception as exc:
            with self.lock:
                self.jobs[corpus_id]["status"] = "failed"
                self.jobs[corpus_id]["error"] = f"{type(exc).__name__}: {exc}"
                self.jobs[corpus_id]["updated_at"] = StudyStore._now()
                self._persist_locked(corpus_id)
            return
        with self.lock:
            self.jobs[corpus_id]["status"] = "completed"
            self.jobs[corpus_id]["result"] = result
            self.jobs[corpus_id]["updated_at"] = StudyStore._now()
            self._persist_locked(corpus_id)

    def get(self, corpus_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(corpus_id)
            return json.loads(json.dumps(job)) if job is not None else None

    def list(self) -> list:
        with self.lock:
            return [
                json.loads(json.dumps(job))
                for job in sorted(
                    self.jobs.values(),
                    key=lambda item: str(item.get("created_at", "")),
                    reverse=True,
                )
            ]


class AutoscalerServer(ThreadingHTTPServer):
    store: StudyStore
    campaign_store: CampaignStore
    corpus_store: CorpusStore
    allowed_origins: set


class RequestHandler(BaseHTTPRequestHandler):
    server: AutoscalerServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"autoscaler-api: {format % args}")

    def _origin(self) -> Optional[str]:
        origin = self.headers.get("Origin")
        if origin is None:
            return None
        parsed = urlparse(origin)
        if parsed.hostname in {"localhost", "127.0.0.1"} or origin in self.server.allowed_origins:
            return origin
        return None

    def _origin_is_forbidden(self) -> bool:
        return self.headers.get("Origin") is not None and self._origin() is None

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("Request body must be between 1 byte and 1 MB")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_OPTIONS(self) -> None:
        if self._origin_is_forbidden():
            self._send(403, {"error": "Origin is not allowed"})
            return
        self.send_response(204)
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            if self.headers.get("Access-Control-Request-Private-Network") == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Vary", "Origin")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path == "/api/health":
            self._send(200, {"status": "ok", "product": "ai-theorist-autoscaler", "schema_version": 4})
            return
        if path == "/api/default-spec":
            self._send(200, {"spec": default_study_spec(quick=True).to_dict()})
            return
        if path == "/api/batch/rules":
            self._send(200, {"rules": transfer_rule_registry()})
            return
        if path == "/api/corpora/catalog":
            self._send(200, {"sources": public_corpus_catalog()})
            return
        if path == "/api/tokenizers":
            self._send(200, {"tokenizers": tokenizer_catalog()})
            return
        if path == "/api/corpora":
            self._send(200, {"corpora": self.server.corpus_store.list()})
            return
        if path.startswith("/api/corpora/"):
            corpus_id = path.rsplit("/", 1)[-1]
            job = self.server.corpus_store.get(corpus_id)
            self._send(200 if job else 404, job or {"error": "Corpus not found"})
            return
        if path == "/api/batch/jobs":
            self._send(200, {"jobs": self.server.campaign_store.list()})
            return
        if path.startswith("/api/batch/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = self.server.campaign_store.get(job_id)
            self._send(200 if job else 404, job or {"error": "Campaign not found"})
            return
        if path == "/api/studies":
            self._send(200, {"studies": self.server.store.list()})
            return
        if path.startswith("/api/studies/"):
            study_id = path.rsplit("/", 1)[-1]
            job = self.server.store.get(study_id)
            self._send(200 if job else 404, job or {"error": "Study not found"})
            return
        self._send(404, {"error": "Route not found"})

    def do_POST(self) -> None:
        if self._origin_is_forbidden():
            self._send(403, {"error": "Origin is not allowed"})
            return
        path = urlparse(self.path).path.rstrip("/")
        try:
            payload = self._body()
            if path == "/api/batch/transfer":
                optimizer_payload = payload.get("optimizer")
                context_payload = payload.get("context")
                if not isinstance(optimizer_payload, dict) or not isinstance(
                    context_payload, dict
                ):
                    raise ValueError("optimizer and context must be objects")
                result = apply_transfer_rule(
                    str(payload.get("rule", "none")),
                    OptimizerHyperparameters(**optimizer_payload),
                    TransferContext(**context_payload),
                    horizon_exponent=float(payload.get("horizon_exponent", 0.32)),
                )
                self._send(200, result.to_dict())
                return
            if path == "/api/corpora":
                spec_payload = payload.get("spec", payload)
                if not isinstance(spec_payload, dict):
                    raise ValueError("spec must be an object")
                self._send(
                    202,
                    self.server.corpus_store.create(
                        PublicCorpusSpec.from_dict(spec_payload)
                    ),
                )
                return
            if path == "/api/batch/jobs":
                campaign = payload.get("campaign")
                config = payload.get("config")
                device = payload.get("device", "cpu")
                if not isinstance(campaign, str) or not isinstance(config, dict):
                    raise ValueError("campaign must be a string and config must be an object")
                if not isinstance(device, str) or not (
                    device == "cpu" or device.startswith("cuda")
                ):
                    raise ValueError("device must be cpu or a cuda device")
                self._send(
                    202,
                    self.server.campaign_store.create(campaign, config, device),
                )
                return
            spec_payload = payload.get("spec", payload)
            if not isinstance(spec_payload, dict):
                raise ValueError("spec must be an object")
            spec = StudySpec.from_dict(spec_payload)
            if path == "/api/compile":
                self._send(200, {"spec": spec.to_dict(), "plan": compile_plan(spec)})
                return
            if path == "/api/studies":
                device = payload.get("device", "cpu")
                if not isinstance(device, str) or not (device == "cpu" or device.startswith("cuda")):
                    raise ValueError("device must be cpu or a cuda device")
                self._send(202, self.server.store.create(spec, device))
                return
            self._send(404, {"error": "Route not found"})
        except (SpecError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})


def serve(host: str = "127.0.0.1", port: int = 8787, run_root: Path = Path("runs/autoscaler")) -> None:
    server = AutoscalerServer((host, port), RequestHandler)
    server.store = StudyStore(run_root)
    server.campaign_store = CampaignStore(run_root)
    server.corpus_store = CorpusStore(run_root)
    configured = os.environ.get("AUTOSCALER_ALLOWED_ORIGINS", "")
    server.allowed_origins = {item.strip() for item in configured.split(",") if item.strip()}
    print(f"AI Theorist Autoscaler API listening on http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the AI Theorist Autoscaler API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--run-root", type=Path, default=Path("runs/autoscaler"))
    args = parser.parse_args()
    serve(args.host, args.port, args.run_root)


if __name__ == "__main__":
    main()
