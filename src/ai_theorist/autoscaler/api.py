from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from uuid import uuid4

from .schema import SpecError, StudySpec, compile_plan, default_study_spec
from .study import run_study


class StudyStore:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def create(self, spec: StudySpec, device: str) -> Dict[str, Any]:
        study_id = uuid4().hex[:12]
        job = {
            "id": study_id,
            "status": "queued",
            "spec": spec.to_dict(),
            "plan": compile_plan(spec),
            "progress": {"phase": "queued", "completed": 0, "total": 0, "message": "Waiting"},
            "result": None,
            "error": None,
        }
        with self.lock:
            self.jobs[study_id] = job
        thread = threading.Thread(target=self._run, args=(study_id, spec, device), daemon=True)
        thread.start()
        return self.get(study_id)  # type: ignore[return-value]

    def _run(self, study_id: str, spec: StudySpec, device: str) -> None:
        with self.lock:
            self.jobs[study_id]["status"] = "running"

        def update(event: Dict[str, Any]) -> None:
            with self.lock:
                self.jobs[study_id]["progress"] = event

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
            return
        with self.lock:
            self.jobs[study_id]["status"] = "completed"
            self.jobs[study_id]["result"] = result

    def get(self, study_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(study_id)
            return json.loads(json.dumps(job)) if job is not None else None

    def list(self) -> list:
        with self.lock:
            return [
                {"id": job["id"], "status": job["status"], "progress": job["progress"]}
                for job in reversed(list(self.jobs.values()))
            ]


class AutoscalerServer(ThreadingHTTPServer):
    store: StudyStore
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
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
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
            self.send_header("Vary", "Origin")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path == "/api/health":
            self._send(200, {"status": "ok", "product": "ai-theorist-autoscaler", "schema_version": 1})
            return
        if path == "/api/default-spec":
            self._send(200, {"spec": default_study_spec(quick=True).to_dict()})
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
        except (SpecError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})


def serve(host: str = "127.0.0.1", port: int = 8787, run_root: Path = Path("runs/autoscaler")) -> None:
    server = AutoscalerServer((host, port), RequestHandler)
    server.store = StudyStore(run_root)
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
