#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EXTENSION_ROOT" >&2
  exit 2
fi

root="$1"
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
current_stage="starting-user-authorized-override"

exec 9>"$root/override-controller.lock"
if ! flock -n 9; then
  echo "another override controller holds $root/override-controller.lock" >&2
  exit 1
fi
echo "$$" > "$root/override-controller.pid"

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    echo "failed:$current_stage" > "$root/stage"
  fi
}
trap on_exit EXIT

required_commit="${REQUIRED_REPO_COMMIT:-}"
if [[ -n "$required_commit" && "$(git rev-parse HEAD)" != "$required_commit" ]]; then
  echo "repository does not match the required override commit" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked repository changes must be clean" >&2
  exit 1
fi
if [[ ! -x "$python" || ! -f "$root/preregistration.json" ]]; then
  echo "forecast runtime or preregistration is missing" >&2
  exit 1
fi
if [[ ! -f "$root/topology/comparison.json" ]]; then
  echo "the failed topology evidence is missing" >&2
  exit 1
fi
if [[ "$(systemctl is-active nvidia-fabricmanager)" != "active" ]]; then
  echo "NVIDIA Fabric Manager is not active" >&2
  exit 1
fi
if (( $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l) != 8 )); then
  echo "the 300M override requires eight GPUs" >&2
  exit 1
fi
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "one or more GPUs report an uncorrected ECC error" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; then
  echo "another GPU process is active" >&2
  exit 1
fi

"$python" - "$root" <<'PY'
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
topology = json.loads((root / "topology" / "comparison.json").read_text())
if topology.get("status") != "failed" or int(topology.get("ddp_replicas", 0)) != 8:
    raise ValueError("override requires the preserved failed eight-GPU topology result")
allowed_prefix = "loss delta "
errors = list(topology.get("errors", ()))
if not errors or any(allowed_prefix not in error for error in errors):
    raise ValueError("override refuses a topology failure other than loss tolerance")
payload = {
    "schema_version": 1,
    "authorization": "explicit_user_request_to_train_after_failed_topology_gate",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "scope": ["one-x", "ten-x"],
    "topology_status": "failed",
    "maximum_allowed_loss_delta": topology["maximum_absolute_loss_delta"],
    "maximum_observed_loss_delta": topology["maximum_observed_loss_delta"],
    "claim_restriction": "exploratory only; topology unqualified; never certified",
}
payload["fingerprint"] = sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
path = root / "execution-override.json"
if path.exists():
    existing = json.loads(path.read_text())
    existing_fingerprint = existing.pop("fingerprint", None)
    if existing_fingerprint != sha256(
        json.dumps(existing, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise ValueError("existing execution override fingerprint is invalid")
    comparable = dict(existing)
    comparable.pop("recorded_at", None)
    expected = dict(payload)
    expected.pop("recorded_at", None)
    expected.pop("fingerprint", None)
    if comparable != expected:
        raise ValueError("incompatible execution override already exists")
else:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY

target_scale="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["one_x"]["target"]["name"])' "$root/preregistration.json")"
task_id="ladder-${target_scale}-theory-eta0.03-seed11"

run_horizon() {
  local label="$1"
  local stage_label="$2"
  local output="$root/$label/trial"
  local manifest="$output/ladder-shard-000.json"
  if [[ -f "$manifest" ]] && "$python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "completed" and len(p["records"]) == 1
' "$manifest"; then
    return
  fi
  current_stage="$stage_label"
  echo "$current_stage" > "$root/stage"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
    -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m ai_theorist.autoscaler.cli forecast-shard \
    "$root/$label/config.json" \
    --phase ladder --shard-index 0 --shard-count 1 \
    --selected-learning-rate 0.03 --task-id "$task_id" --device cuda \
    --output "$output" --progress-jsonl \
    > "$root/$label/worker.log" 2>&1
}

run_horizon "one-x" "running-300m-one-x-user-authorized"
run_horizon "ten-x" "running-300m-ten-x-user-authorized"

current_stage="evaluating-300m-user-authorized-pair"
echo "$current_stage" > "$root/stage"
"$python" scripts/evaluate_fixed_budget_300m_horizon_pair.py "$root" \
  > "$root/evaluation.stdout.json"

trap - EXIT
echo complete-topology-unqualified > "$root/stage"
echo "completed user-authorized topology-unqualified 300M pair: $root/result.json"
