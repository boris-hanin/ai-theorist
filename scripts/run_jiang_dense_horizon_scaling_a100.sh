#!/usr/bin/env bash
set -euo pipefail

repo="${AI_THEORIST_REPO:-/home/ubuntu/ai-theorist}"
python="${AI_THEORIST_PYTHON:-$repo/.venv-forecast/bin/python}"
cli="${AI_THEORIST_AUTOSCALE:-$repo/scripts/ai_theorist_autoscale_python.sh}"
required_commit="${AI_THEORIST_DENSE_HORIZON_COMMIT:?pin the dense horizon commit}"
root="${AI_THEORIST_DENSE_HORIZON_ROOT:-$repo/runs/forecast-production/jiang-dense-300m40-1b20-v1}"
corpus="${AI_THEORIST_FINEWEB_22B_ROOT:-$repo/runs/forecast-corpora/mistral-fineweb-22bt-continuation-v2}"
manifest="$corpus/token-streams/manifest.json"
source_300="$repo/runs/forecast-production/jiang-rho32-300m-horizon-pair-v1"
source_1b="$repo/runs/forecast-production/jiang-mistral-1b-10tpp-exploratory-v1"
receipt="$root/token-stream-verification-receipt.json"
stage=starting

mkdir -p "$root"
exec 9>"$root/controller.lock"
flock -n 9 || { echo "another dense horizon controller holds the lock" >&2; exit 1; }
echo "$$" > "$root/controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$root/stage"; fi' EXIT

[[ "$(git -C "$repo" rev-parse HEAD)" == "$required_commit" ]]
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
[[ -x "$python" && -x "$cli" ]]
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')" == 8 ]]
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq '^NVIDIA A100-SXM4-80GB$'; then
  echo "dense horizon campaign requires eight A100-SXM4-80GB GPUs" >&2
  exit 1
fi
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "volatile uncorrected ECC error detected" >&2
  exit 1
fi
if systemctl list-unit-files nvidia-fabricmanager.service >/dev/null 2>&1; then
  [[ "$(systemctl is-active nvidia-fabricmanager)" == active ]]
fi

export AI_THEORIST_PYTHON="$python"
export AI_THEORIST_AUTOSCALE="$cli"
export PYTHONPATH="$repo/src"

json_is_complete() {
  "$python" - "$1" <<'PY' >/dev/null 2>&1
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and json.load(open(p)).get("status") == "completed" else 1)
PY
}

stage=waiting-for-immutable-21b-token-stream
echo "$stage" > "$root/stage"
while [[ ! -f "$manifest" ]]; do
  corpus_pid="$(cat "$corpus/controller.pid" 2>/dev/null || true)"
  if [[ -z "$corpus_pid" ]] || ! kill -0 "$corpus_pid" 2>/dev/null; then
    echo "22B-token corpus controller is not live" >&2
    exit 1
  fi
  sleep 30
done

stage=fully-verifying-token-stream
echo "$stage" > "$root/stage"
if [[ ! -f "$receipt" ]]; then
  "$python" "$repo/scripts/verify_token_stream_once.py" "$manifest" \
    --output "$receipt" > "$root/token-stream-verification.stdout.json"
fi

stage=compiling-and-preregistering-dense-horizons
echo "$stage" > "$root/stage"
if [[ ! -f "$root/preregistration.json" ]]; then
  "$python" "$repo/scripts/prepare_jiang_dense_horizon_scaling.py" \
    "$source_300" "$source_1b" "$manifest" "$receipt" \
    --output-root "$root" > "$root/preregistration.stdout.json"
fi
"$python" - "$root/preregistration.json" "$required_commit" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "preregistered" and all(p["gates"].values())
assert p["repo_commit"] == sys.argv[2]
assert p["dataset_identity"]["training_tokens"] >= 21_000_000_000
PY

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 10
done

for label in dense-300m-40tpp dense-1b-20tpp; do
  stage="qualifying-${label}-memory-and-optimizer-contract"
  echo "$stage" > "$root/stage"
  if [[ ! -f "$root/$label/runtime-qualification.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$python" \
      "$repo/scripts/qualify_fixed_budget_runtime.py" \
      "$root/$label/config.json" --single-process-ddp-equivalent \
      --output "$root/$label/runtime-qualification.json" \
      > "$root/$label/runtime-qualification.stdout.json"
  fi
done

run_endpoint() {
  local label="$1"
  local key="$2"
  local trial_root="$root/$label/trial"
  local shard="$trial_root/ladder-shard-000.json"
  if json_is_complete "$shard"; then
    return
  fi
  local eta task
  eta="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaigns"][sys.argv[2]]["selected_learning_rate"])' "$root/preregistration.json" "$key")"
  task="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaigns"][sys.argv[2]]["task_id"])' "$root/preregistration.json" "$key")"
  mkdir -p "$trial_root"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
    -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m ai_theorist.autoscaler.cli forecast-shard \
    "$root/$label/config.json" --phase ladder --shard-index 0 --shard-count 1 \
    --selected-learning-rate "$eta" --task-id "$task" --device cuda \
    --output "$trial_root" --progress-jsonl \
    > "$root/$label/worker.log" 2>&1
}

stage=running-300m-to-40tpp-with-10-20-40-retained-states
echo "$stage" > "$root/stage"
run_endpoint dense-300m-40tpp dense_300m_40tpp

stage=verifying-300m-retained-state-files-before-1b
echo "$stage" > "$root/stage"
"$python" - "$root/dense-300m-40tpp/trial/ladder-shard-000.json" <<'PY'
import json, pathlib, sys
p=json.load(open(sys.argv[1])); r=p["records"][0]
rows=r["metadata"]["retained_checkpoints"]
assert len(rows) == 3
assert [row["requested_tokens_per_parameter"] for row in rows] == [10.0,20.0,40.0]
assert all(pathlib.Path(row["base_path"]).with_suffix(".pt").is_file() for row in rows)
PY

stage=running-1b-to-20tpp-with-10-20-retained-states
echo "$stage" > "$root/stage"
run_endpoint dense-1b-20tpp dense_1b_20tpp

stage=verifying-and-hashing-all-retained-horizon-states
echo "$stage" > "$root/stage"
"$python" "$repo/scripts/evaluate_jiang_dense_horizon_scaling.py" \
  "$root/preregistration.json" \
  "$root/dense-300m-40tpp/trial/ladder-shard-000.json" \
  "$root/dense-1b-20tpp/trial/ladder-shard-000.json" \
  --output "$root/result.json" > "$root/evaluation.stdout.json"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader > "$root/final-gpu-health.csv"
trap - EXIT
echo complete > "$root/stage"
echo "completed dense 300M/40-TPP and 1B/20-TPP horizons: $root/result.json"
