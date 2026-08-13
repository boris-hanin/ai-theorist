#!/usr/bin/env bash
set -euo pipefail

worktree=/home/ubuntu/ai-theorist-1b-override
python=/home/ubuntu/ai-theorist/.venv-forecast/bin/python
cli="$worktree/scripts/ai_theorist_autoscale_python.sh"
export AI_THEORIST_PYTHON="$python"
export PYTHONPATH="$worktree/src"
calibration=/home/ubuntu/ai-theorist/runs/forecast-production/jiang-mistral-10tpp-calibration-200m-v1
base_manifest=/home/ubuntu/ai-theorist/runs/forecast-corpora/mistral-fineweb-12g-v1/64538066147e9fbe/token-streams/manifest.json
continuation_root=/home/ubuntu/ai-theorist/runs/forecast-corpora/mistral-fineweb-100bt-continuation-v1
continuation_manifest="$continuation_root/token-streams/manifest.json"
extension=/home/ubuntu/ai-theorist/runs/forecast-production/jiang-mistral-1b-10tpp-exploratory-v1
known_300m=/home/ubuntu/ai-theorist/runs/forecast-production/jiang-rho32-300m-horizon-pair-v1/result.json
completep=/home/ubuntu/ai-theorist/runs/forecast-production/completep-slimpajama-gpt2-paper-3seed-v1
required_commit="${REQUIRED_REPO_COMMIT:?REQUIRED_REPO_COMMIT must pin the launch commit}"
stage=starting

mkdir -p "$extension"
exec 9>"$extension/controller.lock"
flock -n 9 || { echo "another 1B override controller holds the lock" >&2; exit 1; }
echo "$$" > "$extension/controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$extension/stage"; fi' EXIT

[[ "$(git -C "$worktree" rev-parse HEAD)" == "$required_commit" ]]
git -C "$worktree" diff --quiet
git -C "$worktree" diff --cached --quiet
[[ -x "$python" && -x "$cli" && -f "$base_manifest" && -f "$known_300m" ]]
[[ "$(systemctl is-active nvidia-fabricmanager)" == active ]]
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "volatile uncorrected ECC error detected" >&2
  exit 1
fi
cd "$worktree"

stage=waiting-for-verified-100bt-continuation
echo "$stage" > "$extension/stage"
while [[ ! -f "$continuation_manifest" ]]; do
  corpus_pid="$(cat "$continuation_root/controller.pid" 2>/dev/null || true)"
  if [[ -z "$corpus_pid" ]] || ! kill -0 "$corpus_pid" 2>/dev/null; then
    echo "FineWeb continuation controller died before producing a manifest" >&2
    exit 1
  fi
  sleep 30
done

stage=verifying-prefix-and-validation-identity
echo "$stage" > "$extension/stage"
"$python" scripts/verify_token_stream_continuation.py \
  "$base_manifest" "$continuation_manifest" \
  --required-prefix-tokens 2000158720 --minimum-training-tokens 10085203968 \
  --output "$extension/continuation-verification.json" \
  > "$extension/continuation-verification.stdout.json"

stage=compiling-and-preregistering-explicit-override
echo "$stage" > "$extension/stage"
if [[ ! -f "$extension/preregistration.json" ]]; then
  "$python" scripts/prepare_jiang_1b_10tpp_extension.py \
    "$calibration/bound-config.json" "$calibration/aggregate/result.json" \
    "$calibration/result.json" "$continuation_manifest" \
    "$extension/continuation-verification.json" --output-root "$extension" \
    --allow-exploratory-uncertified > "$extension/preparation.stdout.json"
fi
"$python" - "$extension/preregistration.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "preregistered"
assert p["exploratory_override_requested"] is True
assert p["exploratory_override_eligible"] is True
assert p["calibration_failed_gates"] == [
    "aggregate_forecastable", "prospective_1b_ensemble_qualified"
]
assert p["frozen_prediction"]["outcome_seen"] is False
PY

stage=waiting-for-jiang-slimpajama-ladder
echo "$stage" > "$extension/stage"
while [[ -n "$(nvidia-smi -i 1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

stage=pausing-completep-at-task-boundary
echo "$stage" > "$extension/stage"
completep_controller="$(cat "$completep/controller.pid" 2>/dev/null || true)"
if [[ -n "$completep_controller" ]] && kill -0 "$completep_controller" 2>/dev/null; then
  pool_pid="$(pgrep -P "$completep_controller" -f run_forecast_task_pool.py | head -1 || true)"
  if [[ -n "$pool_pid" ]] && kill -0 "$pool_pid" 2>/dev/null; then
    kill -STOP "$pool_pid"
    echo "$pool_pid" > "$extension/paused-completep-pool.pid"
    while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
      sleep 5
    done
    kill -KILL "$pool_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$completep_controller" 2>/dev/null || break
      sleep 1
    done
    echo paused-for-1b-after-task-boundary > "$completep/stage"
    "$python" - "$completep" "$extension/completep-pause.json" <<'PY'
import json, pathlib, sys, time
root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
status={}
for name in ("tune-pool-status.json","ladder-pool-status.json"):
 p=root/name
 if p.exists():
  try: status[name]=json.load(open(p))
  except Exception: pass
json.dump({"schema_version":1,"status":"paused_at_task_boundary","time":time.time(),"pool_status":status},open(out,"w"),indent=2,sort_keys=True)
PY
  elif [[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; then
    kill -TERM "$completep_controller" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$completep_controller" 2>/dev/null || break
      sleep 1
    done
    echo paused-for-1b-at-idle-boundary > "$completep/stage"
  fi
fi

stage=waiting-for-eight-idle-gpus
echo "$stage" > "$extension/stage"
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 10
done

selected_eta="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_learning_rate"])' "$extension/preregistration.json")"
task_id="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_id"])' "$extension/preregistration.json")"

stage=qualifying-1b-memory-and-optimizer-groups
echo "$stage" > "$extension/stage"
CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_fixed_budget_runtime.py \
  "$extension/config.json" --output "$extension/runtime-qualification.json" \
  --single-process-ddp-equivalent \
  > "$extension/runtime-qualification.stdout.json"

stage=qualifying-single-vs-eight-gpu-topology
echo "$stage" > "$extension/stage"
mkdir -p "$extension/topology/single" "$extension/topology/ddp"
if [[ ! -f "$extension/topology/comparison.json" ]]; then
  "$python" scripts/prepare_forecast_runtime_canary.py \
    "$extension/config.json" "$continuation_manifest" \
    "$extension/topology/single-config.json" --steps 3 --seed 11 \
    --learning-rate "$selected_eta" --fused true --checkpoint-steps 0 \
    --checkpoint-seconds 900 --distributed none --num-processes 1 \
    --batch-examples 16 --gradient-accumulation-steps 8 \
    > "$extension/topology/single-preparation.json"
  "$python" scripts/prepare_forecast_runtime_canary.py \
    "$extension/config.json" "$continuation_manifest" \
    "$extension/topology/ddp-config.json" --steps 3 --seed 11 \
    --learning-rate "$selected_eta" --fused true --checkpoint-steps 0 \
    --checkpoint-seconds 900 --distributed ddp --num-processes 8 \
    --batch-examples 16 --gradient-accumulation-steps 1 \
    > "$extension/topology/ddp-preparation.json"
  "$cli" forecast-plan "$extension/topology/single-config.json" > "$extension/topology/single-plan.json"
  "$cli" forecast-plan "$extension/topology/ddp-config.json" > "$extension/topology/ddp-plan.json"
  CUDA_VISIBLE_DEVICES=0 "$cli" forecast-shard "$extension/topology/single-config.json" \
    --phase ladder --shard-index 0 --shard-count 1 --selected-learning-rate "$selected_eta" \
    --task-id "$task_id" --device cuda --output "$extension/topology/single" \
    --progress-jsonl > "$extension/topology/single.log" 2>&1
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" -m torch.distributed.run \
    --standalone --nproc_per_node=8 -m ai_theorist.autoscaler.cli forecast-shard \
    "$extension/topology/ddp-config.json" --phase ladder --shard-index 0 \
    --shard-count 1 --selected-learning-rate "$selected_eta" --task-id "$task_id" \
    --device cuda --output "$extension/topology/ddp" --progress-jsonl \
    > "$extension/topology/ddp.log" 2>&1
  "$python" scripts/evaluate_forecast_shard_topology.py \
    "$extension/topology/single/ladder-shard-000.json" \
    "$extension/topology/ddp/ladder-shard-000.json" \
    "$extension/topology/single-plan.json" --maximum-loss-delta 0.005 \
    --output "$extension/topology/comparison.json" \
    > "$extension/topology/comparison.stdout.json"
fi

stage=running-exploratory-1b-10tpp-endpoint
echo "$stage" > "$extension/stage"
mkdir -p "$extension/trial"
if [[ ! -f "$extension/trial/ladder-shard-000.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" -m torch.distributed.run \
    --standalone --nproc_per_node=8 -m ai_theorist.autoscaler.cli forecast-shard \
    "$extension/config.json" --phase ladder --shard-index 0 --shard-count 1 \
    --selected-learning-rate "$selected_eta" --task-id "$task_id" --device cuda \
    --output "$extension/trial" --progress-jsonl > "$extension/worker.log" 2>&1
fi

stage=evaluating-exploratory-1b-prediction
echo "$stage" > "$extension/stage"
"$python" scripts/evaluate_jiang_1b_10tpp_extension.py \
  "$extension/preregistration.json" "$extension/trial/ladder-shard-000.json" \
  "$extension/topology/comparison.json" --output "$extension/result.json" \
  > "$extension/evaluation.stdout.json"

trap - EXIT
echo complete > "$extension/stage"
