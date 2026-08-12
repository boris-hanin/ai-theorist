#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SOURCE_JIANG_ROOT CORRECTED_PARENT_ROOT EXTENSION_ROOT MANIFEST" >&2
  exit 2
fi

source_root="$1"
parent_root="$2"
extension_root="$3"
manifest="$4"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
current_stage="starting"

mkdir -p "$parent_root" "$extension_root"
exec 9>"$extension_root/controller.lock"
if ! flock -n 9; then
  echo "another 300M extension controller holds $extension_root/controller.lock" >&2
  exit 1
fi
echo "$$" > "$extension_root/controller.pid"

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    echo "failed:$current_stage" > "$extension_root/stage"
  fi
}
trap on_exit EXIT

required_commit="${REQUIRED_REPO_COMMIT:-}"
if [[ -n "$required_commit" && "$(git rev-parse HEAD)" != "$required_commit" ]]; then
  echo "repository does not match the required commit" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked repository changes must be clean" >&2
  exit 1
fi
if [[ ! -f "$manifest" || ! -x "$cli" || ! -x "$python" ]]; then
  echo "manifest or forecast runtime is missing" >&2
  exit 1
fi
if [[ "$(systemctl is-active nvidia-fabricmanager)" != "active" ]]; then
  echo "NVIDIA Fabric Manager is not active" >&2
  exit 1
fi
if (( $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l) != 8 )); then
  echo "the corrected ladder requires eight GPUs" >&2
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
echo "$(git rev-parse HEAD)" > "$extension_root/repo-commit"

collect_caches() {
  local root="$1"
  local phase="$2"
  local -n output="$3"
  while IFS= read -r -d '' directory; do
    output+=(--cache-directory "$directory")
  done < <(find "$root/$phase/tasks" -type d -name trials -print0)
  if (( ${#output[@]} == 0 )); then
    echo "no $phase caches found under $root" >&2
    exit 1
  fi
}

current_stage="selecting-matched-seed-eta"
echo "$current_stage" > "$extension_root/stage"
cp "$source_root/bound-config.json" "$parent_root/bound-config.json"
cp "$source_root/plan.json" "$parent_root/plan.json"
tune_caches=()
collect_caches "$source_root" tune tune_caches
"$cli" forecast-select "$parent_root/bound-config.json" \
  "${tune_caches[@]}" --require-interior \
  --output "$parent_root/reference-selection.json" \
  > "$parent_root/reference-selection.stdout.json"
"$python" -c '
import json, sys
s=json.load(open(sys.argv[1]))
assert s["selection_mode"] == "matched_single_seed_across_all_learning_rates"
assert s["selected_learning_rate"] == 0.03
assert s["selected_seed_count"] == 1
assert s["optimum_is_interior"] is True
' "$parent_root/reference-selection.json"

current_stage="running-corrected-eta003-parent-ladder"
echo "$current_stage" > "$extension_root/stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase ladder \
  --campaign jiang "$parent_root/bound-config.json" "$parent_root" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$parent_root/ladder-pool-status.json"

current_stage="aggregating-corrected-parent"
echo "$current_stage" > "$extension_root/stage"
ladder_caches=()
collect_caches "$parent_root" ladder ladder_caches
"$cli" forecast-aggregate "$parent_root/bound-config.json" \
  "${tune_caches[@]}" "${ladder_caches[@]}" \
  --output "$parent_root/aggregate" \
  > "$parent_root/aggregate.stdout.json"
"$python" -c '
import json, sys
r=json.load(open(sys.argv[1]))
assert r["status"] == "completed"
assert r["forecastable"] is True
assert r["reference_tuning"]["selected_learning_rate"] == 0.03
assert all(row["passed"] for row in r["hidden_scale_backtests"])
' "$parent_root/aggregate/result.json"

current_stage="preregistering-300m-horizon-pair-before-reveal"
echo "$current_stage" > "$extension_root/stage"
"$python" scripts/prepare_fixed_budget_300m_horizon_pair.py \
  "$parent_root" "$manifest" "$extension_root" \
  > "$extension_root/preregistration.stdout.json"

current_stage="qualifying-300m-eight-gpu-ddp"
echo "$current_stage" > "$extension_root/stage"
mkdir -p "$extension_root/topology/single" "$extension_root/topology/ddp"
target_scale="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["one_x"]["target"]["name"])' "$extension_root/preregistration.json")"
task_id="ladder-${target_scale}-theory-eta0.03-seed11"
"$python" scripts/prepare_forecast_runtime_canary.py \
  "$extension_root/one-x/config.json" "$manifest" \
  "$extension_root/topology/single-config.json" \
  --steps 20 --seed 11 --learning-rate 0.03 --fused true \
  --checkpoint-steps 0 --checkpoint-seconds 900 --distributed none \
  --num-processes 1 --gradient-accumulation-steps 32 \
  > "$extension_root/topology/single-preparation.json"
"$python" scripts/prepare_forecast_runtime_canary.py \
  "$extension_root/one-x/config.json" "$manifest" \
  "$extension_root/topology/ddp-config.json" \
  --steps 20 --seed 11 --learning-rate 0.03 --fused true \
  --checkpoint-steps 0 --checkpoint-seconds 900 --distributed ddp \
  --num-processes 8 --gradient-accumulation-steps 32 \
  > "$extension_root/topology/ddp-preparation.json"
"$cli" forecast-plan "$extension_root/topology/single-config.json" \
  > "$extension_root/topology/single-plan.json"
"$cli" forecast-plan "$extension_root/topology/ddp-config.json" \
  > "$extension_root/topology/ddp-plan.json"
CUDA_VISIBLE_DEVICES=0 "$cli" forecast-shard \
  "$extension_root/topology/single-config.json" \
  --phase ladder --shard-index 0 --shard-count 1 \
  --selected-learning-rate 0.03 --task-id "$task_id" --device cuda \
  --output "$extension_root/topology/single" --progress-jsonl \
  > "$extension_root/topology/single.log" 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m ai_theorist.autoscaler.cli forecast-shard \
  "$extension_root/topology/ddp-config.json" \
  --phase ladder --shard-index 0 --shard-count 1 \
  --selected-learning-rate 0.03 --task-id "$task_id" --device cuda \
  --output "$extension_root/topology/ddp" --progress-jsonl \
  > "$extension_root/topology/ddp.log" 2>&1
"$python" scripts/evaluate_forecast_shard_topology.py \
  "$extension_root/topology/single/ladder-shard-000.json" \
  "$extension_root/topology/ddp/ladder-shard-000.json" \
  "$extension_root/topology/single-plan.json" \
  --maximum-loss-delta 0.001 \
  --output "$extension_root/topology/comparison.json" \
  > "$extension_root/topology/comparison.stdout.json"

current_stage="running-300m-one-x-tokens"
echo "$current_stage" > "$extension_root/stage"
mkdir -p "$extension_root/one-x/trial"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m ai_theorist.autoscaler.cli forecast-shard \
  "$extension_root/one-x/config.json" \
  --phase ladder --shard-index 0 --shard-count 1 \
  --selected-learning-rate 0.03 --task-id "$task_id" --device cuda \
  --output "$extension_root/one-x/trial" --progress-jsonl \
  > "$extension_root/one-x/worker.log" 2>&1

current_stage="running-300m-ten-x-tokens"
echo "$current_stage" > "$extension_root/stage"
mkdir -p "$extension_root/ten-x/trial"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m ai_theorist.autoscaler.cli forecast-shard \
  "$extension_root/ten-x/config.json" \
  --phase ladder --shard-index 0 --shard-count 1 \
  --selected-learning-rate 0.03 --task-id "$task_id" --device cuda \
  --output "$extension_root/ten-x/trial" --progress-jsonl \
  > "$extension_root/ten-x/worker.log" 2>&1

current_stage="evaluating-300m-horizon-pair"
echo "$current_stage" > "$extension_root/stage"
"$python" scripts/evaluate_fixed_budget_300m_horizon_pair.py \
  "$extension_root" > "$extension_root/evaluation.stdout.json"

trap - EXIT
echo complete > "$extension_root/stage"
echo "completed corrected Jiang 300M 1x/10x horizon pair: $extension_root/result.json"
