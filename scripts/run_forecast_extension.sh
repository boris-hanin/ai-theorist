#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root="$1"
config="$run_root/bound-config.json"
plan="$run_root/plan.json"
preregistration="$run_root/preregistration.json"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
gpu="${FORECAST_EXTENSION_GPU:-0}"

for path in "$config" "$plan" "$preregistration"; do
  if [[ ! -f "$path" ]]; then
    echo "required extension artifact does not exist: $path" >&2
    exit 2
  fi
done
if [[ ! -x "$cli" ]]; then
  echo "autoscaler executable does not exist: $cli" >&2
  exit 2
fi

expected_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "$preregistration")"
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "repository commit does not match the preregistration" >&2
  exit 1
fi
if ! git diff-index --quiet HEAD --; then
  echo "tracked repository files changed after preregistration" >&2
  exit 1
fi
if [[ "$(systemctl is-active nvidia-fabricmanager)" != "active" ]]; then
  echo "NVIDIA Fabric Manager is not active" >&2
  exit 1
fi

mapfile -t gpu_rows < <(
  nvidia-smi \
    --query-gpu=index,name,memory.total,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total \
    --format=csv,noheader,nounits
)
if (( ${#gpu_rows[@]} != 8 )); then
  echo "forecast extension requires exactly eight visible GPUs" >&2
  exit 1
fi
for row in "${gpu_rows[@]}"; do
  if [[ "$row" != *"NVIDIA A100-SXM4-80GB"* ]] || [[ "$row" != *", 81920, 0, 0" ]]; then
    echo "GPU health contract failed: $row" >&2
    exit 1
  fi
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "another GPU process is already active" >&2
  exit 1
fi

mkdir -p "$run_root/trial"
exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "another extension controller holds $run_root/run.lock" >&2
  exit 1
fi

selected_learning_rate="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_learning_rate"])' "$preregistration")"
target_scale="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target"]["name"])' "$preregistration")"
seed="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$preregistration")"
task_id="ladder-${target_scale}-theory-eta${selected_learning_rate}-seed${seed}"

echo extension_running > "$run_root/stage"
CUDA_VISIBLE_DEVICES="$gpu" "$cli" forecast-shard "$config" \
  --phase ladder \
  --shard-index 0 \
  --shard-count 1 \
  --selected-learning-rate "$selected_learning_rate" \
  --task-id "$task_id" \
  --device cuda \
  --output "$run_root/trial" \
  --progress-jsonl > "$run_root/worker.log" 2>&1

.venv-forecast/bin/python scripts/evaluate_forecast_extension.py "$run_root" \
  > "$run_root/evaluation.stdout.json"
echo extension_complete > "$run_root/stage"
echo "completed forecast extension: $run_root/result.json"
