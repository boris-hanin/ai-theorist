#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BOUND_CONFIG RUN_ROOT" >&2
  exit 2
fi

config="$1"
run_root="$2"
worker_count=8
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"

if [[ ! -f "$config" ]]; then
  echo "bound forecast config does not exist: $config" >&2
  exit 2
fi
if [[ ! -x "$cli" ]]; then
  echo "autoscaler executable does not exist: $cli" >&2
  exit 2
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')"
if (( gpu_count < worker_count )); then
  echo "eight-GPU fleet requires at least 8 visible GPUs; found $gpu_count" >&2
  exit 1
fi

mkdir -p "$run_root"
exec 9>"$run_root/controller.lock"
if ! flock -n 9; then
  echo "another fleet controller holds $run_root/controller.lock" >&2
  exit 1
fi

"$cli" forecast-plan "$config" > "$run_root/plan.json"
"$cli" forecast-tasks "$config" \
  --phase tune \
  --shard-count "$worker_count" > "$run_root/tune-assignments.json"

run_phase() {
  local phase="$1"
  local selected_learning_rate="${2:-}"
  local selected_weight_decay_tau_ema="${3:-}"
  local phase_root="$run_root/$phase"
  local pids=()
  local shard
  mkdir -p "$phase_root"
  for ((shard = 0; shard < worker_count; shard += 1)); do
    local shard_root="$phase_root/shard-$shard"
    mkdir -p "$shard_root"
    local command=(
      "$cli" forecast-shard "$config"
      --phase "$phase"
      --shard-index "$shard"
      --shard-count "$worker_count"
      --device cuda
      --output "$shard_root"
      --progress-jsonl
    )
    if [[ -n "$selected_learning_rate" ]]; then
      command+=(--selected-learning-rate "$selected_learning_rate")
    fi
    if [[ -n "$selected_weight_decay_tau_ema" ]]; then
      command+=(
        --selected-weight-decay-tau-ema "$selected_weight_decay_tau_ema"
      )
    fi
    CUDA_VISIBLE_DEVICES="$shard" "${command[@]}" \
      > "$shard_root/worker.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for ((shard = 0; shard < worker_count; shard += 1)); do
    if ! wait "${pids[$shard]}"; then
      echo "$phase shard $shard failed; see $phase_root/shard-$shard/worker.log" >&2
      failed=1
    fi
  done
  if (( failed != 0 )); then
    return 1
  fi
}

run_phase tune

cache_args=()
for ((shard = 0; shard < worker_count; shard += 1)); do
  cache_args+=(--cache-directory "$run_root/tune/shard-$shard/trials")
done
"$cli" forecast-select "$config" \
  "${cache_args[@]}" \
  --require-interior \
  --output "$run_root/reference-selection.json" > "$run_root/reference-selection.stdout.json"
selected_learning_rate="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected_learning_rate"])' "$run_root/reference-selection.json")"
selected_weight_decay_tau_ema="$(python3 -c 'import json, sys; value=json.load(open(sys.argv[1])).get("selected_weight_decay_tau_ema"); print("" if value is None else value)' "$run_root/reference-selection.json")"

ladder_task_command=(
  "$cli" forecast-tasks "$config"
  --phase ladder
  --shard-count "$worker_count"
  --selected-learning-rate "$selected_learning_rate"
)
if [[ -n "$selected_weight_decay_tau_ema" ]]; then
  ladder_task_command+=(
    --selected-weight-decay-tau-ema "$selected_weight_decay_tau_ema"
  )
fi
"${ladder_task_command[@]}" > "$run_root/ladder-assignments.json"
run_phase ladder "$selected_learning_rate" "$selected_weight_decay_tau_ema"

aggregate_cache_args=()
for phase in tune ladder; do
  for ((shard = 0; shard < worker_count; shard += 1)); do
    aggregate_cache_args+=(
      --cache-directory "$run_root/$phase/shard-$shard/trials"
    )
  done
done
"$cli" forecast-aggregate "$config" \
  "${aggregate_cache_args[@]}" \
  --output "$run_root/aggregate" > "$run_root/aggregate.stdout.json"

echo "completed eight-GPU forecast fleet: $run_root/aggregate/result.json"
