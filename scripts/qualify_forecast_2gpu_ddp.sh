#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 FORECAST_TEMPLATE TOKEN_STREAM_MANIFEST RUN_ROOT [LARGEST_STEPS]" >&2
  exit 2
fi

template="$1"
token_manifest="$2"
run_root="$3"
steps="${4:-100}"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
python_bin="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"

if [[ ! -f "$template" || ! -f "$token_manifest" ]]; then
  echo "template and token-stream manifest must both exist" >&2
  exit 2
fi
if [[ ! -x "$cli" || ! -x "$python_bin" ]]; then
  echo "forecast environment is not installed" >&2
  exit 2
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')"
if (( gpu_count < 2 )); then
  echo "DDP qualification requires at least 2 visible GPUs; found $gpu_count" >&2
  exit 1
fi

mkdir -p "$run_root"
exec 9>"$run_root/controller.lock"
if ! flock -n 9; then
  echo "another qualification controller holds $run_root/controller.lock" >&2
  exit 1
fi

single_config="$run_root/single-config.json"
ddp_config="$run_root/ddp-config.json"
"$python_bin" scripts/prepare_forecast_runtime_canary.py \
  "$template" "$token_manifest" "$single_config" \
  --steps "$steps" \
  --seed 29 \
  --learning-rate 0.001 \
  --checkpoint-steps 0 \
  --distributed none \
  --num-processes 1 > "$run_root/single-preparation.json"
"$python_bin" scripts/prepare_forecast_runtime_canary.py \
  "$template" "$token_manifest" "$ddp_config" \
  --steps "$steps" \
  --seed 29 \
  --learning-rate 0.001 \
  --checkpoint-steps 0 \
  --distributed ddp \
  --num-processes 2 > "$run_root/ddp-preparation.json"

"$cli" forecast-plan "$single_config" > "$run_root/single-plan.json"
"$cli" forecast-plan "$ddp_config" > "$run_root/ddp-plan.json"

CUDA_VISIBLE_DEVICES=0 "$cli" forecast-ladder "$single_config" \
  --device cuda \
  --output "$run_root/single" \
  --progress-jsonl > "$run_root/single.log" 2>&1
CUDA_VISIBLE_DEVICES=0,1 "$cli" forecast-ladder "$ddp_config" \
  --device cuda \
  --output "$run_root/ddp" \
  --progress-jsonl > "$run_root/ddp.log" 2>&1

"$cli" forecast-compare-topology \
  "$run_root/single/result.json" \
  "$run_root/ddp/result.json" \
  --maximum-loss-delta 0.001 \
  --output "$run_root/topology-comparison.json" > "$run_root/topology-comparison.stdout.json"

echo "passed one-GPU versus two-GPU DDP qualification: $run_root/topology-comparison.json"
