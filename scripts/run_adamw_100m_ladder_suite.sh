#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TOKEN_STREAM_MANIFEST SUITE_ROOT" >&2
  exit 2
fi

manifest="$1"
suite_root="$2"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
jiang_root="$suite_root/jiang-chizat-adamw"
completep_root="$suite_root/completep-adamw"
current_stage="starting"

mkdir -p "$suite_root"
exec 9>"$suite_root/controller.lock"
if ! flock -n 9; then
  echo "another suite controller holds $suite_root/controller.lock" >&2
  exit 1
fi

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    echo "failed:$current_stage" > "$suite_root/stage"
  fi
}
trap on_exit EXIT

if [[ ! -f "$manifest" ]]; then
  echo "token-stream manifest does not exist: $manifest" >&2
  exit 2
fi
if [[ ! -x "$cli" ]]; then
  echo "autoscaler executable does not exist: $cli" >&2
  exit 2
fi
if [[ "$(systemctl is-active nvidia-fabricmanager)" != "active" ]]; then
  echo "NVIDIA Fabric Manager is not active" >&2
  exit 1
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')"
if (( gpu_count != 8 )); then
  echo "the AdamW ladder suite requires exactly 8 visible GPUs; found $gpu_count" >&2
  exit 1
fi

current_stage="waiting-for-idle-gpus"
echo "$current_stage" > "$suite_root/stage"
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

mkdir -p "$jiang_root" "$completep_root"
current_stage="binding-jiang"
echo "$current_stage" > "$suite_root/stage"
"$cli" forecast-bind \
  configs/autoscaler/jiang_mistral_100m_adamw_tau_ema.json \
  "$manifest" \
  --output "$jiang_root/bound-config.json" \
  > "$jiang_root/bind-summary.json"

current_stage="running-jiang"
echo "$current_stage" > "$suite_root/stage"
echo "tuning" > "$jiang_root/controller-stage"
scripts/run_forecast_8gpu_fleet.sh \
  "$jiang_root/bound-config.json" \
  "$jiang_root"
echo "complete" > "$jiang_root/controller-stage"

current_stage="freezing-jiang-baseline"
echo "$current_stage" > "$suite_root/stage"
scripts/prepare_completep_comparison_from_jiang.py \
  configs/autoscaler/completep_mistral_100m_adamw_tau_ema.json \
  "$jiang_root/aggregate/result.json" \
  --output "$completep_root/prebound-config.json"
"$cli" forecast-bind \
  "$completep_root/prebound-config.json" \
  "$manifest" \
  --output "$completep_root/bound-config.json" \
  > "$completep_root/bind-summary.json"

current_stage="running-completep"
echo "$current_stage" > "$suite_root/stage"
echo "tuning" > "$completep_root/controller-stage"
scripts/run_forecast_8gpu_fleet.sh \
  "$completep_root/bound-config.json" \
  "$completep_root"
echo "complete" > "$completep_root/controller-stage"

current_stage="complete"
echo "$current_stage" > "$suite_root/stage"
