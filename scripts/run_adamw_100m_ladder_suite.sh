#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TOKEN_STREAM_MANIFEST SUITE_ROOT" >&2
  exit 2
fi

manifest="$1"
suite_root="$2"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
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
if [[ ! -x "$python" ]]; then
  echo "campaign Python executable does not exist: $python" >&2
  exit 2
fi
repo_commit="$(git rev-parse HEAD)"
required_commit="${REQUIRED_REPO_COMMIT:-}"
if [[ -n "$required_commit" && "$repo_commit" != "$required_commit" ]]; then
  echo "repository commit $repo_commit does not match required $required_commit" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked repository changes must be clean before launching the suite" >&2
  exit 1
fi
echo "$repo_commit" > "$suite_root/repo-commit"
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

current_stage="binding-completep"
echo "$current_stage" > "$suite_root/stage"
"$cli" forecast-bind \
  configs/autoscaler/completep_mistral_100m_adamw_tau_ema.json \
  "$manifest" \
  --output "$completep_root/bound-config.json" \
  > "$completep_root/bind-summary.json"

"$cli" forecast-plan "$jiang_root/bound-config.json" > "$jiang_root/plan.json"
"$cli" forecast-plan "$completep_root/bound-config.json" \
  > "$completep_root/plan.json"

current_stage="preregistering-matched-pair"
echo "$current_stage" > "$suite_root/stage"
"$python" scripts/preregister_adamw_100m_pair.py \
  "$jiang_root/bound-config.json" \
  "$completep_root/bound-config.json" \
  --output "$suite_root/preregistration.json" \
  > "$suite_root/preregistration.stdout.json"

collect_cache_arguments() {
  local campaign_root="$1"
  local phase="$2"
  local -n destination="$3"
  while IFS= read -r -d '' directory; do
    destination+=(--cache-directory "$directory")
  done < <(find "$campaign_root/$phase/tasks" -type d -name trials -print0)
  if (( ${#destination[@]} == 0 )); then
    echo "no $phase trial caches found under $campaign_root" >&2
    return 1
  fi
}

current_stage="tuning-both-campaigns"
echo "$current_stage" > "$suite_root/stage"
echo "tuning" > "$jiang_root/controller-stage"
echo "tuning" > "$completep_root/controller-stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase tune \
  --campaign jiang "$jiang_root/bound-config.json" "$jiang_root" \
  --campaign completep "$completep_root/bound-config.json" "$completep_root" \
  --cli "$cli" \
  --gpus 0,1,2,3,4,5,6,7 \
  --status "$suite_root/tune-pool-status.json"

current_stage="selecting-interior-optima"
echo "$current_stage" > "$suite_root/stage"
jiang_tune_cache_args=()
completep_tune_cache_args=()
collect_cache_arguments "$jiang_root" tune jiang_tune_cache_args
collect_cache_arguments "$completep_root" tune completep_tune_cache_args
"$cli" forecast-select "$jiang_root/bound-config.json" \
  "${jiang_tune_cache_args[@]}" \
  --require-interior \
  --output "$jiang_root/reference-selection.json" \
  > "$jiang_root/reference-selection.stdout.json"
"$cli" forecast-select "$completep_root/bound-config.json" \
  "${completep_tune_cache_args[@]}" \
  --require-interior \
  --output "$completep_root/reference-selection.json" \
  > "$completep_root/reference-selection.stdout.json"

current_stage="running-both-ladders"
echo "$current_stage" > "$suite_root/stage"
echo "ladder" > "$jiang_root/controller-stage"
echo "ladder" > "$completep_root/controller-stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase ladder \
  --campaign jiang "$jiang_root/bound-config.json" "$jiang_root" \
  --campaign completep "$completep_root/bound-config.json" "$completep_root" \
  --cli "$cli" \
  --gpus 0,1,2,3,4,5,6,7 \
  --status "$suite_root/ladder-pool-status.json"

current_stage="aggregating"
echo "$current_stage" > "$suite_root/stage"
jiang_ladder_cache_args=()
completep_ladder_cache_args=()
collect_cache_arguments "$jiang_root" ladder jiang_ladder_cache_args
collect_cache_arguments "$completep_root" ladder completep_ladder_cache_args
"$cli" forecast-aggregate "$jiang_root/bound-config.json" \
  "${jiang_tune_cache_args[@]}" \
  "${jiang_ladder_cache_args[@]}" \
  --output "$jiang_root/aggregate" \
  > "$jiang_root/aggregate.stdout.json"
"$cli" forecast-aggregate "$completep_root/bound-config.json" \
  "${completep_tune_cache_args[@]}" \
  "${completep_ladder_cache_args[@]}" \
  --output "$completep_root/aggregate" \
  > "$completep_root/aggregate.stdout.json"
echo "complete" > "$jiang_root/controller-stage"
echo "complete" > "$completep_root/controller-stage"

current_stage="evaluating-matched-pair"
echo "$current_stage" > "$suite_root/stage"
"$python" scripts/evaluate_adamw_100m_pair.py \
  "$suite_root/preregistration.json" \
  "$jiang_root/aggregate/result.json" \
  "$completep_root/aggregate/result.json" \
  --output "$suite_root/pair-result.json" \
  > "$suite_root/pair-result.stdout.json"

current_stage="complete"
echo "$current_stage" > "$suite_root/stage"
