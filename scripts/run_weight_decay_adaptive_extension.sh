#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TOKEN_STREAM_MANIFEST ORIGINAL_PAIR_ROOT EXTENSION_ROOT" >&2
  exit 2
fi

manifest="$1"
original_root="$2"
extension_root="$3"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
current_stage="starting"

mkdir -p "$extension_root"
exec 9>"$extension_root/controller.lock"
if ! flock -n 9; then
  echo "another adaptive controller holds $extension_root/controller.lock" >&2
  exit 1
fi

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    echo "failed:$current_stage" > "$extension_root/stage"
  fi
}
trap on_exit EXIT

if [[ ! -f "$manifest" || ! -f "$original_root/controller.pid" ]]; then
  echo "manifest and original controller PID must exist" >&2
  exit 2
fi
if [[ ! -x "$cli" || ! -x "$python" ]]; then
  echo "campaign CLI and Python must be executable" >&2
  exit 2
fi
repo_commit="$(git rev-parse HEAD)"
required_commit="${REQUIRED_REPO_COMMIT:-}"
if [[ -n "$required_commit" && "$repo_commit" != "$required_commit" ]]; then
  echo "repository commit $repo_commit does not match required $required_commit" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked repository changes must be clean" >&2
  exit 1
fi
echo "$repo_commit" > "$extension_root/repo-commit"

current_stage="waiting-for-original-tuning-gate"
echo "$current_stage" > "$extension_root/stage"
original_pid="$(cat "$original_root/controller.pid")"
while ps -p "$original_pid" >/dev/null 2>&1; do
  sleep 30
done

"$python" -c '
import json, sys
path = sys.argv[1]
payload = json.load(open(path))
if payload.get("status") != "completed" or payload.get("completed_tasks") != 240:
    raise SystemExit("original paired tuning pool is not complete")
' "$original_root/tune-pool-status.json"

current_stage="waiting-for-idle-eight-gpu-node"
echo "$current_stage" > "$extension_root/stage"
if [[ "$(systemctl is-active nvidia-fabricmanager)" != "active" ]]; then
  echo "NVIDIA Fabric Manager is not active" >&2
  exit 1
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')"
if (( gpu_count != 8 )); then
  echo "adaptive extension requires exactly 8 visible GPUs; found $gpu_count" >&2
  exit 1
fi
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

templates="$extension_root/templates"
campaigns="$extension_root/campaigns"
mkdir -p "$templates" "$campaigns"
current_stage="preparing-supplemental-configs"
echo "$current_stage" > "$extension_root/stage"
"$python" scripts/prepare_weight_decay_extension_configs.py \
  configs/autoscaler/jiang_mistral_100m_adamw_tau_ema.json \
  configs/autoscaler/completep_mistral_100m_adamw_tau_ema.json \
  --output-directory "$templates" \
  > "$extension_root/prepare-configs.stdout.json"

for name in jiang-expanded jiang-zero completep-zero; do
  mkdir -p "$campaigns/$name"
  "$cli" forecast-bind \
    "$templates/$name.json" \
    "$manifest" \
    --output "$campaigns/$name/bound-config.json" \
    > "$campaigns/$name/bind-summary.json"
  "$cli" forecast-plan "$campaigns/$name/bound-config.json" \
    > "$campaigns/$name/plan.json"
done

current_stage="preregistering-adaptive-extension"
echo "$current_stage" > "$extension_root/stage"
"$python" scripts/preregister_weight_decay_extension.py \
  "$original_root/preregistration.json" \
  "$original_root/jiang-chizat-adamw/bound-config.json" \
  "$original_root/completep-adamw/bound-config.json" \
  "$campaigns/jiang-expanded/bound-config.json" \
  "$campaigns/jiang-zero/bound-config.json" \
  "$campaigns/completep-zero/bound-config.json" \
  --output "$extension_root/preregistration.json" \
  > "$extension_root/preregistration.stdout.json"

current_stage="running-supplemental-tuning"
echo "$current_stage" > "$extension_root/stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase tune \
  --campaign jiang-expanded \
    "$campaigns/jiang-expanded/bound-config.json" "$campaigns/jiang-expanded" \
  --campaign jiang-zero \
    "$campaigns/jiang-zero/bound-config.json" "$campaigns/jiang-zero" \
  --campaign completep-zero \
    "$campaigns/completep-zero/bound-config.json" "$campaigns/completep-zero" \
  --cli "$cli" \
  --gpus 0,1,2,3,4,5,6,7 \
  --status "$extension_root/tune-pool-status.json"

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

current_stage="selecting-adaptive-weight-decay"
echo "$current_stage" > "$extension_root/stage"
original_jiang_tune=()
original_completep_tune=()
expanded_jiang_tune=()
zero_jiang_tune=()
zero_completep_tune=()
collect_cache_arguments "$original_root/jiang-chizat-adamw" tune original_jiang_tune
collect_cache_arguments "$original_root/completep-adamw" tune original_completep_tune
collect_cache_arguments "$campaigns/jiang-expanded" tune expanded_jiang_tune
collect_cache_arguments "$campaigns/jiang-zero" tune zero_jiang_tune
collect_cache_arguments "$campaigns/completep-zero" tune zero_completep_tune

"$cli" forecast-select "$original_root/jiang-chizat-adamw/bound-config.json" \
  "${original_jiang_tune[@]}" \
  --output "$extension_root/original-jiang-selection.json" >/dev/null
"$cli" forecast-select "$original_root/completep-adamw/bound-config.json" \
  "${original_completep_tune[@]}" \
  --output "$extension_root/original-completep-selection.json" >/dev/null
"$cli" forecast-select "$campaigns/jiang-expanded/bound-config.json" \
  "${expanded_jiang_tune[@]}" \
  --output "$extension_root/expanded-jiang-selection.json" >/dev/null
"$cli" forecast-select "$campaigns/jiang-zero/bound-config.json" \
  "${zero_jiang_tune[@]}" \
  --output "$extension_root/zero-jiang-selection.json" >/dev/null
"$cli" forecast-select "$campaigns/completep-zero/bound-config.json" \
  "${zero_completep_tune[@]}" \
  --output "$extension_root/zero-completep-selection.json" >/dev/null

adaptive_selection_args=()
if [[ -n "${ADAPTIVE_OVERLAP_WAIVER_FILE:-}" ]]; then
  if [[ ! -f "$ADAPTIVE_OVERLAP_WAIVER_FILE" ]]; then
    echo "adaptive overlap waiver does not exist: $ADAPTIVE_OVERLAP_WAIVER_FILE" >&2
    exit 1
  fi
  adaptive_selection_args+=(
    --waive-overlap-gate
    --waiver-record "$ADAPTIVE_OVERLAP_WAIVER_FILE"
  )
fi
"$python" scripts/select_adaptive_weight_decay.py \
  "$extension_root/preregistration.json" \
  "$extension_root/original-jiang-selection.json" \
  "$extension_root/expanded-jiang-selection.json" \
  "$extension_root/zero-jiang-selection.json" \
  "$extension_root/original-completep-selection.json" \
  "$extension_root/zero-completep-selection.json" \
  "${adaptive_selection_args[@]}" \
  --output "$extension_root/decision.json" \
  > "$extension_root/decision.stdout.json"

jiang_source="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["jiang"]["selected_source"])' "$extension_root/decision.json")"
completep_source="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["completep"]["selected_source"])' "$extension_root/decision.json")"

case "$jiang_source" in
  original_jiang)
    jiang_config="$original_root/jiang-chizat-adamw/bound-config.json"
    jiang_selection="$extension_root/original-jiang-selection.json"
    jiang_tune_root="$original_root/jiang-chizat-adamw"
    ;;
  expanded_jiang)
    jiang_config="$campaigns/jiang-expanded/bound-config.json"
    jiang_selection="$extension_root/expanded-jiang-selection.json"
    jiang_tune_root="$campaigns/jiang-expanded"
    ;;
  zero_jiang)
    jiang_config="$campaigns/jiang-zero/bound-config.json"
    jiang_selection="$extension_root/zero-jiang-selection.json"
    jiang_tune_root="$campaigns/jiang-zero"
    ;;
  *) echo "unknown Jiang source: $jiang_source" >&2; exit 1 ;;
esac
case "$completep_source" in
  original_completep)
    completep_config="$original_root/completep-adamw/bound-config.json"
    completep_selection="$extension_root/original-completep-selection.json"
    completep_tune_root="$original_root/completep-adamw"
    ;;
  zero_completep)
    completep_config="$campaigns/completep-zero/bound-config.json"
    completep_selection="$extension_root/zero-completep-selection.json"
    completep_tune_root="$campaigns/completep-zero"
    ;;
  *) echo "unknown CompleteP source: $completep_source" >&2; exit 1 ;;
esac

chosen_jiang="$extension_root/chosen-jiang"
chosen_completep="$extension_root/chosen-completep"
mkdir -p "$chosen_jiang" "$chosen_completep"
cp "$jiang_config" "$chosen_jiang/bound-config.json"
cp "$jiang_selection" "$chosen_jiang/reference-selection.json"
cp "$completep_config" "$chosen_completep/bound-config.json"
cp "$completep_selection" "$chosen_completep/reference-selection.json"

current_stage="running-adaptively-selected-ladders"
echo "$current_stage" > "$extension_root/stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase ladder \
  --campaign jiang "$chosen_jiang/bound-config.json" "$chosen_jiang" \
  --campaign completep "$chosen_completep/bound-config.json" "$chosen_completep" \
  --cli "$cli" \
  --gpus 0,1,2,3,4,5,6,7 \
  --status "$extension_root/ladder-pool-status.json"

current_stage="aggregating-adaptive-ladders"
echo "$current_stage" > "$extension_root/stage"
selected_jiang_tune=()
selected_completep_tune=()
jiang_ladder=()
completep_ladder=()
collect_cache_arguments "$jiang_tune_root" tune selected_jiang_tune
collect_cache_arguments "$completep_tune_root" tune selected_completep_tune
collect_cache_arguments "$chosen_jiang" ladder jiang_ladder
collect_cache_arguments "$chosen_completep" ladder completep_ladder
"$cli" forecast-aggregate "$chosen_jiang/bound-config.json" \
  "${selected_jiang_tune[@]}" "${jiang_ladder[@]}" \
  --output "$chosen_jiang/aggregate" \
  > "$chosen_jiang/aggregate.stdout.json"
"$cli" forecast-aggregate "$chosen_completep/bound-config.json" \
  "${selected_completep_tune[@]}" "${completep_ladder[@]}" \
  --output "$chosen_completep/aggregate" \
  > "$chosen_completep/aggregate.stdout.json"

current_stage="evaluating-adaptive-pair"
echo "$current_stage" > "$extension_root/stage"
"$python" scripts/evaluate_adaptive_weight_decay_pair.py \
  "$extension_root/preregistration.json" \
  "$extension_root/decision.json" \
  "$chosen_jiang/aggregate/result.json" \
  "$chosen_completep/aggregate/result.json" \
  --output "$extension_root/pair-result.json" \
  > "$extension_root/pair-result.stdout.json"

current_stage="complete"
echo "$current_stage" > "$extension_root/stage"
