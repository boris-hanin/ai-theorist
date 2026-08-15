#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 TOKEN_STREAM_MANIFEST BASE_RHO32_ROOT COMPLETEP_CONFIG COMPLETEP_SELECTION COMPLETEP_TUNE_ROOT RUN_ROOT" >&2
  exit 2
fi

manifest="$1"
base_root="$2"
completep_config_source="$3"
completep_selection_source="$4"
completep_tune_root="$5"
run_root="$6"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
required_dataset_fingerprint="${REQUIRED_DATASET_FINGERPRINT:-1b854ee220230e0421acd8312d313a72d396de2234474ec20f63ba1ce4f1d703}"
current_stage="starting"

mkdir -p "$run_root"
exec 9>"$run_root/controller.lock"
if ! flock -n 9; then
  echo "another rho=32 controller holds $run_root/controller.lock" >&2
  exit 1
fi

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    echo "failed:$current_stage" > "$run_root/stage"
  fi
}
trap on_exit EXIT

for path in "$manifest" "$base_root/bound-config.json" "$base_root/controller.pid" "$completep_config_source" "$completep_selection_source"; do
  if [[ ! -f "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$completep_tune_root" || ! -x "$cli" || ! -x "$python" ]]; then
  echo "CompleteP tuning cache, campaign CLI, and Python are required" >&2
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
echo "$$" > "$run_root/controller.pid"
echo "$repo_commit" > "$run_root/repo-commit"

campaigns="$run_root/campaigns"
expanded_root="$campaigns/jiang-expanded-finite-tau"
zero_root="$campaigns/jiang-zero"
completep_root="$campaigns/completep"
mkdir -p "$expanded_root" "$zero_root" "$completep_root" "$run_root/templates"

current_stage="binding-corrected-followup-configs"
echo "$current_stage" > "$run_root/stage"
"$cli" forecast-bind \
  configs/autoscaler/jiang_mistral_100m_rho32_adamw_tau_ema_expanded.json \
  "$manifest" --output "$expanded_root/bound-config.json" \
  > "$expanded_root/bind-summary.json"
"$python" scripts/prepare_rho32_zero_decay_config.py \
  "$base_root/bound-config.json" --output "$run_root/templates/jiang-zero.json" \
  > "$run_root/prepare-zero.stdout.json"
"$cli" forecast-bind "$run_root/templates/jiang-zero.json" "$manifest" \
  --output "$zero_root/bound-config.json" > "$zero_root/bind-summary.json"
cp "$completep_config_source" "$completep_root/bound-config.json"
cp "$completep_selection_source" "$completep_root/reference-selection.json"
for root in "$base_root" "$expanded_root" "$zero_root" "$completep_root"; do
  "$cli" forecast-plan "$root/bound-config.json" > "$root/plan.json"
done
"$python" -c '
import json, sys
expected = sys.argv[1]
for path in sys.argv[2:]:
    plan = json.load(open(path))
    if plan["dataset_identity"]["fingerprint"] != expected:
        raise SystemExit(f"dataset fingerprint mismatch in {path}")
' "$required_dataset_fingerprint" \
  "$base_root/plan.json" "$expanded_root/plan.json" \
  "$zero_root/plan.json" "$completep_root/plan.json"

current_stage="preregistering-rho32-followup"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/preregister_rho32_100m_pair.py \
  "$base_root/bound-config.json" "$expanded_root/bound-config.json" \
  "$zero_root/bound-config.json" "$completep_root/bound-config.json" \
  "$completep_root/reference-selection.json" \
  --output "$run_root/preregistration.json" \
  > "$run_root/preregistration.stdout.json"

current_stage="waiting-for-base-rho32-tuning-gate"
echo "$current_stage" > "$run_root/stage"
base_pid="$(cat "$base_root/controller.pid")"
while ps -p "$base_pid" >/dev/null 2>&1; do
  sleep 30
done

current_stage="checking-idle-eight-gpu-node"
echo "$current_stage" > "$run_root/stage"
if [[ "$(systemctl is-active nvidia-fabricmanager)" != "active" ]]; then
  echo "NVIDIA Fabric Manager is not active" >&2
  exit 1
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')"
if (( gpu_count != 8 )); then
  echo "rho=32 campaign requires exactly eight GPUs; found $gpu_count" >&2
  exit 1
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)" != "NVIDIA A100-SXM4-80GB" ]]; then
  echo "rho=32 campaign requires eight A100-SXM4-80GB GPUs" >&2
  exit 1
fi
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 10
done

current_stage="tuning-rho32-expanded-and-zero-arms"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase tune \
  --campaign jiang-expanded-finite-tau "$expanded_root/bound-config.json" "$expanded_root" \
  --campaign jiang-zero "$zero_root/bound-config.json" "$zero_root" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$run_root/tune-pool-status.json"

collect_caches() {
  local campaign_root="$1"
  local phase="$2"
  local -n destination="$3"
  while IFS= read -r -d '' directory; do
    destination+=(--cache-directory "$directory")
  done < <(find "$campaign_root/$phase" -type d -name trials -print0)
  if (( ${#destination[@]} == 0 )); then
    echo "no $phase caches found under $campaign_root" >&2
    return 1
  fi
}

current_stage="selecting-rho32-reference"
echo "$current_stage" > "$run_root/stage"
base_tune=()
expanded_tune=()
zero_tune=()
collect_caches "$base_root" tune base_tune
collect_caches "$expanded_root" tune expanded_tune
collect_caches "$zero_root" tune zero_tune
"$cli" forecast-select "$base_root/bound-config.json" \
  "${base_tune[@]}" --output "$run_root/base-reference-selection.json" >/dev/null
"$cli" forecast-select "$expanded_root/bound-config.json" \
  "${expanded_tune[@]}" --output "$expanded_root/reference-selection.json" >/dev/null
"$cli" forecast-select "$zero_root/bound-config.json" \
  "${zero_tune[@]}" --output "$zero_root/reference-selection.json" >/dev/null
"$python" scripts/select_rho32_weight_decay.py \
  "$run_root/preregistration.json" "$run_root/base-reference-selection.json" \
  "$expanded_root/reference-selection.json" \
  "$zero_root/reference-selection.json" --output "$run_root/jiang-decision.json" \
  > "$run_root/jiang-decision.stdout.json"

selected_source="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_source"])' "$run_root/jiang-decision.json")"
case "$selected_source" in
  jiang_base_finite_tau) selected_root="$base_root" ;;
  jiang_expanded_finite_tau) selected_root="$expanded_root" ;;
  jiang_zero) selected_root="$zero_root" ;;
  *) echo "unknown selected Jiang source: $selected_source" >&2; exit 1 ;;
esac
chosen_jiang="$run_root/chosen-jiang-rho32"
chosen_completep="$run_root/chosen-completep"
mkdir -p "$chosen_jiang" "$chosen_completep"
cp "$selected_root/bound-config.json" "$chosen_jiang/bound-config.json"
cp "$run_root/jiang-decision.json" "$chosen_jiang/reference-selection.json"
cp "$completep_root/bound-config.json" "$chosen_completep/bound-config.json"
cp "$completep_root/reference-selection.json" "$chosen_completep/reference-selection.json"

current_stage="running-corrected-ladders"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase ladder \
  --campaign jiang-rho32 "$chosen_jiang/bound-config.json" "$chosen_jiang" \
  --campaign completep "$chosen_completep/bound-config.json" "$chosen_completep" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$run_root/ladder-pool-status.json"

current_stage="aggregating-corrected-ladders"
echo "$current_stage" > "$run_root/stage"
selected_tune=()
completep_tune=()
jiang_ladder=()
completep_ladder=()
collect_caches "$selected_root" tune selected_tune
collect_caches "$completep_tune_root" tune completep_tune
collect_caches "$chosen_jiang" ladder jiang_ladder
collect_caches "$chosen_completep" ladder completep_ladder
"$cli" forecast-aggregate "$chosen_jiang/bound-config.json" \
  "${selected_tune[@]}" "${jiang_ladder[@]}" \
  --output "$chosen_jiang/aggregate" > "$chosen_jiang/aggregate.stdout.json"
"$cli" forecast-aggregate "$chosen_completep/bound-config.json" \
  "${completep_tune[@]}" "${completep_ladder[@]}" \
  --output "$chosen_completep/aggregate" > "$chosen_completep/aggregate.stdout.json"

current_stage="evaluating-corrected-pair"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/evaluate_rho32_100m_pair.py \
  "$run_root/preregistration.json" "$run_root/jiang-decision.json" \
  "$chosen_jiang/aggregate/result.json" \
  "$chosen_completep/aggregate/result.json" \
  --output "$run_root/pair-result.json" > "$run_root/pair-result.stdout.json"

current_stage="complete"
echo "$current_stage" > "$run_root/stage"
