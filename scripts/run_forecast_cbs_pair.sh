#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 JIANG_CONFIG JIANG_SELECTION COMPLETEP_CONFIG COMPLETEP_SELECTION RUN_ROOT" >&2
  exit 2
fi

jiang_source="$1"
jiang_selection="$2"
completep_source="$3"
completep_selection="$4"
run_root="$5"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
current_stage="starting"

mkdir -p "$run_root"
exec 9>"$run_root/controller.lock"
if ! flock -n 9; then
  echo "another critical-batch controller holds $run_root/controller.lock" >&2
  exit 1
fi

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    echo "failed:$current_stage" > "$run_root/stage"
  fi
}
trap on_exit EXIT

for path in "$jiang_source" "$jiang_selection" "$completep_source" "$completep_selection"; do
  if [[ ! -f "$path" ]]; then
    echo "required source evidence does not exist: $path" >&2
    exit 2
  fi
done
if [[ ! -x "$cli" || ! -x "$python" ]]; then
  echo "campaign CLI and Python environment are required" >&2
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

current_stage="checking-idle-eight-gpu-node"
echo "$current_stage" > "$run_root/stage"
if [[ "$(systemctl is-active nvidia-fabricmanager)" != "active" ]]; then
  echo "NVIDIA Fabric Manager is not active" >&2
  exit 1
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')"
if (( gpu_count != 8 )); then
  echo "critical-batch campaign requires exactly eight GPUs; found $gpu_count" >&2
  exit 1
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)" != "NVIDIA A100-SXM4-80GB" ]]; then
  echo "critical-batch campaign requires eight A100-SXM4-80GB GPUs" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; then
  echo "GPU node is not idle; refusing to overlap campaigns" >&2
  exit 1
fi

jiang_root="$run_root/jiang-rho32"
completep_root="$run_root/completep"
mkdir -p "$jiang_root" "$completep_root" "$run_root/source-evidence"
cp "$jiang_source" "$run_root/source-evidence/jiang-config.json"
cp "$jiang_selection" "$run_root/source-evidence/jiang-selection.json"
cp "$completep_source" "$run_root/source-evidence/completep-config.json"
cp "$completep_selection" "$run_root/source-evidence/completep-selection.json"

current_stage="binding-source-evidence-and-critical-batch-contracts"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/prepare_forecast_cbs_config.py \
  "$run_root/source-evidence/jiang-config.json" \
  "$run_root/source-evidence/jiang-selection.json" \
  --output "$jiang_root/config.json" > "$jiang_root/prepare.stdout.json"
"$python" scripts/prepare_forecast_cbs_config.py \
  "$run_root/source-evidence/completep-config.json" \
  "$run_root/source-evidence/completep-selection.json" \
  --output "$completep_root/config.json" > "$completep_root/prepare.stdout.json"
"$cli" forecast-cbs-plan "$jiang_root/config.json" > "$jiang_root/plan.json"
"$cli" forecast-cbs-plan "$completep_root/config.json" > "$completep_root/plan.json"

current_stage="preregistering-adaptive-cbs-pair"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/preregister_forecast_cbs_pair.py \
  "$jiang_root/config.json" "$completep_root/config.json" \
  --output "$run_root/preregistration.json" > "$run_root/preregistration.stdout.json"

current_stage="tuning-fresh-horizon-safe-reference-learning-rates"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/run_forecast_cbs_task_pool.py \
  --phase pilot \
  --campaign jiang-rho32 "$jiang_root/config.json" "$jiang_root" \
  --campaign completep "$completep_root/config.json" "$completep_root" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$run_root/pilot-pool-status.json"

current_stage="requiring-interior-horizon-safe-learning-rates"
echo "$current_stage" > "$run_root/stage"
"$cli" forecast-cbs-select "$jiang_root/config.json" --root "$jiang_root" \
  --output "$jiang_root/pilot-selection.json" >/dev/null
"$cli" forecast-cbs-select "$completep_root/config.json" --root "$completep_root" \
  --output "$completep_root/pilot-selection.json" >/dev/null

current_stage="training-fresh-small-batch-baselines"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/run_forecast_cbs_task_pool.py \
  --phase baseline \
  --campaign jiang-rho32 "$jiang_root/config.json" "$jiang_root" \
  --campaign completep "$completep_root/config.json" "$completep_root" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$run_root/baseline-pool-status.json"

current_stage="running-matched-token-local-branches"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/run_forecast_cbs_task_pool.py \
  --phase branch \
  --campaign jiang-rho32 "$jiang_root/config.json" "$jiang_root" \
  --campaign completep "$completep_root/config.json" "$completep_root" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$run_root/branch-pool-status.json"

current_stage="estimating-and-gating-critical-batch-growth"
echo "$current_stage" > "$run_root/stage"
"$cli" forecast-cbs-aggregate "$jiang_root/config.json" --root "$jiang_root" \
  > "$jiang_root/aggregate.stdout.json"
"$cli" forecast-cbs-aggregate "$completep_root/config.json" --root "$completep_root" \
  > "$completep_root/aggregate.stdout.json"

current_stage="evaluating-paired-batch-warmup-contracts"
echo "$current_stage" > "$run_root/stage"
"$python" scripts/evaluate_forecast_cbs_pair.py \
  "$run_root/preregistration.json" "$jiang_root/result.json" \
  "$completep_root/result.json" --output "$run_root/pair-result.json" \
  > "$run_root/pair-result.stdout.json"

current_stage="complete"
echo "$current_stage" > "$run_root/stage"
