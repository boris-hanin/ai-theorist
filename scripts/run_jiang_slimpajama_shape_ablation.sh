#!/usr/bin/env bash
set -euo pipefail

worktree=${AI_THEORIST_SHAPE_WORKTREE:-/home/ubuntu/ai-theorist-shape-ablation}
python=/home/ubuntu/ai-theorist/.venv-forecast/bin/python
cli="$worktree/scripts/ai_theorist_autoscale_python.sh"
export AI_THEORIST_PYTHON="$python"
export PYTHONPATH="$worktree/src"
parent=/home/ubuntu/ai-theorist/runs/forecast-production/jiang-slimpajama6b-gpt2-paper-v3-3seed
active_1b=/home/ubuntu/ai-theorist/runs/forecast-production/jiang-mistral-1b-10tpp-exploratory-v1
root=/home/ubuntu/ai-theorist/runs/forecast-production/jiang-slimpajama-shape-l16-d768-v1
required_commit=${REQUIRED_REPO_COMMIT:?REQUIRED_REPO_COMMIT must pin the launch commit}
stage=starting

mkdir -p "$root"
exec 9>"$root/controller.lock"
flock -n 9 || { echo "another shape-ablation controller holds the lock" >&2; exit 1; }
echo "$$" > "$root/controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$root/stage"; fi' EXIT

[[ "$(git -C "$worktree" rev-parse HEAD)" == "$required_commit" ]]
git -C "$worktree" diff --quiet
git -C "$worktree" diff --cached --quiet
[[ -x "$python" && -x "$cli" ]]
[[ -f "$parent/bound-config.json" && -f "$parent/aggregate/result.json" ]]

cd "$worktree"
stage=preparing-and-preregistering-before-outcomes
echo "$stage" > "$root/stage"
if [[ ! -f "$root/preregistration.json" ]]; then
  "$python" scripts/prepare_jiang_slimpajama_shape_ablation.py \
    "$parent/bound-config.json" "$parent/aggregate/result.json" \
    --output-root "$root" > "$root/preregistration.stdout.json"
fi

stage=waiting-for-active-1b-and-eight-idle-gpus
echo "$stage" > "$root/stage"
while :; do
  one_b_pid="$(cat "$active_1b/controller.pid" 2>/dev/null || true)"
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ -z "$busy" ]] && { [[ -z "$one_b_pid" ]] || ! kill -0 "$one_b_pid" 2>/dev/null; }; then
    break
  fi
  sleep 30
done

[[ "$(systemctl is-active nvidia-fabricmanager)" == active ]]
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "volatile uncorrected ECC error detected" >&2
  exit 1
fi

stage=qualifying-exact-deep-narrow-runtime
echo "$stage" > "$root/stage"
if [[ ! -f "$root/runtime-qualification.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_fixed_budget_runtime.py \
    "$root/config.json" --output "$root/runtime-qualification.json" \
    > "$root/runtime-qualification.stdout.json"
fi
"$python" - "$root/runtime-qualification.json" "$root/preregistration.json" <<'PY'
import json, sys
q=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2]))
assert q["status"] == "passed"
assert {row["plan_fingerprint"] for row in q["campaigns"]} == {p["plan_fingerprint"]}
PY

stage=running-fifteen-preregistered-shape-trials
echo "$stage" > "$root/stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase tune --campaign jiang-shape "$root/config.json" "$root" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$root/tune-pool-status.json"

stage=summarizing-local-learning-rate-diagnostic
echo "$stage" > "$root/stage"
cache_args=()
while IFS= read -r -d '' directory; do
  cache_args+=(--cache-directory "$directory")
done < <(find "$root/tune/tasks" -type d -name trials -print0)
"$cli" forecast-select "$root/config.json" "${cache_args[@]}" \
  --output "$root/reference-selection.json" \
  > "$root/reference-selection.stdout.json"

stage=evaluating-preregistered-transfer-cell
echo "$stage" > "$root/stage"
"$python" scripts/evaluate_jiang_slimpajama_shape_ablation.py \
  "$root/preregistration.json" "$root/reference-selection.json" \
  --output "$root/result.json" > "$root/evaluation.stdout.json"

trap - EXIT
echo complete > "$root/stage"
