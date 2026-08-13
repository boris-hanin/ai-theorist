#!/usr/bin/env bash
set -euo pipefail

worktree=/home/ubuntu/ai-theorist-completep
python=/home/ubuntu/ai-theorist/.venv-forecast/bin/python
cli="$worktree/scripts/ai_theorist_autoscale_python.sh"
export AI_THEORIST_PYTHON="$python"
export PYTHONPATH="$worktree/src"
shape=/home/ubuntu/ai-theorist/runs/forecast-production/jiang-slimpajama-shape-l16-d768-v1
root=/home/ubuntu/ai-theorist/runs/forecast-production/completep-slimpajama-gpt2-paper-3seed-v1
required_commit=555a76dbfe905aa3720a79616cb972e46e5181c5
required_plan=5035171028c91ab1226091459f9055c6b5615ce880ce02432548c1178ebec807
required_dataset=f88477278cb14c6841d78eb97c1b10fd399a79be54752264dc885affcac76017
stage=starting

mkdir -p "$root"
exec 9>"$root/h100-controller.lock"
flock -n 9 || { echo "another H100 CompleteP controller holds the lock" >&2; exit 1; }
echo "$$" > "$root/h100-controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$root/h100-stage"; fi' EXIT

[[ "$(git -C "$worktree" rev-parse HEAD)" == "$required_commit" ]]
git -C "$worktree" diff --quiet
git -C "$worktree" diff --cached --quiet
[[ -x "$python" && -x "$cli" ]]

stage=waiting-for-complete-shape-ablation
echo "$stage" > "$root/h100-stage"
while [[ ! -f "$shape/result.json" ]]; do
  shape_pid="$(cat "$shape/controller.pid" 2>/dev/null || true)"
  if [[ -n "$shape_pid" ]] && ! kill -0 "$shape_pid" 2>/dev/null; then
    echo "shape-ablation controller died without a result" >&2
    exit 1
  fi
  sleep 30
done
"$python" - "$shape/result.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "completed"
PY

stage=verifying-pinned-completep-evidence
echo "$stage" > "$root/h100-stage"
"$python" - "$root/plan.json" "$root/preregistration.json" <<PY
import json, sys
plan=json.load(open(sys.argv[1])); prereg=json.load(open(sys.argv[2]))
assert plan["fingerprint"] == "$required_plan"
assert prereg["status"] == "preregistered"
assert prereg["plan_fingerprint"] == "$required_plan"
assert prereg["dataset_fingerprint"] == "$required_dataset"
PY
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 10
done
[[ "$(systemctl is-active nvidia-fabricmanager)" == active ]]
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "volatile uncorrected ECC error detected" >&2
  exit 1
fi

cd "$worktree"
stage=qualifying-completep-on-h100
echo "$stage" > "$root/h100-stage"
if [[ ! -f "$root/h100-runtime-qualification.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_completep_paper_anchor.py \
    "$root/bound-config.json" --output "$root/h100-runtime-qualification.json" \
    > "$root/h100-runtime-qualification.stdout.json"
fi
"$python" - "$root/h100-runtime-qualification.json" <<PY
import json, sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "passed"
assert {row["plan_fingerprint"] for row in p["campaigns"]} == {"$required_plan"}
PY

stage=resuming-fifteen-trial-tuning-pool
echo "$stage" > "$root/h100-stage"
"$python" scripts/run_forecast_task_pool.py --phase tune \
  --campaign completep "$root/bound-config.json" "$root" --cli "$cli" \
  --gpus 0,1,2,3,4,5,6,7 --status "$root/tune-pool-status.json"

stage=selecting-interior-learning-rate
echo "$stage" > "$root/h100-stage"
cache_args=()
while IFS= read -r -d '' directory; do
  cache_args+=(--cache-directory "$directory")
done < <(find "$root/tune/tasks" -type d -name trials -print0)
"$cli" forecast-select "$root/bound-config.json" "${cache_args[@]}" \
  --require-interior --output "$root/reference-selection.json" \
  > "$root/reference-selection.stdout.json"

stage=running-eighteen-trial-completep-ladder
echo "$stage" > "$root/h100-stage"
"$python" scripts/run_forecast_task_pool.py --phase ladder \
  --campaign completep "$root/bound-config.json" "$root" --cli "$cli" \
  --gpus 0,1,2,3,4,5,6,7 --status "$root/ladder-pool-status.json"

stage=aggregating-completep-result
echo "$stage" > "$root/h100-stage"
ladder_args=()
while IFS= read -r -d '' directory; do
  ladder_args+=(--cache-directory "$directory")
done < <(find "$root/ladder/tasks" -type d -name trials -print0)
"$cli" forecast-aggregate "$root/bound-config.json" \
  "${cache_args[@]}" "${ladder_args[@]}" --output "$root/aggregate" \
  > "$root/aggregate.stdout.json"

trap - EXIT
echo complete > "$root/h100-stage"
