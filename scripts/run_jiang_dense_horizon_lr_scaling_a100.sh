#!/usr/bin/env bash
set -euo pipefail

repo="${AI_THEORIST_REPO:-/home/ubuntu/ai-theorist}"
python="${AI_THEORIST_PYTHON:-$repo/.venv-forecast/bin/python}"
required_commit="${AI_THEORIST_DENSE_LR_HORIZON_COMMIT:?pin the horizon-LR commit}"
root="${AI_THEORIST_DENSE_LR_HORIZON_ROOT:-$repo/runs/forecast-production/jiang-dense-horizon-lr-tminus-third-v1}"
corpus="${AI_THEORIST_FINEWEB_22B_ROOT:-$repo/runs/forecast-corpora/mistral-fineweb-22bt-continuation-v2}"
manifest="$corpus/token-streams/manifest.json"
source_300="$repo/runs/forecast-production/jiang-rho32-300m-horizon-pair-v1"
source_1b="$repo/runs/forecast-production/jiang-mistral-1b-10tpp-exploratory-v1"
failure_root="$repo/runs/forecast-production/jiang-dense-300m40-1b20-v1"
receipt="$failure_root/token-stream-verification-receipt.json"
stage=starting

mkdir -p "$root"
exec 9>"$root/controller.lock"
flock -n 9 || { echo "another target-horizon LR controller holds the lock" >&2; exit 1; }
echo "$$" > "$root/controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$root/stage"; fi' EXIT

[[ "$(git -C "$repo" rev-parse HEAD)" == "$required_commit" ]]
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
[[ -x "$python" ]]
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')" == 8 ]]
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq '^NVIDIA A100-SXM4-80GB$'; then
  echo "target-horizon campaign requires eight A100-SXM4-80GB GPUs" >&2
  exit 1
fi
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "volatile uncorrected ECC error detected" >&2
  exit 1
fi
if systemctl list-unit-files nvidia-fabricmanager.service >/dev/null 2>&1; then
  [[ "$(systemctl is-active nvidia-fabricmanager)" == active ]]
fi

# Each DDP rank maps all 1,358 immutable token shards. The cloud image's
# default soft limit of 1,024 is insufficient even though its hard limit is
# much larger.
if (( $(ulimit -Sn) < 65536 )); then
  ulimit -Sn 65536
fi
(( $(ulimit -Sn) >= 65536 ))

export AI_THEORIST_PYTHON="$python"
export AI_THEORIST_AUTOSCALE="$repo/scripts/ai_theorist_autoscale_python.sh"
export PYTHONPATH="$repo/src"

json_is_complete() {
  "$python" - "$1" <<'PY' >/dev/null 2>&1
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and json.load(open(p)).get("status") == "completed" else 1)
PY
}

stage=preregistering-target-horizon-learning-rate-rule
echo "$stage" > "$root/stage"
if [[ ! -f "$root/preregistration.json" ]]; then
  (
    cd "$repo"
    "$python" "$repo/scripts/prepare_jiang_dense_horizon_lr_scaling.py" \
      "$source_300" "$source_1b" "$manifest" "$receipt" "$failure_root" \
      --output-root "$root"
  ) > "$root/preregistration.stdout.json"
fi
"$python" - "$root/preregistration.json" "$required_commit" <<'PY'
import json, math, sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "preregistered" and all(p["gates"].values())
assert p["repo_commit"] == sys.argv[2]
expected={
    "dense_300m_20tpp": 0.03 * 2.0 ** (-1.0/3.0),
    "dense_300m_40tpp": 0.03 * 4.0 ** (-1.0/3.0),
    "dense_1b_20tpp": 0.02 * 2.0 ** (-1.0/3.0),
}
for key, eta in expected.items():
    assert math.isclose(p["campaigns"][key]["selected_learning_rate"], eta,
                        rel_tol=0.0, abs_tol=1e-15)
PY

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 10
done

for label in dense-300m-20tpp dense-300m-40tpp dense-1b-20tpp; do
  stage="qualifying-${label}-runtime"
  echo "$stage" > "$root/stage"
  if [[ ! -f "$root/$label/runtime-qualification.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$python" \
      "$repo/scripts/qualify_fixed_budget_runtime.py" \
      "$root/$label/config.json" --single-process-ddp-equivalent \
      --output "$root/$label/runtime-qualification.json" \
      > "$root/$label/runtime-qualification.stdout.json"
  fi
  "$python" - "$root/$label/runtime-qualification.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); assert p["status"] == "passed"
PY
done

run_endpoint() {
  local label="$1"
  local key="$2"
  local trial_root="$root/$label/trial"
  local shard="$trial_root/ladder-shard-000.json"
  if json_is_complete "$shard"; then
    return
  fi
  local eta task
  eta="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaigns"][sys.argv[2]]["selected_learning_rate"])' "$root/preregistration.json" "$key")"
  task="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaigns"][sys.argv[2]]["task_id"])' "$root/preregistration.json" "$key")"
  mkdir -p "$trial_root"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
    -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m ai_theorist.autoscaler.cli forecast-shard \
    "$root/$label/config.json" --phase ladder --shard-index 0 --shard-count 1 \
    --selected-learning-rate "$eta" --task-id "$task" --device cuda \
    --output "$trial_root" --progress-jsonl \
    > "$root/$label/worker.log" 2>&1
}

evaluate_phase() {
  local label="$1"
  local key="$2"
  "$python" "$repo/scripts/evaluate_jiang_dense_horizon_lr_phase.py" \
    "$root/preregistration.json" "$key" \
    "$root/$label/trial/ladder-shard-000.json" \
    --output "$root/$label/phase-result.json" \
    > "$root/$label/phase-evaluation.stdout.json"
}

stage=running-fresh-300m-20tpp-eta-tminus-third
echo "$stage" > "$root/stage"
run_endpoint dense-300m-20tpp dense_300m_20tpp
stage=gating-fresh-300m-20tpp-stability
echo "$stage" > "$root/stage"
evaluate_phase dense-300m-20tpp dense_300m_20tpp

stage=running-fresh-300m-40tpp-eta-tminus-third
echo "$stage" > "$root/stage"
run_endpoint dense-300m-40tpp dense_300m_40tpp
stage=gating-fresh-300m-40tpp-stability
echo "$stage" > "$root/stage"
evaluate_phase dense-300m-40tpp dense_300m_40tpp

stage=running-fresh-1b-20tpp-eta-tminus-third
echo "$stage" > "$root/stage"
run_endpoint dense-1b-20tpp dense_1b_20tpp
stage=gating-fresh-1b-20tpp-stability
echo "$stage" > "$root/stage"
evaluate_phase dense-1b-20tpp dense_1b_20tpp

stage=aggregating-target-horizon-learning-rate-results
echo "$stage" > "$root/stage"
"$python" "$repo/scripts/evaluate_jiang_dense_horizon_lr_scaling.py" \
  "$root/preregistration.json" \
  "$root/dense-300m-20tpp/phase-result.json" \
  "$root/dense-300m-40tpp/phase-result.json" \
  "$root/dense-1b-20tpp/phase-result.json" \
  --output "$root/result.json" > "$root/evaluation.stdout.json"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader > "$root/final-gpu-health.csv"
trap - EXIT
echo complete > "$root/stage"
echo "completed T^-1/3 dense horizon campaign: $root/result.json"
