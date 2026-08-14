#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT TOKEN_STREAM_MANIFEST" >&2
  exit 2
fi

repo="${AI_THEORIST_REPO:-/home/ubuntu/ai-theorist}"
python="${AI_THEORIST_PYTHON:-$repo/.venv-forecast-py310/bin/python}"
cli="${AI_THEORIST_AUTOSCALE:-$repo/scripts/ai_theorist_autoscale_python.sh}"
required_commit="${AI_THEORIST_MOE_ACTIVE_1B_COMMIT:?set the pinned campaign commit}"
template="${AI_THEORIST_MOE_ACTIVE_1B_TEMPLATE:-$repo/configs/autoscaler/jiang_moe_slimpajama_rho32_active_1b.json}"
transfer_summary="${AI_THEORIST_MOE_TRANSFER_SUMMARY:-$repo/rounds/018-jiang-moe-constant-rho/transfer-result-summary.json}"
root="$1"
manifest="$2"
stage=starting

mkdir -p "$root"
exec 9>"$root/controller.lock"
flock -n 9 || { echo "another controller holds $root/controller.lock" >&2; exit 1; }
echo "$$" > "$root/controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$root/stage"; fi' EXIT

[[ -x "$python" && -x "$cli" && -f "$manifest" && -f "$template" ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$required_commit" ]]
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')" == 8 ]]
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq '^NVIDIA H100 80GB HBM3$'; then
  echo "the active-1B campaign requires eight H100 80GB GPUs" >&2
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

export AI_THEORIST_PYTHON="$python"
export AI_THEORIST_AUTOSCALE="$cli"
export PYTHONPATH="$repo/src"

json_is_complete() {
  "$python" - "$1" <<'PY' >/dev/null 2>&1
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and json.load(open(p)).get("status") == "completed" else 1)
PY
}

stage=verifying-and-preregistering
echo "$stage" > "$root/stage"
if [[ ! -f "$root/preregistration.json" ]]; then
  cd "$repo"
  "$python" scripts/prepare_jiang_moe_rho32_active_1b.py "$manifest" \
    --output-root "$root" --template "$template" \
    --transfer-summary "$transfer_summary" \
    > "$root/preregistration.stdout.json"
fi
"$python" - "$root/preregistration.json" "$root/plan-single.json" \
  "$root/plan-ddp.json" <<'PY'
import json, sys
pre, single, ddp=(json.load(open(path)) for path in sys.argv[1:])
assert pre["status"] == "preregistered"
assert all(pre["gates"].values())
assert pre["single_plan_fingerprint"] == single["fingerprint"]
assert pre["ddp_plan_fingerprint"] == ddp["fingerprint"]
assert pre["endpoint"]["active_parameters"] == 1_014_509_312
assert pre["endpoint"]["parameters"] == 2_336_879_360
PY

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 10
done

stage=qualifying-endpoint-memory-and-optimizer-contract
echo "$stage" > "$root/stage"
if [[ ! -f "$root/runtime-qualification.json" ]]; then
  cd "$repo"
  CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_fixed_budget_runtime.py \
    "$root/config-ddp.json" --single-process-ddp-equivalent \
    --output "$root/runtime-qualification.json" \
    > "$root/runtime-qualification.stdout.json"
fi
"$python" - "$root/runtime-qualification.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); assert p["status"] == "passed"
c=p["campaigns"][0]; assert c["architecture"] == "jiang_moe_completep_adam_table2"
assert c["single_process_ddp_equivalent"]["production_num_processes"] == 8
assert all(len(row["optimizer_groups"]) == 8 for row in c["scales"])
e=c["scales"][-1]
assert e["parameters"] == 2_336_879_360
assert e["endpoint_canary_loss"] > 0
assert e["endpoint_peak_memory_bytes"] < 80 * 1024**3
assert e["routing_diagnostics"]["maximum_absolute_expert_bias"] > 0
PY

stage=running-eight-way-reference-eta-screen
echo "$stage" > "$root/stage"
mkdir -p "$root/tune"
if [[ ! -f "$root/reference-selection.json" ]]; then
  "$cli" forecast-tasks "$root/config-single.json" --phase tune \
    --shard-count 8 > "$root/tune-assignments.json"
  pids=()
  labels=()
  for shard in $(seq 0 7); do
    shard_root="$root/tune/shard-$shard"
    mkdir -p "$shard_root"
    if json_is_complete "$shard_root/tune-shard-$(printf '%03d' "$shard").json"; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="$shard" "$cli" forecast-shard \
      "$root/config-single.json" --phase tune --shard-index "$shard" \
      --shard-count 8 --device cuda --output "$shard_root" --progress-jsonl \
      > "$shard_root/worker.log" 2>&1 &
    pids+=("$!")
    labels+=("$shard")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "tuning shard ${labels[$index]} failed" >&2
      failed=1
    fi
  done
  (( failed == 0 ))
  cache_args=()
  for shard in $(seq 0 7); do
    cache_args+=(--cache-directory "$root/tune/shard-$shard/trials")
  done
  "$cli" forecast-select "$root/config-single.json" "${cache_args[@]}" \
    --require-interior --output "$root/reference-selection.json" \
    > "$root/reference-selection.stdout.json"
fi
selected_eta="$($python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["learning_rate_optimum_is_interior"] is True; print(p["selected_learning_rate"])' "$root/reference-selection.json")"
eta_tag="$($python -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$selected_eta")"

stage=qualifying-single-vs-eight-gpu-topology
echo "$stage" > "$root/stage"
mkdir -p "$root/topology/single" "$root/topology/ddp"
if [[ ! -f "$root/topology/comparison.json" ]]; then
  cd "$repo"
  "$python" scripts/prepare_forecast_runtime_canary.py \
    "$root/config-single.json" "$manifest" "$root/topology/single-config.json" \
    --steps 2 --seed 11 --learning-rate "$selected_eta" --fused true \
    --checkpoint-steps 0 --checkpoint-seconds 600 --distributed none \
    --num-processes 1 --batch-examples 128 --gradient-accumulation-steps 16 \
    > "$root/topology/single-preparation.json"
  "$python" scripts/prepare_forecast_runtime_canary.py \
    "$root/config-ddp.json" "$manifest" "$root/topology/ddp-config.json" \
    --steps 2 --seed 11 --learning-rate "$selected_eta" --fused true \
    --checkpoint-steps 0 --checkpoint-seconds 600 --distributed ddp \
    --num-processes 8 --batch-examples 128 --gradient-accumulation-steps 2 \
    > "$root/topology/ddp-preparation.json"
  "$cli" forecast-plan "$root/topology/single-config.json" \
    > "$root/topology/single-plan.json"
  task_id="ladder-S9-theory-eta${eta_tag}-seed11"
  if ! json_is_complete "$root/topology/single/ladder-shard-000.json"; then
    CUDA_VISIBLE_DEVICES=0 "$cli" forecast-shard \
      "$root/topology/single-config.json" --phase ladder --shard-index 0 \
      --shard-count 1 --selected-learning-rate "$selected_eta" \
      --task-id "$task_id" --device cuda --output "$root/topology/single" \
      --progress-jsonl > "$root/topology/single.log" 2>&1
  fi
  if ! json_is_complete "$root/topology/ddp/ladder-shard-000.json"; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
      -m torch.distributed.run --standalone --nproc_per_node=8 \
      -m ai_theorist.autoscaler.cli forecast-shard \
      "$root/topology/ddp-config.json" --phase ladder --shard-index 0 \
      --shard-count 1 --selected-learning-rate "$selected_eta" \
      --task-id "$task_id" --device cuda --output "$root/topology/ddp" \
      --progress-jsonl > "$root/topology/ddp.log" 2>&1
  fi
  "$python" scripts/evaluate_forecast_shard_topology.py \
    "$root/topology/single/ladder-shard-000.json" \
    "$root/topology/ddp/ladder-shard-000.json" \
    "$root/topology/single-plan.json" --maximum-loss-delta 0.005 \
    --output "$root/topology/comparison.json" \
    > "$root/topology/comparison.stdout.json"
fi
"$python" - "$root/topology/comparison.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); assert p["status"] == "passed"
assert p["ddp_replicas"] == 8
assert p["maximum_observed_loss_delta"] <= 0.005
PY

cd "$repo"
for scale_index in $(seq 2 8); do
  scale="S${scale_index}"
  stage="running-${scale}-eight-gpu-ddp"
  echo "$stage" > "$root/stage"
  scale_root="$root/ladder/$scale"
  mkdir -p "$scale_root"
  if ! json_is_complete "$scale_root/ladder-shard-000.json"; then
    task_id="ladder-${scale}-theory-eta${eta_tag}-seed11"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
      -m torch.distributed.run --standalone --nproc_per_node=8 \
      -m ai_theorist.autoscaler.cli forecast-shard \
      "$root/config-ddp.json" --phase ladder --shard-index 0 --shard-count 1 \
      --selected-learning-rate "$selected_eta" --task-id "$task_id" \
      --device cuda --output "$scale_root" --progress-jsonl \
      > "$scale_root/worker.log" 2>&1
  fi
done

stage=freezing-S9-prediction-before-reveal
echo "$stage" > "$root/stage"
if [[ ! -f "$root/frozen-prediction.json" ]]; then
  "$python" scripts/freeze_jiang_moe_rho32_active_1b_prediction.py "$root" \
    --output "$root/frozen-prediction.json" \
    > "$root/frozen-prediction.stdout.json"
fi

stage=running-S9-one-billion-active-eight-gpu-ddp
echo "$stage" > "$root/stage"
scale_root="$root/ladder/S9"
mkdir -p "$scale_root"
if ! json_is_complete "$scale_root/ladder-shard-000.json"; then
  task_id="ladder-S9-theory-eta${eta_tag}-seed11"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
    -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m ai_theorist.autoscaler.cli forecast-shard \
    "$root/config-ddp.json" --phase ladder --shard-index 0 --shard-count 1 \
    --selected-learning-rate "$selected_eta" --task-id "$task_id" \
    --device cuda --output "$scale_root" --progress-jsonl \
    > "$scale_root/worker.log" 2>&1
fi

stage=evaluating-holdout-and-scaling-law
echo "$stage" > "$root/stage"
"$python" scripts/evaluate_jiang_moe_rho32_active_1b.py "$root" \
  --output "$root/result.json" > "$root/evaluation.stdout.json"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader > "$root/final-gpu-health.csv"
trap - EXIT
echo complete > "$root/stage"
echo "completed Jiang MoE rho=32 1B-active scaling law: $root/result.json"
