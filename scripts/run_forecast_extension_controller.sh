#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT CORPUS_MANIFEST PARENT_RUN_ROOT" >&2
  exit 2
fi

run_root="$1"
corpus_manifest="$2"
parent_run="$3"
template="configs/autoscaler/jiang_mistral_300m_extension.json"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"

mkdir -p "$run_root"
exec 8>"$run_root/controller.lock"
if ! flock -n 8; then
  echo "another extension controller holds $run_root/controller.lock" >&2
  exit 1
fi

failed_stage() {
  local status="$?"
  if (( status != 0 )); then
    echo extension_failed > "$run_root/stage"
  fi
  exit "$status"
}
trap failed_stage EXIT

while [[ ! -f "$corpus_manifest" ]]; do
  if [[ "$(cat "$run_root/stage" 2>/dev/null || true)" == "extension_failed" ]]; then
    echo "corpus controller previously failed" >&2
    exit 1
  fi
  corpus_pid="$(cat "$run_root/corpus.pid" 2>/dev/null || true)"
  if [[ -z "$corpus_pid" ]] || ! kill -0 "$corpus_pid" 2>/dev/null; then
    echo "corpus manifest is absent and its controller is not live" >&2
    exit 1
  fi
  sleep 30
done

echo extension_preregistering > "$run_root/stage"
.venv-forecast/bin/python scripts/prepare_forecast_extension.py \
  "$template" \
  "$corpus_manifest" \
  "$parent_run" \
  "$run_root" > "$run_root/preregistration.stdout.json"

echo extension_canary > "$run_root/stage"
mkdir -p "$run_root/canary/trial"
.venv-forecast/bin/python scripts/prepare_forecast_runtime_canary.py \
  "$run_root/bound-config.json" \
  "$corpus_manifest" \
  "$run_root/canary/config.json" \
  --steps 10 \
  --seed 11 \
  --learning-rate 0.03 \
  --fused true \
  --checkpoint-steps 0 \
  --checkpoint-seconds 900 > "$run_root/canary/plan.stdout.json"

CUDA_VISIBLE_DEVICES=7 "$cli" forecast-shard "$run_root/canary/config.json" \
  --phase ladder \
  --shard-index 0 \
  --shard-count 1 \
  --selected-learning-rate 0.03 \
  --task-id ladder-S8-theory-eta0.03-seed11 \
  --device cuda \
  --output "$run_root/canary/trial" \
  --progress-jsonl > "$run_root/canary/worker.log" 2>&1

.venv-forecast/bin/python -c '
import json, math, sys
row = json.load(open(sys.argv[1]))
assert row["status"] == "completed"
assert len(row["records"]) == 1
record = row["records"][0]
assert record["parameter_count"] == 303288704
assert record["metadata"]["optimizer_mode"] == "theory"
assert record["metadata"]["optimizer_group_audit"]["complete"] is True
assert record["metadata"]["optimizer_group_audit"]["disjoint"] is True
assert len(record["metadata"]["optimizer_group_audit"]["groups"]) == 7
assert math.isfinite(record["final_validation_loss"])
' "$run_root/canary/trial/ladder-shard-000.json"

echo extension_qualified > "$run_root/stage"
scripts/run_forecast_extension.sh "$run_root" > "$run_root/controller-production.log" 2>&1
trap - EXIT
echo extension_complete > "$run_root/stage"
echo "completed 300M extension: $run_root/result.json"
