#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 MANIFEST KNOWN_300M_RESULT CALIBRATION_ROOT EXTENSION_ROOT CONTINUATION_CORPUS_ROOT" >&2
  exit 2
fi

manifest="$1"
known_300m_result="$2"
calibration_root="$3"
extension_root="$4"
continuation_corpus_root="$5"
template="configs/autoscaler/jiang_mistral_10tpp_calibration_200m.json"
continuation_corpus_config="configs/autoscaler/fineweb_edu_mistral_1b_10tpp_corpus.json"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
current_stage="starting"

mkdir -p "$calibration_root" "$extension_root" "$continuation_corpus_root"
exec 9>"$calibration_root/controller.lock"
if ! flock -n 9; then
  echo "another 10-TPP controller holds $calibration_root/controller.lock" >&2
  exit 1
fi
echo "$$" > "$calibration_root/controller.pid"

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    echo "failed:$current_stage" > "$calibration_root/stage"
  fi
}
trap on_exit EXIT

required_commit="${REQUIRED_REPO_COMMIT:-}"
if [[ -n "$required_commit" && "$(git rev-parse HEAD)" != "$required_commit" ]]; then
  echo "repository does not match the required commit" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked repository changes must be clean" >&2
  exit 1
fi
if [[ ! -f "$manifest" || ! -f "$known_300m_result" || ! -x "$cli" || ! -x "$python" ]]; then
  echo "manifest, prior result, or forecast runtime is missing" >&2
  exit 1
fi
if [[ "$(systemctl is-active nvidia-fabricmanager)" != "active" ]]; then
  echo "NVIDIA Fabric Manager is not active" >&2
  exit 1
fi
if (( $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l) != 8 )); then
  echo "the 10-TPP campaign requires eight GPUs" >&2
  exit 1
fi
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "one or more GPUs report an uncorrected ECC error" >&2
  exit 1
fi
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  current_stage="waiting-for-idle-gpus"
  echo "$current_stage" > "$calibration_root/stage"
  sleep 30
done
echo "$(git rev-parse HEAD)" > "$calibration_root/repo-commit"

continuation_corpus_id="$($python -c \
  'import json,sys; from ai_theorist.autoscaler.public_corpora import PublicCorpusSpec; print(PublicCorpusSpec.from_dict(json.load(open(sys.argv[1]))).fingerprint)' \
  "$continuation_corpus_config")"
continuation_manifest="$continuation_corpus_root/$continuation_corpus_id/token-streams/manifest.json"
continuation_corpus_pid=""
if [[ ! -f "$continuation_manifest" ]]; then
  recorded_corpus_pid="$(cat "$calibration_root/continuation-corpus.pid" 2>/dev/null || true)"
  if [[ -n "$recorded_corpus_pid" ]] && kill -0 "$recorded_corpus_pid" 2>/dev/null; then
    continuation_corpus_pid="$recorded_corpus_pid"
  else
    "$cli" corpus-materialize "$continuation_corpus_config" \
      --output-root "$continuation_corpus_root" --progress-jsonl \
      > "$calibration_root/continuation-corpus.jsonl" 2> "$calibration_root/continuation-corpus.log" &
    continuation_corpus_pid="$!"
    echo "$continuation_corpus_pid" > "$calibration_root/continuation-corpus.pid"
  fi
fi

collect_caches() {
  local root="$1"
  local phase="$2"
  local -n output="$3"
  while IFS= read -r -d '' directory; do
    output+=(--cache-directory "$directory")
  done < <(find "$root/$phase/tasks" -type d -name trials -print0)
  if (( ${#output[@]} == 0 )); then
    echo "no $phase caches found under $root" >&2
    exit 1
  fi
}

current_stage="binding-current-fineweb-mistral-stream"
echo "$current_stage" > "$calibration_root/stage"
"$cli" forecast-bind "$template" "$manifest" \
  --output "$calibration_root/bound-config.json" \
  > "$calibration_root/bind-summary.json"
"$cli" forecast-plan "$calibration_root/bound-config.json" \
  > "$calibration_root/plan.json"
"$python" - "$calibration_root/plan.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p["dataset_identity"]["fingerprint"] == "1b854ee220230e0421acd8312d313a72d396de2234474ec20f63ba1ce4f1d703"
assert p["dataset_identity"]["tokenizer_fingerprint"] == "d52f662783555cbf11f6a0cd8af35016652cda033389db471813c7d30f6958c5"
PY

current_stage="qualifying-200m-endpoint-and-optimizer-groups"
echo "$current_stage" > "$calibration_root/stage"
CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_fixed_budget_runtime.py \
  "$calibration_root/bound-config.json" \
  --output "$calibration_root/runtime-qualification.json" \
  > "$calibration_root/runtime-qualification.stdout.json"

current_stage="preregistering-10tpp-calibration-before-outcomes"
echo "$current_stage" > "$calibration_root/stage"
"$python" scripts/preregister_jiang_10tpp_calibration.py \
  "$calibration_root/bound-config.json" \
  "$calibration_root/runtime-qualification.json" \
  "$known_300m_result" \
  --output "$calibration_root/preregistration.json" \
  > "$calibration_root/preregistration.stdout.json"

current_stage="tuning-reference-eta-at-10tpp"
echo "$current_stage" > "$calibration_root/stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase tune \
  --campaign jiang-10tpp "$calibration_root/bound-config.json" "$calibration_root" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$calibration_root/tune-pool-status.json"

current_stage="selecting-interior-reference-eta"
echo "$current_stage" > "$calibration_root/stage"
tune_caches=()
collect_caches "$calibration_root" tune tune_caches
"$cli" forecast-select "$calibration_root/bound-config.json" \
  "${tune_caches[@]}" --require-interior \
  --output "$calibration_root/reference-selection.json" \
  > "$calibration_root/reference-selection.stdout.json"

current_stage="running-10tpp-ladder-through-200m"
echo "$current_stage" > "$calibration_root/stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase ladder \
  --campaign jiang-10tpp "$calibration_root/bound-config.json" "$calibration_root" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$calibration_root/ladder-pool-status.json"

current_stage="aggregating-10tpp-scaling-law"
echo "$current_stage" > "$calibration_root/stage"
ladder_caches=()
collect_caches "$calibration_root" ladder ladder_caches
"$cli" forecast-aggregate "$calibration_root/bound-config.json" \
  "${tune_caches[@]}" "${ladder_caches[@]}" \
  --output "$calibration_root/aggregate" \
  > "$calibration_root/aggregate.stdout.json"

current_stage="freezing-prospective-1b-prediction"
echo "$current_stage" > "$calibration_root/stage"
"$python" scripts/evaluate_jiang_10tpp_calibration.py \
  "$calibration_root/preregistration.json" \
  "$calibration_root/aggregate/result.json" \
  "$known_300m_result" \
  --output "$calibration_root/result.json" \
  > "$calibration_root/evaluation.stdout.json"

current_stage="compiling-and-preregistering-1b-endpoint"
echo "$current_stage" > "$calibration_root/stage"
if [[ ! -f "$continuation_manifest" ]]; then
  current_stage="waiting-for-verified-fineweb-continuation"
  echo "$current_stage" > "$calibration_root/stage"
  if [[ -n "$continuation_corpus_pid" ]] && [[ "$(ps -o ppid= -p "$continuation_corpus_pid" 2>/dev/null | tr -d ' ')" == "$$" ]]; then
    wait "$continuation_corpus_pid"
  else
    while [[ ! -f "$continuation_manifest" ]]; do
      if [[ -z "$continuation_corpus_pid" ]] || ! kill -0 "$continuation_corpus_pid" 2>/dev/null; then
        echo "FineWeb continuation materializer exited before writing its manifest" >&2
        exit 1
      fi
      sleep 30
    done
  fi
fi
"$python" scripts/verify_token_stream_continuation.py \
  "$manifest" "$continuation_manifest" \
  --required-prefix-tokens 2000158720 \
  --minimum-training-tokens 10085203968 \
  --output "$calibration_root/continuation-verification.json" \
  > "$calibration_root/continuation-verification.stdout.json"
"$python" scripts/prepare_jiang_1b_10tpp_extension.py \
  "$calibration_root/bound-config.json" \
  "$calibration_root/aggregate/result.json" \
  "$calibration_root/result.json" \
  "$continuation_manifest" \
  "$calibration_root/continuation-verification.json" \
  --output-root "$extension_root" \
  > "$extension_root/preparation.stdout.json"
echo "$$" > "$extension_root/controller.pid"
echo "$current_stage" > "$extension_root/stage"

selected_eta="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_learning_rate"])' "$extension_root/preregistration.json")"
task_id="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_id"])' "$extension_root/preregistration.json")"

current_stage="qualifying-1b-memory-and-optimizer-groups"
echo "$current_stage" > "$calibration_root/stage"
echo "$current_stage" > "$extension_root/stage"
CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_fixed_budget_runtime.py \
  "$extension_root/config.json" \
  --output "$extension_root/runtime-qualification.json" \
  > "$extension_root/runtime-qualification.stdout.json"

current_stage="qualifying-1b-single-vs-eight-gpu-topology"
echo "$current_stage" > "$calibration_root/stage"
echo "$current_stage" > "$extension_root/stage"
mkdir -p "$extension_root/topology/single" "$extension_root/topology/ddp"
"$python" scripts/prepare_forecast_runtime_canary.py \
  "$extension_root/config.json" "$continuation_manifest" \
  "$extension_root/topology/single-config.json" \
  --steps 3 --seed 11 --learning-rate "$selected_eta" --fused true \
  --checkpoint-steps 0 --checkpoint-seconds 900 --distributed none \
  --num-processes 1 --gradient-accumulation-steps 32 \
  > "$extension_root/topology/single-preparation.json"
"$python" scripts/prepare_forecast_runtime_canary.py \
  "$extension_root/config.json" "$continuation_manifest" \
  "$extension_root/topology/ddp-config.json" \
  --steps 3 --seed 11 --learning-rate "$selected_eta" --fused true \
  --checkpoint-steps 0 --checkpoint-seconds 900 --distributed ddp \
  --num-processes 8 --gradient-accumulation-steps 32 \
  > "$extension_root/topology/ddp-preparation.json"
"$cli" forecast-plan "$extension_root/topology/single-config.json" \
  > "$extension_root/topology/single-plan.json"
"$cli" forecast-plan "$extension_root/topology/ddp-config.json" \
  > "$extension_root/topology/ddp-plan.json"
CUDA_VISIBLE_DEVICES=0 "$cli" forecast-shard \
  "$extension_root/topology/single-config.json" \
  --phase ladder --shard-index 0 --shard-count 1 \
  --selected-learning-rate "$selected_eta" --task-id "$task_id" --device cuda \
  --output "$extension_root/topology/single" --progress-jsonl \
  > "$extension_root/topology/single.log" 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m ai_theorist.autoscaler.cli forecast-shard \
  "$extension_root/topology/ddp-config.json" \
  --phase ladder --shard-index 0 --shard-count 1 \
  --selected-learning-rate "$selected_eta" --task-id "$task_id" --device cuda \
  --output "$extension_root/topology/ddp" --progress-jsonl \
  > "$extension_root/topology/ddp.log" 2>&1
"$python" scripts/evaluate_forecast_shard_topology.py \
  "$extension_root/topology/single/ladder-shard-000.json" \
  "$extension_root/topology/ddp/ladder-shard-000.json" \
  "$extension_root/topology/single-plan.json" \
  --maximum-loss-delta 0.005 \
  --output "$extension_root/topology/comparison.json" \
  > "$extension_root/topology/comparison.stdout.json"

current_stage="running-prospective-1b-10tpp-endpoint"
echo "$current_stage" > "$calibration_root/stage"
echo "$current_stage" > "$extension_root/stage"
mkdir -p "$extension_root/trial"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m ai_theorist.autoscaler.cli forecast-shard \
  "$extension_root/config.json" \
  --phase ladder --shard-index 0 --shard-count 1 \
  --selected-learning-rate "$selected_eta" --task-id "$task_id" --device cuda \
  --output "$extension_root/trial" --progress-jsonl \
  > "$extension_root/worker.log" 2>&1

current_stage="evaluating-prospective-1b-prediction"
echo "$current_stage" > "$calibration_root/stage"
echo "$current_stage" > "$extension_root/stage"
"$python" scripts/evaluate_jiang_1b_10tpp_extension.py \
  "$extension_root/preregistration.json" \
  "$extension_root/trial/ladder-shard-000.json" \
  "$extension_root/topology/comparison.json" \
  --output "$extension_root/result.json" \
  > "$extension_root/evaluation.stdout.json"

trap - EXIT
echo "complete" > "$calibration_root/stage"
echo "complete" > "$extension_root/stage"
echo "completed Jiang 10-TPP calibration and 1B endpoint"
