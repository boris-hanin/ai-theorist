#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PREVIOUS_RUN_ROOT SUITE_ROOT CORPUS_ROOT" >&2
  exit 2
fi

previous_root="$1"
suite_root="$2"
corpus_root="$3"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
corpus_config="configs/autoscaler/slimpajama_gpt2_300m_corpus.json"
jiang_config="configs/autoscaler/jiang_slimpajama_gpt2_paper_300m_rho32.json"
completep_config="configs/autoscaler/completep_slimpajama_gpt2_paper_anchor.json"
jiang_root="$suite_root/jiang-chizat"
anchor_root="$suite_root/completep-paper-anchor"
current_stage="starting"

mkdir -p "$suite_root" "$corpus_root"
exec 9>"$suite_root/controller.lock"
if ! flock -n 9; then
  echo "another controller holds $suite_root/controller.lock" >&2
  exit 1
fi
echo "$$" > "$suite_root/controller.pid"

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    echo "failed:$current_stage" > "$suite_root/stage"
  fi
}
trap on_exit EXIT

if [[ ! -x "$cli" || ! -x "$python" ]]; then
  echo "forecast virtual environment is incomplete" >&2
  exit 2
fi
repo_commit="$(git rev-parse HEAD)"
required_commit="${REQUIRED_REPO_COMMIT:-}"
if [[ -n "$required_commit" && "$repo_commit" != "$required_commit" ]]; then
  echo "repository commit $repo_commit does not match required $required_commit" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked repository changes must be clean before launch" >&2
  exit 1
fi
echo "$repo_commit" > "$suite_root/repo-commit"

current_stage="waiting-for-previous-300m-run"
echo "$current_stage" > "$suite_root/stage"
previous_pid_file="$previous_root/override-controller.pid"
while [[ -f "$previous_pid_file" ]]; do
  previous_pid="$(cat "$previous_pid_file" 2>/dev/null || true)"
  if [[ -z "$previous_pid" ]] || ! kill -0 "$previous_pid" 2>/dev/null; then
    break
  fi
  sleep 30
done

current_stage="waiting-for-idle-gpus"
echo "$current_stage" > "$suite_root/stage"
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

current_stage="waiting-for-huggingface-slimpajama-access"
echo "$current_stage" > "$suite_root/stage"
while [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && ! -s "$HOME/.cache/huggingface/token" ]]; do
  sleep 60
done
until "$python" -c \
  'from ai_theorist.autoscaler.public_corpora import _source_revision; print(_source_revision("cerebras/SlimPajama-627B"))' \
  > "$suite_root/source-revision-probe.txt" \
  2> "$suite_root/source-access-probe.log"; do
  echo "$current_stage" > "$suite_root/stage"
  sleep 60
done

current_stage="materializing-immutable-slimpajama-gpt2"
echo "$current_stage" > "$suite_root/stage"
"$cli" corpus-materialize "$corpus_config" \
  --output-root "$corpus_root" --progress-jsonl \
  > "$suite_root/corpus-materialization.jsonl"
corpus_id="$("$python" -c \
  'import json,sys; from ai_theorist.autoscaler.public_corpora import PublicCorpusSpec; print(PublicCorpusSpec.from_dict(json.load(open(sys.argv[1]))).fingerprint)' \
  "$corpus_config")"
corpus_manifest="$corpus_root/$corpus_id/manifest.json"
manifest="$corpus_root/$corpus_id/token-streams/manifest.json"
if [[ ! -f "$corpus_manifest" || ! -f "$manifest" ]]; then
  echo "corpus materialization completed without verified manifests" >&2
  exit 1
fi

if [[ "$(systemctl is-active nvidia-fabricmanager)" != "active" ]]; then
  echo "NVIDIA Fabric Manager is not active" >&2
  exit 1
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')"
if (( gpu_count != 8 )); then
  echo "paper rerun requires exactly eight visible GPUs; found $gpu_count" >&2
  exit 1
fi
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "one or more GPUs report volatile uncorrected ECC errors" >&2
  exit 1
fi

mkdir -p "$jiang_root" "$anchor_root"
current_stage="binding-and-compiling-paper-coordinate-plans"
echo "$current_stage" > "$suite_root/stage"
"$cli" forecast-bind "$jiang_config" "$manifest" \
  --output "$jiang_root/bound-config.json" \
  > "$jiang_root/bind-summary.json"
"$cli" forecast-bind "$completep_config" "$manifest" \
  --output "$anchor_root/bound-config.json" \
  > "$anchor_root/bind-summary.json"
"$cli" forecast-plan "$jiang_root/bound-config.json" > "$jiang_root/plan.json"
"$cli" forecast-plan "$anchor_root/bound-config.json" > "$anchor_root/plan.json"

current_stage="qualifying-jiang-runtime"
echo "$current_stage" > "$suite_root/stage"
CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_fixed_budget_runtime.py \
  "$jiang_root/bound-config.json" \
  --output "$suite_root/jiang-runtime-qualification.json" \
  > "$suite_root/jiang-runtime-qualification.stdout.json"

current_stage="qualifying-exact-completep-paper-anchor"
echo "$current_stage" > "$suite_root/stage"
CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_completep_paper_anchor.py \
  "$anchor_root/bound-config.json" \
  --output "$suite_root/completep-anchor-qualification.json" \
  > "$suite_root/completep-anchor-qualification.stdout.json"

current_stage="preregistering-paper-coordinate-rerun"
echo "$current_stage" > "$suite_root/stage"
"$python" scripts/preregister_slimpajama_gpt2_paper_rerun.py \
  "$corpus_manifest" "$jiang_root/bound-config.json" \
  "$anchor_root/bound-config.json" \
  "$suite_root/jiang-runtime-qualification.json" \
  "$suite_root/completep-anchor-qualification.json" \
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

current_stage="tuning-jiang-and-running-completep-anchor"
echo "$current_stage" > "$suite_root/stage"
echo "tuning" > "$jiang_root/controller-stage"
echo "published-lr-anchor" > "$anchor_root/controller-stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase tune \
  --campaign jiang "$jiang_root/bound-config.json" "$jiang_root" \
  --campaign completep-anchor "$anchor_root/bound-config.json" "$anchor_root" \
  --task-id completep-anchor=tune-S1-theory-eta0.00390625-seed11 \
  --task-id completep-anchor=tune-S1-theory-eta0.00390625-seed29 \
  --task-id completep-anchor=tune-S1-theory-eta0.00390625-seed47 \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$suite_root/tune-pool-status.json"

current_stage="selecting-interior-jiang-reference-lr"
echo "$current_stage" > "$suite_root/stage"
jiang_tune_cache_args=()
collect_cache_arguments "$jiang_root" tune jiang_tune_cache_args
"$cli" forecast-select "$jiang_root/bound-config.json" \
  "${jiang_tune_cache_args[@]}" --require-interior \
  --output "$jiang_root/reference-selection.json" \
  > "$jiang_root/reference-selection.stdout.json"

current_stage="running-jiang-paper-coordinate-ladder"
echo "$current_stage" > "$suite_root/stage"
echo "ladder" > "$jiang_root/controller-stage"
"$python" scripts/run_forecast_task_pool.py \
  --phase ladder \
  --campaign jiang "$jiang_root/bound-config.json" "$jiang_root" \
  --cli "$cli" --gpus 0,1,2,3,4,5,6,7 \
  --status "$suite_root/ladder-pool-status.json"

current_stage="aggregating-jiang-scaling-law"
echo "$current_stage" > "$suite_root/stage"
jiang_ladder_cache_args=()
collect_cache_arguments "$jiang_root" ladder jiang_ladder_cache_args
"$cli" forecast-aggregate "$jiang_root/bound-config.json" \
  "${jiang_tune_cache_args[@]}" "${jiang_ladder_cache_args[@]}" \
  --output "$jiang_root/aggregate" \
  > "$jiang_root/aggregate.stdout.json"
echo "complete" > "$jiang_root/controller-stage"
echo "complete" > "$anchor_root/controller-stage"

current_stage="evaluating-paper-coordinate-results"
echo "$current_stage" > "$suite_root/stage"
"$python" scripts/evaluate_slimpajama_gpt2_paper_rerun.py \
  "$suite_root/preregistration.json" \
  "$jiang_root/aggregate/result.json" \
  "$anchor_root/tune/tasks/0006-tune-S1-theory-eta0.00390625-seed11/tune-shard-000.json" \
  "$anchor_root/tune/tasks/0007-tune-S1-theory-eta0.00390625-seed29/tune-shard-000.json" \
  "$anchor_root/tune/tasks/0008-tune-S1-theory-eta0.00390625-seed47/tune-shard-000.json" \
  --output "$suite_root/result.json" \
  > "$suite_root/result.stdout.json"

current_stage="complete"
echo "$current_stage" > "$suite_root/stage"
