#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SUITE_ROOT CORPUS_ROOT" >&2
  exit 2
fi

suite_root="$1"
corpus_root="$2"
cli="${AI_THEORIST_AUTOSCALE:-.venv-forecast/bin/ai-theorist-autoscale}"
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
corpus_config="configs/autoscaler/fineweb_edu_mistral_forecast_corpus.json"
current_stage="starting"

mkdir -p "$suite_root" "$corpus_root"
exec 8>"$suite_root/orchestrator.lock"
if ! flock -n 8; then
  echo "another orchestrator holds $suite_root/orchestrator.lock" >&2
  exit 1
fi
echo "$$" > "$suite_root/orchestrator.pid"

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    echo "failed:$current_stage" > "$suite_root/stage"
  fi
}
trap on_exit EXIT

current_stage="materializing-immutable-mistral-fineweb"
echo "$current_stage" > "$suite_root/stage"
"$cli" corpus-materialize "$corpus_config" \
  --output-root "$corpus_root" --progress-jsonl \
  > "$suite_root/corpus-materialization.jsonl"

corpus_id="$("$python" -c \
  'import json, sys; from ai_theorist.autoscaler.public_corpora import PublicCorpusSpec; print(PublicCorpusSpec.from_dict(json.load(open(sys.argv[1]))).fingerprint)' \
  "$corpus_config")"
manifest="$corpus_root/$corpus_id/token-streams/manifest.json"
if [[ ! -f "$manifest" ]]; then
  echo "materialization completed without token-stream manifest: $manifest" >&2
  exit 1
fi

current_stage="launching-qualified-fixed-budget-pair"
echo "$current_stage" > "$suite_root/stage"
scripts/run_fixed_budget_100m_pair.sh "$manifest" "$suite_root"
