#!/usr/bin/env bash
set -euo pipefail

worktree=/home/ubuntu/ai-theorist-corpus
python=/home/ubuntu/ai-theorist/.venv-forecast/bin/python
root=/home/ubuntu/ai-theorist/runs/forecast-corpora/mistral-fineweb-100bt-continuation-v1
base=/home/ubuntu/ai-theorist/runs/forecast-corpora/mistral-fineweb-12g-v1/64538066147e9fbe
failed=/home/ubuntu/ai-theorist/runs/forecast-corpora/mistral-fineweb-48g-v1/d27db766bf4cd0d1
builder="$root/materialize_fineweb_100bt_continuation.py"
stage=starting

mkdir -p "$root"
exec 9>"$root/controller.lock"
flock -n 9 || { echo "another FineWeb continuation controller holds the lock" >&2; exit 1; }
echo "$$" > "$root/controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$root/stage"; fi' EXIT

common=(
  --output-root "$root"
  --base-stream "$base/token-streams/manifest.json"
  --base-tokenizer "$base/tokenizer/manifest.json"
  --base-train "$base/train.jsonl"
  --base-validation "$base/validation.jsonl"
  --secondary "$failed/.train-secondary.jsonl.partial"
  --secondary-checkpoint "$failed/.train-secondary.jsonl.partial.json"
)
export PYTHONPATH="$worktree/src"
# Encode bounded document batches through the tokenizer's ordered Rayon path.
# Two segment workers share 32 threads, leaving ample CPU and memory headroom.
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=16

stage=initializing-and-verifying-old-prefix
echo "$stage" > "$root/stage"
"$python" "$builder" initialize "${common[@]}" > "$root/initialize.log" 2>&1

stage=materializing-extra-and-tokenizing-reusable-segment
echo "$stage" > "$root/stage"
"$python" "$builder" materialize-extra "${common[@]}" > "$root/materialize-extra.log" 2>&1 &
extra_pid=$!
echo "$extra_pid" > "$root/materialize-extra.pid"
"$python" "$builder" tokenize "${common[@]}" --segment secondary > "$root/tokenize-secondary.log" 2>&1 &
secondary_pid=$!
echo "$secondary_pid" > "$root/tokenize-secondary.pid"
wait "$extra_pid"

stage=tokenizing-deduplicated-sample100bt
echo "$stage" > "$root/stage"
"$python" "$builder" tokenize "${common[@]}" --segment extra > "$root/tokenize-extra.log" 2>&1 &
extra_token_pid=$!
echo "$extra_token_pid" > "$root/tokenize-extra.pid"
wait "$secondary_pid"
wait "$extra_token_pid"

stage=assembling-and-verifying-exact-prefix
echo "$stage" > "$root/stage"
"$python" "$builder" assemble "${common[@]}" > "$root/assemble.log" 2>&1

trap - EXIT
echo complete > "$root/stage"
