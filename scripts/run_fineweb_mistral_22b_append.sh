#!/usr/bin/env bash
set -euo pipefail

repo="${AI_THEORIST_REPO:-/home/ubuntu/ai-theorist}"
python="${AI_THEORIST_PYTHON:-$repo/.venv-forecast/bin/python}"
required_commit="${AI_THEORIST_DENSE_HORIZON_COMMIT:?pin the corpus build commit}"
root="${AI_THEORIST_FINEWEB_22B_ROOT:-$repo/runs/forecast-corpora/mistral-fineweb-22bt-continuation-v2}"
prior="$repo/runs/forecast-corpora/mistral-fineweb-100bt-continuation-v1"
original="$repo/runs/forecast-corpora/mistral-fineweb-12g-v1/64538066147e9fbe"
failed="$repo/runs/forecast-corpora/mistral-fineweb-48g-v1/d27db766bf4cd0d1"
builder="$repo/scripts/materialize_fineweb_100bt_continuation.py"
prior_fingerprint=25a2fcdd8d274875f31df97b6801e78a7a836d8e01be2c55f4875f6b7f46c409
required_tokens=21000000000
extra_text_bytes=$((44 * 1024 * 1024 * 1024))
start_source_row=2997112
stage=starting

mkdir -p "$root"
exec 9>"$root/controller.lock"
flock -n 9 || { echo "another 22B corpus controller holds the lock" >&2; exit 1; }
echo "$$" > "$root/controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$root/stage"; fi' EXIT

[[ "$(git -C "$repo" rev-parse HEAD)" == "$required_commit" ]]
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
[[ -x "$python" && -f "$builder" ]]
[[ -f "$prior/token-streams/manifest.json" ]]
[[ -f "$prior/extra-materialization.json" ]]

export PYTHONPATH="$repo/src"
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-32}"

# Reuse already verified parquet bytes by hard link; newly needed immutable
# source files are downloaded into this new campaign cache.
mkdir -p "$root/source-parquet-100bt"
for source in "$prior"/source-parquet-100bt/*.parquet; do
  target="$root/source-parquet-100bt/$(basename "$source")"
  [[ -e "$target" ]] || ln "$source" "$target"
done

common=(
  --output-root "$root"
  --base-stream "$prior/token-streams/manifest.json"
  --base-tokenizer "$prior/tokenizer/manifest.json"
  --base-train "$original/train.jsonl"
  --base-validation "$original/validation.jsonl"
  --secondary "$failed/.train-secondary.jsonl.partial"
  --secondary-checkpoint "$failed/.train-secondary.jsonl.partial.json"
  --append-only
  --expected-base-fingerprint "$prior_fingerprint"
  --exclude-jsonl "$prior/extra-100bt.jsonl"
  --start-source-row "$start_source_row"
  --required-train-tokens "$required_tokens"
  --extra-text-bytes "$extra_text_bytes"
)

stage=verifying-prior-prefix-and-tokenizer
echo "$stage" > "$root/stage"
"$python" "$builder" initialize "${common[@]}" \
  > "$root/initialize.log" 2>&1

stage=materializing-deduplicated-sample100bt-append
echo "$stage" > "$root/stage"
"$python" "$builder" materialize-extra "${common[@]}" \
  > "$root/materialize-extra.log" 2>&1

stage=tokenizing-append-with-pinned-mistral-tokenizer
echo "$stage" > "$root/stage"
"$python" "$builder" tokenize "${common[@]}" --segment extra \
  > "$root/tokenize-extra.log" 2>&1

stage=assembling-immutable-22b-prefix-stream
echo "$stage" > "$root/stage"
"$python" "$builder" assemble "${common[@]}" \
  > "$root/assemble.log" 2>&1

"$python" - "$root/manifest.json" "$required_tokens" "$prior_fingerprint" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); required=int(sys.argv[2]); prior=sys.argv[3]
assert p["status"] == "complete" and all(p["gates"].values())
assert p["append_only"] is True
assert p["base_stream_fingerprint"] == prior
assert int(p["continuation_train_tokens"]) >= required
PY

trap - EXIT
echo complete > "$root/stage"
echo "completed immutable FineWeb/Mistral 22B-token continuation: $root/token-streams/manifest.json"
