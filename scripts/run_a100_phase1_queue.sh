#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 OUTPUT_ROOT EXPECTED_COMMIT CONFIG [CONFIG ...]" >&2
  exit 2
fi

output_root="$1"
expected_commit="$2"
shift 2
python_bin="${A100_PYTHON_BIN:-python3}"
actual_commit="$(git rev-parse HEAD)"

if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "commit mismatch: expected $expected_commit, got $actual_commit" >&2
  exit 1
fi

mkdir -p "$output_root"
for config in "$@"; do
  campaign="$(basename "$config" .json)"
  output="$output_root/$campaign"
  echo "$(date -u +%FT%TZ) phase1-start config=$config output=$output"
  PYTHONPATH=src "$python_bin" -m ai_theorist.autoscaler.cli run \
    "$config" \
    --device cuda \
    --output "$output" \
    --summary \
    --progress-jsonl \
    2>&1 | tee "$output_root/$campaign.log"
  echo "$(date -u +%FT%TZ) phase1-complete config=$config output=$output"
done

echo "$(date -u +%FT%TZ) phase1-queue-complete"
