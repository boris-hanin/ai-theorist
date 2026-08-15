#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 EXISTING_PID RUN_ROOT CONFIG [CONFIG ...]" >&2
  exit 2
fi

existing_pid="$1"
run_root="$2"
shift 2
python_bin="${A100_PYTHON_BIN:-python3}"

mkdir -p "$run_root"
echo "$(date -u +%FT%TZ) queue-start existing_pid=$existing_pid"
while kill -0 "$existing_pid" 2>/dev/null; do
  echo "$(date -u +%FT%TZ) waiting-for-existing-work pid=$existing_pid"
  sleep 30
done

for config in "$@"; do
  campaign="$(basename "$config" .json)"
  output="$run_root/$campaign"
  echo "$(date -u +%FT%TZ) campaign-start config=$config output=$output"
  PYTHONPATH=src "$python_bin" -m ai_theorist.autoscaler.cli run \
    "$config" \
    --device cuda \
    --output "$output" \
    --summary \
    --progress-jsonl
  echo "$(date -u +%FT%TZ) campaign-complete config=$config output=$output"
done

echo "$(date -u +%FT%TZ) queue-complete"
