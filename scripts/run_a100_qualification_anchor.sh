#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 EXISTING_QUEUE_PID OUTPUT_ROOT EXPECTED_COMMIT" >&2
  exit 2
fi

existing_pid="$1"
output_root="$2"
expected_commit="$3"
python_bin="${A100_PYTHON_BIN:-python3}"
config="configs/autoscaler/a100_qualification_anchor.json"

mkdir -p "$output_root"
echo "$(date -u +%FT%TZ) qualification-wait existing_pid=$existing_pid"
while kill -0 "$existing_pid" 2>/dev/null; do
  echo "$(date -u +%FT%TZ) waiting-for-existing-queue pid=$existing_pid"
  sleep 30
done

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "commit mismatch: expected $expected_commit, got $actual_commit" >&2
  exit 1
fi

{
  echo "timestamp=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "commit=$actual_commit"
  uname -a
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,temperature.gpu,pstate,clocks.sm,ecc.errors.uncorrected.volatile.total --format=csv,noheader
  "$python_bin" --version
  "$python_bin" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")'
} > "$output_root/inventory.txt"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$python_bin" -m pytest -q \
  > "$output_root/tests.log" 2>&1

echo "$(date -u +%FT%TZ) qualification-anchor-start"
PYTHONPATH=src "$python_bin" -m ai_theorist.autoscaler.cli run \
  "$config" \
  --device cuda \
  --output "$output_root/anchor" \
  --summary \
  --progress-jsonl \
  2>&1 | tee "$output_root/anchor.log"
echo "$(date -u +%FT%TZ) qualification-anchor-complete"
