#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 EXISTING_QUEUE_PID OUTPUT_ROOT" >&2
  exit 2
fi

existing_pid="$1"
output_root="$2"
python_bin="${A100_PYTHON_BIN:-python3}"
harness="skills/dmft-attention/scripts/jiang_chizat_transfer.py"

mkdir -p "$output_root"
echo "$(date -u +%FT%TZ) audit-queue-start existing_pid=$existing_pid"
while kill -0 "$existing_pid" 2>/dev/null; do
  echo "$(date -u +%FT%TZ) waiting-for-nugpt pid=$existing_pid"
  sleep 30
done

common=(
  --head-dimension 64
  --vocab-size 128
  --context-length 32
  --n-train 256
  --n-validation 64
  --eta 0.001
  --epsilon0 1e-12
  --steps 1
  --batch-size 16
  --seeds 11 29 47 71
  --rules primary
  --audit-only
  --device cuda
)

echo "$(date -u +%FT%TZ) pure-D-audit-start"
PYTHONPATH=src "$python_bin" "$harness" \
  --shapes D128:4:256:128:128 D192:4:256:192:192 D256:4:256:256:256 D384:4:256:384:384 D512:4:256:512:512 \
  --reference-L 4 --reference-M 256 --reference-D 256 \
  "${common[@]}" --output "$output_root/pure-D.json"

echo "$(date -u +%FT%TZ) pure-M-audit-start"
PYTHONPATH=src "$python_bin" "$harness" \
  --shapes M64:4:64:256:64 M128:4:128:256:128 M256:4:256:256:256 M512:4:512:256:512 M1024:4:1024:256:1024 \
  --reference-L 4 --reference-M 256 --reference-D 256 \
  "${common[@]}" --output "$output_root/pure-M.json"

echo "$(date -u +%FT%TZ) pure-L-audit-start"
PYTHONPATH=src "$python_bin" "$harness" \
  --shapes L1:1:256:256:1 L2:2:256:256:2 L4:4:256:256:4 L8:8:256:256:8 L16:16:256:256:16 \
  --reference-L 4 --reference-M 256 --reference-D 256 \
  "${common[@]}" --output "$output_root/pure-L.json"

echo "$(date -u +%FT%TZ) constant-rho-audit-start"
PYTHONPATH=src "$python_bin" "$harness" \
  --shapes R1:1:128:64:64 R2:2:128:128:128 R3:4:128:256:256 R4:4:192:384:384 R5:6:256:768:768 \
  --reference-L 4 --reference-M 128 --reference-D 256 \
  "${common[@]}" --output "$output_root/constant-rho.json"

echo "$(date -u +%FT%TZ) audit-queue-complete"
