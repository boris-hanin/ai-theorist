#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT EXPECTED_COMMIT" >&2
  exit 2
fi

output_root="$1"
expected_commit="$2"
python_bin="${A100_PYTHON_BIN:-python3}"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "commit mismatch: expected $expected_commit, got $actual_commit" >&2
  exit 1
fi
mkdir -p "$output_root"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$python_bin" -m pytest -q \
  > "$output_root/tests.log" 2>&1

echo "$(date -u +%FT%TZ) chizat-equation22-start"
PYTHONPATH=src "$python_bin" skills/dmft-resnet-depth/scripts/chizat_equation22_transfer.py \
  --shapes S1:1:64:16,S2:1:128:32,S3:2:128:64,S4:2:256:128,S5:4:256:256,S6:4:512:512 \
  --reference-label S3 \
  --eta-us 0.01,0.03,0.1,0.3,1.0 \
  --eta-vs 0.01,0.03,0.1,0.3,1.0 \
  --seeds 11,29,47 \
  --steps 100 \
  --n-train 64 \
  --n-validation 256 \
  --input-dimension 8 \
  --output-dimension 2 \
  --dtype float32 \
  --device cuda \
  --output "$output_root/chizat-equation22.json" \
  > "$output_root/chizat-equation22.log" 2>&1
echo "$(date -u +%FT%TZ) chizat-equation22-complete"

for optimizer in adam sgd; do
  if [[ "$optimizer" == "adam" ]]; then
    etas=(0.0001 0.0003 0.001 0.003 0.01)
  else
    etas=(0.003 0.01 0.03 0.1 0.3)
  fi
  echo "$(date -u +%FT%TZ) mup-$optimizer-start"
  PYTHONPATH=src "$python_bin" skills/dmft-resnet-depth/scripts/mup_mlp_transfer.py \
    --optimizer "$optimizer" \
    --widths 64 128 256 512 1024 \
    --reference-width 256 \
    --depth 4 \
    --etas "${etas[@]}" \
    --seeds 11 29 47 \
    --steps 150 \
    --batch-size 128 \
    --input-dimension 32 \
    --n-train 8192 \
    --n-validation 2048 \
    --device cuda \
    --output "$output_root/mup-$optimizer.json" \
    > "$output_root/mup-$optimizer.log" 2>&1
  echo "$(date -u +%FT%TZ) mup-$optimizer-complete"
done

echo "$(date -u +%FT%TZ) correct-transfer-a-complete"
