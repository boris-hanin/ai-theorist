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

echo "$(date -u +%FT%TZ) jiang-moe-start"
PYTHONPATH=src "$python_bin" skills/dmft-moe/scripts/jiang_moe_transfer.py \
  --shapes S1:2:128:64:4:1:1 S2:2:256:128:4:1:2 S3:4:256:256:8:2:4 S4:4:512:512:8:2:8 \
  --reference-shape S3 \
  --head-dimension 64 \
  --vocab-size 128 \
  --context-length 32 \
  --n-train 4096 \
  --n-validation 512 \
  --etas 0.0001 0.0003 0.001 0.003 0.01 \
  --steps 200 \
  --batch-size 16 \
  --seeds 11 29 47 \
  --device cuda \
  --output "$output_root/jiang-moe.json" \
  > "$output_root/jiang-moe.log" 2>&1
echo "$(date -u +%FT%TZ) jiang-moe-complete"

echo "$(date -u +%FT%TZ) jiang-dense-start"
PYTHONPATH=src:skills/dmft-attention/scripts "$python_bin" skills/dmft-attention/scripts/jiang_chizat_tuned_transfer.py \
  --shapes S1:2:128:64:1 S2:2:256:128:2 S3:4:256:256:4 S4:4:512:512:8 \
  --reference-shape S3 \
  --etas 0.0001 0.0003 0.001 0.003 0.01 \
  --head-dimension 64 \
  --vocab-size 128 \
  --context-length 32 \
  --n-train 4096 \
  --n-validation 512 \
  --steps 200 \
  --batch-size 16 \
  --seeds 11 29 47 \
  --device cuda \
  --output "$output_root/jiang-dense.json" \
  > "$output_root/jiang-dense.log" 2>&1
echo "$(date -u +%FT%TZ) jiang-dense-complete"

for config in \
  configs/autoscaler/a100_nugpt_width_adam.json \
  configs/autoscaler/a100_nugpt_depth_adam.json \
  configs/autoscaler/a100_nugpt_joint_adam.json; do
  campaign="$(basename "$config" .json)"
  echo "$(date -u +%FT%TZ) $campaign-start"
  PYTHONPATH=src "$python_bin" -m ai_theorist.autoscaler.cli run \
    "$config" \
    --device cuda \
    --output "$output_root/$campaign" \
    --summary \
    --progress-jsonl \
    > "$output_root/$campaign.log" 2>&1
  echo "$(date -u +%FT%TZ) $campaign-complete"
done

echo "$(date -u +%FT%TZ) correct-transfer-b-complete"
