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

# The first source-faithful Chizat sweep placed eta_v on the upper boundary.
# Expand both normalized coordinates; do not change the L*M*D raw-rate rule.
echo "$(date -u +%FT%TZ) chizat-equation22-expanded-start"
PYTHONPATH=src "$python_bin" skills/dmft-resnet-depth/scripts/chizat_equation22_transfer.py \
  --shapes S1:1:64:16,S2:1:128:32,S3:2:128:64,S4:2:256:128,S5:4:256:256,S6:4:512:512 \
  --reference-label S3 \
  --eta-us 0.01,0.03,0.1,0.3,1.0,3.0,10.0 \
  --eta-vs 0.03,0.1,0.3,1.0,3.0,10.0,30.0 \
  --seeds 11,29,47 \
  --steps 100 \
  --n-train 64 \
  --n-validation 256 \
  --input-dimension 8 \
  --output-dimension 2 \
  --dtype float32 \
  --device cuda \
  --output "$output_root/chizat-equation22-expanded.json" \
  > "$output_root/chizat-equation22-expanded.log" 2>&1
echo "$(date -u +%FT%TZ) chizat-equation22-expanded-complete"

# Re-run only SGD: the first run completed training but strict JSON rejected a
# NaN diagnostic from its failed control.  The serialization fix converts that
# value to null without changing a single training step or acceptance gate.
echo "$(date -u +%FT%TZ) mup-sgd-report-safe-start"
PYTHONPATH=src "$python_bin" skills/dmft-resnet-depth/scripts/mup_mlp_transfer.py \
  --optimizer sgd \
  --widths 64 128 256 512 1024 \
  --reference-width 256 \
  --depth 4 \
  --etas 0.003 0.01 0.03 0.1 0.3 \
  --seeds 11 29 47 \
  --steps 150 \
  --batch-size 128 \
  --input-dimension 32 \
  --n-train 8192 \
  --n-validation 2048 \
  --device cuda \
  --output "$output_root/mup-sgd-report-safe.json" \
  > "$output_root/mup-sgd-report-safe.log" 2>&1
echo "$(date -u +%FT%TZ) mup-sgd-report-safe-complete"

echo "$(date -u +%FT%TZ) correct-transfer-a-followup-complete"
