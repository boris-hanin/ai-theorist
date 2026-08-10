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

# Width 64 is the sole tuning/base shape.  The previous primary grid put the
# common Adam optimum at its upper edge, so only the normalized eta bracket is
# expanded; depth and every other training variable remain fixed.
echo "$(date -u +%FT%TZ) mup-adam-expanded-smallest-reference-start"
PYTHONPATH=src "$python_bin" skills/dmft-resnet-depth/scripts/mup_mlp_transfer.py \
  --optimizer adam \
  --widths 64 128 256 512 1024 \
  --reference-width 64 \
  --depth 4 \
  --etas 0.001 0.003 0.01 0.03 0.1 0.3 1.0 \
  --seeds 11 29 47 \
  --steps 150 \
  --batch-size 128 \
  --input-dimension 32 \
  --n-train 8192 \
  --n-validation 2048 \
  --oracle-tolerance 1.10 \
  --device cuda \
  --output "$output_root/mup-adam-expanded-smallest-reference.json" \
  > "$output_root/mup-adam-expanded-smallest-reference.log" 2>&1
echo "$(date -u +%FT%TZ) mup-adam-expanded-smallest-reference-complete"

echo "$(date -u +%FT%TZ) correct-mup-adam-extension-complete"
