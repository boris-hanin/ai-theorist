#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root="$1"
python_bin="${A100_PYTHON_BIN:-python3}"
corpus_root="$output_root/corpus"

mkdir -p "$output_root" "$corpus_root"

echo "$(date -u +%FT%TZ) fineweb-materialization-start"
PYTHONPATH=src "$python_bin" -m ai_theorist.autoscaler.cli corpus-materialize \
  configs/autoscaler/fineweb_edu_a100.json \
  --output-root "$corpus_root" \
  --progress-jsonl \
  > "$output_root/corpus-materialization.log" 2>&1

corpus_id="$(PYTHONPATH=src "$python_bin" -c 'import json; from ai_theorist.autoscaler.public_corpora import PublicCorpusSpec; print(PublicCorpusSpec.from_dict(json.load(open("configs/autoscaler/fineweb_edu_a100.json"))).fingerprint)')"
train_path="$corpus_root/$corpus_id/train.jsonl"
validation_path="$corpus_root/$corpus_id/validation.jsonl"
cp "$corpus_root/$corpus_id/manifest.json" "$output_root/corpus-manifest.json"
echo "$(date -u +%FT%TZ) fineweb-materialization-complete corpus_id=$corpus_id"

echo "$(date -u +%FT%TZ) dense-jiang-chizat-real-text-start"
PYTHONPATH=src:skills/dmft-attention/scripts "$python_bin" \
  skills/dmft-attention/scripts/jiang_chizat_tuned_transfer.py \
  --shapes \
    S1:2:128:64:1 \
    S2:2:256:128:2 \
    S3:4:256:256:4 \
    S4:4:512:512:8 \
  --reference-shape S1 \
  --head-dimension 64 \
  --vocab-size 260 \
  --context-length 64 \
  --n-train 8192 \
  --n-validation 512 \
  --train-path "$train_path" \
  --validation-path "$validation_path" \
  --tokenizer byte_v1 \
  --maximum-dataset-bytes 1073741824 \
  --dataset-seed 1729 \
  --etas 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 \
  --multiplier-probes 0.5 1 2 \
  --minimum-relative-multiplier-improvement 0.005 \
  --oracle-tolerance 1.10 \
  --epsilon0 1e-12 \
  --steps 300 \
  --batch-size 16 \
  --seeds 11 29 47 \
  --controls fan_in_down omit_attention_width omit_ffn_hidden_width disable_attention \
  --device cuda \
  --output "$output_root/jiang-dense-fineweb.json" \
  > "$output_root/jiang-dense-fineweb.log" 2>&1
echo "$(date -u +%FT%TZ) dense-jiang-chizat-real-text-complete"

echo "$(date -u +%FT%TZ) sparse-jiang-moe-real-text-start"
PYTHONPATH=src "$python_bin" skills/dmft-moe/scripts/jiang_moe_transfer.py \
  --shapes \
    S1:2:128:64:4:1:1 \
    S2:2:256:128:4:1:2 \
    S3:4:256:256:8:2:4 \
    S4:4:512:512:8:2:8 \
  --reference-shape S1 \
  --head-dimension 64 \
  --vocab-size 260 \
  --context-length 64 \
  --n-train 8192 \
  --n-validation 512 \
  --train-path "$train_path" \
  --validation-path "$validation_path" \
  --tokenizer byte_v1 \
  --maximum-dataset-bytes 1073741824 \
  --dataset-seed 1729 \
  --etas 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 \
  --multiplier-probes 0.5 1 2 \
  --minimum-relative-multiplier-improvement 0.005 \
  --oracle-tolerance 1.10 \
  --epsilon0 1e-12 \
  --expert-bias-learning-rate 0.01 \
  --steps 300 \
  --batch-size 16 \
  --seeds 11 29 47 \
  --controls global_lr_control omit_router_width omit_expert_down_ratio \
  --device cuda \
  --output "$output_root/jiang-moe-fineweb.json" \
  > "$output_root/jiang-moe-fineweb.log" 2>&1
echo "$(date -u +%FT%TZ) sparse-jiang-moe-real-text-complete"

echo "$(date -u +%FT%TZ) fineweb-jiang-transfer-complete"
