#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT CORPUS_ROOT" >&2
  exit 2
fi

output_root="$1"
corpus_root="$2"
python_bin="${A100_PYTHON_BIN:-python3}"
runtime_config="$output_root/fineweb-jiang-chizat-horizon-config.json"

mkdir -p "$output_root" "$output_root/trials"
corpus_id="$(PYTHONPATH=src "$python_bin" -c 'import json; from ai_theorist.autoscaler.public_corpora import PublicCorpusSpec; print(PublicCorpusSpec.from_dict(json.load(open("configs/autoscaler/fineweb_edu_a100.json"))).fingerprint)')"
train_path="$corpus_root/$corpus_id/train.jsonl"
validation_path="$corpus_root/$corpus_id/validation.jsonl"

"$python_bin" -c 'import json, pathlib, sys; source, target, train, validation, cache = sys.argv[1:]; payload = json.load(open(source)); payload["dataset"]["train_path"] = str(pathlib.Path(train).resolve()); payload["dataset"]["validation_path"] = str(pathlib.Path(validation).resolve()); payload["cache_directory"] = str(pathlib.Path(cache).resolve()); pathlib.Path(target).write_text(json.dumps(payload, indent=2) + "\n")' \
  configs/autoscaler/fineweb_jiang_chizat_horizon_a100.json \
  "$runtime_config" \
  "$train_path" \
  "$validation_path" \
  "$output_root/trials"

echo "$(date -u +%FT%TZ) fineweb-jiang-chizat-horizon-start corpus_id=$corpus_id"
PYTHONPATH=src "$python_bin" -m ai_theorist.autoscaler.cli horizon-transfer \
  "$runtime_config" \
  --device cuda \
  --output "$output_root/result.json" \
  --progress-jsonl \
  > "$output_root/horizon-transfer.log" 2>&1
echo "$(date -u +%FT%TZ) fineweb-jiang-chizat-horizon-complete"
