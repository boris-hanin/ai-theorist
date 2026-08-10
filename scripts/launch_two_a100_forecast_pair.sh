#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 KEY JIANG_HOST NUGPT_HOST REMOTE_REPO REMOTE_CORPUS_ROOT REMOTE_RUN_ROOT MATERIALIZE_LOG" >&2
  exit 2
fi

key="$1"
jiang_host="$2"
nugpt_host="$3"
remote_repo="$4"
remote_corpus_root="$5"
remote_run_root="$6"
materialize_log="$7"
ssh_options=(
  -i "$key"
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -o StrictHostKeyChecking=accept-new
)

remote() {
  local host="$1"
  shift
  ssh "${ssh_options[@]}" "ubuntu@$host" "$@"
}

timestamp() {
  date -u +%FT%TZ
}

manifest=""
while [[ -z "$manifest" ]]; do
  manifest="$(remote "$jiang_host" \
    "find '$remote_repo/$remote_corpus_root' -path '*/token-streams/manifest.json' -type f -print -quit 2>/dev/null" || true)"
  if [[ -z "$manifest" ]]; then
    if ! remote "$jiang_host" "pgrep -f 'corpus-materialize.*fineweb_edu_olmo2_forecast_corpus' >/dev/null"; then
      echo "$(timestamp) corpus materializer stopped before producing a manifest" >&2
      remote "$jiang_host" "tail -100 '$remote_repo/$materialize_log'" >&2 || true
      exit 1
    fi
    progress="$(remote "$jiang_host" "tail -1 '$remote_repo/$materialize_log'" || true)"
    echo "$(timestamp) waiting-for-corpus $progress"
    sleep 60
  fi
done

corpus_dir="${manifest%/token-streams/manifest.json}"
corpus_bytes="$(remote "$jiang_host" \
  "du -sb '$corpus_dir/token-streams' '$corpus_dir/tokenizer' | awk '{total += \$1} END {print total}'")"
available_bytes="$(remote "$nugpt_host" "df -B1 --output=avail '$remote_repo' | tail -1 | tr -d ' '")"
reserve_bytes=$((10 * 1024 * 1024 * 1024))
if (( corpus_bytes + reserve_bytes > available_bytes )); then
  echo "$(timestamp) insufficient-space host=$nugpt_host required=$corpus_bytes available=$available_bytes reserve=$reserve_bytes" >&2
  exit 1
fi

echo "$(timestamp) syncing-verified-corpus bytes=$corpus_bytes source=$jiang_host destination=$nugpt_host"
remote "$nugpt_host" "mkdir -p '$corpus_dir'"
remote "$jiang_host" "tar -C '$corpus_dir' -cf - token-streams tokenizer" \
  | remote "$nugpt_host" "tar -C '$corpus_dir' -xf -"

jiang_root="$remote_repo/$remote_run_root/jiang"
nugpt_root="$remote_repo/$remote_run_root/nugpt"
remote "$jiang_host" "mkdir -p '$jiang_root' && cd '$remote_repo' && .venv-forecast/bin/ai-theorist-autoscale forecast-bind configs/autoscaler/jiang_olmo2_100m_ladder.json '$manifest' --output '$jiang_root/bound-config.json' > '$jiang_root/binding.json'"
remote "$nugpt_host" "mkdir -p '$nugpt_root' && cd '$remote_repo' && .venv-forecast/bin/ai-theorist-autoscale forecast-bind configs/autoscaler/nugpt_olmo2_100m_ladder.json '$manifest' --output '$nugpt_root/bound-config.json' > '$nugpt_root/binding.json'"

jiang_dataset="$(remote "$jiang_host" "python3 -c 'import json; print(json.load(open(\"$jiang_root/binding.json\"))[\"dataset_identity\"][\"fingerprint\"])'")"
nugpt_dataset="$(remote "$nugpt_host" "python3 -c 'import json; print(json.load(open(\"$nugpt_root/binding.json\"))[\"dataset_identity\"][\"fingerprint\"])'")"
if [[ "$jiang_dataset" != "$nugpt_dataset" ]]; then
  echo "$(timestamp) copied corpus failed identity parity" >&2
  exit 1
fi

launch_campaign() {
  local host="$1"
  local root="$2"
  local label="$3"
  if remote "$host" "test -f '$root/campaign/result.json'"; then
    echo "$(timestamp) already-complete architecture=$label host=$host"
    return
  fi
  remote "$host" \
    "cd '$remote_repo' && mkdir -p '$root/campaign' && nohup flock -n '$root/campaign.lock' env CUDA_VISIBLE_DEVICES=0 .venv-forecast/bin/ai-theorist-autoscale forecast-ladder '$root/bound-config.json' --device cuda --output '$root/campaign' --progress-jsonl > '$root/campaign.log' 2>&1 < /dev/null & echo \$! > '$root/campaign.pid'"
  sleep 5
  if ! remote "$host" "test -s '$root/campaign.pid' && kill -0 \$(cat '$root/campaign.pid') 2>/dev/null"; then
    echo "$(timestamp) launch-failed architecture=$label host=$host" >&2
    remote "$host" "tail -100 '$root/campaign.log'" >&2 || true
    exit 1
  fi
  echo "$(timestamp) launched architecture=$label host=$host root=$root"
}

launch_campaign "$jiang_host" "$jiang_root" "jiang_chizat"
launch_campaign "$nugpt_host" "$nugpt_root" "nugpt"
echo "$(timestamp) pair-launched dataset_fingerprint=$jiang_dataset"
