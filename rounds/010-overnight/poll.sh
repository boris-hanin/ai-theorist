#!/usr/bin/env bash
set -u

: "${AI_THEORIST_SSH_KEY:?set AI_THEORIST_SSH_KEY to the private-key path}"
: "${AI_THEORIST_REMOTE:?set AI_THEORIST_REMOTE, for example ubuntu@example.org}"

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
results_dir=${AI_THEORIST_RESULTS_DIR:-$script_dir}
remote_dir=${AI_THEORIST_REMOTE_DIR:-/home/ubuntu}
poll_count=${AI_THEORIST_POLL_COUNT:-84}
poll_interval=${AI_THEORIST_POLL_INTERVAL:-600}
files=${AI_THEORIST_RESULT_FILES:-"big_out.json big.log big_out_v1.json big_v1.log"}

mkdir -p "$results_dir"
i=1
while [ "$i" -le "$poll_count" ]; do
  ok=1
  for file in $files; do
    scp -q -i "$AI_THEORIST_SSH_KEY" -o ConnectTimeout=20 \
      -o StrictHostKeyChecking=accept-new \
      "$AI_THEORIST_REMOTE:$remote_dir/$file" "$results_dir/$file" 2>/dev/null || ok=0
  done
  if [ "$ok" -eq 1 ]; then
    lines=$(wc -l < "$results_dir/big.log" 2>/dev/null || true)
    echo "$(date +%H:%M) pulled ok ($lines log lines)" >> "$results_dir/poll.log"
  else
    echo "$(date +%H:%M) partial/failed pull" >> "$results_dir/poll.log"
  fi
  i=$((i + 1))
  [ "$i" -le "$poll_count" ] && sleep "$poll_interval"
done
