#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 KEY HOST0 HOST1 HOST2 REMOTE_REPO REMOTE_ROOT LOCAL_ROOT" >&2
  exit 2
fi

key="$1"
hosts=("$2" "$3" "$4")
remote_repo="$5"
remote_root="$6"
local_root="$7"
config="configs/autoscaler/a100_mlp_adam_hard_transfer.json"
runner="scripts/run_mlp_adam_transfer_campaign.py"
remote_python="/home/ubuntu/.venvs/ai-theorist-torch260-cu124/bin/python3"
ssh_options=(-i "$key" -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)

mkdir -p "$local_root"

remote_file_exists() {
  local host="$1"
  local path="$2"
  ssh "${ssh_options[@]}" "ubuntu@$host" test -f "$path" 2>/dev/null
}

wait_for_phase() {
  local phase="$1"
  for index in 0 1 2; do
    local host="${hosts[$index]}"
    local shard="$phase-shard-$index"
    local completion="$remote_repo/$remote_root/$shard/complete.json"
    local previous_count=-1
    local stagnant_checks=0
    while ! remote_file_exists "$host" "$completion"; do
      local count
      if ! count="$(ssh "${ssh_options[@]}" "ubuntu@$host" \
        "find '$remote_repo/$remote_root/$shard/trials' -type f 2>/dev/null | wc -l")"; then
        echo "$(date -u +%FT%TZ) transient-ssh-failure phase=$phase host=$host shard=$index"
        sleep 40
        continue
      fi
      if [[ "$count" == "$previous_count" ]]; then
        stagnant_checks=$((stagnant_checks + 1))
      else
        stagnant_checks=0
        previous_count="$count"
      fi
      if (( stagnant_checks >= 15 )); then
        echo "$phase shard $index made no progress for ten minutes on $host" >&2
        ssh "${ssh_options[@]}" "ubuntu@$host" \
          tail -100 "$remote_repo/$remote_root/$shard.log" >&2 || true
        return 1
      fi
      echo "$(date -u +%FT%TZ) waiting phase=$phase host=$host shard=$index completed=$count"
      sleep 40
    done
  done
}

collect_phase() {
  local phase="$1"
  for index in 0 1 2; do
    local host="${hosts[$index]}"
    local shard="$phase-shard-$index"
    local destination="$local_root/$shard"
    mkdir -p "$destination"
    ssh "${ssh_options[@]}" "ubuntu@$host" \
      "tar -C '$remote_repo/$remote_root/$shard' -cf - trials" \
      | tar -C "$destination" -xf -
    scp "${ssh_options[@]}" \
      "ubuntu@$host:$remote_repo/$remote_root/$shard/complete.json" \
      "$destination/complete.json"
    scp "${ssh_options[@]}" \
      "ubuntu@$host:$remote_repo/$remote_root/$shard.log" \
      "$destination.log"
  done
}

analyze_phase() {
  local phase="$1"
  local output="$local_root/$phase-analysis.json"
  local arguments=(
    --trials "$local_root/$phase-shard-0"
    --trials "$local_root/$phase-shard-1"
    --trials "$local_root/$phase-shard-2"
    --output "$output"
  )
  if [[ "$phase" == "lr" ]]; then
    PYTHONPATH=src python3 "$runner" analyze-lr "$config" "${arguments[@]}" \
      --trials "$local_root/lr-extension-shard-0" \
      --trials "$local_root/lr-extension-shard-1" \
      --trials "$local_root/lr-extension-shard-2" \
      > "$local_root/lr-analysis.log"
  else
    PYTHONPATH=src python3 "$runner" analyze-followup "$config" \
      --phase "$phase" "${arguments[@]}" > "$local_root/$phase-analysis.log"
  fi
}

launch_phase() {
  local phase="$1"
  local analysis="$local_root/lr-analysis.json"
  for index in 0 1 2; do
    local host="${hosts[$index]}"
    local shard="$phase-shard-$index"
    scp "${ssh_options[@]}" "$analysis" \
      "ubuntu@$host:$remote_repo/$remote_root/lr-analysis.json"
    ssh -n -f "${ssh_options[@]}" "ubuntu@$host" \
      "cd '$remote_repo' && mkdir -p '$remote_root/$shard' && nohup env PYTHONPATH=src '$remote_python' '$runner' run '$config' --phase '$phase' --analysis '$remote_root/lr-analysis.json' --output '$remote_root/$shard' --device cuda --shard-index '$index' --shard-count 3 > '$remote_root/$shard.log' 2>&1 < /dev/null"
    echo "$(date -u +%FT%TZ) launched phase=$phase host=$host shard=$index"
  done
}

wait_for_phase lr
collect_phase lr
launch_phase lr-extension
wait_for_phase lr-extension
collect_phase lr-extension
analyze_phase lr

gate_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"]["status"])' "$local_root/lr-analysis.json")"
followups_allowed="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["followups_allowed"]).lower())' "$local_root/lr-analysis.json")"
echo "$(date -u +%FT%TZ) lr-gate status=$gate_status followups_allowed=$followups_allowed"
if [[ "$followups_allowed" != "true" ]]; then
  echo "LR transfer remains ambiguous; batch and horizon phases were correctly withheld."
  exit 0
fi

launch_phase batch
wait_for_phase batch
collect_phase batch
analyze_phase batch

launch_phase horizon
wait_for_phase horizon
collect_phase horizon
analyze_phase horizon
echo "$(date -u +%FT%TZ) hard-mlp-adam-campaign-complete"
