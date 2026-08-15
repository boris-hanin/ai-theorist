#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 KEY FIRST_HOST FIRST_ANCHOR_PID SECOND_HOST SECOND_ANCHOR_PID REMOTE_REPO REMOTE_RUN_ROOT EXPECTED_COMMIT LOCAL_ROOT" >&2
  exit 2
fi

key="$1"
first_host="$2"
first_anchor_pid="$3"
second_host="$4"
second_anchor_pid="$5"
remote_repo="$6"
remote_run_root="$7"
expected_commit="$8"
local_root="$9"
remote_anchor_result="$remote_repo/$remote_run_root/anchor/result.json"
ssh_options=(-i "$key" -o BatchMode=yes -o ConnectTimeout=15)

mkdir -p "$local_root"

remote_file_exists() {
  local host="$1"
  local path="$2"
  ssh "${ssh_options[@]}" "ubuntu@$host" test -f "$path" 2>/dev/null
}

remote_pid_alive() {
  local host="$1"
  local pid="$2"
  ssh "${ssh_options[@]}" "ubuntu@$host" kill -0 "$pid" 2>/dev/null
}

wait_for_anchor() {
  local host="$1"
  local pid="$2"
  while ! remote_file_exists "$host" "$remote_anchor_result"; do
    if ! remote_pid_alive "$host" "$pid"; then
      echo "anchor runner $pid stopped on $host without a result" >&2
      ssh "${ssh_options[@]}" "ubuntu@$host" \
        tail -100 "$remote_repo/$remote_run_root/qualification-queue.log" >&2 || true
      return 1
    fi
    echo "$(date -u +%FT%TZ) waiting-for-anchor host=$host pid=$pid"
    sleep 40
  done
}

wait_for_anchor "$first_host" "$first_anchor_pid" &
first_wait_pid=$!
wait_for_anchor "$second_host" "$second_anchor_pid" &
second_wait_pid=$!
wait "$first_wait_pid"
wait "$second_wait_pid"

for item in \
  "first:$first_host" \
  "second:$second_host"; do
  label="${item%%:*}"
  host="${item#*:}"
  scp "${ssh_options[@]}" \
    "ubuntu@$host:$remote_anchor_result" \
    "$local_root/$label-anchor-result.json"
  scp "${ssh_options[@]}" \
    "ubuntu@$host:$remote_repo/$remote_run_root/inventory.txt" \
    "$local_root/$label-inventory.txt"
  scp "${ssh_options[@]}" \
    "ubuntu@$host:$remote_repo/$remote_run_root/anchor.log" \
    "$local_root/$label-anchor.log"
done

python3 scripts/compare_a100_anchors.py \
  "$local_root/first-anchor-result.json" \
  "$local_root/second-anchor-result.json" \
  > "$local_root/anchor-gate.json"
echo "$(date -u +%FT%TZ) anchor-gate-accepted"

launch_phase1() {
  local host="$1"
  shift
  ssh "${ssh_options[@]}" "ubuntu@$host" \
    "cd '$remote_repo' && A100_PYTHON_BIN=/home/ubuntu/ai-theorist-round016/.venv-round016/bin/python3 nohup bash scripts/run_a100_phase1_queue.sh '$remote_run_root' '$expected_commit' $* > '$remote_run_root/phase1-queue.log' 2>&1 < /dev/null & echo \$!"
}

first_phase1_pid="$(launch_phase1 "$first_host" \
  configs/autoscaler/a100_phase1_mlp_adam.json \
  configs/autoscaler/a100_phase1_nugpt_adam.json)"
second_phase1_pid="$(launch_phase1 "$second_host" \
  configs/autoscaler/a100_phase1_mlp_sgd.json \
  configs/autoscaler/a100_phase1_moe_adam.json)"
{
  echo "first_host=$first_host first_phase1_pid=$first_phase1_pid"
  echo "second_host=$second_host second_phase1_pid=$second_phase1_pid"
} > "$local_root/phase1-pids.txt"
echo "$(date -u +%FT%TZ) phase1-launched first_pid=$first_phase1_pid second_pid=$second_phase1_pid"

wait_for_phase1() {
  local host="$1"
  local pid="$2"
  while remote_pid_alive "$host" "$pid"; do
    echo "$(date -u +%FT%TZ) waiting-for-phase1 host=$host pid=$pid"
    sleep 40
  done
}

wait_for_phase1 "$first_host" "$first_phase1_pid" &
first_wait_pid=$!
wait_for_phase1 "$second_host" "$second_phase1_pid" &
second_wait_pid=$!
wait "$first_wait_pid"
wait "$second_wait_pid"

for item in \
  "first:$first_host:a100_phase1_mlp_adam" \
  "first:$first_host:a100_phase1_nugpt_adam" \
  "second:$second_host:a100_phase1_mlp_sgd" \
  "second:$second_host:a100_phase1_moe_adam"; do
  label="${item%%:*}"
  remainder="${item#*:}"
  host="${remainder%%:*}"
  campaign="${remainder#*:}"
  result="$remote_repo/$remote_run_root/$campaign/result.json"
  if ! remote_file_exists "$host" "$result"; then
    echo "missing phase-1 result: $host $campaign" >&2
    exit 1
  fi
  scp "${ssh_options[@]}" "ubuntu@$host:$result" \
    "$local_root/$label-$campaign-result.json"
  scp "${ssh_options[@]}" \
    "ubuntu@$host:$remote_repo/$remote_run_root/$campaign.log" \
    "$local_root/$label-$campaign.log"
done
echo "$(date -u +%FT%TZ) phase1-results-collected"
