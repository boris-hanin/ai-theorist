#!/usr/bin/env bash
set -euo pipefail

source_worktree=/home/ubuntu/ai-theorist-500m-source
control_worktree=/home/ubuntu/ai-theorist-500m-control
python=/home/ubuntu/ai-theorist/.venv-forecast-py310/bin/python
cli="$source_worktree/scripts/ai_theorist_autoscale_python.sh"
export AI_THEORIST_PYTHON="$python"
export PYTHONPATH="$source_worktree/src"
root=/home/ubuntu/ai-theorist/runs/forecast-production/jiang-mistral-500m-10tpp-h100-v1
manifest=/home/ubuntu/ai-theorist/runs/forecast-corpora/mistral-fineweb-100bt-continuation-v1/token-streams/manifest.json
tokenizer_manifest=/home/ubuntu/ai-theorist/runs/forecast-corpora/mistral-fineweb-100bt-continuation-v1/tokenizer/manifest.json
required_source_commit=f969cafb3738351ca93fa8d28f6e65abd74a83c5
required_control_commit=${AI_THEORIST_500M_CONTROL_COMMIT:?set the pinned control commit}
stage=starting

mkdir -p "$root"
exec 9>"$root/controller.lock"
flock -n 9 || { echo "another 500M controller holds the lock" >&2; exit 1; }
echo "$$" > "$root/controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$root/stage"; fi' EXIT

[[ "$(git -C "$source_worktree" rev-parse HEAD)" == "$required_source_commit" ]]
[[ "$(git -C "$control_worktree" rev-parse HEAD)" == "$required_control_commit" ]]
git -C "$source_worktree" diff --quiet
git -C "$source_worktree" diff --cached --quiet
git -C "$control_worktree" diff --quiet
git -C "$control_worktree" diff --cached --quiet
[[ -x "$python" && -x "$cli" && -f "$manifest" && -f "$tokenizer_manifest" ]]
[[ "$(systemctl is-active nvidia-fabricmanager)" == active ]]
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')" == 8 ]]
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq '^NVIDIA H100 80GB HBM3$'; then
  echo "500M campaign requires eight H100 80GB GPUs" >&2
  exit 1
fi
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "volatile uncorrected ECC error detected" >&2
  exit 1
fi

"$python" - "$root/launch-provenance.json" "$required_source_commit" \
  "$required_control_commit" <<'PY'
import json, platform, subprocess, sys, torch
output, source_commit, control_commit = sys.argv[1:]
payload = {
    "schema_version": 1,
    "status": "bound",
    "source_training_commit": source_commit,
    "control_commit": control_commit,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines(),
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
print(json.dumps(payload, sort_keys=True))
PY

stage=verifying-complete-token-stream
echo "$stage" > "$root/stage"
"$python" - "$manifest" "$tokenizer_manifest" \
  "$root/dataset-verification.json" <<'PY'
from hashlib import sha256
import json, sys
from pathlib import Path
from ai_theorist.autoscaler.tokenization import load_token_stream_manifest
p = load_token_stream_manifest(Path(sys.argv[1]), verify_files=True)
tokenizer_manifest_path = Path(sys.argv[2])
tokenizer = json.load(open(tokenizer_manifest_path, encoding="utf-8"))
for asset in tokenizer["assets"]:
    path = tokenizer_manifest_path.parent / asset["path"]
    assert path.stat().st_size == int(asset["bytes"])
    assert sha256(path.read_bytes()).hexdigest() == asset["sha256"]
assert tokenizer["fingerprint"] == p["tokenizer_fingerprint"]
out = {
    "schema_version": 1,
    "status": "passed",
    "fingerprint": p["fingerprint"],
    "content_fingerprint": p["content_fingerprint"],
    "tokenizer_fingerprint": p["tokenizer_fingerprint"],
    "tokenizer_assets_verified": True,
    "training_tokens": p["splits"]["train"]["tokens"],
    "validation_tokens": p["splits"]["validation"]["tokens"],
}
json.dump(out, open(sys.argv[3], "w"), indent=2, sort_keys=True)
print(json.dumps(out, sort_keys=True))
PY

stage=verifying-preregistration
echo "$stage" > "$root/stage"
"$python" - "$root/plan.json" "$root/preregistration.json" "$root/dataset-verification.json" <<'PY'
import json, sys
plan, pre, data = (json.load(open(path)) for path in sys.argv[1:])
assert pre["status"] == "preregistered"
assert plan["fingerprint"] == pre["plan_fingerprint"]
assert data["status"] == "passed"
assert plan["dataset_identity"]["fingerprint"] == pre["dataset_fingerprint"] == data["fingerprint"]
assert plan["dataset_identity"]["tokenizer_fingerprint"] == pre["tokenizer_fingerprint"] == data["tokenizer_fingerprint"]
assert data["tokenizer_assets_verified"] is True
assert all(pre["gates"].values())
PY

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 10
done
cd "$source_worktree"
selected_eta="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_learning_rate"])' "$root/preregistration.json")"
task_id="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_id"])' "$root/preregistration.json")"

stage=qualifying-500m-memory-and-optimizer-groups
echo "$stage" > "$root/stage"
CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_fixed_budget_runtime.py \
  "$root/config.json" --output "$root/runtime-qualification.json" \
  --single-process-ddp-equivalent > "$root/runtime-qualification.stdout.json"

stage=qualifying-single-vs-eight-gpu-topology
echo "$stage" > "$root/stage"
mkdir -p "$root/topology/single" "$root/topology/ddp"
if [[ ! -f "$root/topology/comparison.json" ]]; then
  "$python" scripts/prepare_forecast_runtime_canary.py \
    "$root/config.json" "$manifest" "$root/topology/single-config.json" \
    --steps 3 --seed 11 --learning-rate "$selected_eta" --fused true \
    --checkpoint-steps 0 --checkpoint-seconds 600 --distributed none \
    --num-processes 1 --batch-examples 512 --gradient-accumulation-steps 8 \
    > "$root/topology/single-preparation.json"
  "$python" scripts/prepare_forecast_runtime_canary.py \
    "$root/config.json" "$manifest" "$root/topology/ddp-config.json" \
    --steps 3 --seed 11 --learning-rate "$selected_eta" --fused true \
    --checkpoint-steps 0 --checkpoint-seconds 600 --distributed ddp \
    --num-processes 8 --batch-examples 512 --gradient-accumulation-steps 1 \
    > "$root/topology/ddp-preparation.json"
  "$cli" forecast-plan "$root/topology/single-config.json" > "$root/topology/single-plan.json"
  "$cli" forecast-plan "$root/topology/ddp-config.json" > "$root/topology/ddp-plan.json"
  CUDA_VISIBLE_DEVICES=0 "$cli" forecast-shard "$root/topology/single-config.json" \
    --phase ladder --shard-index 0 --shard-count 1 \
    --selected-learning-rate "$selected_eta" --task-id "$task_id" --device cuda \
    --output "$root/topology/single" --progress-jsonl > "$root/topology/single.log" 2>&1
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" -m torch.distributed.run \
    --standalone --nproc_per_node=8 -m ai_theorist.autoscaler.cli forecast-shard \
    "$root/topology/ddp-config.json" --phase ladder --shard-index 0 --shard-count 1 \
    --selected-learning-rate "$selected_eta" --task-id "$task_id" --device cuda \
    --output "$root/topology/ddp" --progress-jsonl > "$root/topology/ddp.log" 2>&1
  "$python" scripts/evaluate_forecast_shard_topology.py \
    "$root/topology/single/ladder-shard-000.json" \
    "$root/topology/ddp/ladder-shard-000.json" \
    "$root/topology/single-plan.json" --maximum-loss-delta 0.005 \
    --output "$root/topology/comparison.json" > "$root/topology/comparison.stdout.json"
fi
"$python" - "$root/topology/comparison.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "passed"
assert p["ddp_replicas"] == 8
assert p["maximum_observed_loss_delta"] <= p["maximum_absolute_loss_delta"]
PY

stage=running-adaptive-500m-10tpp-endpoint
echo "$stage" > "$root/stage"
mkdir -p "$root/trial"
if [[ ! -f "$root/trial/ladder-shard-000.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$python" -m torch.distributed.run \
    --standalone --nproc_per_node=8 -m ai_theorist.autoscaler.cli forecast-shard \
    "$root/config.json" --phase ladder --shard-index 0 --shard-count 1 \
    --selected-learning-rate "$selected_eta" --task-id "$task_id" --device cuda \
    --output "$root/trial" --progress-jsonl > "$root/worker.log" 2>&1
fi

stage=evaluating-adaptive-500m-prediction
echo "$stage" > "$root/stage"
"$python" "$control_worktree/scripts/evaluate_jiang_500m_10tpp_extension.py" \
  "$root/preregistration.json" "$root/trial/ladder-shard-000.json" \
  "$root/topology/comparison.json" --output "$root/result.json" \
  > "$root/evaluation.stdout.json"

trap - EXIT
echo complete > "$root/stage"
