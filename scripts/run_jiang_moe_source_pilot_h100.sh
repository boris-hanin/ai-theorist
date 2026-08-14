#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT TOKEN_STREAM_MANIFEST" >&2
  exit 2
fi

repo="${AI_THEORIST_REPO:-/home/ubuntu/ai-theorist}"
python="${AI_THEORIST_PYTHON:-$repo/.venv-forecast-py310/bin/python}"
required_commit="${AI_THEORIST_MOE_COMMIT:?set the pinned MoE implementation commit}"
root="$1"
manifest="$2"
cli="$repo/scripts/ai_theorist_autoscale_python.sh"
stage=starting

mkdir -p "$root"
exec 9>"$root/controller.lock"
flock -n 9 || { echo "another MoE controller holds $root/controller.lock" >&2; exit 1; }
echo "$$" > "$root/controller.pid"
trap 'code=$?; if (( code != 0 )); then echo "failed:$stage" > "$root/stage"; fi' EXIT

[[ -x "$python" && -x "$cli" && -f "$manifest" ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$required_commit" ]]
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')" == 8 ]]
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq '^NVIDIA H100 80GB HBM3$'; then
  echo "MoE source pilot requires eight H100 80GB GPUs" >&2
  exit 1
fi
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits | grep -Evq '^0$'; then
  echo "volatile uncorrected ECC error detected" >&2
  exit 1
fi
if systemctl list-unit-files nvidia-fabricmanager.service >/dev/null 2>&1; then
  [[ "$(systemctl is-active nvidia-fabricmanager)" == active ]]
fi

export AI_THEORIST_PYTHON="$python"
export AI_THEORIST_AUTOSCALE="$cli"
export PYTHONPATH="$repo/src"

stage=verifying-immutable-token-stream
echo "$stage" > "$root/stage"
"$python" - "$manifest" "$root/dataset-verification.json" <<'PY'
import json, sys
from pathlib import Path
from ai_theorist.autoscaler.tokenization import load_token_stream_manifest
p = load_token_stream_manifest(Path(sys.argv[1]), verify_files=True)
out = {
    "schema_version": 1,
    "status": "passed",
    "fingerprint": p["fingerprint"],
    "content_fingerprint": p["content_fingerprint"],
    "tokenizer_id": p["tokenizer_id"],
    "tokenizer_fingerprint": p["tokenizer_fingerprint"],
    "training_tokens": p["splits"]["train"]["tokens"],
    "validation_tokens": p["splits"]["validation"]["tokens"],
}
assert out["tokenizer_id"] == "gpt2_openai"
json.dump(out, open(sys.argv[2], "w"), indent=2, sort_keys=True)
print(json.dumps(out, sort_keys=True))
PY

stage=preregistering-source-faithful-pilot
echo "$stage" > "$root/stage"
if [[ ! -f "$root/preregistration.json" ]]; then
  cd "$repo"
  "$python" scripts/prepare_jiang_moe_source_pilot.py "$manifest" \
    --output-root "$root" > "$root/preregistration.stdout.json"
fi
"$python" - "$root/preregistration.json" "$root/plan.json" \
  "$root/dataset-verification.json" <<'PY'
import json, sys
pre, plan, data = (json.load(open(path)) for path in sys.argv[1:])
assert pre["status"] == "preregistered"
assert all(pre["gates"].values())
assert pre["plan_fingerprint"] == plan["fingerprint"]
assert pre["dataset_fingerprint"] == data["fingerprint"]
assert pre["tokenizer_fingerprint"] == data["tokenizer_fingerprint"]
PY

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 10
done

stage=qualifying-sparse-runtime-init-and-optimizer-groups
echo "$stage" > "$root/stage"
if [[ ! -f "$root/runtime-qualification.json" ]]; then
  cd "$repo"
  CUDA_VISIBLE_DEVICES=0 "$python" scripts/qualify_fixed_budget_runtime.py \
    "$root/config.json" --output "$root/runtime-qualification.json" \
    > "$root/runtime-qualification.stdout.json"
fi
"$python" - "$root/runtime-qualification.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "passed"
campaign = p["campaigns"][0]
assert campaign["architecture"] == "jiang_moe_completep_adam_table2"
assert all(len(row["optimizer_groups"]) == 8 for row in campaign["scales"])
endpoint = campaign["scales"][-1]
assert endpoint["endpoint_canary_loss"] > 0
assert endpoint["routing_diagnostics"]["maximum_absolute_expert_bias"] > 0
assert endpoint["initialization_contract"]["router_gamma"] == 1.0
PY

stage=running-eight-gpu-reference-and-independent-dimension-ladder
echo "$stage" > "$root/stage"
cd "$repo"
scripts/run_forecast_8gpu_fleet.sh "$root/config.json" "$root/fleet" \
  > "$root/fleet-controller.log" 2>&1

stage=verifying-exploratory-result
echo "$stage" > "$root/stage"
"$python" - "$root/fleet/reference-selection.json" \
  "$root/fleet/aggregate/result.json" "$root/result.json" <<'PY'
import json, shutil, sys
selection = json.load(open(sys.argv[1]))
result = json.load(open(sys.argv[2]))
assert selection["learning_rate_optimum_is_interior"] is True
assert result["status"] in {"completed", "passed"}
shutil.copy2(sys.argv[2], sys.argv[3])
PY

trap - EXIT
echo complete > "$root/stage"
echo "completed exploratory Jiang MoE source pilot: $root/result.json"
