#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ai_theorist.autoscaler.forecast_campaigns import (
    _autocast,
    _build_model_and_groups,
    compile_real_text_scaling_plan,
)
from ai_theorist.autoscaler.pretraining import (
    PretrainingRuntimeSpec,
    close_distributed,
    prepare_distributed,
)
from ai_theorist.autoscaler.study import atomic_write_json


EXPECTED_GROUPS = {
    "jiang_chizat_transformer": {
        "jiang_embeddings",
        "jiang_norms",
        "jiang_final_norm",
        "jiang_attention_qkv",
        "jiang_attention_output",
        "jiang_ffn_up",
        "jiang_ffn_down",
        "jiang_other_biases",
    },
    "completep_transformer": {
        "completep_embeddings",
        "completep_hidden_weights",
        "completep_hidden_biases",
        "completep_hidden_norms",
        "completep_final_norm",
        "completep_unembedding",
    },
}


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"config must be an object: {path}")
    return value


def _qualify_config(path: Path, device: str) -> dict[str, Any]:
    config = _load(path)
    plan = compile_real_text_scaling_plan(config)
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    context = prepare_distributed(runtime, device)
    rows = []
    endpoint = plan["scales"][-1]
    try:
        for scale in plan["scales"]:
            torch.manual_seed(73)
            model, plain_model, optimizer, contract, audit = _build_model_and_groups(
                config=config,
                scale=scale,
                eta=float(plan["learning_rates"][len(plan["learning_rates"]) // 2]),
                weight_decay_tau_ema=None,
                optimizer_mode="theory",
                runtime=runtime,
                context=context,
            )
            group_names = {str(group["name"]) for group in contract}
            block_type = str(scale["block_type"])
            if group_names != EXPECTED_GROUPS[block_type]:
                raise RuntimeError(
                    f"{scale['name']} optimizer groups are incomplete: {group_names}"
                )
            if audit.get("complete") is not True or audit.get("disjoint") is not True:
                raise RuntimeError(f"{scale['name']} optimizer group audit failed")
            if audit.get("optimizer_backend") != "fused_adamw":
                raise RuntimeError(f"{scale['name']} did not construct fused AdamW")
            if any(
                not math.isfinite(float(group["epsilon"]))
                or float(group["epsilon"]) <= 0.0
                for group in contract
            ):
                raise RuntimeError(f"{scale['name']} has an invalid Adam epsilon")
            rows.append(
                {
                    "scale": scale["name"],
                    "parameters": scale["parameters"],
                    "trainable_parameter_tensors": audit[
                        "trainable_parameter_tensors"
                    ],
                    "optimizer_groups": sorted(group_names),
                    "optimizer_backend": audit["optimizer_backend"],
                }
            )
            if scale["name"] == endpoint["name"]:
                torch.cuda.reset_peak_memory_stats(context.device)
                microbatch = int(config["batch_examples"]) // (
                    context.world_size * runtime.gradient_accumulation_steps
                )
                torch.manual_seed(91)
                inputs = torch.randint(
                    int(config["architecture"]["vocab_size"]),
                    (
                        microbatch,
                        int(config["architecture"]["context_length"]),
                    ),
                    device=context.device,
                )
                optimizer.zero_grad(set_to_none=True)
                with _autocast(runtime, context.device):
                    logits = model(inputs)
                    loss = F.cross_entropy(
                        logits.float().reshape(
                            -1, int(config["architecture"]["vocab_size"])
                        ),
                        inputs.reshape(-1),
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("endpoint FlashAttention canary loss is non-finite")
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize(context.device)
                rows[-1]["endpoint_canary_loss"] = float(loss.detach().cpu())
                rows[-1]["endpoint_microbatch_examples"] = microbatch
                rows[-1]["endpoint_peak_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated(context.device)
                )
            del optimizer, model, plain_model
            torch.cuda.empty_cache()
    finally:
        close_distributed(context)
    return {
        "config": str(path),
        "plan_fingerprint": plan["fingerprint"],
        "architecture": plan["architecture_contract"]["parameterization"],
        "scales": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every fixed-budget model scale and run endpoint fused-AdamW "
            "FlashAttention canaries."
        )
    )
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production runtime qualification")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "adam_epsilon_placement": "torch AdamW: sqrt(v_hat) + epsilon",
        "campaigns": [
            _qualify_config(path, args.device) for path in args.configs
        ],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
