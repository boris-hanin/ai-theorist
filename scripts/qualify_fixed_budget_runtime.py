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
from ai_theorist.autoscaler.jiang_moe import JiangMoETransformer
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
    "jiang_moe_transformer": {
        "jiang_moe_embeddings",
        "jiang_moe_norms",
        "jiang_moe_attention_qkv",
        "jiang_moe_attention_output",
        "jiang_moe_router",
        "jiang_moe_expert_up",
        "jiang_moe_expert_down",
        "jiang_moe_other_biases",
    },
}


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"config must be an object: {path}")
    return value


def _qualify_config(
    path: Path,
    device: str,
    single_process_ddp_equivalent: bool = False,
) -> dict[str, Any]:
    config = _load(path)
    plan = compile_real_text_scaling_plan(config)
    runtime_payload = dict(config.get("runtime", {}))
    qualification_override = None
    if single_process_ddp_equivalent:
        distributed = str(runtime_payload.get("distributed", "none"))
        processes = int(runtime_payload.get("num_processes", 1))
        accumulation = int(runtime_payload.get("gradient_accumulation_steps", 1))
        if distributed != "ddp" or processes < 2:
            raise ValueError(
                "--single-process-ddp-equivalent requires a multi-process DDP config"
            )
        runtime_payload.update(
            {
                "distributed": "none",
                "num_processes": 1,
                "gradient_accumulation_steps": processes * accumulation,
            }
        )
        qualification_override = {
            "production_distributed": distributed,
            "production_num_processes": processes,
            "production_gradient_accumulation_steps": accumulation,
            "qualification_distributed": "none",
            "qualification_num_processes": 1,
            "qualification_gradient_accumulation_steps": processes
            * accumulation,
            "microbatch_examples_preserved": True,
        }
    runtime = PretrainingRuntimeSpec.from_dict(runtime_payload)
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
            expected_backend = f"fused_{config['optimizer']['name']}"
            if audit.get("optimizer_backend") != expected_backend:
                raise RuntimeError(
                    f"{scale['name']} did not construct {expected_backend}"
                )
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
                    "initialization_contract": audit.get(
                        "initialization_contract"
                    ),
                    "manual_expert_bias": audit.get("manual_expert_bias"),
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
                if isinstance(plain_model, JiangMoETransformer):
                    plain_model.begin_routing_measurement()
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
                if isinstance(plain_model, JiangMoETransformer):
                    plain_model.update_expert_biases(
                        float(config["optimizer"]["expert_bias_learning_rate"])
                    )
                torch.cuda.synchronize(context.device)
                rows[-1]["endpoint_canary_loss"] = float(loss.detach().cpu())
                rows[-1]["endpoint_microbatch_examples"] = microbatch
                rows[-1]["endpoint_peak_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated(context.device)
                )
                if isinstance(plain_model, JiangMoETransformer):
                    rows[-1]["routing_diagnostics"] = (
                        plain_model.routing_diagnostics()
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
        "single_process_ddp_equivalent": qualification_override,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every fixed-budget model scale and run endpoint fused-Adam/AdamW "
            "FlashAttention canaries."
        )
    )
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--single-process-ddp-equivalent",
        action="store_true",
        help=(
            "Audit a DDP plan on one rank while multiplying gradient accumulation "
            "by the production world size, preserving the per-rank microbatch."
        ),
    )
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
            _qualify_config(
                path,
                args.device,
                single_process_ddp_equivalent=args.single_process_ddp_equivalent,
            )
            for path in args.configs
        ],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
