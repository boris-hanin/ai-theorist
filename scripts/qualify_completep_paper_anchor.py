#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ai_theorist.autoscaler.completep import CompletePTransformer
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


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("CompleteP paper anchor config must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify the exact N=256,L=2 CompleteP paper anchor."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load(args.config)
    plan = compile_real_text_scaling_plan(config)
    architecture = plan["architecture_contract"]
    scale = plan["scales"][0]
    if (
        architecture["parameterization"] != "completep_alpha_1_adamw"
        or architecture["position_encoding"] != "alibi"
        or architecture["attention_scale"] != "QK^T/N"
        or (int(scale["depth"]), int(scale["width"])) != (2, 256)
    ):
        raise ValueError("config is not the preregistered CompleteP paper anchor")

    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    context = prepare_distributed(runtime, args.device)
    try:
        model, plain_model, optimizer, groups, audit = _build_model_and_groups(
            config=config,
            scale=scale,
            eta=0.00390625,
            weight_decay_tau_ema=None,
            optimizer_mode="theory",
            runtime=runtime,
            context=context,
        )
        if not isinstance(plain_model, CompletePTransformer):
            raise RuntimeError("paper anchor did not construct CompletePTransformer")
        if plain_model.position_embedding is not None:
            raise RuntimeError("paper anchor unexpectedly has learned positions")
        if audit.get("complete") is not True or audit.get("disjoint") is not True:
            raise RuntimeError("CompleteP parameter-group audit failed")
        if audit.get("optimizer_backend") != "fused_adamw":
            raise RuntimeError("paper anchor did not construct fused AdamW")
        if any(
            not math.isfinite(float(group["epsilon"]))
            or float(group["epsilon"]) <= 0.0
            for group in groups
        ):
            raise RuntimeError("paper anchor has an invalid per-group Adam epsilon")
        torch.manual_seed(2505)
        inputs = torch.randint(
            int(config["architecture"]["vocab_size"]),
            (1, int(config["architecture"]["context_length"])),
            device=context.device,
        )
        torch.cuda.reset_peak_memory_stats(context.device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(runtime, context.device):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, int(config["architecture"]["vocab_size"])),
                inputs.reshape(-1),
            )
        if not torch.isfinite(loss):
            raise RuntimeError("CompleteP paper-anchor loss is non-finite")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(context.device)
        payload = {
            "schema_version": 1,
            "status": "passed",
            "plan_fingerprint": plan["fingerprint"],
            "scale": scale,
            "paper_contract": {
                "position_encoding": "alibi",
                "learned_position_parameters": 0,
                "attention_scale": "QK^T/N",
                "activation": config["architecture"]["activation"],
                "untied_embeddings": architecture["tied_embeddings"] is False,
                "context_length": int(config["architecture"]["context_length"]),
                "base_learning_rate": 0.00390625,
            },
            "optimizer_groups": [
                {
                    "name": group["name"],
                    "learning_rate": group["peak_learning_rate"],
                    "epsilon": group["epsilon"],
                    "weight_decay": group["weight_decay"],
                }
                for group in groups
            ],
            "canary_loss": float(loss.detach().cpu()),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
            "gpu_name": torch.cuda.get_device_name(context.device),
        }
    finally:
        close_distributed(context)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
