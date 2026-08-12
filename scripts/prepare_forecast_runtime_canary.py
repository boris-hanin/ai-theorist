#!/usr/bin/env python3
"""Bind a forecast template to data and compile one exact upper-rung canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.study import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--fused", choices=("true", "false"), default="true")
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    parser.add_argument("--checkpoint-seconds", type=float, default=900)
    parser.add_argument("--distributed", choices=("none", "ddp"), default="none")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.checkpoint_steps < 0 or args.checkpoint_seconds < 0:
        raise ValueError("checkpoint cadences must be non-negative")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    if (args.distributed == "none") != (args.num_processes == 1):
        raise ValueError("none requires one process and ddp requires multiple processes")

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["run_profile"] = "smoke"
    config.pop("extension_contract", None)
    config["dataset"]["token_stream_manifest_path"] = str(
        args.manifest.resolve()
    )
    config["seeds"] = [args.seed]
    config["bootstrap_samples"] = 0
    config["run_negative_control"] = False
    # Runtime canaries deliberately exercise a large model against a tiny
    # pinned stream. They are never scaling-law evidence, so repetition is
    # allowed and recorded rather than pretending this is a forecast run.
    config["ladder"]["maximum_repetition_ratio"] = 1_000_000.0
    config["runtime"].update(
        {
            "precision": "bf16",
            "attention_backend": "flash",
            "distributed": args.distributed,
            "num_processes": args.num_processes,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "checkpoint_interval_steps": args.checkpoint_steps,
            "checkpoint_interval_seconds": args.checkpoint_seconds,
            "resume": True,
        }
    )
    config["optimizer"]["fused"] = args.fused == "true"
    if args.learning_rate not in {
        float(value) for value in config["optimizer"]["learning_rates"]
    }:
        raise ValueError("learning rate must be in the template grid")

    if "optimizer_steps" in config["ladder"]:
        config["ladder"]["optimizer_steps"] = args.steps
    else:
        config["ladder"]["tokens_per_parameter"] = 0.0001
        provisional = compile_real_text_scaling_plan(config)
        largest = provisional["scales"][-1]
        batch_tokens = int(config["batch_examples"]) * int(
            config["architecture"]["context_length"]
        )
        config["ladder"]["tokens_per_parameter"] = (
            args.steps * batch_tokens / int(largest["parameters"])
        )
    config["validation_interval_steps"] = max(1, args.steps // 4)
    plan = compile_real_text_scaling_plan(config)
    largest = plan["scales"][-1]
    if int(largest["optimizer_steps"]) != args.steps:
        raise RuntimeError("canary token rounding did not preserve the requested steps")
    atomic_write_json(args.output, config)
    print(
        json.dumps(
            {
                "config": str(args.output.resolve()),
                "plan_fingerprint": plan["fingerprint"],
                "parameters": largest["parameters"],
                "optimizer_steps": largest["optimizer_steps"],
                "presented_tokens": largest["presented_tokens"],
                "task_id": (
                    f"ladder-{largest['name']}-theory-"
                    f"eta{args.learning_rate:g}-seed{args.seed}"
                ),
                "fused": config["optimizer"]["fused"],
                "distributed": args.distributed,
                "num_processes": args.num_processes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
