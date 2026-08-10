#!/usr/bin/env python3
"""Fast hardware smoke test for the standard pretraining runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from ai_theorist.autoscaler.batch_scaling import OptimizerHyperparameters
from ai_theorist.autoscaler.pretraining import (
    PretrainingRuntimeSpec,
    StandardTransformerSpec,
    TokenizedTextCorpus,
    TokenizedTextSpec,
    close_distributed,
    prepare_distributed,
    run_standard_pretraining_trial,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/pretraining"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--attention", choices=("auto", "math", "flash"), default="math"
    )
    parser.add_argument(
        "--distributed", choices=("none", "fsdp"), default="none"
    )
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    runtime = PretrainingRuntimeSpec(
        precision=args.precision,
        attention_backend=args.attention,
        distributed=args.distributed,
        num_processes=world_size,
    )
    model_spec = StandardTransformerSpec(
        vocab_size=260,
        context_length=16,
        width=32,
        depth=2,
        num_heads=4,
        mlp_multiplier=4,
    )
    corpus = TokenizedTextCorpus(
        TokenizedTextSpec(
            str(args.data_root / "sample_train.txt"),
            str(args.data_root / "sample_validation.txt"),
        ),
        model_spec.context_length,
        model_spec.vocab_size,
    )
    context = prepare_distributed(runtime, args.device)
    try:
        base_batch = max(8, world_size)
        batch_examples = ((base_batch + world_size - 1) // world_size) * world_size
        record, _ = run_standard_pretraining_trial(
            model_spec=model_spec,
            corpus=corpus,
            runtime=runtime,
            distributed_context=context,
            optimizer_spec=OptimizerHyperparameters(
                "adamw",
                0.001,
                beta1=0.9,
                beta2=0.95,
                epsilon=1e-8,
                weight_decay=0.1,
            ),
            total_tokens=batch_examples * model_spec.context_length * 8,
            batch_examples=batch_examples,
            seed=17,
            validation_interval=4,
            validation_examples=16,
            warmup_steps=1,
        )
        if context.is_primary:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "runtime": asdict(runtime),
                        "device": context.device,
                        "world_size": context.world_size,
                        "final_validation_loss": record.final_validation_loss,
                        "peak_memory_bytes": record.metadata["peak_memory_bytes"],
                        "uses_torch_sdpa": record.metadata["uses_torch_sdpa"],
                        "flash_attention_requested": record.metadata[
                            "flash_attention_requested"
                        ],
                    },
                    sort_keys=True,
                )
            )
    finally:
        close_distributed(context)


if __name__ == "__main__":
    main()
