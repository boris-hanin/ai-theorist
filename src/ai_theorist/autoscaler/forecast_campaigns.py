from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .batch_scaling import BatchRunRecord, OptimizerHyperparameters
from .jiang_chizat import (
    JIANG_COMPLETEP_ADAM_THEORY,
    JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
    JiangChizatBlock,
    JiangChizatReference,
    JiangChizatShape,
    JiangChizatTransformer,
)
from .lr_schedules import LearningRateSchedule
from .normalized_transformer import (
    NUGPT_ADAM_THEORY,
    NormalizedTransformer,
    NormalizedTransformerBlock,
)
from .pretraining import (
    DistributedContext,
    PretrainingRuntimeSpec,
    TokenizedTextCorpus,
    TokenizedTextSpec,
    _autocast,
    clear_runtime_checkpoint,
    close_distributed,
    load_runtime_checkpoint,
    prepare_distributed,
    runtime_checkpoint_due,
    save_runtime_checkpoint,
    wrap_distributed_model,
)
from .scaling import fit_scaling_ensemble
from .schema import ArchitectureTemplate, ScaleLevel
from .study import atomic_write_json
from .tokenization import token_stream_identity


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
CAMPAIGN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TheoryScale:
    name: str
    block_type: str
    target_parameters: int
    parameters: int
    relative_parameter_error: float
    depth: int
    width: int
    hidden_width: Optional[int]
    num_heads: int
    presented_tokens: int
    optimizer_steps: int
    tokens_per_parameter: float
    repetition_ratio: float
    iteration_ratio: float
    heldout: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def jiang_parameter_count(
    *,
    vocab_size: int,
    context_length: int,
    depth: int,
    residual_width: int,
    hidden_width: int,
) -> int:
    per_block = (
        4 * residual_width * residual_width
        + 2 * residual_width * hidden_width
        + 9 * residual_width
        + hidden_width
    )
    return int(
        vocab_size * residual_width
        + context_length * residual_width
        + depth * per_block
        + 2 * residual_width
    )


def nugpt_parameter_count(
    *,
    vocab_size: int,
    depth: int,
    width: int,
    mlp_multiplier: int,
) -> int:
    per_block = (
        (4 + 3 * mlp_multiplier) * width * width
        + (3 + 2 * mlp_multiplier) * width
    )
    return int(2 * vocab_size * width + vocab_size + depth * per_block)


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _generate_ladder(
    config: Mapping[str, Any], identity: Mapping[str, Any]
) -> Tuple[List[TheoryScale], Dict[str, Any]]:
    architecture = config.get("architecture")
    ladder = config.get("ladder")
    if not isinstance(architecture, Mapping) or not isinstance(ladder, Mapping):
        raise ValueError("architecture and ladder must be objects")
    block_type = str(architecture.get("block_type"))
    if block_type not in {"jiang_chizat_transformer", "normalized_transformer"}:
        raise ValueError(
            "architecture.block_type must be jiang_chizat_transformer or "
            "normalized_transformer"
        )
    vocab_size = _positive_int(architecture.get("vocab_size"), "vocab_size", 8)
    if vocab_size != int(identity["vocab_size"]):
        raise ValueError(
            f"architecture requires vocab_size {identity['vocab_size']} for its token stream"
        )
    context_length = _positive_int(
        architecture.get("context_length"), "context_length", 2
    )
    head_dimension = _positive_int(
        architecture.get("head_dimension"), "head_dimension", 2
    )
    targets = tuple(
        _positive_int(value, "target_parameters", 32)
        for value in ladder.get("target_parameters", ())
    )
    if len(targets) < 6 or tuple(sorted(set(targets))) != targets:
        raise ValueError(
            "ladder.target_parameters must contain at least six unique increasing values"
        )
    depths = tuple(
        _positive_int(value, "ladder.depths") for value in ladder.get("depths", ())
    )
    if len(depths) != len(targets):
        raise ValueError("ladder.depths must match target_parameters length")
    maximum_width = _positive_int(
        int(ladder.get("maximum_width", 16_384)), "ladder.maximum_width"
    )
    tolerance = float(ladder.get("maximum_parameter_error_fraction", 0.05))
    if not 0.0 < tolerance < 1.0:
        raise ValueError("maximum_parameter_error_fraction must be in (0,1)")
    heldout_count = _positive_int(
        int(ladder.get("heldout_scale_count", 1)), "heldout_scale_count"
    )
    if heldout_count > len(targets) - 5:
        raise ValueError("at least five non-held-out ladder scales are required")
    tokens_per_parameter = _positive_float(
        ladder.get("tokens_per_parameter"), "tokens_per_parameter"
    )
    batch_examples = _positive_int(config.get("batch_examples"), "batch_examples")
    batch_tokens = batch_examples * context_length
    unique_tokens = int(identity["training_tokens"])

    shapes: List[Dict[str, Any]] = []
    candidates = range(head_dimension, maximum_width + 1, head_dimension)
    if block_type == "jiang_chizat_transformer":
        reference_depth = _positive_int(
            architecture.get("reference_depth"), "reference_depth"
        )
        reference_hidden = _positive_int(
            architecture.get("reference_hidden_width"), "reference_hidden_width"
        )
        reference_residual = _positive_int(
            architecture.get("reference_residual_width"), "reference_residual_width"
        )
        rho = float(
            ladder.get(
                "rho_lm_over_d",
                reference_depth * reference_hidden / reference_residual,
            )
        )
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("rho_lm_over_d must be finite and positive")
        hidden_multiple = _positive_int(
            int(ladder.get("hidden_width_multiple", head_dimension)),
            "hidden_width_multiple",
        )
        for target, depth in zip(targets, depths):
            rows = []
            for residual_width in candidates:
                hidden_width = _nearest_multiple(
                    rho * residual_width / depth, hidden_multiple
                )
                parameters = jiang_parameter_count(
                    vocab_size=vocab_size,
                    context_length=context_length,
                    depth=depth,
                    residual_width=residual_width,
                    hidden_width=hidden_width,
                )
                rows.append(
                    {
                        "target": target,
                        "depth": depth,
                        "width": residual_width,
                        "hidden_width": hidden_width,
                        "parameters": parameters,
                        "num_heads": residual_width // head_dimension,
                    }
                )
            shapes.append(min(rows, key=lambda row: abs(row["parameters"] - target)))
        contract = {
            "parameterization": "jiang_completep_adam",
            "theory": asdict(JIANG_COMPLETEP_ADAM_THEORY),
            "rho_lm_over_d": rho,
            "tied_embeddings": True,
            "attention_scale": "QK^T/d_head",
            "residual_branch_scale": "1/L",
        }
    else:
        mlp_multiplier = _positive_int(
            architecture.get("mlp_multiplier"), "mlp_multiplier"
        )
        reference_width = _positive_int(
            architecture.get("reference_width"), "reference_width"
        )
        reference_depth = _positive_int(
            architecture.get("reference_depth"), "reference_depth"
        )
        del reference_width, reference_depth
        for target, depth in zip(targets, depths):
            rows = [
                {
                    "target": target,
                    "depth": depth,
                    "width": width,
                    "hidden_width": None,
                    "parameters": nugpt_parameter_count(
                        vocab_size=vocab_size,
                        depth=depth,
                        width=width,
                        mlp_multiplier=mlp_multiplier,
                    ),
                    "num_heads": width // head_dimension,
                }
                for width in candidates
            ]
            shapes.append(min(rows, key=lambda row: abs(row["parameters"] - target)))
        contract = {
            "parameterization": "nugpt_mid_alignment",
            "theory": asdict(NUGPT_ADAM_THEORY),
            "tied_embeddings": False,
            "matrix_constraint": "post-step unit-sphere projection",
        }

    scales: List[TheoryScale] = []
    for index, row in enumerate(shapes):
        parameters = int(row["parameters"])
        relative_error = abs(parameters / int(row["target"]) - 1.0)
        if relative_error > tolerance:
            raise ValueError(
                f"no width <= {maximum_width} places S{index + 1} within the "
                f"declared parameter tolerance"
            )
        requested_tokens = tokens_per_parameter * parameters
        presented_tokens = max(
            batch_tokens, int(round(requested_tokens / batch_tokens)) * batch_tokens
        )
        scales.append(
            TheoryScale(
                name=f"S{index + 1}",
                block_type=block_type,
                target_parameters=int(row["target"]),
                parameters=parameters,
                relative_parameter_error=relative_error,
                depth=int(row["depth"]),
                width=int(row["width"]),
                hidden_width=(
                    int(row["hidden_width"])
                    if row["hidden_width"] is not None
                    else None
                ),
                num_heads=int(row["num_heads"]),
                presented_tokens=presented_tokens,
                optimizer_steps=presented_tokens // batch_tokens,
                tokens_per_parameter=presented_tokens / parameters,
                repetition_ratio=presented_tokens / unique_tokens,
                iteration_ratio=1.0,
                heldout=index >= len(shapes) - heldout_count,
            )
        )
    reference_index = int(ladder.get("reference_scale_index", 0))
    if not 0 <= reference_index < len(scales) - heldout_count:
        raise ValueError("reference_scale_index must select a non-held-out scale")
    reference_steps = scales[reference_index].optimizer_steps
    scales = [
        TheoryScale(
            **{
                **row.to_dict(),
                "iteration_ratio": row.optimizer_steps / reference_steps,
            }
        )
        for row in scales
    ]
    return scales, {**contract, "reference_scale_index": reference_index}


def compile_real_text_scaling_plan(config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset must be an object")
    manifest_path = dataset.get("token_stream_manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        raise ValueError("a verified token_stream_manifest_path is required")
    identity = token_stream_identity(Path(manifest_path))
    if dataset.get("tokenizer") != identity["tokenizer_id"]:
        raise ValueError("dataset tokenizer does not match token stream manifest")
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    scales, architecture_contract = _generate_ladder(config, identity)
    block_type = scales[0].block_type
    if block_type == "normalized_transformer" and runtime.distributed == "fsdp":
        raise ValueError(
            "nGPT refuses FSDP because sharded post-step matrix projection is not "
            "yet definition preserving; use DDP or one GPU"
        )
    head_dimension = int(config["architecture"]["head_dimension"])
    if runtime.attention_backend == "flash" and head_dimension % 8:
        raise ValueError("FlashAttention requires head_dimension divisible by eight")
    data_parallel_microbatches = (
        runtime.num_processes * runtime.gradient_accumulation_steps
    )
    batch_examples = _positive_int(config.get("batch_examples"), "batch_examples")
    if batch_examples % data_parallel_microbatches:
        raise ValueError(
            "batch_examples must be divisible by num_processes times "
            "gradient_accumulation_steps"
        )
    validation_examples = _positive_int(
        int(config.get("validation_examples", 256)), "validation_examples"
    )
    if validation_examples % runtime.num_processes:
        raise ValueError("validation_examples must be divisible by num_processes")
    validation_interval_steps = _positive_int(
        config.get("validation_interval_steps"), "validation_interval_steps"
    )
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping) or optimizer.get("name") != "adam":
        raise ValueError("theory scaling ladder currently requires Adam")
    rates = tuple(
        _positive_float(value, "optimizer.learning_rates")
        for value in optimizer.get("learning_rates", ())
    )
    if len(rates) < 3 or tuple(sorted(set(rates))) != rates:
        raise ValueError(
            "optimizer.learning_rates must contain at least three increasing values"
        )
    optimizer_contract = json.loads(json.dumps(dict(optimizer), sort_keys=True))
    fused_optimizer = optimizer_contract.get("fused", False)
    if not isinstance(fused_optimizer, bool):
        raise ValueError("optimizer.fused must be boolean")
    if fused_optimizer and runtime.precision != "bf16":
        raise ValueError("the fused Adam forecast path requires bf16")
    seeds = tuple(int(value) for value in config.get("seeds", (11, 29, 47)))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    run_profile = str(config.get("run_profile", "forecast"))
    if run_profile not in {"smoke", "forecast"}:
        raise ValueError("run_profile must be smoke or forecast")
    if run_profile == "forecast" and len(seeds) < 3:
        raise ValueError("forecast profile requires at least three seeds")
    schedule = LearningRateSchedule.from_payload(config.get("schedule", "cosine_to_10_percent"))
    ladder = config["ladder"]
    target_forecasts = tuple(
        _positive_int(value, "target_forecasts", 32)
        for value in ladder.get("target_forecasts", ())
    )
    if not target_forecasts or tuple(sorted(set(target_forecasts))) != target_forecasts:
        raise ValueError("ladder.target_forecasts must be unique and increasing")
    if target_forecasts[0] <= scales[-1].parameters:
        raise ValueError("every target forecast must exceed the largest ladder scale")
    minimum_span = _positive_float(
        ladder.get("minimum_parameter_span", 30.0), "minimum_parameter_span"
    )
    fit_scales = [row for row in scales if not row.heldout]
    observed_span = fit_scales[-1].parameters / fit_scales[0].parameters
    maximum_repetition = _positive_float(
        ladder.get("maximum_repetition_ratio", 1.0), "maximum_repetition_ratio"
    )
    plan_payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign": "real_text_scaling_ladder",
        "run_profile": run_profile,
        "dataset_identity": identity,
        "architecture_contract": architecture_contract,
        "runtime": asdict(runtime),
        "schedule": schedule.audit(scales[-1].optimizer_steps),
        "batch_examples": batch_examples,
        "validation_examples": validation_examples,
        "learning_rates": list(rates),
        "optimizer_contract": optimizer_contract,
        "seeds": list(seeds),
        "measurement_contract": {
            "validation_examples": validation_examples,
            "validation_interval_steps": validation_interval_steps,
        },
        "scales": [row.to_dict() for row in scales],
        "fit_parameter_span": observed_span,
        "minimum_parameter_span": minimum_span,
        "maximum_repetition_ratio": maximum_repetition,
        "target_forecasts": list(target_forecasts),
        "tuning_trials": len(rates) * len(seeds),
        "scale_trials": len(scales) * len(seeds),
        "negative_control_trials": (
            len(seeds) if bool(config.get("run_negative_control", True)) else 0
        ),
        "planned_grid_trials": (
            len(rates) * len(seeds)
            + len(scales) * len(seeds)
            + (len(seeds) if bool(config.get("run_negative_control", True)) else 0)
        ),
        "execution_order": [
            "verify_tokenizer_and_token_stream",
            "compile_exact_vocab_aware_ladder",
            "recall_architecture_specific_parameter_group_rules",
            "tune_reference_scale",
            "freeze_learning_rate_and_training_path",
            "train_nonheldout_scales",
            "evaluate_hidden_upper_rungs",
            "run_wrong_global_learning_rate_control",
            "rolling_scaling_law_backtests",
            "issue_or_refuse_bounded_forecasts",
        ],
    }
    plan_payload["fingerprint"] = sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return plan_payload


def _distributed_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = value.detach().clone()
    if context.world_size > 1:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        result.div_(context.world_size)
    return result


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    corpus: TokenizedTextCorpus,
    *,
    vocab_size: int,
    validation_examples: int,
    seed: int,
    runtime: PretrainingRuntimeSpec,
    context: DistributedContext,
) -> float:
    model.eval()
    local_examples = validation_examples // context.world_size
    generator = torch.Generator(device="cpu").manual_seed(
        900_001 + seed + 1_000_003 * context.rank
    )
    inputs, targets = corpus.sample_batch(
        "validation", local_examples, generator, context.device
    )
    with _autocast(runtime, context.device):
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.float().reshape(-1, vocab_size), targets.reshape(-1)
        )
    result = float(_distributed_mean(loss, context).cpu())
    model.train()
    return result


def _build_model_and_groups(
    *,
    config: Mapping[str, Any],
    scale: Mapping[str, Any],
    eta: float,
    optimizer_mode: str,
    runtime: PretrainingRuntimeSpec,
    context: DistributedContext,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, List[Dict[str, Any]], Dict[str, Any]]:
    architecture = dict(config["architecture"])
    optimizer_payload = dict(config["optimizer"])
    block_type = str(architecture["block_type"])
    capture_diagnostics = runtime.attention_backend == "math"
    if block_type == "jiang_chizat_transformer":
        shape = JiangChizatShape(
            int(scale["depth"]),
            int(scale["hidden_width"]),
            int(scale["width"]),
            int(architecture["head_dimension"]),
        )
        reference = JiangChizatReference(
            int(architecture["reference_depth"]),
            int(architecture["reference_hidden_width"]),
            int(architecture["reference_residual_width"]),
        )
        plain_model: nn.Module = JiangChizatTransformer(
            shape,
            vocab_size=int(architecture["vocab_size"]),
            context_length=int(architecture["context_length"]),
            reference=reference,
            attention_backend=runtime.attention_backend,
            activation_checkpointing=runtime.activation_checkpointing,
            capture_attention_diagnostics=capture_diagnostics,
        ).to(context.device)
        assert isinstance(plain_model, JiangChizatTransformer)
        multipliers = dict(
            optimizer_payload.get(
                "learning_rate_multipliers", JIANG_DENSE_REPORTED_LR_MULTIPLIERS
            )
        )
        groups = plain_model.optimizer_parameter_groups(
            eta,
            epsilon0=float(optimizer_payload.get("epsilon", 1e-12)),
            learning_rate_multipliers=multipliers,
        )
        group_audit = plain_model.optimizer_contract_audit(
            eta,
            epsilon0=float(optimizer_payload.get("epsilon", 1e-12)),
            learning_rate_multipliers=multipliers,
        )
        block_types: Sequence[type[nn.Module]] = (JiangChizatBlock,)
    else:
        architecture_template = ArchitectureTemplate.from_dict(architecture)
        scale_level = ScaleLevel(
            name=str(scale["name"]),
            width=int(scale["width"]),
            repeats=int(scale["depth"]),
        )
        plain_model = NormalizedTransformer(
            architecture_template,
            scale_level,
            attention_backend=runtime.attention_backend,
            activation_checkpointing=runtime.activation_checkpointing,
            capture_attention_diagnostics=capture_diagnostics,
        ).to(context.device)
        assert isinstance(plain_model, NormalizedTransformer)
        groups = plain_model.optimizer_parameter_groups(
            eta,
            data_multiplier=float(scale["iteration_ratio"]),
            output_learning_rate_multiplier=float(
                optimizer_payload.get("output_learning_rate_multiplier", 0.5)
            ),
            adam_epsilon=float(optimizer_payload.get("epsilon", 1e-16)),
        )
        group_audit = plain_model.optimizer_contract_audit(
            eta,
            data_multiplier=float(scale["iteration_ratio"]),
            output_learning_rate_multiplier=float(
                optimizer_payload.get("output_learning_rate_multiplier", 0.5)
            ),
            adam_epsilon=float(optimizer_payload.get("epsilon", 1e-16)),
        )
        block_types = (NormalizedTransformerBlock,)
    parameter_count = sum(parameter.numel() for parameter in plain_model.parameters())
    if parameter_count != int(scale["parameters"]):
        raise RuntimeError(
            f"compiled parameter count {scale['parameters']} disagrees with model "
            f"construction {parameter_count}"
        )
    model = wrap_distributed_model(plain_model, runtime, context, block_types)
    if optimizer_mode == "wrong_global":
        optimizer_groups: Any = model.parameters()
        group_contract = [
            {
                "name": "wrong_single_global_learning_rate",
                "peak_learning_rate": eta,
                "learning_rate_formula": "eta for every parameter; negative control",
            }
        ]
    elif optimizer_mode == "theory":
        optimizer_groups = groups
        group_contract = [
            {
                "name": str(group["name"]),
                "peak_learning_rate": float(group["lr"]),
                "epsilon": float(group["eps"]),
                "learning_rate_formula": str(group["lr_formula"]),
                "epsilon_formula": str(group["eps_formula"]),
                "scale_factors": dict(group["scale_factors"]),
                "theory_contract_id": str(group["theory_contract_id"]),
            }
            for group in groups
        ]
    else:
        raise ValueError("optimizer_mode must be theory or wrong_global")
    fused = bool(optimizer_payload.get("fused", False))
    if fused and not context.device.startswith("cuda"):
        raise ValueError("fused Adam requires CUDA")
    optimizer = torch.optim.Adam(
        optimizer_groups,
        lr=eta,
        betas=(
            float(optimizer_payload.get("beta1", 0.9)),
            float(optimizer_payload.get("beta2", 0.95)),
        ),
        eps=float(optimizer_payload.get("epsilon", 1e-12)),
        weight_decay=0.0,
        fused=fused,
    )
    group_audit = {**group_audit, "optimizer_backend": "fused_adam" if fused else "adam"}
    return model, plain_model, optimizer, group_contract, group_audit


def _broadcast_record(
    record: Optional[BatchRunRecord], context: DistributedContext
) -> BatchRunRecord:
    if context.world_size == 1:
        assert record is not None
        return record
    payload: List[Any] = [record.to_dict() if record is not None else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return BatchRunRecord.from_dict(payload[0])


def forecast_trial_cache_identity(
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    scale: Mapping[str, Any],
    dataset_fingerprint: str,
    runtime: PretrainingRuntimeSpec,
    eta: float,
    seed: int,
    optimizer_mode: str,
) -> Tuple[str, str]:
    """Return the immutable fingerprint and filename stem for one trial."""

    schedule = LearningRateSchedule.from_payload(config["schedule"])
    identity = {
        "schema_version": 1,
        "plan_fingerprint": plan["fingerprint"],
        "scale": dict(scale),
        "dataset_fingerprint": dataset_fingerprint,
        "runtime": asdict(runtime),
        "eta": eta,
        "seed": seed,
        "optimizer_mode": optimizer_mode,
        "schedule": schedule.audit(int(scale["optimizer_steps"])),
    }
    identity_fingerprint = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_id = (
        f"forecast-{scale['name']}-{optimizer_mode}-eta{eta:g}-s{seed}-"
        f"{identity_fingerprint[:12]}"
    )
    return identity_fingerprint, run_id


def _run_trial(
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    scale: Mapping[str, Any],
    corpus: TokenizedTextCorpus,
    runtime: PretrainingRuntimeSpec,
    context: DistributedContext,
    eta: float,
    seed: int,
    optimizer_mode: str,
    cache_directory: Path,
) -> BatchRunRecord:
    schedule = LearningRateSchedule.from_payload(config["schedule"])
    identity_fingerprint, run_id = forecast_trial_cache_identity(
        config=config,
        plan=plan,
        scale=scale,
        dataset_fingerprint=corpus.identity_fingerprint,
        runtime=runtime,
        eta=eta,
        seed=seed,
        optimizer_mode=optimizer_mode,
    )
    record_path = cache_directory / f"{run_id}.json"
    cache_hit = record_path.is_file() if context.is_primary else False
    if context.world_size > 1:
        marker = [cache_hit]
        torch.distributed.broadcast_object_list(marker, src=0)
        cache_hit = bool(marker[0])
    if cache_hit:
        record = None
        if context.is_primary:
            with record_path.open("r", encoding="utf-8") as handle:
                record = BatchRunRecord.from_dict(json.load(handle))
        return _broadcast_record(record, context)

    torch.manual_seed(seed)
    if context.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(context.device)
    model, plain_model, optimizer, group_contract, group_audit = (
        _build_model_and_groups(
            config=config,
            scale=scale,
            eta=eta,
            optimizer_mode=optimizer_mode,
            runtime=runtime,
            context=context,
        )
    )
    peak_rates = [float(group["lr"]) for group in optimizer.param_groups]
    batch_examples = int(config["batch_examples"])
    local_microbatch_examples = batch_examples // (
        context.world_size * runtime.gradient_accumulation_steps
    )
    steps = int(scale["optimizer_steps"])
    generator = torch.Generator(device="cpu").manual_seed(
        100_003 + seed + 1_000_003 * context.rank
    )
    resume_base = cache_directory / f"{run_id}.resume"
    resumed = load_runtime_checkpoint(
        base_path=resume_base,
        model=model,
        plain_model=plain_model,
        optimizer=optimizer,
        context=context,
        runtime=runtime,
        identity_fingerprint=identity_fingerprint,
        generator=generator,
    )
    checkpoints: List[Dict[str, float]]
    elapsed_before_resume = 0.0
    if resumed is None:
        initial_loss = _evaluate(
            model,
            corpus,
            vocab_size=int(config["architecture"]["vocab_size"]),
            validation_examples=int(config.get("validation_examples", 256)),
            seed=seed,
            runtime=runtime,
            context=context,
        )
        checkpoints = [
            {"step": 0.0, "tokens": 0.0, "validation_loss": initial_loss}
        ]
        start_step = 0
    else:
        start_step = int(resumed["step"])
        checkpoints = [
            dict(row) for row in resumed["extra"]["validation_checkpoints"]
        ]
        elapsed_before_resume = float(
            resumed["extra"].get("elapsed_seconds", 0.0)
        )
        initial_loss = float(checkpoints[0]["validation_loss"])
    validation_interval = _positive_int(
        int(config.get("validation_interval_steps", max(1, steps // 8))),
        "validation_interval_steps",
    )
    started = time.monotonic()
    last_checkpoint_at = started
    model.train()
    for step in range(start_step + 1, steps + 1):
        multiplier = schedule.multiplier(step, steps)
        for group, peak_rate in zip(optimizer.param_groups, peak_rates):
            group["lr"] = peak_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        for accumulation_index in range(runtime.gradient_accumulation_steps):
            inputs, targets = corpus.sample_batch(
                "train",
                local_microbatch_examples,
                generator,
                context.device,
            )
            synchronization = (
                model.no_sync()  # type: ignore[attr-defined]
                if context.world_size > 1
                and accumulation_index + 1 < runtime.gradient_accumulation_steps
                else nullcontext()
            )
            with synchronization, _autocast(runtime, context.device):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.float().reshape(
                        -1, int(config["architecture"]["vocab_size"])
                    ),
                    targets.reshape(-1),
                ) / runtime.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError("theory-faithful pretraining trial diverged")
            loss.backward()
        optimizer.step()
        if isinstance(plain_model, NormalizedTransformer):
            plain_model.project_normalized_weights()
        if step % validation_interval == 0 or step == steps:
            validation_loss = _evaluate(
                model,
                corpus,
                vocab_size=int(config["architecture"]["vocab_size"]),
                validation_examples=int(config.get("validation_examples", 256)),
                seed=seed,
                runtime=runtime,
                context=context,
            )
            checkpoints.append(
                {
                    "step": float(step),
                    "tokens": float(
                        step
                        * batch_examples
                        * int(config["architecture"]["context_length"])
                    ),
                    "validation_loss": validation_loss,
                }
            )
        checkpoint_now = time.monotonic()
        if runtime_checkpoint_due(
            runtime,
            step=step,
            total_steps=steps,
            last_checkpoint_at=last_checkpoint_at,
            now=checkpoint_now,
        ):
            save_runtime_checkpoint(
                base_path=resume_base,
                model=model,
                plain_model=plain_model,
                optimizer=optimizer,
                context=context,
                runtime=runtime,
                identity_fingerprint=identity_fingerprint,
                step=step,
                generator=generator,
                extra={
                    "validation_checkpoints": checkpoints,
                    "elapsed_seconds": elapsed_before_resume
                    + time.monotonic()
                    - started,
                },
            )
            last_checkpoint_at = checkpoint_now
    duration = elapsed_before_resume + time.monotonic() - started
    final_loss = float(checkpoints[-1]["validation_loss"])
    diagnostics: Dict[str, Any] = {}
    if isinstance(plain_model, NormalizedTransformer):
        diagnostics = plain_model.sphere_diagnostics()
    elif isinstance(plain_model, JiangChizatTransformer):
        diagnostics = plain_model.diagnostics()
    record: Optional[BatchRunRecord] = None
    if context.is_primary:
        record = BatchRunRecord(
            run_id=run_id,
            model_family=(
                "nugpt_real_text_scaling"
                if isinstance(plain_model, NormalizedTransformer)
                else "jiang_chizat_real_text_scaling"
            ),
            optimizer=OptimizerHyperparameters(
                name="adam",
                learning_rate=eta,
                beta1=float(config["optimizer"].get("beta1", 0.9)),
                beta2=float(config["optimizer"].get("beta2", 0.95)),
                epsilon=float(config["optimizer"].get("epsilon", 1e-12)),
                weight_decay=0.0,
            ),
            seed=seed,
            parameter_count=int(scale["parameters"]),
            width=int(scale["width"]),
            depth=int(scale["depth"]),
            total_tokens=int(scale["presented_tokens"]),
            batch_tokens=(
                batch_examples * int(config["architecture"]["context_length"])
            ),
            microbatch_tokens=(
                local_microbatch_examples
                * int(config["architecture"]["context_length"])
            ),
            accumulation_steps=runtime.gradient_accumulation_steps,
            data_parallel_replicas=context.world_size,
            optimizer_steps=steps,
            nonpadding_tokens_seen=int(scale["presented_tokens"]),
            learning_rate_schedule=schedule.name,
            final_validation_loss=final_loss,
            estimated_flops=float(
                6 * int(scale["parameters"]) * int(scale["presented_tokens"])
            ),
            wall_time_seconds=duration,
            validation_checkpoints=tuple(checkpoints),
            metadata={
                "scale": dict(scale),
                "dataset_fingerprint": corpus.identity_fingerprint,
                "tokenizer_fingerprint": corpus.tokenizer_fingerprint,
                "tokenizer_is_pinned": corpus.tokenizer_is_pinned,
                "optimizer_mode": optimizer_mode,
                "peak_parameter_group_contract": group_contract,
                "optimizer_group_audit": group_audit,
                "gradient_clipping": "none_source_faithful",
                "activation_checkpointing": runtime.activation_checkpointing,
                "resumed_from_step": start_step,
                "diagnostics": diagnostics,
                "peak_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(context.device))
                    if context.device.startswith("cuda")
                    else 0
                ),
            },
        )
        atomic_write_json(record_path, record.to_dict())
    if context.world_size > 1:
        torch.distributed.barrier()
    clear_runtime_checkpoint(resume_base, context)
    return _broadcast_record(record, context)


def _mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    mean = float(np.mean(values))
    sem = (
        float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return mean, sem


def _progress(
    callback: ProgressCallback,
    phase: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": phase,
                "completed": completed,
                "total": total,
                "message": message,
            }
        )


def run_real_text_scaling_campaign(
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    plan = compile_real_text_scaling_plan(config)
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    context = prepare_distributed(runtime, device)
    completed = 0
    total = int(plan["planned_grid_trials"])
    try:
        dataset_payload = dict(config["dataset"])
        corpus = TokenizedTextCorpus(
            TokenizedTextSpec.from_dict(dataset_payload),
            context_length=int(config["architecture"]["context_length"]),
            vocab_size=int(config["architecture"]["vocab_size"]),
        )
        if not corpus.tokenizer_is_pinned:
            raise ValueError("forecast-grade scaling requires pinned tokenizer provenance")
        cache_directory = Path(
            str(config.get("cache_directory", "runs/autoscaler/forecast-trials"))
        )
        cache_directory.mkdir(parents=True, exist_ok=True)
        scales = [dict(row) for row in plan["scales"]]
        reference_scale = scales[
            int(plan["architecture_contract"]["reference_scale_index"])
        ]
        seeds = [int(value) for value in plan["seeds"]]
        tuning_records: List[BatchRunRecord] = []
        tuning_rows = []
        for eta in plan["learning_rates"]:
            matching = []
            for seed in seeds:
                _progress(
                    progress,
                    "tune-reference",
                    completed,
                    total,
                    f"Reference LR {eta:g} · seed {seed}",
                )
                record = _run_trial(
                    config=config,
                    plan=plan,
                    scale=reference_scale,
                    corpus=corpus,
                    runtime=runtime,
                    context=context,
                    eta=float(eta),
                    seed=seed,
                    optimizer_mode="theory",
                    cache_directory=cache_directory,
                )
                tuning_records.append(record)
                matching.append(record.final_validation_loss)
                completed += 1
            mean, sem = _mean_sem(matching)
            tuning_rows.append(
                {
                    "learning_rate": float(eta),
                    "mean_validation_loss": mean,
                    "sem_validation_loss": sem,
                    "seed_losses": matching,
                }
            )
        selected_index = min(
            range(len(tuning_rows)),
            key=lambda index: tuning_rows[index]["mean_validation_loss"],
        )
        selected_eta = float(tuning_rows[selected_index]["learning_rate"])
        optimum_interior = 0 < selected_index < len(tuning_rows) - 1

        scale_records: List[BatchRunRecord] = []
        for scale in scales:
            for seed in seeds:
                _progress(
                    progress,
                    "train-ladder",
                    completed,
                    total,
                    f"{scale['name']} · {scale['parameters']:,} parameters · seed {seed}",
                )
                scale_records.append(
                    _run_trial(
                        config=config,
                        plan=plan,
                        scale=scale,
                        corpus=corpus,
                        runtime=runtime,
                        context=context,
                        eta=selected_eta,
                        seed=seed,
                        optimizer_mode="theory",
                        cache_directory=cache_directory,
                    )
                )
                completed += 1
        negative_records: List[BatchRunRecord] = []
        if bool(config.get("run_negative_control", True)):
            largest = scales[-1]
            for seed in seeds:
                _progress(
                    progress,
                    "negative-control",
                    completed,
                    total,
                    f"Wrong-global-LR control · {largest['name']} · seed {seed}",
                )
                negative_records.append(
                    _run_trial(
                        config=config,
                        plan=plan,
                        scale=largest,
                        corpus=corpus,
                        runtime=runtime,
                        context=context,
                        eta=selected_eta,
                        seed=seed,
                        optimizer_mode="wrong_global",
                        cache_directory=cache_directory,
                    )
                )
                completed += 1

        aggregates = []
        for scale in scales:
            records = [
                row for row in scale_records if row.metadata["scale"]["name"] == scale["name"]
            ]
            mean, sem = _mean_sem([row.final_validation_loss for row in records])
            aggregates.append(
                {
                    **scale,
                    "mean_validation_loss": mean,
                    "sem_validation_loss": sem,
                    "seed_losses": [row.final_validation_loss for row in records],
                }
            )
        ladder_config = dict(config["ladder"])
        maximum_extrapolation = float(
            ladder_config.get("maximum_extrapolation_factor", 10.0)
        )
        maximum_family_spread = float(
            ladder_config.get("maximum_family_spread", 0.08)
        )
        maximum_backtest_error = float(
            ladder_config.get("maximum_backtest_relative_error", 0.10)
        )
        bootstrap_samples = int(config.get("bootstrap_samples", 200))
        holdout_backtests = []
        for index, row in enumerate(aggregates):
            if not row["heldout"]:
                continue
            prefix = aggregates[:index]
            ensemble = fit_scaling_ensemble(
                [item["parameters"] for item in prefix],
                [item["mean_validation_loss"] for item in prefix],
                [item["sem_validation_loss"] for item in prefix],
                target_size=float(row["parameters"]),
                maximum_extrapolation_factor=maximum_extrapolation,
                maximum_family_spread=maximum_family_spread,
                maximum_backtest_relative_error=maximum_backtest_error,
                bootstrap_samples=bootstrap_samples,
            )
            prediction = float(ensemble["exploratory_prediction"])
            relative_error = abs(prediction / row["mean_validation_loss"] - 1.0)
            passed = bool(ensemble["certified"]) and (
                relative_error <= maximum_backtest_error
            )
            holdout_backtests.append(
                {
                    "scale": row["name"],
                    "parameters": row["parameters"],
                    "observed_loss": row["mean_validation_loss"],
                    "predicted_loss": prediction,
                    "relative_error": relative_error,
                    "passed": passed,
                    "refusal_reasons": list(ensemble["refusal_reasons"]),
                    "fit": ensemble,
                }
            )
        forecasts = []
        for target in plan["target_forecasts"]:
            forecasts.append(
                fit_scaling_ensemble(
                    [item["parameters"] for item in aggregates],
                    [item["mean_validation_loss"] for item in aggregates],
                    [item["sem_validation_loss"] for item in aggregates],
                    target_size=float(target),
                    maximum_extrapolation_factor=maximum_extrapolation,
                    maximum_family_spread=maximum_family_spread,
                    maximum_backtest_relative_error=maximum_backtest_error,
                    bootstrap_samples=bootstrap_samples,
                )
            )
        refusal_reasons: List[str] = []
        if not optimum_interior:
            refusal_reasons.append("reference learning-rate optimum is on the grid boundary")
        if float(plan["fit_parameter_span"]) < float(plan["minimum_parameter_span"]):
            refusal_reasons.append("non-held-out parameter ladder span is too small")
        if max(row["repetition_ratio"] for row in aggregates) > float(
            plan["maximum_repetition_ratio"]
        ):
            refusal_reasons.append("presented-to-unique token repetition exceeds its limit")
        if not holdout_backtests or any(not row["passed"] for row in holdout_backtests):
            refusal_reasons.append("one or more hidden upper-rung predictions failed")
        monotone_transitions = []
        for previous, current in zip(aggregates, aggregates[1:]):
            uncertainty = 2.0 * math.sqrt(
                previous["sem_validation_loss"] ** 2
                + current["sem_validation_loss"] ** 2
            )
            accepted = (
                current["mean_validation_loss"]
                <= previous["mean_validation_loss"] + uncertainty
            )
            monotone_transitions.append(
                {
                    "from_scale": previous["name"],
                    "to_scale": current["name"],
                    "loss_change": (
                        current["mean_validation_loss"]
                        - previous["mean_validation_loss"]
                    ),
                    "two_sem_tolerance": uncertainty,
                    "accepted": accepted,
                }
            )
        if any(not row["accepted"] for row in monotone_transitions):
            refusal_reasons.append(
                "validation loss is non-monotone beyond two-SEM uncertainty"
            )
        negative_control = None
        if negative_records:
            correct = aggregates[-1]["mean_validation_loss"]
            wrong, wrong_sem = _mean_sem(
                [row.final_validation_loss for row in negative_records]
            )
            minimum_degradation = float(
                config.get("negative_control_minimum_degradation", 0.0)
            )
            negative_control = {
                "rule": "wrong_single_global_learning_rate",
                "correct_mean_loss": correct,
                "wrong_mean_loss": wrong,
                "wrong_sem_loss": wrong_sem,
                "relative_degradation": wrong / correct - 1.0,
                "passed": wrong >= correct * (1.0 + minimum_degradation),
            }
            if not negative_control["passed"]:
                refusal_reasons.append("wrong-global-LR negative control was not worse")
        if any(not forecast["certified"] for forecast in forecasts):
            refusal_reasons.append("one or more target forecasts failed ensemble gates")
        result = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "status": "completed",
            "campaign": "real_text_scaling_ladder",
            "plan_fingerprint": plan["fingerprint"],
            "dataset": {
                **plan["dataset_identity"],
                "tokenizer_is_pinned": corpus.tokenizer_is_pinned,
            },
            "architecture_contract": plan["architecture_contract"],
            "runtime": plan["runtime"],
            "reference_tuning": {
                "scale": reference_scale["name"],
                "selected_learning_rate": selected_eta,
                "optimum_is_interior": optimum_interior,
                "grid": tuning_rows,
            },
            "scales": aggregates,
            "hidden_scale_backtests": holdout_backtests,
            "monotonicity_checks": monotone_transitions,
            "negative_control": negative_control,
            "forecasts": forecasts,
            "forecastable": not refusal_reasons,
            "refusal_reasons": refusal_reasons,
            "records": [
                row.to_dict()
                for row in [*tuning_records, *scale_records, *negative_records]
            ],
            "execution_order": plan["execution_order"],
        }
        _progress(
            progress,
            "complete",
            total,
            total,
            (
                "Forecast-grade ladder complete"
                if result["forecastable"]
                else "Ladder complete; forecasts withheld by qualification gates"
            ),
        )
        return result
    finally:
        close_distributed(context)
