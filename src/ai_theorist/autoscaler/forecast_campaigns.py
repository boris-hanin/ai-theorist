from __future__ import annotations

from copy import deepcopy
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
from .completep import (
    COMPLETEP_ADAMW_THEORY,
    CompletePBlock,
    CompletePReference,
    CompletePShape,
    CompletePTransformer,
)
from .jiang_chizat import (
    JIANG_COMPLETEP_ADAM_THEORY,
    JIANG_COMPLETEP_ADAMW_THEORY,
    JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
    JiangChizatBlock,
    JiangChizatReference,
    JiangChizatShape,
    JiangChizatTransformer,
)
from .jiang_moe import (
    JIANG_MOE_ADAM_THEORY,
    JIANG_MOE_REPORTED_LR_MULTIPLIERS,
    JiangMoEBlock,
    JiangMoEReference,
    JiangMoEShape,
    JiangMoETransformer,
    jiang_moe_parameter_counts,
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
    _runtime_checkpoint_path,
    clear_runtime_checkpoint,
    close_distributed,
    load_runtime_checkpoint,
    prepare_distributed,
    synchronized_runtime_checkpoint_due,
    save_runtime_checkpoint,
    wrap_distributed_model,
)
from .scaling import fit_scaling_ensemble
from .schema import ArchitectureTemplate, ScaleLevel
from .study import atomic_write_json
from .tokenization import token_stream_identity


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
CAMPAIGN_SCHEMA_VERSION = 1


def forecast_tokenized_text_spec(config: Mapping[str, Any]) -> TokenizedTextSpec:
    """Translate the campaign dataset envelope to the strict loader schema."""

    payload = dict(config["dataset"])
    task_type = payload.pop("task_type", "tokenized_text")
    if task_type != "tokenized_text":
        raise ValueError("forecast dataset.task_type must be tokenized_text")
    return TokenizedTextSpec.from_dict(payload)


@dataclass(frozen=True)
class TheoryScale:
    name: str
    block_type: str
    target_parameters: int
    parameters: int
    non_embedding_parameters: int
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
    rho_lm_over_d: Optional[float]
    rho_relative_error: Optional[float]
    active_parameters: int
    active_non_embedding_parameters: int
    target_parameter_axis: str
    token_budget_parameter_axis: str
    tokens_per_active_parameter: float
    num_experts: Optional[int]
    active_experts: Optional[int]

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


def completep_parameter_count(
    *,
    vocab_size: int,
    context_length: int,
    depth: int,
    width: int,
    mlp_multiplier: int,
    position_encoding: str = "learned_absolute",
) -> int:
    if position_encoding not in {"alibi", "learned_absolute"}:
        raise ValueError("position_encoding must be alibi or learned_absolute")
    # Untied token embedding/readout, optional learned positions, two affine
    # LayerNorms per block, and biases on all hidden linear maps.
    per_block = (
        (4 + 2 * mlp_multiplier) * width * width
        + (9 + mlp_multiplier) * width
    )
    return int(
        2 * vocab_size * width
        + (context_length * width if position_encoding == "learned_absolute" else 0)
        + depth * per_block
        + 2 * width
    )


def jiang_non_embedding_parameter_count(
    *,
    vocab_size: int,
    context_length: int,
    depth: int,
    residual_width: int,
    hidden_width: int,
) -> int:
    return jiang_parameter_count(
        vocab_size=vocab_size,
        context_length=context_length,
        depth=depth,
        residual_width=residual_width,
        hidden_width=hidden_width,
    ) - (vocab_size + context_length) * residual_width


def completep_non_embedding_parameter_count(
    *,
    vocab_size: int,
    context_length: int,
    depth: int,
    width: int,
    mlp_multiplier: int,
    position_encoding: str = "learned_absolute",
) -> int:
    return completep_parameter_count(
        vocab_size=vocab_size,
        context_length=context_length,
        depth=depth,
        width=width,
        mlp_multiplier=mlp_multiplier,
        position_encoding=position_encoding,
    ) - (
        2 * vocab_size
        + (context_length if position_encoding == "learned_absolute" else 0)
    ) * width


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


def _weight_decay_tau_grid(optimizer: Mapping[str, Any]) -> Tuple[float, ...]:
    raw_grid = optimizer.get("weight_decay_tau_ema_grid")
    if raw_grid is None:
        return ()
    if "weight_decay_tau_ema" in optimizer:
        raise ValueError(
            "optimizer must not declare both weight_decay_tau_ema and "
            "weight_decay_tau_ema_grid"
        )
    if not isinstance(raw_grid, Sequence) or isinstance(raw_grid, (str, bytes)):
        raise ValueError("optimizer.weight_decay_tau_ema_grid must be an array")
    values = tuple(
        _positive_float(value, "optimizer.weight_decay_tau_ema_grid")
        for value in raw_grid
    )
    if len(values) < 3 or tuple(sorted(set(values))) != values:
        raise ValueError(
            "optimizer.weight_decay_tau_ema_grid must contain at least three "
            "unique increasing values"
        )
    return values


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _compile_batch_schedule_contract(
    config: Mapping[str, Any], context_length: int
) -> Optional[Dict[str, Any]]:
    raw = config.get("batch_schedule")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("batch_schedule must be an object")
    reference_batch = _positive_int(
        raw.get("reference_batch_examples"),
        "batch_schedule.reference_batch_examples",
    )
    microbatch = _positive_int(
        raw.get("microbatch_examples"), "batch_schedule.microbatch_examples"
    )
    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, Sequence) or isinstance(stages_raw, (str, bytes)):
        raise ValueError("batch_schedule.stages must be an array")
    stages = []
    for index, stage_raw in enumerate(stages_raw):
        if not isinstance(stage_raw, Mapping):
            raise ValueError("every batch schedule stage must be an object")
        start_tokens = int(stage_raw.get("start_tokens", -1))
        batch_examples = _positive_int(
            stage_raw.get("batch_examples"),
            "batch_schedule.stages.batch_examples",
        )
        if start_tokens < 0:
            raise ValueError("batch schedule start tokens must be non-negative")
        if batch_examples % microbatch:
            raise ValueError("every scheduled batch must be divisible by microbatch_examples")
        if index == 0 and start_tokens != 0:
            raise ValueError("the first batch schedule stage must start at token zero")
        if stages and start_tokens <= stages[-1]["start_tokens"]:
            raise ValueError("batch schedule stage boundaries must be strictly increasing")
        if stages:
            previous_tokens = int(stages[-1]["batch_examples"]) * context_length
            if start_tokens % previous_tokens:
                raise ValueError("stage boundaries must align with the preceding batch")
        if start_tokens % (batch_examples * context_length):
            raise ValueError("stage boundaries must align with the new batch")
        stages.append(
            {"start_tokens": start_tokens, "batch_examples": batch_examples}
        )
    if not stages:
        raise ValueError("batch_schedule.stages cannot be empty")
    initial_batch = _positive_int(config.get("batch_examples"), "batch_examples")
    if stages[0]["batch_examples"] != initial_batch:
        raise ValueError("batch_examples must equal the first scheduled batch")
    if any(
        stages[index]["batch_examples"] < stages[index - 1]["batch_examples"]
        for index in range(1, len(stages))
    ):
        raise ValueError("batch schedule must be monotone non-decreasing")
    if str(raw.get("learning_rate_rule")) != "adam_sqrt":
        raise ValueError("batch_schedule.learning_rate_rule must be adam_sqrt")
    if raw.get("uses_extrapolated_batch") is not False:
        raise ValueError("production batch schedules must refuse extrapolated batches")
    source_result = raw.get("source_critical_batch_result_sha256")
    if (
        not isinstance(source_result, str)
        or len(source_result) != 64
        or any(character not in "0123456789abcdef" for character in source_result)
    ):
        raise ValueError(
            "batch_schedule.source_critical_batch_result_sha256 must be a SHA-256 digest"
        )
    return {
        "reference_batch_examples": reference_batch,
        "microbatch_examples": microbatch,
        "learning_rate_rule": "adam_sqrt",
        "weight_decay_rule": "token_time_sqrt",
        "uses_extrapolated_batch": False,
        "source_critical_batch_result_sha256": source_result,
        "stages": stages,
    }


def _scheduled_token_geometry(
    requested_tokens: float,
    *,
    context_length: int,
    fixed_batch_examples: int,
    batch_schedule: Optional[Mapping[str, Any]],
) -> Tuple[int, int]:
    if batch_schedule is None:
        batch_tokens = fixed_batch_examples * context_length
        presented = max(
            batch_tokens, int(round(requested_tokens / batch_tokens)) * batch_tokens
        )
        return presented, presented // batch_tokens
    stages = list(batch_schedule["stages"])
    active_index = max(
        index
        for index, stage in enumerate(stages)
        if int(stage["start_tokens"]) <= requested_tokens
    )
    active = stages[active_index]
    active_start = int(active["start_tokens"])
    active_batch_tokens = int(active["batch_examples"]) * context_length
    presented = max(
        active_start + active_batch_tokens,
        active_start
        + int(round((requested_tokens - active_start) / active_batch_tokens))
        * active_batch_tokens,
    )
    steps = 0
    for index, stage in enumerate(stages[: active_index + 1]):
        start = int(stage["start_tokens"])
        stop = (
            int(stages[index + 1]["start_tokens"])
            if index < active_index
            else presented
        )
        batch_tokens = int(stage["batch_examples"]) * context_length
        if (stop - start) % batch_tokens:
            raise ValueError("compiled batch schedule segment is not update-aligned")
        steps += (stop - start) // batch_tokens
    return presented, steps


def _generate_ladder(
    config: Mapping[str, Any], identity: Mapping[str, Any]
) -> Tuple[List[TheoryScale], Dict[str, Any]]:
    architecture = config.get("architecture")
    ladder = config.get("ladder")
    if not isinstance(architecture, Mapping) or not isinstance(ladder, Mapping):
        raise ValueError("architecture and ladder must be objects")
    run_profile = str(config.get("run_profile", "forecast"))
    block_type = str(architecture.get("block_type"))
    if block_type not in {
        "completep_transformer",
        "jiang_chizat_transformer",
        "jiang_moe_transformer",
        "normalized_transformer",
    }:
        raise ValueError(
            "architecture.block_type must be completep_transformer, "
            "jiang_chizat_transformer, jiang_moe_transformer, or "
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
    minimum_scales = 2 if run_profile == "comparison" else 6
    if len(targets) < minimum_scales or tuple(sorted(set(targets))) != targets:
        raise ValueError(
            f"ladder.target_parameters must contain at least {minimum_scales} "
            "unique increasing values"
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
    minimum_fit_scales = 1 if run_profile == "comparison" else 5
    if heldout_count > len(targets) - minimum_fit_scales:
        raise ValueError(
            f"at least {minimum_fit_scales} non-held-out ladder scales are required"
        )
    has_tokens_per_parameter = "tokens_per_parameter" in ladder
    has_optimizer_steps = "optimizer_steps" in ladder
    if has_tokens_per_parameter == has_optimizer_steps:
        raise ValueError(
            "ladder must declare exactly one of tokens_per_parameter or optimizer_steps"
        )
    tokens_per_parameter = (
        _positive_float(ladder.get("tokens_per_parameter"), "tokens_per_parameter")
        if has_tokens_per_parameter
        else None
    )
    fixed_optimizer_steps = (
        _positive_int(ladder.get("optimizer_steps"), "ladder.optimizer_steps")
        if has_optimizer_steps
        else None
    )
    batch_examples = _positive_int(config.get("batch_examples"), "batch_examples")
    batch_tokens = batch_examples * context_length
    batch_schedule = _compile_batch_schedule_contract(config, context_length)
    if fixed_optimizer_steps is not None and batch_schedule is not None:
        raise ValueError("fixed-step ladders refuse variable batch schedules")
    unique_tokens = int(identity["training_tokens"])
    allowed_parameter_axes = {
        "parameters",
        "non_embedding_parameters",
        "active_parameters",
        "active_non_embedding_parameters",
    }
    target_parameter_axis = str(
        ladder.get("target_parameter_axis", "parameters")
    )
    token_budget_parameter_axis = str(
        ladder.get("token_budget_parameter_axis", "parameters")
    )
    if target_parameter_axis not in allowed_parameter_axes:
        raise ValueError(
            "ladder.target_parameter_axis must be a supported parameter count"
        )
    if token_budget_parameter_axis not in allowed_parameter_axes:
        raise ValueError(
            "ladder.token_budget_parameter_axis must be a supported parameter count"
        )
    if block_type != "jiang_moe_transformer" and (
        target_parameter_axis.startswith("active_")
        or token_budget_parameter_axis.startswith("active_")
    ):
        raise ValueError("active parameter axes are reserved for sparse MoE ladders")

    shapes: List[Dict[str, Any]] = []
    candidates = range(head_dimension, maximum_width + 1, head_dimension)
    if block_type == "completep_transformer":
        mlp_multiplier = _positive_int(
            architecture.get("mlp_multiplier"), "mlp_multiplier"
        )
        reference_width = _positive_int(
            architecture.get("reference_width"), "reference_width"
        )
        reference_depth = _positive_int(
            architecture.get("reference_depth"), "reference_depth"
        )
        position_encoding = str(
            architecture.get("position_encoding", "learned_absolute")
        )
        if position_encoding not in {"alibi", "learned_absolute"}:
            raise ValueError(
                "CompleteP architecture.position_encoding must be alibi or "
                "learned_absolute"
            )
        if reference_width % head_dimension:
            raise ValueError("reference_width must be divisible by head_dimension")
        for target, depth in zip(targets, depths):
            rows = [
                {
                    "target": target,
                    "depth": depth,
                    "width": width,
                    "hidden_width": mlp_multiplier * width,
                    "parameters": completep_parameter_count(
                        vocab_size=vocab_size,
                        context_length=context_length,
                        depth=depth,
                        width=width,
                        mlp_multiplier=mlp_multiplier,
                        position_encoding=position_encoding,
                    ),
                    "non_embedding_parameters": completep_non_embedding_parameter_count(
                        vocab_size=vocab_size,
                        context_length=context_length,
                        depth=depth,
                        width=width,
                        mlp_multiplier=mlp_multiplier,
                        position_encoding=position_encoding,
                    ),
                    "active_parameters": completep_parameter_count(
                        vocab_size=vocab_size,
                        context_length=context_length,
                        depth=depth,
                        width=width,
                        mlp_multiplier=mlp_multiplier,
                        position_encoding=position_encoding,
                    ),
                    "active_non_embedding_parameters": completep_non_embedding_parameter_count(
                        vocab_size=vocab_size,
                        context_length=context_length,
                        depth=depth,
                        width=width,
                        mlp_multiplier=mlp_multiplier,
                        position_encoding=position_encoding,
                    ),
                    "num_heads": width // head_dimension,
                    "rho_lm_over_d": None,
                    "rho_relative_error": None,
                }
                for width in candidates
            ]
            shapes.append(
                min(rows, key=lambda row: abs(row[target_parameter_axis] - target))
            )
        contract = {
            "parameterization": "completep_alpha_1_adamw",
            "theory": asdict(COMPLETEP_ADAMW_THEORY),
            "reference_width": reference_width,
            "reference_depth": reference_depth,
            "mlp_multiplier": mlp_multiplier,
            "tied_embeddings": False,
            "position_encoding": position_encoding,
            "activation": str(architecture.get("activation", "relu_squared")),
            "attention_scale": "QK^T/N",
            "residual_branch_scale": "(L/L0)^(-1)",
            "unembedding_forward_scale": "(N/N0)^(-1)",
            "hidden_initialization_std": "sigma0 * (N/N0)^(-1/2)",
            "layer_norm_numerical_epsilon": 1e-5,
        }
    elif block_type == "jiang_chizat_transformer":
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
        rho_tolerance = float(ladder.get("maximum_rho_relative_error", 0.25))
        if not 0.0 <= rho_tolerance < 1.0:
            raise ValueError("maximum_rho_relative_error must be in [0,1)")
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
                non_embedding_parameters = jiang_non_embedding_parameter_count(
                    vocab_size=vocab_size,
                    context_length=context_length,
                    depth=depth,
                    residual_width=residual_width,
                    hidden_width=hidden_width,
                )
                actual_rho = depth * hidden_width / residual_width
                rows.append(
                    {
                        "target": target,
                        "depth": depth,
                        "width": residual_width,
                        "hidden_width": hidden_width,
                        "parameters": parameters,
                        "non_embedding_parameters": non_embedding_parameters,
                        "active_parameters": parameters,
                        "active_non_embedding_parameters": non_embedding_parameters,
                        "num_heads": residual_width // head_dimension,
                        "rho_lm_over_d": actual_rho,
                        "rho_relative_error": abs(actual_rho / rho - 1.0),
                    }
                )
            shapes.append(
                min(rows, key=lambda row: abs(row[target_parameter_axis] - target))
            )
        optimizer_payload = config.get("optimizer", {})
        optimizer_name = (
            str(optimizer_payload.get("name", "adam"))
            if isinstance(optimizer_payload, Mapping)
            else "adam"
        )
        jiang_theory = (
            JIANG_COMPLETEP_ADAMW_THEORY
            if optimizer_name == "adamw"
            else JIANG_COMPLETEP_ADAM_THEORY
        )
        contract = {
            "parameterization": f"jiang_completep_{optimizer_name}",
            "theory": asdict(jiang_theory),
            "rho_lm_over_d": rho,
            "maximum_rho_relative_error": rho_tolerance,
            "tied_embeddings": True,
            "attention_scale": "QK^T/d_head",
            "residual_branch_scale": "1/L",
            "unembedding_forward_scale": "(D/D0)^(-1)",
            "layer_norm_numerical_epsilon": 1e-5,
        }
    elif block_type == "jiang_moe_transformer":
        reference_depth = _positive_int(
            architecture.get("reference_depth"), "reference_depth"
        )
        reference_hidden = _positive_int(
            architecture.get("reference_hidden_width"), "reference_hidden_width"
        )
        reference_residual = _positive_int(
            architecture.get("reference_residual_width"),
            "reference_residual_width",
        )
        reference_num_experts = _positive_int(
            architecture.get("reference_num_experts"), "reference_num_experts"
        )
        reference_active_experts = _positive_int(
            architecture.get("reference_active_experts"),
            "reference_active_experts",
        )
        if reference_active_experts > reference_num_experts:
            raise ValueError("reference_active_experts cannot exceed reference_num_experts")
        default_num_experts = _positive_int(
            architecture.get("num_experts", reference_num_experts), "num_experts"
        )
        default_active_experts = _positive_int(
            architecture.get("active_experts", reference_active_experts),
            "active_experts",
        )
        expert_counts = tuple(
            _positive_int(value, "ladder.num_experts")
            for value in ladder.get("num_experts", [default_num_experts] * len(targets))
        )
        active_counts = tuple(
            _positive_int(value, "ladder.active_experts")
            for value in ladder.get(
                "active_experts", [default_active_experts] * len(targets)
            )
        )
        if len(expert_counts) != len(targets) or len(active_counts) != len(targets):
            raise ValueError(
                "ladder.num_experts and ladder.active_experts must match target_parameters"
            )
        reference_sparsity = reference_active_experts / reference_num_experts
        for experts, active in zip(expert_counts, active_counts):
            if active > experts or not math.isclose(
                active / experts, reference_sparsity, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "Jiang MoE scaling requires exact fixed A/E sparsity"
                )
        hidden_multiple = _positive_int(
            int(ladder.get("hidden_width_multiple", head_dimension)),
            "hidden_width_multiple",
        )
        raw_residual_widths = ladder.get("residual_widths")
        raw_expert_widths = ladder.get("expert_widths")
        if (raw_residual_widths is None) != (raw_expert_widths is None):
            raise ValueError(
                "Jiang MoE exact geometry requires both residual_widths and "
                "expert_widths"
            )
        exact_residual_widths: Optional[Tuple[int, ...]] = None
        exact_expert_widths: Optional[Tuple[int, ...]] = None
        if raw_residual_widths is not None and raw_expert_widths is not None:
            exact_residual_widths = tuple(
                _positive_int(value, "ladder.residual_widths")
                for value in raw_residual_widths
            )
            exact_expert_widths = tuple(
                _positive_int(value, "ladder.expert_widths")
                for value in raw_expert_widths
            )
            if (
                len(exact_residual_widths) != len(targets)
                or len(exact_expert_widths) != len(targets)
            ):
                raise ValueError(
                    "ladder residual_widths and expert_widths must match "
                    "target_parameters"
                )
            if any(width % head_dimension for width in exact_residual_widths):
                raise ValueError(
                    "every Jiang MoE residual width must be divisible by head_dimension"
                )
            if any(width % hidden_multiple for width in exact_expert_widths):
                raise ValueError(
                    "every Jiang MoE expert width must satisfy hidden_width_multiple"
                )
        raw_ffn_ratios = ladder.get("ffn_ratios")
        if raw_ffn_ratios is None:
            ffn_ratios = (reference_hidden / reference_residual,) * len(targets)
        else:
            ffn_ratios = tuple(
                _positive_float(value, "ladder.ffn_ratios")
                for value in raw_ffn_ratios
            )
            if len(ffn_ratios) != len(targets):
                raise ValueError("ladder.ffn_ratios must match target_parameters")

        # Constant L*M/D is a useful separately declared shape ablation, but it
        # is not a condition of Jiang et al.'s sparse-MoE transfer theorem.
        # The source result scales D, L, alpha=M/D, and E independently at fixed
        # kappa=A/E.  Never impose the dense hybrid's rho constraint silently.
        enforce_moe_rho = "rho_lm_over_d" in ladder
        moe_rho: Optional[float] = None
        moe_rho_tolerance: Optional[float] = None
        if enforce_moe_rho:
            moe_rho = _positive_float(
                ladder["rho_lm_over_d"], "ladder.rho_lm_over_d"
            )
            moe_rho_tolerance = float(
                ladder.get("maximum_rho_relative_error", 0.25)
            )
            if not 0.0 <= moe_rho_tolerance < 1.0:
                raise ValueError("maximum_rho_relative_error must be in [0,1)")

        for scale_index, (target, depth, experts, active, ffn_ratio) in enumerate(
            zip(targets, depths, expert_counts, active_counts, ffn_ratios)
        ):
            rows = []
            scale_residual_candidates = (
                (exact_residual_widths[scale_index],)
                if exact_residual_widths is not None
                else candidates
            )
            for residual_width in scale_residual_candidates:
                if exact_expert_widths is not None:
                    hidden_width = exact_expert_widths[scale_index]
                elif enforce_moe_rho:
                    assert moe_rho is not None
                    hidden_width = _nearest_multiple(
                        moe_rho * residual_width / depth, hidden_multiple
                    )
                else:
                    hidden_width = _nearest_multiple(
                        ffn_ratio * residual_width, hidden_multiple
                    )
                counts = jiang_moe_parameter_counts(
                    vocab_size=vocab_size,
                    context_length=context_length,
                    depth=depth,
                    residual_width=residual_width,
                    expert_width=hidden_width,
                    num_experts=experts,
                    active_experts=active,
                )
                actual_rho = depth * hidden_width / residual_width
                rho_relative_error = (
                    abs(actual_rho / moe_rho - 1.0)
                    if moe_rho is not None
                    else None
                )
                rows.append(
                    {
                        "target": target,
                        "depth": depth,
                        "width": residual_width,
                        "hidden_width": hidden_width,
                        **{
                            key: counts[key]
                            for key in (
                                "parameters",
                                "non_embedding_parameters",
                                "active_parameters",
                                "active_non_embedding_parameters",
                            )
                        },
                        "num_heads": residual_width // head_dimension,
                        "num_experts": experts,
                        "active_experts": active,
                        "rho_lm_over_d": actual_rho,
                        "rho_relative_error": rho_relative_error,
                    }
                )
            shapes.append(
                min(rows, key=lambda row: abs(row[target_parameter_axis] - target))
            )
        router_gamma = _positive_float(
            architecture.get("router_gamma", 1.0), "architecture.router_gamma"
        )
        if router_gamma < 0.5:
            raise ValueError("architecture.router_gamma must be at least 1/2")
        if not math.isclose(router_gamma, 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "the source-faithful Jiang main-experiment path requires "
                "router_gamma=1"
            )
        initialization_std = _positive_float(
            architecture.get("initialization_std", 0.02),
            "architecture.initialization_std",
        )
        contract = {
            "parameterization": "jiang_moe_completep_adam_table2",
            "theory": asdict(JIANG_MOE_ADAM_THEORY),
            "independently_scalable_dimensions": [
                "residual_width_D",
                "depth_L",
                "expert_width_M",
                "expert_count_E",
            ],
            "rho_lm_over_d_is_not_a_source_transfer_invariant": True,
            "optional_declared_rho_lm_over_d": moe_rho,
            "optional_maximum_rho_relative_error": moe_rho_tolerance,
            "fixed_active_expert_fraction": reference_sparsity,
            "reference_num_experts": reference_num_experts,
            "reference_active_experts": reference_active_experts,
            "router_gamma": router_gamma,
            "reference_initialization_std": initialization_std,
            "embedding_initialization_std": "sigma0",
            "router_initialization_std": (
                "sigma0 * (D/D0)^(-gamma); main-paper gamma=1"
            ),
            "attention_qko_initialization_std": "sigma0 * (D/D0)^(-1/2)",
            "expert_up_initialization_std": "sigma0 * (D/D0)^(-1/2)",
            "expert_down_initialization_std": (
                "sigma0 * (D/D0)^(-1/2) * "
                "(alpha_ffn/alpha_ffn0)^(-1) * 1/4"
            ),
            "value_initialization_multiplier": 1.0 / 16.0,
            "tied_embeddings": True,
            "learned_absolute_positions": True,
            "activation": "gelu",
            "attention_scale": "QK^T/d_head",
            "residual_branch_scale": "1/L",
            "moe_mixing": "sigmoid-weighted hard top-A divided by A",
            "load_balancing": "manual expert bias; no auxiliary loss",
            "unembedding_forward_scale": "(D/D0)^(-1)",
            "layer_norm_numerical_epsilon": 1e-5,
            "parameter_reporting": {
                "total": "shared + all expert banks",
                "active_per_token": "shared + A selected expert banks",
                "router_and_all_routing_biases_are_active": True,
            },
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
                    "non_embedding_parameters": nugpt_parameter_count(
                        vocab_size=vocab_size,
                        depth=depth,
                        width=width,
                        mlp_multiplier=mlp_multiplier,
                    ) - (2 * vocab_size * width + vocab_size),
                    "active_parameters": nugpt_parameter_count(
                        vocab_size=vocab_size,
                        depth=depth,
                        width=width,
                        mlp_multiplier=mlp_multiplier,
                    ),
                    "active_non_embedding_parameters": nugpt_parameter_count(
                        vocab_size=vocab_size,
                        depth=depth,
                        width=width,
                        mlp_multiplier=mlp_multiplier,
                    ) - (2 * vocab_size * width + vocab_size),
                    "num_heads": width // head_dimension,
                    "rho_lm_over_d": None,
                    "rho_relative_error": None,
                }
                for width in candidates
            ]
            shapes.append(
                min(rows, key=lambda row: abs(row[target_parameter_axis] - target))
            )
        contract = {
            "parameterization": "nugpt_mid_alignment",
            "theory": asdict(NUGPT_ADAM_THEORY),
            "tied_embeddings": False,
            "matrix_constraint": "post-step unit-sphere projection",
        }

    scales: List[TheoryScale] = []
    for index, row in enumerate(shapes):
        parameters = int(row["parameters"])
        active_parameters = int(row["active_parameters"])
        relative_error = abs(
            int(row[target_parameter_axis]) / int(row["target"]) - 1.0
        )
        if relative_error > tolerance:
            raise ValueError(
                f"no width <= {maximum_width} places S{index + 1} within the "
                f"declared parameter tolerance"
            )
        if block_type == "jiang_chizat_transformer" and float(
            row["rho_relative_error"]
        ) > rho_tolerance:
            raise ValueError(
                f"S{index + 1} violates the declared L*M/D tolerance"
            )
        if (
            block_type == "jiang_moe_transformer"
            and row["rho_relative_error"] is not None
            and moe_rho_tolerance is not None
            and float(row["rho_relative_error"]) > moe_rho_tolerance
        ):
            raise ValueError(
                f"S{index + 1} violates the optional declared L*M/D ablation tolerance"
            )
        if fixed_optimizer_steps is not None:
            optimizer_steps = fixed_optimizer_steps
            presented_tokens = optimizer_steps * batch_tokens
        else:
            assert tokens_per_parameter is not None
            requested_tokens = tokens_per_parameter * int(
                row[token_budget_parameter_axis]
            )
            presented_tokens, optimizer_steps = _scheduled_token_geometry(
                requested_tokens,
                context_length=context_length,
                fixed_batch_examples=batch_examples,
                batch_schedule=batch_schedule,
            )
        scales.append(
            TheoryScale(
                name=f"S{index + 1}",
                block_type=block_type,
                target_parameters=int(row["target"]),
                parameters=parameters,
                non_embedding_parameters=int(row["non_embedding_parameters"]),
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
                optimizer_steps=optimizer_steps,
                tokens_per_parameter=presented_tokens / parameters,
                repetition_ratio=presented_tokens / unique_tokens,
                iteration_ratio=1.0,
                heldout=index >= len(shapes) - heldout_count,
                rho_lm_over_d=(
                    float(row["rho_lm_over_d"])
                    if row["rho_lm_over_d"] is not None
                    else None
                ),
                rho_relative_error=(
                    float(row["rho_relative_error"])
                    if row["rho_relative_error"] is not None
                    else None
                ),
                active_parameters=active_parameters,
                active_non_embedding_parameters=int(
                    row["active_non_embedding_parameters"]
                ),
                target_parameter_axis=target_parameter_axis,
                token_budget_parameter_axis=token_budget_parameter_axis,
                tokens_per_active_parameter=(
                    presented_tokens / active_parameters
                ),
                num_experts=(
                    int(row["num_experts"])
                    if row.get("num_experts") is not None
                    else None
                ),
                active_experts=(
                    int(row["active_experts"])
                    if row.get("active_experts") is not None
                    else None
                ),
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
    receipt_path = dataset.get("token_stream_verification_receipt_path")
    if receipt_path is None:
        identity = token_stream_identity(Path(manifest_path))
    else:
        identity = token_stream_identity(
            Path(manifest_path),
            verification_receipt_path=Path(str(receipt_path)),
        )
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
    if block_type == "jiang_moe_transformer" and runtime.distributed == "fsdp":
        raise ValueError(
            "Jiang MoE currently requires DDP or one GPU: its manually updated "
            "routing-bias parameter is replicated and synchronized after each "
            "global batch, not FSDP-sharded"
        )
    head_dimension = int(config["architecture"]["head_dimension"])
    if runtime.attention_backend == "flash" and head_dimension % 8:
        raise ValueError("FlashAttention requires head_dimension divisible by eight")
    data_parallel_microbatches = (
        runtime.num_processes * runtime.gradient_accumulation_steps
    )
    batch_examples = _positive_int(config.get("batch_examples"), "batch_examples")
    batch_schedule = _compile_batch_schedule_contract(
        config, int(config["architecture"]["context_length"])
    )
    if batch_schedule is not None and (
        runtime.distributed != "none" or runtime.num_processes != 1
    ):
        raise ValueError(
            "variable-batch forecast workers must be independent one-GPU processes"
        )
    if batch_schedule is None and batch_examples % data_parallel_microbatches:
        raise ValueError(
            "batch_examples must be divisible by num_processes times "
            "gradient_accumulation_steps"
        )
    validation_examples = _positive_int(
        int(config.get("validation_examples", 256)), "validation_examples"
    )
    validation_seed = int(config.get("validation_seed", 900_001))
    if validation_seed < 0:
        raise ValueError("validation_seed must be non-negative")
    if validation_examples % runtime.num_processes:
        raise ValueError("validation_examples must be divisible by num_processes")
    validation_interval_steps = _positive_int(
        config.get("validation_interval_steps"), "validation_interval_steps"
    )
    validation_microbatch_examples = _positive_int(
        int(config.get("validation_microbatch_examples", batch_examples)),
        "validation_microbatch_examples",
    )
    if validation_microbatch_examples > validation_examples // runtime.num_processes:
        raise ValueError(
            "validation_microbatch_examples cannot exceed per-process validation examples"
        )
    optimizer = config.get("optimizer")
    allowed_optimizers = (
        {"adamw"}
        if block_type == "completep_transformer"
        else {"adam", "adamw"}
        if block_type == "jiang_chizat_transformer"
        else {"adam"}
        if block_type == "jiang_moe_transformer"
        else {"adam"}
    )
    if not isinstance(optimizer, Mapping) or optimizer.get("name") not in allowed_optimizers:
        raise ValueError(
            f"{block_type} theory scaling requires one of: "
            + ", ".join(sorted(name.upper() for name in allowed_optimizers))
        )
    rates = tuple(
        _positive_float(value, "optimizer.learning_rates")
        for value in optimizer.get("learning_rates", ())
    )
    if len(rates) < 3 or tuple(sorted(set(rates))) != rates:
        raise ValueError(
            "optimizer.learning_rates must contain at least three increasing values"
        )
    weight_decay_tau_grid = _weight_decay_tau_grid(optimizer)
    include_zero_weight_decay = optimizer.get(
        "include_zero_weight_decay_control", False
    )
    if not isinstance(include_zero_weight_decay, bool):
        raise ValueError("optimizer.include_zero_weight_decay_control must be boolean")
    if include_zero_weight_decay and not weight_decay_tau_grid:
        raise ValueError(
            "include_zero_weight_decay_control requires a finite tau_EMA grid"
        )
    if weight_decay_tau_grid and optimizer.get("name") != "adamw":
        raise ValueError("tau_EMA weight-decay tuning requires AdamW")
    if weight_decay_tau_grid and float(optimizer.get("weight_decay", 0.0)) != 0.0:
        raise ValueError(
            "weight_decay must be zero or omitted when weight_decay_tau_ema_grid "
            "is declared"
        )
    if "weight_decay_tau_ema" in optimizer:
        _positive_float(
            optimizer["weight_decay_tau_ema"],
            "optimizer.weight_decay_tau_ema",
        )
    if optimizer.get("name") == "adam" and float(optimizer.get("weight_decay", 0.0)) != 0.0:
        raise ValueError("forecast Adam uses zero weight decay; use AdamW for decay")
    if block_type == "jiang_moe_transformer":
        if not math.isclose(
            float(optimizer.get("beta1", 0.9)), 0.9, rel_tol=0.0, abs_tol=0.0
        ) or not math.isclose(
            float(optimizer.get("beta2", 0.95)), 0.95, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("Jiang MoE requires Adam betas (0.9, 0.95)")
        if not math.isclose(
            float(optimizer.get("epsilon", 1e-12)),
            1e-12,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("Jiang MoE requires base Adam epsilon 1e-12")
        configured_multipliers = dict(
            optimizer.get(
                "learning_rate_multipliers", JIANG_MOE_REPORTED_LR_MULTIPLIERS
            )
        )
        if set(configured_multipliers) != set(JIANG_MOE_REPORTED_LR_MULTIPLIERS) or any(
            not math.isclose(
                float(configured_multipliers[name]),
                expected,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for name, expected in JIANG_MOE_REPORTED_LR_MULTIPLIERS.items()
        ):
            raise ValueError(
                "Jiang MoE production requires the exact Appendix-D.1 LR "
                "multipliers; tune a separately preregistered reference contract "
                "before changing them"
            )
        _positive_float(
            optimizer.get("expert_bias_learning_rate"),
            "optimizer.expert_bias_learning_rate",
        )
        if weight_decay_tau_grid or include_zero_weight_decay:
            raise ValueError("the Jiang main-paper MoE contract uses Adam with zero decay")
    frozen_optimizer_contract: Optional[Dict[str, Any]] = None
    raw_frozen_optimizer = config.get("frozen_optimizer")
    if raw_frozen_optimizer is not None:
        if not isinstance(raw_frozen_optimizer, Mapping):
            raise ValueError("frozen_optimizer must be an object")
        required_frozen_fields = {
            "selected_learning_rate",
            "selected_weight_decay_tau_ema",
            "source_critical_batch_result_sha256",
            "source_pilot_selection_sha256",
            "source_optimum_is_interior",
            "adaptive_followup",
        }
        if set(raw_frozen_optimizer) != required_frozen_fields:
            raise ValueError(
                "frozen_optimizer must contain exactly: "
                + ", ".join(sorted(required_frozen_fields))
            )
        frozen_optimizer_contract = json.loads(
            json.dumps(dict(raw_frozen_optimizer), sort_keys=True)
        )
        _positive_float(
            frozen_optimizer_contract["selected_learning_rate"],
            "frozen_optimizer.selected_learning_rate",
        )
        frozen_tau = frozen_optimizer_contract["selected_weight_decay_tau_ema"]
        if frozen_tau is not None:
            _positive_float(
                frozen_tau, "frozen_optimizer.selected_weight_decay_tau_ema"
            )
        for name in (
            "source_critical_batch_result_sha256",
            "source_pilot_selection_sha256",
        ):
            digest = frozen_optimizer_contract[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"frozen_optimizer.{name} must be a SHA-256 digest")
        if frozen_optimizer_contract["source_optimum_is_interior"] is not True:
            raise ValueError("frozen optimizer requires an interior source optimum")
        if frozen_optimizer_contract["adaptive_followup"] is not True:
            raise ValueError("frozen optimizer must declare the adaptive follow-up")
        if batch_schedule is None:
            raise ValueError("frozen critical-batch optimizer requires batch_schedule")
    optimizer_contract = json.loads(json.dumps(dict(optimizer), sort_keys=True))
    fused_optimizer = optimizer_contract.get("fused", False)
    if not isinstance(fused_optimizer, bool):
        raise ValueError("optimizer.fused must be boolean")
    if fused_optimizer and runtime.precision != "bf16":
        raise ValueError("the fused Adam/AdamW forecast path requires bf16")
    seeds = tuple(int(value) for value in config.get("seeds", (11, 29, 47)))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    run_profile = str(config.get("run_profile", "forecast"))
    if run_profile not in {
        "smoke",
        "forecast",
        "extension",
        "comparison",
        "fixed_budget_scan",
    }:
        raise ValueError(
            "run_profile must be smoke, forecast, extension, comparison, or "
            "fixed_budget_scan"
        )
    exploratory_single_seed = config.get("exploratory_single_seed", False)
    if not isinstance(exploratory_single_seed, bool):
        raise ValueError("exploratory_single_seed must be boolean")
    if run_profile in {"forecast", "comparison", "fixed_budget_scan"} and len(seeds) < 3:
        if not (
            run_profile == "forecast"
            and exploratory_single_seed is True
            and len(seeds) == 1
        ):
            raise ValueError(f"{run_profile} profile requires at least three seeds")
    base_rates = rates
    tuning_task_rates = rates
    learning_rate_refinement: Optional[Dict[str, Any]] = None
    raw_learning_rate_refinement = optimizer.get("learning_rate_refinement")
    if raw_learning_rate_refinement is not None:
        if run_profile != "fixed_budget_scan":
            raise ValueError(
                "optimizer.learning_rate_refinement requires fixed_budget_scan"
            )
        if not isinstance(raw_learning_rate_refinement, Mapping):
            raise ValueError("optimizer.learning_rate_refinement must be an object")
        required_refinement_fields = {
            "learning_rates",
            "seeds",
            "exploratory_single_seed",
        }
        if set(raw_learning_rate_refinement) != required_refinement_fields:
            raise ValueError(
                "optimizer.learning_rate_refinement must contain exactly: "
                + ", ".join(sorted(required_refinement_fields))
            )
        refinement_rates = tuple(
            _positive_float(
                value,
                "optimizer.learning_rate_refinement.learning_rates",
            )
            for value in raw_learning_rate_refinement["learning_rates"]
        )
        if (
            not refinement_rates
            or tuple(sorted(set(refinement_rates))) != refinement_rates
            or set(refinement_rates).intersection(base_rates)
        ):
            raise ValueError(
                "learning-rate refinement values must be non-empty, increasing, "
                "and disjoint from the base grid"
            )
        refinement_seeds = tuple(
            int(value) for value in raw_learning_rate_refinement["seeds"]
        )
        if (
            len(refinement_seeds) != 1
            or refinement_seeds[0] not in seeds
            or raw_learning_rate_refinement["exploratory_single_seed"] is not True
        ):
            raise ValueError(
                "learning-rate refinement must explicitly use one campaign seed "
                "and declare exploratory_single_seed=true"
            )
        if weight_decay_tau_grid or include_zero_weight_decay:
            raise ValueError(
                "single-seed LR refinement is only supported for a fixed decay choice"
            )
        tuning_task_rates = (*base_rates, *refinement_rates)
        rates = tuple(sorted(tuning_task_rates))
        learning_rate_refinement = {
            "mode": "exploratory_single_seed_lr_refinement",
            "base_learning_rates": list(base_rates),
            "refinement_learning_rates": list(refinement_rates),
            "refinement_seeds": list(refinement_seeds),
            "tuning_task_learning_rates": list(tuning_task_rates),
            "unequal_seed_counts_are_explicit": True,
        }
    if run_profile == "fixed_budget_scan":
        if "optimizer_steps" not in config["ladder"]:
            raise ValueError("fixed_budget_scan requires ladder.optimizer_steps")
        if bool(config.get("run_negative_control", True)):
            raise ValueError("fixed_budget_scan refuses a wrong-LR control")
    if run_profile == "comparison":
        if block_type != "completep_transformer":
            raise ValueError("comparison profile is reserved for completep_transformer")
        if bool(config.get("run_negative_control", True)):
            raise ValueError("comparison profile refuses a wrong-LR control")
    extension_contract: Optional[Dict[str, Any]] = None
    raw_extension_contract = config.get("extension_contract")
    if frozen_optimizer_contract is not None:
        tuning_trials = 0
        scale_trials = (
            len(seeds)
            if run_profile == "comparison"
            else len(scales) * len(seeds)
        )
        negative_control_trials = 0
        execution_order = [
            "verify_frozen_horizon_safe_optimizer_and_critical_batch_evidence",
            "refuse_reference_retuning_after_large_scale_failure",
            "apply_frozen_eta_tau_and_batch_schedule_to_every_scale",
            "evaluate_scaling_law_and_hidden_upper_rung",
        ]
    elif run_profile == "extension":
        if len(seeds) != 1:
            raise ValueError("extension profile requires exactly one seed")
        if bool(config.get("run_negative_control", True)):
            raise ValueError("extension profile refuses a wrong-LR control")
        if not isinstance(raw_extension_contract, Mapping):
            raise ValueError("extension profile requires extension_contract")
        required_extension_fields = {
            "parent_plan_fingerprint",
            "parent_dataset_fingerprint",
            "parent_aggregate_sha256",
            "selected_learning_rate",
            "target_scale",
            "target_seed",
            "expected_target_parameters",
        }
        if set(raw_extension_contract) != required_extension_fields:
            raise ValueError(
                "extension_contract must contain exactly: "
                + ", ".join(sorted(required_extension_fields))
            )
        extension_contract = json.loads(
            json.dumps(dict(raw_extension_contract), sort_keys=True)
        )
        for name in (
            "parent_plan_fingerprint",
            "parent_dataset_fingerprint",
            "parent_aggregate_sha256",
        ):
            digest = extension_contract[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"extension_contract.{name} must be a SHA-256 digest")
        selected_extension_rate = _positive_float(
            extension_contract["selected_learning_rate"],
            "extension_contract.selected_learning_rate",
        )
        if selected_extension_rate not in rates:
            raise ValueError(
                "extension selected_learning_rate is not in the frozen LR grid"
            )
        if extension_contract["target_scale"] != scales[-1].name:
            raise ValueError("extension target_scale must be the largest ladder scale")
        if int(extension_contract["target_seed"]) != seeds[0]:
            raise ValueError("extension target_seed must equal the sole campaign seed")
        if int(extension_contract["expected_target_parameters"]) != scales[-1].parameters:
            raise ValueError(
                "extension expected_target_parameters disagrees with compiled geometry"
            )
    elif raw_extension_contract is not None:
        raise ValueError("extension_contract requires run_profile=extension")
    comparison_contract: Optional[Dict[str, Any]] = None
    raw_comparison_contract = config.get("comparison_contract")
    if run_profile == "comparison":
        if not isinstance(raw_comparison_contract, Mapping):
            raise ValueError("comparison profile requires comparison_contract")
        required_comparison_fields = {
            "baseline_plan_fingerprint",
            "baseline_aggregate_sha256",
            "baseline_dataset_fingerprint",
            "baseline_tokenizer_fingerprint",
            "baseline_architecture",
            "baseline_parameters",
            "baseline_mean_validation_loss",
            "baseline_seed_losses",
        }
        if set(raw_comparison_contract) != required_comparison_fields:
            raise ValueError(
                "comparison_contract must contain exactly: "
                + ", ".join(sorted(required_comparison_fields))
            )
        comparison_contract = json.loads(
            json.dumps(dict(raw_comparison_contract), sort_keys=True)
        )
        for name in (
            "baseline_plan_fingerprint",
            "baseline_aggregate_sha256",
            "baseline_dataset_fingerprint",
            "baseline_tokenizer_fingerprint",
        ):
            digest = comparison_contract[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"comparison_contract.{name} must be a SHA-256 digest")
        if comparison_contract["baseline_dataset_fingerprint"] != identity["fingerprint"]:
            raise ValueError("comparison baseline uses a different token-stream dataset")
        if (
            comparison_contract["baseline_tokenizer_fingerprint"]
            != identity["tokenizer_fingerprint"]
        ):
            raise ValueError("comparison baseline uses a different tokenizer")
        _positive_int(
            comparison_contract["baseline_parameters"],
            "comparison_contract.baseline_parameters",
        )
        _positive_float(
            comparison_contract["baseline_mean_validation_loss"],
            "comparison_contract.baseline_mean_validation_loss",
        )
        baseline_seed_losses = tuple(
            _positive_float(value, "comparison_contract.baseline_seed_losses")
            for value in comparison_contract["baseline_seed_losses"]
        )
        if len(baseline_seed_losses) != len(seeds):
            raise ValueError(
                "comparison baseline and CompleteP target must use the same seed count"
            )
    elif raw_comparison_contract is not None:
        raise ValueError("comparison_contract requires run_profile=comparison")
    schedule = LearningRateSchedule.from_payload(config.get("schedule", "cosine_to_10_percent"))
    ladder = config["ladder"]
    target_forecasts = tuple(
        _positive_int(value, "target_forecasts", 32)
        for value in ladder.get("target_forecasts", ())
    )
    if run_profile in {"comparison", "fixed_budget_scan"}:
        if target_forecasts:
            raise ValueError(
                f"{run_profile} profile does not issue extrapolative forecasts"
            )
    elif run_profile in {"extension", "smoke"} and not target_forecasts:
        # An extension evaluates a forecast frozen by its parent plan, and a
        # runtime smoke canary evaluates no scientific forecast at all. Neither
        # needs to issue another extrapolation beyond the compiled endpoint.
        pass
    else:
        if not target_forecasts or tuple(sorted(set(target_forecasts))) != target_forecasts:
            raise ValueError("ladder.target_forecasts must be unique and increasing")
        # Forecast targets live on the declared fit axis, not necessarily total
        # parameters (MoEs report total and active axes separately).
    fit_parameter_axis = str(
        ladder.get(
            "fit_parameter_axis",
            "non_embedding_parameters"
            if run_profile == "fixed_budget_scan"
            else "parameters",
        )
    )
    if fit_parameter_axis not in {
        "parameters",
        "non_embedding_parameters",
        "active_parameters",
        "active_non_embedding_parameters",
    }:
        raise ValueError(
            "ladder.fit_parameter_axis must be a supported total or active "
            "parameter count"
        )
    if block_type != "jiang_moe_transformer" and fit_parameter_axis.startswith("active_"):
        raise ValueError("active fit axes are reserved for sparse MoE ladders")
    required_fixed_budget_axis = (
        "active_non_embedding_parameters"
        if block_type == "jiang_moe_transformer"
        else "non_embedding_parameters"
    )
    if run_profile == "fixed_budget_scan" and fit_parameter_axis != required_fixed_budget_axis:
        raise ValueError(
            f"fixed_budget_scan requires {required_fixed_budget_axis} as the primary fit axis"
        )
    minimum_span = _positive_float(
        ladder.get("minimum_parameter_span", 1.0 if run_profile == "comparison" else 30.0),
        "minimum_parameter_span",
    )
    fit_scales = [row for row in scales if not row.heldout]
    observed_span = float(getattr(fit_scales[-1], fit_parameter_axis)) / float(
        getattr(fit_scales[0], fit_parameter_axis)
    )
    if target_forecasts and target_forecasts[0] <= getattr(
        scales[-1], fit_parameter_axis
    ):
        raise ValueError("every target forecast must exceed the largest ladder fit scale")
    maximum_repetition = _positive_float(
        ladder.get("maximum_repetition_ratio", 1.0), "maximum_repetition_ratio"
    )
    maximum_extrapolation = _positive_float(
        ladder.get("maximum_extrapolation_factor", 10.0),
        "maximum_extrapolation_factor",
    )
    require_gate_eligible_plan = ladder.get("require_gate_eligible_plan", False)
    if not isinstance(require_gate_eligible_plan, bool):
        raise ValueError("require_gate_eligible_plan must be boolean")
    if require_gate_eligible_plan:
        if observed_span < minimum_span:
            raise ValueError(
                "forecast plan is guaranteed to fail its minimum parameter span gate"
            )
        if max(row.repetition_ratio for row in scales) > maximum_repetition:
            raise ValueError(
                "forecast plan is guaranteed to fail its corpus repetition gate"
            )
        if target_forecasts and target_forecasts[-1] / getattr(
            scales[-1], fit_parameter_axis
        ) > maximum_extrapolation:
            raise ValueError(
                "forecast plan is guaranteed to fail its extrapolation gate"
            )
    if run_profile == "extension":
        tuning_trials = 0
        scale_trials = 1
        negative_control_trials = 0
        execution_order = [
            "verify_parent_evidence_and_frozen_learning_rate",
            "verify_tokenizer_training_stream_and_identical_validation_split",
            "compile_exact_constant_tpp_target_geometry",
            "preregister_parent_fit_prediction_before_reveal",
            "train_one_theory_scaled_target_seed",
            "evaluate_preregistered_prediction_without_retuning",
        ]
    elif run_profile == "comparison":
        tuning_trials = len(rates) * max(1, len(weight_decay_tau_grid)) * len(seeds)
        scale_trials = len(seeds)
        negative_control_trials = 0
        execution_order = [
            "verify_identical_tokenizer_training_stream_and_validation_split",
            "recall_completep_table1_and_tau_ema_before_training",
            "jointly_tune_eta_and_tau_ema_at_declared_L0_N0_reference",
            "require_interior_reference_optimum_in_both_coordinates",
            "freeze_eta_tau_ema_and_apply_all_completep_group_rules",
            "train_three_matched_100m_target_seeds",
            "compare_loss_to_frozen_jiang_chizat_baseline",
        ] if weight_decay_tau_grid else [
            "verify_identical_tokenizer_training_stream_and_validation_split",
            "recall_completep_table1_before_training",
            "tune_eta_at_declared_L0_N0_reference",
            "require_interior_reference_optimum",
            "freeze_eta_and_apply_all_completep_group_rules",
            "train_three_matched_100m_target_seeds",
            "compare_loss_to_frozen_jiang_chizat_baseline",
        ]
    elif run_profile == "fixed_budget_scan":
        tuning_seed_trials = len(base_rates) * len(seeds)
        if learning_rate_refinement is not None:
            tuning_seed_trials += len(
                learning_rate_refinement["refinement_learning_rates"]
            ) * len(learning_rate_refinement["refinement_seeds"])
        tuning_trials = tuning_seed_trials * max(1, len(weight_decay_tau_grid))
        if include_zero_weight_decay:
            tuning_trials += tuning_seed_trials
        scale_trials = (len(scales) - 1) * len(seeds)
        negative_control_trials = 0
        execution_order = [
            "verify_immutable_tokenizer_training_stream_and_fixed_validation_windows",
            "compile_fixed_global_batch_optimizer_steps_and_presented_tokens",
            "audit_every_trainable_tensor_lr_epsilon_decay_and_forward_multiplier",
            "tune_reference_eta_and_decay_at_the_exact_ladder_budget",
            "require_an_interior_eta_and_valid_finite_or_zero_decay_optimum",
            "freeze_optimizer_schedule_batch_and_paired_training_seeds",
            "train_nonheldout_scales_and_reuse_selected_reference_trials",
            "predict_the_hidden_upper_rung_in_nonembedding_parameter_coordinates",
            "report_total_and_nonembedding_parameter_scaling_fits",
        ]
    else:
        tuning_trials = len(rates) * max(1, len(weight_decay_tau_grid)) * len(seeds)
        scale_trials = len(scales) * len(seeds)
        negative_control_trials = (
            len(seeds) if bool(config.get("run_negative_control", True)) else 0
        )
        execution_order = [
            "verify_tokenizer_and_token_stream",
            "compile_exact_vocab_aware_ladder",
            "recall_architecture_specific_parameter_group_rules",
            (
                "jointly_tune_reference_eta_and_tau_ema"
                if weight_decay_tau_grid
                else "tune_reference_scale"
            ),
            (
                "freeze_learning_rate_tau_ema_and_training_path"
                if weight_decay_tau_grid
                else "freeze_learning_rate_and_training_path"
            ),
            "train_nonheldout_scales",
            "evaluate_hidden_upper_rungs",
            "run_wrong_global_learning_rate_control",
            "rolling_scaling_law_backtests",
            "issue_or_refuse_bounded_forecasts",
        ]
    plan_payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign": (
            "real_text_100m_completep_comparison"
            if run_profile == "comparison"
            else "real_text_fixed_budget_scaling_scan"
            if run_profile == "fixed_budget_scan"
            else "real_text_scaling_ladder"
        ),
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
        "exploratory_single_seed": exploratory_single_seed,
        "measurement_contract": {
            "validation_examples": validation_examples,
            "validation_seed": validation_seed,
            "validation_windows_are_identical_across_trials": True,
            "validation_interval_steps": validation_interval_steps,
            "validation_microbatch_examples": validation_microbatch_examples,
        },
        "scales": [row.to_dict() for row in scales],
        "fit_parameter_span": observed_span,
        "fit_parameter_axis": fit_parameter_axis,
        "minimum_parameter_span": minimum_span,
        "maximum_repetition_ratio": maximum_repetition,
        "maximum_extrapolation_factor": maximum_extrapolation,
        "require_gate_eligible_plan": require_gate_eligible_plan,
        "target_forecasts": list(target_forecasts),
        "tuning_trials": tuning_trials,
        "scale_trials": scale_trials,
        "negative_control_trials": negative_control_trials,
        "planned_grid_trials": (
            tuning_trials + scale_trials + negative_control_trials
        ),
        "execution_order": execution_order,
    }
    if runtime.retained_checkpoint_tokens_per_parameter:
        retained_scales: Dict[str, Any] = {}
        for scale in scales:
            parameter_axis = scale.token_budget_parameter_axis
            parameter_count = int(getattr(scale, parameter_axis))
            checkpoints = []
            prior_step = 0
            for tokens_per_parameter in (
                runtime.retained_checkpoint_tokens_per_parameter
            ):
                presented_tokens, optimizer_step = _scheduled_token_geometry(
                    tokens_per_parameter * parameter_count,
                    context_length=int(config["architecture"]["context_length"]),
                    fixed_batch_examples=batch_examples,
                    batch_schedule=batch_schedule,
                )
                if presented_tokens > scale.presented_tokens:
                    raise ValueError(
                        "a retained checkpoint exceeds the compiled training horizon "
                        f"for {scale.name}"
                    )
                if optimizer_step <= prior_step:
                    raise ValueError(
                        "retained checkpoint TPP coordinates collapse to the same "
                        f"optimizer step for {scale.name}"
                    )
                checkpoints.append(
                    {
                        "requested_tokens_per_parameter": tokens_per_parameter,
                        "parameter_axis": parameter_axis,
                        "parameter_count": parameter_count,
                        "optimizer_step": optimizer_step,
                        "presented_tokens": presented_tokens,
                        "effective_tokens_per_parameter": (
                            presented_tokens / parameter_count
                        ),
                    }
                )
                prior_step = optimizer_step
            retained_scales[scale.name] = checkpoints
        plan_payload["retained_checkpoint_contract"] = {
            "schema_version": 1,
            "state": "model_optimizer_and_sampling_generator",
            "coordinates": "presented_tokens_per_token_budget_parameter_axis",
            "survives_successful_trial_cleanup": True,
            "scales": retained_scales,
        }
        plan_payload["execution_order"] = [
            "compile_update_aligned_retained_horizon_checkpoints",
            "retain_full_model_optimizer_and_sampling_state_at_each_horizon",
            *plan_payload["execution_order"],
        ]
    if batch_schedule is not None:
        plan_payload["batch_schedule"] = batch_schedule
        plan_payload["execution_order"] = [
            "verify_critical_batch_result_and_measured_lower_bounds",
            "freeze_power_of_two_batch_warmup_in_token_coordinates",
            "apply_adam_sqrt_batch_factor_to_every_theory_parameter_group",
            "preserve_tau_ema_weight_decay_per_presented_token",
            *plan_payload["execution_order"],
        ]
    if run_profile == "fixed_budget_scan":
        plan_payload["fixed_budget_contract"] = {
            "batch_examples": batch_examples,
            "batch_tokens": batch_examples
            * int(config["architecture"]["context_length"]),
            "optimizer_steps": int(scales[0].optimizer_steps),
            "presented_tokens": int(scales[0].presented_tokens),
            "identical_at_every_scale": all(
                row.optimizer_steps == scales[0].optimizer_steps
                and row.presented_tokens == scales[0].presented_tokens
                for row in scales
            ),
            "batch_learning_rate_scaling": "none; eta tuned at this exact batch",
        }
    if learning_rate_refinement is not None:
        parent_payload = deepcopy(plan_payload)
        parent_payload["learning_rates"] = list(base_rates)
        parent_payload["optimizer_contract"].pop("learning_rate_refinement")
        parent_tuning_trials = (
            len(base_rates)
            * max(1, len(weight_decay_tau_grid))
            * len(seeds)
        )
        if include_zero_weight_decay:
            parent_tuning_trials += len(base_rates) * len(seeds)
        parent_payload["tuning_trials"] = parent_tuning_trials
        parent_payload["planned_grid_trials"] = (
            parent_tuning_trials
            + int(parent_payload["scale_trials"])
            + int(parent_payload["negative_control_trials"])
        )
        learning_rate_refinement["inherited_reference_plan_fingerprint"] = sha256(
            json.dumps(
                parent_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        plan_payload["learning_rate_refinement"] = learning_rate_refinement
        plan_payload["tuning_task_learning_rates"] = list(tuning_task_rates)
    if extension_contract is not None:
        plan_payload["extension_contract"] = extension_contract
    if comparison_contract is not None:
        plan_payload["comparison_contract"] = comparison_contract
    if weight_decay_tau_grid:
        plan_payload["weight_decay_tau_ema_grid"] = list(weight_decay_tau_grid)
    if frozen_optimizer_contract is not None:
        plan_payload["frozen_optimizer"] = frozen_optimizer_contract
    plan_payload["fingerprint"] = sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return plan_payload


def bind_real_text_scaling_config(
    config: Mapping[str, Any], manifest_path: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Bind a forecast template to one verified, immutable token stream.

    Compilation is deliberately part of binding: a deployable config is never
    written unless tokenizer provenance, vocabulary, corpus size, ladder
    geometry, runtime, and repetition limits all pass the production gates.
    """

    resolved_manifest = manifest_path.expanduser().resolve()
    if not resolved_manifest.is_file():
        raise ValueError(f"token stream manifest does not exist: {resolved_manifest}")
    bound = deepcopy(dict(config))
    dataset = bound.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    dataset["token_stream_manifest_path"] = str(resolved_manifest)
    plan = compile_real_text_scaling_plan(bound)
    summary = {
        "schema_version": 1,
        "status": "bound",
        "token_stream_manifest_path": str(resolved_manifest),
        "plan_fingerprint": plan["fingerprint"],
        "dataset_identity": plan["dataset_identity"],
        "architecture_contract": plan["architecture_contract"],
        "scales": plan["scales"],
        "planned_grid_trials": plan["planned_grid_trials"],
        "tuning_trials": plan["tuning_trials"],
        "scale_trials": plan["scale_trials"],
        "negative_control_trials": plan["negative_control_trials"],
    }
    return bound, summary


def _sample_rank_partitioned_batch(
    corpus: TokenizedTextCorpus,
    split: str,
    local_examples: int,
    generator: torch.Generator,
    context: DistributedContext,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw one global batch identically, then give each DDP rank its slice.

    Replicating the lightweight memory-mapped draw makes one-GPU and DDP runs
    consume the same examples in the same order. This is intentionally more
    conservative than independent rank RNG streams because topology must not
    silently change the experiment.
    """

    global_examples = local_examples * context.world_size
    inputs, targets = corpus.sample_batch(
        split, global_examples, generator, context.device
    )
    if context.world_size == 1:
        return inputs, targets
    start = context.rank * local_examples
    stop = start + local_examples
    return inputs[start:stop], targets[start:stop]


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    corpus: TokenizedTextCorpus,
    *,
    vocab_size: int,
    validation_examples: int,
    validation_microbatch_examples: int,
    seed: int,
    runtime: PretrainingRuntimeSpec,
    context: DistributedContext,
) -> float:
    model.eval()
    local_examples = validation_examples // context.world_size
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loss_sum = torch.zeros((), dtype=torch.float64, device=context.device)
    token_count = torch.zeros((), dtype=torch.float64, device=context.device)
    remaining = local_examples
    while remaining:
        current = min(validation_microbatch_examples, remaining)
        inputs, targets = _sample_rank_partitioned_batch(
            corpus, "validation", current, generator, context
        )
        with _autocast(runtime, context.device):
            logits = model(inputs)
            batch_loss = F.cross_entropy(
                logits.float().reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
        loss_sum += batch_loss.double()
        token_count += targets.numel()
        remaining -= current
    if context.world_size > 1:
        torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(token_count, op=torch.distributed.ReduceOp.SUM)
    result = float((loss_sum / token_count).cpu())
    model.train()
    return result


def _build_model_and_groups(
    *,
    config: Mapping[str, Any],
    scale: Mapping[str, Any],
    eta: float,
    weight_decay_tau_ema: Optional[float],
    optimizer_mode: str,
    runtime: PretrainingRuntimeSpec,
    context: DistributedContext,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, List[Dict[str, Any]], Dict[str, Any]]:
    architecture = dict(config["architecture"])
    optimizer_payload = dict(config["optimizer"])
    block_type = str(architecture["block_type"])
    capture_diagnostics = runtime.attention_backend == "math"
    if block_type == "completep_transformer":
        shape = CompletePShape(
            depth=int(scale["depth"]),
            width=int(scale["width"]),
            head_dimension=int(architecture["head_dimension"]),
            mlp_multiplier=int(architecture["mlp_multiplier"]),
        )
        reference = CompletePReference(
            depth=int(architecture["reference_depth"]),
            width=int(architecture["reference_width"]),
        )
        position_encoding = str(
            architecture.get("position_encoding", "learned_absolute")
        )
        plain_model = CompletePTransformer(
            shape,
            vocab_size=int(architecture["vocab_size"]),
            context_length=int(architecture["context_length"]),
            reference=reference,
            initialization_std=float(architecture.get("initialization_std", 0.02)),
            activation=str(architecture.get("activation", "relu_squared")),  # type: ignore[arg-type]
            position_encoding=position_encoding,  # type: ignore[arg-type]
            attention_backend=runtime.attention_backend,
            activation_checkpointing=runtime.activation_checkpointing,
            capture_attention_diagnostics=capture_diagnostics,
        ).to(context.device)
        tau_ema = (
            weight_decay_tau_ema
            if weight_decay_tau_ema is not None
            else optimizer_payload.get("weight_decay_tau_ema")
        )
        if tau_ema is not None:
            tau = _positive_float(tau_ema, "optimizer.weight_decay_tau_ema")
            weight_decay0 = 1.0 / (tau * eta * int(scale["optimizer_steps"]))
        else:
            weight_decay0 = float(optimizer_payload.get("weight_decay", 0.0))
        groups = plain_model.optimizer_parameter_groups(
            eta,
            epsilon0=float(optimizer_payload.get("epsilon", 1e-16)),
            weight_decay0=weight_decay0,
        )
        group_audit = plain_model.optimizer_contract_audit(
            eta,
            epsilon0=float(optimizer_payload.get("epsilon", 1e-16)),
            weight_decay0=weight_decay0,
        )
        group_audit = {
            **group_audit,
            "base_weight_decay": weight_decay0,
            "weight_decay_tau_ema": tau_ema,
            "weight_decay_step_count": int(scale["optimizer_steps"]),
            "weight_decay_timescale_formula": (
                "lambda0 = 1 / (tau_EMA * eta_base * n_steps)"
            ),
            "weight_decay_schedule_assumption": (
                "tau_EMA omits schedule integration; transfer requires the same "
                "schedule family at every scale"
            ),
        }
        block_types: Sequence[type[nn.Module]] = (CompletePBlock,)
    elif block_type == "jiang_chizat_transformer":
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
        optimizer_name = str(optimizer_payload.get("name", "adam"))
        tau_ema = (
            weight_decay_tau_ema
            if weight_decay_tau_ema is not None
            else optimizer_payload.get("weight_decay_tau_ema")
        )
        if tau_ema is not None:
            tau = _positive_float(tau_ema, "optimizer.weight_decay_tau_ema")
            weight_decay0 = 1.0 / (tau * eta * int(scale["optimizer_steps"]))
        else:
            weight_decay0 = float(optimizer_payload.get("weight_decay", 0.0))
        groups = plain_model.optimizer_parameter_groups(
            eta,
            epsilon0=float(optimizer_payload.get("epsilon", 1e-12)),
            weight_decay0=weight_decay0,
            optimizer_name=optimizer_name,  # type: ignore[arg-type]
            learning_rate_multipliers=multipliers,
        )
        group_audit = plain_model.optimizer_contract_audit(
            eta,
            epsilon0=float(optimizer_payload.get("epsilon", 1e-12)),
            weight_decay0=weight_decay0,
            optimizer_name=optimizer_name,  # type: ignore[arg-type]
            learning_rate_multipliers=multipliers,
        )
        group_audit = {
            **group_audit,
            "base_weight_decay": weight_decay0,
            "weight_decay_tau_ema": tau_ema,
            "weight_decay_step_count": int(scale["optimizer_steps"]),
            "weight_decay_timescale_formula": (
                "lambda0 = 1 / (tau_EMA * eta_base * n_steps)"
            ),
            "weight_decay_schedule_assumption": (
                "tau_EMA omits schedule integration; transfer requires the same "
                "schedule family at every scale"
            ),
        }
        block_types: Sequence[type[nn.Module]] = (JiangChizatBlock,)
    elif block_type == "jiang_moe_transformer":
        shape = JiangMoEShape(
            depth=int(scale["depth"]),
            residual_width=int(scale["width"]),
            expert_width=int(scale["hidden_width"]),
            head_dimension=int(architecture["head_dimension"]),
            num_experts=int(scale["num_experts"]),
            active_experts=int(scale["active_experts"]),
        )
        reference = JiangMoEReference(
            depth=int(architecture["reference_depth"]),
            residual_width=int(architecture["reference_residual_width"]),
            expert_width=int(architecture["reference_hidden_width"]),
            num_experts=int(architecture["reference_num_experts"]),
            active_experts=int(architecture["reference_active_experts"]),
        )
        plain_model = JiangMoETransformer(
            shape,
            vocab_size=int(architecture["vocab_size"]),
            context_length=int(architecture["context_length"]),
            reference=reference,
            initialization_std=float(architecture.get("initialization_std", 0.02)),
            router_gamma=float(architecture.get("router_gamma", 1.0)),
            attention_backend=runtime.attention_backend,
            activation_checkpointing=runtime.activation_checkpointing,
            capture_attention_diagnostics=capture_diagnostics,
        ).to(context.device)
        assert isinstance(plain_model, JiangMoETransformer)
        multipliers = dict(
            optimizer_payload.get(
                "learning_rate_multipliers", JIANG_MOE_REPORTED_LR_MULTIPLIERS
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
        expert_bias_learning_rate = _positive_float(
            optimizer_payload["expert_bias_learning_rate"],
            "optimizer.expert_bias_learning_rate",
        )
        group_audit = {
            **group_audit,
            "base_weight_decay": 0.0,
            "weight_decay_tau_ema": None,
            "manual_expert_bias": plain_model.manual_parameter_contract(
                expert_bias_learning_rate
            ),
            "parameter_accounting": plain_model.parameter_accounting(),
            "initialization_contract": plain_model.initialization_contract(),
            "source_constant_initialization_multipliers": {
                "attention_value": 1.0 / 16.0,
                "expert_down": 1.0 / 4.0,
            },
            "router_gamma": float(architecture.get("router_gamma", 1.0)),
        }
        block_types = (JiangMoEBlock,)
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
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "weight_decay_formula": str(
                    group.get("weight_decay_formula", "0")
                ),
                "scale_factors": dict(group["scale_factors"]),
                "theory_contract_id": str(group["theory_contract_id"]),
            }
            for group in groups
        ]
    else:
        raise ValueError("optimizer_mode must be theory or wrong_global")
    fused = bool(optimizer_payload.get("fused", False))
    if fused and not context.device.startswith("cuda"):
        raise ValueError("fused Adam/AdamW requires CUDA")
    optimizer_class = (
        torch.optim.AdamW
        if optimizer_payload.get("name") == "adamw"
        else torch.optim.Adam
    )
    optimizer = optimizer_class(
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
    optimizer_name = str(optimizer_payload.get("name", "adam"))
    group_audit = {
        **group_audit,
        "optimizer_backend": (
            f"fused_{optimizer_name}" if fused else optimizer_name
        ),
    }
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
    weight_decay_tau_ema: Optional[float] = None,
    seed: int,
    optimizer_mode: str,
) -> Tuple[str, str]:
    """Return the immutable fingerprint and filename stem for one trial."""

    schedule = LearningRateSchedule.from_payload(config["schedule"])
    identity_plan_fingerprint = plan["fingerprint"]
    refinement = plan.get("learning_rate_refinement")
    if isinstance(refinement, Mapping):
        reference_index = int(
            plan["architecture_contract"]["reference_scale_index"]
        )
        reference_name = str(plan["scales"][reference_index]["name"])
        if (
            str(scale["name"]) == reference_name
            and float(eta)
            in {float(value) for value in refinement["base_learning_rates"]}
            and int(seed) in {int(value) for value in plan["seeds"]}
            and optimizer_mode == "theory"
            and weight_decay_tau_ema is None
        ):
            identity_plan_fingerprint = refinement[
                "inherited_reference_plan_fingerprint"
            ]
    identity = {
        "schema_version": 1,
        "plan_fingerprint": identity_plan_fingerprint,
        "scale": dict(scale),
        "dataset_fingerprint": dataset_fingerprint,
        "runtime": asdict(runtime),
        "eta": eta,
        "seed": seed,
        "optimizer_mode": optimizer_mode,
        "schedule": schedule.audit(int(scale["optimizer_steps"])),
    }
    if weight_decay_tau_ema is not None:
        identity["weight_decay_tau_ema"] = float(weight_decay_tau_ema)
    identity_fingerprint = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    tau_label = (
        f"-tau{weight_decay_tau_ema:g}"
        if weight_decay_tau_ema is not None
        else ""
    )
    run_id = (
        f"forecast-{scale['name']}-{optimizer_mode}-eta{eta:g}{tau_label}-"
        f"s{seed}-{identity_fingerprint[:12]}"
    )
    return identity_fingerprint, run_id


def _retained_checkpoint_base_path(
    cache_directory: Path,
    run_id: str,
    tokens_per_parameter: float,
) -> Path:
    label = f"{tokens_per_parameter:g}".replace(".", "p")
    # The checkpoint writer replaces the final suffix with .pt (or a sharded
    # rank suffix). Keep a dedicated terminal suffix so decimal eta/TPP labels
    # elsewhere in the run id can never be mistaken for it.
    return cache_directory / f"{run_id}.horizon-{label}tpp.retained"


def _run_trial(
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    scale: Mapping[str, Any],
    corpus: TokenizedTextCorpus,
    runtime: PretrainingRuntimeSpec,
    context: DistributedContext,
    eta: float,
    weight_decay_tau_ema: Optional[float] = None,
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
        weight_decay_tau_ema=weight_decay_tau_ema,
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
            weight_decay_tau_ema=weight_decay_tau_ema,
            optimizer_mode=optimizer_mode,
            runtime=runtime,
            context=context,
        )
    )
    peak_rates = [float(group["lr"]) for group in optimizer.param_groups]
    peak_weight_decays = [
        float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups
    ]
    batch_examples = int(config["batch_examples"])
    context_length = int(config["architecture"]["context_length"])
    batch_schedule = plan.get("batch_schedule")
    update_batch_examples: List[int] = []
    if isinstance(batch_schedule, Mapping):
        stages = list(batch_schedule["stages"])
        presented_tokens = int(scale["presented_tokens"])
        for index, stage in enumerate(stages):
            start_tokens = int(stage["start_tokens"])
            if start_tokens >= presented_tokens:
                break
            stop_tokens = min(
                presented_tokens,
                int(stages[index + 1]["start_tokens"])
                if index + 1 < len(stages)
                else presented_tokens,
            )
            current_batch = int(stage["batch_examples"])
            current_batch_tokens = current_batch * context_length
            if (stop_tokens - start_tokens) % current_batch_tokens:
                raise RuntimeError("compiled variable-batch segment is not update-aligned")
            update_batch_examples.extend(
                [current_batch]
                * ((stop_tokens - start_tokens) // current_batch_tokens)
            )
        if len(update_batch_examples) != int(scale["optimizer_steps"]):
            raise RuntimeError("compiled variable-batch update count is inconsistent")
        reference_batch_examples = int(batch_schedule["reference_batch_examples"])
        microbatch_examples = int(batch_schedule["microbatch_examples"])
        reference_steps = int(scale["presented_tokens"]) // (
            reference_batch_examples * context_length
        )
        if weight_decay_tau_ema is not None:
            # _build_model_and_groups used the actual (variable-batch) update
            # count. Convert lambda0 back to the reference-batch token-time
            # definition before applying the per-stage sqrt(B/B_ref) factor.
            correction = int(scale["optimizer_steps"]) / reference_steps
            peak_weight_decays = [value * correction for value in peak_weight_decays]
    else:
        update_batch_examples = [batch_examples] * int(scale["optimizer_steps"])
        reference_batch_examples = batch_examples
        microbatch_examples = batch_examples // (
            context.world_size * runtime.gradient_accumulation_steps
        )
    cumulative_tokens = [0]
    for current_batch in update_batch_examples:
        cumulative_tokens.append(
            cumulative_tokens[-1] + current_batch * context_length
        )
    if cumulative_tokens[-1] != int(scale["presented_tokens"]):
        raise RuntimeError("compiled update batches do not reach presented_tokens")
    steps = int(scale["optimizer_steps"])
    retained_checkpoint_contract = plan.get("retained_checkpoint_contract")
    retained_checkpoints: Dict[int, Dict[str, Any]] = {}
    if isinstance(retained_checkpoint_contract, Mapping):
        scale_contract = retained_checkpoint_contract["scales"].get(
            str(scale["name"]), []
        )
        retained_checkpoints = {
            int(row["optimizer_step"]): dict(row) for row in scale_contract
        }
    generator = torch.Generator(device="cpu").manual_seed(100_003 + seed)
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
            validation_microbatch_examples=int(
                config.get("validation_microbatch_examples", config["batch_examples"])
            ),
            seed=int(config.get("validation_seed", 900_001)),
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
        resumed_tokens = int(
            resumed["extra"].get("tokens_seen", cumulative_tokens[start_step])
        )
        if resumed_tokens != cumulative_tokens[start_step]:
            raise ValueError("runtime checkpoint token coordinate is inconsistent")
    for retained_step, retained in retained_checkpoints.items():
        if retained_step <= start_step:
            retained_base = _retained_checkpoint_base_path(
                cache_directory,
                run_id,
                float(retained["requested_tokens_per_parameter"]),
            )
            retained_path = _runtime_checkpoint_path(
                retained_base, context, runtime
            )
            retained_exists = retained_path.is_file()
            if context.world_size > 1:
                marker = torch.tensor(
                    1 if retained_exists else 0,
                    dtype=torch.int32,
                    device=context.device,
                )
                torch.distributed.all_reduce(
                    marker, op=torch.distributed.ReduceOp.MIN
                )
                retained_exists = bool(marker.item())
            if not retained_exists:
                raise ValueError(
                    "runtime resume is beyond a missing retained horizon checkpoint: "
                    f"{retained_base}"
                )
    validation_interval = _positive_int(
        int(config.get("validation_interval_steps", max(1, steps // 8))),
        "validation_interval_steps",
    )
    started = time.monotonic()
    last_checkpoint_at = started
    model.train()
    for step in range(start_step + 1, steps + 1):
        current_batch_examples = update_batch_examples[step - 1]
        current_batch_tokens = current_batch_examples * context_length
        tokens_before_update = cumulative_tokens[step - 1]
        batch_lr_multiplier = math.sqrt(
            current_batch_examples / reference_batch_examples
        )
        multiplier = (
            schedule.multiplier_for_token_update(
                tokens_before_update=tokens_before_update,
                batch_tokens=current_batch_tokens,
                total_tokens=int(scale["presented_tokens"]),
            )
            if batch_schedule is not None
            else schedule.multiplier(step, steps)
        )
        for group, peak_rate, peak_weight_decay in zip(
            optimizer.param_groups, peak_rates, peak_weight_decays
        ):
            group["lr"] = peak_rate * batch_lr_multiplier * multiplier
            group["weight_decay"] = peak_weight_decay * batch_lr_multiplier
        accumulation_steps = current_batch_examples // (
            context.world_size * microbatch_examples
        )
        if accumulation_steps <= 0 or (
            current_batch_examples
            % (context.world_size * microbatch_examples)
        ):
            raise RuntimeError("scheduled batch is not divisible by data-parallel microbatches")
        optimizer.zero_grad(set_to_none=True)
        if isinstance(plain_model, JiangMoETransformer):
            plain_model.begin_routing_measurement()
        for accumulation_index in range(accumulation_steps):
            inputs, targets = _sample_rank_partitioned_batch(
                corpus,
                "train",
                microbatch_examples,
                generator,
                context,
            )
            synchronization = (
                model.no_sync()  # type: ignore[attr-defined]
                if context.world_size > 1
                and accumulation_index + 1 < accumulation_steps
                else nullcontext()
            )
            with synchronization, _autocast(runtime, context.device):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.float().reshape(
                        -1, int(config["architecture"]["vocab_size"])
                    ),
                    targets.reshape(-1),
                ) / accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError("theory-faithful pretraining trial diverged")
            loss.backward()
        optimizer.step()
        if isinstance(plain_model, NormalizedTransformer):
            plain_model.project_normalized_weights()
        elif isinstance(plain_model, JiangMoETransformer):
            plain_model.update_expert_biases(
                float(config["optimizer"]["expert_bias_learning_rate"]),
                synchronize=context.world_size > 1,
            )
        if (
            step % validation_interval == 0
            or step == steps
            or step in retained_checkpoints
        ):
            validation_loss = _evaluate(
                model,
                corpus,
                vocab_size=int(config["architecture"]["vocab_size"]),
                validation_examples=int(config.get("validation_examples", 256)),
                validation_microbatch_examples=int(
                    config.get(
                        "validation_microbatch_examples", config["batch_examples"]
                    )
                ),
                seed=int(config.get("validation_seed", 900_001)),
                runtime=runtime,
                context=context,
            )
            checkpoints.append(
                {
                    "step": float(step),
                    "tokens": float(cumulative_tokens[step]),
                    "validation_loss": validation_loss,
                }
            )
        checkpoint_now = time.monotonic()
        if synchronized_runtime_checkpoint_due(
            runtime,
            context,
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
                    "tokens_seen": cumulative_tokens[step],
                    "elapsed_seconds": elapsed_before_resume
                    + time.monotonic()
                    - started,
                },
            )
            last_checkpoint_at = checkpoint_now
        retained = retained_checkpoints.get(step)
        if retained is not None:
            retained_base = _retained_checkpoint_base_path(
                cache_directory,
                run_id,
                float(retained["requested_tokens_per_parameter"]),
            )
            save_runtime_checkpoint(
                base_path=retained_base,
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
                    "tokens_seen": cumulative_tokens[step],
                    "elapsed_seconds": elapsed_before_resume
                    + time.monotonic()
                    - started,
                    "retained_checkpoint_contract": retained,
                },
            )
    duration = elapsed_before_resume + time.monotonic() - started
    final_loss = float(checkpoints[-1]["validation_loss"])
    diagnostics: Dict[str, Any] = {}
    if isinstance(plain_model, NormalizedTransformer):
        diagnostics = plain_model.sphere_diagnostics()
    elif isinstance(plain_model, JiangChizatTransformer):
        diagnostics = plain_model.diagnostics()
    elif isinstance(plain_model, JiangMoETransformer):
        diagnostics = plain_model.routing_diagnostics()
    elif isinstance(plain_model, CompletePTransformer):
        diagnostics = plain_model.diagnostics()
    record: Optional[BatchRunRecord] = None
    if context.is_primary:
        record = BatchRunRecord(
            run_id=run_id,
            model_family=(
                "completep_real_text_comparison"
                if isinstance(plain_model, CompletePTransformer)
                else "nugpt_real_text_scaling"
                if isinstance(plain_model, NormalizedTransformer)
                else "jiang_moe_real_text_scaling"
                if isinstance(plain_model, JiangMoETransformer)
                else "jiang_chizat_real_text_scaling"
            ),
            optimizer=OptimizerHyperparameters(
                name=str(config["optimizer"].get("name", "adam")),
                learning_rate=eta,
                beta1=float(config["optimizer"].get("beta1", 0.9)),
                beta2=float(config["optimizer"].get("beta2", 0.95)),
                epsilon=float(config["optimizer"].get("epsilon", 1e-12)),
                weight_decay=float(group_audit.get("base_weight_decay", 0.0)),
            ),
            seed=seed,
            parameter_count=int(scale["parameters"]),
            width=int(scale["width"]),
            depth=int(scale["depth"]),
            total_tokens=int(scale["presented_tokens"]),
            batch_tokens=(
                max(update_batch_examples) * context_length
            ),
            microbatch_tokens=(
                microbatch_examples * context_length
            ),
            accumulation_steps=max(
                batch // (context.world_size * microbatch_examples)
                for batch in update_batch_examples
            ),
            data_parallel_replicas=context.world_size,
            optimizer_steps=steps,
            nonpadding_tokens_seen=int(scale["presented_tokens"]),
            learning_rate_schedule=schedule.name,
            final_validation_loss=final_loss,
            estimated_flops=float(
                6
                * int(scale.get("active_parameters", scale["parameters"]))
                * int(scale["presented_tokens"])
            ),
            wall_time_seconds=duration,
            validation_checkpoints=tuple(checkpoints),
            metadata={
                "scale": dict(scale),
                "dataset_fingerprint": corpus.identity_fingerprint,
                "tokenizer_fingerprint": corpus.tokenizer_fingerprint,
                "tokenizer_is_pinned": corpus.tokenizer_is_pinned,
                "optimizer_mode": optimizer_mode,
                "parameter_accounting": {
                    "total_parameters": int(scale["parameters"]),
                    "active_parameters_per_token": int(
                        scale.get("active_parameters", scale["parameters"])
                    ),
                    "total_non_embedding_parameters": int(
                        scale["non_embedding_parameters"]
                    ),
                    "active_non_embedding_parameters_per_token": int(
                        scale.get(
                            "active_non_embedding_parameters",
                            scale["non_embedding_parameters"],
                        )
                    ),
                    "tokens_per_total_parameter": float(
                        scale["tokens_per_parameter"]
                    ),
                    "tokens_per_active_parameter": float(
                        scale.get(
                            "tokens_per_active_parameter",
                            scale["tokens_per_parameter"],
                        )
                    ),
                },
                "weight_decay_tau_ema": weight_decay_tau_ema,
                "peak_parameter_group_contract": group_contract,
                "optimizer_group_audit": group_audit,
                "software_contract": {
                    "torch_version": torch.__version__,
                    "cuda_runtime_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "optimizer_backend": group_audit["optimizer_backend"],
                    "adam_epsilon_placement": "sqrt(v_hat) + epsilon",
                    "model_parameter_dtype": str(
                        next(plain_model.parameters()).dtype
                    ),
                    "autocast_precision": runtime.precision,
                },
                "batch_schedule": (
                    dict(batch_schedule) if isinstance(batch_schedule, Mapping) else None
                ),
                "batch_schedule_trace": (
                    [
                        {
                            "start_tokens": int(stage["start_tokens"]),
                            "batch_examples": int(stage["batch_examples"]),
                            "batch_tokens": int(stage["batch_examples"]) * context_length,
                            "learning_rate_multiplier_from_reference": math.sqrt(
                                int(stage["batch_examples"])
                                / reference_batch_examples
                            ),
                        }
                        for stage in batch_schedule["stages"]
                        if int(stage["start_tokens"]) < int(scale["presented_tokens"])
                    ]
                    if isinstance(batch_schedule, Mapping)
                    else []
                ),
                "gradient_clipping": "none_source_faithful",
                "activation_checkpointing": runtime.activation_checkpointing,
                "sampling_contract": "replicated_global_draw_rank_partition_v1",
                "resumed_from_step": start_step,
                "retained_checkpoints": [
                    {
                        **row,
                        "base_path": str(
                            _retained_checkpoint_base_path(
                                cache_directory,
                                run_id,
                                float(row["requested_tokens_per_parameter"]),
                            )
                        ),
                    }
                    for _, row in sorted(retained_checkpoints.items())
                ],
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
    clear_runtime_checkpoint(resume_base, context, runtime)
    return _broadcast_record(record, context)


def _mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    mean = float(np.mean(values))
    sem = (
        float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return mean, sem


def _tuning_seeds_for_learning_rate(
    plan: Mapping[str, Any], eta: float
) -> List[int]:
    refinement = plan.get("learning_rate_refinement")
    if isinstance(refinement, Mapping) and float(eta) in {
        float(value) for value in refinement["refinement_learning_rates"]
    }:
        return [int(value) for value in refinement["refinement_seeds"]]
    return [int(value) for value in plan["seeds"]]


def _selection_seeds_for_tuning(plan: Mapping[str, Any]) -> List[int]:
    """Return the matched seeds used to compare every tuning cell."""

    refinement = plan.get("learning_rate_refinement")
    if isinstance(refinement, Mapping):
        return [int(value) for value in refinement["refinement_seeds"]]
    return [int(value) for value in plan["seeds"]]


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
    if plan["run_profile"] == "comparison":
        raise ValueError(
            "CompleteP comparison must use the two-phase forecast fleet so the "
            "reference LR is frozen before target tasks are constructed"
        )
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    context = prepare_distributed(runtime, device)
    completed = 0
    total = int(plan["planned_grid_trials"])
    try:
        corpus = TokenizedTextCorpus(
            forecast_tokenized_text_spec(config),
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
        tau_grid: List[Optional[float]] = [
            float(value) for value in plan.get("weight_decay_tau_ema_grid", ())
        ]
        if bool(plan["optimizer_contract"].get("include_zero_weight_decay_control")):
            tau_grid = [None, *tau_grid]
        if not tau_grid:
            tau_grid = [None]
        tuning_records: List[BatchRunRecord] = []
        tuning_rows = []
        for eta in plan.get("tuning_task_learning_rates", plan["learning_rates"]):
            for tau_ema in tau_grid:
                matching_by_seed: Dict[int, float] = {}
                tuning_seeds = _tuning_seeds_for_learning_rate(plan, float(eta))
                for seed in tuning_seeds:
                    tau_message = (
                        f" · tau_EMA {tau_ema:g}" if tau_ema is not None else ""
                    )
                    _progress(
                        progress,
                        "tune-reference",
                        completed,
                        total,
                        f"Reference LR {eta:g}{tau_message} · seed {seed}",
                    )
                    record = _run_trial(
                        config=config,
                        plan=plan,
                        scale=reference_scale,
                        corpus=corpus,
                        runtime=runtime,
                        context=context,
                        eta=float(eta),
                        weight_decay_tau_ema=tau_ema,
                        seed=seed,
                        optimizer_mode="theory",
                        cache_directory=cache_directory,
                    )
                    tuning_records.append(record)
                    matching_by_seed[seed] = record.final_validation_loss
                    completed += 1
                matching = [matching_by_seed[seed] for seed in tuning_seeds]
                selection_seeds = _selection_seeds_for_tuning(plan)
                selection_losses = [
                    matching_by_seed[seed] for seed in selection_seeds
                ]
                mean, sem = _mean_sem(matching)
                selection_mean, selection_sem = _mean_sem(selection_losses)
                tuning_rows.append(
                    {
                        "learning_rate": float(eta),
                        "weight_decay_tau_ema": tau_ema,
                        "mean_validation_loss": mean,
                        "sem_validation_loss": sem,
                        "seed_count": len(tuning_seeds),
                        "seeds": tuning_seeds,
                        "selection_mean_validation_loss": selection_mean,
                        "selection_sem_validation_loss": selection_sem,
                        "selection_seed_count": len(selection_seeds),
                        "selection_seeds": selection_seeds,
                        "selection_seed_losses": selection_losses,
                        "selection_evidence": (
                            "matched_single_seed_across_all_learning_rates"
                            if len(selection_seeds) == 1
                            else "matched_multi_seed_mean"
                        ),
                        "seed_losses": matching,
                    }
                )
        selected_index = min(
            range(len(tuning_rows)),
            key=lambda index: tuning_rows[index][
                "selection_mean_validation_loss"
            ],
        )
        selected_eta = float(tuning_rows[selected_index]["learning_rate"])
        selected_tau_ema = tuning_rows[selected_index]["weight_decay_tau_ema"]
        selected_eta_index = list(plan["learning_rates"]).index(selected_eta)
        learning_rate_optimum_interior = (
            0 < selected_eta_index < len(plan["learning_rates"]) - 1
        )
        weight_decay_optimum_interior = True
        if selected_tau_ema is not None:
            finite_tau_grid = [value for value in tau_grid if value is not None]
            selected_tau_index = finite_tau_grid.index(float(selected_tau_ema))
            weight_decay_optimum_interior = (
                0 < selected_tau_index < len(finite_tau_grid) - 1
            )
        optimum_interior = (
            learning_rate_optimum_interior and weight_decay_optimum_interior
        )

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
                        weight_decay_tau_ema=selected_tau_ema,
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
                        weight_decay_tau_ema=selected_tau_ema,
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
        fit_parameter_axis = str(plan.get("fit_parameter_axis", "parameters"))
        holdout_backtests = []
        for index, row in enumerate(aggregates):
            if not row["heldout"]:
                continue
            prefix = aggregates[:index]
            ensemble = fit_scaling_ensemble(
                [item[fit_parameter_axis] for item in prefix],
                [item["mean_validation_loss"] for item in prefix],
                [item["sem_validation_loss"] for item in prefix],
                target_size=float(row[fit_parameter_axis]),
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
                    "non_embedding_parameters": row["non_embedding_parameters"],
                    "fit_parameter_axis": fit_parameter_axis,
                    "fit_parameter_value": row[fit_parameter_axis],
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
                    [item[fit_parameter_axis] for item in aggregates],
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
            if not learning_rate_optimum_interior:
                refusal_reasons.append(
                    "reference learning-rate optimum is on the grid boundary"
                )
            if not weight_decay_optimum_interior:
                refusal_reasons.append(
                    "reference tau_EMA optimum is on the grid boundary"
                )
        if float(plan["fit_parameter_span"]) < float(plan["minimum_parameter_span"]):
            refusal_reasons.append(
                f"non-held-out {fit_parameter_axis} ladder span is too small"
            )
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
        parameter_axis_backtests: Dict[str, List[Dict[str, Any]]] = {}
        for axis in ("parameters", "non_embedding_parameters"):
            axis_rows: List[Dict[str, Any]] = []
            for index, row in enumerate(aggregates):
                if not row["heldout"]:
                    continue
                prefix = aggregates[:index]
                axis_fit = fit_scaling_ensemble(
                    [item[axis] for item in prefix],
                    [item["mean_validation_loss"] for item in prefix],
                    [item["sem_validation_loss"] for item in prefix],
                    target_size=float(row[axis]),
                    maximum_extrapolation_factor=maximum_extrapolation,
                    maximum_family_spread=maximum_family_spread,
                    maximum_backtest_relative_error=maximum_backtest_error,
                    bootstrap_samples=bootstrap_samples,
                )
                prediction = float(axis_fit["exploratory_prediction"])
                axis_rows.append(
                    {
                        "scale": row["name"],
                        "target_size": row[axis],
                        "observed_loss": row["mean_validation_loss"],
                        "predicted_loss": prediction,
                        "relative_error": abs(
                            prediction / row["mean_validation_loss"] - 1.0
                        ),
                        "fit": axis_fit,
                    }
                )
            parameter_axis_backtests[axis] = axis_rows
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
            "fit_parameter_axis": fit_parameter_axis,
            "runtime": plan["runtime"],
            "reference_tuning": {
                "scale": reference_scale["name"],
                "selected_learning_rate": selected_eta,
                "selected_weight_decay_tau_ema": selected_tau_ema,
                "learning_rate_optimum_is_interior": (
                    learning_rate_optimum_interior
                ),
                "weight_decay_optimum_is_interior": (
                    weight_decay_optimum_interior
                ),
                "optimum_is_interior": optimum_interior,
                "grid": tuning_rows,
            },
            "scales": aggregates,
            "hidden_scale_backtests": holdout_backtests,
            "parameter_axis_backtests": parameter_axis_backtests,
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
