from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Dict, List, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .lr_contract import LearningRateTheory, audit_optimizer_groups, theory_group


COMPLETEP_ADAMW_THEORY = LearningRateTheory(
    contract_id="dey-completep-adamw-v4-paper-qk-width",
    architecture="dense pre-LN decoder Transformer with CompleteP alpha=1",
    optimizer="adamw",
    source_title="Don't be lazy: CompleteP enables compute-efficient deep transformers",
    source_url="https://arxiv.org/abs/2505.01618",
    source_version="arXiv:2505.01618v4, Table 1 and Equations 38-40",
    base_coordinate="eta, epsilon0, lambda0, sigma0 at reference (L0, N0)",
    applicability=(
        "AdamW transfer across residual width N and depth L with QK^T/N, "
        "(L/L0)^-1 residual branches, hidden logit scaling, and width-scaled "
        "hidden initialization"
    ),
)


@dataclass(frozen=True)
class CompletePShape:
    depth: int
    width: int
    head_dimension: int
    mlp_multiplier: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("depth", self.depth),
            ("width", self.width),
            ("head_dimension", self.head_dimension),
            ("mlp_multiplier", self.mlp_multiplier),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.head_dimension:
            raise ValueError("width must be divisible by head_dimension")

    @property
    def num_heads(self) -> int:
        return self.width // self.head_dimension

    @property
    def hidden_width(self) -> int:
        return self.mlp_multiplier * self.width


@dataclass(frozen=True)
class CompletePReference:
    depth: int
    width: int

    def __post_init__(self) -> None:
        for name, value in (("depth", self.depth), ("width", self.width)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"reference {name} must be a positive integer")


def _alibi_slopes(num_heads: int) -> Tensor:
    """Return the per-head slopes from the original ALiBi implementation."""

    if num_heads <= 0:
        raise ValueError("num_heads must be positive")

    def power_of_two_slopes(count: int) -> List[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(count) - 3.0)))
        return [start ** (index + 1) for index in range(count)]

    if math.log2(num_heads).is_integer():
        values = power_of_two_slopes(num_heads)
    else:
        lower = 2 ** math.floor(math.log2(num_heads))
        values = power_of_two_slopes(lower)
        values.extend(power_of_two_slopes(2 * lower)[0::2][: num_heads - lower])
    return torch.tensor(values, dtype=torch.float32)


class CompletePAttention(nn.Module):
    def __init__(
        self,
        shape: CompletePShape,
        *,
        initialization_std: float,
        position_encoding: Literal["alibi", "learned_absolute"],
        attention_backend: str,
        capture_diagnostics: bool,
    ) -> None:
        super().__init__()
        if attention_backend not in {"auto", "math", "flash"}:
            raise ValueError("attention_backend must be auto, math, or flash")
        if position_encoding not in {"alibi", "learned_absolute"}:
            raise ValueError("position_encoding must be alibi or learned_absolute")
        if position_encoding == "alibi" and attention_backend == "flash":
            raise ValueError(
                "strict torch SDPA FlashAttention does not support the additive "
                "ALiBi mask; use attention_backend=auto or math"
            )
        self.width = shape.width
        self.num_heads = shape.num_heads
        self.head_dimension = shape.head_dimension
        self.position_encoding = position_encoding
        self.attention_backend = attention_backend
        self.capture_diagnostics = capture_diagnostics
        self.qkv = nn.Linear(shape.width, 3 * shape.width, bias=True)
        self.output = nn.Linear(shape.width, shape.width, bias=True)
        nn.init.normal_(self.qkv.weight, mean=0.0, std=initialization_std)
        nn.init.normal_(self.output.weight, mean=0.0, std=initialization_std)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("last_logit_rms", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_entropy", torch.tensor(float("nan")), persistent=False)
        self.register_buffer(
            "alibi_slopes", _alibi_slopes(shape.num_heads), persistent=False
        )

    def _attention_mask(self, time: int, device: torch.device) -> Tensor | None:
        if self.position_encoding != "alibi":
            return None
        positions = torch.arange(time, dtype=torch.float32, device=device)
        distance = positions[:, None] - positions[None, :]
        bias = -self.alibi_slopes.to(device=device)[:, None, None] * distance[None, :, :]
        causal = torch.ones(time, time, dtype=torch.bool, device=device).triu(1)
        return bias.masked_fill(causal[None, None, :, :], float("-inf"))

    def forward(self, hidden: Tensor) -> Tensor:
        batch, time, _ = hidden.shape
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(value: Tensor) -> Tensor:
            return value.view(
                batch, time, self.num_heads, self.head_dimension
            ).transpose(1, 2)

        q, k, v = (split_heads(value) for value in (q, k, v))
        attention_mask = self._attention_mask(time, hidden.device)
        kernel_context = nullcontext()
        if self.attention_backend != "auto":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backend = (
                SDPBackend.FLASH_ATTENTION
                if self.attention_backend == "flash"
                else SDPBackend.MATH
            )
            kernel_context = sdpa_kernel([backend])
        # Dey et al. use QK^T/N in their experiments, where N is the full
        # residual-stream width rather than the per-head dimension.
        with kernel_context:
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=attention_mask is None,
                scale=1.0 / self.width,
            )
        if self.capture_diagnostics:
            with torch.no_grad():
                logits = torch.matmul(q, k.transpose(-2, -1)) / self.width
                if attention_mask is None:
                    mask = torch.ones(
                        time, time, dtype=torch.bool, device=hidden.device
                    ).triu(1)
                    logits = logits.masked_fill(mask, float("-inf"))
                else:
                    logits = logits + attention_mask
                probabilities = logits.softmax(dim=-1)
                finite = torch.isfinite(logits)
                finite_logits = logits.masked_fill(~finite, 0.0).float()
                self.last_logit_rms.copy_(
                    (finite_logits.square().sum() / finite.sum().clamp_min(1)).sqrt()
                )
                entropy = -(
                    probabilities.float()
                    * probabilities.float().clamp_min(1e-12).log()
                ).sum(dim=-1)
                self.last_entropy.copy_(entropy.mean())
        attended = attended.transpose(1, 2).contiguous().view(batch, time, self.width)
        return self.output(attended)


class CompletePBlock(nn.Module):
    def __init__(
        self,
        shape: CompletePShape,
        reference: CompletePReference,
        *,
        initialization_std: float,
        activation: Literal["gelu", "relu_squared"],
        position_encoding: Literal["alibi", "learned_absolute"],
        attention_backend: str,
        capture_attention_diagnostics: bool,
    ) -> None:
        super().__init__()
        if activation not in {"gelu", "relu_squared"}:
            raise ValueError("activation must be gelu or relu_squared")
        self.residual_scale = reference.depth / shape.depth
        self.activation = activation
        self.attention_norm = nn.LayerNorm(shape.width)
        self.attention = CompletePAttention(
            shape,
            initialization_std=initialization_std,
            position_encoding=position_encoding,
            attention_backend=attention_backend,
            capture_diagnostics=capture_attention_diagnostics,
        )
        self.ffn_norm = nn.LayerNorm(shape.width)
        self.ffn_up = nn.Linear(shape.width, shape.hidden_width, bias=True)
        self.ffn_down = nn.Linear(shape.hidden_width, shape.width, bias=True)
        nn.init.normal_(self.ffn_up.weight, mean=0.0, std=initialization_std)
        nn.init.normal_(self.ffn_down.weight, mean=0.0, std=initialization_std)
        nn.init.zeros_(self.ffn_up.bias)
        nn.init.zeros_(self.ffn_down.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.residual_scale * self.attention(
            self.attention_norm(hidden)
        )
        activated = self.ffn_up(self.ffn_norm(hidden))
        if self.activation == "relu_squared":
            activated = F.relu(activated).square()
        else:
            activated = F.gelu(activated)
        return hidden + self.residual_scale * self.ffn_down(activated)


class CompletePTransformer(nn.Module):
    """Dense CompleteP Transformer with an explicit, auditable AdamW contract."""

    def __init__(
        self,
        shape: CompletePShape,
        *,
        vocab_size: int,
        context_length: int,
        reference: CompletePReference,
        initialization_std: float = 0.02,
        activation: Literal["gelu", "relu_squared"] = "relu_squared",
        position_encoding: Literal["alibi", "learned_absolute"] = "learned_absolute",
        attention_backend: str = "math",
        activation_checkpointing: bool = False,
        capture_attention_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size < 8 or context_length < 2:
            raise ValueError("vocab_size >= 8 and context_length >= 2 are required")
        if not math.isfinite(initialization_std) or initialization_std <= 0.0:
            raise ValueError("initialization_std must be finite and positive")
        if position_encoding not in {"alibi", "learned_absolute"}:
            raise ValueError("position_encoding must be alibi or learned_absolute")
        self.shape = shape
        self.reference = reference
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.position_encoding = position_encoding
        self.activation_checkpointing = activation_checkpointing
        self.width_ratio = shape.width / reference.width
        self.depth_ratio = shape.depth / reference.depth
        hidden_initialization_std = initialization_std / math.sqrt(self.width_ratio)

        # Dey et al. prescribe width-independent read-in/readout initialization,
        # width-scaled hidden initialization, and an untied readout in their runs.
        self.token_embedding = nn.Embedding(vocab_size, shape.width)
        self.position_embedding = (
            nn.Embedding(context_length, shape.width)
            if position_encoding == "learned_absolute"
            else None
        )
        self.blocks = nn.ModuleList(
            CompletePBlock(
                shape,
                reference,
                initialization_std=hidden_initialization_std,
                activation=activation,
                position_encoding=position_encoding,
                attention_backend=attention_backend,
                capture_attention_diagnostics=capture_attention_diagnostics,
            )
            for _ in range(shape.depth)
        )
        self.final_norm = nn.LayerNorm(shape.width)
        self.unembedding = nn.Linear(shape.width, vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=initialization_std)
        if self.position_embedding is not None:
            nn.init.normal_(
                self.position_embedding.weight, mean=0.0, std=initialization_std
            )
        nn.init.normal_(self.unembedding.weight, mean=0.0, std=initialization_std)

    def forward_features(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        if tokens.shape[1] > self.context_length:
            raise ValueError("token sequence exceeds configured context length")
        hidden = self.token_embedding(tokens)
        if self.position_embedding is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            hidden = hidden + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                hidden = activation_checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, tokens: Tensor) -> Tensor:
        # The 1/m_N readout factor is a forward-pass part of CompleteP, not an LR trick.
        return self.unembedding(self.forward_features(tokens)) / self.width_ratio

    def optimizer_parameter_groups(
        self,
        eta: float,
        *,
        epsilon0: float,
        weight_decay0: float,
    ) -> List[Dict[str, object]]:
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if not math.isfinite(epsilon0) or epsilon0 <= 0.0:
            raise ValueError("epsilon0 must be finite and positive")
        if not math.isfinite(weight_decay0) or weight_decay0 < 0.0:
            raise ValueError("weight_decay0 must be finite and non-negative")

        hidden_norms: List[nn.Parameter] = []
        hidden_weights: List[nn.Parameter] = []
        hidden_biases: List[nn.Parameter] = []
        for block in self.blocks:
            hidden_norms.extend(block.attention_norm.parameters())
            hidden_norms.extend(block.ffn_norm.parameters())
            hidden_weights.extend(
                (
                    block.attention.qkv.weight,
                    block.attention.output.weight,
                    block.ffn_up.weight,
                    block.ffn_down.weight,
                )
            )
            hidden_biases.extend(
                (
                    block.attention.qkv.bias,
                    block.attention.output.bias,
                    block.ffn_up.bias,
                    block.ffn_down.bias,
                )
            )
        factors = {
            "width_ratio": self.width_ratio,
            "depth_ratio": self.depth_ratio,
            "alpha": 1.0,
            "base_weight_decay": weight_decay0,
        }
        hidden_epsilon = epsilon0 / self.width_ratio / self.depth_ratio
        endpoint_epsilon = epsilon0 / self.width_ratio
        hidden_rate = eta / self.width_ratio
        hidden_weight_decay = weight_decay0 * self.width_ratio

        embedding_parameters = [self.token_embedding.weight]
        if self.position_embedding is not None:
            embedding_parameters.append(self.position_embedding.weight)
        groups = [
            theory_group(
                name="completep_embeddings",
                params=embedding_parameters,
                lr=eta,
                lr_formula="eta",
                eps=endpoint_epsilon,
                eps_formula="epsilon0 * (N/N0)^(-1)",
                theory=COMPLETEP_ADAMW_THEORY,
                scale_factors=factors,
            ),
            theory_group(
                name="completep_hidden_norms",
                params=hidden_norms,
                lr=eta,
                lr_formula="eta * (L/L0)^(alpha-1) = eta for alpha=1",
                eps=hidden_epsilon,
                eps_formula="epsilon0 * (N/N0)^(-1) * (L/L0)^(-1)",
                theory=COMPLETEP_ADAMW_THEORY,
                scale_factors=factors,
            ),
            theory_group(
                name="completep_hidden_weights",
                params=hidden_weights,
                lr=hidden_rate,
                lr_formula="eta * (N/N0)^(-1) * (L/L0)^(alpha-1)",
                eps=hidden_epsilon,
                eps_formula="epsilon0 * (N/N0)^(-1) * (L/L0)^(-1)",
                theory=COMPLETEP_ADAMW_THEORY,
                scale_factors=factors,
            ),
            theory_group(
                name="completep_hidden_biases",
                params=hidden_biases,
                lr=eta,
                lr_formula="eta * (L/L0)^(alpha-1) = eta for alpha=1",
                eps=hidden_epsilon,
                eps_formula="epsilon0 * (N/N0)^(-1) * (L/L0)^(-1)",
                theory=COMPLETEP_ADAMW_THEORY,
                scale_factors=factors,
            ),
            theory_group(
                name="completep_final_norm",
                params=self.final_norm.parameters(),
                lr=eta,
                lr_formula="eta",
                eps=endpoint_epsilon,
                eps_formula="epsilon0 * (N/N0)^(-1)",
                theory=COMPLETEP_ADAMW_THEORY,
                scale_factors=factors,
            ),
            theory_group(
                name="completep_unembedding",
                params=(self.unembedding.weight,),
                lr=eta,
                lr_formula="eta",
                eps=endpoint_epsilon,
                eps_formula="epsilon0 * (N/N0)^(-1)",
                theory=COMPLETEP_ADAMW_THEORY,
                scale_factors=factors,
            ),
        ]
        for group in groups:
            name = str(group["name"])
            if name in {"completep_embeddings", "completep_unembedding"}:
                group["weight_decay"] = weight_decay0
                group["weight_decay_formula"] = "lambda0"
            elif name == "completep_hidden_weights":
                group["weight_decay"] = hidden_weight_decay
                group["weight_decay_formula"] = "lambda0 * (N/N0)"
            else:
                group["weight_decay"] = 0.0
                group["weight_decay_formula"] = "0 for norms and biases"
        return groups

    def optimizer_contract_audit(
        self,
        eta: float,
        *,
        epsilon0: float,
        weight_decay0: float,
    ) -> Dict[str, object]:
        groups = self.optimizer_parameter_groups(
            eta, epsilon0=epsilon0, weight_decay0=weight_decay0
        )
        audit = audit_optimizer_groups(self, groups, COMPLETEP_ADAMW_THEORY)
        rows = {str(group["name"]): group for group in groups}
        for row in audit["groups"]:  # type: ignore[index]
            group = rows[str(row["name"])]  # type: ignore[index]
            row["weight_decay"] = float(group["weight_decay"])  # type: ignore[index]
            row["weight_decay_formula"] = str(group["weight_decay_formula"])  # type: ignore[index]
        return audit

    def diagnostics(self) -> Dict[str, object]:
        entropies = [float(block.attention.last_entropy) for block in self.blocks]
        logit_rms = [float(block.attention.last_logit_rms) for block in self.blocks]
        finite_entropies = [value for value in entropies if math.isfinite(value)]
        finite_logit_rms = [value for value in logit_rms if math.isfinite(value)]
        return {
            "width_ratio": self.width_ratio,
            "depth_ratio": self.depth_ratio,
            "residual_branch_scale": 1.0 / self.depth_ratio,
            "unembedding_forward_scale": 1.0 / self.width_ratio,
            "attention_logit_scale": 1.0 / self.shape.width,
            "position_encoding": self.position_encoding,
            "mean_attention_entropy": (
                sum(finite_entropies) / len(finite_entropies)
                if finite_entropies
                else None
            ),
            "mean_attention_logit_rms": (
                sum(finite_logit_rms) / len(finite_logit_rms)
                if finite_logit_rms
                else None
            ),
        }
