from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import math
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .lr_contract import LearningRateTheory, audit_optimizer_groups, theory_group


JIANG_COMPLETEP_ADAM_THEORY = LearningRateTheory(
    contract_id="jiang-bordelon-pehlevan-hanin-completep-adam-v4",
    architecture="pre-LN decoder with 1/L MHSA and mean-field FFN residual branches",
    optimizer="adam",
    source_title="Hyperparameter Transfer with Mixture-of-Experts Layers",
    source_url="https://arxiv.org/abs/2601.20205",
    source_version="arXiv:2601.20205v3, Table 2 (dense CompleteP groups)",
    base_coordinate="eta and epsilon0 declared at reference (L0, M0, D0)",
    applicability=(
        "fixed-token-budget Adam transfer across depth L, residual width D, and "
        "mean-field FFN width M with QK^T/d_head and 1/L residual branches"
    ),
)


JIANG_COMPLETEP_ADAMW_THEORY = LearningRateTheory(
    contract_id="jiang-chizat-completep-adamw-tau-ema-v1",
    architecture="pre-LN decoder with 1/L MHSA and mean-field FFN residual branches",
    optimizer="adamw",
    source_title=(
        "Jiang CompleteP LR/epsilon groups with Dey et al. AdamW weight-decay transfer"
    ),
    source_url="https://arxiv.org/abs/2505.01618",
    source_version=(
        "Jiang et al. arXiv:2601.20205v3 Table 2; Dey et al. "
        "arXiv:2505.01618v4 Table 1, Appendix D.4 and Appendix G.1"
    ),
    base_coordinate=(
        "eta, epsilon0, and tau_EMA=(eta*lambda0*n_steps)^(-1) at "
        "reference (L0, M0, D0)"
    ),
    applicability=(
        "fixed-schedule AdamW transfer across depth L, residual width D, and "
        "mean-field FFN width M; Jiang LR/epsilon rules are preserved while "
        "Dey weight decay is scaled by each hidden matrix's input width"
    ),
)


# Appendix D.1 reports the constant-scale values obtained by the authors' one
# coordinate-sweep pass.  They are part of the base parameterization, not
# width/depth exponents and not values to silently reset to one at larger scale.
JIANG_REPORTED_VALUE_INIT_MULTIPLIER = 1.0 / 16.0
JIANG_REPORTED_DOWN_INIT_MULTIPLIER = 1.0 / 4.0
JIANG_DENSE_REPORTED_LR_MULTIPLIERS: Dict[str, float] = {
    "jiang_embeddings": 1.0,
    "jiang_norms": 1.0,
    "jiang_attention_qkv": 1.0 / 16.0,
    "jiang_attention_output": 1.0,
    "jiang_ffn_up": 1.0,
    "jiang_ffn_down": 1.0 / 16.0,
    "jiang_other_biases": 1.0,
}


@dataclass(frozen=True)
class JiangChizatShape:
    depth: int
    hidden_width: int
    residual_width: int
    head_dimension: int

    def __post_init__(self) -> None:
        for name, value in (
            ("depth", self.depth),
            ("hidden_width", self.hidden_width),
            ("residual_width", self.residual_width),
            ("head_dimension", self.head_dimension),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.residual_width % self.head_dimension:
            raise ValueError("residual_width must be divisible by head_dimension")

    @property
    def num_heads(self) -> int:
        return self.residual_width // self.head_dimension

    @property
    def rho(self) -> float:
        return self.depth * self.hidden_width / self.residual_width


@dataclass(frozen=True)
class JiangChizatReference:
    depth: int
    hidden_width: int
    residual_width: int

    def __post_init__(self) -> None:
        for name, value in (
            ("depth", self.depth),
            ("hidden_width", self.hidden_width),
            ("residual_width", self.residual_width),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"reference {name} must be a positive integer")


class JiangChizatAttention(nn.Module):
    def __init__(
        self,
        shape: JiangChizatShape,
        *,
        value_initialization_multiplier: float = JIANG_REPORTED_VALUE_INIT_MULTIPLIER,
        bias: bool = True,
        attention_backend: str = "math",
        capture_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if (
            not math.isfinite(value_initialization_multiplier)
            or value_initialization_multiplier <= 0.0
        ):
            raise ValueError("value_initialization_multiplier must be finite and positive")
        self.width = shape.residual_width
        self.num_heads = shape.num_heads
        self.head_dimension = shape.head_dimension
        self.value_initialization_multiplier = value_initialization_multiplier
        if attention_backend not in {"auto", "math", "flash"}:
            raise ValueError("attention_backend must be auto, math, or flash")
        self.attention_backend = attention_backend
        self.capture_diagnostics = capture_diagnostics
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=bias)
        self.output = nn.Linear(self.width, self.width, bias=bias)
        self.last_attention_logits: Optional[Tensor] = None
        self.last_attention_probabilities: Optional[Tensor] = None
        self.register_buffer("last_logit_rms", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_entropy", torch.tensor(float("nan")), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.width)
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.chunk(3, dim=0)
            nn.init.normal_(q_weight, mean=0.0, std=std)
            nn.init.normal_(k_weight, mean=0.0, std=std)
            nn.init.normal_(
                v_weight,
                mean=0.0,
                std=std * self.value_initialization_multiplier,
            )
        nn.init.normal_(self.output.weight, mean=0.0, std=std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, time, _ = hidden.shape
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(value: Tensor) -> Tensor:
            return value.view(
                batch, time, self.num_heads, self.head_dimension
            ).transpose(1, 2)

        q, k, v = (split_heads(value) for value in (q, k, v))
        kernel_context = nullcontext()
        if self.attention_backend != "auto":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backend = (
                SDPBackend.FLASH_ATTENTION
                if self.attention_backend == "flash"
                else SDPBackend.MATH
            )
            kernel_context = sdpa_kernel([backend])
        # Passing the scale explicitly preserves Jiang et al.'s QK^T/d_head
        # convention under every SDPA backend.
        with kernel_context:
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                scale=1.0 / self.head_dimension,
            )
        if self.capture_diagnostics:
            with torch.no_grad():
                logits = torch.matmul(q, k.transpose(-2, -1)) / self.head_dimension
                causal_mask = torch.ones(
                    time, time, dtype=torch.bool, device=hidden.device
                ).triu(1)
                logits = logits.masked_fill(causal_mask, float("-inf"))
                probabilities = logits.softmax(dim=-1)
                self.last_attention_logits = logits.detach()
                self.last_attention_probabilities = probabilities.detach()
                finite_logits = logits.masked_fill(~torch.isfinite(logits), 0.0).float()
                finite_count = torch.isfinite(logits).sum().clamp_min(1)
                self.last_logit_rms.copy_(
                    (finite_logits.square().sum() / finite_count).sqrt()
                )
                entropy = -(
                    probabilities.float()
                    * probabilities.float().clamp_min(1e-12).log()
                ).sum(dim=-1)
                self.last_entropy.copy_(entropy.mean())
        attended = attended.transpose(1, 2).contiguous().view(batch, time, self.width)
        return self.output(attended)


class JiangChizatBlock(nn.Module):
    def __init__(
        self,
        shape: JiangChizatShape,
        *,
        down_initialization: Literal["mean_field", "fan_in"] = "mean_field",
        disable_attention: bool = False,
        value_initialization_multiplier: float = JIANG_REPORTED_VALUE_INIT_MULTIPLIER,
        down_initialization_multiplier: float = JIANG_REPORTED_DOWN_INIT_MULTIPLIER,
        attention_backend: str = "math",
        capture_attention_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if down_initialization not in {"mean_field", "fan_in"}:
            raise ValueError("down_initialization must be mean_field or fan_in")
        if (
            not math.isfinite(down_initialization_multiplier)
            or down_initialization_multiplier <= 0.0
        ):
            raise ValueError("down_initialization_multiplier must be finite and positive")
        self.shape = shape
        self.disable_attention = disable_attention
        self.attention_norm = nn.LayerNorm(shape.residual_width)
        self.attention = JiangChizatAttention(
            shape,
            value_initialization_multiplier=value_initialization_multiplier,
            bias=True,
            attention_backend=attention_backend,
            capture_diagnostics=capture_attention_diagnostics,
        )
        self.ffn_norm = nn.LayerNorm(shape.residual_width)
        self.ffn_up = nn.Linear(shape.residual_width, shape.hidden_width, bias=True)
        self.ffn_down = nn.Linear(shape.hidden_width, shape.residual_width, bias=True)
        nn.init.normal_(
            self.ffn_up.weight,
            mean=0.0,
            std=1.0 / math.sqrt(shape.residual_width),
        )
        down_std = (
            math.sqrt(shape.residual_width) / shape.hidden_width
            if down_initialization == "mean_field"
            else 1.0 / math.sqrt(shape.hidden_width)
        )
        nn.init.normal_(
            self.ffn_down.weight,
            mean=0.0,
            std=down_std * down_initialization_multiplier,
        )
        nn.init.zeros_(self.ffn_up.bias)
        nn.init.zeros_(self.ffn_down.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        residual_scale = 1.0 / self.shape.depth
        if not self.disable_attention:
            hidden = hidden + residual_scale * self.attention(self.attention_norm(hidden))
        hidden = hidden + residual_scale * self.ffn_down(
            F.gelu(self.ffn_up(self.ffn_norm(hidden)))
        )
        return hidden


class JiangChizatTransformer(nn.Module):
    """Dense interleaved core of Jiang et al. with a Chizat-width FFN."""

    def __init__(
        self,
        shape: JiangChizatShape,
        *,
        vocab_size: int,
        context_length: int,
        reference: JiangChizatReference,
        embedding_initialization_std: float = 0.02,
        down_initialization: Literal["mean_field", "fan_in"] = "mean_field",
        disable_attention: bool = False,
        value_initialization_multiplier: float = JIANG_REPORTED_VALUE_INIT_MULTIPLIER,
        down_initialization_multiplier: float = JIANG_REPORTED_DOWN_INIT_MULTIPLIER,
        attention_backend: str = "math",
        activation_checkpointing: bool = False,
        capture_attention_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size < 8 or context_length < 2:
            raise ValueError("vocab_size >= 8 and context_length >= 2 are required")
        if not math.isfinite(embedding_initialization_std) or embedding_initialization_std <= 0:
            raise ValueError("embedding_initialization_std must be finite and positive")
        self.shape = shape
        self.reference = reference
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.activation_checkpointing = activation_checkpointing
        self.token_embedding = nn.Embedding(vocab_size, shape.residual_width)
        self.position_embedding = nn.Embedding(context_length, shape.residual_width)
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=embedding_initialization_std,
        )
        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=embedding_initialization_std,
        )
        self.blocks = nn.ModuleList(
            JiangChizatBlock(
                shape,
                down_initialization=down_initialization,
                disable_attention=disable_attention,
                value_initialization_multiplier=value_initialization_multiplier,
                down_initialization_multiplier=down_initialization_multiplier,
                attention_backend=attention_backend,
                capture_attention_diagnostics=capture_attention_diagnostics,
            )
            for _ in range(shape.depth)
        )
        self.final_norm = nn.LayerNorm(shape.residual_width)

    def forward_features(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        if tokens.shape[1] > self.context_length:
            raise ValueError("token sequence exceeds configured context length")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                hidden = activation_checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = self.forward_features(tokens)
        # Tied input/output token embeddings, as in the Jiang experiment setup.
        return F.linear(hidden, self.token_embedding.weight)

    def optimizer_parameter_groups(
        self,
        eta: float,
        *,
        epsilon0: float,
        weight_decay0: float = 0.0,
        optimizer_name: Literal["adam", "adamw"] = "adam",
        omit_attention_width_factor: bool = False,
        omit_ffn_hidden_width_factor: bool = False,
        learning_rate_multipliers: Mapping[str, float] | None = None,
    ) -> List[Dict[str, object]]:
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if not math.isfinite(epsilon0) or epsilon0 <= 0.0:
            raise ValueError("epsilon0 must be finite and positive")
        if not math.isfinite(weight_decay0) or weight_decay0 < 0.0:
            raise ValueError("weight_decay0 must be finite and non-negative")
        if optimizer_name not in {"adam", "adamw"}:
            raise ValueError("optimizer_name must be adam or adamw")
        if optimizer_name == "adam" and weight_decay0 != 0.0:
            raise ValueError("Jiang Adam does not accept decoupled weight decay")
        theory = (
            JIANG_COMPLETEP_ADAMW_THEORY
            if optimizer_name == "adamw"
            else JIANG_COMPLETEP_ADAM_THEORY
        )
        depth_ratio = self.shape.depth / self.reference.depth
        hidden_ratio = self.shape.hidden_width / self.reference.hidden_width
        residual_ratio = self.shape.residual_width / self.reference.residual_width
        attention_rate = eta * (
            1.0 if omit_attention_width_factor else residual_ratio ** -1.0
        )
        ffn_down_rate = eta * (
            1.0 if omit_ffn_hidden_width_factor else hidden_ratio ** -1.0
        )
        attention_qkv_parameters: List[nn.Parameter] = []
        attention_output_parameters: List[nn.Parameter] = []
        ffn_up_parameters: List[nn.Parameter] = []
        ffn_down_parameters: List[nn.Parameter] = []
        other_bias_parameters: List[nn.Parameter] = []
        norm_parameters: List[nn.Parameter] = list(self.final_norm.parameters())
        for block in self.blocks:
            attention_qkv_parameters.append(block.attention.qkv.weight)
            attention_output_parameters.append(block.attention.output.weight)
            ffn_up_parameters.append(block.ffn_up.weight)
            ffn_down_parameters.append(block.ffn_down.weight)
            other_bias_parameters.extend(
                parameter
                for parameter in (
                    block.attention.qkv.bias,
                    block.attention.output.bias,
                    block.ffn_up.bias,
                    block.ffn_down.bias,
                )
                if parameter is not None
            )
            norm_parameters.extend(block.attention_norm.parameters())
            norm_parameters.extend(block.ffn_norm.parameters())
        factors = {
            "depth_ratio": depth_ratio,
            "ffn_width_ratio": hidden_ratio,
            "residual_width_ratio": residual_ratio,
            "base_weight_decay": weight_decay0,
        }
        group_names = (
            "jiang_embeddings",
            "jiang_norms",
            "jiang_attention_qkv",
            "jiang_attention_output",
            "jiang_ffn_up",
            "jiang_ffn_down",
            "jiang_other_biases",
        )
        multipliers = dict(JIANG_DENSE_REPORTED_LR_MULTIPLIERS)
        if learning_rate_multipliers is not None:
            unknown = set(learning_rate_multipliers) - set(group_names)
            if unknown:
                raise ValueError(
                    "unknown Jiang-Chizat LR multiplier groups: "
                    + ", ".join(sorted(unknown))
                )
            for name, value in learning_rate_multipliers.items():
                multiplier = float(value)
                if not math.isfinite(multiplier) or multiplier <= 0.0:
                    raise ValueError(f"LR multiplier for {name} must be finite and positive")
                multipliers[name] = multiplier
        groups = [
            theory_group(
                name="jiang_embeddings",
                params=[self.token_embedding.weight, self.position_embedding.weight],
                lr=eta * multipliers["jiang_embeddings"],
                lr_formula="c_embeddings * eta; c_embeddings tuned at reference",
                eps=epsilon0 * residual_ratio ** -1.0,
                eps_formula="epsilon0 * (D/D0)^(-1)",
                theory=theory,
                scale_factors={**factors, "base_lr_multiplier": multipliers["jiang_embeddings"]},
            ),
            theory_group(
                name="jiang_norms",
                params=norm_parameters,
                lr=eta * multipliers["jiang_norms"],
                lr_formula="c_norms * eta; c_norms tuned at reference",
                eps=epsilon0,
                eps_formula="epsilon0",
                theory=theory,
                scale_factors={**factors, "base_lr_multiplier": multipliers["jiang_norms"]},
            ),
            theory_group(
                name="jiang_attention_qkv",
                params=attention_qkv_parameters,
                lr=attention_rate * multipliers["jiang_attention_qkv"],
                lr_formula=(
                    "c_attention_qkv * eta (negative control: D factor omitted)"
                    if omit_attention_width_factor
                    else "c_attention_qkv * eta * (D/D0)^(-1)"
                ),
                eps=epsilon0 * residual_ratio ** -1.0 * depth_ratio ** -1.0,
                eps_formula="epsilon0 * (D/D0)^(-1) * (L/L0)^(-1)",
                theory=theory,
                scale_factors={**factors, "base_lr_multiplier": multipliers["jiang_attention_qkv"]},
            ),
            theory_group(
                name="jiang_attention_output",
                params=attention_output_parameters,
                lr=attention_rate * multipliers["jiang_attention_output"],
                lr_formula=(
                    "c_attention_output * eta (negative control: D factor omitted)"
                    if omit_attention_width_factor
                    else "c_attention_output * eta * (D/D0)^(-1)"
                ),
                eps=epsilon0 * residual_ratio ** -1.0 * depth_ratio ** -1.0,
                eps_formula="epsilon0 * (D/D0)^(-1) * (L/L0)^(-1)",
                theory=theory,
                scale_factors={**factors, "base_lr_multiplier": multipliers["jiang_attention_output"]},
            ),
            theory_group(
                name="jiang_ffn_up",
                params=ffn_up_parameters,
                lr=eta * residual_ratio ** -1.0 * multipliers["jiang_ffn_up"],
                lr_formula="c_ffn_up * eta * (D/D0)^(-1)",
                eps=epsilon0 * hidden_ratio ** -1.0 * depth_ratio ** -1.0,
                eps_formula="epsilon0 * (M/M0)^(-1) * (L/L0)^(-1)",
                theory=theory,
                scale_factors={**factors, "base_lr_multiplier": multipliers["jiang_ffn_up"]},
            ),
            theory_group(
                name="jiang_ffn_down",
                params=ffn_down_parameters,
                lr=ffn_down_rate * multipliers["jiang_ffn_down"],
                lr_formula=(
                    "c_ffn_down * eta (negative control: M factor omitted)"
                    if omit_ffn_hidden_width_factor
                    else "c_ffn_down * eta * (M/M0)^(-1)"
                ),
                eps=(
                    epsilon0
                    * residual_ratio
                    * hidden_ratio ** -2.0
                    * depth_ratio ** -1.0
                ),
                eps_formula="epsilon0 * (D/D0) * (M/M0)^(-2) * (L/L0)^(-1)",
                theory=theory,
                scale_factors={**factors, "base_lr_multiplier": multipliers["jiang_ffn_down"]},
            ),
            theory_group(
                name="jiang_other_biases",
                params=other_bias_parameters,
                lr=eta * multipliers["jiang_other_biases"],
                lr_formula="c_other_biases * eta",
                eps=epsilon0 * depth_ratio ** -1.0,
                eps_formula="epsilon0 * (L/L0)^(-1)",
                theory=theory,
                scale_factors={**factors, "base_lr_multiplier": multipliers["jiang_other_biases"]},
            ),
        ]
        for group in groups:
            name = str(group["name"])
            if name == "jiang_embeddings":
                group["weight_decay"] = weight_decay0
                group["weight_decay_formula"] = "lambda0"
            elif name in {
                "jiang_attention_qkv",
                "jiang_attention_output",
                "jiang_ffn_up",
            }:
                group["weight_decay"] = weight_decay0 * residual_ratio
                group["weight_decay_formula"] = "lambda0 * (D/D0)"
            elif name == "jiang_ffn_down":
                group["weight_decay"] = weight_decay0 * hidden_ratio
                group["weight_decay_formula"] = "lambda0 * (M/M0)"
            else:
                group["weight_decay"] = 0.0
                group["weight_decay_formula"] = "0 for norms and biases"
        return groups

    def optimizer_contract_audit(
        self,
        eta: float,
        *,
        epsilon0: float = 1e-12,
        weight_decay0: float = 0.0,
        optimizer_name: Literal["adam", "adamw"] = "adam",
        omit_attention_width_factor: bool = False,
        omit_ffn_hidden_width_factor: bool = False,
        learning_rate_multipliers: Mapping[str, float] | None = None,
    ) -> Dict[str, object]:
        groups = self.optimizer_parameter_groups(
            eta,
            epsilon0=epsilon0,
            weight_decay0=weight_decay0,
            optimizer_name=optimizer_name,
            omit_attention_width_factor=omit_attention_width_factor,
            omit_ffn_hidden_width_factor=omit_ffn_hidden_width_factor,
            learning_rate_multipliers=learning_rate_multipliers,
        )
        theory = (
            JIANG_COMPLETEP_ADAMW_THEORY
            if optimizer_name == "adamw"
            else JIANG_COMPLETEP_ADAM_THEORY
        )
        audit = audit_optimizer_groups(self, groups, theory)
        rows = {str(group["name"]): group for group in groups}
        for row in audit["groups"]:  # type: ignore[index]
            group = rows[str(row["name"])]  # type: ignore[index]
            row["weight_decay"] = float(group["weight_decay"])  # type: ignore[index]
            row["weight_decay_formula"] = str(group["weight_decay_formula"])  # type: ignore[index]
        return audit

    def make_optimizer(
        self,
        eta: float,
        *,
        epsilon0: float = 1e-12,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay0: float = 0.0,
        optimizer_name: Literal["adam", "adamw"] = "adam",
        omit_attention_width_factor: bool = False,
        omit_ffn_hidden_width_factor: bool = False,
        learning_rate_multipliers: Mapping[str, float] | None = None,
    ) -> torch.optim.Optimizer:
        optimizer_class = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
        return optimizer_class(
            self.optimizer_parameter_groups(
                eta,
                epsilon0=epsilon0,
                weight_decay0=weight_decay0,
                optimizer_name=optimizer_name,
                omit_attention_width_factor=omit_attention_width_factor,
                omit_ffn_hidden_width_factor=omit_ffn_hidden_width_factor,
                learning_rate_multipliers=learning_rate_multipliers,
            ),
            lr=eta,
            betas=(beta1, beta2),
            weight_decay=0.0,
        )

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, Optional[float]]:
        entropies = [block.attention.last_entropy for block in self.blocks]
        logit_rms = [block.attention.last_logit_rms for block in self.blocks]
        finite_entropy = [value for value in entropies if torch.isfinite(value)]
        finite_logits = [value for value in logit_rms if torch.isfinite(value)]
        return {
            "mean_attention_entropy": (
                float(torch.stack(finite_entropy).mean().cpu())
                if finite_entropy
                else None
            ),
            "mean_attention_logit_rms": (
                float(torch.stack(finite_logits).mean().cpu())
                if finite_logits
                else None
            ),
            "rho_LM_over_D": self.shape.rho,
        }

    def semantic_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        for group in self.optimizer_parameter_groups(1.0, epsilon0=1e-12):
            name = str(group["name"])
            for parameter in group["params"]:  # type: ignore[union-attr]
                yield name, parameter
