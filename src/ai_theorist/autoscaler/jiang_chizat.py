from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
    def __init__(self, shape: JiangChizatShape) -> None:
        super().__init__()
        self.width = shape.residual_width
        self.num_heads = shape.num_heads
        self.head_dimension = shape.head_dimension
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=False)
        self.output = nn.Linear(self.width, self.width, bias=False)
        self.last_attention_logits: Optional[Tensor] = None
        self.last_attention_probabilities: Optional[Tensor] = None
        self.register_buffer("last_logit_rms", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_entropy", torch.tensor(float("nan")), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.width)
        nn.init.normal_(self.qkv.weight, mean=0.0, std=std)
        nn.init.normal_(self.output.weight, mean=0.0, std=std)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, time, _ = hidden.shape
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(value: Tensor) -> Tensor:
            return value.view(
                batch, time, self.num_heads, self.head_dimension
            ).transpose(1, 2)

        q, k, v = (split_heads(value) for value in (q, k, v))
        # Jiang et al. use QK^T / d_head, not the standard / sqrt(d_head).
        logits = torch.matmul(q, k.transpose(-2, -1)) / self.head_dimension
        causal_mask = torch.ones(time, time, dtype=torch.bool, device=hidden.device).triu(1)
        logits = logits.masked_fill(causal_mask, float("-inf"))
        probabilities = logits.softmax(dim=-1)
        self.last_attention_logits = logits.detach()
        self.last_attention_probabilities = probabilities.detach()
        with torch.no_grad():
            finite_logits = logits.masked_fill(~torch.isfinite(logits), 0.0).float()
            finite_count = torch.isfinite(logits).sum().clamp_min(1)
            self.last_logit_rms.copy_((finite_logits.square().sum() / finite_count).sqrt())
            entropy = -(
                probabilities.float()
                * probabilities.float().clamp_min(1e-12).log()
            ).sum(dim=-1)
            self.last_entropy.copy_(entropy.mean())
        attended = torch.matmul(probabilities, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, self.width)
        return self.output(attended)


class JiangChizatBlock(nn.Module):
    def __init__(
        self,
        shape: JiangChizatShape,
        *,
        down_initialization: Literal["mean_field", "fan_in"] = "mean_field",
        disable_attention: bool = False,
    ) -> None:
        super().__init__()
        if down_initialization not in {"mean_field", "fan_in"}:
            raise ValueError("down_initialization must be mean_field or fan_in")
        self.shape = shape
        self.disable_attention = disable_attention
        self.attention_norm = nn.LayerNorm(shape.residual_width)
        self.attention = JiangChizatAttention(shape)
        self.ffn_norm = nn.LayerNorm(shape.residual_width)
        self.ffn_up = nn.Linear(shape.residual_width, shape.hidden_width, bias=False)
        self.ffn_down = nn.Linear(shape.hidden_width, shape.residual_width, bias=False)
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
        nn.init.normal_(self.ffn_down.weight, mean=0.0, std=down_std)

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
        omit_attention_width_factor: bool = False,
        omit_ffn_hidden_width_factor: bool = False,
    ) -> List[Dict[str, object]]:
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if not math.isfinite(epsilon0) or epsilon0 <= 0.0:
            raise ValueError("epsilon0 must be finite and positive")
        depth_ratio = self.shape.depth / self.reference.depth
        hidden_ratio = self.shape.hidden_width / self.reference.hidden_width
        residual_ratio = self.shape.residual_width / self.reference.residual_width
        attention_rate = eta * (
            1.0 if omit_attention_width_factor else residual_ratio ** -1.0
        )
        ffn_down_rate = eta * (
            1.0 if omit_ffn_hidden_width_factor else hidden_ratio ** -1.0
        )
        attention_parameters: List[nn.Parameter] = []
        ffn_up_parameters: List[nn.Parameter] = []
        ffn_down_parameters: List[nn.Parameter] = []
        norm_parameters: List[nn.Parameter] = list(self.final_norm.parameters())
        for block in self.blocks:
            attention_parameters.extend(block.attention.parameters())
            ffn_up_parameters.extend(block.ffn_up.parameters())
            ffn_down_parameters.extend(block.ffn_down.parameters())
            norm_parameters.extend(block.attention_norm.parameters())
            norm_parameters.extend(block.ffn_norm.parameters())
        return [
            {
                "name": "jiang_embeddings",
                "params": [self.token_embedding.weight, self.position_embedding.weight],
                "lr": eta,
                "eps": epsilon0 * residual_ratio ** -1.0,
            },
            {
                "name": "jiang_norms",
                "params": norm_parameters,
                "lr": eta,
                "eps": epsilon0,
            },
            {
                "name": "jiang_attention",
                "params": attention_parameters,
                "lr": attention_rate,
                "eps": epsilon0 * residual_ratio ** -1.0 * depth_ratio ** -1.0,
            },
            {
                "name": "jiang_ffn_up",
                "params": ffn_up_parameters,
                "lr": eta * residual_ratio ** -1.0,
                "eps": epsilon0 * hidden_ratio ** -1.0 * depth_ratio ** -1.0,
            },
            {
                "name": "jiang_ffn_down",
                "params": ffn_down_parameters,
                "lr": ffn_down_rate,
                "eps": (
                    epsilon0
                    * residual_ratio
                    * hidden_ratio ** -2.0
                    * depth_ratio ** -1.0
                ),
            },
        ]

    def make_optimizer(
        self,
        eta: float,
        *,
        epsilon0: float = 1e-12,
        beta1: float = 0.9,
        beta2: float = 0.95,
        **group_options: bool,
    ) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.optimizer_parameter_groups(
                eta, epsilon0=epsilon0, **group_options
            ),
            lr=eta,
            betas=(beta1, beta2),
            weight_decay=0.0,
        )

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, float]:
        entropies = [block.attention.last_entropy for block in self.blocks]
        logit_rms = [block.attention.last_logit_rms for block in self.blocks]
        finite_entropy = [value for value in entropies if torch.isfinite(value)]
        finite_logits = [value for value in logit_rms if torch.isfinite(value)]
        return {
            "mean_attention_entropy": (
                float(torch.stack(finite_entropy).mean().cpu())
                if finite_entropy
                else float("nan")
            ),
            "mean_attention_logit_rms": (
                float(torch.stack(finite_logits).mean().cpu())
                if finite_logits
                else float("nan")
            ),
            "rho_LM_over_D": self.shape.rho,
        }

    def semantic_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        for group in self.optimizer_parameter_groups(1.0, epsilon0=1e-12):
            name = str(group["name"])
            for parameter in group["params"]:  # type: ignore[union-attr]
                yield name, parameter
