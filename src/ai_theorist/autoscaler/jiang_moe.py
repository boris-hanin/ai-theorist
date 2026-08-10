from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .jiang_chizat import (
    JIANG_REPORTED_DOWN_INIT_MULTIPLIER,
    JIANG_REPORTED_VALUE_INIT_MULTIPLIER,
    JiangChizatAttention,
    JiangChizatShape,
)
from .lr_contract import LearningRateTheory, audit_optimizer_groups, theory_group


JIANG_MOE_ADAM_THEORY = LearningRateTheory(
    contract_id="jiang-bordelon-pehlevan-hanin-moe-adam-v4",
    architecture="pre-LN decoder with interleaved 1/L MHSA and sparse MoE blocks",
    optimizer="adam",
    source_title="Hyperparameter Transfer with Mixture-of-Experts Layers",
    source_url="https://arxiv.org/abs/2601.20205",
    source_version="arXiv:2601.20205v3, Sections 3.1-3.3 and Table 2",
    base_coordinate="eta, epsilon0, and expert-bias eta declared at reference shape",
    applicability=(
        "fixed-token-budget Adam transfer across depth, residual width, expert width, "
        "and expert count at fixed active-expert fraction kappa"
    ),
)


JIANG_MOE_REPORTED_LR_MULTIPLIERS: Dict[str, float] = {
    "jiang_moe_embeddings": 1.0,
    "jiang_moe_norms": 1.0,
    "jiang_moe_attention_qkv": 1.0 / 16.0,
    "jiang_moe_attention_output": 1.0,
    "jiang_moe_router": 1.0 / 16.0,
    "jiang_moe_expert_up": 1.0,
    "jiang_moe_expert_down": 1.0 / 16.0,
    "jiang_moe_other_biases": 1.0,
}


@dataclass(frozen=True)
class JiangMoEShape:
    depth: int
    residual_width: int
    expert_width: int
    head_dimension: int
    num_experts: int
    active_experts: int

    def __post_init__(self) -> None:
        for name, value in (
            ("depth", self.depth),
            ("residual_width", self.residual_width),
            ("expert_width", self.expert_width),
            ("head_dimension", self.head_dimension),
            ("num_experts", self.num_experts),
            ("active_experts", self.active_experts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.residual_width % self.head_dimension:
            raise ValueError("residual_width must be divisible by head_dimension")
        if self.active_experts > self.num_experts:
            raise ValueError("active_experts cannot exceed num_experts")

    @property
    def num_heads(self) -> int:
        return self.residual_width // self.head_dimension

    @property
    def ffn_ratio(self) -> float:
        return self.expert_width / self.residual_width

    @property
    def sparsity(self) -> float:
        return self.active_experts / self.num_experts

    @property
    def rho_lm_over_d(self) -> float:
        return self.depth * self.expert_width / self.residual_width


@dataclass(frozen=True)
class JiangMoEReference:
    depth: int
    residual_width: int
    expert_width: int
    num_experts: int
    active_experts: int

    def __post_init__(self) -> None:
        for name, value in (
            ("depth", self.depth),
            ("residual_width", self.residual_width),
            ("expert_width", self.expert_width),
            ("num_experts", self.num_experts),
            ("active_experts", self.active_experts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"reference {name} must be a positive integer")
        if self.active_experts > self.num_experts:
            raise ValueError("reference active_experts cannot exceed num_experts")

    @property
    def ffn_ratio(self) -> float:
        return self.expert_width / self.residual_width

    @property
    def sparsity(self) -> float:
        return self.active_experts / self.num_experts


class JiangExpert(nn.Module):
    def __init__(self, residual_width: int, expert_width: int) -> None:
        super().__init__()
        self.up = nn.Linear(residual_width, expert_width, bias=True)
        self.down = nn.Linear(expert_width, residual_width, bias=True)
        nn.init.normal_(self.up.weight, mean=0.0, std=residual_width ** -0.5)
        nn.init.zeros_(self.up.bias)
        nn.init.normal_(
            self.down.weight,
            mean=0.0,
            std=(
                math.sqrt(residual_width)
                / expert_width
                * JIANG_REPORTED_DOWN_INIT_MULTIPLIER
            ),
        )
        nn.init.zeros_(self.down.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(hidden)))


class JiangSparseMoE(nn.Module):
    def __init__(self, shape: JiangMoEShape, *, router_gamma: float = 0.5) -> None:
        super().__init__()
        if not math.isfinite(router_gamma) or router_gamma < 0.5:
            raise ValueError("router_gamma must be finite and at least 1/2")
        self.shape = shape
        self.router = nn.Linear(shape.residual_width, shape.num_experts, bias=False)
        nn.init.normal_(
            self.router.weight,
            mean=0.0,
            std=shape.residual_width ** (-router_gamma),
        )
        self.experts = nn.ModuleList(
            JiangExpert(shape.residual_width, shape.expert_width)
            for _ in range(shape.num_experts)
        )
        self.register_buffer("expert_bias", torch.zeros(shape.num_experts))
        self.register_buffer("last_load", torch.zeros(shape.num_experts), persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        router_logits = self.router(hidden)
        gates = torch.sigmoid(router_logits)
        selected = (gates + self.expert_bias).topk(
            self.shape.active_experts,
            dim=-1,
        ).indices
        mask = torch.zeros_like(gates).scatter_(-1, selected, 1.0)
        with torch.no_grad():
            reduce_dimensions = tuple(range(mask.ndim - 1))
            self.last_load.copy_(mask.mean(dim=reduce_dimensions))
        output = torch.zeros_like(hidden)
        for index, expert in enumerate(self.experts):
            coefficient = (mask[..., index] * gates[..., index]).unsqueeze(-1)
            output = output + coefficient * expert(hidden)
        return output / self.shape.active_experts

    @torch.no_grad()
    def update_expert_bias(self, learning_rate: float) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("expert bias learning rate must be finite and positive")
        self.expert_bias.add_(
            -learning_rate * (self.last_load - self.shape.sparsity)
        )


class JiangMoEBlock(nn.Module):
    def __init__(self, shape: JiangMoEShape) -> None:
        super().__init__()
        self.shape = shape
        self.attention_norm = nn.LayerNorm(shape.residual_width)
        # Reuse the paper-faithful QK^T/d_head attention implementation.
        self.attention = JiangChizatAttention(
            JiangChizatShape(
                depth=shape.depth,
                hidden_width=shape.expert_width,
                residual_width=shape.residual_width,
                head_dimension=shape.head_dimension,
            ),
            value_initialization_multiplier=JIANG_REPORTED_VALUE_INIT_MULTIPLIER,
            bias=True,
        )
        self.moe_norm = nn.LayerNorm(shape.residual_width)
        self.moe = JiangSparseMoE(shape)

    def forward(self, hidden: Tensor) -> Tensor:
        scale = 1.0 / self.shape.depth
        hidden = hidden + scale * self.attention(self.attention_norm(hidden))
        hidden = hidden + scale * self.moe(self.moe_norm(hidden))
        return hidden


class JiangMoETransformer(nn.Module):
    def __init__(
        self,
        shape: JiangMoEShape,
        *,
        vocab_size: int,
        context_length: int,
        reference: JiangMoEReference,
        embedding_initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        if vocab_size < 8 or context_length < 2:
            raise ValueError("vocab_size >= 8 and context_length >= 2 are required")
        if not math.isclose(shape.sparsity, reference.sparsity):
            raise ValueError("Jiang MoE transfer requires fixed active-expert fraction kappa")
        self.shape = shape
        self.reference = reference
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, shape.residual_width)
        self.position_embedding = nn.Embedding(context_length, shape.residual_width)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=embedding_initialization_std)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=embedding_initialization_std)
        self.blocks = nn.ModuleList(JiangMoEBlock(shape) for _ in range(shape.depth))
        self.final_norm = nn.LayerNorm(shape.residual_width)

    def forward_features(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape [batch, time <= context_length]")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, tokens: Tensor) -> Tensor:
        return F.linear(self.forward_features(tokens), self.token_embedding.weight)

    def optimizer_parameter_groups(
        self,
        eta: float,
        *,
        epsilon0: float,
        rule: str = "table2",
        learning_rate_multipliers: Mapping[str, float] | None = None,
    ) -> List[Dict[str, object]]:
        if rule not in {
            "table2",
            "global_lr_control",
            "omit_router_width",
            "omit_expert_down_ratio",
        }:
            raise ValueError("unknown Jiang MoE optimizer rule")
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if not math.isfinite(epsilon0) or epsilon0 <= 0.0:
            raise ValueError("epsilon0 must be finite and positive")
        depth_ratio = self.shape.depth / self.reference.depth
        residual_ratio = self.shape.residual_width / self.reference.residual_width
        expert_ratio = self.shape.expert_width / self.reference.expert_width
        ffn_ratio_ratio = self.shape.ffn_ratio / self.reference.ffn_ratio
        factors = {
            "depth_ratio": depth_ratio,
            "residual_width_ratio": residual_ratio,
            "expert_width_ratio": expert_ratio,
            "ffn_ratio_ratio": ffn_ratio_ratio,
            "expert_count_ratio": self.shape.num_experts / self.reference.num_experts,
            "active_expert_fraction": self.shape.sparsity,
        }
        norms: List[nn.Parameter] = list(self.final_norm.parameters())
        attention_qkv: List[nn.Parameter] = []
        attention_output: List[nn.Parameter] = []
        routers: List[nn.Parameter] = []
        expert_up: List[nn.Parameter] = []
        expert_down: List[nn.Parameter] = []
        other_biases: List[nn.Parameter] = []
        for block in self.blocks:
            norms.extend(block.attention_norm.parameters())
            norms.extend(block.moe_norm.parameters())
            attention_qkv.append(block.attention.qkv.weight)
            attention_output.append(block.attention.output.weight)
            other_biases.extend(
                parameter
                for parameter in (
                    block.attention.qkv.bias,
                    block.attention.output.bias,
                )
                if parameter is not None
            )
            routers.append(block.moe.router.weight)
            for expert in block.moe.experts:
                expert_up.append(expert.up.weight)
                expert_down.append(expert.down.weight)
                other_biases.extend([expert.up.bias, expert.down.bias])
        rates = {
            "jiang_moe_embeddings": eta,
            "jiang_moe_norms": eta,
            "jiang_moe_attention_qkv": eta * residual_ratio ** -1.0,
            "jiang_moe_attention_output": eta * residual_ratio ** -1.0,
            "jiang_moe_router": eta * residual_ratio ** -1.0,
            "jiang_moe_expert_up": eta * residual_ratio ** -1.0,
            "jiang_moe_expert_down": eta * expert_ratio ** -1.0,
            "jiang_moe_other_biases": eta,
        }
        formulas = {
            "jiang_moe_embeddings": "eta",
            "jiang_moe_norms": "eta",
            "jiang_moe_attention_qkv": "eta * (D/D0)^(-1)",
            "jiang_moe_attention_output": "eta * (D/D0)^(-1)",
            "jiang_moe_router": "eta * (D/D0)^(-1)",
            "jiang_moe_expert_up": "eta * (D/D0)^(-1)",
            "jiang_moe_expert_down": "eta * (M/M0)^(-1) = eta * (D/D0)^(-1) * (alpha/alpha0)^(-1)",
            "jiang_moe_other_biases": "eta",
        }
        if rule == "global_lr_control":
            rates = {name: eta for name in rates}
            formulas = {name: "eta (negative control: Table 2 factors omitted)" for name in formulas}
        elif rule == "omit_router_width":
            rates["jiang_moe_router"] = eta
            formulas["jiang_moe_router"] = "eta (negative control: router D factor omitted)"
        elif rule == "omit_expert_down_ratio":
            rates["jiang_moe_expert_down"] = eta * residual_ratio ** -1.0
            formulas["jiang_moe_expert_down"] = "eta * (D/D0)^(-1) (negative control: alpha factor omitted)"
        epsilons = {
            "jiang_moe_embeddings": epsilon0 * residual_ratio ** -1.0,
            "jiang_moe_norms": epsilon0,
            "jiang_moe_attention_qkv": epsilon0 * residual_ratio ** -1.0 * depth_ratio ** -1.0,
            "jiang_moe_attention_output": epsilon0 * residual_ratio ** -1.0 * depth_ratio ** -1.0,
            "jiang_moe_router": epsilon0 * residual_ratio ** -1.0 * depth_ratio ** -1.0,
            "jiang_moe_expert_up": epsilon0 * expert_ratio ** -1.0 * depth_ratio ** -1.0,
            "jiang_moe_expert_down": epsilon0 * residual_ratio * expert_ratio ** -2.0 * depth_ratio ** -1.0,
            "jiang_moe_other_biases": epsilon0 * depth_ratio ** -1.0,
        }
        eps_formulas = {
            "jiang_moe_embeddings": "epsilon0 * (D/D0)^(-1)",
            "jiang_moe_norms": "epsilon0",
            "jiang_moe_attention_qkv": "epsilon0 * (D/D0)^(-1) * (L/L0)^(-1)",
            "jiang_moe_attention_output": "epsilon0 * (D/D0)^(-1) * (L/L0)^(-1)",
            "jiang_moe_router": "epsilon0 * (D/D0)^(-1) * (L/L0)^(-1)",
            "jiang_moe_expert_up": "epsilon0 * (M/M0)^(-1) * (L/L0)^(-1)",
            "jiang_moe_expert_down": "epsilon0 * (D/D0) * (M/M0)^(-2) * (L/L0)^(-1)",
            "jiang_moe_other_biases": "epsilon0 * (L/L0)^(-1)",
        }
        parameters = {
            "jiang_moe_embeddings": [self.token_embedding.weight, self.position_embedding.weight],
            "jiang_moe_norms": norms,
            "jiang_moe_attention_qkv": attention_qkv,
            "jiang_moe_attention_output": attention_output,
            "jiang_moe_router": routers,
            "jiang_moe_expert_up": expert_up,
            "jiang_moe_expert_down": expert_down,
            "jiang_moe_other_biases": other_biases,
        }
        multipliers = dict(JIANG_MOE_REPORTED_LR_MULTIPLIERS)
        if learning_rate_multipliers is not None:
            unknown = set(learning_rate_multipliers) - set(parameters)
            if unknown:
                raise ValueError(
                    "unknown Jiang MoE LR multiplier groups: " + ", ".join(sorted(unknown))
                )
            for name, value in learning_rate_multipliers.items():
                multiplier = float(value)
                if not math.isfinite(multiplier) or multiplier <= 0.0:
                    raise ValueError(f"LR multiplier for {name} must be finite and positive")
                multipliers[name] = multiplier
        for name, multiplier in multipliers.items():
            rates[name] *= multiplier
            formulas[name] = f"c_{name} * ({formulas[name]}); c_{name} tuned at reference"
        return [
            theory_group(
                name=name,
                params=parameters[name],
                lr=rates[name],
                lr_formula=formulas[name],
                eps=epsilons[name],
                eps_formula=eps_formulas[name],
                theory=JIANG_MOE_ADAM_THEORY,
                scale_factors={**factors, "base_lr_multiplier": multipliers[name]},
            )
            for name in parameters
        ]

    def optimizer_contract_audit(
        self,
        eta: float,
        *,
        epsilon0: float,
        rule: str = "table2",
        learning_rate_multipliers: Mapping[str, float] | None = None,
    ) -> Dict[str, object]:
        return audit_optimizer_groups(
            self,
            self.optimizer_parameter_groups(
                eta,
                epsilon0=epsilon0,
                rule=rule,
                learning_rate_multipliers=learning_rate_multipliers,
            ),
            JIANG_MOE_ADAM_THEORY,
        )

    @torch.no_grad()
    def update_expert_biases(self, learning_rate: float) -> None:
        for block in self.blocks:
            block.moe.update_expert_bias(learning_rate)

    @torch.no_grad()
    def routing_diagnostics(self) -> Dict[str, object]:
        loads = [block.moe.last_load.detach().cpu().tolist() for block in self.blocks]
        deviations = [
            abs(value - self.shape.sparsity)
            for layer in loads
            for value in layer
        ]
        return {
            "loads": loads,
            "maximum_absolute_load_deviation": max(deviations) if deviations else 0.0,
            "active_expert_fraction": self.shape.sparsity,
            "rho_LM_over_D": self.shape.rho_lm_over_d,
        }

    def manual_parameter_contract(self, expert_bias_learning_rate: float) -> Dict[str, object]:
        if not math.isfinite(expert_bias_learning_rate) or expert_bias_learning_rate <= 0.0:
            raise ValueError("expert bias learning rate must be finite and positive")
        return {
            "name": "jiang_moe_expert_routing_bias",
            "update": "b_i <- b_i - eta_bias * (Load_i - kappa)",
            "learning_rate": expert_bias_learning_rate,
            "learning_rate_formula": "eta_bias (Theta(1), independent of expert count at fixed kappa)",
            "initialization": "zero",
            "parameter_shapes": [[self.shape.num_experts] for _ in self.blocks],
            "theory_contract_id": JIANG_MOE_ADAM_THEORY.contract_id,
        }
