from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

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


def jiang_moe_parameter_counts(
    *,
    vocab_size: int,
    context_length: int,
    depth: int,
    residual_width: int,
    expert_width: int,
    num_experts: int,
    active_experts: int,
) -> Dict[str, int]:
    """Return exact total and per-token active parameter counts.

    The router and every routing bias participate in every token's routing
    decision, so they are active even though only ``active_experts`` expert
    MLPs are evaluated.  Tied token embedding/unembedding weights are counted
    once.  The manually updated expert biases are model parameters even though
    they are deliberately excluded from Adam.
    """

    shape = JiangMoEShape(
        depth=depth,
        residual_width=residual_width,
        expert_width=expert_width,
        head_dimension=residual_width,
        num_experts=num_experts,
        active_experts=active_experts,
    )
    del shape
    if vocab_size < 8 or context_length < 2:
        raise ValueError("vocab_size >= 8 and context_length >= 2 are required")
    embedding = (vocab_size + context_length) * residual_width
    final_norm = 2 * residual_width
    shared_per_block = (
        4 * residual_width * residual_width
        + 8 * residual_width
        + residual_width * num_experts
        + num_experts
    )
    per_expert = (
        2 * residual_width * expert_width
        + expert_width
        + residual_width
    )
    total = (
        embedding
        + final_norm
        + depth * (shared_per_block + num_experts * per_expert)
    )
    active = (
        embedding
        + final_norm
        + depth * (shared_per_block + active_experts * per_expert)
    )
    return {
        "parameters": int(total),
        "active_parameters": int(active),
        "non_embedding_parameters": int(total - embedding),
        "active_non_embedding_parameters": int(active - embedding),
        "embedding_parameters": int(embedding),
        "shared_parameters_per_block": int(shared_per_block),
        "parameters_per_expert_per_block": int(per_expert),
    }


class JiangExpert(nn.Module):
    def __init__(
        self,
        residual_width: int,
        expert_width: int,
        *,
        up_initialization_std: float,
        down_initialization_std: float,
    ) -> None:
        super().__init__()
        self.up = nn.Linear(residual_width, expert_width, bias=True)
        self.down = nn.Linear(expert_width, residual_width, bias=True)
        nn.init.normal_(self.up.weight, mean=0.0, std=up_initialization_std)
        nn.init.zeros_(self.up.bias)
        nn.init.normal_(self.down.weight, mean=0.0, std=down_initialization_std)
        nn.init.zeros_(self.down.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(hidden)))


class JiangSparseMoE(nn.Module):
    def __init__(
        self,
        shape: JiangMoEShape,
        *,
        router_gamma: float = 1.0,
        router_initialization_std: float,
        expert_up_initialization_std: float,
        expert_down_initialization_std: float,
    ) -> None:
        super().__init__()
        if not math.isfinite(router_gamma) or router_gamma < 0.5:
            raise ValueError("router_gamma must be finite and at least 1/2")
        self.shape = shape
        self.router = nn.Linear(shape.residual_width, shape.num_experts, bias=False)
        nn.init.normal_(
            self.router.weight,
            mean=0.0,
            std=router_initialization_std,
        )
        self.experts = nn.ModuleList(
            JiangExpert(
                shape.residual_width,
                shape.expert_width,
                up_initialization_std=expert_up_initialization_std,
                down_initialization_std=expert_down_initialization_std,
            )
            for _ in range(shape.num_experts)
        )
        # This is a manually updated parameter from equation (2), not an Adam
        # parameter.  Keeping it as a frozen Parameter makes exact model-state
        # accounting include it while the optimizer audit correctly excludes it.
        self.expert_bias = nn.Parameter(
            torch.zeros(shape.num_experts), requires_grad=False
        )
        self.register_buffer("last_load", torch.zeros(shape.num_experts), persistent=False)
        self.register_buffer(
            "last_balancing_load", torch.zeros(shape.num_experts), persistent=False
        )
        self.register_buffer(
            "maximum_balancing_load_deviation",
            torch.zeros(()),
            persistent=False,
        )
        self.register_buffer(
            "routing_assignment_counts",
            torch.zeros(shape.num_experts, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "routing_token_count", torch.zeros((), dtype=torch.float64), persistent=False
        )
        self._routing_measurement_active = False

    def forward(self, hidden: Tensor) -> Tensor:
        router_logits = self.router(hidden)
        gates = torch.sigmoid(router_logits)
        selected = (gates + self.expert_bias).topk(
            self.shape.active_experts,
            dim=-1,
        ).indices
        with torch.no_grad():
            flat_selected = selected.reshape(-1, self.shape.active_experts)
            assignment_counts = torch.bincount(
                flat_selected.reshape(-1), minlength=self.shape.num_experts
            ).to(dtype=torch.float64)
            token_count = flat_selected.shape[0]
            self.last_load.copy_(
                (assignment_counts / max(token_count, 1)).to(self.last_load.dtype)
            )
            if self._routing_measurement_active:
                self.routing_assignment_counts.add_(assignment_counts)
                self.routing_token_count.add_(token_count)

        # Exact token-choice dispatch: only tokens actually assigned to an
        # expert pass through that expert.  The earlier prototype evaluated all
        # experts on all tokens and merely multiplied inactive outputs by zero,
        # which had sparse semantics but dense compute.
        flat_hidden = hidden.reshape(-1, self.shape.residual_width)
        flat_gates = gates.reshape(-1, self.shape.num_experts)
        flat_selected = selected.reshape(-1, self.shape.active_experts)
        selected_gates = flat_gates.gather(1, flat_selected)
        output = torch.zeros_like(flat_hidden)
        for index, expert in enumerate(self.experts):
            token_indices, slots = torch.where(flat_selected == index)
            if token_indices.numel():
                expert_inputs = flat_hidden.index_select(0, token_indices)
                coefficients = selected_gates[token_indices, slots].unsqueeze(-1)
                contributions = coefficients * expert(expert_inputs)
                output = output.index_add(0, token_indices, contributions)
            elif self.training:
                # DDP must see every expert parameter in every backward pass,
                # even in a small batch where an expert receives no tokens.
                zero_anchor = sum(
                    parameter.reshape(-1)[0]
                    for parameter in expert.parameters()
                ) * 0.0
                output = output + zero_anchor.to(output.dtype)
        return output.reshape_as(hidden) / self.shape.active_experts

    @torch.no_grad()
    def begin_routing_measurement(self) -> None:
        self.routing_assignment_counts.zero_()
        self.routing_token_count.zero_()
        self._routing_measurement_active = True

    @torch.no_grad()
    def finish_routing_measurement(self, *, synchronize: bool = False) -> Tensor:
        self._routing_measurement_active = False
        if synchronize:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("routing synchronization requires initialized distributed")
            torch.distributed.all_reduce(
                self.routing_assignment_counts, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                self.routing_token_count, op=torch.distributed.ReduceOp.SUM
            )
        if self.routing_token_count.item() <= 0.0:
            raise RuntimeError("routing measurement observed no tokens")
        self.last_load.copy_(
            (self.routing_assignment_counts / self.routing_token_count).to(
                self.last_load.dtype
            )
        )
        self.last_balancing_load.copy_(self.last_load)
        self.maximum_balancing_load_deviation.copy_(
            (self.last_load - self.shape.sparsity).abs().max()
        )
        return self.last_load

    @torch.no_grad()
    def update_expert_bias(self, learning_rate: float) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("expert bias learning rate must be finite and positive")
        self.expert_bias.add_(
            -learning_rate * (self.last_load - self.shape.sparsity)
        )


class JiangMoEBlock(nn.Module):
    def __init__(
        self,
        shape: JiangMoEShape,
        *,
        router_gamma: float = 1.0,
        hidden_initialization_std: float,
        router_initialization_std: float,
        expert_down_initialization_std: float,
        attention_backend: str = "math",
        capture_attention_diagnostics: bool = True,
    ) -> None:
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
            initialization_std=hidden_initialization_std,
            bias=True,
            attention_backend=attention_backend,
            capture_diagnostics=capture_attention_diagnostics,
        )
        self.moe_norm = nn.LayerNorm(shape.residual_width)
        self.moe = JiangSparseMoE(
            shape,
            router_gamma=router_gamma,
            router_initialization_std=router_initialization_std,
            expert_up_initialization_std=hidden_initialization_std,
            expert_down_initialization_std=expert_down_initialization_std,
        )

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
        initialization_std: float = 0.02,
        router_gamma: float = 1.0,
        attention_backend: str = "math",
        activation_checkpointing: bool = False,
        capture_attention_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size < 8 or context_length < 2:
            raise ValueError("vocab_size >= 8 and context_length >= 2 are required")
        if not math.isfinite(initialization_std) or initialization_std <= 0.0:
            raise ValueError("initialization_std must be finite and positive")
        if not math.isclose(shape.sparsity, reference.sparsity):
            raise ValueError("Jiang MoE transfer requires fixed active-expert fraction kappa")
        self.shape = shape
        self.reference = reference
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.activation_checkpointing = activation_checkpointing
        self.residual_width_ratio = (
            shape.residual_width / reference.residual_width
        )
        self.depth_ratio = shape.depth / reference.depth
        self.ffn_ratio_ratio = shape.ffn_ratio / reference.ffn_ratio
        self.initialization_std = initialization_std
        self.router_gamma = router_gamma
        self.hidden_initialization_std = (
            initialization_std * self.residual_width_ratio ** -0.5
        )
        self.router_initialization_std = (
            initialization_std * self.residual_width_ratio ** -router_gamma
        )
        self.expert_down_initialization_std = (
            self.hidden_initialization_std
            * self.ffn_ratio_ratio ** -1.0
            * JIANG_REPORTED_DOWN_INIT_MULTIPLIER
        )
        self.token_embedding = nn.Embedding(vocab_size, shape.residual_width)
        self.position_embedding = nn.Embedding(context_length, shape.residual_width)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=initialization_std)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=initialization_std)
        self.blocks = nn.ModuleList(
            JiangMoEBlock(
                shape,
                router_gamma=router_gamma,
                hidden_initialization_std=self.hidden_initialization_std,
                router_initialization_std=self.router_initialization_std,
                expert_down_initialization_std=self.expert_down_initialization_std,
                attention_backend=attention_backend,
                capture_attention_diagnostics=capture_attention_diagnostics,
            )
            for _ in range(shape.depth)
        )
        self.final_norm = nn.LayerNorm(shape.residual_width)

    def forward_features(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape [batch, time <= context_length]")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                hidden = activation_checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, tokens: Tensor) -> Tensor:
        # The non-MoE boundary follows CompleteP: tied readout with the inverse
        # residual-width multiplier relative to the tuned reference model.
        return (
            F.linear(self.forward_features(tokens), self.token_embedding.weight)
            / self.residual_width_ratio
        )

    def parameter_accounting(self) -> Dict[str, int]:
        counts = jiang_moe_parameter_counts(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            depth=self.shape.depth,
            residual_width=self.shape.residual_width,
            expert_width=self.shape.expert_width,
            num_experts=self.shape.num_experts,
            active_experts=self.shape.active_experts,
        )
        constructed = sum(parameter.numel() for parameter in self.parameters())
        if constructed != counts["parameters"]:
            raise RuntimeError(
                f"MoE accounting expected {counts['parameters']} parameters but "
                f"constructed {constructed}"
            )
        return counts

    def initialization_contract(self) -> Dict[str, object]:
        """Return the exact source-coordinate initialization used by the model."""

        return {
            "reference_initialization_std": self.initialization_std,
            "residual_width_ratio": self.residual_width_ratio,
            "ffn_ratio_ratio": self.ffn_ratio_ratio,
            "router_gamma": self.router_gamma,
            "embedding_and_unembedding_std": self.initialization_std,
            "position_embedding_std": self.initialization_std,
            "attention_qko_std": self.hidden_initialization_std,
            "attention_value_std": (
                self.hidden_initialization_std
                * JIANG_REPORTED_VALUE_INIT_MULTIPLIER
            ),
            "router_std": self.router_initialization_std,
            "expert_up_std": self.hidden_initialization_std,
            "expert_down_std": self.expert_down_initialization_std,
            "expert_bias_initialization": 0.0,
            "formulas": {
                "embedding_and_unembedding": "sigma0",
                "attention_qko_and_expert_up": "sigma0 * (D/D0)^(-1/2)",
                "attention_value": "sigma0 * (D/D0)^(-1/2) * 1/16",
                "router": "sigma0 * (D/D0)^(-gamma); gamma=1 in main text",
                "expert_down": (
                    "sigma0 * (D/D0)^(-1/2) * "
                    "(alpha/alpha0)^(-1) * 1/4"
                ),
                "expert_bias": "0",
            },
        }

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
    def begin_routing_measurement(self) -> None:
        for block in self.blocks:
            block.moe.begin_routing_measurement()

    @torch.no_grad()
    def update_expert_biases(
        self, learning_rate: float, *, synchronize: bool = False
    ) -> None:
        for block in self.blocks:
            if block.moe._routing_measurement_active:
                block.moe.finish_routing_measurement(synchronize=synchronize)
            block.moe.update_expert_bias(learning_rate)

    @torch.no_grad()
    def routing_diagnostics(self) -> Dict[str, object]:
        loads = [
            block.moe.last_balancing_load.detach().cpu().tolist()
            for block in self.blocks
        ]
        deviations = [
            abs(value - self.shape.sparsity)
            for layer in loads
            for value in layer
        ]
        return {
            "loads": loads,
            "maximum_absolute_load_deviation": max(deviations) if deviations else 0.0,
            "maximum_balancing_load_deviation_by_layer": [
                float(block.moe.maximum_balancing_load_deviation.item())
                for block in self.blocks
            ],
            "active_expert_fraction": self.shape.sparsity,
            "rho_LM_over_D": self.shape.rho_lm_over_d,
            "routing_token_counts": [
                int(block.moe.routing_token_count.item()) for block in self.blocks
            ],
            "expert_biases": [
                block.moe.expert_bias.detach().cpu().tolist()
                for block in self.blocks
            ],
            "maximum_absolute_expert_bias": max(
                (
                    float(block.moe.expert_bias.detach().abs().max().item())
                    for block in self.blocks
                ),
                default=0.0,
            ),
            "parameter_accounting": self.parameter_accounting(),
            "initialization_contract": self.initialization_contract(),
        }

    def manual_parameter_contract(self, expert_bias_learning_rate: float) -> Dict[str, object]:
        if not math.isfinite(expert_bias_learning_rate) or expert_bias_learning_rate <= 0.0:
            raise ValueError("expert bias learning rate must be finite and positive")
        return {
            "name": "jiang_moe_expert_routing_bias",
            "update": "b_i <- b_i - eta_bias * (Load_i - kappa)",
            "learning_rate": expert_bias_learning_rate,
            "learning_rate_formula": "eta_bias (Theta(1), independent of expert count at fixed kappa)",
            "reference_constant_status": (
                "explicitly declared reference hyperparameter; the paper proves "
                "the no-scaling rule but does not report a universal numeric value"
            ),
            "initialization": "zero",
            "parameter_shapes": [[self.shape.num_experts] for _ in self.blocks],
            "theory_contract_id": JIANG_MOE_ADAM_THEORY.contract_id,
        }
