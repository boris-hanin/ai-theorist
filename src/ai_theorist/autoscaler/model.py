from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .schema import ArchitectureTemplate, DatasetSpec, ScaleLevel


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, activation: str, branch_scale: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width)
        self.down = nn.Linear(width, width)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.branch_scale = branch_scale

    def forward(self, x: Tensor) -> Tensor:
        branch = self.down(self.activation(self.up(self.norm(x))))
        return x + self.branch_scale * branch


class MoEExpert(nn.Module):
    def __init__(self, width: int, expert_width: int, activation: str) -> None:
        super().__init__()
        self.up = nn.Linear(width, expert_width)
        self.down = nn.Linear(expert_width, width)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.activation(self.up(x)))


class ResidualMoEBlock(nn.Module):
    """Top-k residual MoE with fixed sparsity and auxiliary-loss-free balancing."""

    def __init__(
        self,
        width: int,
        activation: str,
        branch_scale: float,
        *,
        expert_width: int,
        num_experts: int,
        active_experts: int,
        router_balance_rate: float,
    ) -> None:
        super().__init__()
        self.width = width
        self.expert_width = expert_width
        self.expert_width_multiplier = expert_width / width
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.active_fraction = active_experts / num_experts
        self.router_balance_rate = router_balance_rate
        self.branch_scale = branch_scale
        self.norm = nn.LayerNorm(width)
        self.router = nn.Linear(width, num_experts)
        self.experts = nn.ModuleList(
            MoEExpert(width, self.expert_width, activation) for _ in range(num_experts)
        )
        self.register_buffer("balance_bias", torch.zeros(num_experts))
        self.register_buffer("_last_load", torch.zeros(num_experts), persistent=False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, mean=0.0, std=1.0 / math.sqrt(self.width))
        nn.init.zeros_(self.router.bias)
        self.balance_bias.zero_()
        self._last_load.zero_()
        down_std = math.sqrt(self.width) / self.expert_width
        for expert in self.experts:
            nn.init.normal_(expert.up.weight, mean=0.0, std=1.0 / math.sqrt(self.width))
            nn.init.zeros_(expert.up.bias)
            nn.init.normal_(expert.down.weight, mean=0.0, std=down_std)
            nn.init.zeros_(expert.down.bias)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm(x)
        logits = self.router(normalized)
        if self.num_experts == 1:
            mask = torch.ones_like(logits)
            gates = mask
        else:
            routing_scores = torch.sigmoid(logits) + self.balance_bias
            selected = routing_scores.topk(self.active_experts, dim=-1).indices
            mask = torch.zeros_like(logits).scatter_(-1, selected, 1.0)
            gates = torch.sigmoid(logits) * mask / self.active_experts
        with torch.no_grad():
            reduce_dims = tuple(range(mask.ndim - 1))
            self._last_load.copy_(mask.mean(dim=reduce_dims))
        expert_outputs = torch.stack(
            [expert(normalized) for expert in self.experts], dim=-2
        )
        branch = (expert_outputs * gates.unsqueeze(-1)).sum(dim=-2)
        return x + self.branch_scale * branch

    def update_balance(self, load: Optional[Tensor] = None) -> None:
        if self.num_experts == 1:
            return
        observed = self._last_load if load is None else load.to(self.balance_bias)
        with torch.no_grad():
            self.balance_bias.sub_(
                self.router_balance_rate * (observed - self.active_fraction)
            )

    def optimizer_parameter_groups(self, normalized_eta: float) -> List[Dict[str, object]]:
        """Table-1 Adam rates for router, up, and down projections."""
        return [
            {
                "name": "moe_router",
                "params": list(self.router.parameters()),
                "lr": normalized_eta / self.width,
            },
            {
                "name": "moe_up",
                "params": [parameter for expert in self.experts for parameter in expert.up.parameters()],
                "lr": normalized_eta / self.width,
            },
            {
                "name": "moe_down",
                "params": [
                    parameter for expert in self.experts for parameter in expert.down.parameters()
                ],
                "lr": normalized_eta / self.expert_width,
            },
        ]


class ResidualMLP(nn.Module):
    """Typed Embed -> repeated residual block -> Unembed graph."""

    def __init__(self, architecture: ArchitectureTemplate, scale: ScaleLevel) -> None:
        super().__init__()
        width = scale.width
        self.embed = nn.Linear(architecture.input_dim, width)
        self.block_type = architecture.block_type
        # CompleteP/Chizat depth parameterization: every residual branch is O(1/R).
        # Keep this outside the block-type branches so dense and MoE models cannot
        # silently drift to different depth scalings.
        branch_scale = architecture.residual_multiplier / scale.repeats
        if architecture.block_type == "pre_norm_moe":
            if scale.expert_width is None:
                raise ValueError("MoE scale requires expert_width")
            self.blocks = nn.ModuleList(
                ResidualMoEBlock(
                    width,
                    architecture.activation,
                    branch_scale,
                    expert_width=scale.expert_width,
                    num_experts=architecture.num_experts,
                    active_experts=architecture.active_experts,
                    router_balance_rate=architecture.router_balance_rate,
                )
                for _ in range(scale.repeats)
            )
        else:
            self.blocks = nn.ModuleList(
                ResidualMLPBlock(width, architecture.activation, branch_scale)
                for _ in range(scale.repeats)
            )
        self.final_norm = nn.LayerNorm(width)
        self.unembed = nn.Linear(width, architecture.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(self.embed.in_features))
        nn.init.zeros_(self.embed.bias)
        if self.block_type == "pre_norm_moe":
            nn.init.normal_(self.unembed.weight, mean=0.0, std=1.0 / self.unembed.in_features)
            for block in self.blocks:
                assert isinstance(block, ResidualMoEBlock)
                block.reset_parameters()
        else:
            nn.init.normal_(
                self.unembed.weight,
                mean=0.0,
                std=1.0 / math.sqrt(self.unembed.in_features),
            )
            for block in self.blocks:
                assert isinstance(block, ResidualMLPBlock)
                nn.init.normal_(
                    block.up.weight, mean=0.0, std=1.0 / math.sqrt(block.up.in_features)
                )
                nn.init.zeros_(block.up.bias)
                nn.init.normal_(
                    block.down.weight,
                    mean=0.0,
                    std=0.1 / math.sqrt(block.down.in_features),
                )
                nn.init.zeros_(block.down.bias)
        nn.init.zeros_(self.unembed.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.final_norm(x))

    def routing_loads(self) -> Optional[List[Tensor]]:
        if self.block_type != "pre_norm_moe":
            return None
        return [
            block._last_load.detach().clone()
            for block in self.blocks
            if isinstance(block, ResidualMoEBlock)
        ]

    def update_router_balance(self, loads: Optional[Sequence[Tensor]] = None) -> None:
        if self.block_type != "pre_norm_moe":
            return
        for index, block in enumerate(self.blocks):
            assert isinstance(block, ResidualMoEBlock)
            block.update_balance(None if loads is None else loads[index])

    def optimizer_parameter_groups(self, normalized_eta: float) -> List[Dict[str, object]]:
        if self.block_type != "pre_norm_moe":
            return [{"name": "all", "params": list(self.parameters()), "lr": normalized_eta}]
        width = self.embed.out_features
        groups: List[Dict[str, object]] = [
            {
                "name": "adapters_and_norms",
                "params": list(self.embed.parameters())
                + list(self.final_norm.parameters())
                + [parameter for block in self.blocks for parameter in block.norm.parameters()],
                "lr": normalized_eta,
            },
            {
                "name": "readout_weight",
                "params": [self.unembed.weight],
                "lr": normalized_eta / width,
            },
            {
                "name": "readout_bias",
                "params": [self.unembed.bias],
                "lr": normalized_eta,
            },
        ]
        for block in self.blocks:
            assert isinstance(block, ResidualMoEBlock)
            groups.extend(block.optimizer_parameter_groups(normalized_eta))
        return groups


def make_teacher_dataset(
    architecture: ArchitectureTemplate,
    dataset: DatasetSpec,
    *,
    device: torch.device | str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a deterministic configurable teacher task shared by every scale.

    Training sets are nested prefixes as ``n_train`` grows, while validation is
    generated from a separate fixed stream.  Joint data-scaling studies therefore
    compare every model on exactly the same held-out examples.
    """
    teacher_generator = torch.Generator(device="cpu").manual_seed(dataset.seed + 10_003)
    layer_weights: List[Tensor] = []
    input_width = architecture.input_dim
    for _ in range(dataset.teacher_depth):
        weight = torch.randn(
            input_width,
            dataset.teacher_width,
            generator=teacher_generator,
        ) / math.sqrt(input_width)
        layer_weights.append(weight)
        input_width = dataset.teacher_width
    readout = torch.randn(
        input_width,
        architecture.output_dim,
        generator=teacher_generator,
    ) / math.sqrt(input_width)

    def inputs(count: int, seed: int) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(count, architecture.input_dim, generator=generator)

    def targets(x: Tensor, noise_seed: int) -> Tensor:
        hidden = x
        for layer, weight in enumerate(layer_weights):
            projected = hidden @ weight
            hidden = torch.sin((1.15 + 0.1 * layer) * projected) + 0.2 * torch.tanh(
                projected.square() - 1.0
            )
        y = hidden @ readout
        if dataset.noise_std:
            noise_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
            y = y + dataset.noise_std * torch.randn(y.shape, generator=noise_generator)
        return y

    x_train = inputs(dataset.n_train, dataset.seed)
    x_validation = inputs(dataset.n_validation, dataset.seed + 1)
    y_train = targets(x_train, dataset.seed + 2)
    y_validation = targets(x_validation, dataset.seed + 3)
    calibration_x = inputs(1024, dataset.seed + 4)
    calibration_y = targets(calibration_x, dataset.seed + 5)
    mean_value = calibration_y.mean(dim=0, keepdim=True)
    scale_value = calibration_y.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_train = (y_train - mean_value) / scale_value
    y_validation = (y_validation - mean_value) / scale_value
    return tuple(
        tensor.to(device)
        for tensor in (x_train, y_train, x_validation, y_validation)
    )  # type: ignore[return-value]
