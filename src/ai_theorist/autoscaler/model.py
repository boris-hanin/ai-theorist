from __future__ import annotations

import math
from typing import Tuple

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


class ResidualMLP(nn.Module):
    """The MVP's typed Embed -> repeated Residual MLP -> Unembed graph."""

    def __init__(self, architecture: ArchitectureTemplate, scale: ScaleLevel) -> None:
        super().__init__()
        width = scale.width
        self.embed = nn.Linear(architecture.input_dim, width)
        branch_scale = architecture.residual_multiplier / math.sqrt(scale.repeats)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(width, architecture.activation, branch_scale)
            for _ in range(scale.repeats)
        )
        self.final_norm = nn.LayerNorm(width)
        self.unembed = nn.Linear(width, architecture.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features))
                nn.init.zeros_(module.bias)
        # Start each residual branch near identity without making it exactly inert.
        for block in self.blocks:
            nn.init.normal_(block.down.weight, mean=0.0, std=0.1 / math.sqrt(block.down.in_features))

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.final_norm(x))


def make_teacher_dataset(
    architecture: ArchitectureTemplate,
    dataset: DatasetSpec,
    *,
    device: torch.device | str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create one deterministic nonlinear regression task shared by every scale."""
    generator = torch.Generator(device="cpu").manual_seed(dataset.seed)
    total = dataset.n_train + dataset.n_validation
    x = torch.randn(total, architecture.input_dim, generator=generator)
    w_linear = torch.randn(architecture.input_dim, architecture.output_dim, generator=generator)
    w_quad = torch.randn(architecture.input_dim, architecture.output_dim, generator=generator)
    w_linear /= math.sqrt(architecture.input_dim)
    w_quad /= math.sqrt(architecture.input_dim)
    y = torch.sin(1.3 * (x @ w_linear)) + 0.35 * ((x.square() - 1.0) @ w_quad)
    if dataset.noise_std:
        noise = torch.randn(y.shape, generator=generator)
        y = y + dataset.noise_std * noise
    y = (y - y.mean(dim=0, keepdim=True)) / y.std(dim=0, keepdim=True).clamp_min(1e-6)
    split = dataset.n_train
    return tuple(t.to(device) for t in (x[:split], y[:split], x[split:], y[split:]))  # type: ignore[return-value]
