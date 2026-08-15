from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Literal

import torch
from torch import nn

from .lr_contract import LearningRateTheory, audit_optimizer_groups, theory_group


ChizatRateRule = Literal["lmd", "omit_l", "omit_m", "omit_d", "constant_raw"]


CHIZAT_2LP_GD_THEORY = LearningRateTheory(
    contract_id="chizat-hidden-width-2lp-resnet-gd-v2",
    architecture="Chizat 2LP mean-ODE ResNet, equation (22), fixed input/output maps",
    optimizer="full-batch gradient descent",
    source_title="The Hidden Width of Deep ResNets: Tight Error Bounds and Phase Diagram",
    source_url="https://arxiv.org/abs/2509.10167v2",
    source_version="arXiv:2509.10167v2, equations (22)-(23), critical MLU regime",
    base_coordinate="independent normalized constants eta_u and eta_v",
    applicability=(
        "smooth 2LP residual blocks; sigma_u=sigma_v=sqrt(D); residual factor 1/(L M); "
        "fixed W_in/W_out; vanilla full-batch GD; D=O(L M)"
    ),
)


@dataclass(frozen=True)
class Chizat2LPShape:
    depth: int
    hidden_width: int
    embedding_dimension: int

    def __post_init__(self) -> None:
        if min(self.depth, self.hidden_width, self.embedding_dimension) <= 0:
            raise ValueError("L, M, and D must be positive")

    @property
    def lmd(self) -> int:
        return self.depth * self.hidden_width * self.embedding_dimension

    @property
    def rho(self) -> float:
        """The joint-limit shape coordinate L M / D."""

        return self.depth * self.hidden_width / self.embedding_dimension


class Chizat2LPResNet(nn.Module):
    """Literal implementation of Chizat (2025), equation (22).

    ``U[l, j]`` and ``V[l, j]`` are the particle vectors
    :math:`u^{j,l},v^{j,l}`.  The embedding and unembedding are fixed buffers;
    only U and V are trainable.  The smooth nonlinearity is tanh, matching the
    authors' numerical notebook.
    """

    def __init__(
        self,
        shape: Chizat2LPShape,
        *,
        input_dimension: int,
        output_dimension: int,
        map_seed: int = 202509,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if min(input_dimension, output_dimension) <= 0:
            raise ValueError("input and output dimensions must be positive")
        self.shape = shape
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        map_generator = torch.Generator(device="cpu")
        map_generator.manual_seed(map_seed)
        # With fixed input dimension, iid unit-variance entries make every
        # coordinate of W_in x O(1).  W_out is likewise a fixed iid map and the
        # explicit 1/D factor is applied in forward exactly as in equation (22).
        input_map = torch.randn(
            shape.embedding_dimension, input_dimension, generator=map_generator, dtype=dtype
        )
        output_map = torch.randn(
            shape.embedding_dimension, output_dimension, generator=map_generator, dtype=dtype
        )
        self.register_buffer("input_map", input_map)
        self.register_buffer("output_map", output_map)

        parameter_shape = (shape.depth, shape.hidden_width, shape.embedding_dimension)
        self.U = nn.Parameter(torch.empty(parameter_shape, dtype=dtype))
        self.V = nn.Parameter(torch.empty(parameter_shape, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        critical_std = math.sqrt(self.shape.embedding_dimension)
        nn.init.normal_(self.U, mean=0.0, std=critical_std)
        nn.init.normal_(self.V, mean=0.0, std=critical_std)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs @ self.input_map.T
        denominator = float(self.shape.depth * self.shape.hidden_width)
        dimension = float(self.shape.embedding_dimension)
        for layer in range(self.shape.depth):
            preactivations = hidden @ self.U[layer].T / dimension
            hidden = hidden + torch.tanh(preactivations) @ self.V[layer] / denominator
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_features(inputs)
        return hidden @ self.output_map / float(self.shape.embedding_dimension)

    def optimizer_parameter_groups(
        self,
        *,
        eta_u: float,
        eta_v: float,
        rule: ChizatRateRule = "lmd",
        reference_shape: Chizat2LPShape | None = None,
    ) -> List[Dict[str, object]]:
        if not math.isfinite(eta_u) or eta_u <= 0.0:
            raise ValueError("eta_u must be finite and positive")
        if not math.isfinite(eta_v) or eta_v <= 0.0:
            raise ValueError("eta_v must be finite and positive")

        L = float(self.shape.depth)
        M = float(self.shape.hidden_width)
        D = float(self.shape.embedding_dimension)
        if rule == "lmd":
            multiplier = L * M * D
            factor_formula = "L * M * D"
        elif rule == "omit_l":
            multiplier = M * D
            factor_formula = "M * D (negative control: L omitted)"
        elif rule == "omit_m":
            multiplier = L * D
            factor_formula = "L * D (negative control: M omitted)"
        elif rule == "omit_d":
            multiplier = L * M
            factor_formula = "L * M (negative control: D omitted)"
        elif rule == "constant_raw":
            if reference_shape is None:
                raise ValueError("constant_raw requires reference_shape")
            multiplier = float(reference_shape.lmd)
            factor_formula = "L_ref * M_ref * D_ref (negative control: raw rates held fixed)"
        else:
            raise ValueError(f"unknown Chizat LR rule: {rule}")

        factors = {
            "L": L,
            "M": M,
            "D": D,
            "LM_over_D": self.shape.rho,
            "raw_multiplier": multiplier,
        }
        return [
            theory_group(
                name="particle_u",
                params=[self.U],
                lr=eta_u * multiplier,
                lr_formula=f"eta_u * ({factor_formula})",
                theory=CHIZAT_2LP_GD_THEORY,
                scale_factors=factors,
            ),
            theory_group(
                name="particle_v",
                params=[self.V],
                lr=eta_v * multiplier,
                lr_formula=f"eta_v * ({factor_formula})",
                theory=CHIZAT_2LP_GD_THEORY,
                scale_factors=factors,
            ),
        ]

    def optimizer_contract_audit(
        self,
        *,
        eta_u: float,
        eta_v: float,
        rule: ChizatRateRule = "lmd",
        reference_shape: Chizat2LPShape | None = None,
    ) -> Dict[str, object]:
        groups = self.optimizer_parameter_groups(
            eta_u=eta_u,
            eta_v=eta_v,
            rule=rule,
            reference_shape=reference_shape,
        )
        return audit_optimizer_groups(self, groups, CHIZAT_2LP_GD_THEORY)
