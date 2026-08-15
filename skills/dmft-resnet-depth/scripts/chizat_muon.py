"""Audited single-device Muon + auxiliary Adam for the Chizat harness.

The optimizer routing is semantic:

* every residual-particle ``U`` and ``W`` matrix is updated by Muon;
* the trainable input embed and scalar unembed are updated by auxiliary Adam.

The boundary matrices remain on Adam even though they are two-dimensional.
Using tensor rank as the routing rule would therefore be a correctness bug.
The implementation follows PyTorch's Muon equations while retaining support
for the repository's Python/PyTorch floor and using float32 Newton--Schulz
arithmetic for reproducible numerical validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
ADJUSTMENT_MODES = ("match_rms_adamw", "original", "spectral_unclamped", "none")
TRANSFER_RULES = (
    "group_rms_D",
    "wrong_constant_W",
    "wrong_W_D",
    "wrong_sgd_LMD",
    "freeze_embed",
    "freeze_unembed",
    "wrong_constant_unembed",
)


@dataclass(frozen=True)
class MuonConfig:
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS
    eps: float = 1e-7
    weight_decay: float = 0.0
    adjustment: str = "match_rms_adamw"

    def validate(self) -> "MuonConfig":
        if not math.isfinite(self.momentum) or not 0.0 <= self.momentum < 1.0:
            raise ValueError("Muon momentum must be finite and in [0, 1)")
        if isinstance(self.ns_steps, bool) or not isinstance(self.ns_steps, int):
            raise ValueError("Muon ns_steps must be an integer")
        if not 1 <= self.ns_steps < 100:
            raise ValueError("Muon ns_steps must be in [1, 100)")
        if len(self.ns_coefficients) != 3 or not all(
            math.isfinite(float(value)) for value in self.ns_coefficients
        ):
            raise ValueError("Muon ns_coefficients must contain three finite values")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("Muon eps must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("Muon weight_decay must be finite and non-negative")
        if self.adjustment not in ADJUSTMENT_MODES:
            raise ValueError(f"unsupported Muon adjustment: {self.adjustment}")
        return self

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AuxAdamConfig:
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-10
    weight_decay: float = 0.0

    def validate(self) -> "AuxAdamConfig":
        for name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"auxiliary Adam {name} must be finite and in [0, 1)")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("auxiliary Adam eps must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("auxiliary Adam weight_decay must be finite and non-negative")
        return self

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def zeropower_via_newtonschulz(
    gradient: torch.Tensor,
    *,
    ns_steps: int = 5,
    coefficients: Sequence[float] = DEFAULT_NS_COEFFICIENTS,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the polar factor of one matrix gradient.

    The quintic coefficients intentionally approximate ``U V^T`` rather than
    converging exactly to it.  Float32 is used internally even when the Chizat
    validation harness runs in float64.
    """
    if gradient.ndim != 2:
        raise ValueError("Muon orthogonalization requires a 2D gradient")
    if gradient.is_sparse:
        raise ValueError("Muon does not support sparse gradients")
    if torch.is_complex(gradient):
        raise ValueError("Muon does not support complex gradients")
    if not 1 <= ns_steps < 100:
        raise ValueError("ns_steps must be in [1, 100)")
    if len(coefficients) != 3:
        raise ValueError("coefficients must contain exactly three values")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    original_dtype = gradient.dtype
    x = gradient.to(dtype=torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (torch.linalg.vector_norm(x) + eps)
    a, b, c = (float(value) for value in coefficients)
    for _ in range(ns_steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.mT
    return x.to(dtype=original_dtype)


def learning_rate_adjustment(mode: str, shape: Sequence[int]) -> float:
    if mode not in ADJUSTMENT_MODES:
        raise ValueError(f"unsupported Muon adjustment: {mode}")
    if len(shape) != 2 or min(int(shape[0]), int(shape[1])) <= 0:
        raise ValueError("Muon adjustment requires a positive matrix shape")
    rows, columns = int(shape[0]), int(shape[1])
    if mode == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, columns))
    if mode == "original":
        return math.sqrt(max(1.0, rows / columns))
    if mode == "spectral_unclamped":
        return math.sqrt(rows / columns)
    return 1.0


def muon_direction(
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    config: MuonConfig,
) -> torch.Tensor:
    """Update the momentum state and return the unscaled Muon direction."""
    config.validate()
    if gradient.shape != momentum_buffer.shape:
        raise ValueError("gradient and momentum buffer shapes must match")
    momentum_buffer.lerp_(gradient, 1.0 - config.momentum)
    if config.nesterov:
        direction = torch.lerp(gradient, momentum_buffer, config.momentum)
    else:
        direction = momentum_buffer
    return zeropower_via_newtonschulz(
        direction,
        ns_steps=config.ns_steps,
        coefficients=config.ns_coefficients,
        eps=config.eps,
    )


class SingleDeviceMuon(torch.optim.Optimizer):
    """Small, state-dict-compatible Muon for explicitly routed 2D matrices."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[Mapping[str, object]],
        *,
        lr: float = 0.02,
        config: MuonConfig = MuonConfig(),
    ) -> None:
        config.validate()
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("Muon lr must be finite and positive")
        defaults = {"lr": float(lr), **config.to_dict()}
        super().__init__(params, defaults)
        for group in self.param_groups:
            group_config = self._config_for_group(group)
            group_config.validate()
            group_lr = float(group["lr"])
            if not math.isfinite(group_lr) or group_lr < 0.0:
                raise ValueError("Muon group lr must be finite and non-negative")
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        f"Muon only supports routed 2D matrices, got {tuple(parameter.shape)}"
                    )

    @staticmethod
    def _config_for_group(group: Mapping[str, object]) -> MuonConfig:
        return MuonConfig(
            momentum=float(group["momentum"]),
            nesterov=bool(group["nesterov"]),
            ns_steps=int(group["ns_steps"]),
            ns_coefficients=tuple(float(value) for value in group["ns_coefficients"]),
            eps=float(group["eps"]),
            weight_decay=float(group["weight_decay"]),
            adjustment=str(group["adjustment"]),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            config = self._config_for_group(group)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        parameter.grad, memory_format=torch.preserve_format
                    )
                direction = muon_direction(
                    parameter.grad, state["momentum_buffer"], config
                )
                base_lr = float(group["lr"])
                if config.weight_decay:
                    parameter.mul_(1.0 - base_lr * config.weight_decay)
                effective_lr = base_lr * learning_rate_adjustment(
                    config.adjustment, parameter.shape
                )
                parameter.add_(direction, alpha=-effective_lr)
        return loss


def chizat_muon_learning_rates(
    rule: str,
    *,
    L: int,
    M: int,
    D: int,
    eta: float,
) -> Dict[str, float]:
    """Return raw group rates for the Chizat Muon/Adam hybrid.

    Under RMS-matched Muon, a matrix's post-adjustment update has RMS
    approximately ``0.2 * raw_lr``.  For the scalar mean-field unembed
    ``R_d = O(D^-1)``:

    * a U update contracts once with W and R and is O(raw_lr_U);
    * a W update contracts with R and is O(raw_lr_W / sqrt(D));
    * an Adam embed update is O(raw_lr_embed) in function space;
    * the Adam unembed kernel is O(D), requiring raw_lr_unembed = eta/D.

    Hence the primary coordinate has rates ``(eta, eta, sqrt(D) eta,
    eta/D)`` for ``(embed, U, W, unembed)`` and no extra L or M power.
    """
    if rule not in TRANSFER_RULES:
        raise ValueError(f"unknown Chizat Muon transfer rule: {rule}")
    if min(L, M, D) <= 0:
        raise ValueError("L, M, and D must be positive")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive")
    rates = {
        "embed": eta,
        "U": eta,
        "W": math.sqrt(D) * eta,
        "unembed": eta / D,
    }
    if rule == "wrong_constant_W":
        rates["W"] = eta
    elif rule == "wrong_W_D":
        rates["W"] = D * eta
    elif rule == "wrong_sgd_LMD":
        rates["U"] = L * M * eta / D
        rates["W"] = L * M * D * eta
    elif rule == "freeze_embed":
        rates["embed"] = 0.0
    elif rule == "freeze_unembed":
        rates["unembed"] = 0.0
    elif rule == "wrong_constant_unembed":
        rates["unembed"] = eta
    return {key: float(value) for key, value in rates.items()}


def validate_semantic_partition(
    all_parameters: Sequence[torch.Tensor],
    muon_parameters: Sequence[torch.Tensor],
    auxiliary_parameters: Sequence[torch.Tensor],
) -> None:
    all_ids = [id(parameter) for parameter in all_parameters]
    routed_ids = [id(parameter) for parameter in (*muon_parameters, *auxiliary_parameters)]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("model parameter list contains duplicates")
    if len(set(routed_ids)) != len(routed_ids):
        raise ValueError("optimizer routing assigns a parameter more than once")
    missing = set(all_ids) - set(routed_ids)
    extra = set(routed_ids) - set(all_ids)
    if missing or extra:
        raise ValueError(
            f"optimizer routing mismatch: {len(missing)} missing, {len(extra)} extra"
        )


class ChizatMuonAdam:
    """One checkpointable optimizer facade for the Chizat semantic groups."""

    def __init__(
        self,
        net,
        rates: Mapping[str, float],
        *,
        muon_config: MuonConfig = MuonConfig(),
        aux_config: AuxAdamConfig = AuxAdamConfig(),
    ) -> None:
        muon_config.validate()
        aux_config.validate()
        expected = {"embed", "U", "W", "unembed"}
        if set(rates) != expected:
            raise ValueError(f"rates must have exactly these roles: {sorted(expected)}")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in rates.values()):
            raise ValueError("all optimizer rates must be finite and non-negative")
        groups = net.parameter_groups()
        muon_parameters = [*groups["U"], *groups["W"]]
        auxiliary_parameters = [*groups["embed"], *groups["unembed"]]
        validate_semantic_partition(net.params(), muon_parameters, auxiliary_parameters)
        self.roles = groups
        self.muon_config = muon_config
        self.aux_config = aux_config
        self.rates = {key: float(value) for key, value in rates.items()}
        self.muon = SingleDeviceMuon(
            [
                {"params": groups["U"], "lr": self.rates["U"], "role": "U"},
                {"params": groups["W"], "lr": self.rates["W"], "role": "W"},
            ],
            lr=1.0,
            config=muon_config,
        )
        self.auxiliary = torch.optim.Adam(
            [
                {"params": groups["embed"], "lr": self.rates["embed"], "role": "embed"},
                {
                    "params": groups["unembed"],
                    "lr": self.rates["unembed"],
                    "role": "unembed",
                },
            ],
            betas=(aux_config.beta1, aux_config.beta2),
            eps=aux_config.eps,
            weight_decay=aux_config.weight_decay,
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.auxiliary.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.auxiliary.step()

    def state_dict(self) -> Dict[str, object]:
        return {
            "contract_version": 1,
            "rates": dict(self.rates),
            "muon_config": self.muon_config.to_dict(),
            "aux_config": self.aux_config.to_dict(),
            "muon": self.muon.state_dict(),
            "auxiliary": self.auxiliary.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("contract_version", -1)) != 1:
            raise ValueError("unsupported Chizat Muon optimizer state contract")
        if dict(state["rates"]) != self.rates:
            raise ValueError("checkpoint optimizer rates do not match")
        if dict(state["muon_config"]) != self.muon_config.to_dict():
            raise ValueError("checkpoint Muon configuration does not match")
        if dict(state["aux_config"]) != self.aux_config.to_dict():
            raise ValueError("checkpoint auxiliary Adam configuration does not match")
        self.muon.load_state_dict(state["muon"])
        self.auxiliary.load_state_dict(state["auxiliary"])
