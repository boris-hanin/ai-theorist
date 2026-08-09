"""Audited single-device Muon with semantically routed auxiliary Adam groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch


DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
ADJUSTMENT_MODES = ("match_rms_adamw", "original", "spectral_unclamped", "none")


@dataclass(frozen=True)
class MuonConfig:
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS
    epsilon: float = 1e-7
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
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("Muon epsilon must be finite and positive")
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
    epsilon: float = 1e-10
    weight_decay: float = 0.0

    def validate(self) -> "AuxAdamConfig":
        for name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"auxiliary Adam {name} must be finite and in [0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("auxiliary Adam epsilon must be finite and positive")
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
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Approximate the matrix polar factor using float32 Newton--Schulz steps."""
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
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    original_dtype = gradient.dtype
    x = gradient.to(dtype=torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (torch.linalg.vector_norm(x) + epsilon)
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
    config.validate()
    if gradient.shape != momentum_buffer.shape:
        raise ValueError("gradient and momentum buffer shapes must match")
    momentum_buffer.lerp_(gradient, 1.0 - config.momentum)
    direction = (
        torch.lerp(gradient, momentum_buffer, config.momentum)
        if config.nesterov
        else momentum_buffer
    )
    return zeropower_via_newtonschulz(
        direction,
        ns_steps=config.ns_steps,
        coefficients=config.ns_coefficients,
        epsilon=config.epsilon,
    )


class SingleDeviceMuon(torch.optim.Optimizer):
    """Checkpointable Muon for explicitly routed two-dimensional parameters."""

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
            epsilon=float(group["epsilon"]),
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
                direction = muon_direction(parameter.grad, state["momentum_buffer"], config)
                base_lr = float(group["lr"])
                if config.weight_decay:
                    parameter.mul_(1.0 - base_lr * config.weight_decay)
                effective_lr = base_lr * learning_rate_adjustment(
                    config.adjustment, parameter.shape
                )
                parameter.add_(direction, alpha=-effective_lr)
        return loss


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


class HybridMuonAdam:
    """One optimizer facade for Muon residual matrices and Adam boundaries."""

    def __init__(
        self,
        all_parameters: Sequence[torch.Tensor],
        roles: Mapping[str, Sequence[torch.Tensor]],
        rates: Mapping[str, float],
        *,
        muon_config: MuonConfig = MuonConfig(),
        auxiliary_config: AuxAdamConfig = AuxAdamConfig(),
    ) -> None:
        expected = {"embed", "U", "W", "unembed"}
        if set(roles) != expected or set(rates) != expected:
            raise ValueError(f"roles and rates must contain exactly {sorted(expected)}")
        if any(not math.isfinite(float(rate)) or float(rate) < 0.0 for rate in rates.values()):
            raise ValueError("all optimizer rates must be finite and non-negative")
        muon_config.validate()
        auxiliary_config.validate()
        muon_parameters = [*roles["U"], *roles["W"]]
        auxiliary_parameters = [*roles["embed"], *roles["unembed"]]
        validate_semantic_partition(all_parameters, muon_parameters, auxiliary_parameters)
        self.rates = {name: float(rate) for name, rate in rates.items()}
        self.muon_config = muon_config
        self.auxiliary_config = auxiliary_config
        self.muon = SingleDeviceMuon(
            [
                {"params": list(roles["U"]), "lr": self.rates["U"], "name": "U"},
                {"params": list(roles["W"]), "lr": self.rates["W"], "name": "W"},
            ],
            lr=1.0,
            config=muon_config,
        )
        self.auxiliary = torch.optim.Adam(
            [
                {
                    "params": list(roles["embed"]),
                    "lr": self.rates["embed"],
                    "name": "embed",
                },
                {
                    "params": list(roles["unembed"]),
                    "lr": self.rates["unembed"],
                    "name": "unembed",
                },
            ],
            betas=(auxiliary_config.beta1, auxiliary_config.beta2),
            eps=auxiliary_config.epsilon,
            weight_decay=auxiliary_config.weight_decay,
        )

    @property
    def param_groups(self):
        return [*self.muon.param_groups, *self.auxiliary.param_groups]

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
            "auxiliary_config": self.auxiliary_config.to_dict(),
            "muon": self.muon.state_dict(),
            "auxiliary": self.auxiliary.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("contract_version", -1)) != 1:
            raise ValueError("unsupported hybrid Muon optimizer state contract")
        if dict(state["rates"]) != self.rates:
            raise ValueError("checkpoint optimizer rates do not match")
        if dict(state["muon_config"]) != self.muon_config.to_dict():
            raise ValueError("checkpoint Muon configuration does not match")
        if dict(state["auxiliary_config"]) != self.auxiliary_config.to_dict():
            raise ValueError("checkpoint auxiliary Adam configuration does not match")
        self.muon.load_state_dict(state["muon"])
        self.auxiliary.load_state_dict(state["auxiliary"])
