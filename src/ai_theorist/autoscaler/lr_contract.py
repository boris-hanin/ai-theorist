from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from torch import nn


@dataclass(frozen=True)
class LearningRateTheory:
    """A source-pinned contract for one architecture/optimizer pair."""

    contract_id: str
    architecture: str
    optimizer: str
    source_title: str
    source_url: str
    source_version: str
    base_coordinate: str
    applicability: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def theory_group(
    *,
    name: str,
    params: Iterable[nn.Parameter],
    lr: float,
    lr_formula: str,
    theory: LearningRateTheory,
    scale_factors: Mapping[str, float],
    eps: Optional[float] = None,
    eps_formula: Optional[str] = None,
) -> Dict[str, object]:
    """Build an optimizer group that carries its derivation into the manifest."""

    group: Dict[str, object] = {
        "name": name,
        "params": list(params),
        "lr": float(lr),
        "lr_formula": lr_formula,
        "theory_contract_id": theory.contract_id,
        "theory_source": theory.source_url,
        "scale_factors": {key: float(value) for key, value in scale_factors.items()},
    }
    if eps is not None:
        group["eps"] = float(eps)
    if eps_formula is not None:
        group["eps_formula"] = eps_formula
    return group


def audit_optimizer_groups(
    model: nn.Module,
    groups: Sequence[Mapping[str, object]],
    theory: LearningRateTheory,
) -> Dict[str, object]:
    """Prove complete, disjoint assignment of trainable tensors to LR groups.

    The returned payload is JSON-safe and is intended to be copied verbatim into
    every experiment manifest.  A malformed group contract fails before an
    optimizer or a GPU is touched.
    """

    if not groups:
        raise ValueError("optimizer parameter groups cannot be empty")
    names_by_id = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected = set(names_by_id)
    assigned: Dict[int, str] = {}
    seen_group_names = set()
    rows: List[Dict[str, object]] = []
    for group in groups:
        group_name = str(group.get("name", ""))
        if not group_name or group_name in seen_group_names:
            raise ValueError(f"optimizer group name must be unique and nonempty: {group_name!r}")
        seen_group_names.add(group_name)
        if group.get("theory_contract_id") != theory.contract_id:
            raise ValueError(f"{group_name} does not cite theory contract {theory.contract_id}")
        if not str(group.get("lr_formula", "")):
            raise ValueError(f"{group_name} is missing its learning-rate formula")
        rate = float(group["lr"])
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"{group_name} has a non-positive or non-finite learning rate")
        epsilon = group.get("eps")
        if epsilon is not None and (not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0):
            raise ValueError(f"{group_name} has a non-positive or non-finite Adam epsilon")
        parameters = list(group.get("params", []))  # type: ignore[arg-type]
        if not parameters:
            raise ValueError(f"{group_name} has no parameters")
        parameter_names = []
        parameter_count = 0
        shapes = []
        for parameter in parameters:
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(f"{group_name} contains a non-Parameter value")
            identifier = id(parameter)
            if identifier not in expected:
                raise ValueError(f"{group_name} contains a frozen or foreign parameter")
            if identifier in assigned:
                raise ValueError(
                    f"parameter {names_by_id[identifier]} is assigned to both "
                    f"{assigned[identifier]} and {group_name}"
                )
            assigned[identifier] = group_name
            parameter_names.append(names_by_id[identifier])
            parameter_count += parameter.numel()
            shapes.append(list(parameter.shape))
        rows.append(
            {
                "name": group_name,
                "learning_rate": rate,
                "learning_rate_formula": str(group["lr_formula"]),
                "adam_epsilon": None if epsilon is None else float(epsilon),
                "adam_epsilon_formula": group.get("eps_formula"),
                "scale_factors": dict(group.get("scale_factors", {})),
                "parameter_names": parameter_names,
                "parameter_shapes": shapes,
                "parameter_count": parameter_count,
            }
        )
    missing = expected - set(assigned)
    if missing:
        raise ValueError(
            "trainable parameters missing from optimizer groups: "
            + ", ".join(sorted(names_by_id[identifier] for identifier in missing))
        )
    return {
        "theory": theory.to_dict(),
        "complete": True,
        "disjoint": True,
        "trainable_parameter_tensors": len(expected),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "groups": rows,
    }


def raw_group_rates(groups: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    return {str(group["name"]): float(group["lr"]) for group in groups}


def raw_group_epsilons(groups: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    return {
        str(group["name"]): float(group["eps"])
        for group in groups
        if group.get("eps") is not None
    }
