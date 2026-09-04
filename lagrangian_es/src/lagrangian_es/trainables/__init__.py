"""Evolvable objects.

`trainables/` may import from `systems/` (it needs the descriptors and the
allocator seam) but from nothing above it in the dependency order.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from .base import Trainable

TRAINABLES: Dict[str, Type[Trainable]] = {}


def register_trainable(name: str) -> Callable:
    def deco(cls):
        if name in TRAINABLES:
            raise KeyError(f"trainable {name!r} already registered")
        TRAINABLES[name] = cls
        return cls

    return deco


def make_trainable(name: str, system, **kw) -> Trainable:
    if name not in TRAINABLES:
        raise KeyError(f"unknown trainable {name!r}; registered: {sorted(TRAINABLES)}")
    return TRAINABLES[name](system, **kw)


from .embodied import (                    # noqa: E402
    AGENTS, ArmAgent, EmbodiedAgent, NavAgent, PlanarQuadrotorAgent,
    QuadrotorAgent, QuadrupedAgent,
)
from .energy_shaping import EnergyShaping  # noqa: E402
from .mlp import MLPPolicy                 # noqa: E402
from .pd_baseline import FixedPD           # noqa: E402

register_trainable("energy_shaping")(EnergyShaping)
register_trainable("pd")(FixedPD)
register_trainable("mlp")(MLPPolicy)
for _n, _c in AGENTS.items():
    register_trainable(_n)(_c)

__all__ = [
    "Trainable", "TRAINABLES", "register_trainable", "make_trainable",
    "EnergyShaping", "FixedPD", "MLPPolicy", "EmbodiedAgent", "QuadrotorAgent",
    "QuadrupedAgent", "ArmAgent", "PlanarQuadrotorAgent", "NavAgent",
]
