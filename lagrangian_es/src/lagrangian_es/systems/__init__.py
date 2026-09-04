"""Physics simulation subpackage.

`systems/` must not import from `trainables/`, `rollout.py`, or anything above
them in the dependency order.  If a cycle appears, the seam is in the wrong place.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from .base import LagrangianSystem, State
from .so3 import hat, log_so3, renormalize, rodrigues, vee

SYSTEMS: Dict[str, Type[LagrangianSystem]] = {}


def register_system(name: str) -> Callable:
    def deco(cls):
        if name in SYSTEMS:
            raise KeyError(f"system {name!r} already registered")
        SYSTEMS[name] = cls
        return cls

    return deco


def make_system(name: str, **kw) -> LagrangianSystem:
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; registered: {sorted(SYSTEMS)}")
    return SYSTEMS[name](**kw)


from .planar_quad import PlanarQuadrotor  # noqa: E402
from .quadrotor import QuadrotorSE3        # noqa: E402
from .two_link_arm import TwoLinkArm       # noqa: E402

register_system("quadrotor")(QuadrotorSE3)
register_system("planar_quad")(PlanarQuadrotor)
register_system("two_link_arm")(TwoLinkArm)

__all__ = [
    "LagrangianSystem", "State", "SYSTEMS", "register_system", "make_system",
    "QuadrotorSE3", "PlanarQuadrotor", "TwoLinkArm", "hat", "vee", "rodrigues", "log_so3", "renormalize",
]
