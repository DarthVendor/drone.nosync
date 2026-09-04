"""Physical environments.

Composable obstacle fields, scene presets, compositional mixtures, and CAD import.

`environments/` may import `config`/`util` and nothing else in the package -- in
particular it never imports `systems/`, so the world and the robot stay separable.
"""
from __future__ import annotations

from .base import (
    EPS, Environment, Mixture, ObstacleGroup, State, march,
)
from .primitives import Gate, Hoops, Pillars, Walls

GROUPS = {"pillars": Pillars, "walls": Walls, "gate": Gate, "hoops": Hoops}

#: named scenes -- each is just a list of groups, so a new one is one line
PRESETS = {
    "empty":    lambda: [],
    "sparse":   lambda: [Pillars(n=3, r_lo=0.20, r_hi=0.34)],
    "pillars":  lambda: [Pillars(n=6)],
    "forest":   lambda: [Pillars(n=11, r_lo=0.14, r_hi=0.28, extent=2.8)],
    "slalom":   lambda: [Pillars(n=4, extent=1.8, r_lo=0.26, r_hi=0.40)],
    "gate":     lambda: [Gate()],
    "walls":    lambda: [Walls(n=2)],
    "cluttered": lambda: [Pillars(n=5), Walls(n=2, length=1.3)],
    "gate_forest": lambda: [Gate(), Pillars(n=5, r_lo=0.14, r_hi=0.24)],
    "hoops":         lambda: [Hoops(n=2)],
    "hoop_course":   lambda: [Hoops(n=3, radius=0.55)],
    "hoops_upright": lambda: [Hoops(n=3, radius=0.55, tilt=0.0)],
    "hoops_flat":    lambda: [Hoops(n=3, radius=0.6, tilt=1.5707963267948966,
                                    tilt_lo=1.2)],
    "hoop_slalom":   lambda: [Hoops(n=3, radius=0.45),
                              Pillars(n=3, r_lo=0.16, r_hi=0.26)],
}


#: the regimes a controller trains on, one at a time
REGIMES = lambda: [Pillars(n=6), Walls(n=2), Gate(), Hoops(n=2, radius=0.55)]

#: compositional splits: identical group list, different number live per episode
MIXTURES = {
    "train_mix": (1, 1),      # one regime per episode -- seen in isolation
    "pair_mix":  (2, 2),      # two at once -- never seen combined
    "test_mix":  (2, 3),      # two or three -- the held-out compositions
}


def make_environment(name: str = "empty", **kw) -> Environment:
    """Build a named scene.  `make_environment("forest")`."""
    if name in MIXTURES:
        lo, hi = MIXTURES[name]
        return Mixture(REGIMES(), n_active=(lo, hi), name=name)
    if name not in PRESETS:
        raise KeyError(f"unknown environment {name!r}; "
                       f"presets: {sorted(PRESETS)}; mixtures: {sorted(MIXTURES)}")
    return Environment(PRESETS[name](), name=name)


def make_group(kind: str, **kw) -> ObstacleGroup:
    if kind not in GROUPS:
        raise KeyError(f"unknown obstacle group {kind!r}; registered: {sorted(GROUPS)}")
    return GROUPS[kind](**kw)


def load_dxf(path, **kw) -> Environment:
    """Import a CAD drawing as an environment.  See `environments.cad`."""
    from .cad import dxf_to_environment

    return dxf_to_environment(path, **kw)


__all__ = [
    "Environment", "Mixture", "ObstacleGroup", "State", "EPS", "march",
    "Pillars", "Walls", "Gate", "Hoops",
    "GROUPS", "PRESETS", "MIXTURES", "REGIMES",
    "make_environment", "make_group", "load_dxf",
]
