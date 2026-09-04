"""Frozen configuration dataclasses.

This module holds NO logic: no validation, no IO, no derived quantities that
require computation.  Anything that needs to *do* something with a config lives
in `util.py` (loading) or in the module that consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RolloutCfg:
    """Episode integration and cost weights."""

    dt: float = 0.02
    ep_steps: int = 250
    n_eps: int = 2                # episodes per genome; shared across population
    lambda_e: float = 0.02        # effort weight
    lambda_s: float = 0.10        # plant-specific shaping weight
    dead_cost: float = 5.0        # per-second cost accrued by a crashed vehicle


@dataclass(frozen=True)
class ESCfg:
    """Evolution-strategy hyperparameters."""

    pop: int = 16                 # must be even (mirrored sampling)
    gens: int = 6
    sigma0: float = 0.08
    sigma_min: float = 1e-3
    sigma_max: float = 0.5
    grow: float = 1.06            # step-size multiplier on improvement
    shrink: float = 0.97          # ... and on stagnation
    elite_frac: float = 0.25

    # --- search strategy --------------------------------------------------
    #: "es" = distribution-based (one mean, mirrored sampling, rank recombination)
    #: "ga" = persistent population with tournament selection + crossover
    strategy: str = "es"
    success_target: float = 0.20   # 1/5th rule: grow sigma above this hit rate
    elitism: int = 2               # ga: individuals copied through unchanged
    tournament_k: int = 3          # ga: tournament size
    crossover_rate: float = 0.7    # ga: probability a child is a recombination
    blx_alpha: float = 0.5         # ga: BLX-alpha interpolation slack

    whiten: bool = True           # False => P = I, the isotropic baseline
    metric_every: int = 5         # generations between metric refreshes
    metric_states: int = 96       # states subsampled per refresh
    ridge: float = 1e-3           # added to normalized eigenvalues


@dataclass(frozen=True)
class Config:
    """Top-level experiment description."""

    system: str = "quadrotor"
    trainable: str = "energy_shaping"
    task: str = "waypoint_pair"
    seed: int = 0
    dtype: str = "float64"
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    es: ESCfg = field(default_factory=ESCfg)
