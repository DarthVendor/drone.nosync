"""The `Trainable` abstraction: an evolvable controller.

A trainable owns its own genome layout and nothing else.  It never learns which
plant it is driving beyond the descriptors on `LagrangianSystem`, and the system
never learns the genome layout.  The genome is laid out as

    [ policy params | allocator params ]
      policy_dim        system.allocator_dim

so the trainable owns the first slice and the system owns the second.  That split
is what lets one genome structure serve a fully-actuated arm and an underactuated
quadrotor without either side knowing about the other.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor

from ..systems.base import LagrangianSystem, State


class Trainable(ABC):
    #: does `forward` reduce to the pure gravity feedforward at the goal, for
    #: every genome?  True for the energy-shaping family, False for a free-form
    #: network.  `test_conformance` reads this to know whether the
    #: equilibrium-by-construction check applies.
    equilibrium_exact: bool = False

    def __init__(self, system: LagrangianSystem):
        self.system = system
        self.dtype = system.dtype
        self.device = system.device

    # --- genome layout ------------------------------------------------------
    @property
    @abstractmethod
    def policy_dim(self) -> int:
        """Genome slots this trainable needs, EXCLUDING allocator params."""

    @property
    def dim(self) -> int:
        return (self.policy_dim + self.system.allocator_dim
                + self.system.residual_dim)

    def residual_slice(self, theta: Tensor):
        """The learned-dynamics tail of the genome, or None if the plant has none.

        Note what this implies for the metric: residual parameters change the
        PLANT, not the controller map, so du/dtheta is zero on them exactly as it
        is for an episode-level multiplier.  They contribute a zero block to G.
        With `null_mode="cap"` that block is treated isotropically, which is the
        right default; with a small ridge it would instead be amplified.
        """
        r = self.system.residual_dim
        return theta[..., -r:] if r else None

    def split(self, theta: Tensor):
        """(policy slice, allocator slice).  Trailing-dim indexed, so it works
        batched or under vmap."""
        a = self.system.allocator_dim
        return (theta[..., : self.policy_dim],
                theta[..., self.policy_dim : self.policy_dim + a])

    @classmethod
    def supports(cls, system) -> bool:
        """Is this trainable meant for this plant?

        Generic trainables answer True for everything.  A plant-specific agent
        narrows it, so `test_conformance` can skip pairs that were never intended
        instead of reporting them as failures.
        """
        return True

    def segments(self):
        """Crossover-safe blocks of the genome, as slices.

        A trainable whose genome decomposes into independently meaningful units
        (Lagrangian terms, network layers) exposes them here so that
        `operators.segment_crossover` can exchange whole units between parents
        instead of mixing coordinates that only mean something together.
        Default: the genome is one indivisible block.
        """
        return [slice(0, self.dim)]

    @abstractmethod
    def init(self) -> Tensor:
        """[dim] physical prior.  Must fly badly, not fail to fly."""

    # --- the controller map -------------------------------------------------
    @abstractmethod
    def forward(self, theta: Tensor, s: State, goal: Tensor) -> Tensor:
        """SINGLE-SAMPLE.  theta [dim], unbatched state, goal [task_dim].
        Returns a generalized force [n_force].

        Batching is vmap's job; differentiability is jacrev's business.  So:
        no in-place ops, no `.item()`, no Python `if` on tensor values, every
        normalize floored.  `test_conformance` enforces this by tracing
        `vmap(jacrev(forward))` over every registered pair.
        """

    # --- reporting ----------------------------------------------------------
    def describe(self, theta: Tensor) -> Dict[str, float]:
        """Scalars to log per generation.  Default: nothing."""
        return {}

    def invariants(self, theta: Tensor) -> Dict[str, Tensor]:
        """Quantities the conformance test asserts on.  Default: nothing."""
        return {}

    def _t(self, x) -> Tensor:
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)
