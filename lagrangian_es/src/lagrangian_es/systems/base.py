"""The `LagrangianSystem` abstraction.

State is an opaque pytree: `State = dict[str, Tensor]`, every leaf carrying a
leading batch dim B.  Only the owning system interprets the keys -- `QuadrotorSE3`
uses {p, v, R, om}, `TwoLinkArm` uses {q, dq}.  Downstream code touches state only
through the accessors declared here, which is what keeps `rollout.py` and
`metric.py` plant-agnostic.

Every method must be written batch-rank-agnostically (index with `...`), because
`allocate` is called from inside `vmap(jacrev(...))` where the batch dim is gone.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor

State = Dict[str, Tensor]


class LagrangianSystem(ABC):
    """A plant expressed through its mechanics rather than through its equations."""

    # --- static descriptors -------------------------------------------------
    n_force: int = 0          # dim of the generalized force u
    task_dim: int = 0         # dim of the space the shaped potential lives on
    allocator_dim: int = 0    # genome slots the allocator needs (0 if fully actuated)
    dense_mass: bool = False  # False => inv_mass returns [..., n_force] (diagonal)
    state_keys: tuple = ()    # documentation only; nothing downstream reads it

    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    # --- simulation ---------------------------------------------------------
    @abstractmethod
    def reset(self, B: int, gen: torch.Generator) -> State:
        """Initial state batch of size B, drawn from `gen`."""

    @abstractmethod
    def step(self, s: State, u: Tensor, dt: float) -> State:
        """One integration step.  Pure -- must not mutate `s`.  u: [..., n_force]."""

    @abstractmethod
    def alive(self, s: State) -> Tensor:
        """[...] bool.  False = diverged/crashed; the rollout freezes these."""

    # --- mechanics (this is the part the metric consumes) -------------------
    @abstractmethod
    def inv_mass(self, s: State) -> Tensor:
        """M(q)^-1 in generalized-force coordinates.

        [..., n_force] when diagonal, [..., n_force, n_force] when dense; declare
        which with the `dense_mass` class attribute.

        CONSTANT for a rigid body in body coordinates -- `QuadrotorSE3` ignores its
        argument.  Genuinely state-dependent for articulated systems --
        `TwoLinkArm` does not.  The scope caveat of the whole method lives here:
        only on a plant where this varies does the metric track something no
        single global covariance can represent.
        """

    @abstractmethod
    def gravity_force(self, s: State) -> Tensor:
        """g(q): the task-space force that holds the system against gravity.
        [..., task_dim].  The controller adds this as feedforward."""

    # --- the underactuation seam --------------------------------------------
    @abstractmethod
    def allocate(self, F_des: Tensor, s: State, phi: Tensor) -> Tensor:
        """Map a desired task-space force [..., task_dim] onto a realizable
        generalized force [..., n_force], given allocator params phi
        [..., allocator_dim].

        Fully actuated: identity, `allocator_dim = 0`.
        Quadrotor: thrust along body-z plus a geometric SO(3) attitude loop whose
        gains live in `phi`.

        Must be vmap- and jacrev-safe.
        """

    # --- task-space accessors -----------------------------------------------
    @abstractmethod
    def task_position(self, s: State) -> Tensor:
        """[..., task_dim]"""

    @abstractmethod
    def task_velocity(self, s: State) -> Tensor:
        """[..., task_dim]"""

    @abstractmethod
    def nominal_state(self, x: Tensor, v: Tensor) -> State:
        """A state with `task_position(s) == x` and `task_velocity(s) == v`, in
        the plant's nominal configuration otherwise (level attitude, zero rates).

        This is the plant-agnostic way for a caller to say "put the system here,
        moving like this".  `test_conformance` uses it to place a system exactly
        at its goal (where every equilibrium-by-construction trainable must emit
        the pure gravity feedforward) and absurdly far from it (where the bounded
        potential must still emit something finite).
        """

    # --- cost hooks ---------------------------------------------------------
    def effort(self, u: Tensor, s: State) -> Tensor:
        """Counterforce in the M^-1 metric, gravity feedforward removed.  [...].

        The default assumes the plant is fully actuated (`n_force == task_dim`)
        so that the gravity feedforward is directly comparable with u.  Plants
        where it is not -- the quadrotor, whose u is (thrust, torque) -- override.
        """
        g = self.gravity_force(s)
        if u.shape[-1] != g.shape[-1]:
            raise NotImplementedError(
                f"{type(self).__name__}: n_force={u.shape[-1]} != task_dim={g.shape[-1]}; "
                "override effort() for an underactuated plant."
            )
        du = u - g
        Minv = self.inv_mass(s)
        if self.dense_mass:
            return torch.einsum("...i,...ij,...j->...", du, Minv, du)
        return (du * Minv * du).sum(dim=-1)

    def shaping_cost(self, s: State) -> Tensor:
        """Plant-specific regularizer, [...].  Default: zeros."""
        return torch.zeros_like(self.task_position(s)[..., 0])

    # --- diagnostics --------------------------------------------------------
    def saturation(self, u: Tensor, s: State) -> Tensor:
        """Fraction of actuator channels at a bound, [...] in [0, 1].

        `metric.py` logs the mean of this: a genome living at saturation yields a
        rank-deficient G in exactly the directions that matter, and the ridge
        hides it.  Default: never saturated.
        """
        return torch.zeros_like(u[..., 0])

    def describe(self) -> dict:
        """Static scalars worth recording alongside a run."""
        return {
            "system": type(self).__name__,
            "n_force": self.n_force,
            "task_dim": self.task_dim,
            "allocator_dim": self.allocator_dim,
            "dense_mass": self.dense_mass,
        }

    # --- convenience --------------------------------------------------------
    def _t(self, x) -> Tensor:
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def batch_shape(self, s: State) -> tuple:
        """Leading dims of a state, i.e. everything before the accessor's own dim."""
        return tuple(self.task_position(s).shape[:-1])
