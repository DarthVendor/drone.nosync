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
    residual_dim: int = 0     # genome slots for a LEARNED DYNAMICS residual (0 = none)
    dense_mass: bool = False  # False => inv_mass returns [..., n_force] (diagonal)
    state_keys: tuple = ()    # documentation only; nothing downstream reads it

    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    # --- simulation ---------------------------------------------------------
    @abstractmethod
    def reset(self, B: int, gen: torch.Generator) -> State:
        """Initial state batch of size B, drawn from `gen`."""

    @abstractmethod
    def step(self, s: State, u: Tensor, dt: float, params: Tensor = None) -> State:
        """One integration step.  Pure -- must not mutate `s`.  u: [..., n_force].

        `params` [..., residual_dim] carries the learned dynamics residual when the
        plant declares one.  It is passed in rather than stored so `step` stays
        pure, and it is ignored by plants with `residual_dim == 0`.
        """

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

    def task_mass(self, s: State) -> Tensor:
        """M in TASK coordinates, [..., task_dim, task_dim].

        Distinct from `inv_mass`, which is in generalized-FORCE coordinates: on
        the quadrotor those are (thrust, torque) and 4-dimensional, while the
        shaped potential lives in the 3-dimensional task space.  Only kinetic
        shaping needs this, so the default derives it from `inv_mass` when the
        two coordinate systems coincide and asks for an override otherwise.
        """
        if self.n_force != self.task_dim:
            raise NotImplementedError(
                f"{type(self).__name__}: task_dim={self.task_dim} != n_force={self.n_force}; "
                "override task_mass() to use kinetic shaping on this plant."
            )
        Minv = self.inv_mass(s)
        if not self.dense_mass:
            Minv = torch.diag_embed(Minv)
        return torch.linalg.inv(Minv)

    def residual_init(self) -> Tensor:
        """[residual_dim] prior for the learned residual.

        Should start at the value that makes the residual IDENTICALLY ZERO, so a
        hybrid model begins as the pure constraint model and learns only the
        correction on top of it.
        """
        return torch.zeros(self.residual_dim, dtype=self.dtype, device=self.device)

    def residual_penalty(self, params: Tensor) -> Tensor:
        """Regularizer on the learned residual, [...].

        Co-evolving plant parameters against the controller's own reward is
        reward hacking unless something holds the plant honest: left free, the
        residual will happily invent forces that make the task easier.  This
        penalty is the minimum safeguard; identifying the residual against
        reference data is the real answer.  See README, "Hybrid contact".
        """
        if params is None or self.residual_dim == 0:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        return (params * params).sum(dim=-1)

    def potential_scale(self) -> float:
        """Suggested closed-loop stiffness for the shaped potential, m*omega_n^2.

        The system owns this for the same reason it owns the allocator prior: a
        stiffness sized for one plant's mass is meaningless on another's, and a
        prior that is three orders of magnitude too soft does not "fly badly", it
        fails to respond at all.  Default 3.0 reproduces the quadrotor's
        calibration (0.5 kg at omega_n ~ 2.4 rad/s).
        """
        return 3.0

    def damping_scale(self) -> float:
        """Suggested closed-loop damping K_d for the shaped potential.

        Returned directly rather than derived from a damping ratio, so a plant's
        calibration is exact rather than reconstructed: the quadrotor's 1.44 is a
        measured prototype value and the whole benchmark's difficulty is pinned to
        it.  Roughly 2*zeta*sqrt(K*m) with zeta ~ 0.6.
        """
        return 1.44

    def allocator_init(self) -> Tensor:
        """[allocator_dim] prior for the allocator's own genome slots.

        The system owns the allocator, so it owns the allocator's prior; a
        trainable must never hardcode a plant-shaped phi0.  Default: zeros.
        """
        return torch.zeros(self.allocator_dim, dtype=self.dtype, device=self.device)

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

    # --- rendering ----------------------------------------------------------
    def render_spec(self) -> dict:
        """Static description of what to draw, plant-agnostic.

        Keys:
          `dim`     -- 2 or 3, the drawing space
          `bodies`  -- one entry per rigid body: {type, size, ...}
                       "quadrotor" (box + arms + rotors), "box", "segment"
          `ground`  -- height of the ground plane, or None for no ground
          `scale`   -- a representative length, for camera framing

        The visualizer knows only this vocabulary, so a new plant becomes
        drawable by describing itself rather than by editing the renderer.
        """
        raise NotImplementedError(f"{type(self).__name__} declares no render_spec")

    def render_poses(self, s: State) -> Tensor:
        """Body poses, [..., n_bodies, K].

        K = 12 in 3-D: position (3) then the rotation matrix row-major (9).
        K = 3  in 2-D: (x, z, angle), angle measured from +x.

        Goals are drawn by pushing them through `nominal_state` and calling this,
        so a ghost of the target configuration comes out for free on every plant.
        """
        raise NotImplementedError(f"{type(self).__name__} declares no render_poses")

    def render_extras(self, s: State) -> dict:
        """Optional per-frame overlays, e.g. {"feet": [..., k, 2],
        "contact": [..., k]}.  Default: nothing."""
        return {}

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
