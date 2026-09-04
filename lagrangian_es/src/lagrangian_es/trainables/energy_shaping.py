"""`EnergyShaping` -- the proposal, as a composition of Lagrangian terms.

A genome decodes to a list of terms contributing to a desired Lagrangian
L_d = T_d - V_d, plus dissipation and path-level constraints, which together
define a desired task-space force

    F_des = g(q) - M M_d^-1 ( sum_i grad V_i(e, v, x) + K_d xdot ) .

The default term list -- three `GoalBowl`s and one `DissipationTerm` -- reproduces
the original fixed potential exactly, at the same 39 policy slots.  That is the
degenerate case: T_d = T (no kinetic shaping) and no constraint terms.

Why the list form is the right representation:

  * Lagrangians add and the Euler-Lagrange operator is linear in L, so a conic
    combination of nonnegative terms superposes forces and preserves
    positive-definiteness about the goal.  The equilibrium invariant survives
    composition for free.
  * Every term publishes a `certificate`, so `equilibrium_exact` is the
    CONJUNCTION of its terms' promises rather than a class constant.  Adding a
    barrier that overlaps the goal correctly withdraws the claim instead of
    silently falsifying it.
  * Terms are crossover units.  `segments()` exposes their boundaries so whole
    terms can be swapped between parents GP-style, which is safe precisely
    because each term independently preserves the invariant -- the structural
    recombination that weight-level crossover cannot give you, because weights
    have competing conventions and terms do not.

Gradients are closed-form throughout and never obtained by autograd: `forward`
is vmapped over a whole population every step, and jacrev differentiates it with
respect to theta, not to e.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor

from ..systems.base import LagrangianSystem, State
from .base import Trainable
from .terms import DissipationTerm, GoalBowl, KineticShaping, LagrangianTerm, make_term


class EnergyShaping(Trainable):
    def __init__(
        self,
        system: LagrangianSystem,
        terms: Optional[Sequence[LagrangianTerm]] = None,
        n_bowls: int = 3,
        w0: float = 1.0,
        a0: float = 1.0,
        d0: float = 1.2,
    ):
        super().__init__(system)
        self.d = int(system.task_dim)
        if terms is None:
            terms = [GoalBowl(self.d, w0=w0, a0=a0) for _ in range(n_bowls)]
            terms = list(terms) + [DissipationTerm(self.d, d0=d0)]
        self.terms: List[LagrangianTerm] = list(terms)
        if not self.terms:
            raise ValueError("EnergyShaping needs at least one term")

        self._bounds, n = [], 0
        for t in self.terms:
            self._bounds.append((n, n + t.dim))
            n += t.dim
        self._policy_dim = n
        self._has_kinetic = any(t.kinetic_shape(t.init(self.dtype, self.device), None)
                                is not None for t in self.terms)

    # --- genome layout ------------------------------------------------------
    @property
    def policy_dim(self) -> int:
        return self._policy_dim

    def term_slices(self, theta: Tensor):
        """Per-term views of a genome.  Trailing-dim indexed, so batched or vmapped."""
        return [theta[..., a:b] for a, b in self._bounds]

    def segments(self):
        """Crossover-safe blocks: one per term, plus the allocator.

        Whole terms may be exchanged between parents because each one preserves
        the composition invariant on its own; a block that cut across a term
        would not.
        """
        segs = [slice(a, b) for a, b in self._bounds]
        if self.system.allocator_dim:
            segs.append(slice(self.policy_dim, self.dim))
        return segs

    def init(self) -> Tensor:
        parts = [t.init(self.dtype, self.device) for t in self.terms]
        if self.system.allocator_dim:
            parts.append(self.system.allocator_init())
        return torch.cat(parts, dim=0)

    # --- what the composition promises --------------------------------------
    @property
    def equilibrium_exact(self) -> bool:
        """True only if EVERY term promises to leave the equilibrium at the goal.

        Evaluated with no goal in hand, so a barrier -- whose promise is
        conditional on the goal lying clear of its support -- withdraws the claim.
        `test_conformance` reads this and skips the equilibrium check, which is
        the honest outcome rather than a failure.
        """
        th = self.init()
        return all(t.certificate(s).get("zero_at_goal", False)
                   for t, s in zip(self.terms, self.term_slices(th)))

    def equilibrium_exact_for(self, theta: Tensor, goal: Tensor) -> bool:
        """The conditional version: does this genome leave THIS goal an equilibrium?"""
        return all(t.certificate(s, goal=goal).get("zero_at_goal", False)
                   for t, s in zip(self.terms, self.term_slices(theta)))

    # --- composed quantities ------------------------------------------------
    def potential(self, theta: Tensor, e: Tensor, v: Optional[Tensor] = None,
                  x: Optional[Tensor] = None) -> Tensor:
        """sum_i V_i >= 0.  Velocity and absolute position default to zero so the
        pure-potential reading stays available to tests and to plotting."""
        v = torch.zeros_like(e) if v is None else v
        x = e if x is None else x
        out = None
        for t, s in zip(self.terms, self.term_slices(theta)):
            p = t.potential(s, e, v, x)
            out = p if out is None else out + p
        return out

    def grad_potential(self, theta: Tensor, e: Tensor, v: Optional[Tensor] = None,
                       x: Optional[Tensor] = None) -> Tensor:
        v = torch.zeros_like(e) if v is None else v
        x = e if x is None else x
        out = None
        for t, s in zip(self.terms, self.term_slices(theta)):
            g = t.grad_potential(s, e, v, x)
            out = g if out is None else out + g
        return out

    def desired_mass(self, theta: Tensor, s: State) -> Optional[Tensor]:
        """M_d = M + sum_k W_k W_k^T, or None when no term shapes kinetic energy."""
        if not self._has_kinetic:
            return None
        x = self.system.task_position(s)
        extra = None
        for t, th in zip(self.terms, self.term_slices(theta)):
            k = t.kinetic_shape(th, x)
            if k is not None:
                extra = k if extra is None else extra + k
        if extra is None:
            return None
        return self.system.task_mass(s) + extra

    def damping(self, theta: Tensor) -> Tensor:
        """Total K_d over the dissipation terms."""
        out = None
        for t, s in zip(self.terms, self.term_slices(theta)):
            if isinstance(t, DissipationTerm):
                k = t.damping(s)
                out = k if out is None else out + k
        if out is None:
            shp = theta.shape[:-1] + (self.d, self.d)
            return torch.zeros(shp, dtype=theta.dtype, device=theta.device)
        return out

    def stiffness(self, theta: Tensor) -> Tensor:
        """Hessian of the composed potential at the goal, over bowl terms."""
        out = None
        for t, s in zip(self.terms, self.term_slices(theta)):
            if isinstance(t, GoalBowl):
                k = t.stiffness(s)
                out = k if out is None else out + k
        if out is None:
            shp = theta.shape[:-1] + (self.d, self.d)
            return torch.zeros(shp, dtype=theta.dtype, device=theta.device)
        return out

    # --- the controller map -------------------------------------------------
    def forward(self, theta: Tensor, s: State, goal: Tensor) -> Tensor:
        sysm = self.system
        x = sysm.task_position(s)
        v = sysm.task_velocity(s)
        e = x - goal

        bracket = None
        for t, th in zip(self.terms, self.term_slices(theta)):
            g = t.grad_potential(th, e, v, x)
            bracket = g if bracket is None else bracket + g

        Md = self.desired_mass(theta, s)
        if Md is not None:
            # u = g - M M_d^-1 (...), exact for constant M; see KineticShaping
            A = torch.linalg.solve(Md.transpose(-1, -2), sysm.task_mass(s).transpose(-1, -2))
            bracket = torch.einsum("...ij,...j->...i", A.transpose(-1, -2), bracket)

        F_des = sysm.gravity_force(s) - bracket
        return sysm.allocate(F_des, s, theta[..., self.policy_dim:])

    # --- reporting ----------------------------------------------------------
    def describe(self, theta: Tensor) -> dict:
        with torch.no_grad():
            out = {}
            for i, (t, s) in enumerate(zip(self.terms, self.term_slices(theta))):
                for k, v in t.describe(s).items():
                    out[f"t{i}_{k}"] = v
            ks = torch.linalg.eigvalsh(self.stiffness(theta))
            kd = torch.linalg.eigvalsh(self.damping(theta))
            out.update(K_min=float(ks[0]), K_max=float(ks[-1]),
                       K_aniso=float(ks[-1] / ks[0].clamp_min(1e-12)),
                       Kd_min=float(kd[0]), Kd_max=float(kd[-1]),
                       n_terms=len(self.terms))
            phi = theta[..., self.policy_dim:]
            if phi.numel() >= 6:
                out["kR"] = float((phi[:3] ** 2).mean())
                out["kW"] = float((phi[3:6] ** 2).mean())
                m = getattr(self.system, "m", 1.0)
                J = getattr(self.system, "Jvec", None)
                out["wn_pos"] = float((ks.mean() / m).clamp_min(0).sqrt())
                if J is not None:
                    out["wn_att"] = float(((phi[:2] ** 2) / J[:2]).mean().clamp_min(0).sqrt())
                    out["timescale_sep"] = out["wn_att"] / max(out["wn_pos"], 1e-9)
            return out

    def certificates(self, theta: Tensor, goal: Optional[Tensor] = None) -> list:
        return [t.certificate(s, goal=goal)
                for t, s in zip(self.terms, self.term_slices(theta))]

    def invariants(self, theta: Tensor) -> dict:
        with torch.no_grad():
            zero = torch.zeros(self.d, dtype=theta.dtype, device=theta.device)
            gen = torch.Generator(device=str(theta.device)).manual_seed(0)
            e = 3.0 * torch.randn(512, self.d, generator=gen,
                                  dtype=theta.dtype, device=theta.device)
            V = self.potential(theta.expand(512, -1), e)
            return {
                "V_at_goal": self.potential(theta, zero),
                "V_min": V.min(),
                "V_sampled": V,
                "grad_at_goal": self.grad_potential(theta, zero),
                "Kd_eigs": torch.linalg.eigvalsh(self.damping(theta)),
                "K_eigs": torch.linalg.eigvalsh(self.stiffness(theta)),
            }
