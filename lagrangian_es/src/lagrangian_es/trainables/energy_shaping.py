"""`EnergyShaping` -- the proposal.

A genome decodes to a shaped potential V_d and a PSD dissipation matrix K_d,
which together define a desired task-space force

    F_des = g(q) - grad V_d(x - x_goal) - K_d xdot .

V_d is a sum of NK pseudo-Huber bowls,

    V_d(e) = sum_k w_k^2 ( sqrt(1 + ||A_k e||^2) - 1 ) ,

which is nonnegative, zero only at e = 0, and has a gradient that saturates at
||sum_k w_k^2 A_k^T A_k e / ||A_k e|| || as ||e|| -> infinity.  Two consequences
that the whole method leans on:

  * every genome -- including a randomly initialized one -- is an energy-shaping
    controller whose closed-loop equilibrium is the goal by construction.
    Evolution searches the geometry of the potential, not the question of
    whether the controller is stable;
  * the commanded force is bounded through the potential's own geometry, so
    actuator saturation is a property of the search space rather than a clip
    applied after the fact.

The gradient is written in closed form and never obtained by autograd: `forward`
has to be cheap enough to vmap over a whole population every step, and jacrev
differentiates it with respect to theta, not to e.

Genome layout (NK = 3, d = task_dim = 3 -> policy_dim = 39):

    0        : NK             w   bowl weights, used as w^2 (>= 0)
    NK       : NK + NK*d*d    A   NK bowl shape matrices, d x d each
    ...      : + d*d          D   damping factor, K_d = D D^T (PSD by construction)

then the system appends `allocator_dim` slots (6 for the quadrotor: kR, kW).
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..systems.base import LagrangianSystem, State
from .base import Trainable


class EnergyShaping(Trainable):
    equilibrium_exact = True

    def __init__(
        self,
        system: LagrangianSystem,
        n_bowls: int = 3,
        w0: float = 1.0,
        a0: float = 1.0,
        d0: float = 1.2,
        phi0: tuple = (0.25, 0.25, 0.25, 0.10, 0.10, 0.10),
    ):
        super().__init__(system)
        self.NK = int(n_bowls)
        self.d = int(system.task_dim)
        self.w0, self.a0, self.d0 = float(w0), float(a0), float(d0)
        self.phi0 = tuple(phi0)

        # slice bounds, computed once
        self._i_w = (0, self.NK)
        self._i_A = (self.NK, self.NK + self.NK * self.d * self.d)
        self._i_D = (self._i_A[1], self._i_A[1] + self.d * self.d)

    @property
    def policy_dim(self) -> int:
        return self.NK + self.NK * self.d * self.d + self.d * self.d

    # --- genome decoding ----------------------------------------------------
    def unpack(self, theta: Tensor):
        """theta [..., dim] -> (w2 [..., NK], A [..., NK, d, d], D [..., d, d],
        phi [..., allocator_dim]).  Trailing-dim indexed: batched or vmapped."""
        d, NK = self.d, self.NK
        w = theta[..., self._i_w[0] : self._i_w[1]]
        A = theta[..., self._i_A[0] : self._i_A[1]].reshape(theta.shape[:-1] + (NK, d, d))
        D = theta[..., self._i_D[0] : self._i_D[1]].reshape(theta.shape[:-1] + (d, d))
        phi = theta[..., self.policy_dim :]
        return w * w, A, D, phi

    def init(self) -> Tensor:
        d, NK = self.d, self.NK
        eye = torch.eye(d, dtype=self.dtype, device=self.device)
        parts = [
            torch.full((NK,), self.w0, dtype=self.dtype, device=self.device),
            (self.a0 * eye).expand(NK, d, d).reshape(-1),
            (self.d0 * eye).reshape(-1),
        ]
        if self.system.allocator_dim:
            parts.append(self.system.allocator_init())
        return torch.cat(parts, dim=0)

    # --- the shaped potential ----------------------------------------------
    def potential(self, theta: Tensor, e: Tensor) -> Tensor:
        """V_d(e) >= 0, zero iff e = 0 (given w != 0).  [...]"""
        w2, A, _, _ = self.unpack(theta)
        Ae = torch.einsum("...kij,...j->...ki", A, e)
        r = torch.sqrt(1.0 + (Ae * Ae).sum(dim=-1))
        return (w2 * (r - 1.0)).sum(dim=-1)

    def grad_potential(self, theta: Tensor, e: Tensor) -> Tensor:
        """Closed-form grad V_d, bounded in ||e||.  [..., d]"""
        w2, A, _, _ = self.unpack(theta)
        Ae = torch.einsum("...kij,...j->...ki", A, e)
        r = torch.sqrt(1.0 + (Ae * Ae).sum(dim=-1))
        return torch.einsum("...k,...kij,...ki->...j", w2 / r, A, Ae)

    def damping(self, theta: Tensor) -> Tensor:
        """K_d = D D^T, positive semidefinite by construction.  [..., d, d]"""
        _, _, D, _ = self.unpack(theta)
        return D @ D.transpose(-1, -2)

    def stiffness(self, theta: Tensor) -> Tensor:
        """Hessian of V_d at the goal: sum_k w_k^2 A_k^T A_k.  [..., d, d]"""
        w2, A, _, _ = self.unpack(theta)
        return torch.einsum("...k,...kij,...kil->...jl", w2, A, A)

    # --- the controller map -------------------------------------------------
    def forward(self, theta: Tensor, s: State, goal: Tensor) -> Tensor:
        sysm = self.system
        _, _, D, phi = self.unpack(theta)
        e = sysm.task_position(s) - goal
        gradV = self.grad_potential(theta, e)
        v = sysm.task_velocity(s)
        Kd_v = torch.einsum("...ij,...kj,...k->...i", D, D, v)   # (D D^T) v
        F_des = sysm.gravity_force(s) - gradV - Kd_v
        return sysm.allocate(F_des, s, phi)

    # --- reporting ----------------------------------------------------------
    def describe(self, theta: Tensor) -> dict:
        """Spectrum of the effective stiffness, so you can see whether evolution
        actually uses the extra bowls or whether NK = 1 would do just as well
        (three identical A_k is a valid but boring optimum)."""
        with torch.no_grad():
            ks = torch.linalg.eigvalsh(self.stiffness(theta))
            kd = torch.linalg.eigvalsh(self.damping(theta))
            _, _, _, phi = self.unpack(theta)
            out = {
                "K_min": float(ks[0]), "K_max": float(ks[-1]),
                "K_aniso": float(ks[-1] / ks[0].clamp_min(1e-12)),
                "Kd_min": float(kd[0]), "Kd_max": float(kd[-1]),
            }
            if phi.numel():
                out["kR"] = float((phi[:3] ** 2).mean())
                out["kW"] = float((phi[3:] ** 2).mean())
                # natural frequencies: position loop vs attitude loop.  If these
                # converge the vehicle commands tilts it cannot achieve.
                m = getattr(self.system, "m", 1.0)
                J = getattr(self.system, "Jvec", None)
                out["wn_pos"] = float((ks.mean() / m).clamp_min(0).sqrt())
                if J is not None:
                    out["wn_att"] = float(((phi[:2] ** 2) / J[:2]).mean().clamp_min(0).sqrt())
                    out["timescale_sep"] = out["wn_att"] / max(out["wn_pos"], 1e-9)
            return out

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
