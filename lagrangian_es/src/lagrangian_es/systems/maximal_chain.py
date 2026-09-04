"""Planar chain in MAXIMAL coordinates -- joints as Lagrange constraints.

Each link carries its own (x, z, theta), so the links are not a chain until the
pin joints say so.  Those joints are holonomic constraints enforced by
multipliers (`holonomic.PinJointChain`) rather than being absorbed into the
choice of coordinates.

    q      = [x_i, z_i, theta_i] for each link        3N coordinates
    c(q)   = 2N pin-joint rows                        -> N degrees of freedom
    u      = N joint torques

This exists to be compared with `TwoLinkArm`, which is the SAME physical system
in minimal coordinates, and the comparison makes a point the project needs:

  * minimal coordinates -- M(q) is dense and strongly configuration-dependent,
    and there are no constraints;
  * maximal coordinates -- M is CONSTANT and block-diagonal, there are no
    velocity-product terms at all (each link is just a free rigid body), and
    every bit of the configuration dependence has moved into J(q).

So "M(q) varies" is a statement about coordinates, not about physics.  The
invariant object is the CONSTRAINED inverse inertia
P = M^-1 - M^-1 J^T (J M^-1 J^T)^-1 J M^-1, which is what a generalized force
actually sees and what the mechanical metric should be built from.  It varies
with configuration in both formulations, and `test_maximal_chain.py` checks the
two agree.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .base import LagrangianSystem, State
from .holonomic import ConstraintStack, PinJointChain, constrained_inverse_inertia


class MaximalChain(LagrangianSystem):
    allocator_dim = 0        # fully actuated in joint space
    dense_mass = True
    state_keys = ("q", "dq", "lam")

    def __init__(self, n_links: int = 2, m: float = 1.0, l: float = 1.0,
                 g: float = 9.81, anchor=(0.0, 0.0), tau_max: float = 40.0,
                 dq_max: float = 40.0, reset_noise: float = 0.15,
                 compliance: float = 1e-8,
                 dtype: torch.dtype = torch.float64, device: str = "cpu"):
        self.dtype, self.device = dtype, device
        self.N = int(n_links)
        self.n_force = self.N
        self.task_dim = self.N
        self.m, self.l, self.g = float(m), float(l), float(g)
        self.I = self.m * self.l ** 2 / 12.0
        self.tau_max, self.dq_max = float(tau_max), float(dq_max)
        self.reset_noise = float(reset_noise)
        self.anchor = self._t(anchor)

        # M is CONSTANT here: every link is a free rigid body
        self._Mdiag = self._t([self.m, self.m, self.I]).repeat(self.N)
        self._M = torch.diag(self._Mdiag)
        self._grav = self._t([0.0, self.m * self.g, 0.0]).repeat(self.N)

        # T maps generalized coords -> relative joint angles; S = T^T maps joint
        # torques -> generalized forces.  The two being transposes is the virtual
        # work statement that the actuation and the task share one geometry.
        T = torch.zeros(self.N, 3 * self.N, dtype=dtype, device=device)
        for i in range(self.N):
            T[i, 3 * i + 2] = 1.0
            if i > 0:
                T[i, 3 * (i - 1) + 2] = -1.0
        self.T = T

        self.constraints = ConstraintStack(
            [PinJointChain(self.N, self.l, anchor=anchor, compliance=compliance)])
        self.n_rows = self.constraints.n_rows(self)

    # --- kinematics ---------------------------------------------------------
    def _kin(self, q: Tensor, dq: Tensor) -> dict:
        lead = q.shape[:-1]
        qq = q.reshape(lead + (self.N, 3))
        dd = dq.reshape(lead + (self.N, 3))
        th, dth = qq[..., 2], dd[..., 2]
        u = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)
        w = torch.stack([-torch.sin(th), torch.cos(th)], dim=-1)
        return {"p": qq[..., :2], "th": th, "dth": dth, "u": u, "w": w}

    def forward_kinematics(self, joints: Tensor):
        """Joint angles [..., N] -> (link COM positions [..., N, 2], absolute angles)."""
        th = torch.cumsum(joints, dim=-1)
        u = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)
        ends = self.anchor + self.l * torch.cumsum(u, dim=-2)          # distal ends
        prox = torch.cat([self.anchor.expand(th.shape[:-1] + (1, 2)),
                          ends[..., :-1, :]], dim=-2)
        return prox + 0.5 * self.l * u, th

    # --- LagrangianSystem ---------------------------------------------------
    def _mass(self, lead) -> Tensor:
        return self._M.expand(lead + (3 * self.N, 3 * self.N))

    def reset(self, B: int, gen: torch.Generator) -> State:
        kw = dict(generator=gen, dtype=self.dtype, device=self.device)
        joints = self.reset_noise * torch.randn(B, self.N, **kw)
        djoints = self.reset_noise * torch.randn(B, self.N, **kw)
        return self.nominal_state(joints, djoints)

    def step(self, s: State, u: Tensor, dt: float, params: Tensor = None) -> State:
        q, dq = s["q"], s["dq"]
        tau = u.clamp(-self.tau_max, self.tau_max)
        kin = self._kin(q, dq)
        M = self._mass(q.shape[:-1])
        rhs = torch.einsum("im,...i->...m", self.T, tau) - self._grav
        J, bias, act, eps, _ = self.constraints.assemble(self, q, dq, kin)
        ddq, lam = self.constraints.solve(M, rhs, J, bias, act, eps)
        dq_new = dq + dt * ddq
        return {"q": q + dt * dq_new, "dq": dq_new, "lam": lam}

    def alive(self, s: State) -> Tensor:
        q, dq = s["q"], s["dq"]
        return (torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1)
                & (dq.norm(dim=-1) < self.dq_max))

    def inv_mass(self, s: State) -> Tensor:
        """Joint-space inverse inertia T P T^T.

        P varies with configuration through J even though M is constant -- which
        is precisely the point of this plant.
        """
        kin = self._kin(s["q"], s["dq"])
        J, _, _, _, _ = self.constraints.assemble(self, s["q"], s["dq"], kin)
        P = constrained_inverse_inertia(self._mass(s["q"].shape[:-1]), J)
        return torch.einsum("im,...mn,jn->...ij", self.T, P, self.T)

    def task_mass(self, s: State) -> Tensor:
        return torch.linalg.inv(self.inv_mass(s))

    def gravity_force(self, s: State) -> Tensor:
        """Joint torques holding the chain against gravity, in closed form."""
        jt = self.task_position(s)
        th = torch.cumsum(jt, dim=-1)
        cos = torch.cos(th)
        # prefix sums: C[j] = sum_{k<j} cos(theta_k)
        C = torch.cat([torch.zeros_like(cos[..., :1]),
                       torch.cumsum(cos, dim=-1)[..., :-1]], dim=-1)
        # dz_j/dq_i = l (C[j] - C[i]) + (l/2) cos(theta_j)   for j >= i
        A = self.l * (C[..., None, :] - C[..., :, None]) \
            + 0.5 * self.l * cos[..., None, :]
        tri = torch.triu(torch.ones(self.N, self.N, dtype=jt.dtype, device=jt.device))
        return self.g * self.m * (A * tri).sum(dim=-1)

    def allocate(self, F_des: Tensor, s: State, phi: Tensor) -> Tensor:
        return F_des

    def task_position(self, s: State) -> Tensor:
        return torch.einsum("im,...m->...i", self.T, s["q"])

    def task_velocity(self, s: State) -> Tensor:
        return torch.einsum("im,...m->...i", self.T, s["dq"])

    def nominal_state(self, x: Tensor, v: Tensor) -> State:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        x, v = torch.broadcast_tensors(x, v)
        lead = x.shape[:-1]
        p, th = self.forward_kinematics(x)
        dth = torch.cumsum(v, dim=-1)
        w = torch.stack([-torch.sin(th), torch.cos(th)], dim=-1)
        # COM velocity: distal ends of every preceding link, plus half of this one
        contrib = self.l * dth[..., None] * w
        ends_v = torch.cumsum(contrib, dim=-2)
        prox_v = torch.cat([torch.zeros_like(ends_v[..., :1, :]),
                            ends_v[..., :-1, :]], dim=-2)
        vel = prox_v + 0.5 * contrib
        q = torch.cat([p, th[..., None]], dim=-1).reshape(lead + (3 * self.N,))
        dq = torch.cat([vel, dth[..., None]], dim=-1).reshape(lead + (3 * self.N,))
        lam = torch.zeros(lead + (self.n_rows,), dtype=self.dtype, device=self.device)
        return {"q": q, "dq": dq, "lam": lam}

    def saturation(self, u: Tensor, s: State) -> Tensor:
        return (u.abs() >= self.tau_max - 1e-6).to(u.dtype).mean(dim=-1)

    def describe(self) -> dict:
        d = super().describe()
        d.update(n_links=self.N, n_constraint_rows=self.n_rows,
                 constant_mass=True, coordinates="maximal")
        return d
