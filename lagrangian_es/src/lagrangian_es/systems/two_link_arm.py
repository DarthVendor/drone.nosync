"""Two-link planar arm -- the plant where M(q) actually varies.

State: {q [..., 2], dq [..., 2]}.  Task space IS joint space, so `task_position`
is the identity and `nominal_state` is trivial; the arm is fully actuated, so
`allocate` is the identity and `allocator_dim = 0`.

This is the point of the whole `systems/` abstraction.  On the quadrotor, M is
constant in the body frame, so the trajectory dependence of

    G(theta) = E_traj[ (du/dtheta)^T M(q)^-1 (du/dtheta) ]

enters only through the controller Jacobian.  Here M(q) depends on the elbow
angle q2 strongly -- the inertia seen at the shoulder changes by roughly a factor
of four between a folded and an extended arm -- so `inv_mass` is genuinely
state-dependent and G tracks something no single global covariance can represent.
Running the identical ES here is what upgrades the narrow claim to the strong one.

Dynamics: M(q) qddot + C(q, qdot) qdot + G(q) = tau, with the standard planar
two-link model (Spong & Vidyasagar 6.62).
"""
from __future__ import annotations

import torch
from torch import Tensor

from .base import LagrangianSystem, State


class TwoLinkArm(LagrangianSystem):
    n_force = 2
    task_dim = 2
    allocator_dim = 0        # fully actuated: allocate is the identity
    dense_mass = True        # M(q) is not diagonal, and not constant
    state_keys = ("q", "dq")
    task_labels = ("q1", "q2")

    def __init__(
        self,
        m1: float = 1.0, m2: float = 1.0,
        l1: float = 1.0, l2: float = 1.0,
        lc1: float = 0.5, lc2: float = 0.5,
        I1: float = 1.0 / 12.0, I2: float = 1.0 / 12.0,
        g: float = 9.81,
        tau_max: float = 40.0,
        dq_max: float = 40.0,
        reset_noise: float = 0.15,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        self.dtype, self.device = dtype, device
        self.g = float(g)
        self.m1, self.m2, self.l1, self.lc1, self.lc2 = m1, m2, l1, lc1, lc2
        self.tau_max, self.dq_max = float(tau_max), float(dq_max)
        self.reset_noise = float(reset_noise)

        # standard grouped inertia constants
        self.a = I1 + I2 + m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2)
        self.b = m2 * l1 * lc2
        self.c = I2 + m2 * lc2 ** 2
        self.g1 = (m1 * lc1 + m2 * l1) * g
        self.g2 = m2 * lc2 * g

    # --- mass matrix and its inverse ---------------------------------------
    def _M(self, q: Tensor):
        cos2 = torch.cos(q[..., 1])
        m11 = self.a + 2.0 * self.b * cos2
        m12 = self.c + self.b * cos2
        m22 = torch.full_like(m11, self.c)
        return m11, m12, m22

    def inv_mass(self, s: State) -> Tensor:
        """M(q)^-1, [..., 2, 2].  Depends on the elbow angle -- unlike the
        quadrotor, this is where the metric's state dependence really lives."""
        m11, m12, m22 = self._M(s["q"])
        det = (m11 * m22 - m12 * m12).clamp_min(1e-9)
        row0 = torch.stack([m22 / det, -m12 / det], dim=-1)
        row1 = torch.stack([-m12 / det, m11 / det], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def task_mass(self, s: State) -> Tensor:
        """M(q) itself -- task space IS joint space here.  Because this varies,
        kinetic shaping on the arm omits the Coriolis correction of the IDA-PBC
        matching conditions and is an approximation; on the quadrotors, where M
        is constant, the same law is exact."""
        m11, m12, m22 = self._M(s["q"])
        row0 = torch.stack([m11, m12], dim=-1)
        row1 = torch.stack([m12, m22], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def gravity_force(self, s: State) -> Tensor:
        """G(q): the joint torque that holds the arm against gravity."""
        q = s["q"]
        q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
        return torch.stack([self.g1 * torch.cos(q1) + self.g2 * torch.cos(q12),
                            self.g2 * torch.cos(q12)], dim=-1)

    def _coriolis(self, q: Tensor, dq: Tensor) -> Tensor:
        s2 = torch.sin(q[..., 1])
        d1, d2 = dq[..., 0], dq[..., 1]
        return torch.stack([-self.b * s2 * (2.0 * d1 * d2 + d2 * d2),
                            self.b * s2 * d1 * d1], dim=-1)

    # --- simulation ---------------------------------------------------------
    def reset(self, B: int, gen: torch.Generator) -> State:
        kw = dict(generator=gen, dtype=self.dtype, device=self.device)
        return {"q": self.reset_noise * torch.randn(B, 2, **kw),
                "dq": self.reset_noise * torch.randn(B, 2, **kw)}

    def step(self, s: State, u: Tensor, dt: float, params: Tensor = None) -> State:
        tau = u.clamp(-self.tau_max, self.tau_max)
        q, dq = s["q"], s["dq"]
        rhs = tau - self._coriolis(q, dq) - self.gravity_force(s)
        ddq = torch.einsum("...ij,...j->...i", self.inv_mass(s), rhs)
        dq_new = dq + dt * ddq
        return {"q": q + dt * dq_new, "dq": dq_new}

    def alive(self, s: State) -> Tensor:
        q, dq = s["q"], s["dq"]
        return (torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1)
                & (dq.norm(dim=-1) < self.dq_max) & (q.abs().amax(-1) < 50.0))

    # --- fully actuated: the allocator seam collapses ------------------------
    def allocate(self, F_des: Tensor, s: State, phi: Tensor,
                 goal=None) -> Tensor:
        return F_des

    # --- task space is joint space -----------------------------------------
    def render_spec(self) -> dict:
        return {"dim": 2, "ground": None, "scale": self.l1 + 1.0,
                "bodies": [{"type": "segment", "size": [self.l1, 0.06]},
                           {"type": "segment", "size": [self.l1, 0.05]}]}

    def render_poses(self, s: State) -> Tensor:
        q = s["q"]
        th1 = q[..., 0]
        th2 = q[..., 0] + q[..., 1]
        c1, s1 = torch.cos(th1), torch.sin(th1)
        c2, s2 = torch.cos(th2), torch.sin(th2)
        p1 = torch.stack([self.lc1 * c1, self.lc1 * s1], dim=-1)
        elbow = torch.stack([self.l1 * c1, self.l1 * s1], dim=-1)
        p2 = elbow + torch.stack([self.lc2 * c2, self.lc2 * s2], dim=-1)
        b1 = torch.cat([p1, th1[..., None]], dim=-1)
        b2 = torch.cat([p2, th2[..., None]], dim=-1)
        return torch.stack([b1, b2], dim=-2)                        # [..., 2, 3]

    def task_position(self, s: State) -> Tensor:
        return s["q"]

    def task_velocity(self, s: State) -> Tensor:
        return s["dq"]

    def nominal_state(self, x: Tensor, v: Tensor) -> State:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        x, v = torch.broadcast_tensors(x, v)
        return {"q": x.clone(), "dq": v.clone()}

    # --- diagnostics --------------------------------------------------------
    def saturation(self, u: Tensor, s: State) -> Tensor:
        return (u.abs() >= self.tau_max - 1e-6).to(u.dtype).mean(dim=-1)

    def mass_condition(self, s: State) -> Tensor:
        """cond(M(q)) -- how much the metric's state dependence actually bites."""
        return torch.linalg.cond(torch.linalg.inv(self.inv_mass(s)))
