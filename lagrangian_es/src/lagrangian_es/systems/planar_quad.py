"""Planar (vertical-plane) quadrotor -- the cheap underactuated test plant.

State: {p [..., 2] = (x, z), v [..., 2], th [..., 1], om [..., 1]}.

Same underactuation structure as the SE(3) quadrotor -- two degrees of task-space
force from one thrust magnitude plus an attitude loop -- with a scalar attitude
instead of a rotation matrix, so it exercises the allocator seam at a fraction of
the cost.  Like the full quadrotor, M is constant.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .base import LagrangianSystem, State


class PlanarQuadrotor(LagrangianSystem):
    n_force = 2              # (thrust, torque)
    task_dim = 2             # (x, z)
    allocator_dim = 2        # (kR, kW), used squared
    dense_mass = False
    state_keys = ("p", "v", "th", "om")

    def __init__(self, m: float = 0.5, J: float = 5.0e-3, g: float = 9.81,
                 thrust_ratio: float = 2.2, tau_max: float = 0.30,
                 phi0: tuple = (0.30, 0.12),
                 reset_z: float = 0.5, reset_noise: float = 0.05,
                 z_floor: float = 0.05, p_max: float = 20.0,
                 v_max: float = 30.0, om_max: float = 50.0,
                 dtype: torch.dtype = torch.float64, device: str = "cpu"):
        self.dtype, self.device = dtype, device
        self.m, self.J, self.g = float(m), float(J), float(g)
        self.f_max, self.f_min = thrust_ratio * self.m * self.g, 0.0
        self.tau_max = float(tau_max)
        self.phi0 = tuple(phi0)
        self.reset_z, self.reset_noise = float(reset_z), float(reset_noise)
        self.z_floor, self.p_max = float(z_floor), float(p_max)
        self.v_max, self.om_max = float(v_max), float(om_max)
        self._hover = self.m * self.g
        self._e2 = self._t([0.0, 1.0])

    def reset(self, B: int, gen: torch.Generator) -> State:
        kw = dict(generator=gen, dtype=self.dtype, device=self.device)
        p = self._e2 * self.reset_z + self.reset_noise * torch.randn(B, 2, **kw)
        return {"p": p, "v": self.reset_noise * torch.randn(B, 2, **kw),
                "th": self.reset_noise * torch.randn(B, 1, **kw),
                "om": self.reset_noise * torch.randn(B, 1, **kw)}

    def step(self, s: State, u: Tensor, dt: float, params: Tensor = None) -> State:
        f = u[..., 0:1].clamp(self.f_min, self.f_max)
        tau = u[..., 1:2].clamp(-self.tau_max, self.tau_max)
        th = s["th"]
        body_z = torch.cat([-torch.sin(th), torch.cos(th)], dim=-1)
        acc = body_z * (f / self.m) - self._e2 * self.g
        v = s["v"] + dt * acc
        om = s["om"] + dt * (tau / self.J)
        return {"p": s["p"] + dt * v, "v": v, "th": th + dt * om, "om": om}

    def alive(self, s: State) -> Tensor:
        p, v, om = s["p"], s["v"], s["om"]
        finite = (torch.isfinite(p).all(-1) & torch.isfinite(v).all(-1)
                  & torch.isfinite(om).all(-1))
        return (finite & (p[..., 1] > self.z_floor) & (p.norm(dim=-1) < self.p_max)
                & (v.norm(dim=-1) < self.v_max) & (om[..., 0].abs() < self.om_max))

    def inv_mass(self, s: State) -> Tensor:
        head = self._t([1.0 / self.m, 1.0 / self.J])
        return head.expand(s["p"].shape[:-1] + (self.n_force,))

    def gravity_force(self, s: State) -> Tensor:
        return torch.zeros_like(s["p"]) + self._e2 * self._hover

    def task_mass(self, s: State) -> Tensor:
        eye = torch.eye(2, dtype=self.dtype, device=self.device)
        return (self.m * eye).expand(s["p"].shape[:-1] + (2, 2))

    def allocator_init(self) -> Tensor:
        return self._t(self.phi0)

    def allocate(self, F_des: Tensor, s: State, phi: Tensor,
                 goal=None) -> Tensor:
        """Thrust along body z, plus a scalar attitude loop.

        The angle error is wrapped with atan2(sin d, cos d) rather than a modulo:
        it is smooth everywhere, which `jacrev` requires and a branch would break.
        """
        kR = phi[..., 0:1] ** 2
        kW = phi[..., 1:2] ** 2
        th = s["th"]
        body_z = torch.cat([-torch.sin(th), torch.cos(th)], dim=-1)
        f = (F_des * body_z).sum(dim=-1, keepdim=True).clamp(self.f_min, self.f_max)
        th_d = torch.atan2(-F_des[..., 0:1], F_des[..., 1:2].clamp_min(1e-6))
        d = th - th_d
        err = torch.atan2(torch.sin(d), torch.cos(d))
        tau = (-kR * err - kW * s["om"]).clamp(-self.tau_max, self.tau_max)
        return torch.cat([f, tau], dim=-1)

    def render_spec(self) -> dict:
        return {"dim": 2, "ground": 0.0, "scale": 1.0,
                "bodies": [{"type": "box", "size": [0.30, 0.07]}]}

    def render_poses(self, s: State) -> Tensor:
        return torch.cat([s["p"], s["th"]], dim=-1).unsqueeze(-2)   # [..., 1, 3]

    def task_position(self, s: State) -> Tensor:
        return s["p"]

    def task_velocity(self, s: State) -> Tensor:
        return s["v"]

    def nominal_state(self, x: Tensor, v: Tensor) -> State:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        x, v = torch.broadcast_tensors(x, v)
        z = torch.zeros(x.shape[:-1] + (1,), dtype=self.dtype, device=self.device)
        return {"p": x.clone(), "v": v.clone(), "th": z.clone(), "om": z.clone()}

    def effort(self, u: Tensor, s: State) -> Tensor:
        du_f = u[..., 0] - self._hover
        return du_f * du_f / self.m + u[..., 1] ** 2 / self.J

    def shaping_cost(self, s: State) -> Tensor:
        return 1.0 - torch.cos(s["th"][..., 0])

    def saturation(self, u: Tensor, s: State) -> Tensor:
        tol = 1e-6
        f = u[..., 0]
        sat = ((f <= self.f_min + tol) | (f >= self.f_max - tol)).to(u.dtype)
        return (sat + (u[..., 1].abs() >= self.tau_max - tol).to(u.dtype)) / 2.0
