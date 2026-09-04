"""Full SE(3) quadrotor.

State: {p [...,3], v [...,3], R [...,3,3], om [...,3]}  -- world-frame position and
velocity, body-to-world rotation, body-frame angular rate.

Generalized force is u = (f, tau_x, tau_y, tau_z): a scalar thrust magnitude along
body z plus a body-frame torque.  n_force = 4, task_dim = 3 -- the plant is
underactuated, and `allocate` is where that shows up.

M is CONSTANT here: diag(m, Jx, Jy, Jz) in body coordinates.  `inv_mass` therefore
ignores its state argument.  That is the honest scope limit of the whitening claim
on this plant -- see `two_link_arm.py` for the case where it does not.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .base import LagrangianSystem, State
from .so3 import rodrigues, vee


class QuadrotorSE3(LagrangianSystem):
    n_force = 4
    task_dim = 3
    allocator_dim = 6          # kR [3], kW [3]
    dense_mass = False
    state_keys = ("p", "v", "R", "om")

    def __init__(
        self,
        m: float = 0.5,
        J: tuple = (5.0e-3, 5.0e-3, 9.0e-3),
        g: float = 9.81,
        thrust_ratio: float = 2.2,     # f_max / (m g)
        tau_max: float = 0.30,
        phi0: tuple = (0.25, 0.25, 0.25, 0.10, 0.10, 0.10),   # kR, kW priors
        # --- reset distribution
        reset_z: float = 0.5,
        reset_pos_noise: float = 0.05,
        reset_vel_noise: float = 0.05,
        reset_att_noise: float = 0.03,
        reset_om_noise: float = 0.05,
        # --- liveness envelope
        z_floor: float = 0.05,
        p_max: float = 20.0,
        v_max: float = 30.0,
        om_max: float = 50.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        self.dtype, self.device = dtype, device
        self.m = float(m)
        self.g = float(g)
        self.Jvec = self._t(J)                       # [3]
        self.f_max = float(thrust_ratio) * self.m * self.g
        self.f_min = 0.0
        self.tau_max = float(tau_max)
        self.phi0 = tuple(phi0)

        self.reset_z = float(reset_z)
        self.reset_pos_noise = float(reset_pos_noise)
        self.reset_vel_noise = float(reset_vel_noise)
        self.reset_att_noise = float(reset_att_noise)
        self.reset_om_noise = float(reset_om_noise)

        self.z_floor = float(z_floor)
        self.p_max, self.v_max, self.om_max = float(p_max), float(v_max), float(om_max)

        self._e3 = self._t([0.0, 0.0, 1.0])
        self._e1 = self._t([1.0, 0.0, 0.0])
        self._hover = self.m * self.g

    # --- simulation ---------------------------------------------------------
    def reset(self, B: int, gen: torch.Generator) -> State:
        kw = dict(generator=gen, dtype=self.dtype, device=self.device)
        p0 = self._e3 * self.reset_z
        p = p0 + self.reset_pos_noise * torch.randn(B, 3, **kw)
        v = self.reset_vel_noise * torch.randn(B, 3, **kw)
        om = self.reset_om_noise * torch.randn(B, 3, **kw)
        R = rodrigues(self.reset_att_noise * torch.randn(B, 3, **kw))
        return {"p": p, "v": v, "R": R, "om": om}

    def step(self, s: State, u: Tensor, dt: float) -> State:
        """Semi-implicit Euler on SE(3).  Pure: builds a fresh dict."""
        f = u[..., 0].clamp(self.f_min, self.f_max)          # safety clamp; the
        tau = u[..., 1:4].clamp(-self.tau_max, self.tau_max)  # allocator clamps too

        R, om = s["R"], s["om"]
        b3 = R[..., :, 2]                                     # body z in world
        acc = b3 * (f / self.m)[..., None] - self._e3 * self.g
        v = s["v"] + dt * acc
        p = s["p"] + dt * v

        Jom = om * self.Jvec
        om_dot = (tau - torch.cross(om, Jom, dim=-1)) / self.Jvec
        om_new = om + dt * om_dot
        R_new = R @ rodrigues(om_new * dt)
        return {"p": p, "v": v, "R": R_new, "om": om_new}

    def alive(self, s: State) -> Tensor:
        p, v, om, R = s["p"], s["v"], s["om"], s["R"]
        finite = (
            torch.isfinite(p).all(-1)
            & torch.isfinite(v).all(-1)
            & torch.isfinite(om).all(-1)
            & torch.isfinite(R).all(-1).all(-1)
        )
        return (
            finite
            & (p[..., 2] > self.z_floor)
            & (p.norm(dim=-1) < self.p_max)
            & (v.norm(dim=-1) < self.v_max)
            & (om.norm(dim=-1) < self.om_max)
        )

    # --- mechanics ----------------------------------------------------------
    def inv_mass(self, s: State) -> Tensor:
        """diag(1/m, 1/Jx, 1/Jy, 1/Jz), broadcast to the state's batch shape.

        Deliberately ignores `s`: for a rigid body M is constant in the body frame.
        The trajectory dependence of G on this plant therefore enters only through
        the controller Jacobian.
        """
        head = torch.cat([self._t([1.0 / self.m]), 1.0 / self.Jvec], dim=0)  # [4]
        shape = s["p"].shape[:-1] + (self.n_force,)
        return head.expand(shape)

    def gravity_force(self, s: State) -> Tensor:
        """m g e3 -- the world-frame force that holds the vehicle up."""
        return torch.zeros_like(s["p"]) + self._e3 * self._hover

    # --- the underactuation seam --------------------------------------------
    def allocate(self, F_des: Tensor, s: State, phi: Tensor) -> Tensor:
        """Thrust along body z + a geometric SO(3) attitude loop.

        `phi` = (kR [3], kW [3]) raw; gains are used squared so they stay positive
        under unconstrained evolution.  Yaw reference is fixed at world x.

        vmap/jacrev safe: no branches, no in-place writes, every normalize floored.
        """
        R, om = s["R"], s["om"]
        kR = phi[..., 0:3] ** 2
        kW = phi[..., 3:6] ** 2

        # --- thrust: project the desired force onto the current body z --------
        b3 = R[..., :, 2]
        f = (F_des * b3).sum(dim=-1).clamp(self.f_min, self.f_max)

        # --- desired attitude: body z along F_des, yaw pinned to world x ------
        b3d = F_des / F_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b1c = torch.zeros_like(b3d) + self._e1
        b2d = torch.cross(b3d, b1c, dim=-1)
        b2d = b2d / b2d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b1d = torch.cross(b2d, b3d, dim=-1)
        Rd = torch.stack([b1d, b2d, b3d], dim=-1)            # columns

        # --- geometric attitude error (Lee et al.) ----------------------------
        RdT_R = Rd.transpose(-1, -2) @ R
        eR = 0.5 * vee(RdT_R - RdT_R.transpose(-1, -2))
        eW = om                                              # om_d = 0

        gyro = torch.cross(om, om * self.Jvec, dim=-1)       # feedforward
        tau = (-kR * eR - kW * eW + gyro).clamp(-self.tau_max, self.tau_max)
        return torch.cat([f[..., None], tau], dim=-1)

    # --- task-space accessors -----------------------------------------------
    def task_mass(self, s: State) -> Tensor:
        """m I_3 -- translational inertia is isotropic and CONSTANT, so kinetic
        shaping on this plant is exact rather than an IDA-PBC approximation."""
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        return (self.m * eye).expand(s["p"].shape[:-1] + (3, 3))

    def allocator_init(self) -> Tensor:
        """kR = 0.25, kW = 0.10 (used squared) -> an attitude loop barely faster
        than the position loop.  Deliberately marginal: this is what makes the
        generation-0 prior crash rather than merely track poorly."""
        return self._t(self.phi0)

    def task_position(self, s: State) -> Tensor:
        return s["p"]

    def task_velocity(self, s: State) -> Tensor:
        return s["v"]

    def nominal_state(self, x: Tensor, v: Tensor) -> State:
        """Level attitude, zero body rate, at position x with velocity v."""
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        x, v = torch.broadcast_tensors(x, v)
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        return {
            "p": x.clone(),
            "v": v.clone(),
            "R": eye.expand(x.shape[:-1] + (3, 3)).clone(),
            "om": torch.zeros_like(x),
        }

    # --- cost hooks ---------------------------------------------------------
    def effort(self, u: Tensor, s: State) -> Tensor:
        """u is (f, tau) but gravity_force is a world-frame 3-vector, so the
        default implementation does not apply: the hover feedforward is m g in
        the thrust channel and zero in the torques."""
        du_f = u[..., 0] - self._hover
        du_t = u[..., 1:4]
        return du_f * du_f / self.m + (du_t * du_t / self.Jvec).sum(dim=-1)

    def shaping_cost(self, s: State) -> Tensor:
        """Tilt penalty: 1 - R[2,2], zero when level, 2 when inverted."""
        return 1.0 - s["R"][..., 2, 2]

    # --- diagnostics --------------------------------------------------------
    def saturation(self, u: Tensor, s: State) -> Tensor:
        tol = 1e-6
        f = u[..., 0]
        sat_f = (f <= self.f_min + tol) | (f >= self.f_max - tol)
        sat_t = u[..., 1:4].abs() >= self.tau_max - tol
        n = sat_f.to(u.dtype) + sat_t.to(u.dtype).sum(dim=-1)
        return n / self.n_force

    def describe(self) -> dict:
        d = super().describe()
        d.update(m=self.m, g=self.g, f_max=self.f_max, tau_max=self.tau_max,
                 Jx=float(self.Jvec[0]), Jz=float(self.Jvec[2]))
        return d
