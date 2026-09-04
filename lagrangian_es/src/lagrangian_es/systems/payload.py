"""Quadrotor carrying slung payloads.

The carrier and its packages share one generalized velocity

    w = [ v (3) | omega (3) | v_load (3 per payload) ]

with a block-diagonal mass matrix, and the connectors contribute rows to a single
KKT solve.  Adding a package is therefore a change of *constraints*, not a change
of dynamics code -- which is the whole reason the coupling layer exists.

A slung load is a genuinely hard control problem and a good test of the framework:
the payload is an unactuated pendulum hanging off the very body you are trying to
position, so aggressive tracking excites a swing that then drags the vehicle. The
tilt-plus-swing shaping cost is what tells evolution that "arrive fast" and
"arrive with the package still" are different objectives.

Note the feedforward.  `gravity_force` returns (m_drone + sum m_load) g, not the
drone's own weight: a controller that compensates only its own mass sags under
the package from the first timestep, and would spend the whole search rediscovering
a constant it was never given.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor

from .base import State
from .connectors import Connector, _attach
from .holonomic import ConstraintStack
from .quadrotor import QuadrotorSE3
from .so3 import rodrigues


class QuadrotorPayload(QuadrotorSE3):
    """`QuadrotorSE3` plus point-mass payloads on connectors."""

    state_keys = ("p", "v", "R", "om", "pl", "vl")

    def __init__(self, payload_mass: float = 0.15, cable_length: float = 0.45,
                 connector: str = "cable", n_payload: int = 1,
                 attach: Optional[Sequence] = None, swing_weight: float = 0.6,
                 **kw):
        super().__init__(**kw)
        self.n_pay = int(n_payload)
        self.m_load = float(payload_mass)
        self.cable_length = float(cable_length)
        self.swing_weight = float(swing_weight)

        from .connectors import CONNECTORS, RigidLink
        if connector not in CONNECTORS:
            raise KeyError(f"unknown connector {connector!r}; "
                           f"registered: {sorted(CONNECTORS)}")
        kind = CONNECTORS[connector]
        # a weld has no free length; everything else is defined by one
        mk = (lambda: kind()) if kind is RigidLink \
            else (lambda: kind(length=self.cable_length))
        self.connectors: List[Connector] = [mk() for _ in range(self.n_pay)]
        self.connector_kind = connector

        off = attach if attach is not None else [[0.0, 0.0, -0.02]] * self.n_pay
        self.attach = self._t(off).reshape(self.n_pay, 3)
        # rest position of each payload, relative to the CARRIER, so reset() and
        # nominal_state() start on the constraint manifold rather than off it
        self.rest = self.attach + self._t(
            [c.rest_offset() for c in self.connectors]).reshape(self.n_pay, 3)

        self.n_w = 6 + 3 * self.n_pay
        diag = [self.m] * 3 + [float(j) for j in self.Jvec] \
            + [self.m_load] * (3 * self.n_pay)
        self._Mfull = torch.diag(self._t(diag))
        self.total_mass = self.m + self.n_pay * self.m_load
        self._stack = ConstraintStack([])       # reuse the KKT solver only

    # --- assembly -----------------------------------------------------------
    def _rows(self, s: State):
        """Scatter every connector's rows into the shared generalized velocity."""
        p, R, v, om = s["p"], s["R"], s["v"], s["om"]
        lead = p.shape[:-1]
        Js, resid, acts, eps = [], [], [], []
        for k, conn in enumerate(self.connectors):
            if conn.n_rows() == 0:
                continue
            d = self.attach[k].expand(lead + (3,))
            Jk, ck, ak, bk = conn.rows(p, R, v, om, s["pl"][..., k, :],
                                       s["vl"][..., k, :], d)
            m = Jk.shape[-2]
            full = torch.zeros(lead + (m, self.n_w), dtype=p.dtype, device=p.device)
            full[..., :6] = Jk[..., :6]
            full[..., 6 + 3 * k: 9 + 3 * k] = Jk[..., 6:]
            Js.append(full)
            resid.append(bk)
            acts.append(ak)
            eps.append(torch.full_like(ak, conn.compliance()))
        if not Js:
            return None
        return (torch.cat(Js, -2), torch.cat(resid, -1),
                torch.cat(acts, -1), torch.cat(eps, -1))

    def _compliant(self, s: State):
        """Forces from connectors that work without a multiplier."""
        p, R, v, om = s["p"], s["R"], s["v"], s["om"]
        lead = p.shape[:-1]
        Fv = torch.zeros(lead + (3,), dtype=p.dtype, device=p.device)
        Fom = torch.zeros_like(Fv)
        Fl = torch.zeros(lead + (self.n_pay, 3), dtype=p.dtype, device=p.device)
        for k, conn in enumerate(self.connectors):
            d = self.attach[k].expand(lead + (3,))
            f = conn.force(p, R, v, om, s["pl"][..., k, :], s["vl"][..., k, :], d)
            if f is None:
                continue
            Fl[..., k, :] = f
            _, Rd, _ = _attach(p, R, om, d)
            Fv = Fv - f                                    # reaction on the carrier
            Fom = Fom - torch.cross(Rd, f, dim=-1)
        return Fv, Fom, Fl

    # --- LagrangianSystem ---------------------------------------------------
    def reset(self, B: int, gen: torch.Generator) -> State:
        s = super().reset(B, gen)
        kw = dict(generator=gen, dtype=self.dtype, device=self.device)
        pl = s["p"][:, None, :] + self.rest + 0.01 * torch.randn(B, self.n_pay, 3, **kw)
        s["pl"] = pl
        s["vl"] = 0.02 * torch.randn(B, self.n_pay, 3, **kw)
        return s

    def step(self, s: State, u: Tensor, dt: float, params: Tensor = None) -> State:
        f = u[..., 0].clamp(self.f_min, self.f_max)
        tau = u[..., 1:4].clamp(-self.tau_max, self.tau_max)
        R, om = s["R"], s["om"]
        lead = s["p"].shape[:-1]

        b3 = R[..., :, 2]
        Fv = b3 * f[..., None] - self._e3 * (self.m * self.g)
        Fom = tau - torch.cross(om, om * self.Jvec, dim=-1)
        Fl = -self._e3 * (self.m_load * self.g)
        Fl = Fl.expand(lead + (self.n_pay, 3)).clone()

        cFv, cFom, cFl = self._compliant(s)
        Fv, Fom, Fl = Fv + cFv, Fom + cFom, Fl + cFl

        rhs = torch.cat([Fv, Fom, Fl.reshape(lead + (3 * self.n_pay,))], dim=-1)
        M = self._Mfull.expand(lead + (self.n_w, self.n_w))
        rows = self._rows(s)
        if rows is None:
            wdot = rhs / torch.diagonal(self._Mfull)
        else:
            J, bias, act, eps = rows
            wdot, _ = self._stack.solve(M, rhs, J, bias, act, eps)

        v = s["v"] + dt * wdot[..., :3]
        om_new = om + dt * wdot[..., 3:6]
        vl = s["vl"] + dt * wdot[..., 6:].reshape(lead + (self.n_pay, 3))
        return {"p": s["p"] + dt * v, "v": v,
                "R": R @ rodrigues(om_new * dt), "om": om_new,
                "pl": s["pl"] + dt * vl, "vl": vl}

    def alive(self, s: State) -> Tensor:
        ok = super().alive(s)
        return (ok & torch.isfinite(s["pl"]).all(-1).all(-1)
                & torch.isfinite(s["vl"]).all(-1).all(-1)
                & (s["pl"][..., 2] > -0.5).all(-1))

    def gravity_force(self, s: State) -> Tensor:
        """Holds the drone AND its packages -- see the module docstring."""
        return torch.zeros_like(s["p"]) + self._e3 * (self.total_mass * self.g)

    def nominal_state(self, x: Tensor, v: Tensor) -> State:
        s = super().nominal_state(x, v)
        s["pl"] = (s["p"][..., None, :] + self.rest).expand(
            s["p"].shape[:-1] + (self.n_pay, 3)).clone()
        s["vl"] = torch.zeros_like(s["pl"])
        return s

    def swing_angle(self, s: State) -> Tensor:
        """Angle of each cable from vertical, [..., n_pay]."""
        r = s["pl"] - s["p"][..., None, :]
        n = r.norm(dim=-1).clamp_min(1e-6)
        return torch.acos((-r[..., 2] / n).clamp(-1.0, 1.0))

    def shaping_cost(self, s: State) -> Tensor:
        """Tilt, plus how far the package has swung from hanging straight down."""
        return (1.0 - s["R"][..., 2, 2]) \
            + self.swing_weight * self.swing_angle(s).pow(2).sum(-1)

    # --- rendering ----------------------------------------------------------
    def render_spec(self) -> dict:
        spec = super().render_spec()
        spec["bodies"] = spec["bodies"] + [
            {"type": "box", "size": [0.11, 0.11, 0.11], "shade": 0.8}
            for _ in range(self.n_pay)]
        spec["cables"] = self.n_pay
        return spec

    def render_poses(self, s: State) -> Tensor:
        base = super().render_poses(s)                      # [..., 1, 12]
        eye = torch.eye(3, dtype=s["p"].dtype, device=s["p"].device)
        eye = eye.reshape(9).expand(s["pl"].shape[:-1] + (9,))
        pay = torch.cat([s["pl"], eye], dim=-1)             # [..., n_pay, 12]
        return torch.cat([base, pay], dim=-2)

    def render_extras(self, s: State) -> dict:
        """Cable endpoints, so the renderer can draw the rope."""
        lead = s["p"].shape[:-1]
        a = []
        for k in range(self.n_pay):
            d = self.attach[k].expand(lead + (3,))
            a.append(s["p"] + torch.einsum("...ij,...j->...i", s["R"], d))
        att = torch.stack(a, dim=-2)
        return {"cable": torch.cat([att, s["pl"]], dim=-1)}   # [..., n_pay, 6]

    def describe(self) -> dict:
        d = super().describe()
        d.update(n_payload=self.n_pay, payload_mass=self.m_load,
                 cable_length=self.cable_length, connector=self.connector_kind,
                 total_mass=self.total_mass)
        return d
