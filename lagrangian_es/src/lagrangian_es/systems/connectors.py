"""Connection joints: how bodies are attached to one another.

A connector is a constraint between an attachment point on a carrier and a
payload body, expressed on the shared generalized velocity

    w = [ v (3) | omega (3) | v_load (3 per payload) ]

Three kinds cover most of what you actually want to hang off a robot, and the
distinction between them is entirely in the constraint, not in the code that
solves it:

  * `RigidLink`  -- bilateral, 3 rows.  The payload is welded at a fixed offset;
    always enforced.
  * `Cable`      -- **unilateral**, 1 row.  A rope can pull but not push, so the
    constraint is active only when taut (||r|| >= L) and releases when slack.
    This is the same gating as foot contact, for the same reason: the multiplier
    must be one-signed.  A slack rope is not a weak rope, it is *no constraint*,
    and a bilateral distance constraint would wrongly hold the package UP.
  * `SpringCable` -- the compliant counterpart: a one-sided spring-damper that
    applies force without a multiplier.  Cheaper and always smooth, but it
    stretches, and stiff enough to not stretch is stiff enough to need a tiny dt.

This is why the framework carries both: `Cable` gets the kinematics exactly right
for free, `SpringCable` is the penalty model you fall back to when you want the
constraint softened deliberately.

Sign conventions.  With the attachment point a = p + R d and r = p_load - a,

    rdot = v_load - v + hat(R d) omega

so the velocity Jacobian rows are [-I, hat(R d), I] for a rigid link, and their
projection onto rhat for a cable.  The acceleration bias carries the centripetal
term -omega x (omega x R d) that the attachment point has purely from spinning.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .so3 import hat


class Connector(ABC):
    """A constraint joining a carrier attachment point to one payload body."""

    name: str = "connector"
    unilateral: bool = False

    def __init__(self, compliance: float = 1e-7, alpha: float = 30.0, beta: float = 30.0):
        self._eps = float(compliance)
        self.alpha, self.beta = float(alpha), float(beta)

    @abstractmethod
    def n_rows(self) -> int: ...

    @abstractmethod
    def rows(self, p, R, v, om, pl, vl, d):
        """(J [..., m, 6+3], resid [..., m], act [..., m], bias [..., m]).

        J is expressed on [v | omega | v_load] for THIS payload; the caller
        scatters it into the full generalized velocity.
        """

    def force(self, p, R, v, om, pl, vl, d):
        """Direct force on the payload for compliant connectors.  None if this
        connector works through a multiplier instead."""
        return None

    def rest_offset(self) -> tuple:
        """Payload position at rest, RELATIVE TO THE ATTACHMENT POINT.

        The initial state has to satisfy the connector, or Baumgarte spends the
        first moments yanking the payload into place -- which for a weld means a
        0.43 m correction that visibly jolts the carrier.  A cable hangs at its
        length; a rigid link sits at the attachment point itself.
        """
        return (0.0, 0.0, 0.0)

    def compliance(self) -> float:
        return self._eps


def _attach(p, R, om, d):
    """Attachment point, its velocity contribution, and its centripetal term."""
    Rd = torch.einsum("...ij,...j->...i", R, d)
    a = p + Rd
    cent = torch.cross(om, torch.cross(om, Rd, dim=-1), dim=-1)   # omega x (omega x Rd)
    return a, Rd, cent


class RigidLink(Connector):
    """Bilateral weld: the payload sits at a fixed offset, rigidly."""

    name = "rigid_link"

    def n_rows(self) -> int:
        return 3

    def rows(self, p, R, v, om, pl, vl, d):
        a, Rd, cent = _attach(p, R, om, d)
        r = pl - a
        rdot = vl - v + torch.einsum("...ij,...j->...i", hat(Rd), om)

        lead = r.shape[:-1]
        eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(lead + (3, 3))
        J = torch.cat([-eye, hat(Rd), eye], dim=-1)               # [..., 3, 9]
        act = torch.ones(lead + (3,), dtype=r.dtype, device=r.device)
        bias = cent - 2.0 * self.alpha * rdot - (self.beta ** 2) * r
        return J, r, act, bias


class Cable(Connector):
    """Unilateral distance constraint: taut at length L, absent when slack.

    One row, on the direction of the cable.  The activation is a smooth function
    of slack, so the rope engages continuously at the moment it snaps taut rather
    than switching -- `allocate` is differentiated through this.
    """

    name = "cable"
    unilateral = True

    def __init__(self, length: float = 0.5, sharpness: float = 600.0, **kw):
        super().__init__(**kw)
        self.length = float(length)
        self.sharpness = float(sharpness)

    def n_rows(self) -> int:
        return 1

    def rest_offset(self) -> tuple:
        return (0.0, 0.0, -self.length)

    def rows(self, p, R, v, om, pl, vl, d):
        a, Rd, cent = _attach(p, R, om, d)
        r = pl - a
        dist = r.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rhat = r / dist
        rdot = vl - v + torch.einsum("...ij,...j->...i", hat(Rd), om)

        c = dist.squeeze(-1) - self.length                        # >= 0 => taut
        cdot = (rhat * rdot).sum(-1)
        # curvature term: the part of ddist/dt^2 that is not linear in accelerations
        perp = (rdot * rdot).sum(-1) - cdot * cdot
        jdq = perp / dist.squeeze(-1) - (rhat * cent).sum(-1)

        rows = torch.cat([-rhat, torch.einsum("...i,...ij->...j", rhat, hat(Rd)), rhat],
                         dim=-1)[..., None, :]                    # [..., 1, 9]
        act = torch.sigmoid(c * self.sharpness)[..., None]        # taut => 1
        bias = (-jdq - 2.0 * self.alpha * cdot - (self.beta ** 2) * c)[..., None]
        return rows, c[..., None], act, bias


class SpringCable(Connector):
    """Compliant one-sided cable: a force, not a multiplier.

    Kept for the same reason the quadruped keeps a learned contact penalty
    alongside its contact constraints -- a deliberately soft connector is
    sometimes the model you want, and it makes the constraint-vs-penalty contrast
    available on the same plant.
    """

    name = "spring_cable"

    def __init__(self, length: float = 0.5, k: float = 4.0e3, c: float = 30.0, **kw):
        super().__init__(**kw)
        self.length, self.k, self.c = float(length), float(k), float(c)

    def n_rows(self) -> int:
        return 0

    def rest_offset(self) -> tuple:
        return (0.0, 0.0, -self.length)

    def rows(self, p, R, v, om, pl, vl, d):
        raise RuntimeError("SpringCable is compliant; it contributes no rows")

    def force(self, p, R, v, om, pl, vl, d):
        """Tension along the cable, zero when slack.  Returns the force ON the
        payload; the carrier gets the reaction."""
        a, Rd, _ = _attach(p, R, om, d)
        r = pl - a
        dist = r.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rhat = r / dist
        stretch = (dist.squeeze(-1) - self.length).clamp_min(0.0)
        rdot = vl - v + torch.einsum("...ij,...j->...i", hat(Rd), om)
        rate = (rhat * rdot).sum(-1)
        taut = (stretch > 0).to(r.dtype)
        tension = taut * (self.k * stretch + self.c * rate).clamp_min(0.0)
        return -tension[..., None] * rhat                          # pulls payload back


CONNECTORS = {"rigid_link": RigidLink, "cable": Cable, "spring_cable": SpringCable}


def make_connector(kind: str, **kw) -> Connector:
    if kind not in CONNECTORS:
        raise KeyError(f"unknown connector {kind!r}; registered: {sorted(CONNECTORS)}")
    return CONNECTORS[kind](**kw)
