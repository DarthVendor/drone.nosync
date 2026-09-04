"""Holonomic constraints and their multipliers -- the coupling layer.

Coupling between components is expressed as constraints on the Lagrangian rather
than as penalty forces bolted onto it.  For constraints c(q) = 0 the constrained
Euler-Lagrange equations are

    M(q) qddot + b(q, qdot) = S^T tau + J^T lambda ,      J = dc/dq

and lambda -- the Lagrange multiplier -- IS the coupling force.  Differentiating
the constraint twice gives J qddot + Jdot qdot = 0, so qddot and lambda solve one
saddle-point (KKT) system:

    [  M   -J^T ] [ qddot  ]   [ S^T tau - b            ]
    [  J    E   ] [ lambda ] = [ -Jdot qdot - Baumgarte ]

Two practical terms appear on the second row:

  * **E, a compliance (regularization) block.**  E = 0 is a hard constraint; E > 0
    is a compliant one.  It also makes the system nonsingular when constraints are
    redundant -- four feet on flat ground over-determine a planar base, so the
    hard system is genuinely rank-deficient and would fail without it.
  * **Baumgarte stabilization**, -2*alpha*cdot - beta^2*c.  Enforcing the
    constraint at the acceleration level lets position error drift; this pulls it
    back instead of integrating it forever.

Unilateral constraints (a foot may push but not pull) are gated by a smooth
activation in [0, 1] that scales the compliance: as activation -> 0 the row's
compliance -> infinity and its multiplier -> 0, so the constraint releases
continuously rather than switching.  Continuity matters because `allocate` is
jacrev'd through touchdown.

Simplifications, stated plainly: there is no LCP solve and no impact law, so
lambda is clamped nonnegative rather than solved as a complementarity problem, and
friction uses the previous step's normal multiplier (carried in the state) rather
than being solved jointly with it.  Both are standard for a compliant-constraint
integrator and neither is differentiated through by the search.

Note on what does NOT appear here: joint coupling between links.  The quadruped
uses generalized (minimal) coordinates, so the joints are already implicit in
q -- there is no constraint to add.  This module carries the constraints that
minimal coordinates cannot express: contact with the environment, closed
kinematic loops, and deliberate couplings between joints such as gait symmetry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


class HolonomicConstraint(ABC):
    """One block of rows in the constraint Jacobian."""

    name: str = "constraint"
    unilateral: bool = False

    @abstractmethod
    def n_rows(self, sys) -> int: ...

    @abstractmethod
    def rows(self, sys, q: Tensor, dq: Tensor, kin: Dict) -> Tuple[Tensor, ...]:
        """(J [..., m, n_q], jdot_qdot [..., m], residual [..., m], activation [..., m])

        `kin` carries whatever kinematics the system already computed this step,
        so constraints never recompute forward kinematics.
        """

    def compliance(self) -> float:
        return 1e-6


class GroundContact(HolonomicConstraint):
    """Unilateral contact: each foot's height is constrained to zero while loaded.

    The multiplier of row k is the normal force on foot k -- a real, reportable
    quantity rather than a spring constant's side effect.
    """

    name = "ground_contact"
    unilateral = True

    def __init__(self, compliance: float = 2e-6, sharpness: float = 400.0,
                 baumgarte_alpha: float = 40.0, baumgarte_beta: float = 40.0):
        self._eps = float(compliance)
        self.sharpness = float(sharpness)
        self.alpha, self.beta = float(baumgarte_alpha), float(baumgarte_beta)

    def n_rows(self, sys) -> int:
        return sys.n_feet

    def compliance(self) -> float:
        return self._eps

    def rows(self, sys, q, dq, kin):
        Jf = kin["Jf"]                       # [..., n_feet, 2, n_q]
        pf = kin["pf"]                       # [..., n_feet, 2]
        af = kin["af"]                       # [..., n_feet, 2]  Jdot qdot
        J = Jf[..., 1, :]                    # z-row: dc/dq for c = foot height
        jdq = af[..., 1]
        c = pf[..., 1]
        cdot = torch.einsum("...km,...m->...k", J, dq)
        # smooth unilateral gate: fully on below the surface, releasing above it
        act = torch.sigmoid(-c * self.sharpness)
        bias = -jdq - 2.0 * self.alpha * cdot - (self.beta ** 2) * c
        return J, jdq, c, act, bias


class JointCoupling(HolonomicConstraint):
    """Bilateral coupling between generalized coordinates: q[i] - ratio*q[j] = c0.

    The constraint minimal coordinates cannot express.  Use it for gait symmetry
    (couple diagonal legs into a trot), for gearing, or for closing a kinematic
    loop -- the coupling is enforced by a multiplier rather than hoped for by the
    controller.
    """

    name = "joint_coupling"

    def __init__(self, pairs, ratio: float = 1.0, offset: float = 0.0,
                 compliance: float = 1e-5, baumgarte_alpha: float = 40.0,
                 baumgarte_beta: float = 40.0):
        self.pairs = [tuple(p) for p in pairs]
        self.ratio, self.offset = float(ratio), float(offset)
        self._eps = float(compliance)
        self.alpha, self.beta = float(baumgarte_alpha), float(baumgarte_beta)

    def n_rows(self, sys) -> int:
        return len(self.pairs)

    def compliance(self) -> float:
        return self._eps

    def rows(self, sys, q, dq, kin):
        lead, n = q.shape[:-1], q.shape[-1]
        m = len(self.pairs)
        J = torch.zeros(lead + (m, n), dtype=q.dtype, device=q.device)
        c = torch.zeros(lead + (m,), dtype=q.dtype, device=q.device)
        for r, (i, j) in enumerate(self.pairs):
            J[..., r, i] = 1.0
            J[..., r, j] = -self.ratio
            c[..., r] = q[..., i] - self.ratio * q[..., j] - self.offset
        cdot = torch.einsum("...rm,...m->...r", J, dq)
        jdq = torch.zeros_like(c)            # J is constant, so Jdot = 0
        act = torch.ones_like(c)
        bias = -jdq - 2.0 * self.alpha * cdot - (self.beta ** 2) * c
        return J, jdq, c, act, bias


class ConstraintStack:
    """Assembles the constraint rows and solves the KKT system."""

    def __init__(self, constraints: List[HolonomicConstraint]):
        self.constraints = list(constraints)

    def __len__(self):
        return len(self.constraints)

    def n_rows(self, sys) -> int:
        return sum(c.n_rows(sys) for c in self.constraints)

    def assemble(self, sys, q, dq, kin):
        Js, biases, acts, eps, resid = [], [], [], [], []
        for c in self.constraints:
            J, _, r, act, bias = c.rows(sys, q, dq, kin)
            Js.append(J)
            biases.append(bias)
            acts.append(act)
            resid.append(r)
            eps.append(torch.full_like(act, c.compliance()))
        return (torch.cat(Js, dim=-2), torch.cat(biases, dim=-1),
                torch.cat(acts, dim=-1), torch.cat(eps, dim=-1),
                torch.cat(resid, dim=-1))

    def solve(self, M: Tensor, rhs: Tensor, J: Tensor, bias: Tensor,
              act: Tensor, eps: Tensor):
        """Solve the saddle-point system for (qddot, lambda).

        Compliance is divided by the activation, so a released row acquires
        effectively infinite compliance and a zero multiplier -- a continuous
        release rather than a switch, and one that also keeps the matrix
        nonsingular when the active constraints are redundant (four feet on flat
        ground over-determine a planar base).
        """
        lead = M.shape[:-2]
        n, m = M.shape[-1], J.shape[-2]
        E = torch.diag_embed(eps / act.clamp_min(1e-9))

        top = torch.cat([M, -J.transpose(-1, -2)], dim=-1)          # [..., n, n+m]
        bot = torch.cat([J, E], dim=-1)                             # [..., m, n+m]
        A = torch.cat([top, bot], dim=-2)                           # [..., n+m, n+m]
        y = torch.cat([rhs, bias], dim=-1).unsqueeze(-1)
        sol = torch.linalg.solve(A, y).squeeze(-1)
        return sol[..., :n], sol[..., n:]


class PinJointChain(HolonomicConstraint):
    """Pin joints of a serial chain, in MAXIMAL coordinates.

    Each link carries its own (x, z, theta), and the joints that make them a
    chain are enforced by multipliers instead of being built into the coordinates:

        c_0 = prox_0 - anchor                       (2 rows)
        c_i = prox_i - dist_{i-1},   i >= 1         (2 rows each)

    with prox = p - (l/2) u(theta), dist = p + (l/2) u(theta), u = (cos, sin).
    Because u'' = -u, the acceleration bias is (l/2) thetadot^2 u for each end
    involved -- closed form, like everything else in the plant.

    This is the counterpart to the minimal-coordinate formulation, and the
    comparison is the point: in minimal coordinates M(q) is dense and strongly
    configuration-dependent while there are no constraints; here M is CONSTANT
    and block-diagonal, and every bit of that configuration dependence has moved
    into J(q).  Same physics, and the same constrained inverse inertia -- so
    "M(q) varies" turns out to be a statement about coordinates, while the
    quantity the metric actually needs is invariant.
    """

    name = "pin_joint_chain"

    def __init__(self, n_links: int, length: float, anchor=(0.0, 0.0),
                 compliance: float = 1e-8, baumgarte_alpha: float = 60.0,
                 baumgarte_beta: float = 60.0):
        self.n_links = int(n_links)
        self.length = float(length)
        self.anchor_t = tuple(float(a) for a in anchor)
        self._eps = float(compliance)
        self.alpha, self.beta = float(baumgarte_alpha), float(baumgarte_beta)

    def n_rows(self, sys) -> int:
        return 2 * self.n_links

    def compliance(self) -> float:
        return self._eps

    def rows(self, sys, q, dq, kin):
        N, h = self.n_links, 0.5 * self.length
        p, th = kin["p"], kin["th"]              # [..., N, 2], [..., N]
        dth = kin["dth"]                         # [..., N]
        u, w = kin["u"], kin["w"]                # [..., N, 2]
        lead = q.shape[:-1]
        anchor = torch.as_tensor(self.anchor_t, dtype=q.dtype, device=q.device)

        prox = p - h * u
        dist = p + h * u
        c = torch.empty(lead + (N, 2), dtype=q.dtype, device=q.device)
        c[..., 0, :] = prox[..., 0, :] - anchor
        if N > 1:
            c[..., 1:, :] = prox[..., 1:, :] - dist[..., :-1, :]

        J = torch.zeros(lead + (N, 2, 3 * N), dtype=q.dtype, device=q.device)
        jdq = torch.zeros(lead + (N, 2), dtype=q.dtype, device=q.device)
        for i in range(N):
            J[..., i, 0, 3 * i + 0] = 1.0
            J[..., i, 1, 3 * i + 1] = 1.0
            J[..., i, :, 3 * i + 2] = -h * w[..., i, :]
            jdq[..., i, :] = h * (dth[..., i] ** 2)[..., None] * u[..., i, :]
            if i > 0:
                j = i - 1
                J[..., i, 0, 3 * j + 0] = -1.0
                J[..., i, 1, 3 * j + 1] = -1.0
                J[..., i, :, 3 * j + 2] = -h * w[..., j, :]
                jdq[..., i, :] = jdq[..., i, :] + \
                    h * (dth[..., j] ** 2)[..., None] * u[..., j, :]

        J = J.reshape(lead + (2 * N, 3 * N))
        c = c.reshape(lead + (2 * N,))
        jdq = jdq.reshape(lead + (2 * N,))
        cdot = torch.einsum("...rm,...m->...r", J, dq)
        act = torch.ones_like(c)
        bias = -jdq - 2.0 * self.alpha * cdot - (self.beta ** 2) * c
        return J, jdq, c, act, bias


def constrained_inverse_inertia(M: Tensor, J: Tensor, ridge: float = 1e-9) -> Tensor:
    """P = M^-1 - M^-1 J^T (J M^-1 J^T)^-1 J M^-1.

    The inverse inertia a generalized force actually sees once the constraints
    are enforced -- the maximal-coordinate analogue of M(q)^-1, and the quantity
    the mechanical metric is defined against.  Note that P depends on
    configuration through J even when M itself is constant, which is exactly why
    the constant-M caveat is about coordinates rather than about physics.
    """
    Minv = torch.linalg.inv(M)
    JMi = J @ Minv
    S = JMi @ J.transpose(-1, -2)
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
    inner = torch.linalg.solve(S + ridge * eye, JMi)
    return Minv - JMi.transpose(-1, -2) @ inner
