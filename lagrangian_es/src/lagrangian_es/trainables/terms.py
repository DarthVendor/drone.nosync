"""Composable Lagrangian terms.

A genome no longer decodes to one fixed potential; it decodes to a *list* of
terms, each contributing to a desired Lagrangian L_d = T_d - V_d together with
dissipation and path constraints.

Why composition is safe: Lagrangians add, and the Euler-Lagrange operator is
linear in L.  A conic combination of terms with nonnegative weights therefore
superposes forces, grad(sum_i V_i) = sum_i grad V_i, and preserves
positive-definiteness about the goal.  The "goal is the closed-loop equilibrium"
invariant survives composition for free -- which is the entire reason the
structured genome was worth having.

Each term publishes a `certificate` saying what it actually promises, and the
trainable's equilibrium claim is the CONJUNCTION of its terms' certificates
rather than a class constant.  A barrier that overlaps the goal genuinely does
move the equilibrium, and the certificate is where that stops being a footnote:
`test_conformance` reads it, so a term that cannot keep the promise
automatically removes the claim instead of silently invalidating a test.

Signature note.  The spec sketch passes `(theta, x, xdot)`.  This implementation
passes `(theta, e, v, x)` -- task error, task velocity, and *absolute* task
position -- because a goal-relative frame cannot express an obstacle fixed in the
world, and folding the two into one argument would force every barrier to
reconstruct the other from a goal it was never handed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import torch
from torch import Tensor


class LagrangianTerm(ABC):
    """One additive contribution to the desired Lagrangian.

    All methods are single-sample and must be vmap- and jacrev-safe: no in-place
    ops, no Python branches on tensor values, every normalize floored.
    """

    #: human-readable kind, used in logs and in `describe`
    kind: str = "term"

    #: does this term consume sensor observations?  Terms that do get `obs`
    #: passed through; the rest keep their original signature untouched.
    uses_obs: bool = False

    def __init__(self, d: int):
        self.d = int(d)

    @property
    @abstractmethod
    def dim(self) -> int:
        """Genome slots this term owns."""

    @abstractmethod
    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        """[dim] prior for this term alone."""

    # --- the contributions --------------------------------------------------
    def potential(self, theta: Tensor, e: Tensor, v: Tensor, x: Tensor) -> Tensor:
        """Scalar, >= 0.  Zero for a term that contributes only kinetic shaping."""
        return torch.zeros_like(e[..., 0])

    def grad_potential(self, theta: Tensor, e: Tensor, v: Tensor, x: Tensor) -> Tensor:
        """Closed-form generalized force contribution, [..., d].

        For a potential this is grad_e V; for a Rayleigh dissipation term it is
        dR/dv.  Both are forces, and both are summed into the same bracket.
        """
        return torch.zeros_like(e)

    def kinetic_shape(self, theta: Tensor, x: Tensor) -> Optional[Tensor]:
        """PSD contribution to the desired mass matrix M_d, [..., d, d], or None."""
        return None

    # --- what the term promises --------------------------------------------
    @abstractmethod
    def certificate(self, theta: Tensor, goal: Optional[Tensor] = None) -> Dict:
        """Claims this term makes about itself.

        Required keys:
          `psd`           -- the contribution is nonnegative
          `zero_at_goal`  -- value AND gradient vanish at e = 0, so the term does
                             not move the closed-loop equilibrium.  Barriers may
                             answer conditionally when handed a `goal`.
          `bounded_grad`  -- the force contribution is bounded in ||e||
        """

    def describe(self, theta: Tensor) -> Dict[str, float]:
        return {}

    def _t(self, x, dtype, device) -> Tensor:
        return torch.as_tensor(x, dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
class GoalBowl(LagrangianTerm):
    """One pseudo-Huber bowl: V = w^2 (sqrt(1 + ||A e||^2) - 1).

    Nonnegative, zero exactly at the goal, and with a gradient that saturates at
    w^2 sigma_max(A) as ||e|| -> infinity -- so the commanded force is bounded
    through the potential's own geometry rather than by a clip applied afterwards.

    A bowl is one term rather than the whole potential precisely so that
    term-level crossover can swap individual bowls between parents.
    """

    kind = "goal_bowl"

    def __init__(self, d: int, w0: float = 1.0, a0: float = 1.0):
        super().__init__(d)
        self.w0, self.a0 = float(w0), float(a0)

    @property
    def dim(self) -> int:
        return 1 + self.d * self.d

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        eye = torch.eye(self.d, dtype=dtype, device=device)
        return torch.cat([torch.full((1,), self.w0, dtype=dtype, device=device),
                          (self.a0 * eye).reshape(-1)])

    def _wA(self, theta: Tensor):
        w2 = theta[..., 0] ** 2
        A = theta[..., 1:].reshape(theta.shape[:-1] + (self.d, self.d))
        return w2, A

    def potential(self, theta, e, v, x):
        w2, A = self._wA(theta)
        Ae = torch.einsum("...ij,...j->...i", A, e)
        return w2 * (torch.sqrt(1.0 + (Ae * Ae).sum(-1)) - 1.0)

    def grad_potential(self, theta, e, v, x):
        w2, A = self._wA(theta)
        Ae = torch.einsum("...ij,...j->...i", A, e)
        r = torch.sqrt(1.0 + (Ae * Ae).sum(-1))
        return torch.einsum("...,...ij,...i->...j", w2 / r, A, Ae)

    def stiffness(self, theta: Tensor) -> Tensor:
        """Hessian at the goal: w^2 A^T A."""
        w2, A = self._wA(theta)
        return w2[..., None, None] * torch.einsum("...ij,...ik->...jk", A, A)

    def certificate(self, theta, goal=None):
        w2, A = self._wA(theta)
        return {"kind": self.kind, "psd": True, "zero_at_goal": True,
                "bounded_grad": True,
                "grad_bound": float((w2 * torch.linalg.matrix_norm(A, ord=2)).sum())}

    def describe(self, theta):
        w2, A = self._wA(theta)
        eig = torch.linalg.eigvalsh(self.stiffness(theta))
        return {"w2": float(w2), "K_min": float(eig[0]), "K_max": float(eig[-1])}


class DissipationTerm(LagrangianTerm):
    """Rayleigh dissipation R = 1/2 v^T K_d v with K_d = D D^T, PSD by construction.

    Strictly this is not part of L -- it is the dissipation function that sits
    beside it -- but it enters the same force bracket and obeys the same additive
    algebra, so it composes identically.
    """

    kind = "dissipation"

    def __init__(self, d: int, d0: float = 1.2):
        super().__init__(d)
        self.d0 = float(d0)

    @property
    def dim(self) -> int:
        return self.d * self.d

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return (self.d0 * torch.eye(self.d, dtype=dtype, device=device)).reshape(-1)

    def _D(self, theta):
        return theta.reshape(theta.shape[:-1] + (self.d, self.d))

    def damping(self, theta: Tensor) -> Tensor:
        D = self._D(theta)
        return D @ D.transpose(-1, -2)

    def potential(self, theta, e, v, x):
        D = self._D(theta)
        Dv = torch.einsum("...ji,...j->...i", D, v)      # D^T v
        return 0.5 * (Dv * Dv).sum(-1)                   # 1/2 v^T D D^T v >= 0

    def grad_potential(self, theta, e, v, x):
        D = self._D(theta)
        return torch.einsum("...ij,...kj,...k->...i", D, D, v)   # (D D^T) v

    def certificate(self, theta, goal=None):
        eig = torch.linalg.eigvalsh(self.damping(theta))
        return {"kind": self.kind, "psd": True, "zero_at_goal": True,
                "bounded_grad": False,       # linear in v, unbounded in ||v||
                "Kd_min": float(eig[0]), "Kd_max": float(eig[-1])}

    def describe(self, theta):
        eig = torch.linalg.eigvalsh(self.damping(theta))
        return {"Kd_min": float(eig[0]), "Kd_max": float(eig[-1])}


class KineticShaping(LagrangianTerm):
    """A PSD contribution W W^T to the desired mass matrix M_d.

    With M_d = M + sum_k W_k W_k^T, the control law becomes

        u = g(q) - M M_d^-1 ( sum_i grad V_i + K_d xdot ),

    whose Lyapunov function is E_d = 1/2 xdot^T M_d xdot + V_d, giving
    Edot_d = -xdot^T K_d xdot <= 0.

    That derivation is EXACT when M is constant -- which is precisely the case on
    both quadrotors, where M is constant in the body frame.  On a plant where
    M(q) varies (the arm), it drops the Coriolis correction demanded by the
    IDA-PBC matching conditions, so it is an approximation there; the equilibrium
    invariant still holds exactly, because M M_d^-1 is nonsingular and therefore
    annihilates nothing, but the shaped-kinetic-energy reading is only approximate.

    The same constant-vs-varying M split that bounds the whitening claim bounds
    this one, and for the same reason.
    """

    kind = "kinetic_shaping"

    def __init__(self, d: int, w0: float = 0.0):
        super().__init__(d)
        self.w0 = float(w0)

    @property
    def dim(self) -> int:
        return self.d * self.d

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return torch.full((self.d * self.d,), self.w0, dtype=dtype, device=device)

    def kinetic_shape(self, theta, x):
        W = theta.reshape(theta.shape[:-1] + (self.d, self.d))
        return W @ W.transpose(-1, -2)

    def certificate(self, theta, goal=None):
        M = self.kinetic_shape(theta, None)
        return {"kind": self.kind, "psd": True, "zero_at_goal": True,
                "bounded_grad": True,
                "Md_max": float(torch.linalg.eigvalsh(M)[-1])}


class _Bump(LagrangianTerm):
    """Shared machinery for compactly supported barriers.

    psi(s) = (1 - s)^3 on [0, 1], EXACTLY zero for s >= 1, and extended linearly
    as 1 - 3s for s < 0.

    Compact support on the outside is what lets a barrier keep the equilibrium
    claim: beyond its margin both value and gradient vanish identically, so
    `zero_at_goal` is true whenever the goal sits clear of the barrier -- a
    conditional certificate rather than a blanket forfeit.  A 1/d-style barrier
    could not make that promise.

    The linear extension on the inside matters just as much.  Cubed, (1 - s)
    grows without bound once s goes negative -- i.e. once the limit is actually
    violated -- so a barrier that is merely "compactly supported" commands an
    unbounded force exactly when it is doing its job, destroying the bounded-force
    property the pseudo-Huber bowls were chosen for.  Extending linearly keeps the
    potential C^1 and caps |dpsi/ds| at 3 everywhere, while still pushing back
    monotonically harder the further outside you are.
    """

    @staticmethod
    def _psi(s: Tensor):
        u = (1.0 - s.clamp_min(0.0)).clamp_min(0.0)
        inner, d_inner = u * u * u, -3.0 * u * u
        outside = s < 0.0                       # limit violated: linear, bounded slope
        psi = torch.where(outside, 1.0 - 3.0 * s, inner)
        dpsi = torch.where(outside, torch.full_like(s, -3.0), d_inner)
        return psi, dpsi

    def _margin(self, raw: Tensor) -> Tensor:
        return raw * raw + 1e-3                 # positive and smooth


class ObstacleBarrier(_Bump):
    """Bounded, compactly supported repulsion from a sphere fixed in task space.

    Genome: [weight, margin_raw].  The obstacle's geometry is a property of the
    task, not of the genome -- evolution tunes how hard and how early to push,
    not where the obstacle is.
    """

    kind = "obstacle"

    def __init__(self, d: int, center, radius: float, w0: float = 1.0, margin0: float = 0.6):
        super().__init__(d)
        self.center_t = tuple(float(c) for c in center)
        self.radius = float(radius)
        self.w0, self.margin0 = float(w0), float(margin0)

    @property
    def dim(self) -> int:
        return 2

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return torch.tensor([self.w0, self.margin0 ** 0.5], dtype=dtype, device=device)

    def _s(self, x: Tensor):
        c = torch.as_tensor(self.center_t, dtype=x.dtype, device=x.device)
        rel = x - c
        dist = rel.norm(dim=-1).clamp_min(1e-6)
        return rel, dist

    def potential(self, theta, e, v, x):
        w2 = theta[..., 0] ** 2
        m = self._margin(theta[..., 1])
        _, dist = self._s(x)
        psi, _ = self._psi((dist - self.radius) / m)
        return w2 * psi

    def grad_potential(self, theta, e, v, x):
        w2 = theta[..., 0] ** 2
        m = self._margin(theta[..., 1])
        rel, dist = self._s(x)
        _, dpsi = self._psi((dist - self.radius) / m)
        # dV/ddist * ddist/dx ; negative dpsi => force points away from the sphere
        scale = w2 * dpsi / m / dist
        return scale[..., None] * rel

    def certificate(self, theta, goal=None):
        m = float(self._margin(theta[..., 1]))
        clear = None
        if goal is not None:
            c = torch.as_tensor(self.center_t, dtype=goal.dtype, device=goal.device)
            clear = bool(((goal - c).norm(dim=-1) >= self.radius + m).all())
        return {"kind": self.kind, "psd": True,
                # compact support: exactly zero beyond radius + margin
                "zero_at_goal": True if clear else (False if clear is None else clear),
                "bounded_grad": True, "margin": m, "radius": self.radius,
                "goal_clear_of_barrier": clear}

    def describe(self, theta):
        return {"obs_w2": float(theta[..., 0] ** 2),
                "obs_margin": float(self._margin(theta[..., 1]))}


class JointLimitBarrier(_Bump):
    """Compactly supported repulsion from per-coordinate box limits.

    Genome: [weight, margin_raw].  Written coordinate-wise so it applies to joint
    space on the arm and to a flight envelope on a quadrotor without change.
    """

    kind = "joint_limit"

    def __init__(self, d: int, lo, hi, w0: float = 1.0, margin0: float = 0.25):
        super().__init__(d)
        self.lo_t = tuple(float(v) for v in (lo if hasattr(lo, "__len__") else [lo] * d))
        self.hi_t = tuple(float(v) for v in (hi if hasattr(hi, "__len__") else [hi] * d))
        self.w0, self.margin0 = float(w0), float(margin0)

    @property
    def dim(self) -> int:
        return 2

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return torch.tensor([self.w0, self.margin0 ** 0.5], dtype=dtype, device=device)

    def _bounds(self, x):
        lo = torch.as_tensor(self.lo_t, dtype=x.dtype, device=x.device)
        hi = torch.as_tensor(self.hi_t, dtype=x.dtype, device=x.device)
        return lo, hi

    def potential(self, theta, e, v, x):
        w2 = theta[..., 0] ** 2
        m = self._margin(theta[..., 1])[..., None]
        lo, hi = self._bounds(x)
        p_lo, _ = self._psi((x - lo) / m)
        p_hi, _ = self._psi((hi - x) / m)
        return w2 * (p_lo + p_hi).sum(-1)

    def grad_potential(self, theta, e, v, x):
        w2 = theta[..., 0] ** 2
        m = self._margin(theta[..., 1])[..., None]
        lo, hi = self._bounds(x)
        _, d_lo = self._psi((x - lo) / m)
        _, d_hi = self._psi((hi - x) / m)
        return w2[..., None] * (d_lo - d_hi) / m

    def certificate(self, theta, goal=None):
        m = float(self._margin(theta[..., 1]))
        clear = None
        if goal is not None:
            lo, hi = self._bounds(goal)
            clear = bool(((goal - lo >= m) & (hi - goal >= m)).all())
        return {"kind": self.kind, "psd": True,
                "zero_at_goal": True if clear else (False if clear is None else clear),
                "bounded_grad": True, "margin": m,
                "goal_clear_of_barrier": clear}


TERMS: Dict[str, Type[LagrangianTerm]] = {
    "goal_bowl": GoalBowl, "dissipation": DissipationTerm,
    "kinetic_shaping": KineticShaping, "obstacle": ObstacleBarrier,
    "joint_limit": JointLimitBarrier,
}


def make_term(kind: str, d: int, **kw) -> LagrangianTerm:
    if kind not in TERMS:
        raise KeyError(f"unknown term {kind!r}; registered: {sorted(TERMS)}")
    return TERMS[kind](d, **kw)
