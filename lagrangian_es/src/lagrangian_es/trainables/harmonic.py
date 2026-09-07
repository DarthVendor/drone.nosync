"""A harmonic obstacle potential, built from remembered sensor returns.

Why this exists
---------------
The closed loop is `M xddot = -grad V_d - K_d xdot` with `K_d` PSD, so energy is
non-increasing and LaSalle puts every bounded trajectory on the CRITICAL POINTS
of `V_d`.  Pathing is therefore decided by the critical-point structure of `V_d`,
not by training -- and a goal bowl plus locally-supported repulsive barriers has
spurious minima (Koditschek-Rimon: such a sum is not a navigation function).
Measured on a pillar dead ahead, the soft bump gives an unstable equilibrium
(eigenvalues -0.86, -0.16: the vehicle falls off it, into the pillar or around
it) and a hard log wall gives a STABLE one (+9.51, +61.47: it parks 2 m short of
the goal, permanently).  Neither is pathing.

A harmonic potential cannot do that.  If `lap V = 0` then at any critical point
the Hessian eigenvalues sum to zero, so they cannot all be positive: there is no
interior local minimum, only saddles, and every saddle has a descent direction.

Why the existing barrier is not harmonic
----------------------------------------
It very nearly is.  The force from `-log d` is exactly the point-charge field.
What breaks it is that the beams are BODY-FIXED: as the vehicle moves, beam i
re-aims and lands somewhere else on the obstacle, so the charges slide instead of
staying put, and the field becomes a function of distance-to-surface whose
Laplacian is dominated by `+1/d^2`.  Measured at the same point, same beams, same
weights: sliding sources give `lap V = 69.4155`, sources frozen in the world
frame give `0.0039`.

So the fix is not a different potential.  It is remembering where the returns
came from.

What this term promises, and what it does not
---------------------------------------------
* No local minima away from the goal, because each charge is harmonic and a sum
  of harmonic functions is harmonic.
* The goal stays the unique minimum: `goal_gate` is EXACTLY zero inside `r_goal`,
  so near the goal `V_d` is the bowl alone -- one minimum, bounded force.  The
  gate's transition band is the one place harmonicity is not claimed, and it sits
  where obstacles are excluded by construction.
* The guarantee is against what has been SEEN.  Beams sample the boundary at a
  finite number of points; between charges nothing holds the vehicle out.
* Saddles are permitted, and the vehicle can slow near one.  It cannot stop
  there: the negative-eigenvalue eigenvector is a descent direction, and measured
  on a pillar it comes out almost purely lateral -- (+0.35, -0.94), i.e. "around".
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from .sensor_terms import goal_gate, goal_gate_grad
from .terms import LagrangianTerm

EPS = 1e-9


class HarmonicField(LagrangianTerm):
    """Repulsion from remembered world-frame charges.

    Genome: [weight_raw, softening_raw].  Two slots -- the field's SHAPE is fixed
    by harmonicity and is not the search's to choose; all it may set is how hard
    the charges push and how close they may be approached before the singularity
    is softened.

    `kernel` picks the Green's function, and it has to match what the charges
    stand for.  "log2d" treats each return as a vertical line charge, which is
    exact for pillars and walls and makes the field harmonic in the horizontal
    plane the vehicle navigates in.  "inv3d" treats it as a point charge,
    harmonic in full 3-D, for scenes whose obstacles are not extruded.
    """

    kind = "harmonic_field"
    uses_obs = True
    #: tells the rollout to maintain a world-frame charge memory for this term
    needs_charges = True

    def __init__(self, d: int, w0: float = 1.0, soft0: float = 0.10,
                 kernel: str = "log2d", r_goal: float = 0.35,
                 gate_width: float = 0.35, soft_lo: float = 0.02,
                 soft_hi: float = 0.60):
        super().__init__(d)
        if kernel not in ("log2d", "inv3d"):
            raise ValueError(f"unknown harmonic kernel {kernel!r}")
        self.kernel = kernel
        self.w0, self.soft0 = float(w0), float(soft0)
        self.r_goal, self.gate_width = float(r_goal), float(gate_width)
        self.soft_lo, self.soft_hi = float(soft_lo), float(soft_hi)

    @property
    def dim(self) -> int:
        return 2

    @staticmethod
    def _unsquash(v, lo, hi):
        t = min(max((v - lo) / (hi - lo), 1e-4), 1.0 - 1e-4)
        return math.log(t / (1.0 - t))

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return torch.tensor(
            [self.w0, self._unsquash(self.soft0, self.soft_lo, self.soft_hi)],
            dtype=dtype, device=device)

    def _params(self, theta):
        w2 = theta[..., 0] ** 2
        # bounded, for the same reason the range barrier's are: an unbounded
        # softening is a fitness-flat switch-me-off direction once it exceeds the
        # scene, and crossover extrapolation walks flat directions to infinity
        soft = (self.soft_lo
                + (self.soft_hi - self.soft_lo) * torch.sigmoid(theta[..., 1]))
        return w2, soft

    # --- the field ----------------------------------------------------------
    def _read(self, obs):
        if obs is None or "charges" not in obs:
            return None, None
        return obs["charges"], obs.get("charge_w")

    def _sep(self, x, p):
        """x - p_k, restricted to the plane the kernel lives in."""
        dx = x[..., None, :] - p
        if self.kernel == "log2d":
            dx = torch.cat([dx[..., :2], torch.zeros_like(dx[..., 2:])], dim=-1)
        return dx

    def raw(self, theta, x, p, m):
        """(V, grad_x V) from the charges alone, before the goal gate."""
        w2, soft = self._params(theta)
        dx = self._sep(x, p)
        r2 = (dx * dx).sum(-1) + soft[..., None] ** 2      # softened, never 0
        if self.kernel == "log2d":
            # V = -1/2 log r^2 ; grad = -dx / r^2
            V = -0.5 * torch.log(r2)
            g = -dx / r2[..., None]
        else:
            r = r2.clamp_min(EPS).sqrt()
            V = 1.0 / r
            g = -dx / (r2 * r)[..., None]
        V = (w2[..., None] * m * V).sum(-1)
        g = (w2[..., None, None] * m[..., None] * g).sum(-2)
        return V, g

    def potential(self, theta, e, v, x, obs=None):
        p, m = self._read(obs)
        if p is None:
            return torch.zeros_like(e[..., 0])
        V, _ = self.raw(theta, x, p, m)
        # shifted to be nonnegative: the certificate claims psd, and -log r is
        # negative for r > 1.  A constant offset changes no force.
        return goal_gate(e, self.r_goal, self.gate_width) * torch.relu(V)

    def grad_potential(self, theta, e, v, x, obs=None):
        p, m = self._read(obs)
        if p is None:
            return torch.zeros_like(e)
        V, g = self.raw(theta, x, p, m)
        gate = goal_gate(e, self.r_goal, self.gate_width)
        return (gate[..., None] * g
                + torch.relu(V)[..., None]
                * goal_gate_grad(e, self.r_goal, self.gate_width))

    def certificate(self, theta: Tensor, goal: Optional[Tensor] = None) -> Dict:
        return {"kind": self.kind, "psd": True, "zero_at_goal": True,
                "bounded_grad": True,      # the softening bounds it
                "harmonic": True, "kernel": self.kernel,
                "r_goal": self.r_goal}

    def describe(self, theta: Tensor) -> Dict[str, float]:
        w2, soft = self._params(theta)
        return {"harm_w2": float(w2), "harm_soft": float(soft)}


class HarmonicGoal(LagrangianTerm):
    """Attraction to the goal that is itself HARMONIC.

    Why this exists
    ---------------
    `HarmonicField` removes local minima from the OBSTACLE part of the
    potential, and measured on its own it does: `lap V = 0.0000` exactly.  But
    the potential the vehicle actually descends is the sum, and the three
    quadratic `GoalBowl` terms contribute `lap V = +8.69` between them.  A sum of
    harmonic functions is harmonic; a harmonic function plus a bowl is not, and a
    potential with positive Laplacian may have interior minima.  So the
    guarantee never applied to the composition, which is why swapping only the
    obstacle term left 56% of whole-city tours trapped against the barrier's 65%.

    The attractor has to be harmonic too.  A point SINK is: with the same
    `log2d` kernel the field uses, `V = +1/2 log(r^2)` is harmonic away from the
    origin (a function of the horizontal radius alone has no z curvature, so it
    is harmonic in 3-D as well as 2-D).  Sink plus sources is then harmonic
    everywhere except at the charges and the goal, and a harmonic function has
    no interior minimum at all -- only saddles, every one of which has a way
    down.

    What it costs
    -------------
    A bowl's force vanishes at the goal and a sink's does not: `grad V ~ 1/r`
    grows without bound as the goal is approached.  `soft` is what makes that
    finite, and it sets the residual pull the damping has to absorb, so the
    equilibrium is no longer exactly the goal -- it is within `O(soft)` of it.
    That is the trade: a bowl is exact at the goal and can trap you far from it;
    a sink can never trap you and is approximate at the goal.  `tol` is 0.25 m
    and `soft` is bounded well below that.
    """

    kind = "harmonic_goal"

    def __init__(self, d: int, w0: float = 1.0, soft0: float = 0.12,
                 kernel: str = "log2d", soft_lo: float = 0.04,
                 soft_hi: float = 0.40):
        super().__init__(d)
        if kernel not in ("log2d", "inv3d"):
            raise ValueError(f"unknown harmonic kernel {kernel!r}")
        self.kernel = kernel
        self.w0, self.soft0 = float(w0), float(soft0)
        self.soft_lo, self.soft_hi = float(soft_lo), float(soft_hi)

    @property
    def dim(self) -> int:
        return 2

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return torch.tensor(
            [self.w0, HarmonicField._unsquash(self.soft0, self.soft_lo,
                                              self.soft_hi)],
            dtype=dtype, device=device)

    def _params(self, theta):
        w2 = theta[..., 0] ** 2
        soft = (self.soft_lo
                + (self.soft_hi - self.soft_lo) * torch.sigmoid(theta[..., 1]))
        return w2, soft

    def _plane(self, e):
        if self.kernel == "log2d":
            return torch.cat([e[..., :2], torch.zeros_like(e[..., 2:])], dim=-1)
        return e

    def potential(self, theta, e, v, x, obs=None):
        w2, soft = self._params(theta)
        d = self._plane(e)
        r2 = (d * d).sum(-1) + soft ** 2
        if self.kernel == "log2d":
            return w2 * 0.5 * torch.log(r2)
        return -w2 / r2.clamp_min(EPS).sqrt()

    def grad_potential(self, theta, e, v, x, obs=None):
        w2, soft = self._params(theta)
        d = self._plane(e)
        r2 = (d * d).sum(-1) + soft ** 2
        if self.kernel == "log2d":
            g = d / r2[..., None]
        else:
            r = r2.clamp_min(EPS).sqrt()
            g = d / (r2 * r)[..., None]
        out = w2[..., None] * g
        if self.kernel == "log2d":
            # the plane kernel says nothing about height, so height is left to
            # the bowls or the dissipation; returning a fabricated z force here
            # would be a claim the potential does not make
            out = torch.cat([out[..., :2], torch.zeros_like(out[..., 2:])], -1)
        return out

    def certificate(self, theta, goal=None):
        w2, soft = self._params(theta)
        return {"kind": self.kind, "psd": False,          # a sink is unbounded below
                "zero_at_goal": False,                    # V(0) = w^2 log(soft)
                "bounded_grad": True,
                "grad_bound": float((w2 / soft).sum()),
                "harmonic": True}

    def describe(self, theta):
        w2, soft = self._params(theta)
        return {"w2": float(w2), "soft": float(soft), "kernel": self.kernel}
