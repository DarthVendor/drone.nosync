"""Sensing as a deformation of the desired Lagrangian.

Rather than estimating state and evaluating V_d(x_hat), a sensor perturbs the
energy landscape directly: L_d = L_0(theta) + dL(theta, obs).  That is strictly
weaker to guarantee than an estimator -- it needs a sign condition and a support
condition rather than unbiasedness -- and it composes as an ordinary
`LagrangianTerm`.

**The partition by argument kind is not cosmetic.** `SensorPotential` consumes
'position_like' and 'range' channels; `SensorDissipation` consumes
'velocity_like' ones.  A single unstructured dL depending on both produces forces
with no sign guarantee -- gyroscopic terms that are neither conservative nor
dissipative, and which no amount of K_d will stabilize.

Two conditions hold by construction here, and both are asserted in tests:

1. **dV >= 0, and dV == 0 for ||e|| < r_goal.**  This is what makes the
   equilibrium immune to sensor bias.  Unlike V_d(x_hat), whose minimum sits
   wherever the estimator's bias vanishes, dV cannot move the goal no matter what
   the sensor reports, because it has no support there at all.  The gate is a
   smoothstep that is EXACTLY zero below r_goal, not merely small.

2. **A slew limit on the parameters.**  A time-varying potential has
   dV/dt = grad V . xdot + partial V/partial t, and that second term is a power
   input: a sensor-driven potential that snaps around pumps energy into the
   closed loop however well-behaved K_d is.  `SlewLimited` interpolates between
   parameter updates instead of stepping -- a 1 Hz *stepped* potential is far
   worse than a 1 Hz *slewed* one, because every step is an impulsive
   partial V/partial t.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .terms import LagrangianTerm


def goal_gate(e: Tensor, r_goal: float, width: float) -> Tensor:
    """Smoothstep, EXACTLY zero for ||e|| <= r_goal and 1 beyond r_goal + width."""
    t = ((e.norm(dim=-1) - r_goal) / max(width, 1e-9)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def goal_gate_grad(e: Tensor, r_goal: float, width: float) -> Tensor:
    """d(gate)/d(e), [..., d].  Zero inside the goal ball, so the whole term --
    value and gradient -- vanishes there."""
    n = e.norm(dim=-1).clamp_min(1e-9)
    w = max(width, 1e-9)
    t = ((n - r_goal) / w).clamp(0.0, 1.0)
    dt = 6.0 * t * (1.0 - t) / w
    return (dt / n)[..., None] * e


class SensorPotential(LagrangianTerm):
    """dV(theta, obs) >= 0, with no support near the goal."""

    kind = "sensor_potential"
    uses_obs = True

    def __init__(self, d: int, sensor_name: str, r_goal: float = 0.35,
                 gate_width: float = 0.35):
        super().__init__(d)
        self.sensor_name = sensor_name
        self.r_goal, self.gate_width = float(r_goal), float(gate_width)

    # --- subclasses supply these two ---------------------------------------
    def raw_potential(self, theta: Tensor, obs: Tensor) -> Tensor:
        raise NotImplementedError

    def raw_grad(self, theta: Tensor, obs: Tensor) -> Tensor:
        """d(raw_potential)/d(obs), [..., obs_dim]."""
        raise NotImplementedError

    # --- composed ----------------------------------------------------------
    def _read(self, obs):
        if obs is None or self.sensor_name not in obs:
            return None, None
        return obs[self.sensor_name], obs.get(self.sensor_name + "/J")

    def potential(self, theta, e, v, x, obs=None):
        z, _ = self._read(obs)
        if z is None:
            return torch.zeros_like(e[..., 0])
        return goal_gate(e, self.r_goal, self.gate_width) * self.raw_potential(theta, z)

    def grad_potential(self, theta, e, v, x, obs=None):
        z, J = self._read(obs)
        if z is None:
            return torch.zeros_like(e)
        g = goal_gate(e, self.r_goal, self.gate_width)
        dV = self.raw_grad(theta, z)                       # [..., obs_dim]
        pull = torch.einsum("...o,...od->...d", dV, J) if J is not None \
            else torch.zeros_like(e)
        # product rule: the gate depends on e as well as scaling the sensor term
        return (g[..., None] * pull
                + self.raw_potential(theta, z)[..., None]
                * goal_gate_grad(e, self.r_goal, self.gate_width))

    def certificate(self, theta, goal=None):
        return {"kind": self.kind, "psd": True, "zero_at_goal": True,
                "bounded_grad": True, "r_goal": self.r_goal,
                "sensor": self.sensor_name}


class FovBarrier(SensorPotential):
    """Keep image features away from the frame border.

    Features leaving the frame is image-based servoing's best-known failure mode,
    and it is a pure geometry problem -- exactly the sort of thing a barrier in
    the potential handles better than a term in the reward.

    Genome: [weight, margin_raw].  The barrier is compactly supported in pixel
    space, so a feature comfortably inside the image contributes nothing at all.
    """

    kind = "fov_barrier"

    def __init__(self, d: int, sensor_name: str, width: int, height: int,
                 n_landmarks: int, w0: float = 0.5, margin_px: float = 30.0, **kw):
        super().__init__(d, sensor_name, **kw)
        self.W, self.H, self.K = float(width), float(height), int(n_landmarks)
        self.w0, self.margin_px = float(w0), float(margin_px)

    @property
    def dim(self) -> int:
        return 2

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return torch.tensor([self.w0, self.margin_px ** 0.5], dtype=dtype, device=device)

    def _margin(self, raw):
        return raw * raw + 1.0

    def _bump(self, m):
        """psi(m) = clamp(1-m, 0, 1)^3 -- exactly zero once a feature is more
        than `margin` inside the border, and bounded when it is outside."""
        u = (1.0 - m.clamp_min(0.0)).clamp(0.0, 1.0)
        return u * u * u, -3.0 * u * u * (m >= 0).to(m.dtype)

    def _borders(self, uv):
        """Distance to each of the four borders, in units of the margin."""
        u, v = uv[..., 0], uv[..., 1]
        return torch.stack([u, self.W - u, v, self.H - v], dim=-1)

    def raw_potential(self, theta, obs):
        w2 = theta[..., 0] ** 2
        m = self._margin(theta[..., 1])[..., None, None]
        uv = obs.reshape(obs.shape[:-1] + (self.K, 2))
        psi, _ = self._bump(self._borders(uv) / m)
        # MEAN over landmarks, not sum: the barrier's strength should not depend
        # on how many features the environment happens to contain, and the pixel
        # pullback is already O(100 px/m), so a sum over K puts the commanded
        # force several times the vehicle's own weight at the prior.
        return w2 * psi.sum(-1).mean(-1)

    def raw_grad(self, theta, obs):
        w2 = theta[..., 0] ** 2
        m = self._margin(theta[..., 1])[..., None, None]
        uv = obs.reshape(obs.shape[:-1] + (self.K, 2))
        _, d = self._bump(self._borders(uv) / m)
        d = d / m
        du = d[..., 0] - d[..., 1]              # +u border, -u border
        dv = d[..., 2] - d[..., 3]
        g = w2[..., None, None] * torch.stack([du, dv], dim=-1) / self.K
        return g.reshape(obs.shape)

    def describe(self, theta):
        return {"fov_w2": float(theta[..., 0] ** 2),
                "fov_margin_px": float(self._margin(theta[..., 1]))}


class SensorDissipation(LagrangianTerm):
    """Rayleigh term on a velocity-like channel, R = 1/2 z^T (E E^T) z.

    Parameterized as E E^T so it is PSD by construction -- the same reason
    `DissipationTerm` uses D D^T.  Kept separate from `SensorPotential` so that a
    single net can never straddle position-like and velocity-like observations.
    """

    kind = "sensor_dissipation"
    uses_obs = True

    def __init__(self, d: int, sensor_name: str, obs_dim: int, e0: float = 0.0):
        super().__init__(d)
        self.sensor_name = sensor_name
        self.obs_dim = int(obs_dim)
        self.e0 = float(e0)

    @property
    def dim(self) -> int:
        return self.d * self.obs_dim

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return torch.full((self.dim,), self.e0, dtype=dtype, device=device)

    def _E(self, theta):
        return theta.reshape(theta.shape[:-1] + (self.d, self.obs_dim))

    def damping(self, theta):
        E = self._E(theta)
        return E @ E.transpose(-1, -2)

    def potential(self, theta, e, v, x, obs=None):
        if obs is None or self.sensor_name not in obs:
            return torch.zeros_like(e[..., 0])
        z = obs[self.sensor_name]
        Ez = torch.einsum("...ij,...j->...i", self._E(theta), z)
        return 0.5 * (Ez * Ez).sum(-1)

    def grad_potential(self, theta, e, v, x, obs=None):
        if obs is None or self.sensor_name not in obs:
            return torch.zeros_like(e)
        z = obs[self.sensor_name]
        E = self._E(theta)
        return torch.einsum("...ij,...kj,...k->...i", E, E, z)

    def certificate(self, theta, goal=None):
        eig = torch.linalg.eigvalsh(self.damping(theta))
        return {"kind": self.kind, "psd": True, "zero_at_goal": True,
                "bounded_grad": False, "R_min": float(eig[..., 0].min())}


class RangeBarrier(SensorPotential):
    """Obstacle avoidance from MEASUREMENTS rather than from known geometry.

    Consumes a 'range' channel and pushes away from whatever is close, without
    ever being told where the obstacles are.  That is the difference that makes
    generalization meaningful: an `ObstacleBarrier` term is handed the geometry at
    construction and cannot transfer to a layout it has not been given, while this
    one sees only beams and so is indifferent to the particular scene.

    Genome: [weight, safe_raw, margin_raw].  Compactly supported -- a beam
    reporting more than `safe + margin` contributes exactly nothing, so open
    space costs neither force nor gradient.
    """

    kind = "range_barrier"

    def __init__(self, d: int, sensor_name: str, n_beams: int, w0: float = 1.0,
                 safe0: float = 0.45, margin0: float = 0.9, **kw):
        super().__init__(d, sensor_name, **kw)
        self.n_beams = int(n_beams)
        self.w0, self.safe0, self.margin0 = float(w0), float(safe0), float(margin0)

    @property
    def dim(self) -> int:
        return 3

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return torch.tensor([self.w0, self.safe0 ** 0.5, self.margin0 ** 0.5],
                            dtype=dtype, device=device)

    def _params(self, theta):
        w2 = theta[..., 0] ** 2
        safe = theta[..., 1] ** 2 + 0.05
        margin = theta[..., 2] ** 2 + 0.05
        return w2, safe, margin

    def _bump(self, m):
        u = (1.0 - m.clamp_min(0.0)).clamp(0.0, 1.0)
        return u * u * u, -3.0 * u * u * (m >= 0).to(m.dtype)

    def raw_potential(self, theta, obs):
        w2, safe, margin = self._params(theta)
        m = (obs - safe[..., None]) / margin[..., None]
        psi, _ = self._bump(m)
        return w2 * psi.mean(-1)

    def raw_grad(self, theta, obs):
        w2, safe, margin = self._params(theta)
        m = (obs - safe[..., None]) / margin[..., None]
        _, d = self._bump(m)
        return w2[..., None] * d / margin[..., None] / self.n_beams

    def describe(self, theta):
        w2, safe, margin = self._params(theta)
        return {"rng_w2": float(w2), "rng_safe": float(safe),
                "rng_margin": float(margin)}
