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

import math
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
                 safe0: float = 0.45, margin0: float = 0.9,
                 safe_lo: float = 0.05, safe_hi: float = 1.50,
                 margin_lo: float = 0.10, margin_hi: float = 2.00,
                 mode: str = "bump", d_min: float = 0.12,
                 floor: float = 0.02, **kw):
        super().__init__(d, sensor_name, **kw)
        self.n_beams = int(n_beams)
        self.w0, self.safe0, self.margin0 = float(w0), float(safe0), float(margin0)
        self.safe_lo, self.safe_hi = float(safe_lo), float(safe_hi)
        self.margin_lo, self.margin_hi = float(margin_lo), float(margin_hi)
        if mode not in ("bump", "log"):
            raise ValueError(f"unknown RangeBarrier mode {mode!r}")
        self.mode = mode
        self.d_min, self.floor = float(d_min), float(floor)

    @property
    def dim(self) -> int:
        return 3

    @staticmethod
    def _unsquash(v, lo, hi):
        t = min(max((v - lo) / (hi - lo), 1e-4), 1.0 - 1e-4)
        return math.log(t / (1.0 - t))

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        return torch.tensor(
            [self.w0,
             self._unsquash(self.safe0, self.safe_lo, self.safe_hi),
             self._unsquash(self.margin0, self.margin_lo, self.margin_hi)],
            dtype=dtype, device=device)

    def _params(self, theta):
        w2 = theta[..., 0] ** 2
        # BOUNDED, via a squash.  Unbounded, `safe` and `margin` are a free
        # "switch me off" direction: once safe exceeds the sensor's max range
        # every beam has m < 0, the bump is constant, its gradient vanishes, and
        # the term stops influencing u at all.  Its parameters are then
        # fitness-flat, and BLX-alpha extrapolation walks them off to infinity --
        # which is exactly what happened, ending at safe = 2.8e29 with the range
        # sensor contributing nothing to the commanded force.
        safe = self.safe_lo + (self.safe_hi - self.safe_lo) * torch.sigmoid(theta[..., 1])
        margin = (self.margin_lo
                  + (self.margin_hi - self.margin_lo) * torch.sigmoid(theta[..., 2]))
        return w2, safe, margin

    def _bump(self, m):
        """Cubic bump on [0, 1], continued LINEARLY below 0.

        The cubic alone is flat for m < 0 -- i.e. the repulsion switches off
        inside the safe radius, exactly where a collision is imminent.  The
        linear continuation keeps the gradient pinned at its maximum there
        instead of dropping to zero, so the force stays bounded (|dpsi/dm| <= 3,
        which is what the certificate needs) without ever vanishing in the danger
        zone.  C1 at m = 0: value 1, slope -3 on both sides.
        """
        mc = m.clamp(0.0, 1.0)
        u = 1.0 - mc
        below = (-m).clamp_min(0.0)          # depth inside `safe`, 0 outside
        return u * u * u + 3.0 * below, -3.0 * u * u

    def _wall(self, obs, safe):
        """Hard-wall barrier: -log((d - d_min) / (safe - d_min)).

        A real wall is an INFINITE potential, and that buys two things a bounded
        bump cannot.  The sublevel set the episode starts in never reaches the
        obstacle, so contact is excluded by the geometry of V rather than merely
        made expensive; and because the resulting force is purely NORMAL, the
        goal pull's tangential component survives untouched and the vehicle
        SLIDES along the obstacle instead of stalling against it.  That sliding is
        the steering a radial repulsion could never produce.

        It is infinite only in continuous time.  At dt = 0.02 and 4 m/s the
        vehicle covers 8 cm per step and can step straight through the wall, so
        the argument is floored -- large but finite, and honest about it.  Note
        also that `obs` is BEAM RANGE, so the invariance is with respect to what
        the sensor sees; between beams there is nothing holding the vehicle out.
        """
        span = (safe[..., None] - self.d_min).clamp_min(1e-6)
        z = ((obs - self.d_min) / span).clamp(self.floor, 1.0)
        # SHIFTED log: -ln z + (z - 1).  Both the value and the slope vanish at
        # z = 1, so the barrier meets open space smoothly instead of with the
        # kink a bare -ln z leaves at the edge of its support.
        inside = (obs < safe[..., None]).to(obs.dtype)
        psi = inside * (-torch.log(z) + (z - 1.0))
        # Saturates at the floor rather than switching off there.  Clamping z is
        # unavoidable -- the vehicle can step through the wall in one dt -- but
        # the push must stay at its maximum once inside, not vanish, or tunnelling
        # through leaves nothing pushing back out.
        d = inside * (1.0 - 1.0 / z) / span
        return psi, d

    def raw_potential(self, theta, obs):
        w2, safe, margin = self._params(theta)
        if self.mode == "log":
            psi, _ = self._wall(obs, safe)
        else:
            psi, _ = self._bump((obs - safe[..., None]) / margin[..., None])
        return w2 * psi.mean(-1)

    def raw_grad(self, theta, obs):
        w2, safe, margin = self._params(theta)
        if self.mode == "log":
            _, d = self._wall(obs, safe)
            return w2[..., None] * d / self.n_beams
        m = (obs - safe[..., None]) / margin[..., None]
        _, d = self._bump(m)
        return w2[..., None] * d / margin[..., None] / self.n_beams

    def describe(self, theta):
        w2, safe, margin = self._params(theta)
        return {"rng_w2": float(w2), "rng_safe": float(safe),
                "rng_margin": float(margin)}

    def certificate(self, theta, goal=None):
        c = super().certificate(theta, goal)
        # the wall's gradient is bounded only by the floor, not by O(1)
        c["bounded_grad"] = self.mode != "log"
        c["mode"] = self.mode
        return c


class RangeDamper(SensorPotential):
    """Dissipation that bites on CLOSING motion toward what the beams see.

    Why a potential cannot do this job
    ----------------------------------
    Stopping needs `v^2 / 2a` of room.  A potential supplies a force that depends
    only on POSITION, so its deceleration `a(d)` is fixed and it can only stop an
    approach from `v <= sqrt(2 a d)`.  Measured on the pillar field: the barrier
    delivers about 2 m/s^2 over 1.35 m of support, good for 2.3 m/s, while 100%
    of deaths are obstacle contact at a mean of 3.09 m/s.  Turning the gain up
    fixes the speed limit and breaks everything else -- at high gain the field
    deflects so hard that reach collapses to 0.21.  The missing dependence is on
    velocity, and in a Lagrangian that is a Rayleigh term, not a potential.

    The construction
    ----------------
        R(v) = 1/2 * sum_i k_i * relu(-J_i . v)^2 ,   k_i = c * g(d_i)

    `J_i = d(range_i)/dx` points AWAY from what beam i sees, so `J_i . v` is the
    rate of change of that range and `relu(-J_i . v)` is the closing speed, zero
    when receding.  The force is `-dR/dv = sum_i k_i relu(-J_i . v) J_i`, which:

      * is strictly dissipative -- `dR/dv . v = sum k relu(...)^2 >= 0` -- so
        `H = T + V_d` is still non-increasing and the certificate is untouched;
      * vanishes on receding motion, so it never pushes the vehicle along;
      * damps ONLY the closing component, leaving tangential motion free, so it
        slows an approach without preventing going around;
      * scales with speed, so the achievable deceleration grows with the speed
        that needs killing instead of being fixed by geometry.

    Genome: [strength_raw, reach_raw].  Both bounded, for the reason the
    barrier's are: an unbounded activation range is a fitness-flat switch-me-off
    direction once it exceeds the scene.

    Only the EXCESS speed is damped
    -------------------------------
    A damper keyed on distance alone brakes whenever anything is within reach,
    and in a field of six pillars that is nearly always -- it pays a toll on
    every episode to rescue the ~17% that would have hit something.  Priced on
    the pillar field that trade is a net loss: the safe policy scores 3.9703
    against the crashing one's 3.7952, so selection correctly dismantles it, and
    600 generations drove the strength from 120 to 6.

    So the damping keys on the SAFE SPEED for the distance in hand,

        v_safe(d) = sqrt(2 * a * d)

    and resists only `relu(v_close - v_safe)`.  Inside the envelope -- cruising
    past a pillar, however close -- the term is exactly zero and costs nothing.
    It engages only when the approach could no longer be stopped in the distance
    remaining, which is the situation it exists for.  Compact support falls out
    for free: far away `v_safe` is large and nothing triggers.

    Genome: [strength_raw, decel_raw] -- how hard to resist, and the deceleration
    the envelope assumes it can command.  Both bounded, and `accel_hi` tightly:
    a LARGE assumed deceleration widens `v_safe` until the term never fires, so a
    loose upper bound is just another switch-me-off direction.  Given the run at
    accel_hi = 20, two of three seeds pinned it at 19.98 and 19.68 and lost reach
    (0.676, 0.688) while the seed that kept 2.20 scored 0.828.  The envelope must
    assume what the plant can ACTUALLY deliver, and measured over the approach to
    a crash this one is saturated 59% of the time with its realised force 49 deg
    off the demand -- so the honest value is small.
    """

    kind = "range_damper"

    def __init__(self, d: int, sensor_name: str, n_beams: int, c0: float = 200.0,
                 accel0: float = 2.0, accel_lo: float = 0.5,
                 accel_hi: float = 6.0, **kw):
        super().__init__(d, sensor_name, **kw)
        self.n_beams = int(n_beams)
        self.c0, self.accel0 = float(c0), float(accel0)
        self.accel_lo, self.accel_hi = float(accel_lo), float(accel_hi)

    @property
    def dim(self) -> int:
        return 2

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        t = min(max((self.accel0 - self.accel_lo)
                    / (self.accel_hi - self.accel_lo), 1e-4), 1.0 - 1e-4)
        return torch.tensor([self.c0 ** 0.5, math.log(t / (1.0 - t))],
                            dtype=dtype, device=device)

    def _params(self, theta):
        c = theta[..., 0] ** 2
        accel = (self.accel_lo + (self.accel_hi - self.accel_lo)
                 * torch.sigmoid(theta[..., 1]))
        return c, accel

    def _v_safe(self, z, accel):
        """Speed the remaining distance can still absorb, sqrt(2 a d)."""
        return (2.0 * accel[..., None] * z.clamp_min(0.0)).clamp_min(0.0).sqrt()

    # The parent's potential/grad machinery is for V(obs); this is R(v, obs), so
    # both are overridden rather than reusing raw_potential/raw_grad.
    def raw_potential(self, theta, obs):
        return torch.zeros_like(obs[..., 0])

    def raw_grad(self, theta, obs):
        return torch.zeros_like(obs)

    def potential(self, theta, e, v, x, obs=None):
        return torch.zeros_like(e[..., 0])

    def grad_potential(self, theta, e, v, x, obs=None):
        z, J = self._read(obs)
        if z is None or J is None:
            return torch.zeros_like(e)
        c, accel = self._params(theta)
        close = torch.einsum("...od,...d->...o", J, v).neg().clamp_min(0.0)
        # only the part of the approach the remaining distance cannot absorb
        excess = (close - self._v_safe(z, accel)).clamp_min(0.0)
        k = c[..., None] / self.n_beams
        # dR/dv = -sum_i k_i * excess_i * J_i ; the caller subtracts it, so the
        # applied force is +sum_i k_i excess_i J_i -- opposing the approach
        dRdv = -torch.einsum("...o,...od->...d", k * excess, J)
        return goal_gate(e, self.r_goal, self.gate_width)[..., None] * dRdv

    def certificate(self, theta, goal=None):
        return {"kind": self.kind, "psd": True, "zero_at_goal": True,
                "bounded_grad": False,     # grows with closing speed, by design
                "dissipative": True, "sensor": self.sensor_name}

    def describe(self, theta):
        c, accel = self._params(theta)
        return {"dmp_c": float(c), "dmp_accel": float(accel)}
