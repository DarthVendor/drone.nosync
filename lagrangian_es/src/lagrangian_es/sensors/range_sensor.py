"""`RangeSensor` -- a fan of horizontal beams, sonar/lidar style.

Ray intersections against the environment are closed form, so there is no
marching loop and the whole thing stays vmap/jacrev safe.  Beams are fanned about
the body yaw, so what the vehicle sees rotates with it.

"No return" reads as `max_range` with ZERO gradient.  That is the honest
derivative of a clamped measurement, and it is also the safe reading: a barrier
built on this cannot become confident just because a beam missed.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from ..systems.base import State
from ..environments import EPS
from .base import Sensor

TWO_PI = 2.0 * 3.141592653589793


class RangeSensor(Sensor):
    """Horizontal fan of range beams.

    24 beams, updated every step.  Both were measured, not guessed: 12 beams over
    2*pi sit 30 deg apart, so adjacent rays are 2 d sin(15 deg) = 0.52 d apart and
    at 1 m a 0.36-0.76 m pillar fits entirely between two of them.  Going 12 -> 24
    took the crash rate 0.027 -> 0.014, and updating every step rather than every
    fifth took it 0.058 -> 0.027 (0.1 s of blindness is 30 cm at 3 m/s).
    """

    kind = "range"
    name = "range"
    # Re-marching the scene every step is the single largest cost in the whole
    # rollout -- profiled at 97% of it on the imported city.  A stride of 8 is
    # 0.16 s of staleness at dt = 0.02.
    #
    # Evaluated COLD it looks expensive: nav99, which was trained at stride 1,
    # goes from 0.0015 to 0.0114 crash when the sensor is strided under it.  But
    # that measures transfer to an input it never saw, not what the stride costs
    # a policy that trains on it -- a controller that always reads 0.16 s late
    # can lead the target, damp harder, or simply stand further off, and none of
    # those are available to one that is handed staleness at test time only.
    update_every = 8

    def __init__(self, system, n_beams: int = 24, max_range: float = 6.0,
                 spread: float = TWO_PI, sigma: float = 0.02,
                 latency_steps: int = 1, elevations: tuple = (0.0,),
                 name: str = "range", body_fixed: bool = True):
        self.system = system
        # instance name: a downward fan and a forward fan are both range
        # sensors and both have to live in one observation dict
        self.name = str(name)
        # `body_fixed`: the fan is bolted to the airframe and turns with yaw
        # AND tilt (the default, the realistic mounting).  False keeps the fan
        # yaw-stabilised in the world-vertical frame, the mounting the earlier
        # hand-designed-stack findings were measured under.
        self.body_fixed = bool(body_fixed)
        self.n_beams = int(n_beams)
        # A purely horizontal fan is blind to anything off its own altitude.
        # That is fine for extruded obstacles like pillars and walls, and
        # wrong for anything with vertical structure -- a flat hoop's ring can
        # be entirely invisible.  `elevations` tilts extra fans out of plane;
        # the default keeps the single horizontal ring.
        self.elevations = tuple(float(e) for e in elevations)
        self.obs_dim = self.n_beams * len(self.elevations)
        self.max_range = float(max_range)
        self.spread = float(spread)
        self.sigma = float(sigma)
        self.latency_steps = int(latency_steps)

    @classmethod
    def supports(cls, system) -> bool:
        return hasattr(system, "raycast")

    def _yaw(self, s: State) -> Tensor:
        R = s["R"]
        return torch.atan2(R[..., 1, 0], R[..., 0, 0])

    def _dirs(self, s: State) -> Tensor:
        """Beam directions in the WORLD frame, from a body-fixed fan.

        The fan is mounted on the vehicle, so it turns with yaw and dips with
        tilt: a banking drone's forward beams look into the ground a little,
        and a downward fan swings off vertical.  Built in the body frame --
        forward is body +x, azimuth in the body xy plane, elevation out of it
        -- then rotated by the full R, not by yaw alone.
        """
        R = s["R"]
        k = torch.arange(self.n_beams, dtype=R.dtype, device=R.device)
        off = (k / self.n_beams - 0.5) * self.spread    # [n_beams], body azimuth
        out = []
        for el in self.elevations:
            ce, se = math.cos(el), math.sin(el)
            out.append(torch.stack([torch.cos(off) * ce, torch.sin(off) * ce,
                                    torch.full_like(off, se)], dim=-1))
        body = torch.cat(out, dim=-2)                   # [n_beams*n_elev, 3]
        if not self.body_fixed:
            # yaw-stabilised: rotate about world z by the yaw only
            yaw = self._yaw(s); c, sn = torch.cos(yaw), torch.sin(yaw)
            Rz = torch.stack([torch.stack([c, -sn, torch.zeros_like(c)], -1),
                              torch.stack([sn, c, torch.zeros_like(c)], -1),
                              torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], -1)], -2)
            return torch.einsum("...ij,kj->...ki", Rz, body)
        # world = R @ body, for every beam
        return torch.einsum("...ij,kj->...ki", R, body)  # [..., n_beams*n_elev, 3]

    def observe(self, s: State, gen: torch.Generator) -> Tensor:
        rng, _ = self.system.raycast(s, self._dirs(s), self.max_range)
        if self.sigma > 0 and gen is not None:
            rng = rng + self.sigma * self.crn_noise(rng.shape, gen, rng.dtype, rng.device)
        return rng.clamp(0.0, self.max_range)

    stateless = True

    def measure(self, s: State):
        """One march, both outputs, no noise."""
        return self.system.raycast(s, self._dirs(s), self.max_range)

    def perturb(self, x: Tensor, gen) -> Tensor:
        if self.sigma > 0 and gen is not None:
            x = x + self.sigma * self.crn_noise(x.shape, gen, x.dtype, x.device)
        return x.clamp(0.0, self.max_range)

    def observe_with_jacobian(self, s: State, gen):
        """One march for both.  The noise draw is identical to `observe`'s, so
        common random numbers are unaffected."""
        rng, grad = self.measure(s)
        return self.perturb(rng, gen), grad

    def jacobian(self, s: State) -> Tensor:
        """d(range)/d(position), [..., n_beams, 3].

        The environment supplies the full 3-D gradient.  For vertical primitives
        its height channel is exactly zero, which is the truth; for a hoop it is
        not, because how high you are really does change how far the ring is.
        """
        _, grad = self.system.raycast(s, self._dirs(s), self.max_range)
        return grad

    def valid(self, s: State, gen=None) -> Tensor:
        rng, _ = self.system.raycast(s, self._dirs(s), self.max_range)
        return rng < self.max_range - EPS

    def describe(self) -> dict:
        d = super().describe()
        d.update(n_beams=self.n_beams, max_range=self.max_range, sigma=self.sigma)
        return d

    def render_spec(self) -> dict:
        return {"type": "range", "n_beams": self.n_beams, "max_range": self.max_range}

    def render_frame(self, s: State) -> dict:
        rng, _ = self.system.raycast(s, self._dirs(s), self.max_range)
        return {"range": rng, "dirs": self._dirs(s)}
