"""The `Sensor` abstraction and per-sensor delay.

A camera does not change the plant Lagrangian -- M(q)qddot + C qdot + g = u is
indifferent to what we measure -- so M^-1 in G(theta) stays a ground-truth,
design-time quantity and no estimator ever enters the metric.  What sensing
changes is the DESIRED Lagrangian: rather than estimating state and evaluating
V_d(x_hat), the sensor perturbs the energy landscape directly,

    L_d = L_0(theta) + dL(theta, obs) .

That is strictly weaker to guarantee -- it needs dV >= 0 with dV == 0 near the
goal, rather than an unbiased estimator -- and it composes as an ordinary
`LagrangianTerm`.

Dependency position: `systems/ -> sensors/ -> trainables/`.  This subpackage may
import `systems/` (it needs `State` and `so3`) and nothing above it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import Tensor

from ..systems.base import State


class Sensor(ABC):
    """One measurement channel: what it reports, how late, and how noisily."""

    obs_dim: int = 0
    latency_steps: int = 0          # measured in SIM STEPS, not seconds

    #: refresh the measurement every N control steps, holding the last reading in
    #: between.  1 = every step.
    #:
    #: This is physical, not only a speed knob: a real camera runs at 30-60 Hz and
    #: a ToF at 50-200 Hz against a loop that wants 250-500 Hz, so sensing at loop
    #: rate is the unrealistic case.  A FIXED stride is also safer than the
    #: difference-triggered skipping the spec describes, because latency stays
    #: deterministic -- threshold skipping makes it state-dependent (little change
    #: -> skip -> open-loop drift -> big change -> correction burst), which is a
    #: limit-cycle generator that fires in hover.
    #:
    #: The held value is the last MEASUREMENT, not a dead-reckoned estimate; with
    #: a large stride that staleness is a real cost the search will trade against
    #: (fitness degrades 10.31 -> 11.66 going from every step to every tenth).
    #:
    #: Default 5.  At dt = 0.02 the control loop is 50 Hz, so this is a 10 Hz
    #: sensor -- about right for a scanning lidar, and sensing at full loop rate
    #: was the unphysical case.  It also costs: ray-traced scenes run 4.3x faster
    #: at N=5, and a hoop course 6.8x faster at N=10.
    update_every: int = 5
    kind: str = "position_like"     # 'position_like' | 'velocity_like' | 'range'
    name: str = "sensor"

    #: Size of the common-random-numbers group (the rollout sets this to n_eps).
    #: Noise is drawn for one group and tiled across the population so that every
    #: genome in a generation sees the SAME noise realization -- without it,
    #: sensor stochasticity turns straight into fitness-ranking variance, and ES
    #: is already variance-limited.
    crn_group: int = 1

    @classmethod
    def supports(cls, system) -> bool:
        """Is this sensor meaningful on this plant?

        Mirrors `Trainable.supports`: a range sensor needs a plant that has an
        environment to cast against, so the conformance sweep skips pairs that
        were never intended rather than reporting them as failures.
        """
        return True

    @abstractmethod
    def observe(self, s: State, gen: torch.Generator) -> Tensor:
        """[B, obs_dim].  Noise draws come from `gen`, so they take part in
        common random numbers."""

    @abstractmethod
    def jacobian(self, s: State) -> Tensor:
        """d(obs)/d(task_position), [B, obs_dim, task_dim].

        Used to pull potential gradients back into task space.  Must be vmap- and
        jacrev-safe: sensors enter the metric only through du/dtheta, which
        `jacrev` picks up automatically once `forward` consumes observations.
        """

    def valid(self, s: State, gen: Optional[torch.Generator] = None) -> Tensor:
        """[B, obs_dim] bool.  False = dropout / out of FOV / no return.

        `gen` is optional and, when omitted, only DETERMINISTIC invalidity is
        reported (geometry: behind the camera, outside the image).  Stochastic
        dropout needs a generator so it joins common random numbers rather than
        becoming an unseeded source of fitness variance.

        Treat "no return" as UNKNOWN, not as clear.  A barrier that vanishes when
        an obstacle stops being visible is worse than no barrier, because the
        controller becomes confident exactly when it should not be.
        """
        return torch.ones(self._batch(s) + (self.obs_dim,), dtype=torch.bool,
                          device=self._device(s))

    # --- helpers ------------------------------------------------------------
    @staticmethod
    def _batch(s: State) -> tuple:
        return tuple(next(iter(s.values())).shape[:1])

    @staticmethod
    def _device(s: State):
        return next(iter(s.values())).device

    def crn_noise(self, shape, gen: torch.Generator, dtype, device) -> Tensor:
        """Gaussian noise shared across the population, independent per episode.

        The rollout lays the batch out as index = member * n_eps + episode, so a
        draw of [n_eps, ...] repeated `pop` times gives every genome the same
        realization while still varying across episodes.
        """
        B = shape[0]
        g = int(self.crn_group)
        if g > 1 and B % g == 0:
            base = torch.randn((g,) + tuple(shape[1:]), generator=gen,
                               dtype=dtype, device=device)
            return base.repeat((B // g,) + (1,) * (len(shape) - 1))
        return torch.randn(tuple(shape), generator=gen, dtype=dtype, device=device)

    def describe(self) -> dict:
        return {"sensor": type(self).__name__, "obs_dim": self.obs_dim,
                "latency_steps": self.latency_steps, "kind": self.kind,
                "update_every": self.update_every}

    # --- rendering ----------------------------------------------------------
    def render_spec(self) -> Optional[dict]:
        """Static description of what this sensor draws, or None if nothing.

        Kept on the ABC deliberately: adding the hook now costs nothing and is
        awkward to retrofit once several sensors exist.
        """
        return None

    def render_frame(self, s: State) -> dict:
        """Per-frame overlay data (landmark projections, beams, FOV)."""
        return {}


class DelayBuffer:
    """Fixed-depth ring buffer holding past observations for ONE sensor.

    Per-sensor, not global: flow and IMU run at ~2 ms, ToF at 5-20 ms, vision at
    30-80 ms, and collapsing them onto one delay throws away the distinction the
    allocator/potential timescale split depends on.

    Why this is worth having before any real sensor exists: delay costs omega*tau
    of phase margin.  Evolve without it and evolution discovers stiff, lightly
    damped potentials that are optimal in sim and oscillate on hardware.  It is
    the classic sim-to-real failure for evolved gains -- cheap to prevent,
    expensive to diagnose after the fact.
    """

    def __init__(self, latency_steps: int = 0):
        if latency_steps < 0:
            raise ValueError(f"latency_steps must be >= 0, got {latency_steps}")
        self.latency = int(latency_steps)
        self._buf: List[Tensor] = []

    def reset(self, obs0: Tensor) -> None:
        """Prime with `latency` copies, so the first reads are the initial value
        rather than zeros."""
        self._buf = [obs0.clone() for _ in range(self.latency)]

    def push(self, obs: Tensor) -> Tensor:
        """Append the newest observation, return the one `latency` steps old.

        With `latency = 0` this returns `obs` itself, which is what makes the
        zero-delay path bit-identical to having no buffer at all.
        """
        self._buf.append(obs)
        return self._buf.pop(0)

    def __len__(self) -> int:
        return self.latency + 1
