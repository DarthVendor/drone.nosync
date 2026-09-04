"""`FullState` -- the identity sensor.

Reports `system.task_position(s)` with no noise, no dropout and zero latency, so
the framework's existing behaviour becomes a special case of the sensing path
rather than a parallel one.

This is the regression gate.  With `FullState` and `latency_steps = 0`, every
section 6 acceptance number must reproduce BIT-IDENTICALLY.  If it does not, the
sensor seam is in the wrong place and nothing else in the addendum should be
built until it is.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..systems.base import LagrangianSystem, State
from .base import Sensor


class FullState(Sensor):
    kind = "position_like"
    name = "full_state"
    latency_steps = 0
    #: Pinned to every step, unlike the physical sensors.  This is the identity
    #: baseline -- the thing the no-sensor path must reproduce bit-for-bit -- so
    #: striding it would quietly make it something other than the identity.
    update_every = 1

    def __init__(self, system: LagrangianSystem, latency_steps: int = 0):
        self.system = system
        self.obs_dim = int(system.task_dim)
        self.latency_steps = int(latency_steps)

    def observe(self, s: State, gen: torch.Generator) -> Tensor:
        return self.system.task_position(s)

    def jacobian(self, s: State) -> Tensor:
        eye = torch.eye(self.obs_dim, dtype=self.system.dtype,
                        device=self.system.device)
        return eye.expand(self._batch(s) + (self.obs_dim, self.obs_dim))


class FullStateVelocity(Sensor):
    """Companion identity channel for velocity-like observations."""

    kind = "velocity_like"
    name = "full_state_velocity"
    update_every = 1          # identity baseline; see FullState

    def __init__(self, system: LagrangianSystem, latency_steps: int = 0):
        self.system = system
        self.obs_dim = int(system.task_dim)
        self.latency_steps = int(latency_steps)

    def observe(self, s: State, gen: torch.Generator) -> Tensor:
        return self.system.task_velocity(s)

    def jacobian(self, s: State) -> Tensor:
        eye = torch.eye(self.obs_dim, dtype=self.system.dtype,
                        device=self.system.device)
        return eye.expand(self._batch(s) + (self.obs_dim, self.obs_dim))


class NoisyPosition(Sensor):
    """Task position with Gaussian noise, dropout and latency.

    The minimum sensor that actually exercises the seam -- noise for common
    random numbers, dropout for `valid`, latency for the delay buffer -- without
    any of the projection machinery.  Stands in for a mocap or GNSS channel.
    """

    kind = "position_like"
    name = "noisy_position"

    def __init__(self, system: LagrangianSystem, sigma: float = 0.02,
                 latency_steps: int = 0, dropout: float = 0.0):
        self.system = system
        self.obs_dim = int(system.task_dim)
        self.sigma = float(sigma)
        self.latency_steps = int(latency_steps)
        self.dropout = float(dropout)

    def observe(self, s: State, gen: torch.Generator) -> Tensor:
        x = self.system.task_position(s)
        if self.sigma == 0.0:
            return x
        n = self.crn_noise(x.shape, gen, x.dtype, x.device)
        return x + self.sigma * n

    def jacobian(self, s: State) -> Tensor:
        eye = torch.eye(self.obs_dim, dtype=self.system.dtype,
                        device=self.system.device)
        return eye.expand(self._batch(s) + (self.obs_dim, self.obs_dim))

    def valid(self, s: State, gen=None) -> Tensor:
        if self.dropout <= 0.0 or gen is None:
            return super().valid(s, gen)
        x = self.system.task_position(s)
        u = torch.rand(x.shape, generator=gen, dtype=x.dtype, device=x.device)
        return u >= self.dropout

    def describe(self) -> dict:
        d = super().describe()
        d.update(sigma=self.sigma, dropout=self.dropout)
        return d
