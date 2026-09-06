"""`FixedPD` -- the sanity floor.

Diagonal Kp / Kd on the task-space error, with the same gravity feedforward every
other trainable gets.  Six policy slots.  Not part of the 2x2: it exists so that
"the structured genome helps" can be read against something with almost no
structure to search at all.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..systems.base import LagrangianSystem, State
from .base import Trainable


class FixedPD(Trainable):
    equilibrium_exact = True

    def __init__(self, system: LagrangianSystem, kp0: float = 1.732, kd0: float = 1.2,
                 phi0: tuple = (0.25, 0.25, 0.25, 0.10, 0.10, 0.10)):
        super().__init__(system)
        self.d = int(system.task_dim)
        self.kp0, self.kd0, self.phi0 = float(kp0), float(kd0), tuple(phi0)

    @property
    def policy_dim(self) -> int:
        return 2 * self.d

    def init(self) -> Tensor:
        parts = [
            torch.full((self.d,), self.kp0, dtype=self.dtype, device=self.device),
            torch.full((self.d,), self.kd0, dtype=self.dtype, device=self.device),
        ]
        if self.system.allocator_dim:
            parts.append(self.system.allocator_init())
        if self.system.residual_dim:
            parts.append(self.system.residual_init())
        return torch.cat(parts, dim=0)

    def forward(self, theta: Tensor, s: State, goal: Tensor, obs=None) -> Tensor:
        sysm = self.system
        kp = theta[..., : self.d] ** 2
        kd = theta[..., self.d : 2 * self.d] ** 2
        _, phi = self.split(theta)
        e = sysm.task_position(s) - goal
        F_des = sysm.gravity_force(s) - kp * e - kd * sysm.task_velocity(s)
        return sysm.allocate(F_des, s, phi, goal)

    def describe(self, theta: Tensor) -> dict:
        with torch.no_grad():
            return {
                "kp_mean": float((theta[: self.d] ** 2).mean()),
                "kd_mean": float((theta[self.d : 2 * self.d] ** 2).mean()),
            }
