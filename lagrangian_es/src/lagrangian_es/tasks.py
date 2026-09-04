"""Goal distributions.

A task owns what the controller is asked to do and nothing about how.  It is
handed the system so it can size its samples and evaluate success in task space,
but it never reads a raw state key.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Type

import torch
from torch import Tensor

from .systems.base import LagrangianSystem, State
from .util import uniform


class Task(ABC):
    n_legs: int = 1
    tol: float = 0.25          # success radius, task-space units

    def __init__(self, system: LagrangianSystem):
        self.system = system
        self.task_dim = system.task_dim

    @abstractmethod
    def sample(self, n: int, gen: torch.Generator) -> Tensor:
        """[n, n_legs, task_dim]"""

    @abstractmethod
    def goal_at(self, goals: Tensor, t: int, ep_steps: int) -> Tensor:
        """Active goal at integration step t.  goals [B, n_legs, task_dim] -> [B, task_dim]."""

    def leg_end_steps(self, ep_steps: int) -> list:
        """Last step index of each leg -- where per-leg error is measured."""
        n = self.n_legs
        return [(i + 1) * ep_steps // n - 1 for i in range(n)]

    def success(self, s: State, goal: Tensor) -> Tensor:
        """[B] bool, for reporting only -- never part of the fitness."""
        return (self.system.task_position(s) - goal).norm(dim=-1) < self.tol


class WaypointPair(Task):
    """Takeoff, translate, re-target mid-episode, hover.

    Leg A holds for t < ep_steps / 2, then the goal jumps to leg B.  The jump is
    the point: it is a step input applied to an already-moving vehicle, which is
    where a controller that merely reaches the first waypoint falls apart.
    """

    n_legs = 2

    def __init__(self, system, xy: float = 2.0, z_lo: float = 1.0, z_hi: float = 2.5,
                 tol: float = 0.25):
        super().__init__(system)
        if system.task_dim != 3:
            raise ValueError(f"WaypointPair needs task_dim 3, got {system.task_dim}")
        self.xy, self.z_lo, self.z_hi, self.tol = float(xy), float(z_lo), float(z_hi), float(tol)

    def sample(self, n: int, gen: torch.Generator) -> Tensor:
        kw = dict(gen=gen, dtype=self.system.dtype, device=self.system.device)
        xy = uniform((n, self.n_legs, 2), -self.xy, self.xy, **kw)
        z = uniform((n, self.n_legs, 1), self.z_lo, self.z_hi, **kw)
        return torch.cat([xy, z], dim=-1)

    def goal_at(self, goals: Tensor, t: int, ep_steps: int) -> Tensor:
        return goals[:, 0] if t < ep_steps // 2 else goals[:, 1]


class JointTarget(Task):
    """A single joint-space setpoint, for the articulated plants where the task
    space is joint space and `nominal_state` is therefore trivial."""

    n_legs = 1

    def __init__(self, system, lo: float = -1.2, hi: float = 1.2, tol: float = 0.15):
        super().__init__(system)
        self.lo, self.hi, self.tol = float(lo), float(hi), float(tol)

    def sample(self, n: int, gen: torch.Generator) -> Tensor:
        return uniform((n, self.n_legs, self.task_dim), self.lo, self.hi,
                       gen=gen, dtype=self.system.dtype, device=self.system.device)

    def goal_at(self, goals: Tensor, t: int, ep_steps: int) -> Tensor:
        return goals[:, 0]


class JointPair(JointTarget):
    """JointTarget with the same mid-episode re-target as WaypointPair, so the
    arm transfer exercises the identical task structure."""

    n_legs = 2

    def goal_at(self, goals: Tensor, t: int, ep_steps: int) -> Tensor:
        return goals[:, 0] if t < ep_steps // 2 else goals[:, 1]


class BasePose(Task):
    """Floating-base pose targets for a legged robot: (x, height, pitch).

    Squat, shift and lean while keeping the feet planted -- a balance task rather
    than a locomotion one.  The mid-episode re-target is the same structure as
    `WaypointPair`: a step input applied to an already-moving multi-body system,
    which is where a controller that merely reaches the first pose falls apart.

    Not walking.  Gaits need a contact schedule and swing-leg tracking on top of
    this, which is a separate piece of machinery -- see README, "What the
    quadruped does and does not do".
    """

    n_legs = 2

    def __init__(self, system, x_range: float = 0.08, z_lo: float = 0.28,
                 z_hi: float = 0.40, pitch: float = 0.12, tol: float = 0.03):
        super().__init__(system)
        if system.task_dim != 3:
            raise ValueError(f"BasePose needs task_dim 3, got {system.task_dim}")
        self.x_range, self.z_lo, self.z_hi = float(x_range), float(z_lo), float(z_hi)
        self.pitch, self.tol = float(pitch), float(tol)

    def sample(self, n: int, gen: torch.Generator) -> Tensor:
        kw = dict(gen=gen, dtype=self.system.dtype, device=self.system.device)
        x = uniform((n, self.n_legs, 1), -self.x_range, self.x_range, **kw)
        z = uniform((n, self.n_legs, 1), self.z_lo, self.z_hi, **kw)
        th = uniform((n, self.n_legs, 1), -self.pitch, self.pitch, **kw)
        return torch.cat([x, z, th], dim=-1)

    def goal_at(self, goals: Tensor, t: int, ep_steps: int) -> Tensor:
        return goals[:, 0] if t < ep_steps // 2 else goals[:, 1]


TASKS: Dict[str, Type[Task]] = {
    "base_pose": BasePose,
    "waypoint_pair": WaypointPair,
    "joint_target": JointTarget,
    "joint_pair": JointPair,
}


def register_task(name: str) -> Callable:
    def deco(cls):
        if name in TASKS:
            raise KeyError(f"task {name!r} already registered")
        TASKS[name] = cls
        return cls

    return deco


def make_task(name: str, system: LagrangianSystem, **kw) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(TASKS)}")
    return TASKS[name](system, **kw)
