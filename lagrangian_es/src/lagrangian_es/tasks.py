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
from .util import make_gen, uniform


class Task(ABC):
    n_legs: int = 1
    tol: float = 0.25          # success radius, task-space units

    #: "time"    -- the goal advances on a fixed schedule
    #: "arrival" -- it advances when the vehicle actually reaches the waypoint
    #:
    #: Arrival gating is what makes speed worth anything.  Under a timer the
    #: vehicle is scored on where it is at pre-set moments, so dawdling to a
    #: waypoint costs nothing as long as it arrives before the switch; under
    #: arrival gating, reaching a waypoint early buys a longer tail of low cost
    #: at the next one, so the fastest route is the cheapest one.
    gating: str = "time"

    def __init__(self, system: LagrangianSystem):
        self.system = system
        self.task_dim = system.task_dim

    @abstractmethod
    def sample(self, n: int, gen: torch.Generator) -> Tensor:
        """[n, n_legs, task_dim]"""

    @abstractmethod
    def goal_at(self, goals: Tensor, t: int, ep_steps: int) -> Tensor:
        """Active goal under TIME gating.  [B, n_legs, d] -> [B, d]."""

    def leg_end_steps(self, ep_steps: int) -> list:
        """Last step index of each leg -- where per-leg error is measured."""
        n = self.n_legs
        return [(i + 1) * ep_steps // n - 1 for i in range(n)]

    def goal_for_leg(self, goals: Tensor, leg: Tensor) -> Tensor:
        """Active goal under ARRIVAL gating: whichever leg each episode is on."""
        idx = leg.clamp(0, self.n_legs - 1)
        return goals.gather(1, idx[:, None, None].expand(-1, 1, goals.shape[-1])
                            ).squeeze(1)

    def success(self, s: State, goal: Tensor) -> Tensor:
        """[B] bool, for reporting only -- never part of the fitness."""
        return (self.system.task_position(s) - goal).norm(dim=-1) < self.tol


class WaypointPair(Task):
    """Takeoff, translate, re-target, hover.

    Leg A holds for t < ep_steps / 2, then the goal jumps to leg B.  The jump is
    the point: it is a step input applied to an already-moving vehicle, which is
    where a controller that merely reaches the first waypoint falls apart.
    """

    n_legs = 2

    def __init__(self, system, xy: float = 2.0, z_lo: float = 1.0, z_hi: float = 2.5,
                 tol: float = 0.25, gating: str = "arrival"):
        super().__init__(system)
        self.gating = gating
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

    def __init__(self, system, x_range: float = 0.06, z_lo: float = 0.30,
                 z_hi: float = 0.37, pitch: float = 0.10, tol: float = 0.03):
        # Heights must stay inside the legs' reach.  With 0.2 m links the feet can
        # never be more than 0.40 m below the hips, so a goal at 0.40 asks for the
        # fully-straight-leg singularity and a goal above it is unreachable
        # outright -- the controller cannot be blamed for missing either.
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


class HoopCourse(Task):
    """A sequence of gates to fly through, in order.

    Waypoints are the hoop centres, so reaching the goal and passing through the
    gate are the same event -- the vehicle cannot score by drifting past the ring.
    The plant places the hoops on these points via `place_course`.
    """

    def __init__(self, system, n_gates: int = 3, radius: float = 2.0,
                 z_lo: float = 1.0, z_hi: float = 2.0, tol: float = 0.3,
                 gating: str = "arrival"):
        super().__init__(system)
        self.gating = gating
        if system.task_dim != 3:
            raise ValueError(f"HoopCourse needs task_dim 3, got {system.task_dim}")
        self.n_legs = int(n_gates)
        self.radius, self.z_lo, self.z_hi = float(radius), float(z_lo), float(z_hi)
        self.tol = float(tol)

    def sample(self, n: int, gen: torch.Generator) -> Tensor:
        kw = dict(gen=gen, dtype=self.system.dtype, device=self.system.device)
        # gates spread around the vehicle in sequence, so consecutive legs turn
        base = uniform((n, 1), 0.0, 6.283185307179586, **kw)
        step = uniform((n, self.n_legs), 1.2, 2.6, **kw)
        ang = base + torch.cumsum(step, dim=-1)
        rad = uniform((n, self.n_legs), 0.65 * self.radius, self.radius, **kw)
        z = uniform((n, self.n_legs), self.z_lo, self.z_hi, **kw)
        return torch.stack([rad * torch.cos(ang), rad * torch.sin(ang), z], dim=-1)

    def goal_at(self, goals: Tensor, t: int, ep_steps: int) -> Tensor:
        leg = min(t * self.n_legs // ep_steps, self.n_legs - 1)
        return goals[:, leg]


class FreeSpaceWaypoints(Task):
    """Waypoints drawn from the free space of a FIXED scene.

    The counterpart to `place_course`: where a random obstacle field is nudged
    aside to keep a waypoint reachable, imported geometry stays put and the
    waypoints move instead -- a building that dodges the drone is not that
    building any more.

    The pool is precomputed once, because imported geometry is identical across
    episodes and rejection sampling it per episode would be pure waste.
    """

    def __init__(self, system, n_legs: int = 2, pool: int = 4096,
                 margin: float = 0.40, extent: float = 3.0,
                 z_range=(1.0, 2.0), tol: float = 0.25, seed: int = 12345,
                 gating: str = "arrival"):
        super().__init__(system)
        self.gating = gating
        self.n_legs = int(n_legs)
        self.tol = float(tol)
        env = getattr(system, "env", None)
        if env is None:
            raise ValueError("FreeSpaceWaypoints needs a plant with an environment")
        self.pool = env.free_points(int(pool), make_gen(seed), extent=extent,
                                    margin=margin, z_range=z_range,
                                    dtype=system.dtype, device=system.device)

    def sample(self, n: int, gen: torch.Generator) -> Tensor:
        idx = torch.randint(self.pool.shape[0], (n, self.n_legs), generator=gen)
        return self.pool[idx]

    def goal_at(self, goals: Tensor, t: int, ep_steps: int) -> Tensor:
        leg = min(t * self.n_legs // ep_steps, self.n_legs - 1)
        return goals[:, leg]


TASKS: Dict[str, Type[Task]] = {
    "base_pose": BasePose,
    "free_space": FreeSpaceWaypoints,
    "hoop_course": HoopCourse,
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


class ObserveTarget(Task):
    """Hold a vantage point on a target that obstacles can hide.

    The waypoint tasks ask the vehicle to REACH a point, and occlusion resolves
    itself on arrival -- you cannot be blocked from somewhere you are standing.
    This one asks it to WATCH a point from a standoff, which is a different
    problem: no heading sees through a pillar, so when the line is blocked the
    only remedy is to move somewhere else.

    That makes it the task where a camera earns its place.  `sight_cost` is
    fixed by turning and `visibility` only by repositioning, so the two together
    ask for something a pure waypoint controller never has to do -- give up the
    direct route and fly around to where the target can actually be seen.

    Success is deliberately a conjunction of all three: inside the standoff
    band, line of sight clear, and pointed at it.  Any two without the third is
    not an observation.
    """

    n_legs = 1

    def __init__(self, system, xy: float = 2.2, z_lo: float = 0.9, z_hi: float = 2.2,
                 r_near: float = 1.2, r_far: float = 3.0, tol: float = 0.25,
                 look_tol: float = 0.6, r_min: float = 1.9,
                 gating: str = "time"):
        super().__init__(system)
        self.gating = gating
        if system.task_dim != 3:
            raise ValueError(f"ObserveTarget needs task_dim 3, got {system.task_dim}")
        self.xy, self.z_lo, self.z_hi = float(xy), float(z_lo), float(z_hi)
        self.r_near, self.r_far = float(r_near), float(r_far)
        self.tol, self.look_tol = float(tol), float(look_tol)
        self.r_min = float(r_min)

    def sample(self, n: int, gen: torch.Generator) -> Tensor:
        """Targets pushed out to at least `r_min` from where the vehicle starts.

        An occluder placed ON the start-to-target line needs room at both ends,
        and a short leg has none -- those episodes fall back to a scattered
        occluder and the opening view is clear, which is exactly the case the
        task is not about.  Enforcing a minimum leg took reliably-obscured starts
        from 62% to over 90%.
        """
        kw = dict(gen=gen, dtype=self.system.dtype, device=self.system.device)
        xy = uniform((n, self.n_legs, 2), -self.xy, self.xy, **kw)
        r = xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        xy = xy * (self.r_min / r).clamp_min(1.0)
        z = uniform((n, self.n_legs, 1), self.z_lo, self.z_hi, **kw)
        return torch.cat([xy, z], dim=-1)

    def goal_at(self, goals: Tensor, t: int, ep_steps: int) -> Tensor:
        return goals[:, 0]

    def success(self, s: State, goal: Tensor) -> Tensor:
        p = self.system.task_position(s)
        r = (p - goal).norm(dim=-1)
        in_band = (r > self.r_near) & (r < self.r_far)
        seen = self.system.visibility(s, goal) > 0.5
        aimed = self.system.sight_cost(s, goal) < self.look_tol
        return in_band & seen & aimed

register_task("observe")(ObserveTarget)


class AcquireThenReach(ObserveTarget):
    """Line the camera up first, then fly in.

    A waypoint task rewards arriving however you got there, so a controller that
    charges blind is scored the same as one that looked first.  Here the target
    starts hidden behind an obstacle, and credit needs BOTH: the vehicle has to
    establish line of sight at some point, and then reach the target.

    The ordering is not enforced by the scoring -- it falls out of the geometry.
    Sight has to come first because the only way to lose it is to be behind
    something, and the only way to regain it is to move; once the vehicle is at
    the target it trivially sees it, so a "reached" episode that never acquired
    is one that flew in blind. `acquired` is carried in the rollout rather than
    recomputed at the end for exactly that reason: it is a claim about the whole
    trajectory, not about its last frame.
    """

    def __init__(self, system, tol: float = 0.3, **kw):
        kw.setdefault("gating", "time")
        super().__init__(system, tol=tol, **kw)

    def success(self, s: State, goal: Tensor) -> Tensor:
        reached = (self.system.task_position(s) - goal).norm(dim=-1) < self.tol
        return reached & (self.system.visibility(s, goal) > 0.5)


register_task("acquire_then_reach")(AcquireThenReach)
