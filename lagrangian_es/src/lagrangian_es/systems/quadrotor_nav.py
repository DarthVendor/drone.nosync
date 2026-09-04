"""Quadrotor in an obstacle field.

`QuadrotorSE3` plus an `Environment`.  The obstacle geometry lives IN THE STATE,
batched per episode, so a held-out evaluation seed is automatically a held-out
set of layouts -- generalization is measured rather than assumed, and there is no
separate bookkeeping to get wrong.

Collision enters through `alive`, which is the honest place for it: hitting a
pillar ends the episode the same way hitting the ground does, and the search
never differentiates through it.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import State
from .environment import Environment, Hoops, Mixture, make_environment
from .quadrotor import QuadrotorSE3


class QuadrotorNav(QuadrotorSE3):
    state_keys = ("p", "v", "R", "om")          # plus the environment's own fields

    def __init__(self, environment: str = "pillars",
                 env: Optional[Environment] = None, goal_margin: float = 0.30, **kw):
        super().__init__(**kw)
        self.env = env if env is not None else make_environment(environment)
        self._env_keys: tuple = ()
        self.goal_margin = float(goal_margin)
        # In a mixture a hoop is just another obstacle and may not even be live
        # this episode, so pinning it to a waypoint would both un-park it and
        # delete the obstacle.  Only a dedicated hoop scene makes them gates.
        self.place_hoops = not isinstance(self.env, Mixture)
        # Always: hoops have to be placed ON the route, and every other group has
        # to be pushed OFF it.  Both need the goals, so both go through the hook.
        self.needs_course = bool(self.env.groups)

    # --- state carries the scene --------------------------------------------
    def reset(self, B: int, gen: torch.Generator) -> State:
        s = super().reset(B, gen)
        field = self.env.sample(B, gen, self.dtype, self.device)
        self._env_keys = tuple(field)
        s.update(field)
        return s

    def step(self, s: State, u: Tensor, dt: float, params: Tensor = None) -> State:
        out = super().step(s, u, dt, params)
        for k in s:                              # geometry is constant, carry it
            if k not in out:
                out[k] = s[k]
        return out

    def place_course(self, s: State, goals: Tensor) -> State:
        """Put one hoop on each waypoint, facing along the leg that reaches it.

        A racing gate faces down the course, so the heading for leg k is the
        direction from the previous waypoint (the start, for k = 0) to this one.
        Each hoop keeps the tilt it sampled, so a course still mixes upright,
        tilted and flat gates -- it is only the position and the facing that the
        route dictates.
        """
        # Clear FIRST, then place.  Without this a few percent of goals sit inside
        # geometry and are simply unreachable -- a permanent ceiling on the success
        # rate that no amount of training removes, i.e. a measurement artefact
        # masquerading as a controller limitation.  Placement runs afterwards so a
        # gate still lands exactly on its waypoint.
        pts = torch.cat([s["p"][:, None, :], goals], dim=1)
        out = dict(s)
        out.update(self.env.clear_points(out, pts, self.goal_margin))
        if self.place_hoops:
            prev = torch.cat([s["p"][:, None, :], goals[:, :-1, :]], dim=1)
            head = goals - prev
            for g in self.env.groups:
                if isinstance(g, Hoops):
                    out.update(g.place(out, goals, head))
        return out

    def alive(self, s: State) -> Tensor:
        return super().alive(s) & (self.clearance(s) > 0.0)

    # --- environment queries -------------------------------------------------
    def clearance(self, s: State) -> Tensor:
        """Signed distance to the nearest obstacle, [...]."""
        return self.env.sdf(s["p"], s)

    def raycast(self, s: State, dirs: Tensor, max_range: float = 4.0):
        return self.env.raycast(s["p"], dirs, s, max_range)

    def shaping_cost(self, s: State) -> Tensor:
        """Tilt, plus a proximity penalty that bites only near an obstacle."""
        tilt = 1.0 - s["R"][..., 2, 2]
        near = (0.6 - self.clearance(s)).clamp_min(0.0)
        return tilt + 2.0 * near * near

    # --- rendering ----------------------------------------------------------
    def render_spec(self) -> dict:
        spec = super().render_spec()
        spec["environment"] = self.env.describe()
        return spec

    def render_static(self, s: State) -> dict:
        """Obstacle geometry, one entry per group."""
        return {"obstacles": self.env.render(s)}

    def render_extras(self, s: State) -> dict:
        return {}

    def describe(self) -> dict:
        d = super().describe()
        d.update(self.env.describe())
        return d
