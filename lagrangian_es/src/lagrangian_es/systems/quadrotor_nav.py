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
from ..environments import Environment, Hoops, Mixture, make_environment
from .quadrotor import QuadrotorSE3


class QuadrotorNav(QuadrotorSE3):
    state_keys = ("p", "v", "R", "om")          # plus the environment's own fields

    def __init__(self, environment: str = "pillars",
                 env: Optional[Environment] = None, goal_margin: float = 0.75,
                 free_start: bool = False, prox_gain: float = 70.0,
                 prox_band: float = 0.60, **kw):
        super().__init__(**kw)
        self.prox_gain = float(prox_gain)
        self.prox_band = float(prox_band)
        self.env = env if env is not None else make_environment(environment)
        self._env_keys: tuple = ()
        # Must exceed the barrier's standoff (safe = 0.45) or the task is
        # ill-posed: the vehicle is pushed away from its own waypoint.  Measured
        # by waypoint clearance, reach was 0.929 for waypoints inside 0.45 m of a
        # pillar against 0.990-1.000 outside it -- the entire shortfall from 99%.
        self.goal_margin = float(goal_margin)
        # In a mixture a hoop is just another obstacle and may not even be live
        # this episode, so pinning it to a waypoint would both un-park it and
        # delete the obstacle.  Only a dedicated hoop scene makes them gates.
        self.place_hoops = not isinstance(self.env, Mixture)
        # Always: hoops have to be placed ON the route, and every other group has
        # to be pushed OFF it.  Both need the goals, so both go through the hook.
        self.needs_course = bool(self.env.groups)
        self.start_pool = None
        if free_start:
            self.use_free_start()

    def use_free_start(self, n: int = 2048, margin: float = 0.45,
                       extent: float = 3.0, z_range=(0.6, 1.6), seed: int = 999):
        """Start episodes in free space rather than at the origin.

        Needed for imported scenes, where the drawing's origin is very often
        inside a wall and every episode would begin in collision.
        """
        from ..util import make_gen
        self.start_pool = self.env.free_points(
            n, make_gen(seed), extent=extent, margin=margin, z_range=z_range,
            dtype=self.dtype, device=self.device)
        return self

    # --- state carries the scene --------------------------------------------
    def reset(self, B: int, gen: torch.Generator) -> State:
        s = super().reset(B, gen)
        field = self.env.sample(B, gen, self.dtype, self.device)
        self._env_keys = tuple(field)
        s.update(field)
        if self.start_pool is not None:
            # inside imported geometry the origin is very often a wall
            idx = torch.randint(self.start_pool.shape[0], (B,), generator=gen)
            s["p"] = self.start_pool[idx].clone()
            s["v"] = torch.zeros_like(s["v"])
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
        """Tilt, plus a proximity penalty that bites only near an obstacle.

        The gain is set so the penalty rate at zero clearance MEETS `dead_cost`:

            lambda_s * prox_gain * prox_band^2  ==  dead_cost

        which for the curriculum's lambda_s = 0.2, dead_cost = 5.0 and a 0.60 m
        band gives 70.  Without that match the objective has a 35x step at the
        collision boundary -- a cliff exactly where the search needs a slope.
        Measured at the old gain of 2.0, this term carried 0.07% of the episode
        cost against 58% for the crash term it exists to anticipate, i.e. 869:1
        against the only smooth obstacle signal in the objective.

        `prox_band` must stay BELOW the aperture a gate is meant to be flown
        through, or passing one correctly is penalized: a 0.55 m hoop offers at
        most 0.48 m of clearance at its centre, so a course wants a band of
        ~0.35 (and, to keep the boundary match above, a gain of ~200).
        """
        tilt = 1.0 - s["R"][..., 2, 2]
        near = (self.prox_band - self.clearance(s)).clamp_min(0.0)
        return tilt + self.prox_gain * near * near

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
