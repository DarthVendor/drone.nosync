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
                 free_start: bool = False, difficulty: float = 1.0,
                 los_range: float = 12.0,
                 los_soft: float = 0.05, los_radius: float = 0.30,
                 los_samples: int = 8, n_occluders: int = 0,
                 occ_lo: float = 0.85, occ_hi: float = 1.35,
                 occ_clear: float = 0.80,
                 occ_seed: int = 12345, occ_t_lo: float = 0.35,
                 occ_t_hi: float = 0.65, prox_gain: float = 70.0,
                 prox_band: float = 0.60, **kw):
        super().__init__(**kw)
        self.los_range = float(los_range)
        self.los_soft = float(los_soft)
        self.los_radius = float(los_radius)
        self.los_samples = int(los_samples)
        self.n_occluders = int(n_occluders)
        self.occ_lo, self.occ_hi = float(occ_lo), float(occ_hi)
        self.occ_clear = float(occ_clear)
        self.occ_seed = int(occ_seed)
        self.occ_t_lo, self.occ_t_hi = float(occ_t_lo), float(occ_t_hi)
        self.prox_gain = float(prox_gain)
        self.prox_band = float(prox_band)
        self.env = env if env is not None else make_environment(environment)
        # Scene difficulty as the fraction of obstacles left ACTIVE.
        #
        # Size was the obvious dial and it does not work: tripling the pillar
        # radii moved the crash rate only 0.0020 -> 0.0088, because the learned
        # controller simply handles bigger obstacles.  Nor does taking warning
        # away -- it still flies at 1 m of sensor range, 0.8 m of range noise and
        # 0.4 s of blindness.  DENSITY is the axis that bites: 6 -> 32 pillars
        # takes crash 0.0020 -> 0.0684 and reach 0.9941 -> 0.7939.
        #
        # Applied by parking the surplus rather than by resampling a smaller
        # field, so the state schema is identical at every difficulty and one
        # genome trains across all of them.
        self.difficulty = float(difficulty)
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
        if self.difficulty < 1.0:
            field = self._thin(field, B)
        self._env_keys = tuple(field)
        s.update(field)
        if self.start_pool is not None:
            # inside imported geometry the origin is very often a wall
            idx = torch.randint(self.start_pool.shape[0], (B,), generator=gen)
            s["p"] = self.start_pool[idx].clone()
            s["v"] = torch.zeros_like(s["v"])
        return s

    def _thin(self, field: State, B: int) -> State:
        """Park a trailing fraction of every group, keeping shapes fixed."""
        out = dict(field)
        for grp in self.env.groups:
            keys = [k for k in grp._keys() if k in out]
            if not keys:
                continue
            n = out[keys[0]].shape[1]
            keep = max(1, int(round(n * self.difficulty)))
            off = torch.zeros(B, n, dtype=torch.bool, device=self.device)
            off[:, keep:] = True
            out.update(grp.deactivate(out, off))
        return out

    def step(self, s: State, u: Tensor, dt: float, params: Tensor = None) -> State:
        out = super().step(s, u, dt, params)
        for k in s:                              # geometry is constant, carry it
            if k not in out:
                out[k] = s[k]
        return out

    def place_occluders(self, s: State, goals: Tensor) -> State:
        """Move `n_occluders` pillars into a ring around the target.

        The stock scene barely tests occlusion -- measured, only 10% of a 1.8 m
        viewing shell is blocked, so the vehicle almost always settles somewhere
        it can already see and repositioning is never asked for.  Placing an
        obstacle at `occ_lo`..`occ_hi` from the target puts a real shadow on the
        shell: a 0.3 m pillar 0.8 m from the target hides roughly a 44 deg arc of
        it, so a few of them make a clear vantage something the vehicle has to go
        and find.

        They go NEAR the target, never on it -- `clear_points` still runs first,
        so the target itself stays in free space and a clear view always exists.
        """
        if not self.n_occluders:
            return {}
        from ..environments.primitives import Boxes, Pillars
        out = {}
        g = goals[:, 0]
        start = s["p"]
        for grp in self.env.groups:
            if not isinstance(grp, (Pillars, Boxes)):
                continue
            c = s[grp._k("c")].clone()
            B, n = c.shape[0], c.shape[1]
            k = min(self.n_occluders, n)
            gen = torch.Generator(device="cpu").manual_seed(self.occ_seed)
            # The FIRST occluder goes on the line from where the vehicle starts
            # to the target, so the opening view is blocked every episode rather
            # than most of them.  Scattering them all in a ring leaves the start
            # clear a good fraction of the time, and then the task never asks for
            # the thing it exists to ask for -- moving to see.
            t = (self.occ_t_lo + (self.occ_t_hi - self.occ_t_lo)
                 * torch.rand(B, generator=gen, dtype=c.dtype)).to(c.device)
            # Keep it clear of BOTH ends.  `place_occluders` runs after
            # `clear_points`, so nothing else protects the target from it, and a
            # short start-to-target leg would otherwise put the occluder on top
            # of the very thing it is meant to hide -- measured, that drove the
            # goal's clearance to -0.18, i.e. buried, making those episodes
            # unsolvable rather than hard.
            leg = (g[:, :2] - start[:, :2])
            L = leg.norm(dim=-1).clamp_min(1e-6)
            t_lo, t_hi = self.occ_clear / L, 1.0 - self.occ_clear / L
            t = torch.minimum(torch.maximum(t, t_lo), t_hi)
            on_line = start[:, :2] + t[:, None] * leg
            # A leg shorter than twice the clearance has NO valid point on it --
            # anywhere far enough from the target is too close to the start.
            # Clamping into an empty interval silently collapses to its endpoint,
            # which is how the occluder ended up 0.013 m from the goal and buried
            # it.  Fall back to the ring for those episodes instead.
            room = (L > 2.0 * self.occ_clear)[:, None]
            ang0 = torch.rand(B, generator=gen, dtype=c.dtype) * 6.283185307
            ring0 = g[:, :2] + torch.stack(
                [torch.cos(ang0), torch.sin(ang0)], -1).to(c.device) * self.occ_lo
            c[:, 0] = torch.where(room, on_line, ring0)
            if k > 1:
                ang = torch.rand(B, k - 1, generator=gen,
                                 dtype=c.dtype) * 6.283185307
                rad = (self.occ_lo + (self.occ_hi - self.occ_lo)
                       * torch.rand(B, k - 1, generator=gen, dtype=c.dtype))
                rad = rad.clamp_min(self.occ_clear)
                off = torch.stack([torch.cos(ang) * rad,
                                   torch.sin(ang) * rad], -1)
                c[:, 1:k] = g[:, None, :2] + off.to(c.device)
            out = {kk: s[kk] for kk in grp._keys()}
            out[grp._k("c")] = c
            break
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
        out.update(self.place_occluders(out, goals))
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

    def visibility(self, s: State, goal: Tensor) -> Tensor:
        """Fraction of the target that is visible, in [0, 1] -- a PENUMBRA.

        The obvious implementation casts one ray at the target and smooths the
        result, but that smooths along the wrong axis.  A single ray's overshoot
        measures how far BEYOND the target the line reaches, so its gradient
        points down the line of sight -- while the direction that actually
        recovers a blocked view is LATERAL, out of the shadow.  A point target
        behind a point occluder has a hard umbra with no lateral gradient
        anywhere, so a search would have to stumble onto a clear vantage instead
        of being led to one.

        Giving the target extent fixes that.  Integrating visibility over a small
        disc facing the viewer -- a density, estimated here by `los_samples` rays
        over the disc -- turns the shadow edge into a penumbra: the visible
        fraction falls off smoothly across it, and its gradient points out of the
        shadow, which is the information the vehicle needs to reposition.

        `los_radius` is that extent.  It is a modelling choice, not a fudge: it
        says how big the thing being watched is, and the width of the penumbra
        follows from it and the geometry.
        """
        p = self.task_position(s)
        d = goal - p
        dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        n = d / dist
        # an orthonormal pair spanning the disc facing the viewer; the seed axis
        # is chosen away from n so the cross product never degenerates
        up = torch.zeros_like(n)
        up[..., 2] = 1.0
        alt = torch.zeros_like(n)
        alt[..., 0] = 1.0
        up = torch.where((n[..., 2:3].abs() > 0.9), alt, up)
        e1 = torch.cross(n, up, dim=-1)
        e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        e2 = torch.cross(n, e1, dim=-1)
        k = self.los_samples
        th = torch.arange(k, dtype=p.dtype, device=p.device) * (6.283185307 / k)
        # a ring plus the centre: cheap, and the ring is what resolves the edge
        offs = (torch.cos(th)[:, None] * e1[..., None, :]
                + torch.sin(th)[:, None] * e2[..., None, :]) * self.los_radius
        pts = goal[..., None, :] + offs                     # [..., k, 3]
        pts = torch.cat([goal[..., None, :], pts], dim=-2)  # [..., k+1, 3]
        dv = pts - p[..., None, :]
        dn = dv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rng, _ = self.env.raycast(s["p"], dv / dn, s, self.los_range)
        clear = torch.sigmoid((rng - dn[..., 0]) / self.los_soft)
        return clear.mean(-1)

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
