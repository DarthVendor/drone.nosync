"""Obstacle fields: the abstraction.

The environment layer is deliberately separate from `systems/`: a plant is a set
of equations of motion, a scene is the world those equations move through, and
keeping them apart means a robot can be dropped into any scene and a scene can be
reused by any robot.  `environments/` does not import `systems/` at all.

An environment is a LIST of obstacle groups, exactly as a controller is a list of
Lagrangian terms and a robot is a list of connectors.  Each group knows how to

  * `sample` its own geometry, batched, one layout per episode;
  * report a signed distance to it;
  * be hit by a ray, analytically or by marching.

Geometry lives in the STATE, batched per episode.  That is what makes
generalization measurable rather than assumed: `reset` draws a fresh layout from
its generator, so a held-out evaluation seed is automatically a held-out set of
layouts, with no separate bookkeeping to get wrong.

Everything here is closed form or a fixed-step march, so it stays vmap- and
jacrev-safe and the search never differentiates through it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import torch
from torch import Tensor

EPS = 1e-6
State = Dict[str, Tensor]


PARK = 1.0e3          # where evicted or deactivated geometry goes


def _circle_violates(c: Tensor, r: Tensor, p2: Tensor, margin: float) -> Tensor:
    d = (c[..., :, None, :] - p2[..., None, :, :]).norm(dim=-1)
    return (d < r[..., :, None] + margin).any(dim=-1)


def _evict(x: Tensor, bad: Tensor, ref: Tensor = None) -> Tensor:
    """Move any primitive that still overlaps a waypoint far out of the arena.

    Pushing obstacles apart is iterative and cannot always succeed: a wall pinched
    between two waypoints on opposite sides has no translation that clears both.
    Rather than leave a goal buried -- a permanent, invisible ceiling on the
    success rate -- the offending primitive is removed from that episode.

    It fires on roughly 0.05% of episodes, so the effect on obstacle density is
    negligible, and it converts a silent measurement artefact into a slightly
    sparser scene, which is the trade worth making.
    """
    off = torch.zeros_like(x)
    off[..., 0] = PARK
    if ref is not None:
        return torch.where(bad[..., None], x - ref + off, x)
    return torch.where(bad[..., None], off, x)


def _pad3(g: Tensor) -> Tensor:
    """Planar gradient -> 3-D, with an exactly zero vertical channel.

    Zero rather than omitted: these primitives are vertical, so height truly does
    not change the range, and a pullback that acted on a fudged value would be
    acting on a lie.
    """
    z = torch.zeros(g.shape[:-1] + (1,), dtype=g.dtype, device=g.device)
    return torch.cat([g, z], dim=-1)


def march(group, origin: Tensor, dirs: Tensor, f: State, max_range: float,
          steps: int = 10, tol: float = 5e-3):
    """Sphere marching against a group's own signed distance function.

    This is what lets a new primitive be defined by `sdf` alone.  Groups with a
    closed-form ray intersection (pillars, walls) override `raycast` and are both
    exact and faster; anything with an awkward surface -- a torus, say -- gets
    correct ranges here for free.

    At a hit, dt/d(origin) = -n/(d.n) for the surface normal n.  A ray that misses
    reports `max_range` with zero gradient: the honest derivative of a clamped
    measurement, and the safe one, since a barrier built on it cannot become
    confident merely because a beam found nothing.
    """
    o = origin[..., None, :]
    # Start the march at the group's own bounding interval rather than at the ray
    # origin.  Under vmap there is no early exit, so every ray pays for every
    # step; concentrating those steps on the span that can actually contain a
    # surface is what makes a marched primitive affordable next to the closed-form
    # ones -- 10 steps across a ring's bounding sphere resolve better than 28
    # across the whole sensor range.
    t, t_max = group.ray_bounds(origin, dirs, f, max_range)
    for _ in range(steps):
        d = group.sdf(o + t[..., None] * dirs, f, extra=1)
        t = torch.minimum(t + d.clamp_min(1e-3), t_max)
    end = o + t[..., None] * dirs
    hit = group.sdf(end, f, extra=1) < tol
    n = group.normal(end, f, extra=1)
    dn = (dirs * n).sum(-1)
    grad = -n / torch.where(dn.abs() < EPS, torch.full_like(dn, EPS), dn)[..., None]
    t = torch.where(hit, t, torch.full_like(t, max_range))
    grad = torch.where(hit[..., None], grad, torch.zeros_like(grad))
    return t.clamp(0.0, max_range), grad

class ObstacleGroup(ABC):
    """One family of primitives, sampled and queried as a batch."""

    key: str = "group"
    kind: str = "group"

    @abstractmethod
    def sample(self, B: int, gen: torch.Generator, dtype, device) -> State:
        """Batched geometry, namespaced under `self.key`."""

    @abstractmethod
    def sdf(self, p: Tensor, f: State, extra: int = 0) -> Tensor:
        """Signed distance to this group, [...].  Positive outside.

        `extra` is the number of axes `p` carries beyond the batch (1 when the
        query is per-beam), so the group knows how far to unsqueeze its own
        geometry.  Making that explicit is what lets one implementation serve both
        a point query and a ray march.
        """

    def normal(self, p: Tensor, f: State, extra: int = 0) -> Tensor:
        """Outward unit normal, by central differences unless overridden."""
        h = 1e-4
        g = []
        for i in range(p.shape[-1]):
            d = torch.zeros_like(p)
            d[..., i] = h
            g.append((self.sdf(p + d, f, extra) - self.sdf(p - d, f, extra)) / (2 * h))
        n = torch.stack(g, dim=-1)
        return n / n.norm(dim=-1, keepdim=True).clamp_min(EPS)

    def ray_bounds(self, origin: Tensor, dirs: Tensor, f: State, max_range: float):
        """(t_enter, t_exit) bracketing where this group could possibly be hit.

        Default: the whole ray.  A group with compact geometry should narrow it,
        because the marcher spends its fixed step budget inside this interval.
        """
        shp = dirs.shape[:-1]
        return (torch.zeros(shp, dtype=origin.dtype, device=origin.device),
                torch.full(shp, max_range, dtype=origin.dtype, device=origin.device))

    def raycast(self, origin: Tensor, dirs: Tensor, f: State, max_range: float):
        """(range [..., m], d_range/d_origin [..., m, 3]).

        `origin` is [..., 3] and `dirs` [..., m, 3] for EVERY group, even the ones
        whose geometry is planar: a hoop's distance genuinely depends on height,
        and letting flat primitives set a 2-D convention is what makes a torus
        crash the moment it is added to the same scene.

        Default: sphere-march this group's own SDF, so a NEW PRIMITIVE ONLY HAS TO
        IMPLEMENT `sdf` -- which is the whole point of the abstraction.  Groups
        with a closed-form intersection (pillars, walls) override this and are
        both faster and exact.

        The gradient at a hit is the standard result dt/d(origin) = -n/(d.n) for
        the surface normal n; a ray that misses reports `max_range` with zero
        gradient, the honest derivative of a clamped measurement.
        """
        return march(self, origin, dirs, f, max_range)

    def deactivate(self, f: State, off: Tensor) -> State:
        """Park this group's geometry wherever `off` [...] is True.

        Deactivating rather than resizing keeps the state schema FIXED across
        episodes, which is what lets one controller train on a mixture of regimes:
        the tensors are always present and always the same shape, only some of
        them are somewhere irrelevant.
        """
        out = dict(f)
        for k in self._keys():
            v = f[k]
            m = off.reshape(off.shape + (1,) * (v.ndim - off.ndim))
            park = torch.zeros_like(v)
            park[..., 0] = PARK
            out[k] = torch.where(m, v + park, v)
        return out

    def _keys(self) -> tuple:
        """State keys this group owns that carry POSITION (not size/tilt)."""
        return (self._k("c"),)

    def clear_points(self, f: State, pts: Tensor, margin: float) -> State:
        """Move geometry so every point in `pts` has at least `margin` clearance.

        Obstacles move, waypoints do not.  Rejection-sampling the goals instead
        would quietly bias the task distribution toward open space -- exactly the
        easy cases -- and the bias would grow with obstacle density, so a "harder"
        scene would silently become a *different* task rather than a harder one.
        Moving obstacles leaves the goal distribution exactly as specified.

        Default: no-op, for groups that are meant to sit on the waypoints.
        """
        return f

    @abstractmethod
    def render(self, f: State) -> dict:
        """Geometry for the visualizer."""

    def _k(self, name: str) -> str:
        return f"{self.key}/{name}"

class Environment:
    """A composed scene: a list of obstacle groups."""

    def __init__(self, groups: Sequence[ObstacleGroup] = (), name: str = "custom"):
        self.groups: List[ObstacleGroup] = list(groups)
        self.name = name

    def __len__(self):
        return len(self.groups)

    @property
    def empty(self) -> bool:
        return not self.groups

    def sample(self, B, gen, dtype=torch.float64, device="cpu") -> State:
        out: State = {}
        for g in self.groups:
            out.update(g.sample(B, gen, dtype, device))
        return out

    def sdf(self, p: Tensor, f: State) -> Tensor:
        """Distance to the nearest obstacle of any group, [...]."""
        if not self.groups:
            return torch.full(p.shape[:-1], 1e3, dtype=p.dtype, device=p.device)
        d = None
        for g in self.groups:
            gi = g.sdf(p, f)
            d = gi if d is None else torch.minimum(d, gi)
        return d

    def raycast(self, origin: Tensor, dirs: Tensor, f: State, max_range: float = 4.0):
        """Nearest hit across all groups, with the matching gradient."""
        if not self.groups:
            shp = dirs.shape[:-1]
            return (torch.full(shp, max_range, dtype=origin.dtype, device=origin.device),
                    torch.zeros(shp + (3,), dtype=origin.dtype, device=origin.device))
        rng, grad = None, None
        for g in self.groups:
            ri, gi = g.raycast(origin, dirs, f, max_range)
            if rng is None:
                rng, grad = ri, gi
            else:
                take = (ri < rng)[..., None]
                grad = torch.where(take, gi, grad)
                rng = torch.minimum(rng, ri)
        return rng, grad

    def clear_points(self, f: State, pts: Tensor, margin: float = 0.25) -> State:
        """Guarantee clearance at every point, across all groups."""
        out = dict(f)
        for g in self.groups:
            out.update(g.clear_points(out, pts, margin))
        return out

    def free_points(self, n: int, gen, extent: float = 3.0,
                    margin: float = 0.35, z_range=(1.0, 2.0), tries: int = 40,
                    dtype=torch.float64, device="cpu") -> Tensor:
        """Rejection-sample `n` points with at least `margin` clearance, [n, 3].

        For IMPORTED geometry this replaces `clear_points`: a random obstacle
        field may be nudged aside to keep a waypoint reachable, but a building may
        not, or the simulation stops being a simulation of that building.  So the
        obstacles stay put and the waypoints move instead.

        Static geometry is identical across episodes, so one sampled field is
        enough to test against.
        """
        f1 = self.sample(1, gen, dtype, device)
        lo, hi = float(z_range[0]), float(z_range[1])
        keep = []
        have = 0
        for _ in range(tries):
            k = max(n * 4, 256)
            xy = (torch.rand(k, 2, generator=gen, dtype=dtype, device=device)
                  * 2.0 - 1.0) * extent
            z = lo + (hi - lo) * torch.rand(k, 1, generator=gen, dtype=dtype,
                                            device=device)
            cand = torch.cat([xy, z], dim=-1)
            ok = cand[self.sdf(cand, f1) >= margin]
            if ok.numel():
                keep.append(ok)
                have += ok.shape[0]
            if have >= n:
                break
        if not keep:
            raise ValueError(
                f"{self.name}: no free space at margin {margin} within "
                f"extent {extent}. The drawing may be scaled wrongly, or the "
                f"clearance may exceed the width of every corridor in it.")
        return torch.cat(keep, dim=0)[:n]

    def render(self, f: State) -> list:
        return [g.render(f) for g in self.groups]

    def describe(self) -> dict:
        return {"environment": self.name,
                "groups": [{"kind": g.kind, "key": g.key} for g in self.groups]}


class Mixture(Environment):
    """A scene that activates a random SUBSET of its groups each episode.

    This is what makes a compositional train/test split possible.  Train with
    exactly one group live and the controller meets each obstacle regime in
    isolation; test with two or three live and it faces combinations it has never
    seen -- while the state schema, and therefore the genome and the sensor, stay
    identical across both.

    Inactive groups are parked rather than removed, so nothing about the tensor
    shapes depends on which regime an episode drew.
    """

    def __init__(self, groups: Sequence[ObstacleGroup], n_active=(1, 1),
                 name: str = "mixture"):
        super().__init__(groups, name=name)
        self.lo, self.hi = int(n_active[0]), int(n_active[1])

    def sample(self, B, gen, dtype=torch.float64, device="cpu") -> State:
        f = super().sample(B, gen, dtype, device)
        n = len(self.groups)
        k = torch.randint(self.lo, self.hi + 1, (B,), generator=gen, device=device)
        # rank groups by a per-episode random key and keep the lowest k
        key = torch.rand(B, n, generator=gen, dtype=dtype, device=device)
        rank = key.argsort(dim=-1).argsort(dim=-1)
        active = rank < k[:, None]
        for i, g in enumerate(self.groups):
            f.update(g.deactivate(f, ~active[:, i]))
        f["mix/active"] = active.to(dtype)
        return f

    def describe(self) -> dict:
        d = super().describe()
        d.update(n_active=[self.lo, self.hi])
        return d
