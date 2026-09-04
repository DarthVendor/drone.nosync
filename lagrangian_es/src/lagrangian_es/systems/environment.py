"""Physical environments: composable obstacle fields.

An environment is a LIST of obstacle groups, exactly as a controller is a list of
Lagrangian terms and a robot is a list of connectors.  Each group knows how to

  * `sample` its own geometry, batched, one layout per episode;
  * report a signed distance to it;
  * be hit by a ray, analytically.

so composing a scene is a list literal and adding a new primitive is one class:

    env = Environment([Pillars(n=6), Walls(n=2)])
    env = make_environment("forest")          # or a named preset

Geometry lives in the STATE, batched per episode.  That is what makes
generalization measurable rather than assumed: `reset` draws a fresh layout from
its generator, so a held-out evaluation seed is automatically a held-out set of
layouts, with no separate bookkeeping to get wrong and no chance of silently
testing on the training scenes.

Everything here is closed form -- signed distances and ray intersections both --
so it stays vmap- and jacrev-safe with no sphere marching, and the search never
differentiates through it.
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
          steps: int = 28, tol: float = 5e-3):
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
    t = torch.zeros(dirs.shape[:-1], dtype=origin.dtype, device=origin.device)
    o = origin[..., None, :]
    for _ in range(steps):
        d = group.sdf(o + t[..., None] * dirs, f, extra=1)
        t = torch.minimum(t + d.clamp_min(1e-3), torch.full_like(t, max_range))
    end = o + t[..., None] * dirs
    hit = group.sdf(end, f, extra=1) < tol
    n = group.normal(end, f, extra=1)
    dn = (dirs * n).sum(-1)
    grad = -n / torch.where(dn.abs() < EPS, torch.full_like(dn, EPS), dn)[..., None]
    t = torch.where(hit, t, torch.full_like(t, max_range))
    grad = torch.where(hit[..., None], grad, torch.zeros_like(grad))
    return t.clamp(0.0, max_range), grad


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
class Pillars(ObstacleGroup):
    """Vertical cylinders.

    Pillars rather than spheres because they force the vehicle to go *around*
    rather than over, which is what makes a navigation task a navigation task
    instead of an altitude task.
    """

    kind = "pillars"

    def __init__(self, n: int = 6, key: str = "pillars", extent: float = 2.6,
                 r_lo: float = 0.18, r_hi: float = 0.38, keep_clear: float = 0.55):
        self.n, self.key = int(n), key
        self.extent, self.r_lo, self.r_hi = float(extent), float(r_lo), float(r_hi)
        self.keep_clear = float(keep_clear)

    def sample(self, B, gen, dtype, device):
        kw = dict(generator=gen, dtype=dtype, device=device)
        c = (torch.rand(B, self.n, 2, **kw) * 2.0 - 1.0) * self.extent
        # push obstacles off the origin: an episode that starts already inside a
        # pillar measures nothing at all
        d = c.norm(dim=-1, keepdim=True).clamp_min(EPS)
        c = c * (self.keep_clear / d).clamp_min(1.0)
        r = self.r_lo + (self.r_hi - self.r_lo) * torch.rand(B, self.n, **kw)
        return {self._k("c"): c, self._k("r"): r}

    def sdf(self, p, f, extra: int = 0):
        c, r = f[self._k("c")], f[self._k("r")]
        for _ in range(extra):                 # make room for the beam axis
            c, r = c.unsqueeze(-3), r.unsqueeze(-2)
        return ((p[..., None, :2] - c).norm(dim=-1) - r).min(dim=-1).values

    def raycast(self, origin, dirs, f, max_range):
        """Analytic ray/circle intersection.

        For a = origin - centre, the near root is t = -f - sqrt(f^2 - g) with
        f = a.d and g = |a|^2 - r^2, so dt/d(origin) = -d - (f d - a)/sqrt(disc).
        A ray that misses reports `max_range` with zero gradient, which is the
        honest derivative of a clamped measurement.
        """
        c, r = f[self._k("c")], f[self._k("r")]
        o2, d2 = origin[..., :2], dirs[..., :2]          # cylinders ignore height
        a = o2[..., None, None, :] - c[..., None, :, :]
        dd = d2[..., :, None, :]
        fa = (a * dd).sum(-1)
        g = (a * a).sum(-1) - r[..., None, :] ** 2
        disc = fa * fa - g
        root = torch.sqrt(disc.clamp_min(EPS))
        t = -fa - root
        hit = (disc > 0) & (t > 0)
        t = torch.where(hit, t, torch.full_like(t, max_range))
        rng, idx = t.min(dim=-1)

        sel = idx[..., None]
        f_s = fa.gather(-1, sel).squeeze(-1)
        root_s = root.gather(-1, sel).squeeze(-1)
        # `a` carries a singleton beam axis (it does not depend on the beam), so
        # it has to be expanded before gathering the selected obstacle per beam
        a_e = a.expand(fa.shape + (2,))
        a_s = a_e.gather(-2, sel[..., None].expand(sel.shape + (2,))).squeeze(-2)
        hit_s = hit.gather(-1, sel).squeeze(-1)
        grad = -d2 - (f_s[..., None] * d2 - a_s) / root_s[..., None].clamp_min(EPS)
        grad = torch.where(hit_s[..., None], grad, torch.zeros_like(grad))
        return rng.clamp(0.0, max_range), _pad3(grad)

    def clear_points(self, f, pts, margin):
        """Push each pillar radially off any waypoint it covers.

        Iterated a few times because a pillar pushed clear of one waypoint can
        land on another; three passes settles every layout tested, and the final
        state is asserted rather than assumed.
        """
        c, r = f[self._k("c")].clone(), f[self._k("r")]
        p2 = pts[..., :2]
        for _ in range(6):
            delta = c[..., :, None, :] - p2[..., None, :, :]      # [..., n, k, 2]
            dist = delta.norm(dim=-1)                             # [..., n, k]
            need = r[..., :, None] + margin
            viol = (need - dist)
            worst, idx = viol.max(dim=-1)
            take = idx[..., None, None].expand(idx.shape + (1, 2))
            dsel = delta.gather(-2, take).squeeze(-2)             # [..., n, 2]
            nrm = dsel.norm(dim=-1, keepdim=True)
            # a pillar exactly on a waypoint has no direction to flee; pick one
            fallback = torch.zeros_like(dsel)
            fallback[..., 0] = 1.0
            unit = torch.where(nrm > EPS, dsel / nrm.clamp_min(EPS), fallback)
            c = c + unit * worst.clamp_min(0.0)[..., None]
        c = _evict(c, _circle_violates(c, r, p2, margin))
        return {self._k("c"): c, self._k("r"): r}

    def render(self, f):
        return {"kind": "pillars",
                "c": f[self._k("c")], "r": f[self._k("r")]}


class Walls(ObstacleGroup):
    """Finite vertical wall segments, with thickness."""

    kind = "walls"

    def __init__(self, n: int = 2, key: str = "walls", extent: float = 2.4,
                 length: float = 1.6, thickness: float = 0.12,
                 keep_clear: float = 0.7):
        self.n, self.key = int(n), key
        self.extent, self.length = float(extent), float(length)
        self.thickness, self.keep_clear = float(thickness), float(keep_clear)

    def sample(self, B, gen, dtype, device):
        kw = dict(generator=gen, dtype=dtype, device=device)
        mid = (torch.rand(B, self.n, 2, **kw) * 2.0 - 1.0) * self.extent
        ang = torch.rand(B, self.n, **kw) * 3.141592653589793
        half = 0.5 * self.length * torch.stack(
            [torch.cos(ang), torch.sin(ang)], dim=-1)
        a, b = mid - half, mid + half
        # Clear the origin by the SEGMENT's closest approach, not its midpoint:
        # a long wall whose centre is pushed aside can still lie across the start,
        # and an episode that begins in collision measures nothing.
        ab = b - a
        tt = ((-a * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(EPS)).clamp(0.0, 1.0)
        near = a + tt[..., None] * ab
        dist = near.norm(dim=-1, keepdim=True).clamp_min(EPS)
        need = (self.keep_clear + self.thickness)
        shift = (near / dist) * (need - dist).clamp_min(0.0)
        return {self._k("a"): a + shift, self._k("b"): b + shift}

    def _closest(self, p, f, extra: int = 0):
        a, b = f[self._k("a")], f[self._k("b")]
        for _ in range(extra):
            a, b = a.unsqueeze(-3), b.unsqueeze(-3)
        ab = b - a
        ap = p[..., None, :2] - a
        tt = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(EPS)).clamp(0.0, 1.0)
        return a + tt[..., None] * ab

    def sdf(self, p, f, extra: int = 0):
        q = self._closest(p, f, extra)
        return ((p[..., None, :2] - q).norm(dim=-1) - self.thickness).min(dim=-1).values

    def raycast(self, origin, dirs, f, max_range):
        """Ray/segment intersection in the plane, then a thickness offset.

        The gradient is the exact derivative of the line-line solution; parallel
        or behind-the-ray cases fall through to `max_range` with zero gradient.
        """
        a, b = f[self._k("a")], f[self._k("b")]
        A = a[..., None, :, :]
        AB = (b - a)[..., None, :, :]
        D = dirs[..., :2][..., :, None, :]               # walls ignore height
        O = origin[..., :2][..., None, None, :]
        # O + t D = A + u AB   ->   cross products give t and u
        den = D[..., 0] * AB[..., 1] - D[..., 1] * AB[..., 0]   # [..., m, n]
        # A - O carries a singleton beam axis; expand to the full [beam, seg] grid
        # so the per-beam gather below has something to index
        oa = (A - O).expand(den.shape + (2,))
        safe_den = torch.where(den.abs() < EPS, torch.full_like(den, EPS), den)
        t = (oa[..., 0] * AB[..., 1] - oa[..., 1] * AB[..., 0]) / safe_den
        u = (oa[..., 0] * D[..., 1] - oa[..., 1] * D[..., 0]) / safe_den
        hit = (den.abs() > EPS) & (t > 0) & (u >= 0.0) & (u <= 1.0)
        t = torch.where(hit, (t - self.thickness).clamp_min(0.0),
                        torch.full_like(t, max_range))
        rng, idx = t.min(dim=-1)
        # d t / d origin = -(n) / (D . n) with n the segment normal
        nx, ny = AB[..., 1], -AB[..., 0]
        nn = torch.sqrt((nx * nx + ny * ny).clamp_min(EPS))
        nrm = torch.stack([nx / nn, ny / nn], dim=-1).expand(den.shape + (2,))
        dn = (D * nrm).sum(-1)
        gradf = -nrm / torch.where(dn.abs() < EPS,
                                   torch.full_like(dn, EPS), dn)[..., None]
        sel = idx[..., None]
        grad = gradf.gather(-2, sel[..., None].expand(sel.shape + (2,))).squeeze(-2)
        hit_s = hit.gather(-1, sel).squeeze(-1)
        grad = torch.where(hit_s[..., None], grad, torch.zeros_like(grad))
        return rng.clamp(0.0, max_range), _pad3(grad)

    def _keys(self):
        return (self._k("a"), self._k("b"))

    def clear_points(self, f, pts, margin):
        """Translate each wall off any waypoint it covers, keeping its heading."""
        a, b = f[self._k("a")].clone(), f[self._k("b")].clone()
        p2 = pts[..., :2]
        need = self.thickness + margin
        for _ in range(6):
            ab = (b - a)[..., :, None, :]                         # [..., n, 1, 2]
            ap = p2[..., None, :, :] - a[..., :, None, :]         # [..., n, k, 2]
            tt = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(EPS)).clamp(0., 1.)
            q = a[..., :, None, :] + tt[..., None] * ab           # closest points
            delta = q - p2[..., None, :, :]
            dist = delta.norm(dim=-1)
            worst, idx = (need - dist).max(dim=-1)
            take = idx[..., None, None].expand(idx.shape + (1, 2))
            dsel = delta.gather(-2, take).squeeze(-2)
            nrm = dsel.norm(dim=-1, keepdim=True)
            fallback = torch.zeros_like(dsel)
            fallback[..., 1] = 1.0
            unit = torch.where(nrm > EPS, dsel / nrm.clamp_min(EPS), fallback)
            shift = unit * worst.clamp_min(0.0)[..., None]
            a, b = a + shift, b + shift
        # final guarantee (see `_evict`)
        ab = (b - a)[..., :, None, :]
        ap = p2[..., None, :, :] - a[..., :, None, :]
        tt = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(EPS)).clamp(0., 1.)
        q = a[..., :, None, :] + tt[..., None] * ab
        bad = ((q - p2[..., None, :, :]).norm(dim=-1) < need).any(dim=-1)
        mid = 0.5 * (a + b)
        a = _evict(a, bad, ref=mid)
        b = _evict(b, bad, ref=mid)
        return {self._k("a"): a, self._k("b"): b}

    def render(self, f):
        return {"kind": "walls", "a": f[self._k("a")], "b": f[self._k("b")],
                "thickness": self.thickness}


class Gate(ObstacleGroup):
    """Two pillars forming a gap the vehicle has to fly through.

    Unlike a random field, a gate cannot be solved by drifting around the edge of
    the scene -- it is the cheapest layout that actually requires planning a route
    rather than a heading.
    """

    kind = "gate"

    def __init__(self, key: str = "gate", gap_lo: float = 0.7, gap_hi: float = 1.2,
                 radius: float = 0.28, dist_lo: float = 0.9, dist_hi: float = 1.8):
        self.key = key
        self.gap_lo, self.gap_hi = float(gap_lo), float(gap_hi)
        self.radius = float(radius)
        self.dist_lo, self.dist_hi = float(dist_lo), float(dist_hi)

    def sample(self, B, gen, dtype, device):
        kw = dict(generator=gen, dtype=dtype, device=device)
        ang = torch.rand(B, **kw) * 2.0 * 3.141592653589793
        dist = self.dist_lo + (self.dist_hi - self.dist_lo) * torch.rand(B, **kw)
        gap = self.gap_lo + (self.gap_hi - self.gap_lo) * torch.rand(B, **kw)
        mid = torch.stack([dist * torch.cos(ang), dist * torch.sin(ang)], dim=-1)
        perp = torch.stack([-torch.sin(ang), torch.cos(ang)], dim=-1)
        half = (0.5 * gap + self.radius)[..., None] * perp
        c = torch.stack([mid - half, mid + half], dim=-2)          # [B, 2, 2]
        r = torch.full((B, 2), self.radius, dtype=dtype, device=device)
        return {self._k("c"): c, self._k("r"): r}

    # geometry is circles, so reuse the pillar maths -- INCLUDING clearance.
    # Borrowing methods by assignment is easy to under-do: `Gate` originally took
    # `sdf` and `raycast` and silently kept the base class's no-op `clear_points`,
    # so gate posts alone still swallowed waypoints.
    sdf = Pillars.sdf
    raycast = Pillars.raycast
    clear_points = Pillars.clear_points

    def render(self, f):
        return {"kind": "pillars", "c": f[self._k("c")], "r": f[self._k("r")]}


class Hoops(ObstacleGroup):
    """Rings to fly THROUGH -- a racing course.

    Only the ring material is solid; the opening is free, so a hoop is the one
    obstacle that cannot be satisfied by simply going around.  The signed distance
    is the standard torus form: with q = p - centre, z = q.axis and the radial
    component u = q - z*axis,

        d = sqrt((|u| - R)^2 + z^2) - tube

    Note this group implements only `sdf` and `normal` and inherits ray casting
    from the base class, which is exactly the property the abstraction is for.
    """

    kind = "hoops"

    def __init__(self, n: int = 3, key: str = "hoops", radius: float = 0.55,
                 tube: float = 0.07, jitter: float = 0.35, z_lo: float = 0.9,
                 z_hi: float = 2.0, extent: float = 2.0,
                 tilt: float = 1.5707963267948966, tilt_lo: float = 0.0):
        """`tilt` is the maximum angle of the hoop axis away from horizontal.

        0 keeps every hoop vertical (axis horizontal -- you fly through it
        sideways); pi/2 allows a fully horizontal hoop (axis vertical -- you fly
        up through it); the default samples the whole range, so a course mixes
        vertical, tilted and horizontal gates and the controller cannot get by
        with a single approach direction.
        """
        self.n, self.key = int(n), key
        self.radius, self.tube = float(radius), float(tube)
        self.jitter = float(jitter)
        self.z_lo, self.z_hi, self.extent = float(z_lo), float(z_hi), float(extent)
        self.tilt, self.tilt_lo = float(tilt), float(tilt_lo)

    def sample(self, B, gen, dtype, device):
        """A ring of hoops around the origin, each facing the centre.

        Replaced by `place_course` when the plant lays the course out along the
        episode's actual waypoints.
        """
        kw = dict(generator=gen, dtype=dtype, device=device)
        ang = torch.rand(B, self.n, **kw) * 2.0 * 3.141592653589793
        rad = self.extent * (0.6 + 0.4 * torch.rand(B, self.n, **kw))
        c = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang),
                         self.z_lo + (self.z_hi - self.z_lo)
                         * torch.rand(B, self.n, **kw)], dim=-1)
        r = torch.full((B, self.n), self.radius, dtype=dtype, device=device)
        tl = self._tilts(ang.shape, gen, dtype, device)
        return {self._k("c"): c, self._k("a"): self._axes(ang, tl),
                self._k("r"): r, self._k("tilt"): tl}

    def _tilts(self, shape, gen, dtype, device):
        kw = dict(generator=gen, dtype=dtype, device=device)
        tl = self.tilt_lo + (self.tilt - self.tilt_lo) * torch.rand(shape, **kw)
        sgn = torch.where(torch.rand(shape, **kw) < 0.5,
                          -torch.ones_like(tl), torch.ones_like(tl))
        return tl * sgn

    def axes_about(self, heading: Tensor, tilt: Tensor) -> Tensor:
        """Tilt a heading out of the horizontal by `tilt`, keeping it unit.

        Used when a course is laid along the waypoints: the gate faces down the
        route, then tips by the angle this group already sampled.
        """
        h = heading.clone()
        h[..., 2] = 0.0
        h = h / h.norm(dim=-1, keepdim=True).clamp_min(EPS)
        up = torch.zeros_like(h)
        up[..., 2] = 1.0
        return torch.cos(tilt)[..., None] * h + torch.sin(tilt)[..., None] * up

    def _axes(self, ang, tl):
        """Hoop axes: an in-plane heading tilted up by the sampled angle.

        Built from an orthonormal pair rather than by perturbing a vector, so the
        result is a unit axis at exactly the sampled tilt -- the torus SDF assumes
        |axis| = 1 and would otherwise scale the whole ring.
        """
        horiz = torch.stack([-torch.sin(ang), torch.cos(ang),
                             torch.zeros_like(ang)], dim=-1)
        up = torch.zeros_like(horiz)
        up[..., 2] = 1.0
        return torch.cos(tl)[..., None] * horiz + torch.sin(tl)[..., None] * up

    def _fields(self, f, extra):
        c, a, r = f[self._k("c")], f[self._k("a")], f[self._k("r")]
        for _ in range(extra):
            c, a, r = c.unsqueeze(-3), a.unsqueeze(-3), r.unsqueeze(-2)
        return c, a, r

    def sdf(self, p, f, extra: int = 0):
        c, a, r = self._fields(f, extra)
        q = p[..., None, :] - c
        z = (q * a).sum(-1)
        u = q - z[..., None] * a
        radial = u.norm(dim=-1)
        d = torch.sqrt(((radial - r) ** 2 + z * z).clamp_min(EPS)) - self.tube
        return d.min(dim=-1).values

    def normal(self, p, f, extra: int = 0):
        """Analytic: the nearest point on the ring circle is c + R*uhat."""
        c, a, r = self._fields(f, extra)
        q = p[..., None, :] - c
        z = (q * a).sum(-1)
        u = q - z[..., None] * a
        uh = u / u.norm(dim=-1, keepdim=True).clamp_min(EPS)
        w = q - r[..., None] * uh
        d = torch.sqrt((w * w).sum(-1).clamp_min(EPS)) - self.tube
        idx = d.argmin(dim=-1)[..., None, None].expand(d.shape[:-1] + (1, 3))
        wn = w.gather(-2, idx).squeeze(-2)
        return wn / wn.norm(dim=-1, keepdim=True).clamp_min(EPS)

    def place(self, f: State, centres: Tensor, headings: Tensor) -> State:
        """Move the hoops onto a route: centre k at waypoint k, facing the leg."""
        n = min(self.n, centres.shape[-2])
        c = f[self._k("c")].clone()
        a = f[self._k("a")].clone()
        tl = f[self._k("tilt")]
        c[..., :n, :] = centres[..., :n, :]
        a[..., :n, :] = self.axes_about(headings[..., :n, :], tl[..., :n])
        return {self._k("c"): c, self._k("a"): a,
                self._k("r"): f[self._k("r")], self._k("tilt"): tl}

    def clear_points(self, f, pts, margin):
        """Push the ring off any waypoint whose clearance is too small.

        Note this is a real clearance, not a no-op.  When hoops ARE the gates the
        plant calls `place` AFTER clearing, so placement wins and the ring lands
        back on its waypoint; when hoops are merely obstacles in a mixture, they
        get cleared like anything else.  Skipping this is what let a ring swallow
        a waypoint in ~0.15% of mixture episodes.

        A waypoint inside the OPENING already has positive signed distance, so it
        is left alone -- flying through the hole is not a collision.
        """
        c = f[self._k("c")].clone()
        for _ in range(6):
            g = {self._k("c"): c, self._k("a"): f[self._k("a")],
                 self._k("r"): f[self._k("r")], self._k("tilt"): f[self._k("tilt")]}
            # pts carries a waypoint axis beyond the batch, so extra=1
            d = self.sdf(pts, g, extra=1)                       # [..., k]
            worst, widx = (margin - d).max(dim=-1)
            pk = pts.gather(-2, widx[..., None, None].expand(
                widx.shape + (1, pts.shape[-1]))).squeeze(-2)   # [..., 3]
            n = self.normal(pk, g, extra=0)                     # outward at p
            shift = -n * worst.clamp_min(0.0)[..., None]
            c = c + shift[..., None, :]
        return {self._k("c"): c, self._k("a"): f[self._k("a")],
                self._k("r"): f[self._k("r")], self._k("tilt"): f[self._k("tilt")]}

    def render(self, f):
        return {"kind": "hoops", "c": f[self._k("c")], "a": f[self._k("a")],
                "r": f[self._k("r")], "tube": self.tube}


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
GROUPS = {"pillars": Pillars, "walls": Walls, "gate": Gate, "hoops": Hoops}

#: named scenes -- each is just a list of groups, so a new one is one line
PRESETS = {
    "empty":    lambda: [],
    "sparse":   lambda: [Pillars(n=3, r_lo=0.20, r_hi=0.34)],
    "pillars":  lambda: [Pillars(n=6)],
    "forest":   lambda: [Pillars(n=11, r_lo=0.14, r_hi=0.28, extent=2.8)],
    "slalom":   lambda: [Pillars(n=4, extent=1.8, r_lo=0.26, r_hi=0.40)],
    "gate":     lambda: [Gate()],
    "walls":    lambda: [Walls(n=2)],
    "cluttered": lambda: [Pillars(n=5), Walls(n=2, length=1.3)],
    "gate_forest": lambda: [Gate(), Pillars(n=5, r_lo=0.14, r_hi=0.24)],
    "hoops":         lambda: [Hoops(n=2)],
    "hoop_course":   lambda: [Hoops(n=3, radius=0.55)],
    "hoops_upright": lambda: [Hoops(n=3, radius=0.55, tilt=0.0)],
    "hoops_flat":    lambda: [Hoops(n=3, radius=0.6, tilt=1.5707963267948966,
                                    tilt_lo=1.2)],
    "hoop_slalom":   lambda: [Hoops(n=3, radius=0.45),
                              Pillars(n=3, r_lo=0.16, r_hi=0.26)],
}


#: the regimes a controller trains on, one at a time
REGIMES = lambda: [Pillars(n=6), Walls(n=2), Gate(), Hoops(n=2, radius=0.55)]

#: compositional splits: identical group list, different number live per episode
MIXTURES = {
    "train_mix": (1, 1),      # one regime per episode -- seen in isolation
    "pair_mix":  (2, 2),      # two at once -- never seen combined
    "test_mix":  (2, 3),      # two or three -- the held-out compositions
}


def make_environment(name: str = "empty", **kw) -> Environment:
    """Build a named scene.  `make_environment("forest")`."""
    if name in MIXTURES:
        lo, hi = MIXTURES[name]
        return Mixture(REGIMES(), n_active=(lo, hi), name=name)
    if name not in PRESETS:
        raise KeyError(f"unknown environment {name!r}; "
                       f"presets: {sorted(PRESETS)}; mixtures: {sorted(MIXTURES)}")
    return Environment(PRESETS[name](), name=name)


def make_group(kind: str, **kw) -> ObstacleGroup:
    if kind not in GROUPS:
        raise KeyError(f"unknown obstacle group {kind!r}; registered: {sorted(GROUPS)}")
    return GROUPS[kind](**kw)
