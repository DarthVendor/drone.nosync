"""Concrete obstacle primitives.

Adding one is a single class: `ObstacleGroup.raycast` defaults to sphere marching
the group's own SDF, so a new shape only has to implement `sdf` (and optionally
`normal`).  `Pillars` and `Walls` override it with closed-form intersections and
are exact and faster; `Hoops` inherits the marcher.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .base import EPS, ObstacleGroup, State, _circle_violates, _evict, _pad3


class Pillars(ObstacleGroup):
    """Vertical cylinders.

    Pillars rather than spheres because they force the vehicle to go *around*
    rather than over, which is what makes a navigation task a navigation task
    instead of an altitude task.
    """

    kind = "pillars"

    def __init__(self, n: int = 6, key: str = "pillars", extent: float = 2.6,
                 r_lo: float = 0.18, r_hi: float = 0.38, keep_clear: float = 0.55,
                 cull_k: int = 0):
        self.n, self.key = int(n), key
        self.extent, self.r_lo, self.r_hi = float(extent), float(r_lo), float(r_hi)
        self.keep_clear = float(keep_clear)
        # CHUNKING: how many nearby pillars a ray march considers.
        #
        # Exact if and only if `cull_k` >= the number of pillars within the
        # sensor's reach, since anything beyond that provably cannot be hit.  The
        # reason it works is that the vehicle's neighbourhood does not grow with
        # the world: measured at a 6 m sensor, roughly 35-43 pillars are in reach
        # whether the field holds 40 or 1000 of them.  So a fixed `cull_k` makes
        # the per-step cost independent of world size -- 8.6x faster at 1000
        # pillars with ZERO range error.
        #
        # It buys nothing on a SMALL arena, where everything is in reach anyway
        # (1.1x at extent 5), and setting it below the occupancy is not slow-but-
        # correct, it is silently WRONG: k=8 against 80 pillars ran 4.6x faster
        # and lost hits by up to 5.16 m.  `chunk_occupancy` reports the number to
        # beat, and `test_environment` asserts it for the shipped presets.
        self.cull_k = int(cull_k)

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
        """Analytic ray/circle intersection, over NEARBY pillars only.

        For a = origin - centre, the near root is t = -f - sqrt(f^2 - g) with
        f = a.d and g = |a|^2 - r^2, so dt/d(origin) = -d - (f d - a)/sqrt(disc).
        A ray that misses reports `max_range` with zero gradient, which is the
        honest derivative of a clamped measurement.
        """
        c, r = f[self._k("c")], f[self._k("r")]
        # `Gate` borrows this method by class assignment and defines neither
        # `cull_k` nor `r_hi`, so read them defensively rather than assuming the
        # owner is a Pillars.  A two-pillar gate has nothing to cull anyway.
        cull_k = getattr(self, "cull_k", 0)
        if cull_k and c.shape[-2] > cull_k:
            # Only pillars the beams could actually reach.  All beams share an
            # origin, so this runs once per vehicle instead of once per ray.
            idx, _ = self.near_indices(origin, f, cull_k, self._k("c"),
                                       max_range + getattr(self, "r_hi", 0.0))
            c = c.gather(-2, idx[..., None].expand(idx.shape + (2,)))
            r = r.gather(-1, idx)
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

    def chunk_occupancy(self, origin, f, reach):
        c, r = f[self._k("c")], f[self._k("r")]
        d = (origin[..., None, :2] - c).norm(dim=-1) - r
        return (d < reach).to(origin.dtype).sum(-1)

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


class Boxes(ObstacleGroup):
    """Upright rectangular blocks, yaw-rotated, with finite height.

    A cylinder's shadow has no edges -- its silhouette is the same from every
    bearing, so the boundary between seeing a target and not seeing it is soft
    and featureless.  A box has FACES: the shadow it casts has a definite width
    that changes with viewing angle, and stepping past a corner recovers the view
    sharply.  For an occlusion task that is the difference between a gradient
    that merely exists and one that says something.

    Finite height also matters, unlike `Pillars` which extrude forever.  A block
    can be looked over as well as around, so the vantage problem is genuinely
    three-dimensional -- which is the case a horizontal beam fan cannot see and a
    forward camera can.

    Only `sdf` is implemented: the base class sphere-marches it, which is the
    whole point of that abstraction.  The exact box SDF is the standard one,
    q = |p| - h with distance ||max(q,0)|| + min(max(q), 0) -- correct both
    outside and inside, so the marcher never steps through a face.
    """

    kind = "boxes"

    def __init__(self, n: int = 4, key: str = "boxes", extent: float = 2.4,
                 half_lo: float = 0.16, half_hi: float = 0.42,
                 h_lo: float = 0.7, h_hi: float = 2.0,
                 keep_clear: float = 0.65):
        self.n, self.key = int(n), key
        self.extent = float(extent)
        self.half_lo, self.half_hi = float(half_lo), float(half_hi)
        self.h_lo, self.h_hi = float(h_lo), float(h_hi)
        self.keep_clear = float(keep_clear)

    def sample(self, B, gen, dtype, device):
        kw = dict(generator=gen, dtype=dtype, device=device)
        c = (torch.rand(B, self.n, 2, **kw) * 2.0 - 1.0) * self.extent
        d = c.norm(dim=-1, keepdim=True).clamp_min(EPS)
        c = c * (self.keep_clear / d).clamp_min(1.0)
        hx = self.half_lo + (self.half_hi - self.half_lo) * torch.rand(B, self.n, **kw)
        hy = self.half_lo + (self.half_hi - self.half_lo) * torch.rand(B, self.n, **kw)
        hz = self.h_lo + (self.h_hi - self.h_lo) * torch.rand(B, self.n, **kw)
        ang = torch.rand(B, self.n, **kw) * 3.141592653589793
        return {self._k("c"): c, self._k("h"): torch.stack([hx, hy, hz], -1),
                self._k("a"): ang}

    def sdf(self, p, f, extra: int = 0):
        c, h, a = f[self._k("c")], f[self._k("h")], f[self._k("a")]
        for _ in range(extra):
            c, h, a = c.unsqueeze(-3), h.unsqueeze(-3), a.unsqueeze(-2)
        d = p[..., None, :2] - c                       # into each box's frame
        ca, sa = torch.cos(a), torch.sin(a)
        lx = d[..., 0] * ca + d[..., 1] * sa
        ly = -d[..., 0] * sa + d[..., 1] * ca
        # boxes stand on the floor, so the vertical extent runs 0..h_z
        lz = p[..., None, 2] - 0.5 * h[..., 2]
        q = torch.stack([lx.abs() - h[..., 0], ly.abs() - h[..., 1],
                         lz.abs() - 0.5 * h[..., 2]], dim=-1)
        outside = q.clamp_min(0.0).norm(dim=-1)
        inside = q.max(dim=-1).values.clamp_max(0.0)
        return (outside + inside).min(dim=-1).values

    def ray_bounds(self, origin, dirs, f, max_range):
        """Bracket on the group's bounding sphere, so the marcher spends its
        step budget where the boxes actually are."""
        c, h = f[self._k("c")], f[self._k("h")]
        rad = h.norm(dim=-1).max(dim=-1).values + 1e-3          # [...]
        ctr = c.mean(dim=-2)
        span = (c - ctr[..., None, :]).norm(dim=-1).max(dim=-1).values + rad
        a = origin[..., :2] - ctr
        t_mid = -(a[..., None, :] * dirs[..., :2]).sum(-1)
        half = span[..., None]
        lo = (t_mid - half).clamp_min(0.0)
        hi = (t_mid + half).clamp(0.0, max_range)
        return lo, torch.maximum(hi, lo)

    def clear_points(self, f, pts, margin):
        """Push each box off any waypoint it covers, along the shortest escape.

        Uses the box's own bounding radius rather than a face-exact test: the
        margin only has to guarantee free space, and over-clearing by the corner
        distance is the safe direction to be wrong in.
        """
        c, h, a = f[self._k("c")].clone(), f[self._k("h")], f[self._k("a")]
        r = h[..., :2].norm(dim=-1)                    # bounding radius in plan
        p2 = pts[..., :2]
        for _ in range(6):
            delta = c[..., :, None, :] - p2[..., None, :, :]
            dist = delta.norm(dim=-1)
            need = r[..., :, None] + margin
            viol = need - dist
            worst, idx = viol.max(dim=-1)
            take = idx[..., None, None].expand(idx.shape + (1, 2))
            dsel = delta.gather(-2, take).squeeze(-2)
            nrm = dsel.norm(dim=-1, keepdim=True)
            fallback = torch.zeros_like(dsel)
            fallback[..., 0] = 1.0
            unit = torch.where(nrm > EPS, dsel / nrm.clamp_min(EPS), fallback)
            c = c + unit * worst.clamp_min(0.0)[..., None]
        c = _evict(c, _circle_violates(c, r, p2, margin))
        return {self._k("c"): c, self._k("h"): h, self._k("a"): a}

    def render(self, f):
        return {"kind": "boxes", "c": f[self._k("c")], "h": f[self._k("h")],
                "a": f[self._k("a")]}


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

    def ray_bounds(self, origin, dirs, f, max_range):
        """Analytic bounding-sphere interval around the rings.

        A torus has no cheap closed-form ray intersection, but its bounding
        sphere does, and that is enough to tell the marcher where to spend its
        steps.  Rays that miss every sphere get an empty interval and cost
        nothing beyond the test.
        """
        c, _, r = self._fields(f, 1)
        rad = (r + self.tube)[..., None, :] if r.dim() == 2 else r + self.tube
        a = origin[..., None, None, :] - c
        b = (a * dirs[..., :, None, :]).sum(-1)
        disc = b * b - ((a * a).sum(-1) - rad * rad)
        hit = disc > 0
        root = torch.sqrt(disc.clamp_min(EPS))
        t0 = torch.where(hit, (-b - root).clamp_min(0.0),
                         torch.full_like(b, max_range))
        t1 = torch.where(hit, (-b + root), torch.zeros_like(b))
        return (t0.min(dim=-1).values.clamp(0.0, max_range),
                t1.max(dim=-1).values.clamp(0.0, max_range))

    def render(self, f):
        return {"kind": "hoops", "c": f[self._k("c")], "a": f[self._k("a")],
                "r": f[self._k("r")], "tube": self.tube}
