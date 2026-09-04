"""Import CAD drawings as simulation environments.

A 2-D DXF floor plan is almost exactly the geometry this framework already
models: the obstacle primitives are vertical extrusions, so a plan's walls and
columns become `StaticWalls` and `StaticPillars` with no approximation beyond
extruding them upward.

    env = dxf_to_environment("floor.dxf", fit=6.0)
    sys = make_system("quadrotor_nav", env=env)

Mapping:

| DXF entity | becomes |
|---|---|
| `LINE` | one wall segment |
| `LWPOLYLINE`, `POLYLINE` | a wall segment per span (closed loops included) |
| `CIRCLE` | a pillar |
| `ARC` | segments along the arc |
| `INSERT` | recursed into, so blocks are not silently dropped |

`ezdxf` is used when installed and a minimal ASCII reader covers the entity types
above otherwise, so importing a plan never becomes a hard dependency.

**Imported geometry never moves.**  `clear_points` is a no-op on the static
groups: a random obstacle field may be nudged aside to keep a waypoint reachable,
but a building may not, or the simulation stops being a simulation of that
building.  Waypoints are instead sampled from free space -- see
`Environment.free_points` and the `free_space` task.
"""
from __future__ import annotations

import math
import pathlib
from typing import List, Optional, Tuple

import torch

from .base import EPS, Environment
from .primitives import Pillars, Walls

ARC_SEGMENTS = 16


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _arc_points(cx, cy, r, a0, a1, n=ARC_SEGMENTS):
    if a1 < a0:
        a1 += 360.0
    return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in (a0 + (a1 - a0) * i / n for i in range(n + 1))]


def _chain(points, closed=False):
    segs = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    if closed and len(points) > 2:
        segs.append((points[-1], points[0]))
    return segs


def _read_ezdxf(path):
    import ezdxf

    doc = ezdxf.readfile(str(path))
    segs: List = []
    circles: List = []

    def walk(entities, depth=0):
        for e in entities:
            t = e.dxftype()
            if t == "LINE":
                segs.append(((e.dxf.start.x, e.dxf.start.y),
                             (e.dxf.end.x, e.dxf.end.y)))
            elif t == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points()]
                segs.extend(_chain(pts, e.closed))
            elif t == "POLYLINE":
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                segs.extend(_chain(pts, e.is_closed))
            elif t == "CIRCLE":
                circles.append((e.dxf.center.x, e.dxf.center.y, e.dxf.radius))
            elif t == "ARC":
                segs.extend(_chain(_arc_points(e.dxf.center.x, e.dxf.center.y,
                                               e.dxf.radius, e.dxf.start_angle,
                                               e.dxf.end_angle)))
            elif t == "INSERT" and depth < 4:
                # blocks are extremely common in real drawings; dropping them
                # silently imports an empty building
                walk(e.virtual_entities(), depth + 1)

    walk(doc.modelspace())
    return segs, circles


def _read_ascii(path):
    """Minimal ASCII DXF reader: group code / value pairs, entities section only.

    Covers LINE, LWPOLYLINE, POLYLINE, CIRCLE and ARC -- enough for a floor plan
    exported flat.  Blocks are not expanded here; install `ezdxf` for those.
    """
    raw = pathlib.Path(path).read_text(errors="ignore").splitlines()
    if len(raw) % 2:
        raw.append("")
    pairs = [(raw[i].strip(), raw[i + 1].strip()) for i in range(0, len(raw) - 1, 2)]

    segs: List = []
    circles: List = []
    cur: Optional[str] = None
    d: dict = {}
    poly: List = []

    def flush():
        nonlocal cur, d, poly
        try:
            if cur == "LINE":
                segs.append(((float(d["10"]), float(d["20"])),
                             (float(d["11"]), float(d["21"]))))
            elif cur == "CIRCLE":
                circles.append((float(d["10"]), float(d["20"]), float(d["40"])))
            elif cur == "ARC":
                segs.extend(_chain(_arc_points(
                    float(d["10"]), float(d["20"]), float(d["40"]),
                    float(d.get("50", 0.0)), float(d.get("51", 360.0)))))
            elif cur in ("LWPOLYLINE", "POLYLINE") and len(poly) > 1:
                segs.extend(_chain(poly, int(d.get("70", 0)) & 1 == 1))
        except (KeyError, ValueError):
            pass
        cur, d, poly = None, {}, []

    xs: List[float] = []
    for code, val in pairs:
        if code == "0":
            flush()
            if val in ("LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE", "VERTEX"):
                if val == "VERTEX":
                    cur, d = "VERTEX", {}
                else:
                    cur = val
        elif cur:
            if cur in ("LWPOLYLINE",) and code == "10":
                xs.append(float(val))
            elif cur in ("LWPOLYLINE",) and code == "20" and xs:
                poly.append((xs.pop(), float(val)))
            else:
                d[code] = val
    flush()
    return segs, circles


def read_dxf(path) -> Tuple[List, List]:
    """(segments, circles) in the drawing's own units."""
    try:
        return _read_ezdxf(path)
    except ImportError:
        return _read_ascii(path)


# --------------------------------------------------------------------------- #
# static groups
# --------------------------------------------------------------------------- #
class StaticWalls(Walls):
    """Wall segments loaded from a drawing.  Identical geometry every episode."""

    kind = "static_walls"

    def __init__(self, segments, key: str = "walls", thickness: float = 0.08):
        super().__init__(n=max(1, len(segments)), key=key, thickness=thickness)
        seg = torch.as_tensor(list(segments), dtype=torch.float64)
        if seg.numel() == 0:
            seg = torch.zeros(1, 2, 2, dtype=torch.float64) + 1e4
        self.seg = seg.reshape(-1, 2, 2)
        self.n = self.seg.shape[0]

    def sample(self, B, gen, dtype, device):
        a = self.seg[:, 0].to(dtype=dtype, device=device).expand(B, self.n, 2)
        b = self.seg[:, 1].to(dtype=dtype, device=device).expand(B, self.n, 2)
        return {self._k("a"): a.clone(), self._k("b"): b.clone()}

    def clear_points(self, f, pts, margin):
        """No-op: a building does not move to make a waypoint reachable."""
        return f


class StaticPillars(Pillars):
    """Columns loaded from a drawing."""

    kind = "static_pillars"

    def __init__(self, circles, key: str = "pillars"):
        super().__init__(n=max(1, len(circles)), key=key)
        c = torch.as_tensor(list(circles), dtype=torch.float64)
        if c.numel() == 0:
            c = torch.tensor([[1e4, 1e4, 0.1]], dtype=torch.float64)
        self.circ = c.reshape(-1, 3)
        self.n = self.circ.shape[0]

    def sample(self, B, gen, dtype, device):
        c = self.circ[:, :2].to(dtype=dtype, device=device).expand(B, self.n, 2)
        r = self.circ[:, 2].to(dtype=dtype, device=device).expand(B, self.n)
        return {self._k("c"): c.clone(), self._k("r"): r.clone()}

    def clear_points(self, f, pts, margin):
        return f


# --------------------------------------------------------------------------- #
def dxf_to_environment(path, fit: Optional[float] = 6.0, scale: Optional[float] = None,
                       centre: bool = True, thickness: float = 0.08,
                       min_length: float = 1e-3, name: Optional[str] = None
                       ) -> Environment:
    """Load a DXF drawing as a simulation environment.

    `fit` rescales so the drawing's largest dimension spans that many metres,
    which is what makes a plan drawn in millimetres usable without the caller
    having to know its units; pass `scale` instead for an explicit factor.
    """
    segs, circles = read_dxf(path)
    pts = [p for s in segs for p in s] + [(c[0], c[1]) for c in circles]
    if not pts:
        raise ValueError(f"{path}: no usable geometry "
                         "(LINE / LWPOLYLINE / POLYLINE / CIRCLE / ARC)")

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))) if centre else (0., 0.)
    span = max(max(xs) - min(xs), max(ys) - min(ys), EPS)
    k = float(scale) if scale is not None else (float(fit) / span if fit else 1.0)

    S = [(((a[0] - cx) * k, (a[1] - cy) * k), ((b[0] - cx) * k, (b[1] - cy) * k))
         for a, b in segs]
    S = [s for s in S if math.dist(s[0], s[1]) > min_length]
    C = [((c[0] - cx) * k, (c[1] - cy) * k, c[2] * k) for c in circles if c[2] * k > min_length]

    groups = []
    if S:
        groups.append(StaticWalls(S, thickness=thickness))
    if C:
        groups.append(StaticPillars(C))
    env = Environment(groups, name=name or pathlib.Path(path).stem)
    env.imported = {"segments": len(S), "circles": len(C), "scale": k,
                    "extent": span * k}
    return env
