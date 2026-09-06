"""Turn an OSM-derived DXF into a flyable city: blocks to fly around, streets
to fly down, and waypoints on the road network.

The map has no buildings -- OSM's DXF export here carries roads, water, railways
and coastline, nothing else.  So the built form is recovered as the COMPLEMENT
of the street network: rasterise the roads at their real widths, and what is
left inside the land is the city blocks.  That is not a guess about where
buildings are, it is the definition of a block, and it means the canyons the
drone flies down are Singapore's actual streets rather than invented ones.

Two scales are in play and they do not match.  The map is real metres; the
simulated quadrotor lives in a +/-2 m world and covers about 5 m in a 24 s
episode.  Flying true-scale Singapore would mean 100 s to cross one junction, so
the geometry is divided by `--scale` on the way out.  The default is chosen so a
street corridor comes out a little wider than the drone's 6 m sensor range,
which is the property that actually decides whether the task is navigable: it
has to be able to see both walls of the canyon it is in.

Usage:
  python scripts/dxf_city.py maps/singapore/singapore.dxf out.json \
      --cx 81000 --cy 54000 --half 600 --scale 12
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np

#: metres of half-width to stamp for each road class, per side.  OSM's three
#: road layers are roughly motorway/trunk, secondary, residential-and-service.
ROAD_HALF = {"roads_1": 12.0, "roads_2": 8.0, "roads_3": 5.0}


def read_dxf(path, layers):
    """Stream LWPOLYLINEs out of a DXF.  Group codes 8 (layer), 10/20 (vertex).

    No ezdxf dependency: this subset of the format is a flat sequence of
    (code, value) line pairs, and entities run back to back, so a `0` marker is
    both the next entity's start and the previous one's terminator.
    """
    out, cur, in_ent, sec = defaultdict(list), None, False, None
    with open(path, "r", errors="replace") as fh:
        it = iter(fh)
        for raw in it:
            code = raw.strip()
            try:
                val = next(it).strip()
            except StopIteration:
                break
            if code == "2" and val in ("HEADER", "CLASSES", "TABLES", "BLOCKS",
                                       "ENTITIES", "OBJECTS"):
                sec = val
            if code == "0":
                if in_ent and cur and len(cur[1]) > 1 and cur[0] in layers:
                    out[cur[0]].append(np.asarray(cur[1], dtype=np.float64))
                if val == "LWPOLYLINE" and sec == "ENTITIES":
                    cur, in_ent = ["", [], 0], True
                else:
                    cur, in_ent = None, False
            elif in_ent and cur is not None:
                if code == "8":
                    cur[0] = val
                elif code == "10":
                    cur[1].append([float(val), 0.0])
                elif code == "20" and cur[1]:
                    cur[1][-1][1] = float(val)
    return out


def stamp_segments(mask, polys, half, cell, x0, y0):
    """Mark every cell within `half` metres of any segment of any polyline.

    Walks each segment at sub-cell steps and paints a square of the right radius
    -- a dilation done at stamp time, which avoids needing a distance transform
    (and so avoids scipy) at the cost of being a square rather than a disc.  For
    a corridor that is metres wide the corner overshoot is under a cell.
    """
    H, W = mask.shape
    r = max(int(round(half / cell)), 1)
    for a in polys:
        d = np.linalg.norm(np.diff(a, axis=0), axis=1)
        n = np.maximum((d / (0.5 * cell)).astype(int) + 1, 1)
        for k in range(len(d)):
            t = np.linspace(0.0, 1.0, n[k] + 1)[:, None]
            pts = a[k] * (1 - t) + a[k + 1] * t
            ix = ((pts[:, 0] - x0) / cell).astype(int)
            iy = ((pts[:, 1] - y0) / cell).astype(int)
            keep = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            for cx, cy in zip(ix[keep], iy[keep]):
                mask[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1] = True


def largest_rectangle(mask):
    """Largest all-True axis-aligned rectangle: histogram method, row by row.

    Returns (area, y0, x0, h, w).  O(H*W) per call, which is what makes the
    greedy decomposition below affordable.
    """
    H, W = mask.shape
    heights = np.zeros(W, dtype=np.int64)
    best = (0, 0, 0, 0, 0)
    for y in range(H):
        row = mask[y]
        heights = np.where(row, heights + 1, 0)
        stack = []                       # (start_index, height), increasing
        for x in range(W + 1):
            h = heights[x] if x < W else 0
            start = x
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (x - s)
                if area > best[0]:
                    best = (area, y - sh + 1, s, sh, x - s)
                start = s
            stack.append((start, h))
    return best


def decompose(mask, min_cells):
    """Greedy cover of a block mask by axis-aligned rectangles.

    Each block keeps its real footprint to within the leftovers, and a rectangle
    is exactly what `Boxes` can represent, so nothing is lost in translation.
    Stops when the largest remaining rectangle is smaller than `min_cells` --
    the residue is sub-building speckle along block edges.
    """
    work = mask.copy()
    rects = []
    while True:
        area, y, x, h, w = largest_rectangle(work)
        if area < min_cells:
            return rects, work
        rects.append((y, x, h, w))
        work[y:y + h, x:x + w] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dxf")
    ap.add_argument("out")
    ap.add_argument("--cx", type=float, default=81000.0)
    ap.add_argument("--cy", type=float, default=54000.0)
    ap.add_argument("--half", type=float, default=600.0, help="window half-size, m")
    ap.add_argument("--cell", type=float, default=4.0, help="raster cell, m")
    ap.add_argument("--scale", type=float, default=12.0,
                    help="real metres per simulated metre")
    ap.add_argument("--min-area", type=float, default=400.0,
                    help="drop block rectangles below this many real m^2")
    ap.add_argument("--wp-spacing", type=float, default=60.0,
                    help="minimum real-metre spacing between waypoints")
    ap.add_argument("--min-clear", type=float, default=0.90,
                    help="required SIM-metre clearance at a waypoint; must "
                         "exceed the controller's standoff (prox_band 0.60) or "
                         "the drone is asked to hover somewhere it will not go")
    ap.add_argument("--z-lo", type=float, default=1.0)
    ap.add_argument("--z-hi", type=float, default=2.0,
                    help="waypoint altitude band, SIM metres")
    ap.add_argument("--h-lo", type=float, default=18.0)
    ap.add_argument("--h-hi", type=float, default=90.0, help="building height, m")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    want = set(ROAD_HALF) | {"water"}
    L = read_dxf(a.dxf, want)
    x0, y0 = a.cx - a.half, a.cy - a.half
    n = int(round(2 * a.half / a.cell))
    road = np.zeros((n, n), dtype=bool)
    for layer, half in ROAD_HALF.items():
        stamp_segments(road, L.get(layer, []), half, a.cell, x0, y0)
    water = np.zeros((n, n), dtype=bool)
    stamp_segments(water, L.get("water", []), a.cell, a.cell, x0, y0)

    block = ~road & ~water
    block[0, :] = block[-1, :] = block[:, 0] = block[:, -1] = False   # open rim
    rects, residue = decompose(block, int(round(a.min_area / a.cell ** 2)))

    rng = np.random.default_rng(a.seed)
    S = a.scale
    cs, hs = [], []
    for (y, x, h, w) in rects:
        wx, wy = w * a.cell, h * a.cell
        cx = x0 + (x + 0.5 * w) * a.cell
        cy = y0 + (y + 0.5 * h) * a.cell
        hz = float(rng.uniform(a.h_lo, a.h_hi))
        cs.append([(cx - a.cx) / S, (cy - a.cy) / S])
        hs.append([0.5 * wx / S, 0.5 * wy / S, hz / S])

    # waypoints: road vertices thinned to a spacing, kept clear of every block
    verts = np.concatenate([v for k in ROAD_HALF for v in L.get(k, [])])
    m = (np.abs(verts[:, 0] - a.cx) < a.half * 0.92) & \
        (np.abs(verts[:, 1] - a.cy) < a.half * 0.92)
    verts = verts[m]
    rng.shuffle(verts)
    keep = []
    for p in verts:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > a.wp_spacing ** 2
               for q in keep):
            keep.append(p)
    wps = np.array(keep)
    C = np.array(cs)
    Hh = np.array(hs)
    sim = np.stack([(wps[:, 0] - a.cx) / S, (wps[:, 1] - a.cy) / S], 1)
    # a waypoint has to sit in free space: reject any inside a block's plan
    # footprint plus a margin, since the drone has to be able to hover there
    def clearance(P):
        """Exact plan-view distance from each point to the nearest block face."""
        dx = np.abs(P[:, None, 0] - C[None, :, 0]) - Hh[None, :, 0]
        dy = np.abs(P[:, None, 1] - C[None, :, 1]) - Hh[None, :, 1]
        out = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
        ins = np.minimum(np.maximum(dx, dy), 0.0)
        return (out + ins).min(1)

    clear = clearance(sim)
    ok = clear >= a.min_clear
    sim, wps, clear = sim[ok], wps[ok], clear[ok]
    z = rng.uniform(a.z_lo, a.z_hi, size=len(sim))
    sim3 = np.stack([sim[:, 0], sim[:, 1], z], 1)

    # what the free space actually looks like, so the scale choice is auditable
    gg = np.linspace(-a.half / S, a.half / S, 300)
    GX, GY = np.meshgrid(gg, gg)
    cg = clearance(np.stack([GX.ravel(), GY.ravel()], 1))

    span = float(a.half / S)
    data = {
        "meta": {
            "source": a.dxf, "centre": [a.cx, a.cy], "half_m": a.half,
            "cell_m": a.cell, "scale": S, "span": span,
            "seconds": round(time.time() - t0, 1),
            "n_blocks": len(cs), "n_waypoints": int(len(sim)),
            "road_fraction": float(road.mean()),
            "block_fraction": float(block.mean()),
            "covered_fraction": float(1.0 - residue.sum() / max(block.sum(), 1)),
            "height_m": [a.h_lo, a.h_hi],
            "min_clear": a.min_clear,
            "waypoint_clear": [float(clear.min()), float(np.median(clear))],
            "corridor_halfwidth": [float(np.median(cg[cg > 0])),
                                   float(np.percentile(cg[cg > 0], 90))],
            "flyable_fraction": float((cg > 0.60).mean()),
        },
        "boxes": {"c": C.tolist(), "h": Hh.tolist(),
                  "a": [0.0] * len(cs)},
        "waypoints": sim3.tolist(),
    }
    json.dump(data, open(a.out, "w"))
    print(json.dumps(data["meta"], indent=2))


if __name__ == "__main__":
    main()
