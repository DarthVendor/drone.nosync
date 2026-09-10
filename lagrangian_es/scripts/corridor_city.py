"""A city that is nothing but corridors.

A Manhattan grid of tall blocks separated by streets of one width, ringed by
an outer wall, so that every point a vehicle can be at is inside a corridor
or at a junction of corridors: walls on at least two sides, never open ground.
Built for the enclosed-space failure the city measurement singled out.

    python scripts/corridor_city.py [--width 4] [--pitch 12] [--n 5] [--height 8]

Writes `src/lagrangian_es/environments/maps/corridors.json` in the same schema
as the DXF-derived maps (`boxes` c/h/a, `waypoints`, `meta`), so it loads
through `city_to_environment` like the others.  Waypoints sit at every
junction and every corridor midpoint, at 1.5 m: a leg between neighbouring
junctions runs straight down a street, a leg across a block's diagonal is
occluded and has to go round the corner.
"""
import argparse, json, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=float, default=4.0, help="street width, simulated metres")
ap.add_argument("--pitch", type=float, default=12.0, help="block-centre spacing (block + one street)")
ap.add_argument("--n", type=int, default=5, help="blocks per side")
ap.add_argument("--height", type=float, default=8.0, help="block height (the hz slot)")
ap.add_argument("--wall", type=float, default=0.5, help="outer ring wall thickness")
ap.add_argument("--z", type=float, default=1.5, help="waypoint height")
ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "src/lagrangian_es/environments/maps/corridors.json"))
a = ap.parse_args()

w, P, n, H = a.width, a.pitch, a.n, a.height
half_block = 0.5 * (P - w)
centres = [(-0.5 * (n - 1) * P + i * P) for i in range(n)]             # block centres per axis
streets = [c - 0.5 * P for c in centres] + [centres[-1] + 0.5 * P]     # street centre-lines per axis (n + 1)
outer = streets[-1] + 0.5 * w                                         # the outer streets' far edge
C, Hh, A = [], [], []
for cx in centres:
    for cy in centres:
        C.append([cx, cy]); Hh.append([half_block, half_block, H]); A.append(0.0)
# the ring wall: four slabs just outside the outer streets
t = a.wall
for sx, sy, hx, hy in ((outer + 0.5 * t, 0.0, 0.5 * t, outer + t), (-(outer + 0.5 * t), 0.0, 0.5 * t, outer + t),
                       (0.0, outer + 0.5 * t, outer + t, 0.5 * t), (0.0, -(outer + 0.5 * t), outer + t, 0.5 * t)):
    C.append([sx, sy]); Hh.append([hx, hy, H]); A.append(0.0)
wps = []
for i, sx in enumerate(streets):
    for j, sy in enumerate(streets):
        wps.append([sx, sy, a.z])                                         # the junction
        if i + 1 < len(streets):
            wps.append([0.5 * (sx + streets[i + 1]), sy, a.z])            # midpoint of the street heading +x
        if j + 1 < len(streets):
            wps.append([sx, 0.5 * (sy + streets[j + 1]), a.z])            # midpoint of the street heading +y
data = {"meta": {"kind": "corridors", "width": w, "pitch": P, "n": n, "height_m": H, "span": outer + t,
                 "n_blocks": len(C), "n_waypoints": len(wps),
                 "note": "a Manhattan grid ringed by a wall: every waypoint is inside a corridor or at a junction"},
        "boxes": {"c": C, "h": Hh, "a": A}, "waypoints": wps}
pathlib.Path(a.out).write_text(json.dumps(data))
print(f"{a.out}: {len(C)} boxes ({n}x{n} blocks + 4 ring walls), {len(wps)} waypoints, streets {w} m wide, span {outer + t:.1f} m")
