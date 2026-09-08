#!/usr/bin/env python3
"""A map on which every waypoint is behind a wall from every other.

The city measurement: at legs of up to 20 m, a leg whose straight line crosses
a building is lost 91% of the time against 24% with line of sight.  This map
isolates that one skill.  Each waypoint sits inside a U-shaped pocket of thin
walls; every pocket opens toward +x, so any two waypoints -- same row, same
column, or diagonal -- have a wall on the straight line between them.  Getting anywhere means leaving the pocket the way it opens
and going round.  Nothing here is a shortest path to anything.

    python scripts/occluded_map.py            # writes maps/occluded.json
"""
from __future__ import annotations

import json
import pathlib

ROWS, COLS, PITCH = 4, 4, 12.0          # 16 waypoints, 36 m across
INNER, DEPTH, WALL, HEIGHT = 4.0, 5.0, 0.5, 4.0   # pocket width, depth, wall thickness, height
Z = 1.5                                  # cruise altitude, same as the city


def build():
    boxes_c, boxes_h, boxes_a, wps = [], [], [], []
    for r in range(ROWS):
        for c in range(COLS):
            cx = (c - (COLS - 1) / 2) * PITCH
            cy = (r - (ROWS - 1) / 2) * PITCH
            # Every pocket opens the same way.  Alternating by row put +x
            # openings opposite -x ones and nine of 120 pairs could see each
            # other along a shallow diagonal; with one direction a line that
            # leaves a pocket through its opening travels +x and meets the next
            # pocket's closed back or side before its interior, for every pair.
            open_px = True
            sgn = 1.0 if open_px else -1.0
            # back wall: across the closed side, at x = cx - sgn*DEPTH/2
            boxes_c.append([cx - sgn * DEPTH / 2, cy]); boxes_h.append([WALL / 2, INNER / 2 + WALL, HEIGHT]); boxes_a.append(0.0)
            # two side walls, running along x from the back wall to the opening
            for side in (-1.0, 1.0):
                boxes_c.append([cx, cy + side * (INNER / 2 + WALL / 2)])
                boxes_h.append([DEPTH / 2, WALL / 2, HEIGHT]); boxes_a.append(0.0)
            wps.append([cx, cy, Z])
    span = max(COLS, ROWS) * PITCH / 2 + 2.0
    return {"meta": {"source": "scripts/occluded_map.py", "span": span, "height_m": HEIGHT,
                     "n_blocks": len(boxes_c), "n_waypoints": len(wps),
                     "pocket": {"inner": INNER, "depth": DEPTH, "wall": WALL, "pitch": PITCH},
                     "note": "every waypoint pair is occluded by construction"},
            "boxes": {"c": boxes_c, "h": boxes_h, "a": boxes_a}, "waypoints": wps}


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parents[1] / "src/lagrangian_es/environments/maps/occluded.json"
    out.write_text(json.dumps(build(), indent=1))
    print(f"wrote {out}: {len(build()['waypoints'])} waypoints, {len(build()['boxes']['c'])} walls")
