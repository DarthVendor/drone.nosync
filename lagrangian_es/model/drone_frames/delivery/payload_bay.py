"""
Payload bay v2: rigid insulated box on a compact cardan (universal) joint.

Replaces the 90 mm gimbal ring, which was undersized by >2x for a real order
and would have needed 400 mm diameter to work. A cardan joint at the top of the
box does the same job with ~80 mm of hardware and self-levels under gravity.

Bag never leaves the box, so the aerodynamics stay those of a rigid bluff body.
"""
import math, os
import cadquery as cq

BAY_X, BAY_Y, BAY_Z = 280.0, 240.0, 260.0   # internal, fits 3 entrees + 2 drinks
WALL = 4.0
PIVOT_D = 6.2          # M6 shoulder bolt
YOKE_W = 90.0
YOKE_T = 8.0
ARM_H = 46.0
BOLT_D = 3.4


def build_cardan_outer():
    """U-yoke bolted under the hub. Carries the roll axis."""
    base = (cq.Workplane("XY").box(YOKE_W + 24, 44.0, YOKE_T,
                                   centered=(True, True, False))
            .edges("|Z").fillet(6.0))
    for sx in (-1, 1):
        for sy in (-1, 1):
            base = base.cut(cq.Workplane("XY")
                            .center(sx * (YOKE_W / 2 + 6), sy * 14)
                            .circle(BOLT_D / 2).extrude(YOKE_T + 2)
                            .translate((0, 0, -1)))
    for sx in (-1, 1):
        arm = (cq.Workplane("XY").center(sx * YOKE_W / 2, 0)
               .box(YOKE_T, 34.0, ARM_H, centered=(True, True, False))
               .translate((0, 0, YOKE_T))
               .edges("|X").fillet(5.0))
        base = base.union(arm)
        base = base.cut(cq.Workplane("YZ").center(0, YOKE_T + ARM_H - 13)
                        .circle(PIVOT_D / 2).extrude(YOKE_W, both=True))
    return base


def build_cardan_cross():
    """Cross link: roll pins outboard on X, pitch pins outboard on Y."""
    c = (cq.Workplane("XY").box(YOKE_W - 6, 30.0, 22.0,
                                centered=(True, True, False))
         .edges("|Z").fillet(4.0))
    c = c.union(cq.Workplane("XY").box(30.0, YOKE_W - 26, 22.0,
                                       centered=(True, True, False))
                .edges("|Z").fillet(4.0))
    c = c.cut(cq.Workplane("YZ").center(0, 11.0).circle(PIVOT_D / 2)
              .extrude(YOKE_W, both=True))
    c = c.cut(cq.Workplane("XZ").center(0, 11.0).circle(PIVOT_D / 2)
              .extrude(YOKE_W, both=True))
    return c


def build_bay_bracket():
    """Lid bracket: pitch pivots up to the cross, flange down to the box."""
    b = (cq.Workplane("XY").box(120.0, YOKE_W + 20, YOKE_T,
                                centered=(True, True, False))
         .edges("|Z").fillet(8.0))
    for sy in (-1, 1):
        arm = (cq.Workplane("XY").center(0, sy * (YOKE_W - 26) / 2)
               .box(34.0, YOKE_T, ARM_H - 8, centered=(True, True, False))
               .translate((0, 0, YOKE_T)).edges("|Y").fillet(5.0))
        b = b.union(arm)
        b = b.cut(cq.Workplane("XZ").center(0, YOKE_T + ARM_H - 21)
                  .circle(PIVOT_D / 2).extrude(YOKE_W, both=True))
    for sx in (-1, 1):
        for sy in (-1, 1):
            b = b.cut(cq.Workplane("XY").center(sx * 48, sy * 46)
                      .circle(BOLT_D / 2).extrude(YOKE_T + 2)
                      .translate((0, 0, -1)))
    return b


def geometry_report():
    swing_clear = math.degrees(math.atan2(BAY_X / 2, BAY_Z))
    gear_needed = BAY_Z + 90 + 60          # box + cardan stack + ground clearance
    return {
        "bay_internal_mm": [BAY_X, BAY_Y, BAY_Z],
        "bay_volume_L": round(BAY_X * BAY_Y * BAY_Z / 1e6, 1),
        "cardan_stack_height_mm": 90,
        "landing_gear_height_mm": gear_needed,
        "free_swing_before_corner_strike_deg": round(swing_clear, 1),
        "old_ring_radius_needed_mm": 196,
        "cardan_footprint_mm": YOKE_W + 24,
    }


if __name__ == "__main__":
    out = "/mnt/user-data/outputs/cad_delivery"
    os.makedirs(out, exist_ok=True)
    for name, s in {"cardan_outer": build_cardan_outer(),
                    "cardan_cross": build_cardan_cross(),
                    "bay_bracket": build_bay_bracket()}.items():
        cq.exporters.export(s, f"{out}/{name}.step")
        cq.exporters.export(s, f"{out}/{name}.stl",
                            tolerance=0.02, angularTolerance=0.15)
        print("exported", name)
    import json; print(json.dumps(geometry_report(), indent=2))
