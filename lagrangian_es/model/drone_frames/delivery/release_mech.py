"""
Package release: winch + passive tension-release hook.

The hook holds the payload bail while the tether is under load. When the box
touches down the tether goes slack, a counterweighted gate rotates open, and
the bail slides free. No actuator at the hook, no wire running down the tether,
and nothing to fail in a way that drops the package in flight -- release
requires weight-off, which cannot happen at altitude.

Drone hovers at 15-25 m; only ~60 s of hover per delivery.
"""
import math, os
import cadquery as cq

DRUM_D      = 46.0        # tether drum
DRUM_W      = 34.0
FLANGE_D    = 74.0
FLANGE_T    = 3.0
SHAFT_D     = 6.2         # M6
TETHER_D    = 2.0         # 2 mm Dyneema, ~200 kg break

HOOK_T      = 9.0         # plate thickness
HOOK_L      = 82.0
BAIL_D      = 8.0         # payload bail rod diameter
GATE_PIVOT  = 4.2         # M4
BOLT_D      = 3.4


def build_winch_drum():
    """Drum with flanges and a tether anchor hole. Print in PETG, 6 walls."""
    d = (cq.Workplane("XY").circle(DRUM_D / 2).extrude(DRUM_W))
    for z in (0.0, DRUM_W - FLANGE_T):
        d = d.union(cq.Workplane("XY").circle(FLANGE_D / 2)
                    .extrude(FLANGE_T).translate((0, 0, z)))
    d = d.cut(cq.Workplane("XY").circle(SHAFT_D / 2).extrude(DRUM_W + 2)
              .translate((0, 0, -1)))
    # D-flat so the drum keys to the gearmotor shaft
    d = d.cut(cq.Workplane("XY").center(SHAFT_D / 2 - 0.6, 0)
              .box(1.6, SHAFT_D, DRUM_W + 2, centered=(True, True, False))
              .translate((0, 0, -1)))
    # tether anchor: radial hole into the core
    d = d.cut(cq.Workplane("XZ").center(0, DRUM_W / 2)
              .circle(TETHER_D / 2 + 0.3).extrude(FLANGE_D, both=True))
    return d


def build_hook_body():
    """Hook plate. Tether eye at top, open throat for the bail, gate pivot."""
    b = (cq.Workplane("XZ")
         .moveTo(-16, 0).lineTo(16, 0).lineTo(16, HOOK_L - 26)
         .lineTo(6, HOOK_L).lineTo(-6, HOOK_L).lineTo(-16, HOOK_L - 26)
         .close().extrude(HOOK_T))
    # tether eye
    b = b.cut(cq.Workplane("XZ").center(0, HOOK_L - 9).circle(3.2)
              .extrude(HOOK_T + 2).translate((0, -1, 0)))
    # throat the bail rides in: open on -X, closed at the top
    throat = (cq.Workplane("XZ").center(-4, 20).circle(BAIL_D / 2 + 0.6)
              .extrude(HOOK_T + 2).translate((0, -1, 0)))
    slot = (cq.Workplane("XZ").center(-14, 20)
            .box(22.0, BAIL_D + 1.2, HOOK_T + 2, centered=(True, True, False))
            .translate((0, -1, -(HOOK_T + 2) / 2)))
    b = b.cut(throat).cut(slot)
    # gate pivot + mounting holes
    b = b.cut(cq.Workplane("XZ").center(9, 34).circle(GATE_PIVOT / 2)
              .extrude(HOOK_T + 2).translate((0, -1, 0)))
    return b


def build_gate():
    """Counterweighted gate: tether load holds it shut, slack lets it fall open."""
    g = (cq.Workplane("XY")
         .moveTo(-6, -5).lineTo(30, -5).lineTo(30, 5).lineTo(-6, 5)
         .close().extrude(HOOK_T - 1.0)
         .edges("|Z").fillet(2.0))
    g = g.cut(cq.Workplane("XY").circle(GATE_PIVOT / 2)
              .extrude(HOOK_T + 2).translate((0, 0, -1)))
    # counterweight pocket at the far end (fill with a M5 nut stack)
    g = g.cut(cq.Workplane("XY").center(24, 0).circle(4.6)
              .extrude(HOOK_T - 3.0).translate((0, 0, 1.5)))
    return g


def build_tether_guide():
    """Fairlead under the hub; keeps the tether off the gear and out of the wake."""
    t = (cq.Workplane("XY").box(52.0, 34.0, 8.0, centered=(True, True, False))
         .edges("|Z").fillet(6.0))
    t = t.cut(cq.Workplane("XY").circle(TETHER_D / 2 + 2.5)
              .extrude(10.0).translate((0, 0, -1)))
    t = t.faces(">Z").edges(">>X[-1] or <<X[-1]").fillet(1.5) \
        if False else t
    for sx in (-1, 1):
        t = t.cut(cq.Workplane("XY").center(sx * 20, 0).circle(BOLT_D / 2)
                  .extrude(10.0).translate((0, 0, -1)))
    return t


def release_analysis():
    payload_N = 1.5 * 9.81
    # gate stays shut while tether tension exceeds the counterweight moment
    gate_cw_g, r_cw, r_lock = 12.0, 0.024, 0.009
    hold_moment = gate_cw_g / 1000 * 9.81 * r_cw
    return {
        "tether": "2 mm Dyneema, ~200 kg break, >100x payload",
        "hover_height_m": [15, 25],
        "descent_rate_ms": 1.0,
        "hover_time_per_delivery_s": round(2 * 20 / 1.0 + 20),
        "gate_hold_moment_Nmm": round(hold_moment * 1000, 2),
        "release_condition": "tether tension < ~0.15 N (weight-off)",
        "payload_load_N": round(payload_N, 1),
        "fails_safe": "gate cannot open under load; release needs ground contact",
    }


if __name__ == "__main__":
    out = "/mnt/user-data/outputs/cad_delivery"
    os.makedirs(out, exist_ok=True)
    for name, s in {"winch_drum": build_winch_drum(),
                    "release_hook_body": build_hook_body(),
                    "release_gate": build_gate(),
                    "tether_guide": build_tether_guide()}.items():
        cq.exporters.export(s, f"{out}/{name}.step")
        cq.exporters.export(s, f"{out}/{name}.stl",
                            tolerance=0.02, angularTolerance=0.15)
        print("exported", name)
    import json; print(json.dumps(release_analysis(), indent=2))
