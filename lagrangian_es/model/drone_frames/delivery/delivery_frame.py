"""
Delivery-class quadrotor: 650 mm wheelbase, 15" props, 1.5 kg takeout payload.

NOT a scale-up of the coreless frame. At this size the arms are carbon tube;
printed parts are joints, mounts, and the payload gimbal only. A printed arm
carrying 2.6 kg of thrust at 325 mm is a fatigue failure waiting to happen.

The payload hangs in a passive 2-axis gimbal so soup stays level while the
vehicle banks. That gimbal is also why this airframe is interesting for the
research: a swinging payload makes M(q) genuinely configuration-dependent.
"""
import math, os
import cadquery as cq

# ------------------------------------------------------------------ params
WHEELBASE   = 650.0
PROP_D      = 15 * 25.4
TUBE_OD     = 25.0
TUBE_ID     = 23.0

HUB_X       = 200.0
HUB_Y       = 200.0
HUB_T       = 4.0
HUB_FILLET  = 10.0

CLAMP_R     = 85.0          # clamp centre distance from hub origin
CLAMP_L     = 60.0
CLAMP_W     = 34.0
CLAMP_H     = 34.0
BOLT_D      = 3.4           # M3 clearance

MOTOR_BOLT  = 25.0          # square pattern, M3
MOUNT_PLATE = 40.0
MOUNT_T     = 5.0

GIMBAL_R    = 90.0          # outer ring radius
GIMBAL_W    = 10.0
GIMBAL_T    = 8.0
PIVOT_D     = 6.2           # M6 shoulder bolt

OFF = WHEELBASE / 2 / math.sqrt(2)


def build_hub():
    """Top hub plate: arm clamp seats, gimbal pivot bosses, battery strap slots."""
    hub = (cq.Workplane("XY")
           .box(HUB_X, HUB_Y, HUB_T, centered=(True, True, False))
           .edges("|Z").fillet(HUB_FILLET))

    # arm clamp bolt pattern, 4 pairs at 45 deg
    for ang in (45, 135, 225, 315):
        a = math.radians(ang)
        for d in (-14, 14):
            for r in (CLAMP_R - 20, CLAMP_R + 20):
                x = r * math.cos(a) - d * math.sin(a)
                y = r * math.sin(a) + d * math.cos(a)
                if abs(x) < HUB_X / 2 - 6 and abs(y) < HUB_Y / 2 - 6:
                    hub = hub.cut(cq.Workplane("XY").center(x, y)
                                  .circle(BOLT_D / 2).extrude(HUB_T + 2)
                                  .translate((0, 0, -1)))

    # gimbal roll-axis pivot bosses on the +/-X edges
    for sx in (-1, 1):
        b = (cq.Workplane("YZ").center(0, HUB_T)
             .circle(11.0).extrude(10.0)
             .translate((sx * (HUB_X / 2 - 5), 0, 0)))
        hub = hub.union(b)
        hub = hub.cut(cq.Workplane("YZ").center(0, HUB_T).circle(PIVOT_D / 2)
                      .extrude(14.0)
                      .translate((sx * (HUB_X / 2 - 9), 0, 0)))

    # battery strap slots + central lightening window
    for sy in (-1, 1):
        hub = hub.cut(cq.Workplane("XY").center(0, sy * 45)
                      .box(3.0, 30.0, HUB_T + 2, centered=(True, True, False))
                      .translate((0, 0, -1)))
    hub = hub.cut(cq.Workplane("XY").center(0, 0).circle(28.0)
                  .extrude(HUB_T + 2).translate((0, 0, -1)))
    return hub


def build_arm_clamp():
    """Half of a two-piece tube clamp. Print 8, bolt in pairs around the tube."""
    c = (cq.Workplane("XY")
         .box(CLAMP_L, CLAMP_W, CLAMP_H / 2, centered=(True, True, False))
         .edges("|Z").fillet(4.0))
    c = c.cut(cq.Workplane("XZ").center(0, CLAMP_H / 2)
              .circle(TUBE_OD / 2).extrude(CLAMP_L, both=True))
    for dx in (-20, 20):
        for dy in (-14, 14):
            c = c.cut(cq.Workplane("XY").center(dx, dy)
                      .circle(BOLT_D / 2).extrude(CLAMP_H).translate((0, 0, -1)))
    return c


def build_motor_mount():
    """Tube end cap carrying the motor bolt pattern."""
    m = (cq.Workplane("XY").circle(TUBE_OD / 2 + 3.0).extrude(30.0))
    m = m.cut(cq.Workplane("XY").circle(TUBE_OD / 2 + 0.15).extrude(24.0))
    m = m.union(cq.Workplane("XY").box(MOUNT_PLATE, MOUNT_PLATE, MOUNT_T,
                                       centered=(True, True, False))
                .edges("|Z").fillet(5.0).translate((0, 0, 30.0)))
    for sx in (-1, 1):
        for sy in (-1, 1):
            m = m.cut(cq.Workplane("XY")
                      .center(sx * MOTOR_BOLT / 2, sy * MOTOR_BOLT / 2)
                      .circle(BOLT_D / 2).extrude(10.0).translate((0, 0, 28.0)))
    # clamping slot + pinch bolt
    m = m.cut(cq.Workplane("XY").box(2.5, TUBE_OD + 8, 22.0,
                                     centered=(True, True, False)))
    m = m.cut(cq.Workplane("XZ").center(0, 11.0).circle(BOLT_D / 2)
              .extrude(TUBE_OD + 10, both=True))
    return m


def build_gimbal_ring():
    """Outer ring of the passive 2-axis payload gimbal.

    Pivots on the hub's roll axis; the payload basket hangs inside it on the
    orthogonal pitch axis. Gravity does the levelling — no actuators.
    """
    g = (cq.Workplane("XY").circle(GIMBAL_R).circle(GIMBAL_R - GIMBAL_W)
         .extrude(GIMBAL_T))
    for sx in (-1, 1):     # roll pivots (to hub)
        g = g.union(cq.Workplane("XY").center(sx * (GIMBAL_R - GIMBAL_W / 2), 0)
                    .circle(10.0).extrude(GIMBAL_T))
        g = g.cut(cq.Workplane("XY").center(sx * (GIMBAL_R - GIMBAL_W / 2), 0)
                  .circle(PIVOT_D / 2).extrude(GIMBAL_T + 2).translate((0, 0, -1)))
    for sy in (-1, 1):     # pitch pivots (to basket)
        g = g.union(cq.Workplane("XY").center(0, sy * (GIMBAL_R - GIMBAL_W / 2))
                    .circle(10.0).extrude(GIMBAL_T))
        g = g.cut(cq.Workplane("XY").center(0, sy * (GIMBAL_R - GIMBAL_W / 2))
                  .circle(PIVOT_D / 2).extrude(GIMBAL_T + 2).translate((0, 0, -1)))
    return g


if __name__ == "__main__":
    out = "/mnt/user-data/outputs/cad_delivery"
    os.makedirs(out, exist_ok=True)
    parts = {"hub_plate": build_hub(),
             "arm_clamp": build_arm_clamp(),
             "motor_mount": build_motor_mount(),
             "gimbal_ring": build_gimbal_ring()}
    for name, s in parts.items():
        cq.exporters.export(s, f"{out}/{name}.step")
        cq.exporters.export(s, f"{out}/{name}.stl",
                            tolerance=0.02, angularTolerance=0.15)
        print("exported", name)

    adj = WHEELBASE / math.sqrt(2)
    print(f"adjacent motor spacing {adj:.0f} mm | prop {PROP_D:.0f} mm | "
          f"tip gap {adj - PROP_D:.0f} mm")
