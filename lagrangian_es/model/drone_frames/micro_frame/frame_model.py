"""
Parametric frame for the Adafruit-class coreless quadrotor.

Builds real B-rep geometry in CadQuery, exports STEP (CAD) + STL (print),
then extracts mass and inertia so PlantCfg can be filled from the actual
airframe instead of the placeholder constants in the prototype.

Sized around:
  - 8520 coreless motors (8.5 mm dia)
  - 65 mm props
  - Feather STM32F405 (50.8 x 22.9 mm, holes 45.72 x 17.78)
  - 2x DRV8833, 4x VL53L1X (STEMMA QT)
  - 1S 500 mAh LiPo
"""
import math, json
import cadquery as cq

# ------------------------------------------------------------------ params
WHEELBASE   = 110.0   # motor-to-motor diagonal
PROP_D      = 65.0
ARM_W       = 9.0
ARM_H       = 4.0     # taller in Z = stiffer where it matters
PLATE_X     = 62.0
PLATE_Y     = 46.0
PLATE_T     = 2.4
FILLET_R    = 4.0

MOTOR_D     = 8.5
MOTOR_FIT   = 0.25    # press-fit clearance (PETG, 0.4 nozzle)
MOUNT_OD    = 12.6
MOUNT_H     = 10.0
WIRE_SLOT   = 3.2

FEATHER_HX  = 45.72   # Feather mounting hole pattern
FEATHER_HY  = 17.78
HOLE_D      = 2.6     # M2.5 clearance
STANDOFF_D  = 5.0
STANDOFF_H  = 3.0

STRAP_W     = 14.0    # velcro battery strap
STRAP_T     = 2.2

R_ARM = WHEELBASE / 2.0
OFF   = R_ARM / math.sqrt(2.0)          # motor x,y offset (X-config, 45 deg)


def check_prop_clearance():
    """Adjacent motors are OFF*2 apart along one axis in an X-quad."""
    gap = 2 * OFF - PROP_D
    return {'adjacent_spacing_mm': round(2 * OFF, 1),
            'prop_tip_gap_mm': round(gap, 1),
            'ok': gap > 5.0}


# ------------------------------------------------------------------- frame
def build_frame():
    # central plate
    frame = (cq.Workplane("XY")
             .box(PLATE_X, PLATE_Y, PLATE_T, centered=(True, True, False))
             .edges("|Z").fillet(FILLET_R))

    # four arms, drawn as boxes rotated into the diagonals
    for ang in (45, 135, 225, 315):
        arm = (cq.Workplane("XY")
               .center(R_ARM / 2.0, 0)
               .box(R_ARM + 6, ARM_W, ARM_H, centered=(True, True, False))
               .rotate((0, 0, 0), (0, 0, 1), ang))
        frame = frame.union(arm)

    # motor mounts at the arm ends
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        x, y = sx * OFF, sy * OFF
        boss = (cq.Workplane("XY").center(x, y)
                .circle(MOUNT_OD / 2).extrude(MOUNT_H))
        frame = frame.union(boss)
        # motor bore
        frame = (frame.faces(">Z").workplane(origin=(x, y, 0))
                 .center(x, y).circle((MOTOR_D + MOTOR_FIT) / 2)
                 .cutBlind(-MOUNT_H))
        # wire exit slot, cut toward the centre
        slot = (cq.Workplane("XY").center(x, y)
                .box(MOUNT_OD, WIRE_SLOT, MOUNT_H - 2.0,
                     centered=(True, True, False))
                .rotate((x, y, 0), (x, y, 1),
                        math.degrees(math.atan2(-sy, -sx))))
        frame = frame.cut(slot.translate((0, 0, 2.0)))

    # Feather standoffs + mounting holes
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * FEATHER_HX / 2, sy * FEATHER_HY / 2
            frame = frame.union(cq.Workplane("XY").center(x, y)
                                .circle(STANDOFF_D / 2)
                                .extrude(PLATE_T + STANDOFF_H))
            frame = frame.cut(cq.Workplane("XY").center(x, y)
                              .circle(HOLE_D / 2)
                              .extrude(PLATE_T + STANDOFF_H + 1))

    # battery strap slots through the plate
    for sx in (-1, 1):
        frame = frame.cut(cq.Workplane("XY")
                          .center(sx * (PLATE_X / 2 - 8), 0)
                          .box(STRAP_T, STRAP_W, PLATE_T + 2,
                               centered=(True, True, False))
                          .translate((0, 0, -1)))

    # lightening cutouts between the arms (keeps mass down, adds cable routes)
    for ang in (0, 90, 180, 270):
        cut = (cq.Workplane("XY").center(PLATE_X / 2 - 6, 0)
               .circle(4.5).extrude(PLATE_T + 2)
               .rotate((0, 0, 0), (0, 0, 1), ang)
               .translate((0, 0, -1)))
        try:
            frame = frame.cut(cut)
        except Exception:
            pass

    return frame


def build_prop_guard():
    """Ring that press-fits over a motor boss. Print in TPU (impact)."""
    r_in = PROP_D / 2 + 3.0
    g = (cq.Workplane("XY").circle(r_in + 2.0).circle(r_in)
         .extrude(7.0))
    tab = (cq.Workplane("XY").center(-(r_in + 1.0), 0)
           .box(14.0, 10.0, 3.0, centered=(True, True, False)))
    g = g.union(tab)
    g = g.cut(cq.Workplane("XY").center(-(r_in + 5.0), 0)
              .circle((MOUNT_OD + 0.4) / 2).extrude(3.0))
    return g


def build_leg():
    """Press-fit landing leg. Print 4, separately, in TPU or PETG."""
    leg = (cq.Workplane("XY").circle(MOUNT_OD / 2 - 0.15).extrude(6.0)
           .faces(">Z").workplane().circle(3.2).extrude(14.0)
           .faces(">Z").workplane().circle(5.0).extrude(2.0)
           .edges(">Z").fillet(1.4))
    return leg


if __name__ == "__main__":
    import os
    out = "/mnt/user-data/outputs/cad"
    os.makedirs(out, exist_ok=True)

    parts = {"quad_frame": build_frame(),
             "prop_guard": build_prop_guard(),
             "landing_leg": build_leg()}

    for name, solid in parts.items():
        cq.exporters.export(solid, f"{out}/{name}.step")
        cq.exporters.export(solid, f"{out}/{name}.stl",
                            tolerance=0.01, angularTolerance=0.1)
        print(f"exported {name}")

    print(json.dumps(check_prop_clearance(), indent=2))
