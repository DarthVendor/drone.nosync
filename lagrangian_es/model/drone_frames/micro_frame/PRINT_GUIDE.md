# Quadrotor frame — print and build guide

110 mm wheelbase, 65 mm props, X-config. Designed around the Adafruit
electronics stack (Feather STM32F405, DRV8833 ×2, VL53L1X ×4, LSM6DSOX).

Files in `cad/`:

| file | format | purpose |
|---|---|---|
| `quad_frame.step` / `.stl` | B-rep / mesh | main frame, print 1 |
| `prop_guard.step` / `.stl` | B-rep / mesh | guard ring, print 4 |
| `landing_leg.step` / `.stl` | B-rep / mesh | press-fit leg, print 4 |

STEP files are real parametric CAD — open in Fusion, FreeCAD, Onshape, or
SolidWorks to modify. Edit `frame_model.py` to change wheelbase, prop size,
or the board footprint and re-export.

---

## Read this before printing

**Thrust-to-weight comes out at 1.8, not the 3.0 your sim assumes.**

Computed from the CAD plus a component mass budget: frame 15.5 g, all-up
87.6 g, and 8520 coreless motors give ~40 g static thrust each on 1S. That is
4 × 40 / 87.6 = **1.83**.

This matters more for your project than for a normal build. A vehicle at
T/W 1.8 spends most of an aggressive maneuver at the actuator limit, and
saturation is one of the silent failure modes in the spec — `jacrev` through a
saturated `clamp` returns zero, so a genome living at `f_max` yields a
rank-deficient `G` in exactly the directions that matter, with the ridge hiding
it. Evolving against a saturated plant also produces a controller that has
learned the saturation rather than the dynamics.

Three options:

| config | all-up | T/W | notes |
|---|---|---|---|
| **A** full sensor suite, coreless 8520 | 87.6 g | 1.83 | as modelled; marginal |
| **B** strip 3 lateral ToF, guards, legs | 75.1 g | 2.13 | loses your barrier sensing |
| **C** brushless 1103 on 2S, 4-in-1 ESC | 111.8 g | **3.22** | matches the sim |

**Config C is the one I'd build.** It keeps every Adafruit board — only the
motors, ESC, and battery change, and those were never Adafruit parts anyway.
Set `PlantCfg.mass = 0.112` and `inertia = [6.6e-5, 6.6e-5, 1.27e-4]`.

Config A is still worth printing first as a bench article for fit checks and
system-ID practice. Just don't evolve against it.

---

## Print settings

| | main frame | prop guards | legs |
|---|---|---|---|
| material | PETG | **TPU 95A** | TPU 95A |
| nozzle | 0.4 mm | 0.4 mm | 0.4 mm |
| layer height | 0.15 mm | 0.20 mm | 0.20 mm |
| perimeters | **5** | 3 | 3 |
| top/bottom | 5 / 5 | 3 / 3 | 4 / 4 |
| infill | 35 % gyroid | 20 % gyroid | 40 % gyroid |
| nozzle temp | 240 °C | 230 °C | 230 °C |
| bed | 80 °C | 45 °C | 45 °C |
| cooling | 50 % | 30 % | 30 % |
| speed | 40 mm/s | 20 mm/s | 20 mm/s |
| supports | **none** | none | none |
| adhesion | brim, 5 mm | skirt | brim |

**Perimeter count is doing the work here, not infill.** Arms are 9 mm wide, so
5 perimeters at 0.4 mm is 4 mm of solid wall — nearly the full section. Raising
infill instead adds mass without adding stiffness.

**Orientation.** Main frame flat, plate down, arms in the XY plane. The motor
bosses then print as vertical walls with no overhang, and the whole part needs
no supports. Do not print it on edge; the layer lines end up perpendicular to
the bending load in the arms and they will snap at the root.

**Why TPU for the guards and legs.** These are the impact parts. Generation-0
crashed 92 % of the time, and evolved genomes fail in ways that don't look like
normal control failures. TPU deforms and returns; PETG guards shatter and take
an arm with them.

If you have a hardened nozzle and an enclosure, PA-CF (nylon–carbon) for the
main frame is a real upgrade — stiffer and far more fatigue-resistant at the arm
roots. PETG is the practical default and prints anywhere.

---

## Assembly order

1. **Deburr the motor bores.** They're modelled at 8.75 mm for an 8.5 mm motor
   (0.25 mm press-fit). Test on one before committing all four — elephant's
   foot on the first layers will tighten the bottom of the bore.
2. Feed motor wires through the slot in each boss **before** seating the motor.
3. Press motors in. They should need firm thumb pressure. If they slide in
   freely, increase `MOTOR_FIT` in the model and reprint; a loose motor changes
   your inertia between flights.
4. Mount the Feather on the four integrated standoffs, M2.5 × 6 mm.
5. DRV8833 boards on the plate flanks, LSM6DSOX at centre-rear. Keep the IMU
   as close to the CoM as you can and use a foam pad — coreless motors put a
   lot of high-frequency noise into the gyro.
6. Down-facing VL53L1X and PMW3901 flow sensor under the plate.
7. Battery under the plate, velcro through the two strap slots.
8. Legs press into the underside of each motor boss; guards clip over the top.

---

## System ID — do not skip this

The frame mass and inertia above are computed from CAD geometry and estimated
component masses. Your metric `G = E[(∂u/∂θ)ᵀ M⁻¹ (∂u/∂θ)]` consumes `M⁻¹`
directly, so an error here does not cause an obvious failure — it quietly makes
the whitening the wrong preconditioner, which is indistinguishable from the
whitening not working. That is the one error that corrupts the ablation rather
than breaking the drone.

1. **Mass.** Weigh the finished airframe, flight-ready with battery. Expect the
   printed frame to come in 5–15 % above the CAD figure — the model assumes an
   0.85 fill factor and slicers vary.
2. **Inertia.** Bifilar pendulum: hang the airframe from two parallel threads a
   known distance apart, twist gently, time 20 oscillations.
   `I = (m·g·d²·T²)/(16·π²·L)` for thread separation `d` and length `L`. Repeat
   about each axis. Takes an afternoon and needs about $20 of wire plus a
   printed fixture.
3. **Thrust and torque.** Load cell plus HX711 on a printed arm gives
   thrust-vs-throttle and the drag-torque coefficient. These set `f_max` and
   the thrust-to-torque ratio in the mixer.
4. **Update `PlantCfg` with measured values, then re-run system ID after any
   airframe change.** Reprinting an arm changes the numbers.

---

## Modifying the design

Everything is parametric in `frame_model.py`. The constants worth touching:

```python
WHEELBASE = 110.0   # motor-to-motor diagonal
PROP_D    = 65.0    # clearance is checked automatically
ARM_W     = 9.0     # arm width
ARM_H     = 4.0     # arm depth in Z — the stiffness knob
MOTOR_D   = 8.5     # 8520 coreless; 1103 brushless needs a mount plate instead
MOTOR_FIT = 0.25    # press-fit clearance
```

`check_prop_clearance()` verifies adjacent props don't intersect. Current
geometry gives 77.8 mm adjacent spacing and a 12.8 mm tip gap.

**For config C**, the 1103 brushless motors bolt on rather than press in, so
replace the bore with a 9 mm bolt circle and 2 × M2 holes in each boss. That's
a ten-line change to `build_frame`.
