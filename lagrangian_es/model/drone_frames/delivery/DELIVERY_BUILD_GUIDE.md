# Delivery quadrotor — 650 mm, 1.5 kg takeout payload

**This is not a scaled version of the coreless frame. It is a different
vehicle, and it is dangerous.** 15" carbon props at 5,000 rpm remove fingers.
A 5.3 kg airframe falling from 30 m is lethal. Read §5 before building.

Files in `cad_delivery/` — STEP (parametric) and STL (print) for each part.

---

## 1. Sizing

| | |
|---|---|
| wheelbase | 650 mm, X-config |
| props | 15" (381 mm), 79 mm tip clearance |
| motors | MN4014-class, ~340 KV, 4× |
| battery | 6S 10 Ah (222 Wh) |
| **empty** | **3.84 kg** |
| **payload** | **1.5 kg** (28% of AUW) |
| **AUW** | **5.34 kg** |
| thrust | 10.4 kg → **T/W 1.95 loaded, 2.71 empty** |
| hover | ~593 W → 22 min, **~16 min usable** |

Mass budget: arms/hub/hardware 800 g, motors 620 g, ESCs 220 g, props 120 g,
battery 1200 g, avionics 250 g, PDB/wiring 200 g, gimbal bay 250 g, gear 180 g.

16 minutes usable is roughly a 3–5 km round trip with reserve. That is a real
takeout radius, and it is also the number that collapses first — every gram of
payload past 1.5 kg comes straight out of it.

T/W 1.95 loaded is the practical floor. Below ~1.8 you have no authority left
for gusts or a failed motor.

---

## 2. What is printed and what is not

**Printed:** hub plates, arm clamps, motor mounts, gimbal ring, landing feet.
**Not printed:** the arms.

25 mm OD carbon tube for arms, non-negotiable. Each arm carries 2.6 kg of
thrust at a 325 mm moment arm, cycling at rotor frequency for the whole flight.
A printed arm at that scale is a fatigue crack propagating along layer lines —
it will survive the bench test and fail in the air.

| part | qty | est. mass (PETG) | notes |
|---|---|---|---|
| `hub_plate` | 2 | 176 g each | print both; sandwich construction |
| `arm_clamp` | 8 | 29 g each | two halves per arm |
| `motor_mount` | 4 | 21 g each | pinch-bolt onto tube end |
| `gimbal_ring` | 1 | 52 g | passive 2-axis payload gimbal |

Printed total ≈ 750 g, which is most of the 800 g structure allowance. If you
have access to carbon plate at Berkeley, cut the hub plates instead and save
~250 g — that is 250 g of payload or endurance back.

**Print settings, all parts:** PETG or PA-CF, 0.4 mm nozzle, 0.2 mm layers,
**6 perimeters**, 40% gyroid, 245 °C / 85 °C, 30 mm/s, brim. No supports if the
hub prints flat and the motor mounts print bore-up.

PA-CF is worth the hardened nozzle here in a way it was not at 88 g. These are
load-bearing joints.

---

## 3. The payload gimbal

Takeout needs to stay level. A hard-mounted bay tilts with the vehicle, and at
a 20° bank your soup is in the bag rather than the container. The `gimbal_ring`
part is a passive 2-axis gimbal: roll pivots to the hub, pitch pivots to the
basket, gravity does the levelling. No actuators, no control loop.

**This is also the most interesting thing about the airframe for your research.**
A swinging payload makes the inertia tensor a function of the swing angle:

| swing | `I_xx` | `I_xz` coupling |
|---|---|---|
| 0° | 0.159 kg·m² | 0 |
| 30° | 0.150 kg·m² | 0.017 kg·m² |

`I_xx` moves 6%, but the off-diagonal coupling goes from zero to non-zero —
the mass matrix stops being diagonal at all. That is the configuration-dependent
`M(q)` I said the quadrotor didn't have and the two-link arm did. Here it
appears on a flying vehicle, which means the strong form of the whitening claim
becomes testable without switching plants.

Two caveats before you build a paper on it. The gimbal adds two unactuated DOF,
so the matching conditions from §3.4 now bind for real — you cannot choose `T_d`
and `V_d` independently. And a liquid payload sloshes, which is a distributed
system that a single pendulum DOF models badly.

Worth knowing: Sreenath's lab at Berkeley has published geometric control for
quadrotors with cable-suspended and elastically-suspended payloads. That is
directly upstream of this and worth reading before you design the controller.

---

## 4. Assembly notes

1. Bond nothing until the tubes are cut to identical length — measure from the
   clamp face, not the tube end. Arm length asymmetry shows up as a persistent
   attitude bias that will look like a controller bug.
2. Clamp halves bolt around the tube with M3 × 30. Do not overtighten onto
   carbon; crush the tube and you lose most of its stiffness.
3. Motor mounts pinch-bolt onto the tube ends. Add a witness mark so you can
   check for rotation after a hard landing.
4. Route ESC wires inside the tubes. Cleaner, and it keeps the ESCs in prop wash
   for cooling.
5. Battery on the top plate, over the CoM. The gimbal hangs below through the
   central window.
6. Balance the airframe empty before first flight, then re-check loaded.

---

## 5. Before you fly this

**Do not evolve on this vehicle.** Generation-0 crashed 92% of the time, and
evolved genomes fail in ways that don't resemble normal control failures. That
was free on a 88 g coreless quad bouncing off a net. At 5.3 kg with 15" carbon
it is a serious injury. Train in sim, validate a fixed genome in sim across
seeds and disturbances, then fly only a frozen controller with a manual
override that cuts to a known-good PID.

**Regulatory.** Over 250 g requires FAA registration. Delivery is a commercial
operation, so Part 107 applies, and flight over people or beyond visual line of
sight needs specific authorisation. Berkeley will have its own UAS policy and
almost certainly requires approval and insurance for anything at this scale.
Check both before ordering parts — it may change what you build.

**Test progression.** Tethered hover at 1 m over a net, in a large indoor space
or a fenced outdoor area, with props guarded and a spotter. Then untethered
hover, then translation, then payload. Never skip to the last step because the
sim looked good.

---

## 6. Scaling the model

All parameters are at the top of `delivery_frame.py`:

```python
WHEELBASE  = 650.0    # clearance is checked on export
PROP_D     = 15 * 25.4
TUBE_OD    = 25.0     # carbon arm tube
MOTOR_BOLT = 25.0     # motor bolt square pattern
GIMBAL_R   = 90.0     # payload ring radius
```

For a larger payload the honest move is a hexacopter, not bigger props. Six
rotors give you redundancy — a quad carrying 1.5 kg over a street has no
response to a single motor failure other than falling.

---

## 7. Payload bay v2 — why the bag goes in a box

The §3 gimbal ring was wrong twice, and both errors came from assuming a bag is
an easy payload.

**Wrong size.** A real order — three clamshell entrees stacked plus two drinks —
needs a 280 × 240 × 260 mm bay, 17.5 L. The ring I modelled gave 180 mm of clear
diameter; it needed 196 mm of *radius*, so a 400 mm ring, which would have had
to reach past the hub on outriggers.

**Wrong assumption.** A hanging plastic bag is close to the worst payload this
vehicle could carry:

| | rigid box | hanging bag |
|---|---|---|
| downwash load (13.7 m/s wake) | contained | **28.9 N, unsteady** |
| cruise drag at 10 m/s | 6.1 N | 12.7 N |
| pitch trim | 6.6° | 13.7° |
| prop-strike risk | none | billows upward into the disc |

28.9 N is 2.95 kg — twice the payload's own weight, fluctuating, and completely
unmodeled. A bag is a light, high-drag, shape-changing bluff body sitting
directly in a 13.7 m/s wake. Wing and Zipline both carry rigid containers and
lower them on a winch specifically to keep the package out of the rotor wash.

**The fix keeps pickup simple.** A human opens the lid and drops the bag in. The
box is a rigid insulated container, so the flight dynamics stay those of a rigid
body and the §3 inertia analysis remains valid.

### New parts

| part | qty | est. PETG | role |
|---|---|---|---|
| `cardan_outer` | 1 | 73 g | U-yoke under the hub, roll axis |
| `cardan_cross` | 1 | 83 g | cross link, roll + pitch pins |
| `bay_bracket` | 1 | 143 g | pitch pivots up, flange down to box |

A compact cardan joint replaces the ring: 114 mm footprint, 90 mm stack, versus
a 400 mm ring on outriggers. Same passive self-levelling, gravity does the work,
no actuators. M6 shoulder bolts at both axes; use flanged bearings if you can,
plain bushings will do.

**Consequence to plan for: the landing gear grows to 410 mm.** Box height plus
cardan stack plus ground clearance. That is a tall, tippy vehicle — build the
gear as a wide skid frame rather than four legs, and expect to re-check the
empty CoM after the change.

Free swing before a box corner meets the gear is **28.3°**. Add hard stops in the
cardan at ±25° so the box cannot reach the gear or the arms under a gust.

### Still worth thinking about

Slosh. Drinks in a self-levelling box are better off than in a tilting one, but
liquid in a partly full container is a distributed system, not a pendulum DOF.
If you carry drinks, either constrain them upright in a cup holder or accept
that the payload dynamics have a mode your model does not contain.

---

## 8. Endurance — 60 min hover does not close

Solving the battery-mass fixed point (add battery → more mass → more power →
more battery) for 60 min hover with 1.5 kg payload:

| config | AUW | battery | hover |
|---|---|---|---|
| **current: 4 × 15", LiPo 190 Wh/kg** | — | — | **diverges, no solution** |
| 4 × 18", LiPo 190 | — | — | **diverges** |
| 4 × 22", Li-ion 250 | 8.4 kg | 3.9 kg | 777 W |
| 6 × 22", Li-ion 250 | 8.6 kg | 3.3 kg | 658 W |
| 8 × 24", Li-ion 280 (best case) | 9.0 kg | 2.5 kg | 560 W |

"Diverges" means the spiral runs away: every gram of battery costs more than it
carries, so no pack size reaches 60 minutes. The current airframe cannot get
there at any price.

The configs that do close are 8–9 kg machines with the battery at 40–46% of AUW,
on 22–24" props and high-density Li-ion at a C-rate that barely supports the
hover current. For reference, a DJI Matrice 350 gets roughly 55 min **with no
payload**. Sixty minutes of hover *with* 1.5 kg is beyond current commercial
heavy-lift multirotors.

### The reframe

**Hovering is 3.2× more expensive than cruising.** Same airframe, 60 min:

| | 5.34 kg | 7.0 kg |
|---|---|---|
| hover | 579 W | 869 W |
| cruise (L/D 8, 18 m/s) | 183 W | 240 W |

For delivery you don't need an hour of hover — you need an hour of *endurance*,
and hover is only needed for takeoff, package release, and landing. That is
about 2–3 minutes per trip.

The current 5.34 kg / 222 Wh design already cruises roughly **8 km out and back**.
Before redesigning around a much larger vehicle, worth deciding whether the
requirement is really 60 min of hover or 60 min of mission time — they lead to
completely different aircraft. If it is mission time, the answer is a hybrid
VTOL (wing for cruise, rotors for hover), which is what Wing flies for exactly
this reason.

If it genuinely must hover for an hour, budget for the 6 × 22" configuration and
accept 8.6 kg AUW, which pushes the vehicle into a much heavier regulatory and
safety category than anything discussed above.

---

## 9. Package release — winch and passive hook

The drone hovers at 15–25 m and lowers the box on a tether. It never descends
into trees, pets, parked cars, or people, and hover time is ~60 s per delivery.

### Parts

| part | qty | est. PETG | role |
|---|---|---|---|
| `winch_drum` | 1 | 81 g | tether drum, D-flat keys to gearmotor |
| `release_hook_body` | 1 | 22 g | throat holds the payload bail |
| `release_gate` | 1 | 3 g | counterweighted gate |
| `tether_guide` | 1 | 15 g | fairlead under the hub |

Plus: 2 mm Dyneema (~200 kg break, >100× payload), a geared DC motor with
encoder, and a current-sense line on the winch driver.

### How release works

Tether tension holds the gate shut against its counterweight. When the box
touches down the tether goes slack, tension drops below ~0.15 N, and the gate
rotates open under a 2.8 N·mm counterweight moment, letting the bail slide out
of the throat. The winch then retracts an empty hook.

**This fails safe by geometry.** The gate physically cannot open while loaded,
so release requires weight-off, which cannot occur at altitude. There is no
actuator at the hook, no wire running down the tether, and no software path that
drops a package in flight.

### Detecting touchdown

Winch motor current is the primary signal — it falls sharply when the load
transfers to the ground. Back that with drum encoder counts against commanded
payout, so an early snag reads as a stall rather than a touchdown. Do not rely
on the vehicle's altitude estimate; the tether is longer than the barometer is
accurate.

### Failure modes to design against

- **Snag on retract.** If the hook catches, the vehicle is now tethered to the
  ground and will pull itself down. Fit a tension limiter and an emergency
  tether cutter, and treat this as the top hazard in the flight-test plan.
- **Pendulum.** A 20 m tether with 1.5 kg on the end is a slow pendulum that
  couples into attitude. Lower and retract at ≤1 m/s and hold position tightly.
- **Downwash on the box during descent.** It leaves the strong wake within a few
  metres, but the first 2–3 m are turbulent; expect swing to build there.

### Simpler v1

Land-and-release with a servo latch on the bay is far easier to test and needs
no tether at all. Recommended for first flights — prove the airframe and the
controller before adding a mechanism whose worst failure mode is tethering the
aircraft to the ground.
