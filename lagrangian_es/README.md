# Physics-Whitened Evolution Strategies for Underactuated Control

Evolutionary control where **both the search space and the variation operator are
derived from Lagrangian mechanics** rather than chosen generically.

1. A genome decodes not to torques or network weights but to a shaped potential
   `V_d` and a PSD dissipation matrix `K_d`, giving
   `F_des = g(q) − ∇V_d(x − x_goal) − K_d·ẋ`. Because `V_d` is nonnegative with a
   unique minimum at the goal, **every** individual — including a randomly
   initialized one — is an energy-shaping controller whose closed-loop equilibrium
   is the target by construction. Evolution searches the geometry of the
   potential, not whether the controller is stable.
2. Mutation is whitened by a mechanical metric. A force perturbation `δu` costs
   energy `δuᵀM(q)⁻¹δu`; pulling that back through the controller gives
   `G(θ) = E_traj[(∂u/∂θ)ᵀ M(q)⁻¹ (∂u/∂θ)]`, and sampling `δθ ~ N(0, σ²G⁻¹)` makes
   variation isotropic in **energy** rather than in parameter coordinates.

Only the controller map is differentiated. Nothing differentiates through the
integrator, the liveness logic, or the contact model, so the method stays
gradient-free with respect to the dynamics — the property that motivates
evolutionary search on non-smooth systems in the first place.

Beyond the v2 spec, the same two seams now carry a quadruped with constraint-based
contact, drones with slung payloads on unilateral cables, obstacle fields and hoop
courses, and a sensing layer in which a camera deforms the *desired* Lagrangian
rather than feeding an estimator. Each is a composition, not a fork: one loop, one
metric, one set of operators throughout.

## Quickstart

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m pytest -q                                    # 550 tests, ~35 s

.venv/bin/python scripts/train.py --config configs/smoke.yaml    # ~1 s
.venv/bin/python scripts/train.py --config configs/quadrotor_default.yaml
.venv/bin/python scripts/ablate.py --seeds 0 1 2 3 4             # the 2x2
.venv/bin/python scripts/replay.py --out traj.json               # visualizer data
```

Configs: `smoke`, `quadrotor_default`, `ablation_2x2`, `payload_delivery`,
`quadruped`, `arm_transfer`, `maximal_chain`.

Everything composes from four registries, so a new robot, controller, scene or
sensor is a class plus a line:

```python
system  = make_system("quadrotor_nav", environment="hoop_course")
agent   = make_trainable("quadrotor_agent", system)
sensor  = make_sensor("range", system, n_beams=12)
env     = Environment([Pillars(n=6), Hoops(n=3)])
```

## The navigation prototype

Sensor-guided waypoint navigation on randomised pillar fields. The scene lives in
the state, so a different evaluation seed is a different set of layouts.

| | n | reach | crash | timeout |
|---|---|---|---|---|
| pillars, eval seed | 8192 | **0.9940** | 0.0020 | 0.0040 |
| pillars, held-out seed | 4096 | **0.9961** | 0.0010 | 0.0029 |

Reproduce with `assets/nav99_genome.json` and library defaults; `tests/test_prototype.py`
guards it. Transfer to scenes it was never tuned on — the obstacle terms consume
beams and are never handed geometry:

| sparse | gate | gate_forest | slalom | forest | cluttered | walls | hoop_* |
|---|---|---|---|---|---|---|---|
| 1.000 | 0.999 | 0.992 | 0.990 | 0.988 | 0.886 | 0.845 | 0.66–0.86 |

The hoop scenes are a **known limitation, not a tuning gap**: a repulsive barrier
pushes away from the ring it is supposed to fly through. That is an aperture
problem and needs a term that can distinguish a gap from an obstacle.

### What the number rests on

Each of these was measured, and several overturned an earlier guess of mine.

**Three terms, not one.** A potential's force depends on position alone, so it
delivers a fixed deceleration and can only arrest an approach from
`v <= sqrt(2 a d)` — about 2.3 m/s here, while 100% of collisions happened at a
mean of 3.09 m/s. Raising the barrier's gain fixes the speed limit and destroys
navigation (reach 0.21 at high gain). Only a Rayleigh term sees velocity, hence
`RangeDamper`; and `RangeVortex` is workless (`F ⊥ v`), so it redirects the
stragglers without touching the energy certificate.

**Sensing geometry.** 12 beams over 2π sit 30° apart, leaving 0.52 m gaps at 1 m
against pillars 0.36–0.76 m wide — one can hide entirely between two rays. 24
beams took crashes 0.027 → 0.014; refreshing every step rather than every fifth
took them 0.058 → 0.027.

**`goal_margin` must exceed the barrier's standoff.** At 0.30 against a `safe`
radius of 0.45, half of all waypoints sat inside the repulsion and the vehicle
was pushed off its own target: reach 0.929 for those against 0.990–1.000 for the
rest. That was the entire shortfall from 99%, and it is a well-posedness
condition rather than a difficulty setting.

**The objective needs both halves.** `dead_mode="frozen"` made a crash cheaper
than a careful arrival (3.7952 against 3.9703), so selection correctly dismantled
obstacle avoidance — the range barrier's weight went to 0.00 on two of three
seeds. `dead_cost` alone over-corrects: crashes fall to 0.085 but reach falls to
0.716, because punishing a crash is not the same as paying for an arrival. With
`dead_cost=6.0` **and** `goal_bonus=15.0`, rank correlation between fitness and
reach across a reckless-to-paralysed spectrum is +1.00.

**Selection cannot polish this further.** At a 0.1% crash rate, 32 episodes per
genome contain 0.03 crashes; resolving it would need ~1000 episodes per genome.
Training past this point reliably *degrades* it (0.9707 → 0.9395), because the
search optimises the only thing it can measure. The flight controller is trained;
the sensor constants are measured.

## Results

Held-out evaluation, 128–192 unseen tasks, 60 generations, identical search loop
throughout — only the plant and its Lagrangian terms change.

| plant | task | error before → after | failures before → after |
|---|---|---|---|
| `QuadrotorSE3` | two waypoints | 1.85 → **0.017 m** | 12% → **0%** |
| `QuadrotorSE3` + landmark camera | two waypoints | 2.09 → **0.041 m** | 78% → **0%** |
| `QuadrotorPayload` | two waypoints, slung load | 1.05 → **0.046 m** | 2% → **0%** |
| `TwoLinkArm` (minimal coords) | joint pair | 0.558 → **0.039 rad** | 0% → 0% |
| `MaximalChain` (maximal coords) | joint pair | 0.559 → **0.035 rad** | 0% → 0% |
| `PlanarQuadruped` | base pose | 0.052 → 0.062 m | 7% → 28% |

Against the spec's §6 acceptance criteria (quadrotor, GA, 60 generations):
**crash 0%, leg-B error 0.09 m (< 0.15), 98% within 25 cm (> 80%)** — all met.

Two rows deserve reading carefully:

- The **arm in both coordinate systems** starts from near-identical priors
  (0.558 / 0.559) and trains to near-identical results (0.039 / 0.035). That is
  the coordinate-invariance claim confirmed end to end, not just at a single state.
- The **quadruped does not improve.** The plant is verified independently
  (SPD mass matrix, gravity wrench recovering the legs' 2.446 N·m moment
  analytically, contact multipliers carrying full weight, feet held to 0.0000 m),
  but the wrench-to-torque allocator through a redundant contact set still needs
  work. Treat it as a working plant with an unfinished controller, not a solved
  task.

The generation-0 prior flies badly but does not fail to fly — which is the point.
If it already succeeded, the task would be too easy and the ablation would have no
dynamic range.

## Plants

| system | n_force | task_dim | allocator | `M(q)` | notes |
|---|---|---|---|---|---|
| `QuadrotorSE3` | 4 | 3 | 6 | constant | the primary plant |
| `QuadrotorPayload` | 4 | 3 | 6 | constant | + slung packages on connectors |
| `QuadrotorNav` | 4 | 3 | 6 | constant | + obstacle field in the state |
| `PlanarQuadrotor` | 2 | 2 | 2 | constant | cheap 2-D test plant |
| `PlanarQuadruped` | 8 | 3 | 4 | **varies** | 11 coords, 3 unactuated, contact |
| `TwoLinkArm` | 2 | 2 | 0 | **varies** | minimal coordinates |
| `MaximalChain` | 2 | 2 | 0 | *constant* | maximal coordinates, joints as constraints |

## The two abstractions

Everything downstream of these two ABCs is written against them and never mentions
a quadrotor. `tests/test_lint_seam.py` enforces that mechanically.

**`systems/base.py` — `LagrangianSystem`.** State is an opaque pytree
(`dict[str, Tensor]`); only the owning system interprets the keys. Downstream code
touches state only through `task_position` / `task_velocity` / `nominal_state`.

**`trainables/base.py` — `Trainable`.** Owns its genome layout and nothing else.
The genome is `[policy params | allocator params]`: the trainable owns the first
slice, the system owns the second, and neither knows the other's layout. That is
what lets one genome structure serve a fully-actuated arm and an underactuated
quadrotor.

| system | n_force | task_dim | allocator_dim | M(q) |
|---|---|---|---|---|
| `QuadrotorSE3` | 4 | 3 | 6 (`kR`,`kW`) | **constant** |
| `PlanarQuadrotor` | 2 | 2 | 2 | **constant** |
| `TwoLinkArm` | 2 | 2 | 0 (fully actuated) | **varies** |

| trainable | policy_dim | role |
|---|---|---|
| `EnergyShaping` | 39 | the proposal |
| `FixedPD` | 6 | sanity floor |
| `MLPPolicy` | 1379 | unstructured arm of the 2x2 |

`test_conformance.py` is parameterized over the full 3x3 product and asserts trace
safety under `vmap(jacrev(...))`, shape agreement, `step` purity, freeze
correctness, finiteness at `‖e‖=1e4`, and — for trainables that claim it —
equilibrium at the goal for *arbitrary* genomes, not just the trained one.

## The scope caveat, made structural

For a rigid body `M` is **constant in the body frame**. On the quadrotor the
trajectory dependence of `G` therefore enters only through `∂u/∂θ`, not through
`M(q)`. The stronger claim — that `G` tracks a configuration-dependent inertia no
single global covariance can represent — needs a plant where `M(q)` actually varies.

This is why `inv_mass(state)` is a method on the system rather than a constant.
Measured over 256 sampled states:

| system | relative spread of `M⁻¹` | `cond(M(q))` |
|---|---|---|
| `QuadrotorSE3` | 0.0000 | constant |
| `PlanarQuadrotor` | 0.0000 | constant |
| `TwoLinkArm` | 1.0824 | 2.62 … 44.26 |

The caveat is a row in a table rather than a footnote, and running the identical
ES on the arm is what upgrades the narrow claim to the strong one.

## Composable Lagrangians

A genome does not decode to one fixed potential; it decodes to a **list of terms**
contributing to a desired Lagrangian `L_d = T_d − V_d`, plus dissipation and
path constraints:

```
F_des = g(q) − M·M_d⁻¹ · ( Σᵢ ∇Vᵢ(e, v, x) + K_d·ẋ )
```

The default list — three `GoalBowl`s and one `DissipationTerm` — reproduces the
original fixed potential exactly, at the same 39 policy slots. That is the
degenerate case: `T_d = T`, no constraint terms.

**Why composition is safe.** Lagrangians add and the Euler-Lagrange operator is
linear in `L`, so a conic combination of nonnegative terms superposes forces,
`∇(ΣVᵢ) = Σ∇Vᵢ`, and preserves positive-definiteness about the goal. The
"goal is the closed-loop equilibrium" invariant survives composition for free.

| term | dim (d=3) | contributes | zero at goal |
|---|---|---|---|
| `GoalBowl` | 10 | pseudo-Huber `V`, bounded `∇V` | always |
| `DissipationTerm` | 9 | Rayleigh `R = ½vᵀK_d v` | always |
| `KineticShaping` | 9 | PSD block of `M_d` | always |
| `ObstacleBarrier` | 2 | compact repulsion from a sphere | if goal clears it |
| `JointLimitBarrier` | 2 | compact repulsion from a box | if goal clears it |

**Certificates, not class constants.** Each term publishes what it actually
promises (`psd`, `zero_at_goal`, `bounded_grad`), and `equilibrium_exact` is the
*conjunction* of its terms' promises. Adding a barrier that overlaps the goal
correctly withdraws the equilibrium claim instead of silently falsifying it —
`test_conformance` reads the property and skips, which is the honest outcome.
`equilibrium_exact_for(theta, goal)` gives the conditional answer.

**Terms are crossover units.** `segments()` exposes term boundaries so whole
terms can be swapped GP-style (`operators.segment_crossover`). This is safe
*precisely because* each term preserves the invariant alone, and it is the
structural recombination weight-level crossover cannot give you: unit 7 of one
network has nothing to do with unit 7 of another, whereas a `GoalBowl` means the
same thing in every genome.

**Kinetic shaping and the constant-M caveat.** With `M_d = M + Σ W_kW_kᵀ` the law
above has Lyapunov function `E_d = ½ẋᵀM_dẋ + V_d` with `Ė_d = −ẋᵀK_dẋ ≤ 0`. That
derivation is **exact when M is constant** — which is the case on both quadrotors.
On the arm, where `M(q)` varies, it drops the Coriolis correction the IDA-PBC
matching conditions demand, so it is an approximation there. The same
constant-vs-varying split that bounds the whitening claim bounds this one.

## Robot-specific agents

`EnergyShaping` is the composition machinery; it knows how to sum terms and
certify the result, but nothing about any particular robot. *Which* terms a robot
should have is a property of that robot, so each one declares its own:

```
Trainable                      the evolvable-object interface
  └─ EnergyShaping             composition of LagrangianTerms
       └─ EmbodiedAgent        an agent that knows its robot
            ├─ QuadrotorAgent  bowls + damping + ground-clearance barrier
            ├─ QuadrupedAgent  … + kinetic shaping + a base-pose envelope
            └─ ArmAgent        … + joint-limit barriers
```

A subclass may read its own plant's attributes — `z_floor`, `stand_height`, link
lengths — because that is exactly what the specialization buys. `supports()`
records which plants an agent is for, so the conformance sweep **skips** pairs
that were never intended rather than failing them.

The barrier is not decoration. On the drone, crashing is the untrained prior's
dominant failure, and a compactly-supported repulsion just above `z_floor` makes
clearance a property of the search space instead of something evolution must
discover by killing individuals (3 seeds, 192 held-out tasks):

| | genome | gen-0 crash | trained crash | trained leg-B |
|---|---|---|---|---|
| `energy_shaping` | 45 | 69% | 0% | 0.011 m |
| `quadrotor_agent` | 47 | **14%** | 0% | 0.018 m |

Five times fewer crashes before any learning, for two extra genome slots. The
trade is honest: final accuracy is slightly worse, because the barrier is one
more thing shaping the potential near the floor.

Priors are sized by the plant (`potential_scale`, `damping_scale`), because a
stiffness calibrated for a 0.5 kg drone produces ~0.15 N of correction on an
11 kg robot — three orders of magnitude below its own weight. It does not "fly
badly"; it fails to respond at all. The quadrotor's calibration (K = 3.00,
K_d = 1.44) is pinned by a test so it cannot drift.

## Multi-body robots: the quadruped

`PlanarQuadruped` is a sagittal-plane "robot dog": a floating trunk plus four
two-link legs.

```
q = [x, z, pitch,  (hip, knee) x 4]     11 coordinates, only 8 actuated
```

Three things make it qualitatively harder than the drone, and all three are what
the abstractions were built for:

| | quadrotor | quadruped |
|---|---|---|
| `M(q)` | constant in the body frame | dense, 11x11, strongly configuration-dependent |
| unactuated DOF | 1 (thrust direction) | 3 — the base has **no actuators at all** |
| contact | none | four feet, made and broken |

Dynamics are assembled in closed form, not by autograd: for a planar tree every
body's absolute angle is a fixed linear function of `q`, so `M`, the bias force
and every Jacobian follow analytically. That matters for the central claim —
nothing in the plant is ever handed to autograd, so the search stays gradient-free
with respect to the dynamics.

## Coupling through Lagrange constraints

Coupling is expressed as constraints on the Lagrangian rather than penalty forces
bolted onto it. For `c(q) = 0`,

```
[  M   -Jᵀ ] [ q̈ ]   [ Sᵀτ - b            ]
[  J    E  ] [ λ ] = [ -J̇q̇ - Baumgarte   ]
```

and **λ is the coupling force**. The ground reaction is a multiplier, not a
spring constant's side effect — `contact_forces(s)` returns it directly.

Two terms earn their place on the second row. `E` is a compliance block: four
feet on flat ground over-determine a planar base, so the *hard* system is
genuinely rank-deficient and would fail without it. Baumgarte stabilization pulls
position drift back, since enforcing a constraint at the acceleration level
otherwise integrates error forever. Unilateral rows are gated by a smooth
activation that scales the compliance, so a foot **releases continuously rather
than switching** — continuity matters because `allocate` is jacrev'd through
touchdown.

`holonomic.py` carries the couplings minimal coordinates cannot express: contact,
closed loops, and deliberate joint couplings (`JointCoupling`, e.g. binding
diagonal legs into a trot).

### Both coordinate formulations

Joint coupling between links needs no constraint in minimal coordinates — the
joints are already implicit in `q`. `MaximalChain` does it the other way: each
link carries its own `(x, z, θ)` and the joints *are* pin-joint constraints. Same
physics, and the comparison makes a point the project needs:

| | minimal (`TwoLinkArm`) | maximal (`MaximalChain`) |
|---|---|---|
| `M` | dense, configuration-dependent | **constant**, block-diagonal |
| velocity-product terms | Coriolis + centrifugal | **none** — each link is a free body |
| coupling | implicit in the coordinates | 2N constraint rows |

Measured agreement between the two independent implementations: gravity torques
to **3.6e-15**, constrained inverse inertia to **8.9e-9**.

So *"M(q) varies"* is a statement about **coordinates**, not about physics. The
invariant object is the constrained inverse inertia
`P = M⁻¹ − M⁻¹Jᵀ(JM⁻¹Jᵀ)⁻¹JM⁻¹`, which is what a generalized force actually sees.
Its relative spread across configurations is **1.1331 in both formulations** —
identical. That is the quantity the mechanical metric should be built from.

### What the quadruped does and does not do

**Does:** stand and hold a commanded base pose with all four feet in contact,
under constraint-based contact, with the identical ES/GA loop, metric and
operators used for the drone. The plant is verified independently — mass matrix
SPD, gravity wrench recovering the legs' +2.446 N·m moment analytically, contact
multipliers carrying the full weight, feet held to 0.0000 m.

**Does not:** walk. Gaits need a contact schedule and swing-leg tracking, which
is a separate piece of machinery this does not have.

**Honest status of the learning:** the drone's numbers do not carry over. On the
balance task the trained controller improves pose error only modestly, and the
generic `energy_shaping` arm does not improve it at all. The bottleneck is the
controller/allocator, not the plant or the search: a wrench-to-torque map through
a redundant contact set has far more scope to be badly conditioned than a
quadrotor's thrust-plus-attitude loop, and several drone-calibrated constants had
to be re-derived before the loop responded to a goal at all —

- `pos_eps`, the cost's smoothing floor, was `1e-2`, sized for metre-scale
  errors. At the quadruped's 0.1 m scale the entire task spanned 0.100→0.141 in
  cost: a nearly flat objective no amount of search would fix.
- `gravity_force` returned `(0, Mg, 0)`. The legs' mass sits off the trunk
  centreline, so equilibrium needs a real pitch moment; without it the trunk
  accelerated at several rad/s² while the controller believed it was balanced.
- the allocator's contact weight was `sigmoid(−h·k)`, which is **½ for a foot
  exactly on the ground** — the normal standing case — halving the grasp matrix
  and doubling every commanded contact force.
- `BasePose` asked for heights up to 0.40 m, exactly the legs' full reach, i.e.
  the straight-leg singularity.

All four are fixed and covered by tests. Treat the quadruped as a working plant
with a controller that still needs work, not as a solved task.

### Hybrid contact: constraint + learned residual

The constraint layer gets the physics approximately right for free, but it *is*
an approximation, in four identifiable ways: finite compliance, Baumgarte
position drift, friction lagged by one step, and no impact law at touchdown.
Those errors are systematic and state-dependent — exactly what a small learned
penalty absorbs well, and exactly what a multiplier cannot, since the multiplier
is pinned by the constraint it enforces.

So the two are complementary. Enable with `learned_contact=True`; the plant then
declares `residual_dim = 4`, the genome grows by four slots, and the residual is
initialized at **exactly zero** so the hybrid *starts* as the pure constraint
model and can only learn a correction on top of it.

**Two caveats, both structural rather than stylistic:**

- Co-evolving plant parameters against the controller's own reward is reward
  hacking. Left free, the residual will invent forces that make the task easier.
  `residual_penalty` (weight `lambda_r`) is the minimum safeguard; identifying the
  residual against reference data is the real answer.
- Residual parameters change the **plant**, not the controller map, so
  `∂u/∂θ = 0` on them — they contribute a zero block to `G`, exactly as an
  episode-level multiplier would. `test_residual_params_form_a_zero_block_in_G`
  measures it. This is why `null_mode="cap"` is the default: it treats an
  uninformative direction isotropically instead of amplifying it by `ridge^(-1/2)`.

## Connection joints: attaching things to robots

Packages, tools and tethers are attached through the same constraint layer. A
connector joins an attachment point on a carrier to a payload body on the shared
generalized velocity `w = [v | ω | v_load]`, contributing rows to one KKT solve —
so **adding a package is a change of constraints, not a change of dynamics code**.

| connector | rows | kind | rest offset |
|---|---|---|---|
| `RigidLink` | 3 | bilateral weld | at the attachment |
| `Cable` | 1 | **unilateral** — taut only | hangs at length L |
| `SpringCable` | 0 | compliant force, no multiplier | hangs at length L |

The unilateral case is the interesting one, and it is the same machinery as foot
contact for the same reason: **a rope pulls but cannot push**, so its multiplier
must be one-signed. A bilateral distance constraint would hold the package *up*
when slack — not a small modelling error but a different machine. Activation is a
smooth function of slack, so the rope engages continuously at the moment it snaps
taut, which matters because `allocate` is differentiated through it.

```python
sys = make_system("quadrotor_payload", connector="cable",
                  payload_mass=0.15, cable_length=0.45)
```

Verified in `test_connectors.py`: a taut cable holds **0.45000 m** exactly with
the drone at **1.00000 m**; a `SpringCable` stretches by `mg/k` to within 2e-4; a
weld sits at 0.00000; a slack rope lets the package fall freely instead of
pushing it; and free fall with a swinging package conserves energy to **0.15%**,
confirming the constraint is workless.

Two details that bite if you skip them:

- **`rest_offset`.** Each connector declares where its payload sits at rest, so
  `reset` and `nominal_state` start *on* the constraint manifold. Hanging a
  welded payload at cable length instead violates the weld by 0.43 m at t=0, and
  Baumgarte then yanks it into place — visibly jolting the carrier.
- **`gravity_force` returns `(m_drone + Σm_load)·g`.** A controller compensating
  only its own mass sags under the package from the first timestep, and would
  spend the entire search rediscovering a constant it was never given.

A slung load is also a genuinely hard control problem — an unactuated pendulum
hanging off the body you are trying to position — which is why `shaping_cost`
scores tilt *and* swing: "arrive fast" and "arrive with the package still" are
different objectives.

## Constraints act at three levels

Putting it together, the same idea appears three times and the distinctions matter:

| level | where λ lives | how λ is obtained | in θ? |
|---|---|---|---|
| **dynamics** | KKT system (`holonomic.py`, `connectors.py`) | *solved* — algebraic consequence of `c(q)=0` | no |
| **path** | barrier term in `V_d` | no multiplier; soft, in the potential | **yes** |
| **episode** | fitness (`constraints.py`) | *ascended* — dual step in the outer loop | no |

Only the path-level constraint belongs in the genome, because only it changes the
commanded force. The other two are searched by nothing and solved or ascended
instead.

### The two multiplier levels in detail

**Path level** — a barrier term folded into `V_d`. It changes the commanded force,
so it lives in `θ` and is whitened by `G` along with everything else.

**Episode level** — attached to fitness, `L(θ,λ) = J(θ) + Σᵢ λᵢ(cᵢ(θ) − budgetᵢ)`.
These change only selection pressure.

**λ must not go inside θ.** A multiplier affects fitness, not the wrench, so
`∂u/∂λ = 0` identically and λ contributes zero rows *and columns* to
`G(θ) = E[(∂u/∂θ)ᵀM⁻¹(∂u/∂θ)]`. Put λ in the genome and the metric is exactly
singular there, with the ridge silently doing all the work in the very
coordinates meant to be searched — and it is silent, because the ridge keeps
`eigh` well posed so nothing errors.
`test_constraints.py::test_a_multiplier_inside_theta_would_zero_a_block_of_G`
builds that wrong design on purpose and measures the zero block rather than
asserting the claim in a comment.

So multipliers are updated by dual ascent in the **outer loop**. Expect
primal-dual oscillation: the multiplier keeps integrating through the dead time
in which the population has already complied, overshoots, and rings.
`PIDMultiplier` (Stooke et al. 2020) is the standard fix — classic dual ascent is
exactly its integral term, so `kp = kd = 0` recovers it bit-for-bit, and the
proportional term is what damps the ringing. In the repo's lagged-loop test it
cuts overshoot from 28% to 15%.

```python
cs = ConstraintSet([CrashBudget(0.0), SaturationBudget(0.02)], multiplier="pid")
train(cfg, system, trainable, task, constraints=cs)   # theta's dimension is unchanged
```

## Physical environments

`environments/` is a **separate layer from `systems/`**, and the separation is
enforced: a plant is a set of equations of motion, a scene is the world those
equations move through, and `environments/` imports neither `systems/` nor
anything above it (`test_lint_seam.py::test_environments_never_import_a_robot`).
So a robot drops into any scene and a scene is reused by any robot.

```
environments/
  base.py        ObstacleGroup ABC, Environment, Mixture, the SDF marcher
  primitives.py  Pillars, Walls, Gate, Hoops
  cad.py         DXF import -> StaticWalls / StaticPillars
  __init__.py    registries, presets, mixtures, make_environment, load_dxf
```

A scene is a **list of obstacle groups**, exactly as a controller is a list of
Lagrangian terms and a robot is a list of connectors:

```python
env = Environment([Pillars(n=6), Walls(n=2)])
env = make_environment("forest")            # or a named preset
sys = make_system("quadrotor_nav", environment="hoop_course")
```

| group | shape | you must |
|---|---|---|
| `Pillars` | vertical cylinders | go **around** — not over |
| `Walls` | finite wall segments | go around or over |
| `Gate` | two posts with a gap | plan a route, not a heading |
| `Hoops` | rings (torus) | fly **through** |

Presets: `empty`, `sparse`, `pillars`, `forest`, `slalom`, `gate`, `walls`,
`cluttered`, `gate_forest`, `hoops`, `hoop_course`, `hoops_upright`,
`hoops_flat`, `hoop_slalom`.

**Adding a primitive is one class.** `ObstacleGroup.raycast` defaults to sphere
marching the group's own SDF, so a new shape only has to implement `sdf`
(and optionally `normal`). `Pillars` and `Walls` override it with closed-form
intersections and are exact and faster; `Hoops` inherits the marcher and gets
correct ranges for free — verified by a test asserting `Hoops` defines no
`raycast` of its own.

**Geometry lives in the state, one layout per episode.** That is what makes
generalization *measurable* rather than assumed: `reset` draws a fresh scene from
its generator, so a held-out evaluation seed is automatically a held-out set of
layouts, with no separate bookkeeping to get wrong and no way to silently test on
the training scenes.

### Hoops are gates, not decoration

Hoop tilt is sampled across the full range — vertical (fly through sideways),
horizontal (fly up through), and everything between — so a course cannot be
solved with one approach direction. `hoops_upright` and `hoops_flat` pin the
extremes.

The waypoints **are** the hoop centres. Geometry and goals are drawn from
different generators, so a gate can only be guaranteed to sit on the route if one
is derived from the other: `place_course` puts hoop *k* on waypoint *k*, facing
along the leg that reaches it, keeping the tilt the group already sampled. A gate
the vehicle is not required to pass through is not a gate.

### Importing CAD drawings

A 2-D DXF floor plan is almost exactly the geometry this already models — the
obstacle primitives are vertical extrusions, so a plan's walls and columns become
`StaticWalls` and `StaticPillars` with no approximation beyond extruding upward.

```python
env = load_dxf("floor.dxf", fit=8.0)          # rescale so the plan spans 8 m
sys = make_system("quadrotor_nav", env=env, free_start=True)
task = make_task("free_space", sys, n_legs=2)
```

| DXF entity | becomes |
|---|---|
| `LINE` | one wall segment |
| `LWPOLYLINE`, `POLYLINE` | a wall segment per span, closed loops included |
| `CIRCLE` | a pillar |
| `ARC` | segments along the arc |
| `INSERT` | recursed into, so blocks are not silently dropped |

`ezdxf` is used when installed, with a built-in ASCII reader covering the same
entity types otherwise, so importing a plan is never a hard dependency.

`fit` rescales so the drawing's largest dimension spans that many metres — which
is what lets a plan drawn in millimetres be used without the caller knowing its
units. Blocks matter more than they look: `INSERT` is everywhere in real drawings,
and importing them as nothing yields a convincingly empty building.

**Imported geometry never moves.** `clear_points` is a deliberate no-op on the
static groups: a random obstacle field may be nudged aside to keep a waypoint
reachable, but a building that dodges the drone is not that building any more. So
the reverse applies — the *waypoints* move instead:

- `Environment.free_points` rejection-samples positions with a required clearance,
  and raises with a diagnosis (wrong scale, or a margin wider than every corridor)
  rather than returning an empty tensor;
- the `free_space` task draws waypoints from that pool;
- `free_start=True` starts episodes there too, because a drawing's origin is very
  often inside a wall and every episode would otherwise begin in collision.

### Does obstacle navigation generalize?

Trained on `pillars`, evaluated on held-out **layouts** of that preset and on five
presets never seen in training. Two arms matched on seed, population and budget:
*blind* is the standard genome with no sensor at all; *ranged* adds a
`RangeSensor` and a `RangeBarrier` that sees only beams and is never told where
anything is. 3 seeds, 192 held-out tasks each, 60 generations.

| test scene | blind fail | ranged fail | blind reach | ranged reach |
|---|---|---|---|---|
| `pillars` *(trained here)* | 0.29 | **0.18** | 0.38 | 0.36 |
| `sparse` | 0.17 | **0.10** | 0.46 | 0.38 |
| `forest` | 0.34 | **0.24** | 0.36 | 0.34 |
| `gate` | 0.14 | **0.06** | 0.46 | 0.42 |
| `walls` | 0.21 | **0.14** | 0.43 | 0.37 |
| `cluttered` | 0.31 | **0.22** | 0.38 | 0.34 |

**What holds up.** Range sensing cuts collisions in *every* scene, by roughly a
third relative, and the benefit transfers intact to five presets — including wall
segments and gates, geometry the controller never trained against. That is the
payoff of a barrier built on measurements rather than on known geometry.

**Why there is no train/test gap.** Performance tracks scene *difficulty*, not
whether the scene was trained on: the training preset is among the harder rows,
and easier unseen scenes (`gate`, `sparse`) score better than it. Layouts are
resampled every episode and every generation, so the controller never sees the
same scene twice and has nothing to memorize — generalization is a property of the
training distribution here, not an achievement of the controller.

**What does not hold up.** The absolute numbers are poor. Reaching the goal
34–46% of the time while colliding 6–24% of the time is a vehicle that navigates
*somewhat*, not a competent navigator. Two contributors are worth separating:

- 2.6–8% of sampled waypoints land inside geometry and are simply unreachable —
  the task samples goals independently of obstacles. Real, but far too small to
  explain the gap.
- The rest is the controller. A reactive barrier on 12 beams has no memory and no
  plan; it cannot back out of a pocket or route around a wall it is already
  pressed against. Getting past this needs a policy with state, not a better
  barrier — which is a different piece of work, not a longer training run.

The honest summary: **the sensing benefit generalizes; the navigation competence
is not there yet.**

### Sensing the scene

`RangeSensor` fans horizontal beams and reports distances, so avoidance can come
from **measurements** rather than known geometry — `RangeBarrier` is handed beams
and is never told where anything is, which is the difference that makes transfer
to an unseen layout meaningful. `ObstacleBarrier`, by contrast, is given geometry
at construction and cannot transfer.

Rays are 3-D throughout even for planar primitives. Letting flat shapes set a 2-D
convention is exactly what crashed the first hoop added to a shared scene: a
torus's distance genuinely depends on height. Vertical primitives report an
*exactly zero* height gradient — the truth — rather than a fudged small value the
pullback would then act on.

## Sensing (spec addendum §3.5)

A camera does not change the plant Lagrangian — `M(q)q̈ + Cq̇ + g = u` is
indifferent to what we measure — so **`M⁻¹` in `G(θ)` stays a ground-truth,
design-time quantity and no estimator ever enters the metric**. `metric.py`,
`operators.py` and `es.py` are untouched. What sensing changes is the *desired*
Lagrangian: `L_d = L₀(θ) + ΔL(θ, obs)`.

### The regression gate

With `FullState` and `latency_steps = 0`, training, evaluation and traces
reproduce **bit-identically** to the sensor-free path
(`test_full_state_training_is_bit_identical`). The sensor-free path also still
uses the literal 3-argument `vmap`, so "no sensors" is unchanged rather than
merely equivalent.

### Delay and common random numbers, built before any real sensor

Both are cheap now and expensive to retrofit:

- **Per-sensor `DelayBuffer`**, not one global lag — flow/IMU at ~2 ms, ToF at
  5–20 ms, vision at 30–80 ms. Delay costs `ω·τ` of phase margin; evolve without
  it and evolution finds stiff, lightly-damped potentials that are optimal in sim
  and oscillate on hardware.
- **Sensor noise joins CRN** — drawn once per episode and tiled across the
  population, exactly like goals and reset noise. Otherwise sensor stochasticity
  becomes fitness-ranking variance, and ES is already variance-limited.

### `ΔL` as a `LagrangianTerm`

Partitioned **by argument kind**: `SensorPotential` takes `position_like`/`range`,
`SensorDissipation` takes `velocity_like`. Never one net across both — that
produces forces with no sign guarantee, gyroscopic terms that are neither
conservative nor dissipative.

`ΔV ≥ 0` and `ΔV ≡ 0` for `‖e‖ < r_goal`, enforced by a smoothstep gate that is
*exactly* zero inside the ball (value **and** gradient). That is what replaces
unbiasedness:

> **Bias immunity, measured.** Inject a constant **+60 pixel** bias into the
> observation and the closed-loop equilibrium is still the goal to **< 1e-3 m**.
> An estimator-based controller settles wherever its bias vanishes; `ΔV` cannot
> move the goal because it has no support there.

`FovBarrier` keeps features off the frame border — image-based servoing's
best-known failure mode — and is compactly supported in pixel space, so a
centred feature contributes nothing.

### Lenses and the landmark camera

`Pinhole` and `DoubleSphere` (Usenko et al. 2018) round-trip to **2e-16** with
closed-form Jacobians matching central differences to **5e-8**. The fisheye sees
1233 of 2000 random directions where the pinhole sees 501 — wide FOV makes the
barrier less binding.

`LandmarkCamera` **projects, never renders**: 192 vehicles × 250 steps would be
48,000 images per generation. Trained end-to-end with the camera and barrier in
the loop: error **2.092 → 0.041 m**, crashes **78% → 0%**.

**Runtime deviates from the spec's target.** §7 asks for <15% at `B=192, K=32`;
measured it is ~15% at K=4 and ~45% at K=32. The cost is per-op *dispatch*, not
arithmetic — the ~12 tensor ops per step are paid whatever K is — so the lever is
K or fusing the projection, not batch size. The test asserts the level actually
achieved rather than a target that is not met.

### Not built

Steps 4–5 of the addendum's build order (adaptive skipping with `skip_thresh` and
`heartbeat` in the genome; `EventCamera`) are deliberately absent, per the scope
warning: past step 3 this stops being an evolutionary operator for actuator
control and becomes a vision-based control system. The `Sensor` ABC keeps the
door open at near-zero cost.

## Search strategies

Two loops over the same genome and the same whitened variation operator, so the
whitening claim can be tested under either rather than being entangled with one
population model:

- **`strategy: es`** — distribution-based. One moving mean, mirrored sampling,
  rank-weighted recombination. This is the loop the ablation runs.
- **`strategy: ga`** — a persistent population with tournament selection, elitism,
  and BLX-α crossover performed **in the whitened frame** (plain coordinate-wise
  BLX would draw its box along parameter axes, reintroducing exactly the
  coordinate dependence the metric exists to remove). Converges faster at small
  budgets; see the results table.

With `P = I` both reduce exactly to their textbook isotropic forms — the arms flow
through the same code path and differ only in `P`, which is what makes the
comparison controlled.

## Diagnostics for the silent failures

Three of §7's failure modes make the ablation look like a null result when it is
actually a bug. All three are logged every metric refresh:

- **`metric_dist_I`** — `‖P − I‖_F`. Near zero means the ridge dominates and the
  whitened arm has quietly *become* the baseline: you are comparing an arm to
  itself. Typical healthy value here is ≈ 5.2 with `cond(G) ≈ 10⁴`.
- **`metric_sat`** — fraction of sampled states at an actuator bound. `jacrev`
  through a saturated `clamp` returns zero, so a genome living at `f_max` yields a
  rank-deficient `G` in exactly the directions that matter, and the ridge hides it.
- **`th_timescale_sep`** — `ω_att / ω_pos`. Falling means the vehicle commands
  tilts it cannot achieve; the symptom is a crash rate that rises mid-training.

`EnergyShaping.describe()` also returns the spectrum of `Σ w_k² A_kᵀA_k`, so you can
see whether evolution uses the extra bowls or whether `NK = 1` would do as well.
At the prior the three bowls are identical and their Jacobian blocks are
bit-identical, so `G` is genuinely rank-19 of 45; symmetry-breaking lifts it to ~33.

## Known deviations from the v2 spec

- **`n_eps = 8`, not 4, for the 60-generation run.** At 4 episodes the fitness
  estimate is noisy enough that leg-B error plateaus near 0.17 m. Eight episodes
  clears the 0.15 m bar and costs almost nothing: rollouts are
  Python-overhead-bound, not batch-size-bound (B=32 → 0.074 s, B=192 → 0.087 s).
- **`MLPPolicy` defaults to 2×32, not 2×64.** The spec says "2×64" in prose but
  "~1–3k parameters" in the table; 64 units give 4,803. The default lands in the
  stated band and keeps the metric's O(dim³) `eigh` from making the 2×2
  budget-mismatched in wall clock. `hidden=64` is one argument away.
- **Step-size adaptation uses a within-generation 1/5th rule.** Goals are
  resampled every generation, so "did the elite beat its own best-ever score" is
  confounded by task difficulty and shrinks σ monotonically regardless of progress.
  Both loops instead compare against a reference genome scored on the *same* goals
  — the ES arm appends the parent to the batch, the GA re-evaluates its elites —
  which costs no extra rollout because batch size is nearly free.
- **Generation-0 is ~69–78% crashes, not ~90%,** with leg-A error ~0.82 m rather
  than ~0.5 m. Same regime and ample dynamic range (0% success), but the exact
  plant constants behind the prototype's numbers were not recoverable.
- **Runtime is ~3.4× faster than the spec's budget** (250-step rollout at B=192:
  0.087 s vs ~0.3 s), so the refactor-regression gate passes comfortably.
- **`null_mode="cap"` is the default, not a bare ridge.** With `ridge=1e-3` the
  whitened arm was measurably *worse* than isotropic — §7's first failure mode, but
  in the opposite direction from how the spec describes it. See below.
- **`LandmarkCamera` runtime misses §7's <15% target** (~15% at K=4, ~45% at
  K=32). The cost is per-op dispatch, not arithmetic, so the lever is K or fusing
  the projection. The test asserts what is actually achieved rather than a target
  that is not met.
- **Sensing steps 4–5 are deliberately absent** (adaptive skipping in the genome;
  `EventCamera`), per the addendum's own scope warning: past step 3 this stops
  being an evolutionary operator for actuator control and becomes a vision-based
  control system.
- **The quadruped's controller is unfinished.** Several drone-calibrated constants
  had to be re-derived before its loop responded to a goal at all — the cost's
  smoothing floor `pos_eps` (sized for metre-scale errors, flattening a 0.1 m
  task), `gravity_force` ignoring the legs' pitch moment, an allocator contact
  weight of ½ for a foot exactly on the ground, and goals at the straight-leg
  singularity. All four are fixed and tested; the wrench allocator still is not.

### What the ablation found

The first 2×2 said whitening **hurt** on 0/5 seeds. That was a defaults bug, not a
property of the method: `ridge=1e-3` amplified `G`'s null space (rank ≈ 0.73 of 45)
by up to 31×, spending the step budget on directions that barely move the
commanded force. A sign-flipped control (`G^{+1/2}`) was catastrophic (leg-B 2.32),
confirming the direction was right.

With `null_mode="cap"` — whitening applied one-sidedly, so an uninformative
direction gets an isotropic step rather than an amplified one — the paired
difference flips:

| genome | paired Δ leg-B (whitened − isotropic) | helped on |
|---|---|---|
| structured | **−0.0045** | **5/5 seeds** |
| unstructured (MLP) | +0.125 (sd 0.24) | 2/5 seeds |

So whitening helps the structured genome consistently and shows no clear effect on
the unstructured one — which is a sharper result than "whitening helps", and the
kind the 2×2 exists to produce.

Worth noting how the bug presented: §7 warns that a *large* ridge collapses `P → I`
so the whitened arm silently becomes the baseline. The damage here came from the
ridge being too **small**. Both diagnostics (`metric_dist_I`, `metric_cond`) were
logged and healthy throughout — the metric was genuinely anisotropic, it was just
anisotropic in useless directions.

## Layout

```
src/lagrangian_es/
  config.py  util.py           frozen dataclasses; pytree + RNG helpers
  systems/
    base.py so3.py             LagrangianSystem ABC; batched SO(3)
    quadrotor.py  payload.py   SE(3) drone; + slung packages
    quadrotor_nav.py           + obstacle field carried in the state
    planar_quad.py             cheap 2-D drone
    planar_quadruped.py        11-coord legged robot with contact
    two_link_arm.py            minimal coordinates
    maximal_chain.py           maximal coordinates, joints as constraints
    holonomic.py               constraint rows + the KKT solve
    connectors.py              welds, cables, compliant links
    environment.py             composable obstacle groups
  trainables/
    base.py energy_shaping.py  Trainable ABC; composition of terms
    terms.py sensor_terms.py   LagrangianTerm library; ΔL from observations
    embodied.py                robot-specific agents
    pd_baseline.py mlp.py      sanity floor; unstructured arm of the 2×2
  sensors/
    base.py full_state.py      Sensor ABC + DelayBuffer; the identity gate
    lens.py landmarks.py       pinhole/fisheye; project-never-render camera
    range_sensor.py            beam ranging against the environment
  tasks.py  rollout.py         goal distributions; plant-agnostic episodes
  constraints.py               episode-level budgets + multipliers
  metric.py operators.py es.py G(θ) and P; variation/selection; the loops
  evaluate.py viz.py           held-out metrics; figures
scripts/  train.py  ablate.py  replay.py
```

Registries: `SYSTEMS`, `TRAINABLES`, `TERMS`, `SENSORS`, `CONNECTORS`, `GROUPS`,
`PRESETS` (scenes), `TASKS`, `CONSTRAINTS`, `MULTIPLIERS`, `LENSES`.

Dependency order is strictly acyclic and asserted by
`test_lint_seam.py::test_dependency_order_is_acyclic`.

## Visualizer

`scripts/replay.py` trains, then exports trajectories as JSON for an interactive
replay: trained genome and untrained prior flying the identical episode.

Nothing in the renderer is plant-specific. Each system publishes

- `render_spec()` — dimensionality, body shapes, ground plane;
- `render_poses(s)` — `[n_bodies, 12]` (position + rotation) in 3-D, or
  `[n_bodies, 3]` (`x, z, angle`) in 2-D;
- `render_static(s)` / `render_extras(s)` — scene geometry, cables, feet, beams;

and the page knows only that vocabulary, so **a new robot becomes drawable by
describing itself**. The target is drawn by pushing the goal through the plant's
own `nominal_state` and rendering it, so a ghost of the target configuration
comes out for free on every plant.

Plant-specific detail still reaches the screen: cable lines for a slung load,
foot markers that swell with contact force (a Lagrange multiplier read from the
state, not a spring), pillars and walls and tilted hoops, and a live camera panel
showing which landmarks are in frame against the FOV barrier's margin band.

```bash
.venv/bin/python scripts/replay.py --robots quadrotor quadrotor_hoops --out t.json
```

Training defaults to the drone family; the quadruped and arms stay registered and
selectable by name.
