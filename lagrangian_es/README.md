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
integrator, the liveness logic, or (later) the contact model, so the method stays
gradient-free with respect to the dynamics — the property that motivates
evolutionary search on non-smooth systems in the first place.

## Quickstart

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m pytest -q                                   # 150+ tests, ~15 s
.venv/bin/python scripts/train.py --config configs/smoke.yaml   # ~1 s
.venv/bin/python scripts/train.py --config configs/quadrotor_default.yaml
.venv/bin/python scripts/ablate.py --seeds 0 1 2 3 4            # the 2x2
```

## Results

Held-out (192 unseen tasks), quadrotor + `energy_shaping`, 60 generations:

| | crash | leg-A err | leg-B err | within 25 cm |
|---|---|---|---|---|
| generation-0 prior | 69% | 0.82 m | 2.04 m | 0% |
| trained (GA)       | **0%** | 0.09 m | **0.09 m** | **98%** |
| §6 acceptance      | 0% | — | < 0.15 m | > 80% |

The prior flies badly but does not fail to fly — which is the point. If it already
succeeded, the task would be too easy and the ablation would have no dynamic range.

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

## Layout

```
src/lagrangian_es/
  config.py util.py            frozen dataclasses; pytree + RNG helpers
  systems/                     so3, base, quadrotor, planar_quad, two_link_arm
  trainables/                  base, energy_shaping, pd_baseline, mlp
  tasks.py rollout.py          goal distributions; plant-agnostic episodes
  metric.py operators.py es.py G(θ) and P; variation/selection; the loops
  evaluate.py viz.py           held-out metrics; figures
scripts/  train.py  ablate.py  replay.py
```

Dependency order is strictly acyclic and asserted by
`test_lint_seam.py::test_dependency_order_is_acyclic`.

## Visualizer

`scripts/replay.py` exports flight trajectories as JSON (position, orientation,
liveness, goals) for an interactive 3D replay that renders the airframe as an
oriented box and the waypoints as points, flying the trained genome and the
untrained prior through the same episode side by side.
