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
