# Experiment log: the composer

A chronological record of every composer training arm run in this project,
what each measured, what broke, and what the measurement changed. Companion to
[`scripts/composer/README.md`](../scripts/composer/README.md), which describes
the architecture as it stands today; this file is the trail that got there —
useful when a result looks wrong and the question is "have we seen this
before," or when a fix needs re-justifying.

Dates are 2026. Arrival/reach numbers are fractions unless stated otherwise.
"Judge" means the held-out scoring pass (a fixed task set, never trained on);
"batch"/"training" means the rollout the update itself was computed from —
the two are read separately throughout because they move independently more
than once below.

---

## Phase 0 — the plant has to fly before anything is layered on it

Before any composer existed, the frozen low level itself had to clear an empty
map. It didn't, twice, for reasons that had nothing to do with learning:

- **No thrust envelope.** The allocator asked for 24.6 N against an actuator
  limit of 10.8 N and pointed body-z 78–88° off the force it could actually
  produce; 30 of 32 fresh genomes crashed on an *empty* map. Fixed by clamping
  `F_z` into `[0.1 f_max, f_max]` and scaling the horizontal component to fit
  under `|F| ≤ f_max`.
- **A deliberately marginal attitude prior** (the old prior was tuned to be
  right at the edge of stability, "to make sure it doesn't crash" via
  perturbation rather than by flying well). Replaced with a prior derived from
  the plant's own actuator bandwidth.

Result: empty-map crash rate 30/32 → 1/32; the fresh pair's judged score on
the full city went from 0.000/1.000 (reach/crash) to 0.133/0.633 with zero
training. **Lesson that recurred constantly afterward: if an untrained
controller can't fly an empty map, the bug is in the plant, not in whatever
learner sits on top of it — test the bare low level alone first.**

A second plant-level fix followed once flights were reaching the city: crashes
were 100% into buildings at 10.3 m/s, 1.5 s after takeoff, with sensor
blinding (fans, camera) changing *nothing* (cost 198.08 vs 198.15 blind). A 6 m
beam at 10 m/s is 0.6 s of warning — not a sensing problem, a speed problem.
Fixed with a plant-level airspeed limit (`speed_limit=5.0`), not a loss
penalty. Related: the learned braking term had independently saturated to
zero on every beam (a softplus stuck at its floor), invisible until a force
diagnostic on crashing flights found it; reset to its init and given its own
term (`split_terms=True`, PULL and BRAKE evolved and prioritized separately).

Also found in this phase: `yaw_mode="learned"` was required for any heading
behavior at all — without it the allocator's look-at blend was dead code and
the drone never rotated, full stop, in every prior config. And turning itself
was made an implicit property of the Lagrangian rather than a rule: yaw torque
is `-∂V/∂ψ`, the potential's own gradient through the body-fixed beams,
verified against a finite-difference ray-cast (median relative error < 1e-4).

---

## Phase 1 — a token composer, discrete vocabulary (`cotrain_v5`–`v9`, Sept 8–9)

**Design (user-directed):** "Think of it as an LLM" — one *action token* per
report (`HOLD` / `PLACE(bearing, range)` / `RAISE`/`LOWER` (term priority) /
`TURN` / `LOOK`), categorical PPO, nothing hand-tuned in the decision itself.
`HOLD` is silence; the prior was built so continuing straight at full reach is
the default (~80% HOLD / ~11% straight-PLACE at init) — the identity action is
the straight line, matching the untrained low level's own behavior.

**What had to be fixed before the batch cost was informative at all:**

- **Report stream ended before the cost settled.** `returns_from_stream` read
  the stream's cumulative cost, but the last report landed before a crashed
  row's remaining death charge, a still-flying row's hover charge, and the
  arrival bonus were applied. A half-batch gradient-agreement check (cosine
  +0.85) showed the update was *not* noise, yet the judged cost never moved —
  the target itself was wrong, not the estimator. Fixed with a closing token
  at the settled cost.
- **Distance-to-active-waypoint, not distance-to-go.** On a multi-leg tour,
  finishing a leg jumped the position charge from ~0.3 m to ~20 m for every
  remaining second, so parking 0.28 m short of the first waypoint beat
  finishing the tour by ~50 cost units. Fixed by charging distance to the
  active waypoint *plus* the remaining legs' lengths, continuous through an
  arrival.
- **Death cheaper than living.** At `dead_cost=6/s` against a per-second
  living charge of up to ~11 (distance-to-go on 15–20 m legs), crashing
  *early* was the cheapest outcome available. Raised to 40/s — above the
  largest distance-to-go a living drone can owe.
- **Re-forking the worker pool every iteration** killed a worker at iteration
  2 (`BrokenProcessPool`) once the parent had run anything multithreaded
  (judge, PPO) — forking a process with a live thread pool is unsafe. Fixed:
  fork the pool once at startup while single-threaded; workers reload weights
  by file mtime instead of being re-forked.

**Result, once those were fixed:** a bandit-only positive control
(`speak_update`, one log-prob × one paired advantage on the type decision
alone) correctly drove `speak` 0.370 → 0.001 in 13 iterations, i.e. it
*learned to say nothing*, with training arrival rising only because emitting
nothing is cheaper. That is a working estimator, not a working composer — see
Phase "waypoints are direction-blind," next.

---

## Phase 1b — the action space turned out to be a 2-armed bandit, not a spatial one

Measured directly (256 paired tasks, forced `WAYPOINT` at three bearings
spanning 180°, one identical seed each):

| condition | arrival |
| --- | --- |
| muted (no tokens) | 0.539 |
| forced waypoint, 0° | 0.285 |
| forced waypoint, +90° | 0.281 |
| forced waypoint, +180° | 0.285 |

Emitting *anything* cost ~0.25 arrival, and three bearings spanning half the
compass did **identical** damage to three decimal places. **Where** a subgoal
went carried no signal; **whether** one was emitted carried all of it — the
action space was effectively binary, and any objective trained only on the
placement arguments was training the one thing that didn't matter.

A follow-up investigation found *why* geometry couldn't help yet: a start
geometry trace showed 55% of judge starts have a building within 10 m on the
straight line to the goal, and 100% of those crash; the exploration noise on
subgoal placement was ~0.5 m (σ 0.05 × 10 m reach) against a detour that needs
several meters — the policy gradient could never see a working detour to
reinforce it. Separately, a per-subgoal charge (3.0) forbade the short, slow
subgoals that would have let the beams matter at all.

This diagnosis chain — silence-as-optimum, then unreachable good behavior
under the exploration scale — is the same shape the project ran into twice
more, at larger scale, in Phases 4 and 5 below.

---

## Phase 2 — continuous arguments (`cotrain_v10`, Sept 9–10)

**Design (user-directed):** "make the tokens continuous... I still want
`[waypoint(x,y,z)] [EOS]`... maybe `waypoint(r, theta)`." Five token types
(`EOS`, `WAYPOINT(r,θ)`, `TURN(θ)`, `PRIORITY(w)`, `LOOK`) replace the
25-component discrete grid; each carries a diagonal-Gaussian argument over an
unsquashed real, squashed by `tanh` when the action is built (so the density
stays exact with no Jacobian correction needed).

Three defects, each blocking in a way that looked like a different failure
until isolated:

1. **The identity had to be a large detour, not the goal.** An early
   parameterization biased the range mean near +1 and measured bearing from
   the *nose*, so the untrained policy's default action was "go straight off
   the nose at half reach" — nothing arrived (reach 0.000 vs 0.229 for no
   composer at all). This is the direct predecessor of Phase 3's fix.
2. **Units mismatch.** `goal_ego` was in reach units; a distance in meters was
   compared against it directly, putting every commanded waypoint ~2 m from
   the vehicle instead of at its intended reach. The vehicle crawled and
   stalled.
3. **A silent record-dropping bug.** `parallel._work` rebuilds worker records
   from a fixed field list; new continuous-argument fields (`u`, `n_args`,
   `mu`, `log_std`) weren't in it, so the update saw zero samples and trained
   on nothing while looking like it ran.

**Control:** the discrete-token composer (`cotrain_v9`) was flat over 50
updates at reach 0.28 / crash 0.72 under the same frozen low level and tasks —
the comparison this phase's continuous arm was measured against.

---

## Phase 3 — a real PPO signal (`cotrain_v11`, `LOSS=ppo`, Sept 11–12): ten defects

The single densest phase. The composer's continuous PPO update was rewritten
as `ppo_update_cont`, and it took **ten** separate defects — each invisible
until the previous one was fixed — before arrival showed a real trend at all.
In rough order of when each was found:

1. **Bearing measured from the nose.** The identity action was unreachable —
   an untrained net arrived 0.000. Fixed by measuring bearing *from the
   goal*: `theta = bearing(g_ego) + π·tanh(a₁)`, so `a₁ = 0` means "fly
   straight at the goal." A student distilled onto a 0.576 teacher under the
   old (nose-relative) frame still hit RMS ~33° of bearing error and arrived
   0.000 on all 213 flights — nearly all network capacity had gone to
   reproducing `atan2(g_y, g_x)` at every decision, none left for routing.
   Verified: fresh untrained net now arrives 0.549 vs muted 0.555 (parity).
2. **The range formula put the origin at half radius.** `r = (a₀+1)/2 · radius`
   meant a mandatory router (see `route_only` below) began every flight with
   a harmful placement it had no way to decline. Fixed:
   `r = (1 + min(0, a₀)) · radius`, so `a₀ = 0` is full reach.
3. **Flight-level advantage, zero within-flight variance.** Under "make it
   simple," a single log-prob × one paired advantage per flight was tried —
   this has *exactly zero* variance within a flight by construction, so a
   router deciding ~60 times per flight got no distinguishing signal between
   its own decisions. Fixed: per-decision returns from the cost stream, minus
   a learned state baseline `V(s)`, monitored via explained variance (`EV`).
4. **Importance ratio scored at the wrong width.** Actions were drawn at
   `log_std × temperature` but scored at bare `log_std`; at temperature 8 this
   is a 64× variance mismatch. Measured consequence: KL 100.97 against a 0.02
   cap, 97% of samples clipped, policy loss ~3×10⁵ — which then starved the
   critic entirely, since gradient-norm clipping rescales the *whole*
   gradient and left the critic ~3×10⁻⁶ of it (`EV` sat at −0.006).
5. Off-policy selection (`var_temp`) mixed with raw temperature-8 draws flown
   directly, compounding (4).
6. A KL cap was *measured* every update but never *enforced*.
7. Once enforced, it was checked against the wrong reference — the pipeline's
   stale starting point rather than this update's own drift — which froze the
   policy solid.
8. A signed log-ratio was used as a KL estimate, making the trust region
   one-sided.
9. **A per-placement charge on a mandatory action.** Under `route_only`
   (waypoint forced every report — see below), the composer cannot decline to
   place a subgoal, so a nonzero `SUBGOAL_COST` taxed something it could not
   opt out of. Zeroed for this arm.
10. **Crash luck dominated the advantage** (z = +331 for crash/survive alone).
    Fixed with a per-decision common-random-numbers baseline read from a paired
    muted control flown on the identical task and seed.

**Also found in this phase, orthogonal to the ten above:** `arg_w2` (the
continuous-argument output layer) was initialized to *exactly* zero, so the
policy would start at the hand-built prior in the bias term. But
`∂μ/∂arg_w1 ∝ arg_w2`, so at exactly zero the gradient to the input weights,
the shared trunk, and the entire scene encoder was *also* exactly zero. The
argument head was a constant for the whole run (`r` had sd 0.0088 across 6,945
decisions); blinding every beam and camera pixel changed the waypoint
arguments by 0.0000. Fixed with a small nonzero init (`std = 0.05·(1/H)^0.5`),
sized on realistic tokens after a naive scale choice overshot 10× and put the
bearing prior 88° off. **General form: when a head looks dead, compare its
weights against a fresh initialization, not against zero — "small" and "never
moved" look identical otherwise, and check whether any layer multiplies the
gradient of everything beneath it by a quantity that can be zero.**

Result after all ten: training arrival 0.368 → 0.418 over 42 uninterrupted
iterations, trend t = +4.43 (a stalled comparison run sat at t = 0.79 over the
same span); `EV` +0.005 → +0.410 (t = +11.89). **Not yet established** at this
point: the judge was only +1.39 SE over silence, and the sensor-use metric
(`sens`, not yet reliable — see Phase 5) read ~0.003.

---

## Phase 4 — beating silence, and finding out why it was geometry (CBD, Sept 12)

**The blocker:** disjoint halves of *one* task draw agreed on gradient
direction at cosine +0.46; two *independent* draws agreed at **−0.45**.
Consecutive updates were undoing each other — 90 updates, each moving the
policy by |Δμ| ≈ 0.024, netted only 0.018 of total drift. More task draws per
*rollout* didn't fit in 16 GB (batches of 4,608 and 2,304 episodes were both
killed before completing one iteration).

**Fix:** gradient accumulation across task draws (`LES_ACCUM=N`). The policy
is held fixed across N rollouts, the gradient is summed, and the optimizer
steps once — the same effective batch size, traded from memory into
wall-clock. Because the policy provably does not move during the window,
every one of the N batches is exactly on-policy (ratio ≡ 1, clipping inert).
At `LES_ACCUM=4`: arrival sat flat at 0.42 for iterations 1–4 (the window),
then jumped to 0.506 the instant the first accumulated step landed.

**Result — the first configuration in this project to beat silence with
significance:**

held-out judge 0.277 → 0.348 → 0.379 → 0.426 → 0.441 against a silence
baseline of 0.2692 ± 0.0045 (**+5.54 SE**, trend t = +7.77 across the five
judge points); training arrival 0.411 → 0.636 over 42 uninterrupted
iterations (t = +13.20).

**What the win was actually made of, once measured properly:** `sens` (the
sensor-dependence metric) read 1.6% of the placement's own spread at both 25%
and 100% building density. The composer won by placing *more* and
*better-sited* subgoals (64 → 102 a flight) — geometry, not perception. This
particular `sens` reading was later retracted as unmeasured once the metric's
own noise floor was characterized (Phase 5) — but the qualitative point it
was pointing at held up under later, trustworthy measurement: on a greedily
navigable map, geometry alone suffices, so a correct learner has no reason to
pay for perception. Testing whether it *can* use its sensors needed a map
where geometry alone fails — the `occluded` map, next phase.

A parallel finding, `corridors` (a Manhattan grid, also greedily navigable):
the composer solved it by staging subgoals along the already-clear grid, and
`sens` fell rather than rose — same story, different map.

A hand-built, non-learned router (fixed beam-ring candidates, `argmin` on
`|sub| + |goal−sub|`) was measured as the bar any learned composer has to
clear: **+0.027 ± 0.011 arrival over muted, t = +2.58**, over 1,024 paired
tasks. Adding a composer-side avoidance term to that same router was
monotonically *harmful* as its weight rose (arrival fell 0.578 → 0.476, lam 0
→ 2, t = −4.42 at the top) — it buys safety by not going anywhere, and local
avoidance is redundant with the low level's own beam-braking. `var_lam`
defaults to 0 for this reason: **the composer routes, the low level avoids.**

---

## Phase 5 — the routing question: occluded map, a broken instrument, held exploration (Sept 12, this session)

**The setup.** `occluded` (U-shaped pockets) was chosen because, unlike
`corridors`, it is **not greedily navigable** — the frozen low level alone is
0/1.000 there, and a genuine detour is required to arrive at all. This is
where "does the composer need its sensors" is actually being tested.

**The geometry, precisely** (`scripts/occluded_map.py`): a 4x4 grid of 16
U-shaped pockets, inner width 4 m, depth 5 m, wall 0.5 m, pitch 12 m, one
waypoint at the centre of each. **Every pocket opens the same way (+x)**, and
`use_free_start` takes the start pool directly from the map's waypoints — so
BOTH ends of every leg are down a dead end. The maneuver is therefore not just
"escape a pocket": exit your own through its +x mouth, travel, then go around
the target pocket and re-enter *its* mouth from +x.

Because every pocket faces +x, the exit heading is +x whatever the goal
bearing. Over the 84 ordered waypoint pairs within a 20 m leg, the correction
required at takeoff *just to leave the start pocket*:

| required correction | share of legs |
| --- | --- |
| under 45 deg | 14.3% |
| 45-90 deg | 21.4% |
| 90-135 deg | 28.6% |
| 135-180 deg (fly AWAY from the goal) | 35.7% |

**Median required correction: 90 deg**, against a measured on-policy median of
15.1 deg and 0.000 of decisions beyond 90 deg. That is the gap in the map's own
units — at the median, the map demands precisely the correction the policy
never makes.

**Caveat on `LES_DIFF` for this map — it does not thin uniformly.**
`QuadrotorNav._thin` parks a TRAILING fraction of each group in build order
(`off[:, keep:] = True`), and the generator emits pockets row-major, three
walls each. At `LES_DIFF=0.7` (34 of 48 walls live) this deletes **the entire
last row**: 11 pockets fully intact, 1 partial, and 4 waypoints (25%) left
standing in open space along a contiguous strip at y = +18 — deterministic and
identical in every episode, so it is learnable structure rather than a random
distractor. Of the 84 legs, 62% have both ends in intact pockets, 31% have one
end open, 7% are trivially open-to-open. **So nearly 40% of the training
distribution does not exercise the property the map was chosen for**, and any
arrival number read off this rung — including silence's ~0.057 — is likely
concentrated on the easy fraction. Worth fixing (randomise which pockets thin,
or pick a difficulty that does not carve a contiguous open row) before reading
too much into a small arrival gain here.

**First attempt, `LES_ACCUM=16` (scaled up from 4 for the ~5× sparser
arrival), flat.** Judge sat bit-identical at 0.082 for ten iterations; `EV`
collapsed to +0.015 against +0.58 on the CBD. Read initially as a
variance failure needing more draws per step (consistent with Phase 4's
lesson) — and separately, as a possible signal that "`sens` FELL" on this map
too. That second reading turned out to be an artifact of a broken instrument,
below.

**The instrument was lying.** `_sens` sampled 8 fresh records off whatever had
just been flown and shuffled them with an **unseeded** `randperm` — so on a
policy that was *provably* bit-frozen for fifteen iterations (no optimizer
step yet), `sens` still read 0.0034 → 0.0044 → 0.0057 → 0.0069 → 0.0058 →
0.0066 → 0.0053: a 2× swing with zero underlying change. **Every earlier
claim in this log that "`sens` fell"** — on the corridor grid, on the
occluded map — **was reading that noise**, and has been marked retracted
above. Fixed by capturing a fixed probe once (24 records × 256 rows, seeded
permutation) and replaying it every iteration, the same common-random-numbers
discipline the judge already used. Verified: identical across four different
task draws on a frozen network, and moves only when the weights do
(regression tests: `test_sens_holds_still_while_the_policy_does`,
`test_sens_still_moves_when_the_policy_does`).

**A second, cheaper fix found alongside it:** `lp_start`, a full forward pass
over every recorded decision (up to 20,000), was computed every iteration and
read by *nothing* in the accumulation branch — which takes no optimizer step,
and so has no drift-from-start to bound. 26.8% of the update's network compute
was dead weight; skipped when `accum > 1`.

**The diagnosis, once `sens` was trustworthy.** A behavioral portrait of the
frozen policy on 192 occluded flights:

| | at ε=0 (policy alone) | at ε=0.25 (as flown) |
| --- | --- | --- |
| median bearing correction off straight-at-goal | 15.1° | 19.8–20.2° |
| decisions beyond 45° (a real detour) | 4.0% | ~21–22% |
| decisions beyond 90° (around a pocket) | **0.000** | **12.4–12.6%** |
| `sens` (beam/pixel shuffle ÷ placement's own spread) | 0.63% | — |

The router flies almost straight at the goal and barely consults its sensors
— but the escapes it needs are *already being sampled* at almost exactly the
uniform-over-the-circle rate. So this is not an unreachable-action problem,
and more task draws (Phase 4's fix) cannot touch it. **The real problem:
clearing a U-shaped pocket is a temporally extended action.** One 90° detour
in isolation spends distance walking away from the goal and the flight dies
anyway — it is *correctly* punished — and only a *consistent run* of such
decisions pays off. Independent, per-decision exploration gives a run of
length *k* with probability ε^k (0.126³ ≈ 0.002). Measured directly: mean
consecutive-detour run length 1.13, only 1.4% of runs reached length 3+ —
matching the i.i.d. geometric null (p² = 0.0146) to three digits. Confirmed
from a third angle: the very first accumulated gradient step moved `sens`
**down**, 0.0045 → 0.0026 — the update was actively teaching the policy to
rely on its sensors *less*, because on i.i.d. exploration, looking never pays.

**The fix: held exploration (`LES_HOLD` / `noise_hold`).** The exploration
*mean*, not the sampled action, is drawn once and held for `noise_hold`
decisions, so a detour is committed to rather than re-rolled every report.
Holding the mean rather than the action is the load-bearing design choice:
holding the action makes the per-decision density a delta function with no
importance ratio, and since it carries no θ, gives *exactly zero* gradient
from every escape sample. Holding the mean keeps `μ_θ(s)` in the mixture's
first component, so an escape sample still pushes the policy, attenuated by
the posterior probability that it came from the policy rather than the
exploration term (~8% at 90° and σ=0.12, not zero). The mixture density was
verified to integrate to 1 (numerically, error < 2×10⁻³), to match the claimed
closed form pointwise, to collapse to the plain Gaussian at ε=0, and to still
carry gradient on an escape sample — four separate tests.

Measured effect on 192 flights, hold=6:

| | i.i.d. (hold=1) | held (hold=6) |
| --- | --- | --- |
| decisions beyond 90° | 0.126 | 0.124 (unchanged, as it should be) |
| mean consecutive-detour run length | 1.13 | **4.11** |
| runs reaching length 3+ | 0.014 | **0.577** (41×) |
| arrival | 0.047 | 0.062 |

Same detour rate, organized into coherent maneuvers instead of isolated
flickers, at slightly *better* arrival — a coherent detour is less
destructive than random flailing even before any learning acts on it.

**A second dead knob, found while measuring the first held run.** The initial
`hold=6` measurement came back at *exactly* the i.i.d. prediction — the
signature of a knob doing nothing. `noise_hold` had been defined in the
trainer, threaded through `composer_kw`, and documented in the launcher for
the entire life of the project, and `grep -rn noise_hold src/` returned
**zero hits**: `PolicyComposer.__init__` pops each kwarg it recognizes
explicitly, and the base `Composer` class does not `setattr` whatever is
left, so an unclaimed kwarg is silently accepted and silently dropped. Every
run that had ever set `LES_HOLD` had explored i.i.d. regardless. Fixed by
adding the `kw.pop("noise_hold", 1)` and binding it; guarded going forward
with a test that greps for the pop *and* asserts the base class still has no
generic kwarg-to-attribute path (`test_noise_hold_actually_reaches_the_composer`,
`test_the_base_class_does_not_setattr_kwargs`).

**Known open approximation, not yet resolved:** holding which mixture
component to draw from lets the *state* at a decision leak whether this
particular hold is exploratory, so the per-decision density used in the PPO
ratio is no longer exactly the true conditional — the same bias every
correlated-exploration method carries (Ornstein–Uhlenbeck noise,
parameter-space noise). This is being treated as "verify empirically, don't
assume," not as solved.

**Status of the held-exploration training run at the time of writing** (see
the composer README's "Status" section for anything that postdates this
file): `hold_1928` (`LES_ENV=occluded LES_DIFF=0.7 LES_EXPLORE=0.25
LES_ACCUM=16 LES_HOLD=6`), launched after a clean 1039-passed/0-failed test
suite, compared against a same-config i.i.d. control (`router_1821`, killed
after its own first step showed `sens` falling 0.0045 → 0.0026). The held
run's first accumulated step landed at iteration 16 (visible at 17, the usual
one-iteration pipeline lag) and moved `sens` 0.0046 → 0.0038 — confirmed
stable across the following iteration, not a transient. **Also down, and by
the same ~1.7× factor as the control's drop** (control: 0.0045 → 0.0026,
1.7×; held: 0.0046 → 0.0038, 1.2×) — smaller, but the same direction, on a
single step. One step is not enough to call this a difference in kind rather
than degree; whether `sens` keeps falling, flattens, or turns upward across
*subsequent* steps — the actual test of the mechanism — is still open and
being watched.

Two constants introduced this phase are picked, not derived, and are flagged
for a sweep once the mechanism is confirmed to pay: `noise_hold=6` and
`LES_ACCUM=16` (the latter inherited from Phase 4's sparsity-scaling rule,
which assumed a variance failure — a diagnosis this phase's own measurements
supersede for the occluded map specifically; ACCUM may now be over-sized for
the problem actually being solved here).

---

## Threads still open

- Does `sens` rise across *multiple* accumulated steps under held
  exploration, and does judge arrival clear silence's noise floor on
  `occluded` (silence ≈ 0.057–0.070 depending on exact seed set; a judge
  result needs roughly ±0.035 to mean anything at 256 episodes)?
- `noise_hold` and `LES_ACCUM` are both picked constants on the occluded arm
  and want a joint sweep once the qualitative mechanism is confirmed —
  candidate: if coherent escapes carry much more signal per draw than the
  i.i.d. case did, fewer task draws per step (lower `ACCUM`) may suffice,
  trading some variance for a much faster feedback loop (currently ~40 min a
  step at ACCUM=16).
- The correlated-exploration density approximation (state leaks which
  decisions are "held") has not been quantified, only flagged.
- Whether the held-exploration mechanism, if it works on `occluded`, changes
  anything on `corridors` or `singapore_cbd` — those maps didn't need it
  (geometry already sufficed), so a regression there would be a red flag for
  the mechanism, not evidence for it.

---

## Operational lessons that shaped how these experiments were run

Not physics or RL findings, but process rules that repeatedly saved or cost
wall-clock, worth keeping visible:

- **This Mac has 16 GB, shared with a browser, two editors, and Spotify.**
  Load averages of 70–100 during a "stalled" run were swap thrashing, not
  compute contention or a deadlock — `vm_stat` (free pages, compressor size)
  and `sysctl vm.swapusage` diagnose this before blaming the code. Worker
  count is sized by memory headroom (~3 GB available to training), not core
  count; 8 workers gained nothing over 4 here and pushed the machine into
  swap.
- **The Mac sleeps mid-run.** A 100+ minute gap between eval points that
  otherwise land ~7 minutes apart is sleep, not a stall — launch long runs
  under `caffeinate`.
- **Diagnostics that need their own rollout compete with training for the
  same four performance cores.** Prefer reading the training run's own batch
  (its records, its judge points) over a separate full-batch measurement;
  when a separate measurement is unavoidable, run it once, small, detached,
  and record the answer rather than repeating it.
- **A falling loss proves the objective is optimizable, not that it's
  connected to the policy.** Four different credit-assignment schemes failed
  in one session for the *same* underlying reason (a blocked gradient path);
  the one-line check that would have caught all four sooner: assert the
  deciding parameter's weights actually move on a fixture.
- **When a metric is central to a conclusion, test the metric on a case where
  the truth is known** — a frozen policy (nothing should move), a
  beam-blind head (shuffling its inputs should do nothing), a fresh vs. zero
  initialization (distinguishes "small" from "dead"). Several of the defects
  above cost days precisely because the instrument, not the model, was
  measured.
