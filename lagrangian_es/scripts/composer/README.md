# The composer: a transformer router over a frozen Lagrangian low level

Two layers, trained in two stages. Stage A (see the repo root `README.md`)
evolves the Lagrangian low level — a shaped potential and dissipation matrix
that flies the drone toward wherever it is currently told to go. Stage B,
documented here, freezes that low level (`FREEZE_LOW = True`, one genome `P`)
and trains a small transformer, the **composer**, to decide *where* to send it,
continuously, over the sensors it actually has.

The split is deliberate: **the composer routes in real time; the low level
handles the kinematics.** The composer never touches motor commands. It reads
a token per beam/pixel/self-state and every `EVERY_M` low-level steps emits one
continuous action — a subgoal in the vehicle's own frame — which the low level
then flies toward exactly as it would fly toward the flight's real goal.

## Why a router, not a planner

Earlier arms (`cotrain_v5`–`v9`) gave the composer a *discrete* action
vocabulary — `HOLD` / `PLACE` / `RAISE` / `LOWER` / `TURN` / `LOOK`, one token
per report, trained by categorical PPO. That died to a specific and repeatable
failure: the action space was binary in effect (waypoint or not) and
direction-blind, so the policy's easiest improvement was to say nothing —
`speak -> 0.000` in thirteen iterations, judge parked at silence
(`[[waypoints-are-direction-blind-and-harmful]]`, `[[composer-emits-action-tokens]]`).

`cotrain_v11` (this file) replaces that with:

- **A continuous vocabulary** (`actions_cont.py`): one token, `WAYPOINT(r,
  theta, phi)`, whose arguments are a diagonal Gaussian over unsquashed
  reals, squashed through `tanh`. `EOS` still exists as a token but —
- **`route_only` (`LES_ROUTE=1`) masks `EOS` out of the type logits
  entirely.** The composer always emits a waypoint; the only decision left is
  *where*. This is the one config change that turned a router that learns to
  be silent into one that has to route.
- **A goal-relative action frame.** The commanded bearing is
  `theta = bearing(g_ego) + pi*tanh(a1)`, not a nose-relative angle. At
  `a1 = 0` (an untrained net's prior) this is the identity: fly straight at
  the goal. The nose-relative frame gave an untrained net arrival 0.000; the
  goal-relative frame gives 0.549 against a muted 0.555
  (`[[put-the-identity-action-at-the-origin]]`) — the composer starts at a
  sane default and learns *corrections* to it, not a heading from scratch.

## The objective and its instruments

No shaping, no hand-tuned bonus curve (`[[loss-as-simple-as-possible]]`,
user's standing rule — *"nothing should be hand tuned"*). The composer is
trained by PPO (`policy_cont.ppo_update_cont`) against the same cost the low
level was evolved on: distance-to-go, a goal bonus paid on arrival, a death
cost, `SUBGOAL_COST` per placement (zero under `route_only` — a router that
must always speak should not be taxed for speaking,
`[[composer-routes-low-level-avoids]]`). Per-decision credit
(`returns_from_stream`, a state baseline `V(s)`, EV monitored) — a
flight-level advantage has zero within-flight variance and teaches nothing
(`[[per-decision-credit-or-nothing-learns]]`).

Two instruments exist specifically to catch this training lying to you, and
both have caught something:

- **`EV`** (explained variance of the critic). A batch that cannot rank
  anything reads `EV ~ 0`, however good the arrival trend looks — this is
  what tells you the batch itself carried no signal, independent of whether
  the policy moved.
- **`sens`** (`_sens` in `cotrain_v11.py`): shuffles the beam/pixel rows on a
  *fixed, seeded probe* and reports how far the commanded subgoal moves, in
  reach units, as a fraction of the placement's own spread. This is the
  measure of whether the composer is actually looking. It used to draw fresh
  records with an unseeded shuffle every call and swung 2x on a policy that
  had not changed by one bit — every earlier claim that "`sens` fell" was
  reading that noise (`[[a-frozen-policy-is-a-free-null-test]]`). Fixed to a
  captured probe, replayed every iteration: now identical across task draws
  on a frozen network, and moves only when the weights do.

## Gradient accumulation across task draws

The single biggest lever this project found for whether the composer learns
at all is not in the loss — it's in how many task draws back one gradient
step. Disjoint halves of *one* task draw agree at gradient cosine +0.46;
*independent* draws agree at -0.45 (`[[task-variance-was-the-blocker]]`).
Consecutive updates on independent draws undo each other. `LES_ACCUM=N` holds
the policy fixed across N rollouts, sums the gradient, and steps once — the
same effective batch for wall-clock instead of memory (a 16 GB machine can't
hold N rollouts at once). Scale N with sparsity: CBD (28% arrival) needed 4;
occluded (5.7% arrival) needed 16 (`[[scale-task-draws-not-difficulty]]`).

**This is expensive to read.** During the N-1 iterations before a step, the
policy is bit-for-bit frozen, so anything that moves in the log during that
window is noise, not learning — treat every such window as a free null test
of your own instruments (`[[a-frozen-policy-is-a-free-null-test]]`). At
`ACCUM=16` a step lands roughly every 40 minutes.

Two bugs specific to the accumulation path were found and fixed by exactly
this kind of check:

- `lp_start` — a full forward pass over every recorded decision — was
  computed every iteration and read by *nothing* in the accumulation branch,
  which takes no step and so has no drift-from-start to cap (26.8% of the
  update's network compute, now skipped).
- The gradient itself does correctly accumulate (verified: norm grows
  monotonically across calls, weights hold still until the external
  `opt.step()`).

## Held exploration: escaping a pocket is a temporally extended action

The `occluded` map (U-shaped pockets) is chosen because it is **not greedily
navigable** — unlike `corridors`, which the composer solved by staging
subgoals along an already-clear grid while `sens` stayed low. On `occluded`
the composer flies straight at the goal (median bearing correction 15
degrees; 0.000 of on-policy decisions point past 90 degrees) and reads its
sensors at ~0.6% of its placement's own spread.

The escapes are not unreachable — at `LES_EXPLORE=0.25`, 12.6% of *explored*
decisions already point past 90 degrees, matching uniform-over-the-circle
exactly. They are never reinforced, because clearing a pocket needs a
*consistent run* of such decisions (one 90-degree detour in isolation just
spends distance and then the flight dies anyway), and i.i.d. exploration
gives a run of length k with probability `eps^k`
(`[[escapes-need-temporal-coherence]]`).

`LES_HOLD=N` (`noise_hold`) holds the exploration *mean* — not the sampled
action — for N decisions, so a detour is committed to instead of re-rolled
every report. Holding the mean rather than the action matters: the density
stays `(1-eps)*N(mu_theta, sigma) + eps*N(explore_mu, sigma)`, so an escape
sample still carries a gradient through `mu_theta`, attenuated by the
posterior that it came from the policy. A held *action* would make the
per-decision density a delta with no importance ratio and exactly zero
gradient. Measured at `LES_HOLD=6`: detour rate unchanged (12.4%), mean
consecutive-detour run 1.13 -> 4.11, runs reaching length 3+ 0.014 -> 0.577.

Caveat carried forward, not hidden: holding the component choice lets the
state at decision t leak whether this hold is exploratory, so the
per-decision density is no longer exactly the conditional — the standard
correlated-exploration bias (OU noise, parameter-space noise share it).
`noise_hold=6` is a picked constant, not derived, and wants a sweep.

**A general lesson from finding this:** `noise_hold` was accepted as a
config key and threaded through `composer_kw` for the whole life of the
project, and read by *nothing* — `PolicyComposer.__init__` pops each kwarg it
knows about explicitly, and the base `Composer` does not `setattr` the rest,
so an unclaimed kwarg is silently dropped
(`[[a-plumbed-knob-can-be-read-by-nothing]]`). When adding a knob: add the
`kw.pop(...)` in `composer/policy.py` *and* assert the attribute exists
before trusting any result that depends on it.

## Running a training arm

```
LES_ENV=occluded LES_DIFF=0.7 LES_EXPLORE=0.25 LES_ACCUM=16 LES_HOLD=6 \
  WORK=~/drone_runs/<name> \
  caffeinate -i bash scripts/composer/start_router.sh
```

`start_router.sh` copies `cotrain_v11.py` into `WORK` and launches it there —
edits to the source after launch do not reach a running job; relaunch to pick
them up. See the top of `start_router.sh` for the full list of defaults and
the measured bar each configuration has to clear (a hand-built router beating
silence at t=+2.58 over 1024 paired tasks is the reference).

### Every `LES_*` variable, what it does, and whether it reaches anything

| var | trainer name | default | reaches |
|---|---|---|---|
| `LES_ENV` | `TRAIN_ENV` | `singapore_cbd` | which built map (`singapore_cbd`, `corridors`, `occluded`) |
| `LES_DIFF` | `DIFF_FIX` | `-1` (curriculum) | fraction of buildings present; `-1` lets the curriculum bar move it |
| `LES_LEG` | `LEG_FIX` | `0` (default) | leg length in metres |
| `LES_LOSS` | `LOSS` | `ce` | which update runs: `ce` / `error` / `route` / `speak` / `time` / `ppo` (this file's arm) |
| `LES_ROUTE` | `ROUTE_ONLY` | `0` | composer key `route_only` — masks EOS, forces a waypoint every report |
| `LES_INJECT` | `INJECT` | `1` | composer key `inject` — 0 decides at every report; >0 samples sparsely |
| `LES_SPEAK` | `SPEAK_FLOOR` | `0.0` | composer key `speak_floor` (moot under `route_only`) |
| `LES_ACCUM` | `ACCUM` | `1` | gradient steps every N task draws (see above); `1` = step every iteration |
| `LES_EPOCHS` | `EPOCHS_C` | `3` | optimizer epochs per update, **not** wall-clock — steps are the scarce resource |
| `LES_MB` | `MB_C` | `1024` | minibatch size |
| `LES_MAXS` | `MAX_SAMPLES` | `40000` | cap on decisions sampled into one update; keep lower (e.g. `20000`) under `LES_ACCUM>1` — the accumulation pass collates every sample at once |
| `LES_KCHAIN` | `K_CHAIN` | — | composer key `k_chain`, the transformer's chain length |
| `LES_TEMP` | `TEMPERATURE` | — | composer key `temperature` on the sampling distribution |
| `LES_EXPLORE` | `EXPLORE_EPS` | `0.0` | composer key `explore_eps` — uniform-in-action mixture weight |
| `LES_HOLD` | `NOISE_HOLD` | `1` | composer key `noise_hold` — decisions an exploration draw is held (see above; **dead before this session's fix**) |
| `LES_VART` | `VAR_T` | `2.0` | composer key `var_temp` — variational re-draw over the policy's own candidates |
| `LES_VARK` | `VAR_K` | `16` | composer key `var_k` — candidates considered per variational re-draw |
| `LES_VARLAM` | `VAR_LAM` | `0.0` | composer key `var_lam` — composer-side avoidance weight (0: avoidance stays the low level's job, see `[[composer-routes-low-level-avoids]]`) |
| `LES_REPEAT` | `REPEAT` | `1` | fly E/k distinct tasks k times each (imitation arms) |
| `LES_SIGNED` | `SIGNED` | `1` | can an imitation weight go negative (imitation arms only) |
| `LES_ALPHA`, `LES_BETA` | `ERR_ALPHA`, `ERR_BETA` | `1.0` each | weights in the `error` loss arm only |
| `LES_TAU` | `WEIGHT_TAU` | `1.0` | imitation-weight temperature (`imitate` arm only) |
| `LES_DIFF0` | `DIFF0` | `0.1` | curriculum starting difficulty, composer key `difficulty` |
| `LES_W0` | `W0` | — | optional warm-start weights path |

### Reading the iteration log

```
iter 17  buildings 70% legs 20m  arrive 0.043 crash 0.957 t_arr 0.388  cost 1339.91 subgoals 28.6
  | loss -0.007 v 1.001 EV +0.007 kl 0.0000 clip 0.00 steps 20 on 20000 decisions
  | speak 0.400 std 0.120 sens 0.0045  [roll 136s update 137s | 41.6m]
```

- `arrive` — training-batch arrival, noisy per iteration (task-draw
  dominated; do not read iteration-to-iteration wobble as progress —
  `[[task-variance-was-the-blocker]]`).
- `EV` — explained variance of the critic; near 0 under a frozen policy or an
  uninformative batch, not a bug in either case.
- `kl` / `clip` — read `0.0000` / `0.00` for every iteration inside an
  `ACCUM` window by construction (no step taken yet); only meaningful at the
  iteration a step lands.
- `sens` — see above; flat within a window is correct, a step should move it.
- `[roll Ns update Ns | Tm]` — rollout time, update time (the accumulation
  pass runs whether or not it steps), cumulative wall-clock.

Separately, every 10 iterations, a full judge pass over the held-out city
(256 fixed paired tasks) prints `judge N city20 <p>/<1-p> cost ... crashes
...`. This is the actual scoring metric, not the training batch — compare it
to the **silent-flight baseline** for the environment/difficulty pair (see
`start_router.sh`), not to 0. At `p~0.08` and 256 episodes, `SE ~ 0.017`; a
result needs to clear roughly 2 SE (~0.035) to mean anything.

## Files

| file | role |
|---|---|
| `actions.py` | the discrete action vocabulary (superseded arms, `v5`-`v9`) |
| `actions_cont.py` | `ContVocab`: the continuous `WAYPOINT` token, its goal-relative geometry, and the mixture `log_prob` (plain + held-exploration) |
| `base.py` | `Composer` base class — kwarg handling lives here; see the dead-knob caveat above |
| `policy.py` | `PolicyComposer`: shared training machinery, `explore_eps`/`noise_hold`/etc. attribute binding |
| `policy_cont.py` | `ContPolicyNet` + `ContComposer`: the transformer, `choose` (sampling, incl. held exploration), `ppo_update_cont` |
| `spec.py` | `TaskSpec` — the non-learned machinery that makes an emitted spec safe to apply |
| `tokens.py` | ego-centric tokenization of what the vehicle perceives (beams, pixels, self-state) |
| `transformer.py` | `TransformerComposer` base: chaining, token assembly, `reset`/`tokens`/`_apply` |
| `variational.py` | probabilistic waypoint re-draw over the policy's own candidates (`var_temp`/`var_k`) |
| `distill.py`, `oracle.py`, `ga.py`, `explain.py` | earlier/alternate arms: imitation from a planner oracle, an evolvable composer, decision tracing |

## Environments

Built maps live in `src/lagrangian_es/environments/maps/*.json`
(`singapore_cbd`, `corridors`, `occluded`). Chosen for a specific property
each:

- **`singapore_cbd`** — dense but greedily navigable; the composer's first
  win (judge 0.277 -> 0.453, +6.0 SE) came from placing more, better-sited
  subgoals along an already-viable route, not from reading sensors.
- **`corridors`** — a Manhattan grid, also greedily navigable; composer
  solved it by staging, and `sens` fell rather than rose
  (`[[decomposition-needs-safer-subgoals-not-shorter]]`).
- **`occluded`** — U-shaped pockets, **not** greedily navigable. The frozen
  low level alone is 0/1.000 here; a genuine detour is required. This is
  where the routing question — does the composer need its sensors — is
  actually being asked, and where `LES_HOLD` is being tested.

## Status as of this session

`FREEZE_LOW=True`, `singapore_cbd`/`corridors` results above are historical.
Current work is on `occluded` at `LES_DIFF=0.7`, comparing i.i.d. exploration
against `LES_HOLD=6`, both at `LES_ACCUM=16`. See the memory files linked
throughout for the full experimental trail; `[[escapes-need-temporal-coherence]]`
and `[[a-plumbed-knob-can-be-read-by-nothing]]` are the two findings from this
exact comparison.
