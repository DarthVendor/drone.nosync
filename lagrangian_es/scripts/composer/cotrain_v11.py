"""Co-training v7: v6 with a 360-degree range fan (24 beams, 15 degrees apart) in place of the 120-degree front fan --
does the low level learn to use its beams when it can see all round?

v6: yaw is the LAGRANGIAN's -- no heading rule; the potential's gradient with respect to yaw, through the
body-fixed beams, is the yaw torque (yaw_mode="lagrangian").  Otherwise v5:

v5: the composer is a language model over the flight -- one ACTION TOKEN per report from the chain of
measurement and action tokens (HOLD, PLACE on a bearing/range grid, RAISE/LOWER a term, TURN, LOOK), a categorical policy
with exact likelihoods and no scale to tune; the low level on the crash-aware GA (16 x 288); the scene curriculum as v4.

v4:

v3:

v2: a front-mounted fan, a downward fan, the camera and the tilt;
the composer may command heading.  Same simultaneous scheme as v1.  The low
level starts as the all-round-beam genome, extended with zero rows for the new
channels -- so it begins confused about what its beams now mean, and whether
it adapts is the experiment."""
import json, math, shutil, sys, time, torch
sys.path.insert(0, "/Users/maddoxnoon/Desktop/drone.nosync/lagrangian_es/src")
from lagrangian_es.config import Config, RolloutCfg
from lagrangian_es.composer import center_by_task, returns_from_stream, returns_goal_only
from lagrangian_es.composer.policy_cont import error_update, imitate_update, ppo_update_cont, time_update
from lagrangian_es.es import build, build_composer, build_sensors
from lagrangian_es.composer.policy_cont import paired_advantage, projected_time
from lagrangian_es.metric import identity_preconditioner
from lagrangian_es.operators import whitened_mutation
from lagrangian_es.parallel import ParallelRollout
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen
from lagrangian_es.widen import extend_inputs
SP = sys.argv[1]; OUTER = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SIGMA0_COMPOSER = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 100         # the LONGEST a placed subgoal is held (steps of 0.02 s)
EVERY_M = 20                                                   # the drone reports every EVERY_M steps; decisions fall on reports
SUBGOAL_COST = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0   # what the composer pays per subgoal it places
# The death charge, per second of episode left -- the user's 6.  (The reach-0
# parking the composer found at judge 10 was the distance term's doing, not
# this one: finishing a leg raised the charge.  The rollout now charges the
# distance TO GO along the tour, so that spot is gone.)
# Death must never be cheaper than living: a drone d metres from its goal pays
# d a second alive, and the largest distance to go on these tours is ~35 m, so
# at 6 a second a fresh drone was selected for crashing EARLY (an 18 s episode
# cost ~108 + 11 t for a crash at t).  40 a second exceeds any living charge.
DEAD_COST = float(sys.argv[6]) if len(sys.argv) > 6 else 40.0
STOP_FINISHED = 1.0
# NO early stop.  A training batch used to end once this fraction of the
# survivors had arrived, which BOUNDS the reach the batch can report -- the cap
# has to sit above the curriculum bar or the bar is unreachable by
# construction, and at 0.8 against a 0.9 bar difficulty was pinned at zero
# buildings however well the composer flew.  With the bar now at 0.95 there is
# no headroom left for a cap at all, so batches run to completion and the
# reported reach is the real one.  It costs time: every flight runs its full
# budget instead of the batch ending early.
# The cap must sit ABOVE the curriculum bar, and it did not.  A training batch
# ends once this fraction of the survivors has arrived, which BOUNDS the reach
# the batch can report: at 0.8 the measured reach could not exceed ~0.8, while
# the curriculum needed 0.9 to step up, so difficulty was pinned at zero
# buildings no matter how well the composer flew.  Measured at difficulty 0 with
# the batch run to completion: reach 0.906, crash 0.021, and every remaining
# failure ended WITHIN the 0.25 m arrival tolerance -- flights that arrived as
# the clock ran out, not navigation failures.
GOAL_BONUS = 60.0                                          # per leg held (was 15): a finish must outrank the 50-80 spread the placements put on the time integral (user)
# Measured 2026-09-08 (diag_genome.py): at 48 x 96 and sigma 0.012 the spread of
# fitness across mutants (6.5) was SMALLER than one genome's 96-episode noise
# (10.4) and the ranking on two batches agreed at Spearman -0.11 -- selection
# was ranking noise, the genome a random walk.  At sigma 0.036 the ranking
# agreed at +0.28 and the elite of one batch beat the rest on the other by
# 4.4.  So: a third of the genomes with three times the episodes each (noise
# 6.0 < spread), and double the mutation.  Same 4608 rows an iteration.
K_CHAIN = 128            # the composer's memory, in events.  A flight is 1800 steps
# reporting every 20, so its whole history is ~90 measurement events plus its
# own instructions; 32 kept only the last ~15 seconds and everything earlier
# fell off, which is why it could not know it had already tried a corridor.
# 128 covers a whole flight.  Measured cost: a rollout goes from ~12.7s to
# ~17.7s at this batch size.
EXPLORE_EPS = 0.0
# No INJECTED exploration.  Cross-entropy has no importance ratio to correct for
# the distribution its samples came from, so imitating rows drawn from a
# policy-plus-uniform mixture teaches the policy that mixture.  Measured: with
# 30% uniform over the five token types, the speak rate walked to 0.323 in six
# updates -- and 0.7*0.12 + 0.3*0.8 = 0.324, so it had landed exactly on the
# sampling distribution, entropy rising and crashes rising with it.
#
# Exploration is now IMPLICIT, which is the only kind consistent with imitating
# what you sampled: the token type is drawn from the categorical and the
# arguments from the Gaussian on every decision, so the policy already varies,
# and the variation being reinforced is its own.
# No injected exploration.  Under cross-entropy there is no importance ratio to
# correct for the behaviour distribution, so imitating rows drawn from a
# policy-plus-uniform mixture teaches the policy that mixture.  Measured: with
# 30% uniform over five token types the policy's speak rate walked to 0.323 in
# six updates, and 0.7*0.12 + 0.3*0.8 = 0.324 -- it had converged exactly onto
# the sampling distribution, entropy rising, crashes rising with it.
#
# A continuous policy explores on its own: the token type is sampled from the
# categorical and the arguments from the Gaussian, every decision.  That is the
# exploration, and it is the same distribution being imitated, so the update is
# consistent.                                        # the recorded rows draw from the policy mixed with a uniform: rare tokens ~100-170 samples a batch instead of 3-31 (0.6 wrecked the exploring flights: reach 0.014)
VCOEF = 0.5                                              # STATE baseline: advantage = return - V(state).  Measured on a batch: with a mean baseline HOLD carried +0.31 and every other token -0.45 to -1.02 (the states they are chosen in, not their effect); with V they read +0.02 / -0.04 / -0.13 / +0.11 and the half-batch gradients agreed at 0.95 vs 0.75
FREEZE_LOW = True        # stage B: the low level is FROZEN at what stage A found on the
# empty map, and ONLY the composer trains.  With no mutants to rank there is no
# reason to fly 16 copies of one genome, so the population collapses to 1 and
# the episode count rises to keep the same number of flights per iteration --
# and every one of them is now a DISTINCT task instead of the same 288 flown
# sixteen times.
#
# The cost: `center_by_task` needs at least two samples of a task to estimate
# its difficulty, and at P=1 there is exactly one, so it becomes a no-op and
# that variance reduction is gone.  The value head (vcoef) is the remaining
# baseline.  If the returns get too noisy without it, the fix is P=2 flying
# each task twice with independent composer draws.
# REC = 1.0 with the low level FROZEN: record EVERY flight, not half of them.
# The split existed so the unrecorded half could give the GA fitness untainted
# by the exploration the recorded half carried.  There is no GA here (SIGMA 0,
# selection skipped) and EXPLORE_EPS is 0, so both halves sample from the
# identical policy and the unrecorded 288 were flown and discarded.  Recording
# them doubles the update's data at zero rollout cost.
WORKERS, GAMMA, JUDGE_EVERY, REC = 6, 0.99, 10, (1.0 if FREEZE_LOW else 0.25)
# REC was held down because a recorded row EXPLORES, and an exploring row is no
# use to the genetic ranking.  With the low level frozen there is no ranking to
# protect, so half the batch can be recorded instead of a quarter: ~1150
# recorded flights an update, the same volume as before, from a batch of
# distinct tasks rather than 288 flown sixteen times.   # FULL horizon: a token's effect on the expected outcome keeps growing past 2 s (straight PLACE +0.5 at 2 s, +23 over the flight); the 2 s horizon threw that away    # gamma per report (0.2 s): a 2 s horizon, the measured time for two near-identical flights to separate; at 0.99 (20 s) the half-batch gradients agreed at cos 0.42, at 0.9 at 0.87   # a quarter of the rows explore and record: one token per report per row is many samples
MAX_SAMPLES = 10_000
#: how far the vehicle covers in one whole episode at its airspeed limit; the
#: projection charges outstanding distance in these units, so "one episode of
#: flying still to do" costs exactly 1.0 of extra time
SPAN = 5.0 * 1800 * 0.02
# --- the 2x2 -----------------------------------------------------------------
# Arrival stalled at ~0.91 on an empty map while the cost it was actually
# optimising fell 53%: the update trains on flights that already arrived, and
# ranked them across TASKS, so "the fastest 30%" was "the 30% nearest goals".
# Two independent fixes, tested combinatorially at EQUAL flight budget:
#
#   REPEAT k   fly E/k distinct tasks k times each, so a task has k samples and
#              the weight can be centred WITHIN it -- the spread is then credit,
#              not difficulty.  k=1 is the control (centring falls back to the
#              batch mean, i.e. no baseline).  Total flights per iteration is E
#              either way, so the arms cost the same.
#   SIGNED     False: imitation.  A failure has no arrival time and takes weight
#              zero, so the loss can only reallocate probability among successes
#              -- it cannot lower the failure rate, only speed up what works.
#              True: the weight is the negated centred arrival time, so a
#              failure (the slowest possible) is pushed DOWN.  Still ONE
#              cross-entropy term; the sign is in the weight, not a new loss.
import os as _os                                           # the arms come from the environment, not argv (argv is already full)
REPEAT = int(_os.environ.get("LES_REPEAT", "1"))            # samples per task
SIGNED = _os.environ.get("LES_SIGNED", "1") == "1"          # can the weight go negative
# DEFAULT ON now that the score is soft time.  With finish_frac the weight had
# to be non-negative: a failure had no arrival time, so imitating it would have
# made the failure MORE likely and it could only be dropped.  That capped the
# objective at "reallocate probability among the successes" -- it could make
# good flights quicker and could not make bad flights rarer.  Soft time ranks
# failures too, so the weight can go negative on them and the update can push
# probability DOWN.  It is still a weighted log-probability -- without
# differentiating the simulator the score function is the only estimator
# available -- but it is a policy gradient now, not imitation.
WEIGHT_TAU = float(_os.environ.get("LES_TAU", "1.0"))       # in units of the batch's own spread
W_MAX = 2.0                                                # -log p is unbounded below; cap the push-down
LOSS = _os.environ.get("LES_LOSS", "ce")                   # "ce" | "error" | "time"
# PER-DECISION credit for the cross-entropy loss.  Flight-level weighting
# reinforces every decision inside a kept flight equally, including the ones the
# flight arrived DESPITE -- measured, the advantage separated crashed from
# surviving flights at z = -166 while correlating -0.013 with the braking
# argument.  It is also a positive feedback loop on the composer's own
# behaviour: subgoals climbed 8 -> 24.5 while arrival stalled, and at short legs
# it took 0.977 down to 0.707.  Weighting each decision by the distance it
# actually closed makes the credit local; the loss is unchanged.
# PER-DECISION CREDIT IS GONE, with `progress_weights` itself.  It scored a
# decision by the distance closed before the NEXT one, which under sparse
# injection is unmeasurable (a flight has one or two decisions, and the last is
# never credited) and confounded when it did fire (the distance is the FROZEN
# LOW LEVEL's, not the token's).  Measured on the live checkpoint: the speak
# floor emitted 13 WAYPOINTs and 0 of 13 survived it.  Credit is now the paired
# counterfactual -- see PAIRED below.
# SPARSE INJECTION (LES_INJECT).  0 = the composer decides at every report,
# ~90 times a flight.  n > 0 = it decides at exactly n report steps drawn at
# random per flight and is silent otherwise, so the flight's outcome credits
# those n tokens against a baseline that succeeds on its own.  Measured on the
# empty map at 8 m legs, composer at initialisation:
#     inject 0 -> arrive 0.000, crash 0.469, 29.2 subgoals   (nothing to learn from)
#     inject 1 -> arrive 0.917, crash 0.000,  0.46 subgoals
#     inject 5 -> arrive 0.698, crash 0.000,  2.25 subgoals
# The frozen low level alone is 1.000 on that rung, and a SILENT composer is
# now also 1.000 -- it was 0.000 until the scripted opening placement was
# removed, which is what had kept every flight failing and every batch empty.
INJECT = int(_os.environ.get("LES_INJECT", "1"))     # ONE token a flight: with
# two, both share the flight's advantage and neither is attributable.
# VARIATIONAL WAYPOINTS (LES_VART, 0 = off).  See composer/variational.py.
VAR_T = float(_os.environ.get("LES_VART", "2.0"))
VAR_K = int(_os.environ.get("LES_VARK", "16"))
# LES_SPEAK: probability a decision may not open on EOS, so it must say
# something.  Guards the absorbing state -- see `PolicyComposer.speak_floor`.
SPEAK_FLOOR = float(_os.environ.get("LES_SPEAK", "0.0"))
# Where the composer is actually worth having, measured with no training at all
# (128 episodes, frozen low level, arrive/crash):
#
#     setting            no composer     untrained composer
#     8 m legs, 10%      0.992 / 0.01    0.977 / 0.01
#     20 m legs, 100%    0.250 / 0.75    0.227 / 0.73
#     50 m legs, 100%    0.039 / 0.96    0.070 / 0.89   <- the only cell it helps
#
# It nearly doubles arrival and cuts 7 points of crash at 50 m legs in the full
# city, and is mildly harmful everywhere else -- at 8 m legs the plant is already
# at 0.99 and a navigator can only subtract.  Success there is 4-7%, which is
# also why the loss matters: self-imitation needs successes to copy and there
# are almost none, while the error loss has a target for every decision of every
# flight including the 93% that crash.
ERR_ALPHA = float(_os.environ.get("LES_ALPHA", "1.0"))     # weight on e_reach
ERR_BETA = float(_os.environ.get("LES_BETA", "1.0"))       # weight on e_aim
LEG_FIX = float(_os.environ.get("LES_LEG", "0") or 0)      # >0 pins the leg length
DIFF_FIX = float(_os.environ.get("LES_DIFF", "-1"))        # >=0 pins the difficulty
ARM = (f"k{REPEAT}{'s' if SIGNED else 'u'}" + ("" if LOSS == "ce" else f"-err{ERR_ALPHA:g}_{ERR_BETA:g}")
       + "-pair")
COMPILE_WORKERS = False                                    # the workers compile the controller's forward passes (one compile thread each; the parent warms the compiler before the fork)
TOK_FRAC = 1.0   # KEEP EVERY RECORDED ROW'S TOKENS.  At 0.1 nine of ten rows
# were flown, recorded, and then made untrainable because their scene tokens
# were dropped -- `tok_keep` gates whether a decision can be a sample at all.
# The 0.1 was sized for a composer making ~90 decisions a flight (~150k samples,
# 600 MB a worker); sparse injection makes one or two, so the whole batch is a
# few thousand samples and a few tens of MB.  With REC 1.0 beside it this is
# ~20x the data per update at no extra rollout cost.
T_TRAIN = 1800                                             # the JUDGE's horizon: on 18 s episodes a crash at 10 s cost 320, on the judge's 36 s 1040, and the composer drifted bold (crash +1.5 points per update) while its training cost fell -- train on what is judged (user: the curriculum must match the judge)
# TWO copies of the frozen genome, not one.  `center_by_task` needs at least
# two flights of a task to estimate its difficulty, and at P=1 it had exactly
# one, so it became a no-op and the return spread went straight back to where
# it was before that fix (measured: 262->136 with a population, 284->284
# frozen).  The two copies are identical, so the ONLY difference between a
# task's two flights is the composer's own draws -- which is precisely the
# quantity the update is trying to estimate.  The continuous composer samples
# each row independently (no paired uniforms), so the two flights genuinely
# differ.
# 576 flights an iteration, not 2304.  Measured: cost is LINEAR in flights
# (~100 ms each, flat from 384 up to 2304), so the batch was costing 230 s of a
# ~260 s iteration.  The update caps at MAX_SAMPLES tokens anyway, so a quarter
# of the flights still fills most of it, and four times as many iterations is
# worth more than four times the data per iteration when nothing has yet been
# shown to move.
#
# The real fix is the worker pool, which ran this at 19 ms a flight earlier
# today -- five times better, because the work is dispatch-bound and separate
# PROCESSES help where threads do not (measured: 1 thread 12.1 s, 6 threads
# 16.1 s, slower).  The pool shards by POPULATION though, and a frozen low
# level is one genome, so using it needs episode sharding in `parallel.py`.
# EPISODES PER ITERATION (LES_EPS).  Raised because only the flights whose
# injected tokens actually changed the arrival time carry any signal -- the
# rest tie with their control at advantage 0 and are dropped.  More tasks is
# the only way to buy more of those; the update's minibatch was never the
# constraint (it is 4096 against a few hundred samples).
P, E, SIGMA = (1, int(_os.environ.get("LES_EPS", "1152")), 0.0) if FREEZE_LOW else (16, 288, 0.024)                                   # measured: at 48 x 96 / 0.012 the ranking was noise (Spearman -0.11); here the noise (6.0) is below the spread
# --- the curriculum: legs INSIDE buildings -----------------------------------
# Legs climb a ladder; when the top rung holds, the buildings step up and the
# legs drop back to the bottom.  So the composer learns to chain a long tour on
# an easy scene, then relearns it on a harder one -- rather than meeting longer
# legs and more obstacles at the same time and failing at both.
#
# The rungs are set by what the MAP can actually generate, measured from
# singapore_cbd.json's 33 waypoints rather than guessed:
#
#   closest pair anywhere      5.12 m     <- so max_leg=5 yields ZERO legal legs
#   nearest-neighbour median   6.29 m        and the task builder raises outright
#   longest pair              65.9  m     <- so a 100 m leg cannot exist either
#
#   max_leg   8    15    30    50    pairs available
#             27   92   303   485
#
# Hence 8 at the bottom (the shortest rung with any legal pairs at all) and 50
# at the top, roughly doubling the reachable task pool at each step.
LEG_LADDER = (8.0, 15.0, 30.0, 50.0)
LEG0, LEG_STEP, LEG_MAX, LEG_UP_AT = LEG_LADDER[0], 2.5, LEG_LADDER[-1], 0.9      # the judge's legs from the start, for the same reason      # legs start 5 m short of the judge's and step back up once the buildings are all in
LR_C = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-4        # the composer's rate, FIXED: no KL cap, no early stop, no backtracking (user's call)
# GRADIENT STEPS PER ITERATION (LES_EPOCHS), the scarce resource here.  The
# minibatch (4096) is larger than the whole batch of samples, so each epoch is
# ONE optimizer step -- at epochs=2 the composer moved twice per 25 s of wall
# clock, ~200 steps in 100 iterations for a 964k-parameter net.  Measured
# movement per update at lr 1e-4: token KL ~0.0001 and the argument mean
# shifting 0.02-0.06 sigma, about 100x below the 0.02 KL this project once used
# as a CAP.  The rollout costs 20 s and the update a fraction of that, so steps
# taken per rollout are nearly free and were being left on the table.
EPOCHS_C = int(_os.environ.get("LES_EPOCHS", "10"))
TEMPERATURE = 1.0        # MEASURED: raising it does not help.  384 flights on the
# same tasks at T = 1.0 / 1.4 / 1.8 / 2.5 moved the best-30% finish time by under
# 1% (0.2785 -> 0.2771 -> 0.2808 -> 0.2777) while reach collapsed 0.880 -> 0.185.
# More noise destroys good flights without finding better ones -- good behaviour
# is not a random perturbation away from the current policy.
#
# And variation was never the shortage.  Flying the SAME task twice with
# independent sampling, finish times differ by 0.120 on average against a
# between-task spread of 0.103, and only 9% of the ranking is a property of the
# task.  The policy already varies more than enough to select from; what was
# missing is the selecting.
# SHARPENS -- every update raises the likelihood of what was already done -- so
# without something widening the proposal the loop converges to its own mean and
# stops, which is exactly what the flat five iterations looked like (loss 0.256,
# match 0.82, reach 0.90, nothing moving).  This scales the token logits and the
# argument spread together, at SAMPLING time only: the likelihood the update
# fits is still the untempered policy, so it is a proposal, not a new objective.
KEEP_FRAC = 1.0          # RETIRED in favour of the arrival-time WEIGHT (see the
# 2x2 above): the cut discarded two thirds of the arrivals outright and
# quantized the rest into one bucket, and with a single flight per task it was
# ranking goals rather than decisions.  Kept at 1.0 so nothing is cut; the
# weight is the only selector.  The old comment, still true about why some
# selection is needed at all:
#                        # imitate only the FASTEST 30% of the flights that arrived.
# With ~90% of flights arriving, "keep the successes" kept almost everything,
# and cross-entropy on 90% of your own behaviour is a fixed point: measured, the
# loss sat at 0.256 and the batch reach at 0.90 for five straight iterations
# with nothing moving.  Imitation only improves a policy when the kept set is
# BETTER than the average, so the successes are ranked by how much of the
# episode they needed (`finish_frac`) and only the quickest are copied.  That
# adds no term to the objective -- arriving sooner is already implicit in the
# goal -- it only chooses which successes to learn from.
IMITATE = True           # train by CROSS-ENTROPY on the composer's own successful
# flights (user: "is there no cross entropy function we can use?").  Sample,
# keep the flights that arrived, raise the likelihood of exactly the tokens
# they emitted -- rejection-sampling fine-tuning, as a language model is
# post-trained.  It uses the outcome as a FILTER over whole flights instead of
# as a per-decision signal, which is the step that fails here: measured, the
# advantage knew the outcome at z = -166 and the braking argument at -0.013,
# because consecutive decisions in a flight differ by 3 against a spread of 241.
# It can only learn from success, so it depends on the curriculum keeping a
# supply of flights that arrive.
GOAL_ONLY = True         # reward ONLY arriving (user: "honestly just reward achieving
# the goal and see if that works").  The shaped return had a spread of 241,
# nearly all of it the death charge and the distance still owed, and it told the
# update almost nothing about WHICH decision mattered: measured on a real batch
# it separated crashed from surviving flights at z = -166 while its correlation
# with the braking argument r was -0.013.  A binary outcome makes the value
# head predict the PROBABILITY of arriving -- bounded, well conditioned, and
# exactly what a baseline should carry.
# START WITH OBSTACLES, not on an empty map.  The composer's perception is
# only worth anything when there is something to perceive, and the frozen low
# level already reaches 1.000 on an empty map by itself -- so every iteration
# spent at 0% buildings trains the one part of the net that was just repaired
# (the argument head, and the scene encoder underneath it) on a scene that
# cannot reward using a beam.  The previous run spent its first 65 iterations
# there clearing the leg ladder, then stalled for 506 iterations the moment
# buildings appeared.  10% is exactly where it stalled, so it is where an
# improvement shows up soonest.
# START EMPTY and scale.  The argument for starting at 10% was that
# perception is useless on an empty map -- true, but it assumed the composer
# already knew where the goal was.  With every hand-coded term removed it does
# not: an untrained head points at chance (measured, 85.7 deg mean bearing
# error against 6.4 with the residual).  Aiming at the goal has to be learned
# before avoiding anything means anything, and an empty map is where that is
# learnable.
# START AT 10% BUILDINGS, not on the empty map (LES_DIFF0 overrides).
# Measured: starting empty TEACHES SILENCE and silence is absorbing.  The
# frozen low level alone is 1.000 on the empty map at 8 m legs, so every token
# the composer emits costs it points (0.917 for one injected token), signed
# weights correctly push those tokens down, and arrival climbs 0.832 -> 0.950
# purely by shutting up -- subgoals 0.3 -> 0.0 by iteration 38.  Then
# `progress_weights` credits ONLY a decision that placed a subgoal, so a
# composer that places nothing yields zero samples, a nan loss and no gradient
# at all.  The curriculum meanwhile read 0.950 as "ready" and promoted it to
# 10% buildings, where silence crashes 18% of flights and nothing can be
# learned because nothing is emitted.  An empty map measures nothing: there is
# no use for a composer on it, so the only lesson available there is to be
# quiet, and that lesson cannot be unlearned.
DIFF0 = float(_os.environ.get("LES_DIFF0", "0.1"))
DIFF_STEP, DIFF_UP_AT = 0.1, 0.95
DIFF_WINDOW = 5          # the bar must hold over the LAST 5 iterations, not one.
# A single batch of 1152 flights still swings by several points draw to draw, so
# promoting on one reading steps up on a lucky sample and then sits stuck at a
# difficulty it never actually mastered.  The window resets on every step, so
# each new level has to be earned from scratch rather than coasting on the
# scores from the easier one.
# Start on an EMPTY map and raise the difficulty in small steps -- the same
# treatment that made the low level converge in ten generations after fifty
# composer updates had moved nothing.  `difficulty` is the fraction of the 60
# buildings left active (floor of one), so 0.0 is effectively open ground.
#
# The reason is measured: on the full city 72% of flights crash, crashes there
# flip under perturbations the policy does not control, and the return spread
# is ~285 against a decision worth ~25.  An empty map removes exactly that
# noise, which is what the low level needed and never had here.  Ten steps of
# 0.1 rather than four of 0.25, because 0 -> 0.25 is already fifteen buildings.
#
# The JUDGE stays on the whole city at every step, so the yardstick never moves
# while the training scene fills in.
LR_LL = float(sys.argv[7]) if len(sys.argv) > 7 else 5e-3      # the low level's Adam rate: slots have median magnitude 0.33, so 1.5% a step
N_BPTT, B_BPTT, K_BPTT = 4, 48, 100                            # BPTT batches an iteration, episodes a batch, steps a window
# A HELD lateral offset (sigma 0.5 on the subgoal's lateral axes, drawn once
# a flight) was tried and reverted: a subgoal held beside a 0.25 m tolerance
# is never achieved, every explored flight failed, and two updates on those
# returns took the mean policy to 0.000 reach.  Exploration must vanish at
# the goal; a held bearing needs a state-scaled sigma, not this.
SIGMA_XY = float(sys.argv[8]) if len(sys.argv) > 8 else SIGMA0_COMPOSER
NOISE_HOLD = sys.argv[9] if len(sys.argv) > 9 else 1                # decisions a draw is held; "flight" = once a flight
JUDGE_N = 256                                                  # judged every 10 on 256 episodes: half the noise of 128 every 5, same cost
TARGET_KL, LR0, LR_MAX = 0.02, 2e-5, 1e-3     # the composer's trust region; the rate adapts to it
W = f"{SP}/v11_{ARM}_composer.pt"; G = f"{SP}/v11_{ARM}_genome.json"; STATE = f"{SP}/v11_{ARM}_state.json"
W0 = ""   # no warm start: the token checkpoint has a different action head
# The frozen stage-A low level every arm flies.  v10 never named this because
# its genome file already existed; the arms each get their own path, so the
# seed has to be explicit or the first arm to run NameErrors here.
G0 = f"{SP}/cotrain_v10_genome.json"
NET_SEED = 1234   # all four arms start from the SAME composer, or the 2x2 measures the draw
SENSORS = ("range", "range_down", "depth_camera", "tilt", "map_prior")   # v9: BOTH maps
# The carried survey (nearest 12 footprints inside a 30 m viewport) and the map
# the vehicle builds from its own returned beams, offered as aged measurement
# tokens.  See composer/tokens.py and mapping.py; the 2x2 in map_ab.py is the
# controlled comparison, this is the run that commits to the combination.
# A small credit for airspeed in the COMPOSER's return only (the plant cost the
# GA evolves on is untouched).  0.25 per 0.4 s interval at the 5 m/s limit is
# about 16 of a ~400 return, ~4%: enough to break a tie between loitering and
# moving, far too small to outweigh the distance charge, so circling to farm it
# still loses.  Aimed at the ~20% of flights that end alive but unfinished.
SPEED_BONUS, SPEED_REF = 0.25, 5.0
MAP_PRIOR_KW = (("k", 12), ("max_range", 30.0))
BUILT_MAP_KW = (("cell", 2.0), ("k", 12), ("extent", 72.0), ("max_range", 30.0))
SENSOR_KW = (("range", (("spread", 6.2831853),)), ("map_prior", MAP_PRIOR_KW))                 # a 360-degree fan: 24 beams, 15 degrees apart
TKW = (("learned", True), ("damp_mode", "beams"), ("extra_obs", (("range_down", 4), ("tilt", 3))), ("split_terms", True))   # PULL and BRAKE as separate constraint terms

def cfg_for(env, max_leg, steps, n, composer="policy_cont", early=False, sensors=SENSORS, skw=SENSOR_KW, tkw=TKW, yaw=True, weights=None):
    w = W if weights is None else weights
    # fresh exploration noise per decision: a decision is now an event, not a tick
    ckw = (("reach", 10.0), ("every", EVERY), ("measure_every", EVERY_M), ("k_chain", K_CHAIN), ("temperature", TEMPERATURE), ("noise_hold", NOISE_HOLD), ("explore_eps", EXPLORE_EPS), ("tok_frac", TOK_FRAC), ("inject", INJECT), ("speak_floor", SPEAK_FLOOR),
           ("var_temp", VAR_T), ("var_k", VAR_K)) + ((("weights", w),) if w else ()) if composer else ()   # kids fly the parent's decisions: the GA compares low levels, not dice
    return Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment=env,
                  sensors=sensors, sensor_kw=skw, gating="arrival", seed=0, composer=composer, composer_kw=ckw,
                  task_kw=(("n_legs", 2), ("max_leg", max_leg)),
                  system_kw=(("prox_gain", 30.0), ("free_start", True), ("reset_yaw", 3.14159265), ("speed_limit", 5.0)) + ((("yaw_mode", "lagrangian"),) if yaw else ()),   # 5 m/s airspeed limit (user)
                  trainable_kw=tkw,
                  # the task and nothing else: distance over time, the bonus for
                  # arriving, the charge for dying.  No saturation, effort or
                  # time-to-collision terms -- those are for the layers to learn.
                  rollout=RolloutCfg(n_eps=n, ep_steps=steps, lambda_s=0.0, lambda_e=0.0, dead_mode="constant", compile_forward=True,
                                     dead_cost=DEAD_COST, goal_bonus=GOAL_BONUS, stop_on_arrival=early, lambda_ttc=0.0,
                                     # 1.0, NOT 0.9.  `rollout.py` reads this as "end the batch
                                     # once this FRACTION OF EPISODES HAS ARRIVED", and every
                                     # flight still in the air at that moment is scored as a
                                     # failure.  At 0.9 the reported arrival rate was truncated
                                     # at the quantile by construction: v10 sat at 0.908 over
                                     # thirteen iterations while the identical config measured
                                     # 0.988 with the early exit off.  The curriculum bar is
                                     # 0.95, so it could never be met and the composer spent its
                                     # entire life at 0% buildings -- and `finish_frac` was 1.0
                                     # for those truncated rows, so the imitation filter was
                                     # told ~8% of the batch had failed when it was still flying.
                                     # At 1.0 the exit is EXACT (everyone arrived or died), which
                                     # still skips the dead tail and costs only the time the
                                     # stragglers actually need.
                                     stop_quantile=1.0,
                                     built_map=True, built_map_kw=BUILT_MAP_KW,
                                     stop_finished=STOP_FINISHED if early else 0.0))   # training batches end once this fraction of the survivors has arrived




def log(m): print(m, flush=True); open(f"{SP}/cotrain_v8.log", "a").write(m + "\n")

import os, shutil
tcfg = cfg_for("singapore_cbd", LEG_MAX, T_TRAIN, E, early=True)
sysm, tr, task = build(tcfg)
FRESH = False                                             # v8 = v7's best pair placed in the corridor city (user: it struggles most in enclosed environments)
if not FRESH and not os.path.exists(G): shutil.copy(G0, G)
if W0 and not FRESH and not os.path.exists(W): shutil.copy(W0, W)   # W0 empty: start from the prior
if FRESH and not os.path.exists(G):
    json.dump({"theta": tr.init().tolist()}, open(G, "w"))     # the low level's own prior: the quadratic bowl, braking at 0.69 a beam
th = torch.tensor(json.load(open(G))["theta"], dtype=torch.float64)
if th.numel() == tr.dim + 3:
    # a genome from yaw_mode='learned': its last three slots were the look-at weights, which the Lagrangian mode has no use for
    th = th[:tr.dim]; json.dump({"theta": th.tolist()}, open(G, "w"))
assert th.numel() == tr.dim, (th.numel(), tr.dim)
torch.manual_seed(NET_SEED)
comp = build_composer(tcfg if os.path.exists(W) else cfg_for("singapore_cbd", LEG_MAX, T_TRAIN, E, early=True, weights=""), sysm, tr)
torch.save(comp.net.state_dict(), W)                 # a fresh net's random body and its HOLD / continue-straight prior, when no file yet
V = comp.net.vocab
torch.set_num_threads(1)
# `ParallelRollout` used to shard only the POPULATION, so with the low level
# frozen at a single genome there was nothing to split: one process flew all
# 576 episodes while nine cores idled.  It now splits the EPISODE axis whenever
# the population cannot fill the workers, which is exactly this case.  One
# thread per worker: six workers at two threads each oversubscribes a 4+6 core
# machine, and the parent needs cores for the update running beside them.
_W = WORKERS
par = ParallelRollout({"cfg": tcfg, "compile_workers": COMPILE_WORKERS}, workers=_W,
                      min_pop=max(1, _W), shards=P, shard_axis="auto", threads=1)
par._pool_up()
torch.set_num_threads(3)   # the parent runs the update concurrently with the workers' rollout
import hashlib
def gkey(t): return hashlib.sha1(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
evidence = {}                                            # genome -> (sum of paired diffs vs the parent, sum of squares, episodes)
Pm = identity_preconditioner(th.shape[-1], th.dtype, th.device)
TH = (th[None].clone() if FREEZE_LOW else
      (lambda t: (t.__setitem__(0, th), t)[1])(whitened_mutation(th.expand(P, -1).clone(), SIGMA, Pm.P, make_gen(77))))
tasks = {}
def sampler(max_leg):
    if max_leg not in tasks:
        _, _, tasks[max_leg] = build(cfg_for("singapore_cbd", max_leg, T_TRAIN, E, early=True))
    return tasks[max_leg]

def token_use(instr):
    """What the composer said, over a judged batch: tokens per flight and their kinds."""
    if not instr: return 0.0, {}
    ids = torch.cat([e[1][e[2]] for e in instr]); n = ids.numel() / instr[0][1].shape[0]
    # five token TYPES now, so report them by name rather than by id ranges
    kinds = {"place": float((ids == V.WAYPOINT).double().mean()),
             "priority": float((ids == V.PRIORITY).double().mean()),
             "heading": float(((ids == V.TURN) | (ids == V.LOOK)).double().mean())}
    return n, kinds

def judge(env, max_leg, steps, seed, n=JUDGE_N):
    c = cfg_for(env, max_leg, steps, n)
    s2, t2, k2 = build(c); comp.stochastic = True; comp.records = []; comp.record_rows = torch.zeros(0, dtype=torch.long); comp.reset(n)   # the ONE policy, sampled; nothing recorded
    r2 = Rollout(s2, t2, k2, c.rollout, build_sensors(c, s2), composer=comp)
    with torch.no_grad():
        res = r2.run(th[None], k2.sample(n, make_gen(seed)), seed + 1)
    spoke, kinds = token_use(comp._instr)
    dead = ~res.alive; t_dead = res.death_step[dead].double() * c.rollout.dt
    crash = (float(t_dead.median()) if dead.any() else float("nan"), float((t_dead < 1.0).double().mean()) if dead.any() else float("nan"))
    return (float(res.success.double().mean()), 1 - float(res.alive.double().mean()), float(res.fitness.mean()),
            float(res.cost.std() / n ** 0.5), float(r2.n_subgoals.double().mean()), spoke, kinds, crash)

def judge_all(tag):
    c = judge("singapore_cbd", 20.0, 1800, 9_900_001)
    k = c[6]
    log(f"  {tag:>9}  city20 {c[0]:.3f}/{c[1]:.3f}  cost {c[2]:7.2f} +-{c[3]:4.1f}  |  crashes: median {c[7][0]:4.1f} s, {c[7][1]:.0%} inside 1 s  |  "
        f"subgoals/flight {c[4]:4.1f}  spoke {c[5]:4.1f}/flight (place {k.get('place', 0):.0%} priority {k.get('priority', 0):.0%} heading {k.get('heading', 0):.0%})")
    return c

state = json.load(open(STATE)) if os.path.exists(STATE) else {"best_cost": float("inf"), "leg": LEG0, "difficulty": DIFF0}
_recent = []                                              # the last DIFF_WINDOW batch reaches
best_cost = float(state["best_cost"]); DIFF = float(state.get("difficulty", DIFF0))
LEG_I = int(state.get("leg_i", 0)); LEG = LEG_LADDER[min(LEG_I, len(LEG_LADDER) - 1)]
if LEG_FIX > 0:
    LEG = LEG_FIX; LEG_I = len(LEG_LADDER) - 1      # pinned: no promotion off this rung
if DIFF_FIX >= 0:
    DIFF = DIFF_FIX
opt_c = torch.optim.Adam([p for p in comp.net.parameters() if p.requires_grad], lr=LR_C)
def save_state(): json.dump({"best_cost": best_cost, "leg": LEG, "leg_i": LEG_I, "difficulty": DIFF}, open(STATE, "w"))
def save_best():
    shutil.copy(W, f"{SP}/v11_{ARM}_composer_best.pt"); shutil.copy(G, f"{SP}/v11_{ARM}_genome_best.json"); log("             <-- BEST pair saved")
log(f"\n{'='*78}\nco-training v7: a 360-degree fan; airspeed limited to 5 m/s; yaw is the Lagrangian's (the potential's gradient through the beams is the yaw torque; no heading rule); the composer emits TYPED TOKENS WITH CONTINUOUS ARGUMENTS ({V.V} types: EOS, WAYPOINT(r,theta), TURN(theta), PRIORITY(w), LOOK -- r=-1 is a full stop, r=+1 the whole reach, theta from the nose), one per report every {EVERY_M} steps; the low level by the GA ({P} x {E}, mutation {SIGMA})\n"
    f"the judge's legs ({LEG:.0f} m); the scene from {DIFF:.0%} of the buildings, +{DIFF_STEP:.0%} whenever the mean half reaches {DIFF_UP_AT:.0%}; judged on the whole city ({JUDGE_N} episodes) every {JUDGE_EVERY}\n"
    f"cost = distance to go + bonus {GOAL_BONUS:.0f} + death {DEAD_COST:.1f}/s, {SUBGOAL_COST:.1f} per PLACE; PPO uncapped at rate {LR_C:.1e}, 3 epochs, Adam kept; ONE policy: every flight samples its tokens, training and judging alike\n{'='*78}")
start = judge_all("start")
if start[2] < best_cost: best_cost = start[2]; save_best()
save_state(); t0 = time.time()
# The composer's update runs CONCURRENTLY with the next rollout: batch k+1
# flies the weights published after update k-1 while update k runs on batch
# k's samples, which carry their behaviour logits for the ratio.  An
# iteration is then the longer phase, not the sum (measured: rollout ~42 s,
# update ~60 s, sequential).
import threading
pending, st_prev, t_u_prev = None, {"kl": float("nan"), "clipfrac": float("nan"), "speak": float("nan"), "entropy": float("nan"), "n": 0}, 0.0
def _update(groups_r, groups_R, holder, groups_win=None, groups_score=None, groups_task=None, groups_left=None, groups_adv=None):
    t = time.time()
    # target_kl is a REAL cap here: the argument mean is unbounded and the last
    # continuous composer this project ran was uncapped and reached KL 18
    if LOSS == "time":
        # PURELY TIME: fit T_hat to the time each decision actually needed,
        # then move the policy down T_hat.  Nothing else is in it.
        holder["st"] = time_update(comp.net, groups_r, groups_left, ep_steps=T_TRAIN,
                                   epochs=EPOCHS_C, batch=4096, lr=LR_C, opt=opt_c,
                                   max_samples=MAX_SAMPLES)
    elif LOSS == "error":
        # No reward, no return, no labels: the composer claims the vehicle can
        # reach g by the next decision and reality answers.  See error_update.
        holder["st"] = error_update(comp.net, groups_r, reach=10.0, epochs=EPOCHS_C, batch=4096,
                                    lr=LR_C, opt=opt_c, max_samples=MAX_SAMPLES,
                                    alpha=ERR_ALPHA, beta=ERR_BETA)
    elif IMITATE:
        holder["st"] = imitate_update(comp.net, groups_r, groups_win, epochs=EPOCHS_C, batch=4096,
                                      lr=LR_C, opt=opt_c, max_samples=MAX_SAMPLES,
                                      keep_frac=KEEP_FRAC, score=groups_score, advantage=groups_adv,
                                      weight_tau=WEIGHT_TAU, task=groups_task,
                                      signed=SIGNED, w_max=W_MAX,
                                      reach=10.0)
    else:
        holder["st"] = ppo_update_cont(comp.net, groups_r, groups_R, comp.n_terms, epochs=EPOCHS_C, batch=4096, lr=LR_C, vcoef=VCOEF, ent=0.0, target_kl=0.02, opt=opt_c, max_samples=MAX_SAMPLES)
    holder["t"] = time.time() - t
# The composer's updates train a CHALLENGER.  Measured (Sept 9, corridor
# city, judge-matched training): each update is near neutral on the batch and
# the crash rate creeps ~1.5 points per update; ten of them took the judge from
# 406 to 504 with the batch cost flat -- drift, not signal -- and a one-shot
# paired gate on 128 tasks resolves only +-40 where an update's effect is
# +-10 (two policies diverge within seconds, so pairing buys little).  So the
# evidence is POOLED: every iteration the challenger (the weights PPO keeps
# training) and the kept policy fly the same fresh tasks with paired draws;
# the paired difference accumulates; at pooled z >= 2.5 the challenger is
# adopted, at z <= -2.5 it reverts to the kept policy, and in between it keeps
# training.  The batches fly the challenger (PPO stays on-policy); the kept
# policy exists only to be beaten.
GA_ADOPT_Z = 0.0          # the low level adopts ANY kid that beats the parent on paired
# evidence, with no significance bar (user's call: "just let it adopt all
# changes and see if it changes quicker ... it might just be more minute").
# The 2.5 bar was set when a batch's return spread was ~415 and two task draws
# ranked the same 16 genomes at Spearman +0.02; it let only 2 adoptions through
# in the whole run while pooled scores sat at 0.5-2.1, just under it.  The
# per-task baseline has since halved that spread, so the paired evidence behind
# each comparison is better than when the bar was chosen.  The risk is the known
# one: at zero bar, selection on a noisy ranking is a random walk of the genome.
# The judge every 10 iterations is the check, and the best-judged pair is still
# checkpointed, so nothing is lost if it wanders.
# (gate constants removed with the adopt/revert mechanic)
# The revert was set at 1.0 pooled sigma after ten ungated updates drifted the
# judge 408 -> 512.  Measured here, it costs more than it saves: three reverts
# in fifteen iterations at z -1.1 and -1.3, which is noise, and each one rolls
# the live policy back AND resets the pooled evidence -- so a challenger that
# was climbing (+0.3, +0.5, +1.4, +1.0 over iterations 12-15) can never reach
# the +2.5 it needs to be adopted.  Training now always moves forward; the
# gate still MEASURES, so the log says whether it is actually improving, and
# the best-judged pair is still checkpointed as the safety net.
# The adopt/revert GATE is gone.  It trained a challenger, flew it and the kept
# policy on GATE_N fresh tasks every iteration, pooled the paired difference and
# adopted at z >= 2.5.  Removed because it cost two extra 96-episode rollouts an
# iteration (~20s) to arbitrate a decision that is no longer in doubt: the run
# trains continuously, REVERT_Z was already infinite so nothing was ever rolled
# back, and its verdict was reported one iteration out of step with the batch
# beside it.  The safety net that remains is the judge every JUDGE_EVERY
# iterations on a held-out batch, which still checkpoints the best pair.
def _join(it_=0):
    """Finish the update that has been running beside this rollout, and publish
    its weights so the next batch flies them."""
    global pending, st_prev, t_u_prev
    if pending is None:
        st_prev = dict(st_prev, n=0); t_u_prev = 0.0
        return
    _t0 = time.time(); pending[0].join()
    st_prev = pending[1].get("st", st_prev); t_u_prev = pending[1].get("t", 0.0); pending = None
    # the workers reload W by mtime, so a fresh write is how a new policy ships
    torch.save(comp.net.state_dict(), W)
    import os as _os
    log(f"    [phase it {it_}: waited on update {time.time() - _t0:.0f}s "
        f"(update itself {t_u_prev:.0f}s), load {_os.getloadavg()[0]:.0f}]")


def _sens(groups_r, n=192):
    """How much the composer's output depends on its PERCEPTION, measured on the
    batch it just flew -- no extra rollout.

    The training batch already records scene tokens to feed the update, then
    discards them; the judge records outcomes, not tokens, so it cannot answer
    this at all.  Here each sampled decision is re-run twice on IDENTICAL state:
    once as it was, once with the beam and pixel rows SHUFFLED between rows,
    which destroys their information while leaving the marginal distribution
    exactly as it was.  Blinding them to "nothing within range" is the weaker
    test -- most beams already read nothing, so it barely perturbs anything.

    Returned as the fraction of the token distribution that moves.  It was
    0.0001 while the composer was ignoring its sensors entirely.
    """
    from lagrangian_es.composer.tokens import BEAM, PIXEL
    toks = [r["tok"] for g in groups_r for r in g if r.get("tok")]
    if not toks:
        return float("nan")
    step = max(1, len(toks) // 8)
    dP, k = 0.0, 0
    with torch.no_grad():
        for tk in toks[::step][:8]:
            ty = tk["ent_types"]
            sel = (ty == BEAM) | (ty == PIXEL)
            if not bool(sel.any()) or ty.shape[0] < 2:
                continue
            t1 = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in tk.items()}
            e = t1["entities"].clone()
            m0 = sel[0]
            e[:, m0] = e[torch.randperm(e.shape[0])][:, m0]
            t1["entities"] = e
            p0 = torch.softmax(comp.net.pre(tk)[0], -1)
            p1 = torch.softmax(comp.net.pre(t1)[0], -1)
            dP += float((p0 - p1).abs().sum(-1).mean()); k += 1
    return dP / max(1, k)


row_prev = None
def _emit(r, st, t_u):
    """One iteration's batch, its own update, and the gate verdict on it."""
    _sel = (f"ess {st.get('ess', 0.0):5.0f}" if not SIGNED
            else f"up {st.get('w_up', float('nan')):.2f} |w| {st.get('w_abs', float('nan')):.2f}")
    log(f"  [{ARM}] iter {r['it']:>4}  buildings {r['diff']:4.0%} legs {r['leg']:4.0f}m  arrive {r['arrive']:.3f}"
        f" crash {r['crash']:.3f} t_arr {r['t_arr']:.3f} cost {r['cost']:7.2f} subgoals {r['subs']:4.1f}"
        # `nll` -- the pure TOKEN cross-entropy -- beside `loss`.  The loss is
        # not comparable across iterations: with sigma frozen at 0.12 each
        # Gaussian argument adds a constant +0.701 of log-density, so a
        # WAYPOINT (3 arguments) subtracts 2.104 from it and an EOS subtracts
        # nothing.  The loss therefore FALLS whenever the policy simply speaks
        # more -- measured: speak 0.250 -> 0.291 took the loss 0.072 -> 0.007
        # while arrival went 0.736 -> 0.719.  `nll` carries no argument term
        # and is the one to read.
        f"  | loss {st.get('ce', float('nan')):7.3f} nll {st.get('nll', float('nan')):5.3f} match {st.get('match', float('nan')):.2f}"
        f" on {st.get('n', 0)} tokens from {st.get('kept_flights', 0)}/{st.get('flights', 0)} weighted {_sel}"
        f" | speak {st.get('speak', float('nan')):.3f} std {st.get('std', float('nan')):.3f} sens {r.get('sens', float('nan')):.4f}"
        f"  [roll {r['t_r']:.0f}s update {t_u:.0f}s | {r['mins']:.1f}m]{r['note']}")


for it in range(1, OUTER + 1):
    task_it = sampler(LEG)
    t_r = time.time()
    # E/REPEAT distinct tasks, each flown REPEAT times.  The task seed is the
    # same on every pass, so the goals, the initial states and the sensor noise
    # are identical and only the composer's draws differ -- the k samples of a
    # task differ by DECISION alone, which is what makes their spread credit.
    E_T = E // REPEAT
    goals_it = task_it.sample(E_T, make_gen(5_000_000 + it))
    shards, res_list = [], []
    for _r in range(REPEAT):
        _res, _sh = par.run_with_records(TH, goals_it, 5_100_000 + it, stochastic=True,
                                         record_frac=REC, difficulty=DIFF, noise=_r)
        for _s in _sh:
            # carried with the shard, because the merged result of pass _r is
            # not addressable once the passes are pooled
            _s["win"] = _res.success[_s["rows"]]
            # SOFT TIME, not finish_frac.  finish_frac is 1.0 for every flight
            # that did not arrive, so all failures tie and the objective can
            # only ever give them weight zero -- which is why nothing in it
            # could reduce the failure rate.  soft_time accumulates
            # sigmoid((|err| - tol)/tol) while the flight was still going, so a
            # vehicle that nearly made it outranks one that crashed at once and
            # EVERY flight is rankable.
            _s["score"] = _res.soft_time[_s["rows"]]
            # PROJECTED time, not raw finish_frac: a flight that never arrived
            # is charged 1.0 plus what its remaining distance would have taken
            # at cruise, so two failures no longer tie at zero and a crash is
            # the most expensive outcome rather than an unranked one.
            _s["ff"] = projected_time(_res.finish_frac, _res.final_err,
                                      _res.success, SPAN)[_s["rows"]]
            # time still to run from each decision, for LOSS="time".  1.0 means
            # the flight never arrived, which is exactly "forever" as far as the
            # time model is concerned.
            _s["left"] = _res.finish_frac[_s["rows"]]
            _s["task"] = _s["rows"] % E_T           # P = 1, so the row IS the task
        shards += _sh; res_list.append(_res)
    # PAIRED CONTROL: the same tasks, the same seed, the same initial states and
    # sensor noise, flown with the composer MUTED.  What the frozen low level
    # does unaided is then present in both flights and cancels, so the
    # difference is what the composer's tokens were worth and nothing else.
    # Only the paired losses need a control.  LOSS="time" regresses the time a
    # decision actually needed and never looks at a twin, so flying one would
    # double the rollout for nothing.
    _ctl = None
    if LOSS != "time":
        _ctl, _ = par.run_with_records(TH, goals_it, 5_100_000 + it, stochastic=True,
                                       record_frac=0.0, difficulty=DIFF, noise=0, mute=True)
    # FINISH_FRAC, not soft_time.  soft_time accumulates only while a row is
    # alive, so a flight that crashes at step 100 scores LOWER -- looks faster --
    # than one that flies 500 steps and arrives; as a paired advantage that
    # would pay the composer to crash.  finish_frac is 1.0 for anything that
    # never arrived, which is exactly "took forever".  The reason soft_time was
    # preferred (failures all tie at 1.0 and cannot be ranked) does not apply
    # here: the pairing ranks them -- a failure the control also failed scores
    # 0, one the control won scores -1.
    if _ctl is not None:
        _ctl_ff = projected_time(_ctl.finish_frac, _ctl.final_err,
                                 _ctl.success, SPAN).reshape(-1)[:E_T].clone()
        for _s in shards:
            _s["adv"] = paired_advantage(_s["ff"], _ctl_ff[_s["task"]])
    else:
        for _s in shards:
            _s["adv"] = torch.zeros_like(_s["ff"], dtype=torch.float64)
    t_r = time.time() - t_r
    _cat = lambda f: torch.cat([getattr(_x, f) for _x in res_list])
    res = res_list[0]
    fit = torch.stack([_x.fitness for _x in res_list]).mean(0); order = fit.argsort()
    if FREEZE_LOW:
        # nothing to select: TH stays exactly as it was, and the genome on disk
        # is the stage-A one so a restart picks the same low level back up
        guard_note = "low level FROZEN"
        z_best = 0.0
        stride = max(1, int(round(1.0 / REC))); pol = (torch.arange(E) % stride) != 0
    else:
        # The parent of record (TH[0]) keeps its place unless a kid beats it by two
        # PAIRED standard errors on this batch.  Measured (Sept 9, corridor city):
        # two task draws ranked the same 16 genomes at Spearman +0.02, so an elite
        # chosen on raw means is a coin flip and the saved genome drifts; with the
        # composer's token draws paired across the population (crn_sample) a
        # kid-vs-parent difference on identical tasks and draws resolves 5 of 15
        # mutants at 2 sigma, so THAT comparison is the one worth acting on.
        # ... and the evidence is POOLED across batches: a kid that ranks into the
        # elite is re-flown next iteration anyway, so its paired difference against
        # the parent accumulates (one mutation moves the cost by about +-10 against
        # a 288-episode error of 10, so no single batch can clear 2 sigma; four to
        # six batches can).  Adopted at 2.5 pooled sigma over at least two batches.
        stride = max(1, int(round(1.0 / REC))); pol = (torch.arange(E) % stride) != 0     # the recorded (exploring) rows are every `stride`-th episode
        cpe = res.cost.reshape(P, E)[:, pol]; Ep = int(pol.sum()); fit = cpe.mean(1); order = fit.argsort()
        d = cpe - cpe[0:1]
        for i in range(1, P):
            k = gkey(TH[i]); s_, q_, n_ = evidence.get(k, (0.0, 0.0, 0))
            evidence[k] = (s_ + float(d[i].sum()), q_ + float((d[i] * d[i]).sum()), n_ + Ep)
        z_best, i_best, n_pooled = 0.0, None, 0
        for i in range(1, P):
            s_, q_, n_ = evidence[gkey(TH[i])]
            if n_ >= 2 * Ep:
                n_pooled += 1; m_ = s_ / n_; se_ = max((q_ / n_ - m_ * m_), 1e-12) ** 0.5 / n_ ** 0.5; z_ = -m_ / se_
                if z_ > z_best: z_best, i_best = z_, i
        adopted = i_best is not None and z_best >= GA_ADOPT_Z
        # PERSISTENCE by pooled evidence: the kids that stay in the elite (and are
        # re-flown, accumulating) are the ones whose pooled paired difference vs
        # the parent is best, not the ones a single batch's coin flip favoured;
        # a kid whose pooled z has fallen to -2.5 is out.
        score = torch.full((P,), float("-inf"), dtype=torch.float64)
        for i in range(1, P):
            s_, q_, n_ = evidence[gkey(TH[i])]; m_ = s_ / n_; se_ = max(q_ / n_ - m_ * m_, 1e-12) ** 0.5 / n_ ** 0.5; zi = -m_ / se_
            score[i] = zi if zi > -2.5 else float("-inf")
        if adopted:
            order = torch.cat([torch.tensor([i_best]), score.argsort(descending=True)]); evidence.clear()
        else:
            order = torch.cat([torch.zeros(1, dtype=torch.long), score.argsort(descending=True)])
        seen = set(); order = torch.tensor([int(i) for i in order.tolist() if int(i) not in seen and not seen.add(int(i))])
        elite = TH[order[: P // 2]]
        guard_note = f"pooled {n_pooled} kids, best z {z_best:4.1f}" + (" ADOPTED" if adopted else "")
        th = elite[0].clone(); json.dump({"theta": th.tolist()}, open(G, "w"))
        kids = whitened_mutation(elite.repeat(2, 1)[: P - P // 2], SIGMA, Pm.P, make_gen(7_000_000 + it))
        TH = torch.cat([elite, kids], 0)
    groups_r, groups_R, n_dec, n_sub = [], [], 0, []
    groups_task = []; groups_left = []; groups_adv = []
    _Rs, _rows, groups_win, groups_score = [], [], [], []
    for sh in shards:
        if GOAL_ONLY:
            # the shard numbers its rows by POSITION within the rows it kept, so
            # the success flags have to be gathered the same way
            R = returns_goal_only(sh["records"], sh["win"].double(), gamma=0.98)
        else:
            R = returns_from_stream(sh["records"], sh["chain"], GAMMA, subgoal_cost=SUBGOAL_COST, unit=EVERY_M,
                                    speed_bonus=SPEED_BONUS, speed_ref=SPEED_REF).double()
        if sh.get("n_sub") is not None: n_sub.append(float(sh["n_sub"]))
        _Rs.append(R); _rows.append(sh["rows"])
    # The TASK's difficulty, not the policy's doing, is most of a return's
    # spread: a crash books ~1100 against ~250 for a finish, and 55% of starts
    # on this city have a building within 10 m on the line to the goal.  Every
    # genome flies the same task list, so the mean over genomes of one task
    # estimates that difficulty, and removing it leaves what the decisions
    # changed.  Pooled ACROSS shards, because the pool splits the population
    # and one task's samples are scattered over all of them.
    _v0 = float(torch.cat([r.reshape(-1) for r in _Rs]).std())
    center_by_task(_Rs, _rows, E_T)
    _v1 = float(torch.cat([r.reshape(-1) for r in _Rs]).std())
    for sh, R in zip(shards, _Rs):                       # then the residual per-genome offset
        gid = sh["rows"] // E_T
        for g in gid.unique():
            m = gid == g; R[:, m] = R[:, m] - R[:, m].mean(1, keepdim=True)
        groups_r.append(sh["records"]); groups_R.append(R); groups_win.append(sh["win"]); groups_score.append(sh["score"]); groups_task.append(sh["task"]); groups_left.append(sh["left"]); groups_adv.append(sh["adv"]); n_dec += sum(int((r["alive"] & r["tok_keep"]).sum()) if r.get("tok_keep") is not None else int(r["alive"].sum()) for r in sh["records"])   # the samples the update can use
    sens = _sens(groups_r)                                # free: the batch it just flew
    _join(it)                                             # update k-1 done: gate, publish, then start update k
    if n_dec:
        holder = {}; th_ = threading.Thread(target=_update, args=(groups_r, groups_R, holder, groups_win, groups_score, groups_task, groups_left, groups_adv), daemon=True); th_.start()
        pending = (th_, holder)
    st, t_u = st_prev, t_u_prev
    # Every row, not the unrecorded half.  The recorded rows used to explore on
    # top of the policy, so the batch numbers were read off the others; with
    # EXPLORE_EPS at 0 both halves sample from the same policy and splitting
    # them only made the printed arrival rate disagree with the update's.
    _suc = _cat("success").double()
    tr_r = float(_suc.mean()); tr_c = 1 - float(_cat("alive").double().mean())
    subs = sum(n_sub) / len(n_sub) if n_sub else float("nan")
    _ff = _cat("finish_frac").double()
    tr_t = float(_ff[_ff < 1.0].mean()) if bool((_ff < 1.0).any()) else float("nan")   # mean arrival time, arrivals only
    de_r, de_cost = tr_r, float(_cat("cost").mean())
    note = ""; flown_diff, flown_leg = DIFF, LEG
    _recent.append(de_r)
    if len(_recent) > DIFF_WINDOW:
        _recent.pop(0)
    _held = (len(_recent) == DIFF_WINDOW and (sum(_recent) / DIFF_WINDOW) >= DIFF_UP_AT
             and LEG_FIX <= 0 and DIFF_FIX < 0)      # a pinned setting never promotes
    if _held:
        if LEG_I < len(LEG_LADDER) - 1:
            # the inner ladder: longer legs on the scene it already handles
            LEG_I += 1; LEG = LEG_LADDER[LEG_I]
            note = f"  -> legs {LEG:.0f} m"
        elif DIFF < 1.0:
            # the top rung held, so the scene gets harder and the legs restart
            DIFF = min(DIFF + DIFF_STEP, 1.0)
            LEG_I = 0; LEG = LEG_LADDER[0]
            note = f"  -> buildings {DIFF:.0%}, legs back to {LEG:.0f} m"
        _recent.clear()
    save_state()
    # ONE LINE, ONE ITERATION.  The update runs concurrently with the next
    # rollout, so when iteration k prints, the freshest update results belong
    # to batch k-1 -- the line used to mix batch k's arrival rate with update
    # k-1's loss and gate verdict, and an ADOPTED read as though it belonged to
    # the batch beside it.  Rather than give up the concurrency (which would
    # cost roll + update instead of max(roll, update)), the batch stats are
    # HELD and printed one iteration later, next to their own update.  The log
    # trails real time by an iteration; nothing on a line is from a different
    # one.
    row = {"it": it, "diff": flown_diff, "leg": flown_leg, "arrive": tr_r, "crash": tr_c,
           "t_arr": tr_t, "cost": de_cost, "subs": subs, "note": note, "t_r": t_r,
           "mins": (time.time() - t0) / 60, "sens": sens}
    if row_prev is not None:
        _emit(row_prev, st, t_u)
    row_prev = row

    if it % JUDGE_EVERY == 0:
        _join(it)                                         # the judge flies the finished weights
        c = judge_all(f"judge {it}")
        if c[2] < best_cost:
            best_cost = c[2]; save_state(); save_best()
_join()
if row_prev is not None:
    _emit(row_prev, st_prev, t_u_prev)      # the final batch's own update
par.close()
log(f"COTRAIN V11 {ARM} DONE")
