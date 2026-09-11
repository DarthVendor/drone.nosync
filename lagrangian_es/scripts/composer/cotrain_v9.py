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
from lagrangian_es.composer import center_by_task, ppo_update, returns_from_stream
from lagrangian_es.es import build, build_composer, build_sensors
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
STOP_FINISHED = 0.8                                       # the adaptive cap BOUNDS the batch reach, so it must sit above the curriculum bar (0.8 held reach at 0.80 under a 90% bar)
GOAL_BONUS = 60.0                                          # per leg held (was 15): a finish must outrank the 50-80 spread the placements put on the time integral (user)
# Measured 2026-09-08 (diag_genome.py): at 48 x 96 and sigma 0.012 the spread of
# fitness across mutants (6.5) was SMALLER than one genome's 96-episode noise
# (10.4) and the ranking on two batches agreed at Spearman -0.11 -- selection
# was ranking noise, the genome a random walk.  At sigma 0.036 the ranking
# agreed at +0.28 and the elite of one batch beat the rest on the other by
# 4.4.  So: a third of the genomes with three times the episodes each (noise
# 6.0 < spread), and double the mutation.  Same 4608 rows an iteration.
EXPLORE_EPS = 0.3                                        # the recorded rows draw from the policy mixed with a uniform: rare tokens ~100-170 samples a batch instead of 3-31 (0.6 wrecked the exploring flights: reach 0.014)
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
WORKERS, GAMMA, JUDGE_EVERY, REC = 6, 0.99, 10, (0.5 if FREEZE_LOW else 0.25)
# REC was held down because a recorded row EXPLORES, and an exploring row is no
# use to the genetic ranking.  With the low level frozen there is no ranking to
# protect, so half the batch can be recorded instead of a quarter: ~1150
# recorded flights an update, the same volume as before, from a batch of
# distinct tasks rather than 288 flown sixteen times.   # FULL horizon: a token's effect on the expected outcome keeps growing past 2 s (straight PLACE +0.5 at 2 s, +23 over the flight); the 2 s horizon threw that away    # gamma per report (0.2 s): a 2 s horizon, the measured time for two near-identical flights to separate; at 0.99 (20 s) the half-batch gradients agreed at cos 0.42, at 0.9 at 0.87   # a quarter of the rows explore and record: one token per report per row is many samples
MAX_SAMPLES = 10_000
COMPILE_WORKERS = False                                    # the workers compile the controller's forward passes (one compile thread each; the parent warms the compiler before the fork)
TOK_FRAC = 0.1                                             # scene tokens kept for this fraction of the recorded rows: ~15k samples for a 10k update instead of ~150k (600 MB per worker)                                       # sized so the update (3 threads, beside the rollout) finishes inside the rollout
T_TRAIN = 1800                                             # the JUDGE's horizon: on 18 s episodes a crash at 10 s cost 320, on the judge's 36 s 1040, and the composer drifted bold (crash +1.5 points per update) while its training cost fell -- train on what is judged (user: the curriculum must match the judge)
P, E, SIGMA = (1, 2304, 0.0) if FREEZE_LOW else (16, 288, 0.024)                                   # measured: at 48 x 96 / 0.012 the ranking was noise (Spearman -0.11); here the noise (6.0) is below the spread
LEG0, LEG_STEP, LEG_MAX, LEG_UP_AT = 20.0, 2.5, 20.0, 0.9      # the judge's legs from the start, for the same reason      # legs start 5 m short of the judge's and step back up once the buildings are all in
LR_C = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-4        # the composer's rate, FIXED: no KL cap, no early stop, no backtracking (user's call)
DIFF0, DIFF_STEP, DIFF_UP_AT = 1.0, 0.25, 0.9                   # the scene: buildings from none to the whole city, +25% whenever the batch reaches 90% (user: the bar was 60%); the judge is always the whole city
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
W = f"{SP}/cotrain_v9_composer.pt"; G = f"{SP}/cotrain_v9_genome.json"; STATE = f"{SP}/cotrain_v9_state.json"
W0 = f"{SP}/cotrain_v8_composer_best.pt"; G0 = f"{SP}/cotrain_v8_genome_best.json"   # v7's judged-best pair (judge 110: 0.488/0.277, cost 581 on the city)
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

def cfg_for(env, max_leg, steps, n, composer="policy", early=False, sensors=SENSORS, skw=SENSOR_KW, tkw=TKW, yaw=True, weights=None):
    w = W if weights is None else weights
    # fresh exploration noise per decision: a decision is now an event, not a tick
    ckw = (("reach", 10.0), ("every", EVERY), ("measure_every", EVERY_M), ("noise_hold", NOISE_HOLD), ("explore_eps", EXPLORE_EPS), ("explore_balance", True), ("follow_parent", True), ("tok_frac", TOK_FRAC)) + ((("weights", w),) if w else ()) if composer else ()   # kids fly the parent's decisions: the GA compares low levels, not dice
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
                                     stop_quantile=0.9 if early else 1.0,
                                     built_map=True, built_map_kw=BUILT_MAP_KW,
                                     stop_finished=STOP_FINISHED if early else 0.0))   # training batches end once this fraction of the survivors has arrived




def log(m): print(m, flush=True); open(f"{SP}/cotrain_v8.log", "a").write(m + "\n")

import os, shutil
tcfg = cfg_for("singapore_cbd", LEG_MAX, T_TRAIN, E, early=True)
sysm, tr, task = build(tcfg)
FRESH = False                                             # v8 = v7's best pair placed in the corridor city (user: it struggles most in enclosed environments)
if not FRESH and not os.path.exists(G): shutil.copy(G0, G)
if not FRESH and not os.path.exists(W): shutil.copy(W0, W)
if FRESH and not os.path.exists(G):
    json.dump({"theta": tr.init().tolist()}, open(G, "w"))     # the low level's own prior: the quadratic bowl, braking at 0.69 a beam
th = torch.tensor(json.load(open(G))["theta"], dtype=torch.float64)
if th.numel() == tr.dim + 3:
    # a genome from yaw_mode='learned': its last three slots were the look-at weights, which the Lagrangian mode has no use for
    th = th[:tr.dim]; json.dump({"theta": th.tolist()}, open(G, "w"))
assert th.numel() == tr.dim, (th.numel(), tr.dim)
comp = build_composer(tcfg if os.path.exists(W) else cfg_for("singapore_cbd", LEG_MAX, T_TRAIN, E, early=True, weights=""), sysm, tr)
torch.save(comp.net.state_dict(), W)                 # a fresh net's random body and its HOLD / continue-straight prior, when no file yet
V = comp.net.vocab
torch.set_num_threads(1)
# `ParallelRollout` shards the POPULATION, so with the low level frozen at a
# single genome there is nothing for it to split: it would fork six workers,
# hand everything to one, and leave the rest holding ~500 MB each.  One process
# with all the threads is both faster and lighter here.
_W = 1 if FREEZE_LOW else WORKERS
par = ParallelRollout({"cfg": tcfg, "compile_workers": COMPILE_WORKERS}, workers=_W, min_pop=max(1, _W), shards=P)
if not FREEZE_LOW:
    par._pool_up()   # P shards over 8 workers on 4 performance + 6 efficiency cores
torch.set_num_threads(6 if FREEZE_LOW else 2)   # frozen: one process, so give it the cores; otherwise leave them to the workers
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
    kinds = {"place": float(V.is_place(ids).double().mean()), "priority": float(((ids >= V.RAISE0) & (ids < V.TURN0)).double().mean()),
             "heading": float((ids >= V.TURN0).double().mean())}
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
best_cost = float(state["best_cost"]); LEG = float(state.get("leg", LEG0)); DIFF = float(state.get("difficulty", DIFF0))
opt_c = torch.optim.Adam([p for p in comp.net.parameters() if p.requires_grad], lr=LR_C)
def save_state(): json.dump({"best_cost": best_cost, "leg": LEG, "difficulty": DIFF}, open(STATE, "w"))
def save_best():
    shutil.copy(W, f"{SP}/cotrain_v9_composer_best.pt"); shutil.copy(G, f"{SP}/cotrain_v9_genome_best.json"); log("             <-- BEST pair saved")
log(f"\n{'='*78}\nco-training v7: a 360-degree fan; airspeed limited to 5 m/s; yaw is the Lagrangian's (the potential's gradient through the beams is the yaw torque; no heading rule); the composer emits ACTION TOKENS (vocabulary of {V.V}: HOLD, {V.n_place} placements, {2 * V.n} priority moves, {len(V.TURNS)} turns, LOOK), one per report every {EVERY_M} steps; the low level by the GA ({P} x {E}, mutation {SIGMA})\n"
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
def _update(groups_r, groups_R, holder):
    t = time.time()
    holder["st"] = ppo_update(comp.net, groups_r, groups_R, comp.n_terms, epochs=2, batch=1024, lr=LR_C, vcoef=VCOEF, ent=0.0, target_kl=0.0, opt=opt_c, max_samples=MAX_SAMPLES)
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
GATE_N, GATE_Z, REVERT_Z = 96, 2.5, float("inf")         # adopt at 2.5 pooled sigma; NEVER revert (user's call: "just let it train")
# The revert was set at 1.0 pooled sigma after ten ungated updates drifted the
# judge 408 -> 512.  Measured here, it costs more than it saves: three reverts
# in fifteen iterations at z -1.1 and -1.3, which is noise, and each one rolls
# the live policy back AND resets the pooled evidence -- so a challenger that
# was climbing (+0.3, +0.5, +1.4, +1.0 over iterations 12-15) can never reach
# the +2.5 it needs to be adopted.  Training now always moves forward; the
# gate still MEASURES, so the log says whether it is actually improving, and
# the best-judged pair is still checkpointed as the safety net.
gate_cfg = cfg_for("singapore_cbd", LEG_MAX, 1800, GATE_N)
gate_sys, gate_tr, gate_task = build(gate_cfg); gate_sens = build_sensors(gate_cfg, gate_sys)
kept = {k: v.detach().clone() for k, v in comp.net.state_dict().items()}
chal_ev = [0.0, 0.0, 0]                                  # sum of paired diffs (challenger - kept), sum of squares, n
gate_note = ""
def gate_costs(state, it_):
    # on the worker pool: the workers reload W by mtime, so write the weights there first (a fresh mtime each time)
    torch.save(state, W); time.sleep(1.05)
    r, _ = par.run_with_records(th[None], gate_task.sample(GATE_N, make_gen(9_900_000 + it_)), 9_910_000 + it_, stochastic=True, record_frac=0.0, difficulty=DIFF)
    return r.cost.clone()
def _join(it_=0):
    global pending, st_prev, t_u_prev, kept, gate_note, chal_ev
    if pending is not None:
        _t0 = time.time(); pending[0].join(); st_prev = pending[1].get("st", st_prev); t_u_prev = pending[1].get("t", 0.0); pending = None
        _t1 = time.time(); cand = {k: v.detach().clone() for k, v in comp.net.state_dict().items()}
        _c1 = gate_costs(cand, it_); _t2 = time.time(); _c0 = gate_costs(kept, it_); _t3 = time.time()
        d = _c1 - _c0
        import os as _os; log(f"    [phase it {it_}: waited on update {_t1 - _t0:.0f}s (update itself {t_u_prev:.0f}s), gate cand {_t2 - _t1:.0f}s, gate kept {_t3 - _t2:.0f}s, load {_os.getloadavg()[0]:.0f}]")
        chal_ev[0] += float(d.sum()); chal_ev[1] += float((d * d).sum()); chal_ev[2] += GATE_N
        n_ = chal_ev[2]; m_ = chal_ev[0] / n_; se_ = max(chal_ev[1] / n_ - m_ * m_, 1e-12) ** 0.5 / n_ ** 0.5; z_ = -m_ / se_
        if n_ >= 2 * GATE_N and z_ >= GATE_Z:
            kept = cand; chal_ev = [0.0, 0.0, 0]; gate_note = f"composer ADOPTED (pooled {m_:+.1f} +- {se_:.1f}, z {z_:.1f})"
        elif n_ >= 2 * GATE_N and z_ <= -REVERT_Z:
            cand = kept; chal_ev = [0.0, 0.0, 0]; gate_note = f"composer REVERTED (pooled {m_:+.1f} +- {se_:.1f}, z {z_:.1f})"
        else:
            gate_note = f"composer challenger pooled {m_:+.1f} +- {se_:.1f} z {z_:+.1f} n {n_}"
        comp.net.load_state_dict(cand)
        torch.save(comp.net.state_dict(), W)             # published: the next batch flies these
for it in range(1, OUTER + 1):
    task_it = sampler(LEG)
    t_r = time.time()
    res, shards = par.run_with_records(TH, task_it.sample(E, make_gen(5_000_000 + it)), 5_100_000 + it, stochastic=True, record_frac=REC, difficulty=DIFF)
    t_r = time.time() - t_r
    fit = res.fitness; order = fit.argsort()
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
    _Rs, _rows = [], []
    for sh in shards:
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
    center_by_task(_Rs, _rows, E)
    _v1 = float(torch.cat([r.reshape(-1) for r in _Rs]).std())
    for sh, R in zip(shards, _Rs):                       # then the residual per-genome offset
        gid = sh["rows"] // E
        for g in gid.unique():
            m = gid == g; R[:, m] = R[:, m] - R[:, m].mean(1, keepdim=True)
        groups_r.append(sh["records"]); groups_R.append(R); n_dec += sum(int((r["alive"] & r["tok_keep"]).sum()) if r.get("tok_keep") is not None else int(r["alive"].sum()) for r in sh["records"])   # the samples the update can use
    _join(it)                                             # update k-1 done: gate, publish, then start update k
    if n_dec:
        holder = {}; th_ = threading.Thread(target=_update, args=(groups_r, groups_R, holder), daemon=True); th_.start()
        pending = (th_, holder)
    st, t_u = st_prev, t_u_prev
    _stride = max(1, int(round(1.0 / REC))); _pol = (torch.arange(E) % _stride) != 0        # the batch numbers are the POLICY rows' (the recorded rows explore)
    tr_r = float(res.success.reshape(P, E)[:, _pol].double().mean()); tr_c = 1 - float(res.alive.reshape(P, E)[:, _pol].double().mean()); subs = sum(n_sub) / len(n_sub) if n_sub else float("nan")
    de_r, de_cost = tr_r, float(res.cost.mean())
    note = ""; flown_diff, flown_leg = DIFF, LEG
    if de_r >= DIFF_UP_AT and DIFF < 1.0:
        DIFF = min(DIFF + DIFF_STEP, 1.0); note = f"  -> buildings {DIFF:.0%}"
    elif de_r >= LEG_UP_AT and DIFF >= 1.0 and LEG < LEG_MAX:
        LEG = min(LEG + LEG_STEP, LEG_MAX); note = f"  -> legs {LEG:.1f} m"
    save_state()
    log(f"  iter {it:>4}  buildings {flown_diff:4.0%} legs {flown_leg:4.1f}  batch {tr_r:.3f}/{tr_c:.3f} cost {de_cost:7.2f} subgoals/flight {subs:4.1f}  "
        f"elite {float(fit[order[0]]):8.3f} (spread {(float(fit.std()) if fit.numel() > 1 else 0.0):6.3f}, {guard_note}; {gate_note})  ppo n {st.get('n', 0)} kl {st['kl']:.4f} clip {st['clipfrac']:.2f} speak {st['speak']:.3f} H {st['entropy']:.2f} ev {st.get('ev', float('nan')):.2f}  "
        f"[ret sd {_v0:.0f}->{_v1:.0f} dec {n_dec} roll {t_r:.0f}s ppo(k-1) {t_u:.0f}s | {(time.time()-t0)/60:.1f}m]{note}")
    if it % JUDGE_EVERY == 0:
        _join(it)                                         # the judge flies the finished weights
        c = judge_all(f"judge {it}")
        if c[2] < best_cost:
            best_cost = c[2]; save_state(); save_best()
_join(); par.close()
log("COTRAIN V7 DONE")
