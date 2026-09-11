"""Stage A: teach the low level to FLY, alone, on an empty map.

The diagnosis this comes from: with the composer removed entirely the motion is
unchanged -- 1.55 m/s against a 5 m/s limit, acceleration changing ~20% every
step, and crashes arriving at walking pace with the beams already reporting the
wall.  So the erratic flying is the controller's, and co-training was never
giving it a clean signal: on a city map its fitness is dominated by whether it
crashed, which is chaotic near walls, so the part that says 'fly straight and
fast' is buried.

An empty map removes every obstacle, so the cost is almost purely 'how quickly
did you get there'.  Same cost structure as the real thing (distance to go, the
arrival bonus, the charge for dying) so the genome transfers, and no composer,
so the goal is the task's own waypoint.

Stage B puts the composer back on Singapore, warm-started from what this finds.
"""
import json, sys, time, torch
SP = sys.argv[1]
GENS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
# Take cotrain_v9's HEADER only.  The old marker was
# "torch.set_num_threads(1)\npar = ParallelRollout"; a comment block was later
# inserted between those two lines, `str.split` silently returned the WHOLE
# file, and exec ran all 1000 iterations of v9's training loop instead of this
# script.  A missing marker must fail, not fall back to running everything.
_MARK = "\npar = ParallelRollout"
_src = open(f"{SP}/cotrain_v9.py").read()
assert _MARK in _src, f"header marker {_MARK!r} missing from cotrain_v9.py: exec would run the whole training script"
src = _src.split(_MARK)[0]
_a = sys.argv; sys.argv = ["cotrain_v9.py", SP, "1000", "0.05", "100", "3.0", "40.0", "2e-4"]
try: exec(compile(src, "hdr", "exec"))
finally: sys.argv = _a
from dataclasses import replace
from lagrangian_es.config import ESCfg
from lagrangian_es.es import build, build_sensors, train_ga
from lagrangian_es.parallel import ParallelRollout
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen
torch.set_num_threads(2)
POP, E_, SIGMA, WORKERS = 16, 192, 0.03, 6
OUT_G = f"{SP}/stage_b_genome.json"
log_path = f"{SP}/stage_b.out"
def log(m):
    print(m, flush=True)
    with open(log_path, "a") as f: f.write(m + "\n")

def cfg_a(n, steps=900):
    # An empty map carries no waypoints, so the city tour cannot run on it;
    # `waypoint_pair` generates its own goals.
    #
    # `xy` MUST keep those goals inside the plant's own arena: `alive` fails at
    # |p| >= 20 m, and a corner goal is sqrt(2)*xy away.  Set to the city leg
    # length (20) it put goals 27 m out, so the vehicle flew at a target
    # outside the world and died on the boundary -- 100% of "crashes" on a map
    # with nothing in it.  9.0 keeps the far corner near 13 m, which is still
    # far enough to reach the 5 m/s limit and settle.
    c = cfg_for("singapore_cbd", LEG_MAX, steps, n, composer="", early=False)
    return replace(c, environment="empty", task="waypoint_pair", composer="", composer_kw=(),
                   task_kw=(("xy", 9.0), ("z_lo", 1.0), ("z_hi", 2.5),
                            ("tol", 0.25), ("gating", "arrival")),
                   es=ESCfg(pop=POP, sigma0=SIGMA, gens=GENS, elitism=2),
                   rollout=replace(c.rollout, n_eps=n, ep_steps=steps))

cfg = cfg_a(E_)
sysm, tr, task = build(cfg)
th0 = torch.tensor(json.load(open(G))["theta"], dtype=torch.float64)
log(f"\n===== stage B started {time.ctime()}: low level ALONE (NO composer) on an EMPTY map -- speed and control, "
    f"pop {POP} x {E_} episodes, sigma {SIGMA}, {GENS} generations, warm start from the city genome =====")

def smooth(theta, tag):
    """What we are actually trying to fix: speed, and how much the thrust chatters."""
    c = cfg_a(64, steps=900)
    s2, t2, k2 = build(c)
    r2 = Rollout(s2, t2, k2, c.rollout, build_sensors(c, s2))
    with torch.no_grad():
        trc = r2.trace(theta[None], k2.sample(64, make_gen(4242)), 4243, freeze_arrivals=True)
    v = s2.task_velocity(trc.states); live = trc.alive.double(); T = live.shape[0]
    a = ((v[1:] - v[:-1]) / c.rollout.dt)[:T]
    j = (a[1:] - a[:-1]) / c.rollout.dt
    m = lambda t, w: float((t * w).sum() / w.sum().clamp(min=1))
    res = r2.run(theta[None], k2.sample(64, make_gen(4242)), 4243)
    x = s2.task_position(trc.states); dead = ~trc.alive[-1]
    if bool(dead.any()):
        kd = trc.alive.double().sum(0).long().clamp(max=trc.alive.shape[0] - 1)
        zs = torch.stack([x[int(kd[b]), b, 2] for b in dead.nonzero().flatten().tolist()])
        log(f"    {'':16s} of the {int(dead.sum())} deaths: median z at death {float(zs.median()):+.2f} m, "
            f"{100*float((zs < 0.3).double().mean()):.0f}% on the floor")
    log(f"    {tag:16s} speed {m(v.norm(dim=-1)[:T], live):5.2f} m/s  |jerk| {m(j.norm(dim=-1), live[:j.shape[0]]):7.1f}  "
        f"reach {float(res.success.double().mean()):.3f}  crash {1-float(res.alive.double().mean()):.3f}  "
        f"cost {float(res.cost.mean()):7.2f}")

smooth(th0, "warm start")
par = ParallelRollout({"cfg": cfg}, workers=WORKERS, min_pop=WORKERS, shards=POP); par._pool_up()
t_start = time.time()
_seen = {"g": 0}
def cb(rec, theta):
    g = _seen["g"]; _seen["g"] += 1
    json.dump({"theta": theta.tolist()}, open(OUT_G, "w"))     # always keep the latest elite
    # AND a snapshot per generation.  The original kept one file and overwrote
    # it, so every intermediate elite was lost and "use the gen 35 version" was
    # unanswerable -- the only survivor was whatever `train_ga` returned at the
    # end.  Which generation transfers best to the city is an empirical
    # question, so keep them all and measure.
    json.dump({"theta": theta.tolist()}, open(f"{SP}/stage_b_snaps/g{g:03d}.json", "w"))
    if g % 5 == 0 or g == GENS - 1:
        # `success_rate`, NOT `success_frac`.  stage_a.py printed the latter
        # under the heading "reach", and they are unrelated quantities:
        # `success_frac` (es.py:155) is the fraction of OFFSPRING that beat the
        # parent -- the 1/5th-rule statistic that drives sigma.  So "reach
        # 0.000" at generations 35-45 meant the search had converged, not that
        # the vehicle never arrived, and the whole run gave no visible signal
        # about the thing it was being trained for.
        log(f"  gen {g:>3}  elite {rec.get('fitness_parent', float('nan')):8.3f}  "
            f"sigma {rec.get('sigma', float('nan')):.4f}  "
            f"reach {rec.get('success_rate', float('nan')):.3f}  "
            f"crash {rec.get('crash_rate', float('nan')):.3f}  "
            f"t_arr {rec.get('finish_frac', float('nan')):.3f}  "
            f"err {rec.get('final_err', float('nan')):5.2f}  "
            f"| mut {rec.get('success_frac', float('nan')):.2f}  [{(time.time()-t_start)/60:.1f}m]")
    if g and g % 20 == 0:
        smooth(theta, f"gen {g}")        # speed and jerk: the thing this stage exists to fix
try:
    out = train_ga(cfg, sysm, tr, task, callback=cb, verbose=False, evaluator=par, theta0=th0)
finally:
    par.close()
best = out.theta
json.dump({"theta": best.tolist()}, open(OUT_G, "w"))
log("\n  after stage A:")
smooth(best, "trained")
log(f"  genome saved to {OUT_G}")
log("STAGE A DONE")
