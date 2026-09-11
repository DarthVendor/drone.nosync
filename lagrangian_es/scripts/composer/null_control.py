"""Is the composer HELPING?  The control nobody ran.

The low level alone reaches 1.000 on an empty map; the composer + low level
reached 0.865 on its own empty-map setting.  Those were different configs, so
this flies all three arms on the SAME config, the SAME tasks and the SAME
initial states -- the composer's own training setting at difficulty 0:

    no composer        the subgoal IS the goal; the frozen low level flies at it
    composer sampled   exactly as it is trained and judged
    composer argmax    the same policy with the sampling noise removed

Any difference is the composer and nothing else.
"""
import json, sys, math, time

import torch

SP = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
_a = sys.argv
sys.argv = ["cotrain_v10.py", SP, "1000", "0.05", "100", "3.0", "40.0", "2e-4"]
_MARK = "\npar = ParallelRollout"
_src = open(f"{SP}/cotrain_v10.py").read()
assert _MARK in _src, "header marker missing: exec would run the whole training script"
try:
    exec(compile(_src.split(_MARK)[0], "hdr", "exec"))
finally:
    sys.argv = _a
from lagrangian_es.es import build, build_composer, build_sensors
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen

torch.set_num_threads(3)
DIFF = 0.0                       # the curriculum level the composer has lived at
SEED_TASK, SEED_ROLL = 777, 778
th = torch.tensor(json.load(open(G))["theta"], dtype=torch.float64)[None]

c0 = cfg_for("singapore_cbd", LEG_MAX, T_TRAIN, N, composer="", early=False)
s0, t0_, k0 = build(c0)
goals = k0.sample(N, make_gen(SEED_TASK))          # ONE task list, used by every arm

print(f"{N} episodes, singapore_cbd at {DIFF:.0%} buildings, {T_TRAIN} steps, identical tasks\n")
print(f"{'arm':<22} {'reach':>7} {'+-':>5} {'crash':>7} {'stuck':>7} {'t_arr':>7} {'cost':>9}")
rows = {}
for name, kind, stoch in (("no composer", "", None),
                          ("composer sampled", "policy_cont", True),
                          ("composer argmax", "policy_cont", False)):
    c = cfg_for("singapore_cbd", LEG_MAX, T_TRAIN, N, composer=kind, early=False)
    s2, t2, k2 = build(c)
    s2.difficulty = DIFF
    comp = build_composer(c, s2, t2) if kind else None
    if comp is not None:
        comp.stochastic = bool(stoch)
        comp.records = []
        comp.record_rows = torch.arange(N) if stoch else None
        comp.reset(N); comp.pair(N, 11)
    rig = Rollout(s2, t2, k2, c.rollout, build_sensors(c, s2), composer=comp)
    torch.manual_seed(0)
    with torch.no_grad():
        r = rig.run(th, goals, SEED_ROLL)
    suc = r.success.double(); alv = r.alive.double(); ff = r.finish_frac.double()
    reach = float(suc.mean()); se = (reach * (1 - reach) / N) ** 0.5
    stuck = float(((alv > 0) & (suc == 0)).double().mean())
    t_arr = float(ff[ff < 1.0].mean()) if bool((ff < 1.0).any()) else float("nan")
    print(f"{name:<22} {reach:7.3f} {se:5.3f} {1-float(alv.mean()):7.3f} {stuck:7.3f} {t_arr:7.3f} {float(r.cost.mean()):9.2f}")
    rows[name] = (r, comp)

a = rows["no composer"][0].success.double()
b = rows["composer sampled"][0].success.double()
d = b - a
se = float(d.std() / N ** 0.5)
print(f"\npaired, same tasks: composer - none = {float(d.mean()):+.3f} +- {se:.3f}"
      f"   (won {int((d>0).sum())}, lost {int((d<0).sum())}, same {int((d==0).sum())})")

# --- if it is losing, what is it saying? ------------------------------------
comp = rows["composer sampled"][1]
recs = getattr(comp, "records", None)
if recs:
    V = comp.net.vocab
    act = torch.cat([x["act"] for x in recs])
    u = torch.cat([x["u"] for x in recs])
    print(f"\n{act.numel()} tokens emitted:")
    for t in range(V.V):
        print(f"   {V.name(t):<20} {100*float((act==t).double().mean()):6.2f}%")
    w = act == V.WAYPOINT
    if bool(w.any()):
        r_lever = torch.tanh(u[w, 0])
        bearing = math.pi * torch.tanh(u[w, 1]) * 180 / math.pi
        print(f"\nWAYPOINT arguments ({int(w.sum())} of them):")
        print(f"   r     (-1 = stop on the spot, +1 = full distance to the goal): "
              f"mean {float(r_lever.mean()):+.3f}  sd {float(r_lever.std()):.3f}")
        print(f"   theta (0 = straight at the nose):  mean |{float(bearing.abs().mean()):.1f}| deg"
              f"  sd {float(bearing.std()):.1f}  |  >45 deg off: {100*float((bearing.abs()>45).double().mean()):.1f}%")
        frac = (r_lever + 1.0) * 0.5
        print(f"   so the subgoal sits at {float(frac.mean()):.2f} of the way to the goal on average")
print("\nNULL CONTROL DONE")
