"""Which generation of stage A actually flies best -- measured, on the map it
was trained on.

The stage-A log's `reach` column is not usable: it printed 0.000 at generations
35-45 while the elite fitness was improving, so the run gave no honest signal
about which elite to freeze, and the one genome it kept was whatever `train_ga`
returned at the end.  Every generation is snapshotted now, and they are all
flown HERE, together, as one population: a single rollout puts every genome on
the SAME tasks from the same initial states (common random numbers), so the
differences between them are the genomes and nothing else.
"""
import json, sys, glob, os, time

import torch

SP = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 384
_a = sys.argv
sys.argv = ["cotrain_v9.py", SP, "1000", "0.05", "100", "3.0", "40.0", "2e-4"]
# Take cotrain_v9's HEADER only.  The old marker was
# "torch.set_num_threads(1)\npar = ParallelRollout"; a comment block was later
# inserted between those two lines, `str.split` silently returned the WHOLE
# file, and exec ran all 1000 iterations of v9's training loop instead of this
# script.  A missing marker must fail, not fall back to running everything.
_MARK = "\npar = ParallelRollout"
_src = open(f"{SP}/cotrain_v9.py").read()
assert _MARK in _src, f"header marker {_MARK!r} missing from cotrain_v9.py: exec would run the whole training script"
src = _src.split(_MARK)[0]
try:
    exec(compile(src, "hdr", "exec"))
finally:
    sys.argv = _a
from dataclasses import replace
from lagrangian_es.config import ESCfg
from lagrangian_es.es import build
from lagrangian_es.parallel import ParallelRollout
from lagrangian_es.util import make_gen

STEPS = 900
def cfg_a(n, steps=STEPS):
    # identical to stage_a.py's own config, so this measures the thing that was trained
    c = cfg_for("singapore_cbd", LEG_MAX, steps, n, composer="", early=False)
    return replace(c, environment="empty", task="waypoint_pair", composer="", composer_kw=(),
                   task_kw=(("xy", 9.0), ("z_lo", 1.0), ("z_hi", 2.5),
                            ("tol", 0.25), ("gating", "arrival")),
                   es=ESCfg(pop=16, sigma0=0.03, gens=1, elitism=2),
                   rollout=replace(c.rollout, n_eps=n, ep_steps=steps))

cfg = cfg_a(N)
sysm, tr, task = build(cfg)
snaps = sorted(glob.glob(f"{SP}/stage_b_snaps/g*.json"))
rows = [(os.path.basename(f)[1:4], torch.tensor(json.load(open(f))["theta"], dtype=torch.float64)) for f in snaps]
# the genome currently frozen, as the reference every arm is measured against
ref = torch.tensor(json.load(open(f"{SP}/stage_a_genome_frozen.json"))["theta"], dtype=torch.float64)
rows.append(("FROZEN", ref))
TH = torch.stack([t for _, t in rows])
print(f"{len(rows)} genomes x {N} episodes on the EMPTY map ({STEPS} steps), one batch, common random numbers")
torch.set_num_threads(1)
par = ParallelRollout({"cfg": cfg}, workers=6, min_pop=6, shards=len(rows))
par._pool_up()
t0 = time.time()
res = par.run(TH, task.sample(N, make_gen(4242)), 4243)
par.close()
suc = res.success.reshape(len(rows), N).double().mean(1)
alv = res.alive.reshape(len(rows), N).double().mean(1)
cst = res.cost.reshape(len(rows), N).double().mean(1)
se = (suc * (1 - suc) / N).clamp_min(1e-12) ** 0.5
print(f"flown in {time.time()-t0:.0f}s\n")
print(f"{'gen':>7} {'reach':>7} {'+-':>5} {'crash':>7} {'cost':>9}")
for i, (g, _) in enumerate(rows):
    print(f"{g:>7} {float(suc[i]):7.3f} {float(se[i]):5.3f} {1-float(alv[i]):7.3f} {float(cst[i]):9.2f}")
# Reach is 1.000 for every generation here, so it cannot rank anything.  Cost
# is what stage B was actually optimising (speed and control), so rank on that
# among the genomes that still arrive perfectly.
ok = suc >= suc.max() - 1e-9
pen = cst.clone(); pen[~ok] = float("inf")
best = int(pen.argmin())
print(f"\nbest: gen {rows[best][0]}  cost {float(cst[best]):.2f} reach {float(suc[best]):.3f}"
      f"   vs FROZEN cost {float(cst[-1]):.2f} reach {float(suc[-1]):.3f}"
      f"   ({float(cst[best]-cst[-1]):+.2f} cost)")
json.dump({"theta": rows[best][1].tolist()}, open(f"{SP}/stage_b_pick_genome.json", "w"))
print(f"saved -> {SP}/stage_b_pick_genome.json")
print("STAGE A PICK DONE")
