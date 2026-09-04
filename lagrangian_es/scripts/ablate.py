#!/usr/bin/env python3
"""The 2x2 the paper rests on.

    (structured vs unstructured genome) x (whitened vs isotropic mutation)

matched on seed, population and evaluation budget.  Because the genome structure
and the mutation operator are independent design choices, each is removed on its
own; the cross terms are what separate "the potential does the work" from "the
metric does the work".

Single-seed results are not evidence.  Every cell runs on the same seed list and
the headline number is the PAIRED difference across seeds -- whitened minus
isotropic, within genome type, seed by seed -- because the seed effect is large
relative to the effect being measured.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lagrangian_es import viz                                   # noqa: E402
from lagrangian_es.config import Config, ESCfg, RolloutCfg      # noqa: E402
from lagrangian_es.es import build, train                       # noqa: E402
from lagrangian_es.evaluate import evaluate                     # noqa: E402

METRICS = ("legB_err", "success_rate", "crash_rate", "fitness", "final_err_median")


def cell_name(trainable, whiten):
    return f"{'structured' if trainable == 'energy_shaping' else 'unstructured'} x " \
           f"{'whitened' if whiten else 'isotropic'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--n-eps", type=int, default=8)
    ap.add_argument("--strategy", default="es", choices=["es", "ga"])
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--eval-tasks", type=int, default=192)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = [(t, w) for t in ("energy_shaping", "mlp") for w in (True, False)]
    cells, raw = {}, []
    t_start = time.time()

    print(f"2x2 ablation | {len(args.seeds)} seeds | {args.gens} gens | pop {args.pop} "
          f"| n_eps {args.n_eps} | strategy {args.strategy}")
    print(f"budget per cell: {args.gens * args.pop * args.n_eps} episode evaluations\n", flush=True)

    for trainable, whiten in grid:
        name = cell_name(trainable, whiten)
        cells[name] = {m: [] for m in METRICS}
        for seed in args.seeds:
            cfg = Config(system="quadrotor", trainable=trainable, task="waypoint_pair",
                         seed=seed, rollout=RolloutCfg(n_eps=args.n_eps),
                         es=ESCfg(pop=args.pop, gens=args.gens, sigma0=args.sigma0,
                                  elite_frac=0.5, strategy=args.strategy, whiten=whiten,
                                  metric_every=5, metric_states=96, ridge=1e-3))
            system, tr, task = build(cfg)
            t0 = time.time()
            res = train(cfg, system, tr, task)
            ev = evaluate(system, tr, task, res.theta, cfg.rollout, n_tasks=args.eval_tasks)
            for m in METRICS:
                cells[name][m].append(ev[m])
            raw.append({"cell": name, "trainable": trainable, "whiten": whiten,
                        "seed": seed, "eval": ev, "wall_time": time.time() - t0,
                        "final": res.final})
            print(f"  {name:28s} seed {seed}  legB {ev['legB_err']:.3f}  "
                  f"succ {ev['success_rate']:.2f}  crash {ev['crash_rate']:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- paired differences: whitened - isotropic, within genome type ----
    diffs = {}
    for trainable in ("energy_shaping", "mlp"):
        w = cells[cell_name(trainable, True)]
        i = cells[cell_name(trainable, False)]
        label = "structured" if trainable == "energy_shaping" else "unstructured"
        diffs[label] = [a - b for a, b in zip(w["legB_err"], i["legB_err"])]

    print("\n=== cell means (held-out) ===")
    print(f"{'cell':30s} {'legB':>8} {'success':>8} {'crash':>7}")
    for name, d in cells.items():
        print(f"{name:30s} {statistics.mean(d['legB_err']):>8.3f} "
              f"{statistics.mean(d['success_rate']):>8.2f} "
              f"{statistics.mean(d['crash_rate']):>7.2f}")

    print("\n=== paired difference in legB_err: whitened - isotropic ===")
    print("    (negative = whitening helps; per seed, so the seed effect cancels)")
    for label, d in diffs.items():
        mean = statistics.mean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        se = sd / max(len(d), 1) ** 0.5
        wins = sum(x < 0 for x in d)
        print(f"  {label:14s} mean {mean:+.4f}  sd {sd:.4f}  se {se:.4f}  "
              f"helped on {wins}/{len(d)} seeds")
        print(f"  {'':14s} per seed: {[round(x, 4) for x in d]}")

    (out / "results.json").write_text(json.dumps(
        {"cells": cells, "raw": raw, "paired_diffs": diffs,
         "config": vars(args)}, indent=2))
    viz.ablation_grid(cells, out / "ablation_grid.png")
    viz.paired_differences(diffs, out / "paired_differences.png")
    print(f"\ntotal {time.time() - t_start:.0f}s  ->  {out}/")


if __name__ == "__main__":
    main()
