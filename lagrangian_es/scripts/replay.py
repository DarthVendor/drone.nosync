#!/usr/bin/env python3
"""Export flight trajectories for the visualizer (and for matplotlib replay).

Trains a genome, then re-flies it -- and the untrained prior -- on the SAME
held-out episodes, so the two are directly comparable frame by frame.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lagrangian_es.config import Config, ESCfg, RolloutCfg          # noqa: E402
from lagrangian_es.es import build, train                            # noqa: E402
from lagrangian_es.evaluate import evaluate                          # noqa: E402
from lagrangian_es.rollout import Rollout                            # noqa: E402
from lagrangian_es.util import make_gen                              # noqa: E402


def capture(roll, task, theta, goals, seed, decimals=3):
    """Fly one genome on a batch of episodes and pack the trajectory."""
    tr = roll.trace(theta[None], goals, seed)
    T, B = tr.goals.shape[0], tr.goals.shape[1]
    p = tr.states["p"][: T + 1]                       # [T+1, B, 3]
    R = tr.states["R"][: T + 1]                       # [T+1, B, 3, 3]
    alive = tr.alive                                  # [T, B]
    r = lambda x: [round(float(v), decimals) for v in x.reshape(-1)]
    eps = []
    for b in range(B):
        eps.append({
            "p": r(p[:, b]),
            "R": r(R[:, b]),
            "alive": [int(v) for v in alive[:, b]],
            "goals": r(goals[b]),
        })
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="trajectories.json")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--n-eps", type=int, default=4)
    ap.add_argument("--strategy", default="ga")
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(seed=args.seed, rollout=RolloutCfg(n_eps=args.n_eps),
                 es=ESCfg(pop=args.pop, gens=args.gens, strategy=args.strategy,
                          sigma0=args.sigma0, elite_frac=0.5, metric_every=5))
    system, trainable, task = build(cfg)
    print(f"training {args.strategy} for {args.gens} generations ...", flush=True)
    res = train(cfg, system, trainable, task, verbose=False)
    theta0 = trainable.init()

    ev_t = evaluate(system, trainable, task, res.theta, cfg.rollout, n_tasks=192)
    ev_0 = evaluate(system, trainable, task, theta0, cfg.rollout, n_tasks=192)
    print(f"  prior  : crash {ev_0['crash_rate']:.2f}  legB {ev_0['legB_err']:.3f}  "
          f"success {ev_0['success_rate']:.2f}")
    print(f"  trained: crash {ev_t['crash_rate']:.2f}  legB {ev_t['legB_err']:.3f}  "
          f"success {ev_t['success_rate']:.2f}")

    roll = Rollout(system, trainable, task, cfg.rollout)
    goals = task.sample(args.episodes, make_gen(4242))
    payload = {
        "dt": cfg.rollout.dt,
        "ep_steps": cfg.rollout.ep_steps,
        "switch_step": cfg.rollout.ep_steps // 2,
        "arm_len": 0.16,
        "runs": {
            "trained": capture(roll, task, res.theta, goals, 4243),
            "prior": capture(roll, task, theta0, goals, 4243),
        },
        "eval": {"trained": ev_t, "prior": ev_0},
        "curve": {
            "fitness_elite": [h["fitness_elite"] for h in res.history],
            "crash_rate": [h["crash_rate"] for h in res.history],
            "legB_err": [h["legB_err"] for h in res.history],
            "sigma": [h["sigma"] for h in res.history],
        },
        "config": {"strategy": args.strategy, "gens": args.gens, "pop": args.pop,
                   "n_eps": args.n_eps, "sigma0": args.sigma0, "seed": args.seed},
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{args.episodes} episodes x 2 runs)")


if __name__ == "__main__":
    main()
