#!/usr/bin/env python3
"""Train one configuration and report it against the section 6 criteria."""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lagrangian_es import viz                                        # noqa: E402
from lagrangian_es.es import build, train                            # noqa: E402
from lagrangian_es.evaluate import accepts, evaluate, report         # noqa: E402
from lagrangian_es.rollout import Rollout                            # noqa: E402
from lagrangian_es.util import load_config, make_gen, replace_cfg    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quadrotor_default.yaml")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--gens", type=int)
    ap.add_argument("--pop", type=int)
    ap.add_argument("--strategy", choices=["es", "ga"])
    ap.add_argument("--no-whiten", action="store_true")
    ap.add_argument("--trainable")
    ap.add_argument("--eval-tasks", type=int, default=192)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    over = {}
    if args.seed is not None:      over["seed"] = args.seed
    if args.gens is not None:      over["es.gens"] = args.gens
    if args.pop is not None:       over["es.pop"] = args.pop
    if args.strategy:              over["es.strategy"] = args.strategy
    if args.trainable:             over["trainable"] = args.trainable
    if args.no_whiten:             over["es.whiten"] = False
    cfg = replace_cfg(cfg, **over)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    system, trainable, task = build(cfg)
    print(f"{cfg.system} + {cfg.trainable} ({trainable.dim} genome slots) | "
          f"{cfg.es.strategy} | whiten={cfg.es.whiten} | seed={cfg.seed}")

    res = train(cfg, system, trainable, task, verbose=not args.quiet)

    ev0 = evaluate(system, trainable, task, trainable.init(), cfg.rollout, n_tasks=args.eval_tasks)
    ev = evaluate(system, trainable, task, res.theta, cfg.rollout, n_tasks=args.eval_tasks)
    print(f"\n{report(ev0, 'generation-0 prior')}")
    print(f"\n{report(ev, 'trained')}")
    print("\n--- section 6 acceptance ---")
    for k, v in accepts(ev).items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\ntrained in {res.wall_time:.1f}s")

    torch.save({"theta": res.theta, "cfg": dataclasses.asdict(cfg)}, out / "genome.pt")
    (out / "history.json").write_text(json.dumps(
        {"history": res.history, "eval": ev, "eval_prior": ev0,
         "cfg": dataclasses.asdict(cfg), "wall_time": res.wall_time}, indent=2))

    viz.learning_curves({"run": res.history}, out / "learning_curves.png")
    viz.metric_panel(res.history, out / "metric_panel.png")
    roll = Rollout(system, trainable, task, cfg.rollout)
    goals = task.sample(6, make_gen(4242))
    viz.trajectories(system,
                     [roll.trace(trainable.init()[None], goals, 4243),
                      roll.trace(res.theta[None], goals, 4243)],
                     goals, out / "trajectories.png", labels=["prior", "trained"])
    print(f"artifacts -> {out}/")


if __name__ == "__main__":
    main()
