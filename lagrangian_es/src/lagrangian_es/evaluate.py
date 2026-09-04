"""Held-out evaluation.

Training goals are resampled every generation and there are only a handful of
them, so a training-set number is far too noisy to accept a run on.  Everything
in section 6 is measured here instead, on a fixed held-out task set drawn from a
seed that training never touches.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .rollout import Rollout
from .util import make_gen

HELDOUT_SEED = 987_654_321


def evaluate(system, trainable, task, theta: Tensor, cfg, n_tasks: int = 128,
             seed: int = HELDOUT_SEED, roll: Optional[Rollout] = None,
             tol: float = 0.25) -> dict:
    """Run one genome on `n_tasks` unseen episodes.

    `tol` is the success radius used for the "within 25 cm" criterion; it is a
    reporting threshold only and never enters the fitness.
    """
    roll = roll or Rollout(system, trainable, task, cfg)
    goals = task.sample(n_tasks, make_gen(seed))
    res = roll.run(theta[None], goals, seed=seed + 1)

    err = res.final_err
    within = (err < tol) & res.alive
    out = {
        "n_tasks": n_tasks,
        "fitness": float(res.fitness.mean()),
        "crash_rate": res.crash_rate,
        "success_rate": float(within.to(torch.float64).mean()),
        "final_err_mean": float(err.mean()),
        "final_err_median": float(err.median()),
        "final_err_p90": float(err.quantile(0.9)),
        "saturation": float(res.saturation.mean()),
    }
    for i in range(res.leg_err.shape[1]):
        out[f"leg{chr(ord('A') + i)}_err"] = float(res.leg_err[:, i].mean())
    return out


def accepts(ev: dict, leg_b_max: float = 0.15, success_min: float = 0.80,
            crash_max: float = 0.0) -> dict:
    """Check a held-out result against the section 6 acceptance criteria."""
    checks = {
        "legB_err < %.2f" % leg_b_max: ev.get("legB_err", float("inf")) < leg_b_max,
        "success_rate > %.2f" % success_min: ev["success_rate"] > success_min,
        "crash_rate <= %.2f" % crash_max: ev["crash_rate"] <= crash_max,
    }
    checks["ALL"] = all(checks.values())
    return checks


def report(ev: dict, title: str = "held-out") -> str:
    lines = [f"--- {title} ({ev['n_tasks']} tasks) ---"]
    for k, v in ev.items():
        if k != "n_tasks":
            lines.append(f"  {k:18s} {v:.4f}")
    return "\n".join(lines)
