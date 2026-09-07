"""Mining a controller's own failures, and keeping only the useful ones.

Once crashes are rare the crash term stops ranking anything: a handful of
failures in a batch ties across most of the population, and selection sees a
gate rather than a gradient.  The way through is to make failures common again
on purpose, by restarting episodes in the situations that produced them --
`ReplayStart` in `tasks.py` does the restarting, and this module chooses what to
put in its pool.

The choice is not free.  A replayed situation only carries selection pressure if
its outcome still depends on the genome.  Measured on `sparse`, crash rate when
replayed from each lead:

    0.20s 90.9%   0.30s 92.7%   0.50s 84.4%    hopeless
    0.80s 56.7%   1.20s 30.6%   1.80s 26.7%    contested

Training on fixed leads of 0.3/0.5/0.8 s moved held-out crash rate by -1.0%,
which is noise.  Training on the contested band moved it -26.3%, better on six
of six held-out seeds.  The hopeless leads had been adding a flat `dead_cost` to
every member and diluting the batch.

The band is a property of the controller and the world together, not a constant:
`pillars` measures 0.4-1.2 s where `sparse` measures 0.8-1.8 s, and the same
controller's band moves later as it gets better at close-range recovery.  So
probe it, per stage, rather than carrying a number over.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor

from .rollout import Rollout
from .tasks import Task, make_task
from .util import make_gen

__all__ = ["harvest_pre_crash", "recoverability", "contested_leads"]

Pool = Tuple[Dict[str, Tensor], Tensor]


def harvest_pre_crash(roll: Rollout, task: Task, theta: Tensor, seed: int,
                      leads: Sequence[int], blocks: int = 8,
                      block: int = 256) -> Tuple[Dict[int, Pool], int, int]:
    """States `leads` steps before each of this controller's crashes.

    Returns one pool per lead, the episodes examined, and the crashes found.
    The whole state is kept, obstacle field included -- a recovery in a
    different scene is a different problem -- along with the goal that was
    ACTIVE at that instant.  Under arrival gating a replay restarts on leg 0,
    so replaying the whole waypoint list would aim the vehicle at somewhere it
    had already been rather than at where it was going when it died.
    """
    pools: Dict[int, Tuple[Dict[str, List[Tensor]], List[Tensor]]] = {
        int(l): ({}, []) for l in leads}
    seen = crashes = 0
    for b in range(blocks):
        goals = task.sample(block, make_gen(seed + b))
        tr = roll.trace(theta[None], goals, seed + 700 + b)
        alive = tr.alive.bool()
        died = ~alive[-1]
        seen += block
        if not bool(died.any()):
            continue
        crashes += int(died.sum())
        first = (~alive).to(torch.int8).argmax(0)     # first step not alive
        for i in died.nonzero().flatten().tolist():
            for lead in pools:
                k = int(first[i]) - lead
                if k < 1:
                    continue
                states, goal_list = pools[lead]
                for key, val in tr.states.items():
                    states.setdefault(key, []).append(val[k, i])
                goal_list.append(tr.goals[k, i].expand(task.n_legs, -1).clone())
    out: Dict[int, Pool] = {}
    for lead, (states, goal_list) in pools.items():
        if goal_list:
            out[lead] = ({k: torch.stack(v) for k, v in states.items()},
                         torch.stack(goal_list))
    return out, seen, crashes


def recoverability(system, trainable, sensors, rollout_cfg, theta: Tensor,
                   pool: Pool, base: str = "waypoint_pair",
                   gating: str = "arrival", n: int = 384,
                   seed: int = 9_001, **task_kw) -> float:
    """Crash rate when this controller is restarted from `pool`."""
    states, goals = pool
    task = make_task("replay_start", system, base=base, mix=1.0,
                     states={k: v.tolist() for k, v in states.items()},
                     goals=goals.reshape(-1).tolist(), gating=gating, **task_kw)
    roll = Rollout(system, trainable, task, rollout_cfg, sensors)
    m = min(int(goals.shape[0]), n)
    res = roll.run(theta[None], task.sample(m, make_gen(seed)), seed + 1)
    return 1.0 - float(res.alive.float().mean())


def contested_leads(pools: Dict[int, Pool], rates: Dict[int, float],
                    band: Tuple[float, float] = (0.15, 0.85),
                    keep: int = 4) -> List[int]:
    """The leads whose outcome still depends on the genome.

    Outside the band a replay start ranks nothing: below it every member
    survives, above it every member dies, and either way the episode
    contributes the same number to every fitness.  When nothing is contested --
    a controller that has stopped crashing, or one that cannot recover at any
    range -- fall back to the middle lead so the caller always has a pool.
    """
    lo, hi = band
    good = sorted(l for l in pools if lo <= rates.get(l, 0.0) <= hi)
    if not good:
        order = sorted(pools)
        good = [order[len(order) // 2]] if order else []
    return good[-keep:]
