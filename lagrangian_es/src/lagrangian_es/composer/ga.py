"""A composer small enough to EVOLVE, with time as the fitness.

Why a different shape.  Every gradient objective tried on the transformer
composer failed, each in its own way and each verified by measurement:

    cross-entropy on own successes   destructive: 0.977 -> 0.707 at short legs
    signed soft time                 diverged, loss -7.5e11 (centred weights)
    reach-only error                 degenerate: e_reach -> 0.009, arrive -> 0.026
    reach + aim                      aim outweighed reach 3:1, it over-reached
    purely time (learned model)      policy gamed the model, T_hat went negative

A GA needs none of what those needed: no differentiability, no credit
assignment across ~90 decisions, no log-density, no baseline.  It needs only a
fitness that can be MEASURED, and time can be.

But the transformer is 984k parameters and ES variance scales with dimension --
a few dozen genomes a generation would be a random walk there.  So the policy
here is deliberately tiny, a linear map from a compact view to the token it
emits.  That also buys the thing that makes evolution affordable: every genome
in the population flies in ONE rollout, because a small policy can carry a
DIFFERENT weight matrix per row as a batched matmul.  The population costs the
same as a single policy.

The view is the navigator's, and nothing more:

    goal in the vehicle frame (3)   where to go
    beam minima in 8 sectors  (8)   what is in the way -- the sensors, directly
    speed in the vehicle frame(3)   how fast it is already going
    bias                      (1)

and the output is what the token needs: r, theta, phi and whether to speak.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from .actions_cont import PHI_MAX, ContVocab
from .base import Composer
from .tokens import to_ego, to_world, yaw_of

N_SECT = 8                       # beam sectors around the vehicle
N_FEAT = 3 + N_SECT + 3 + 1      # goal, sectors, velocity, bias
N_OUT = 4                        # r, theta, phi, speak


def genome_dim() -> int:
    return N_FEAT * N_OUT


def features(ctx: Dict, reach: float, vmax: float = 5.0) -> Tensor:
    """The compact view, [B, N_FEAT].  Ego-framed, scaled to order one."""
    x, v, goal, s = ctx["x"], ctx["v"], ctx["goal"], ctx["state"]
    B = x.shape[0]; dt = x.dtype; dev = x.device
    psi = yaw_of(s["R"]) if "R" in s else torch.zeros(B, dtype=dt, device=dev)
    g = to_ego(goal - x, psi) / reach
    ve = to_ego(v, psi) / max(vmax, 1e-6)
    obs = ctx.get("obs") or {}
    sect = torch.ones(B, N_SECT, dtype=dt, device=dev)
    for key, o in obs.items():
        if not torch.is_tensor(o) or o.ndim != 2 or o.shape[1] < N_SECT:
            continue
        # the fan, folded into N_SECT equal wedges: the MINIMUM in each, which
        # is what "can I go that way" depends on
        n = (o.shape[1] // N_SECT) * N_SECT
        r = o[:, :n].reshape(B, N_SECT, -1).min(-1).values
        sect = torch.minimum(sect, (r / r.new_tensor(o.max().clamp_min(1e-6))).clamp(0, 1).to(dt))
        break
    return torch.cat([g, sect, ve, torch.ones(B, 1, dtype=dt, device=dev)], -1)


class GAComposer(Composer):
    """One linear policy per genome, all of them flying in the same batch."""

    kind = "ga"

    def __init__(self, system, trainable, reach: float = 10.0, every: int = 20,
                 n_terms: int = 1, **kw):
        self.system = system
        self.reach = float(reach)
        self.every = int(every)
        self.measure_every = int(kw.get("measure_every", self.every))
        self.vocab = ContVocab(n_terms)
        self.n_terms = int(n_terms)
        self.theta: Optional[Tensor] = None     # [P, N_FEAT, N_OUT]
        self.n_eps = 1
        self.records = []
        self.record_rows = None
        self.stochastic = False

    # --- the population -----------------------------------------------------
    def set_population(self, theta: Tensor, n_eps: int) -> None:
        """`theta` [P, genome_dim]; rows of the batch are member * n_eps + ep."""
        self.theta = theta.reshape(theta.shape[0], N_FEAT, N_OUT)
        self.n_eps = int(n_eps)

    def _out(self, ctx: Dict, rows: Tensor) -> Tensor:
        f = features(ctx, self.reach).to(self.theta.dtype)          # [n, N_FEAT]
        who = torch.div(rows, self.n_eps, rounding_mode="floor").clamp(0, self.theta.shape[0] - 1)
        W = self.theta[who]                                          # [n, N_FEAT, N_OUT]
        return torch.einsum("nf,nfo->no", f, W)

    def reset(self, B: int) -> None:
        self.records = []

    def pair(self, *a, **k) -> None:
        pass

    def decide(self, ctx: Dict, rows: Optional[Tensor] = None):
        """`[WAYPOINT(r, theta, phi)]` or silence, per row."""
        x, goal = ctx["x"], ctx["goal"]
        B = x.shape[0]
        idx = rows if rows is not None else torch.arange(B, device=x.device)
        out = self._out(ctx, idx)
        speak = out[:, 3] > 0.0
        a = torch.tanh(out[:, :3])
        s = ctx["state"]
        psi = yaw_of(s["R"]) if "R" in s else torch.zeros(B, dtype=x.dtype, device=x.device)
        g_ego = to_ego(goal - x, psi) / self.reach
        L0 = g_ego.norm(dim=-1)
        radius = L0.clamp(max=1.0)
        r = (a[:, 0] + 1.0) * 0.5 * radius
        th = math.pi * a[:, 1]
        ph = PHI_MAX * a[:, 2]
        cph = torch.cos(ph)
        sub_ego = torch.stack([r * cph * torch.cos(th), r * cph * torch.sin(th),
                               r * torch.sin(ph)], -1)
        sub = x + to_world(sub_ego * self.reach, psi)
        return sub, speak
