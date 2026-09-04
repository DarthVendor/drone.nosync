"""Episode evaluation -- system- and trainable-agnostic.

Nothing in this file names a plant or reads a raw state key.  It talks to the
system through the accessors on `LagrangianSystem` and to the controller through
`Trainable.forward`; `tests/test_lint_seam.py` enforces that mechanically.

Common random numbers are load-bearing.  Every member of a generation faces
identical goals AND identical reset noise: the population is laid out as
`index = member * n_eps + episode`, built by repeating one shared batch of E
initial states P times and interleaving each genome E times.  Without this, ES
variance swamps the effect the ablation is trying to measure.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
from torch.func import vmap

from .config import RolloutCfg
from .systems.base import LagrangianSystem, State
from .tasks import Task
from .trainables.base import Trainable
from .util import make_gen, tree_repeat, tree_stack, tree_where


@dataclass
class RolloutResult:
    fitness: Tensor             # [P]  mean episode cost per genome -- what ES ranks
    cost: Tensor                # [B]  raw per-episode cost
    alive: Tensor               # [B]  survived the whole episode
    leg_err: Tensor             # [B, n_legs]  task-space error at the end of each leg
    final_err: Tensor           # [B]
    success: Tensor             # [B]  within task tolerance at the end of the last leg
    saturation: Tensor          # [B]  mean fraction of channels at a bound
    effort: Tensor              # [B]  mean counterforce in the M^-1 metric
    shaping: Tensor             # [B]  mean plant-specific regularizer
    n_eps: int

    def per_genome(self, x: Tensor) -> Tensor:
        """Aggregate an episode-level [B] quantity to a per-genome [P] mean.

        Episode-level constraints are budgets on a genome's whole behaviour, not
        on a single episode, so they are always read through this.
        """
        return x.view(-1, self.n_eps).mean(dim=1)

    def genome_slice(self, sl) -> "RolloutResult":
        """Restrict to a contiguous range of genomes.

        Fitness is per genome [P] while the episode-level fields are [B = P*E], so
        a genome slice has to be widened by the episode count.  Used to report
        population statistics over offspring only, excluding the parent that is
        appended to the batch purely to get a free improvement signal.
        """
        E = self.n_eps
        start, stop, _ = sl.indices(self.fitness.shape[0])
        ep = slice(start * E, stop * E)
        return RolloutResult(
            fitness=self.fitness[sl], cost=self.cost[ep], alive=self.alive[ep],
            leg_err=self.leg_err[ep], final_err=self.final_err[ep],
            success=self.success[ep], saturation=self.saturation[ep],
            effort=self.effort[ep], shaping=self.shaping[ep], n_eps=E,
        )

    @property
    def crash_rate(self) -> float:
        return float((~self.alive).to(torch.float64).mean())

    @property
    def success_rate(self) -> float:
        return float(self.success.to(torch.float64).mean())

    def summary(self) -> dict:
        out = {
            "fitness": float(self.fitness.mean()),
            "fitness_best": float(self.fitness.min()),
            "crash_rate": self.crash_rate,
            "success_rate": self.success_rate,
            "final_err": float(self.final_err.mean()),
            "saturation": float(self.saturation.mean()),
            "effort": float(self.effort.mean()),
        }
        for i in range(self.leg_err.shape[1]):
            out[f"leg{chr(ord('A') + i)}_err"] = float(self.leg_err[:, i].mean())
        return out


@dataclass
class Trace:
    """Stacked trajectory, produced only on demand."""
    states: State               # leaves [T+1, B, ...]
    goals: Tensor               # [T, B, task_dim]
    us: Tensor                  # [T, B, n_force]
    alive: Tensor               # [T, B] -- alive *entering* each step


class Rollout:
    """Holds the vmapped controller map so it is built once, not per generation.

    Rebuilding `vmap(trainable.forward)` inside the loop is the single most likely
    cause of a v2 that is much slower than the prototype.
    """

    def __init__(self, system: LagrangianSystem, trainable: Trainable, task: Task,
                 cfg: RolloutCfg):
        self.system, self.trainable, self.task, self.cfg = system, trainable, task, cfg
        # in_dims=(0, 0, 0): every batch entry carries its own genome, state, goal.
        self.forward_batch = vmap(trainable.forward, in_dims=(0, 0, 0))

    # --- layout ------------------------------------------------------------
    def _expand(self, TH: Tensor, goals: Tensor, seed: int):
        """(P, dim) x (E, n_legs, d) -> the B = P*E common-random-numbers batch."""
        P, E = TH.shape[0], goals.shape[0]
        s = self.system.reset(E, make_gen(seed))     # E shared initial states ...
        s = tree_repeat(s, P)                        # ... reused by every genome
        TH_b = TH.repeat_interleave(E, dim=0)        # index = member * E + episode
        goals_b = goals.repeat(P, 1, 1)
        # learned-dynamics residual, if the plant declares one
        res_b = self.trainable.residual_slice(TH_b)
        return s, TH_b, goals_b, res_b, P, E

    # --- the hot path ------------------------------------------------------
    @torch.no_grad()
    def run(self, TH: Tensor, goals: Tensor, seed: int) -> RolloutResult:
        sysm, task, cfg = self.system, self.task, self.cfg
        T, dt = cfg.ep_steps, cfg.dt
        s, TH_b, goals_b, res_b, P, E = self._expand(TH, goals, seed)
        B = P * E

        cost = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)
        sat = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)
        eff_acc = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)
        shp_acc = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)
        alive = sysm.alive(s)
        leg_ends = task.leg_end_steps(T)
        leg_err = torch.zeros(B, task.n_legs, dtype=sysm.dtype, device=sysm.device)
        dead = torch.full((B,), cfg.dead_cost, dtype=sysm.dtype, device=sysm.device)

        for t in range(T):
            goal = task.goal_at(goals_b, t, T)
            u = self.forward_batch(TH_b, s, goal)
            s_new = sysm.step(s, u, dt, res_b)
            # crashed vehicles freeze; never integrate a diverged state
            s = tree_where(alive, s_new, s)

            err = sysm.task_position(s) - goal
            pos = torch.sqrt((err * err).sum(-1) + cfg.pos_eps)
            eff = sysm.effort(u, s)
            shp = sysm.shaping_cost(s)
            live = pos + cfg.lambda_e * eff + cfg.lambda_s * shp
            if res_b is not None:
                live = live + cfg.lambda_r * sysm.residual_penalty(res_b)
            cost = cost + torch.where(alive, live, dead) * dt
            sat = sat + sysm.saturation(u, s)
            eff_acc = eff_acc + eff
            shp_acc = shp_acc + shp

            if t in leg_ends:
                leg_err[:, leg_ends.index(t)] = err.norm(dim=-1)
            alive = alive & sysm.alive(s)

        final_goal = task.goal_at(goals_b, T - 1, T)
        return RolloutResult(
            fitness=cost.view(P, E).mean(dim=1),
            cost=cost,
            alive=alive,
            leg_err=leg_err,
            final_err=(sysm.task_position(s) - final_goal).norm(dim=-1),
            success=task.success(s, final_goal) & alive,
            saturation=sat / T,
            effort=eff_acc / T,
            shaping=shp_acc / T,
            n_eps=E,
        )

    # --- diagnostics path --------------------------------------------------
    @torch.no_grad()
    def trace(self, TH: Tensor, goals: Tensor, seed: int) -> Trace:
        """Same dynamics, but keeps every state.  Separate from `run` so the hot
        path allocates no trace buffers."""
        sysm, task, cfg = self.system, self.task, self.cfg
        T, dt = cfg.ep_steps, cfg.dt
        s, TH_b, goals_b, res_b, P, E = self._expand(TH, goals, seed)

        alive = sysm.alive(s)
        states, gs, us, al = [s], [], [], []
        for t in range(T):
            goal = task.goal_at(goals_b, t, T)
            u = self.forward_batch(TH_b, s, goal)
            s = tree_where(alive, sysm.step(s, u, dt, res_b), s)
            gs.append(goal)
            us.append(u)
            al.append(alive)
            states.append(s)
            alive = alive & sysm.alive(s)
        return Trace(
            states=tree_stack(states),
            goals=torch.stack(gs),
            us=torch.stack(us),
            alive=torch.stack(al),
        )


# --------------------------------------------------------------------------- #
# functional wrappers (spec signatures); the ES loop reuses a `Rollout` instead
# --------------------------------------------------------------------------- #
def rollout(system, trainable, TH, goals, cfg, seed, task: Task, record: bool = False):
    """Convenience wrapper.  The ES loop reuses a `Rollout` instead, so that the
    vmapped controller map is built once rather than per generation."""
    r = Rollout(system, trainable, task, cfg)
    return r.trace(TH, goals, seed) if record else r.run(TH, goals, seed)


def state_trace(system, trainable, TH, goals, cfg, seed, task: Task):
    return Rollout(system, trainable, task, cfg).trace(TH, goals, seed)
