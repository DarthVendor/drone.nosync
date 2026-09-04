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
from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch.func import vmap

from .config import RolloutCfg
from .sensors.base import DelayBuffer, Sensor
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
    success: Tensor             # [B]  reached the final waypoint
    legs_done: Tensor           # [B]  waypoints reached, arrival gating only
    finish_frac: Tensor         # [B]  fraction of the episode taken to finish (1 = never)
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
            success=self.success[ep], legs_done=self.legs_done[ep],
            finish_frac=self.finish_frac[ep], saturation=self.saturation[ep],
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
            "legs_done": float(self.legs_done.to(torch.float64).mean()),
            "finish_frac": float(self.finish_frac.mean()),
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
                 cfg: RolloutCfg, sensors: Optional[Sequence[Sensor]] = None):
        self.system, self.trainable, self.task, self.cfg = system, trainable, task, cfg
        self.sensors: List[Sensor] = list(sensors or [])
        # Per-sensor delay, not one global lag: flow and IMU run at ~2 ms, ToF at
        # 5-20 ms, vision at 30-80 ms, and collapsing them loses the very
        # timescale separation the allocator/potential split depends on.
        self.buffers = [DelayBuffer(s.latency_steps) for s in self.sensors]
        self._held: dict = {}
        # The pullback Jacobian costs about as much as the projection itself, so
        # it is computed only when some term declares it consumes observations.
        self._needs_jac = any(getattr(t, "uses_obs", False)
                              for t in getattr(trainable, "terms", ()))
        for sen in self.sensors:
            sen.crn_group = cfg.n_eps          # noise shared across the population
        # in_dims: every batch entry carries its own genome, state, goal (and obs).
        # The sensor-free path keeps the original 3-argument signature verbatim,
        # so "no sensors" is bit-identical rather than merely equivalent.
        self.forward_batch = vmap(trainable.forward,
                                  in_dims=(0, 0, 0, 0) if self.sensors
                                  else (0, 0, 0))

    # --- sensing ------------------------------------------------------------
    def _prime(self, s: State, gen) -> None:
        self._held = {}
        for sen, buf in zip(self.sensors, self.buffers):
            buf.reset(sen.observe(s, gen))

    def _observe(self, s: State, gen, step: int = 0) -> dict:
        """Delayed observations plus their pullback Jacobians.

        The MEASUREMENT is delayed; the Jacobian is evaluated at the current
        state.  That asymmetry is deliberate and matches how these loops are
        actually flown: the camera is late, but the vehicle's own pose comes from
        the IMU at loop rate, so the geometry used to pull image error back into
        task space is fresh even when the pixels are not.  It is the same
        dead-reckon-between-updates rule as the skip policy.
        """
        out = {}
        for sen, buf in zip(self.sensors, self.buffers):
            k = max(1, int(getattr(sen, "update_every", 1)))
            if step % k == 0 or sen.name not in self._held:
                fresh = sen.observe(s, gen)
                jac = sen.jacobian(s) if self._needs_jac else None
                self._held[sen.name] = (fresh, jac)
            else:
                fresh, jac = self._held[sen.name]
            # the delay buffer still advances every step, so a strided sensor is
            # stale by (stride - 1) steps on top of its own latency
            out[sen.name] = buf.push(fresh)
            if jac is not None:
                out[sen.name + "/J"] = jac
        return out

    def _u(self, TH_b, s, goal, obs):
        return (self.forward_batch(TH_b, s, goal, obs) if self.sensors
                else self.forward_batch(TH_b, s, goal))

    # --- layout ------------------------------------------------------------
    def _expand(self, TH: Tensor, goals: Tensor, seed: int):
        """(P, dim) x (E, n_legs, d) -> the B = P*E common-random-numbers batch."""
        P, E = TH.shape[0], goals.shape[0]
        s = self.system.reset(E, make_gen(seed))     # E shared initial states ...
        s = tree_repeat(s, P)                        # ... reused by every genome
        TH_b = TH.repeat_interleave(E, dim=0)        # index = member * E + episode
        goals_b = goals.repeat(P, 1, 1)
        if getattr(self.system, "needs_course", False):
            # geometry and waypoints come from different generators, so a plant
            # that wants gates ON the route has to be handed the route
            s = self.system.place_course(s, goals_b)
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
        arrival = getattr(task, "gating", "time") == "arrival"
        leg = torch.zeros(B, dtype=torch.long, device=sysm.device)
        finish = torch.full((B,), float(T), dtype=sysm.dtype, device=sysm.device)

        # Sensor noise joins common random numbers: one stream per generation,
        # drawn per episode and tiled across the population.  Without this,
        # sensor stochasticity becomes fitness-ranking variance and ES is
        # already variance-limited.
        sgen = make_gen(seed + 5_701_889)
        self._prime(s, sgen)

        for t in range(T):
            goal = task.goal_for_leg(goals_b, leg) if arrival \
                else task.goal_at(goals_b, t, T)
            u = self._u(TH_b, s, goal, self._observe(s, sgen, t))
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

            if arrival:
                # advance only on ARRIVAL, so reaching a waypoint early buys a
                # longer tail of low cost at the next one -- the incentive that
                # makes the fastest route the cheapest
                reached = (err.norm(dim=-1) < task.tol) & alive
                last = leg >= task.n_legs - 1
                finish = torch.where(reached & last & (finish >= T),
                                     torch.full_like(finish, float(t)), finish)
                leg = torch.where(reached & ~last, leg + 1, leg)
            elif t in leg_ends:
                leg_err[:, leg_ends.index(t)] = err.norm(dim=-1)
            alive = alive & sysm.alive(s)

        final_goal = task.goal_for_leg(goals_b, leg) if arrival \
            else task.goal_at(goals_b, T - 1, T)
        done = (finish < T) if arrival else task.success(s, final_goal)
        if arrival:
            leg_err[:, -1] = (sysm.task_position(s) - final_goal).norm(dim=-1)
        return RolloutResult(
            fitness=cost.view(P, E).mean(dim=1),
            cost=cost,
            alive=alive,
            leg_err=leg_err,
            final_err=(sysm.task_position(s) - final_goal).norm(dim=-1),
            success=(done & alive) if arrival
                    else (task.success(s, final_goal) & alive),
            legs_done=leg + done.to(leg.dtype),
            finish_frac=finish / T,
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
        sgen = make_gen(seed + 5_701_889)
        self._prime(s, sgen)
        # the replay has to advance goals exactly as training did, or a trace
        # shows a different flight from the one that was scored
        arrival = getattr(task, "gating", "time") == "arrival"
        leg = torch.zeros(goals_b.shape[0], dtype=torch.long, device=sysm.device)
        states, gs, us, al = [s], [], [], []
        for t in range(T):
            goal = task.goal_for_leg(goals_b, leg) if arrival \
                else task.goal_at(goals_b, t, T)
            u = self._u(TH_b, s, goal, self._observe(s, sgen, t))
            s = tree_where(alive, sysm.step(s, u, dt, res_b), s)
            gs.append(goal)
            us.append(u)
            al.append(alive)
            states.append(s)
            if arrival:
                err = (sysm.task_position(s) - goal).norm(dim=-1)
                reached = (err < task.tol) & alive
                leg = torch.where(reached & (leg < task.n_legs - 1), leg + 1, leg)
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
