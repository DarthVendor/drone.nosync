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
    legs: Tensor                # [T, B] -- which waypoint was the target.
                                # Under arrival gating this advances on ARRIVAL,
                                # so a replay cannot infer it from the frame
                                # index: two episodes of the same length sit on
                                # different legs at the same instant, and a
                                # crashed one never advances again.


class ChargeMemory:
    """Ring buffer of world-frame sensor returns.

    Fixed shape and a step index that never depends on tensor values, so it stays
    vmap-safe.  Misses are written too, carrying weight zero: dropping them
    instead would make the write length depend on the data and break the shape.
    """

    def __init__(self, slots: int = 96):
        self.slots = int(slots)
        self.p: Optional[Tensor] = None
        self.w: Optional[Tensor] = None
        self.i = 0

    def reset(self) -> None:
        self.p, self.w, self.i = None, None, 0

    def write(self, hit: Tensor, seen: Tensor) -> None:
        B, n, d = hit.shape
        # a strided sensor hands back a held reading whose batch may be narrower
        # than the state's; the weights must line up with the points
        if seen.shape[0] != B:
            seen = seen.expand(B, *seen.shape[1:])
        if self.p is None or self.p.shape[0] != B:
            self.p = torch.zeros(B, self.slots, d, dtype=hit.dtype,
                                 device=hit.device)
            self.w = torch.zeros(B, self.slots, dtype=hit.dtype,
                                 device=hit.device)
            self.i = 0
        idx = (torch.arange(n, device=hit.device) + self.i) % self.slots
        # out-of-place: the buffer is read inside the vmapped controller map
        self.p = self.p.index_copy(1, idx, hit)
        self.w = self.w.index_copy(1, idx, seen)
        self.i = (self.i + n) % self.slots

    def read(self):
        return self.p, self.w


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
        # A harmonic obstacle field needs its charges FIXED in the world frame.
        # Body-fixed beams re-aim as the vehicle moves, so a potential written as
        # a function of range is a field of sliding sources and is not harmonic
        # (measured: lap V = 69.4 sliding vs 0.004 frozen, same beams).  Freezing
        # them is the whole fix, so the memory lives here.
        self._needs_charges = any(getattr(t, "needs_charges", False)
                                  for t in getattr(trainable, "terms", ()))
        self.charge_mem = ChargeMemory(cfg.charge_slots) if self._needs_charges \
            else None
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
        if self.charge_mem is not None:
            self.charge_mem.reset()
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
                if self.charge_mem is not None and sen.kind == "range":
                    # J = d(range)/d(x) = -beam direction, so the return landed at
                    # x + d*u = x - d*J.  Recorded in WORLD coordinates and kept,
                    # which is what makes the resulting field harmonic.
                    x = self.system.task_position(s)
                    hit = x[..., None, :] - jac * fresh[..., None]
                    seen = (fresh < sen.max_range * 0.98).to(fresh.dtype)
                    self.charge_mem.write(hit, seen)
        if self.charge_mem is not None:
            out["charges"], out["charge_w"] = self.charge_mem.read()
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
        # The final leg never advances `leg`, so its arrival test keeps firing for
        # every step the vehicle sits inside tol -- paying the bonus per step
        # would make hovering on the goal an unbounded reward.  Credit it once.
        paid_last = torch.zeros(B, dtype=torch.bool, device=sysm.device)
        credits = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)
        frozen_dead = cfg.dead_mode == "frozen"
        forfeit_dead = cfg.dead_mode == "forfeit"
        # what the episode owed before it moved: a crash is charged this, so the
        # progress it made is handed back and dying late buys nothing
        start_err = torch.sqrt(
            ((sysm.task_position(s) - task.goal_for_leg(goals_b, leg))
             ** 2).sum(-1) + cfg.pos_eps) if forfeit_dead else None

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
            # in-place: this runs under no_grad, and the accumulators were each
            # allocating a fresh [B] tensor on every one of 250 steps
            # A dead vehicle is frozen at its crash site, so under "frozen" it
            # keeps paying the position term from there and needs no constant of
            # its own.  Effort and shaping are dropped: it is not actuating.
            charge = (pos if frozen_dead
                      else start_err if forfeit_dead else dead)
            cost.add_(torch.where(alive, live, charge), alpha=dt)
            sat.add_(sysm.saturation(u, s))
            eff_acc.add_(eff)
            shp_acc.add_(shp)

            if arrival:
                # advance only on ARRIVAL, so reaching a waypoint early buys a
                # longer tail of low cost at the next one -- the incentive that
                # makes the fastest route the cheapest
                reached = (err.norm(dim=-1) < task.tol) & alive
                last = leg >= task.n_legs - 1
                finish = torch.where(reached & last & (finish >= T),
                                     torch.full_like(finish, float(t)), finish)
                if cfg.goal_bonus:
                    # An intermediate leg can only be credited once because
                    # reaching it advances `leg`; the last one needs `paid_last`.
                    hit = (reached & ~last) | (reached & last & ~paid_last)
                    credits.add_(hit.to(sysm.dtype))
                    paid_last = paid_last | (reached & last)
                leg = torch.where(reached & ~last, leg + 1, leg)
            elif t in leg_ends:
                leg_err[:, leg_ends.index(t)] = err.norm(dim=-1)
            alive = alive & sysm.alive(s)

        final_goal = task.goal_for_leg(goals_b, leg) if arrival \
            else task.goal_at(goals_b, T - 1, T)
        # Bonus is paid at the END, and only to survivors.  Crediting it on
        # contact instead would make "touch the goal, then crash" score almost as
        # well as completing the task -- fitness would improve while `success`,
        # which requires being alive, fell.  An objective that disagrees with the
        # metric it is judged by reads exactly like a plateau.
        if cfg.goal_bonus:
            cost.sub_(credits * alive.to(sysm.dtype), alpha=cfg.goal_bonus)
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
        states, gs, us, al, lg = [s], [], [], [], []
        for t in range(T):
            goal = task.goal_for_leg(goals_b, leg) if arrival \
                else task.goal_at(goals_b, t, T)
            u = self._u(TH_b, s, goal, self._observe(s, sgen, t))
            s = tree_where(alive, sysm.step(s, u, dt, res_b), s)
            gs.append(goal)
            us.append(u)
            al.append(alive)
            lg.append(leg.clone())
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
            legs=torch.stack(lg),
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
