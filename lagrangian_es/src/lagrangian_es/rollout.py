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
from .composer.spec import TaskSpec
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
    # The LOW LEVEL's objective when a composer is in the loop: the same cost
    # with the distance term measured to the subgoal the composer PLACED (the
    # hold's target) instead of to the task goal.  Without a composer both
    # are the task cost.  The composer is judged on `cost`; a co-training run
    # ranks genomes on `fitness_sub`, so each layer answers for its own job.
    cost_sub: Tensor = None     # [B]
    fitness_sub: Tensor = None  # [P]
    death_step: Tensor = None   # [B]  step the episode crashed at; ep_steps if it never did

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
            cost_sub=None if self.cost_sub is None else self.cost_sub[ep],
            fitness_sub=None if self.fitness_sub is None else self.fitness_sub[sl],
            death_step=None if self.death_step is None else self.death_step[ep],
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
                 cfg: RolloutCfg, sensors: Optional[Sequence[Sensor]] = None,
                 composer=None):
        self.system, self.trainable, self.task, self.cfg = system, trainable, task, cfg
        self.composer = composer
        # the vehicle's own map, if the config asks for one: it duck-types a
        # sensor closely enough to attach through the same path, but it is NOT
        # one -- it consumes what the range sensor returned rather than the world
        self.built_map = None
        self._beam_sen = next((x for x in (sensors or ())
                               if getattr(x, "kind", "") == "range" and hasattr(x, "_dirs")), None)
        if getattr(cfg, "built_map", False) and self._beam_sen is not None:
            from .mapping import BuiltMap
            self.built_map = BuiltMap(**dict(getattr(cfg, "built_map_kw", ()) or ()))
            self.built_map.kind, self.built_map.name = "map", "map_built"
        if composer is not None and hasattr(composer, "attach"):
            composer.attach(list(sensors or []) + ([self.built_map] if self.built_map is not None else []))
        self.chain: list = []          # measurement tokens, one entry per interval
        self._last_obs: dict = {}
        self.sensors: List[Sensor] = list(sensors or [])
        if composer is not None:
            # A sensor nobody reads more often than the composer does is cast
            # at the composer's cadence.  Profiled on the v2 rig: the depth
            # camera was 70% of an iteration, cast every step, read every ten
            # -- by the composer only, since no term names it.  Refresh and
            # decision share the same phase (`step % k == 0`), so the composer
            # still sees a frame fresh up to the sensor's own latency.
            read_by_terms = set()
            for t in getattr(trainable, "terms", ()):
                n = getattr(t, "sensor_name", None)
                read_by_terms.update(n if isinstance(n, (tuple, list)) else ([n] if n else []))
            # a decision can fall on any report step, so the composer's own
            # sensors are cast at the report cadence
            every = int(getattr(composer, "measure_every", getattr(composer, "every", 1)))
            for sen in self.sensors:
                if sen.name not in read_by_terms and every > int(getattr(sen, "update_every", 1)):
                    sen.update_every = every
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
        # The composer path is a SEPARATE vmapped map, so a rollout without one
        # is the same function object as before rather than merely equivalent.
        # The spec rides as a (delta, weight) tuple: vmap batches tensors, not
        # dataclasses.
        if composer is not None:
            self.forward_spec_yaw = (
                vmap(lambda th, st, g, o, sp: trainable.forward(th, st, g, o, spec=sp),
                     in_dims=(0, 0, 0, 0, (0, 0, 0, 0))) if self.sensors else
                vmap(lambda th, st, g, sp: trainable.forward(th, st, g, spec=sp),
                     in_dims=(0, 0, 0, (0, 0, 0, 0))))
            self.forward_spec = (
                vmap(lambda th, st, g, o, sp: trainable.forward(th, st, g, o, spec=sp),
                     in_dims=(0, 0, 0, 0, (0, 0))) if self.sensors else
                vmap(lambda th, st, g, sp: trainable.forward(th, st, g, spec=sp),
                     in_dims=(0, 0, 0, (0, 0))))
        if getattr(cfg, "compile_forward", False):
            # Compiled OUTSIDE the vmap, not inside it: `compile(vmap(f))` is
            # one graph over the whole population, while `vmap(compile(f))`
            # re-enters the compiled callable per batch element.  Measured 1.78x
            # on the learned rig, bit-matching eager.
            #
            # Guarded: a compile failure must degrade to eager rather than take
            # the run down, and every backend/version combination is its own
            # question.
            try:
                self.forward_batch = torch.compile(self.forward_batch,
                                                   dynamic=False)
                if composer is not None:
                    # the spec paths are the ones a composer drives every
                    # step; eager they were the largest cost of a rollout
                    # (35%: hundreds of small ops per step in the learned trunk)
                    self.forward_spec = torch.compile(self.forward_spec, dynamic=False)
                    self.forward_spec_yaw = torch.compile(self.forward_spec_yaw, dynamic=False)
            except Exception:
                pass

    # --- sensing ------------------------------------------------------------
    def _prime(self, s: State, gen) -> None:
        self._held = {}
        self._raw = {}          # noiseless readings, kept so frozen episodes
                                # can be skipped without re-marching them
        # per-row state version: bumped every step the row is integrated, so a
        # sensor knows which rows moved since IT last read them (a strided
        # sensor reads every k steps; a row that moved and then froze between
        # two reads is not in any per-step mask)
        any_ = next(iter(s.values()))        # plant-agnostic: the batch size and device of the state
        self._ver = torch.zeros(any_.shape[0], dtype=torch.long, device=any_.device)
        # NOT cached: skipping the controller for arrived episodes was tried
        # and is in the git history as a dead end.  It is 44% of a city rollout,
        # but `s` carries per-episode obstacle geometry, so indexing the batch
        # down to the awake rows costs more than the evaluation it saves -- 0.96x
        # on the city, 1.12x on pillars.  It was also not bit-exact: the control
        # cached on the step an episode arrives was computed from the state
        # BEFORE that step, not the frozen state after it.
        if self.charge_mem is not None:
            self.charge_mem.reset()
        if self.built_map is not None:                # every episode starts knowing nothing
            self.built_map.reset(any_.shape[0], any_.device, self.system.task_position(s).dtype)
        for sen, buf in zip(self.sensors, self.buffers):
            buf.reset(sen.observe(s, gen))

    def _observe(self, s: State, gen, step: int = 0, live=None) -> dict:
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
                raw, jac = self._measure(sen, s, live)
                # noise is drawn AFTER, at full batch width, from the same
                # generator in the same order -- so skipping frozen episodes
                # cannot shift the common-random-numbers stream
                fresh = sen.perturb(raw, gen) if sen.stateless \
                    else (sen.observe(s, gen) if not self._needs_jac
                          else sen.observe_with_jacobian(s, gen)[0])
                self._held[sen.name] = (fresh, jac)
            else:
                fresh, jac = self._held[sen.name]
            # the delay buffer still advances every step, so a strided sensor is
            # stale by (stride - 1) steps on top of its own latency
            out[sen.name] = buf.push(fresh)
            if jac is not None:
                out[sen.name + "/J"] = jac
                if hasattr(sen, "_dirs"):
                    # the beam directions, for the potential's yaw gradient:
                    # rotating the body rotates them, and a range's sensitivity
                    # to that is r (J . (e_z x d)) from the same Jacobian
                    out[sen.name + "/dir"] = sen._dirs(s)
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
        if self.built_map is not None:
            # Built from what the BEAMS RETURNED, never from the world: the same
            # delayed, strided, noisy numbers the rest of the stack sees, so a
            # remembered wall can be in the wrong place exactly as the vehicle
            # believes it to be.
            #
            # Only on a step the fan actually refreshed.  A strided sensor HOLDS
            # its reading in between, so marking every step counted one sighting
            # five times over -- inflating the confidence, and refreshing the
            # last-seen stamp so that a cell's age could never grow past the
            # stride.  The map is a record of measurements, so it advances when
            # a measurement arrives.
            if step % max(1, int(getattr(self._beam_sen, "update_every", 1))) == 0 or step == 0:
                x = self.system.task_position(s)
                self.built_map.update(x, self._beam_sen._dirs(s), out[self._beam_sen.name],
                                      self._beam_sen.max_range, live=live, t=float(step))
        return out

    def _measure(self, sen, s: State, live):
        """Noiseless reading and Jacobian, marching only what is still flying.

        An episode that has crashed or arrived is frozen by `tree_where`, so its
        state -- vehicle AND geometry -- is bit-identical to the step before.  A
        sensor whose reading is a pure function of that state therefore returns
        exactly what it returned last time, which makes the cached value the
        right answer rather than a stale one.  Skipping it is an optimisation
        with no approximation in it, and the test asserts that.

        The batch is NOT compacted.  `crn_noise` draws `[n_eps, ...]` and tiles
        it across the population, so a smaller batch would change `B % n_eps`,
        fall through to an untiled draw, and silently give every episode
        different noise.  Indexing the expensive call while leaving the batch
        shape alone keeps that stream exactly where it was.

        Which rows to march is decided by the state VERSION each row carried
        when this sensor last read it, not by who is awake now: a strided
        sensor reads every k steps, and a row that moved after its last read
        and then crashed or arrived is frozen now yet its cached reading is
        of a state it left (measured 5.7 m off across a pillar edge, feeding
        the frozen row's actuation and effort for the rest of the episode).
        `live=None` forces a full march (the acceptance test's reference).
        """
        cached = self._raw.get(sen.name)
        dirty = None if (live is None or cached is None) else (self._ver != cached[2])
        if not sen.stateless or dirty is None or bool(dirty.all()):
            if sen.stateless:
                out = sen.measure(s)
                raw, jac = (out[0], out[1] if self._needs_jac else None)
            else:
                raw = sen.observe(s, None)
                jac = sen.jacobian(s) if self._needs_jac else None
            self._raw[sen.name] = (raw, jac, self._ver.clone())
            return raw, jac
        idx = dirty.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return cached[0], cached[1]
        sub = {k: v[idx] for k, v in s.items()}
        r_sub, j_sub = sen.measure(sub)
        raw = cached[0].clone()
        raw[idx] = r_sub
        jac = None
        if self._needs_jac and cached[1] is not None:
            # `cached[1] is None` for a sensor that has no pullback at all --
            # the carried map is read by the composer and never by the
            # potential, so there is no Jacobian to patch (`MapPrior.measure`
            # returns None for it).
            jac = cached[1].clone()
            jac[idx] = j_sub
        self._raw[sen.name] = (raw, jac, self._ver.clone())
        return raw, jac

    def _u(self, TH_b, s, goal, obs, spec=None):
        if spec is None:
            return (self.forward_batch(TH_b, s, goal, obs) if self.sensors
                    else self.forward_batch(TH_b, s, goal))
        sp = (spec.delta, spec.weight) if spec.yaw is None else \
            (spec.delta, spec.weight, spec.yaw, spec.yaw_gate if spec.yaw_gate is not None else torch.ones_like(spec.yaw))
        fn = self.forward_spec_yaw if spec.yaw is not None else self.forward_spec
        return (fn(TH_b, s, goal, obs, sp) if self.sensors else fn(TH_b, s, goal, sp))

    # --- the task-level layer -------------------------------------------------
    def _hold_for(self, B: int):
        """A fresh zero-order hold for this batch, rated from the plant's own
        bandwidth rather than a tuned number."""
        from .composer import SpecHold
        sysm, comp = self.system, self.composer
        omega = float(getattr(comp, "omega_n", 0.0)) or \
            float(sysm.potential_scale()) ** 0.5
        return SpecHold(B, sysm.task_dim, comp.n_terms, self.cfg.dt, sysm.dtype,
                        sysm.device, omega_n=omega,
                        reach=float(getattr(comp, "reach", 10.0)))

    def _emit_live(self, comp, s, goal, alive, arrived, leg, t, hold, rows=None):
        """Ask the composer about `rows` (or the live rows); the context
        carries each row's current target spec, which a token-emitting
        composer applies its action to."""
        """Ask the composer only about rows still flying -- or, with `rows`,
        exactly the rows whose turn it is.

        A dead or arrived row's state is frozen: its observations are never
        read again and its decisions change nothing in the cost.  At worker
        scale the composer cost 1.68 s per call on 1152 rows, 180 calls an
        episode-batch, on every row to the last step -- with 70% of them dead
        for most of it.  Rows not asked keep the target they had.
        """
        live = (alive & ~arrived) if rows is None else rows
        n_live = int(live.sum())
        B = alive.shape[0]
        if n_live == B or (rows is None and not getattr(comp, "live_only", True)):
            ctx = self._context(s, goal, self._last_obs, alive, arrived, leg, t); ctx["spec"] = hold.target
            return comp.emit(ctx)
        if n_live == 0:
            return hold.target
        idx = live.nonzero().flatten()
        sub = {k: (v[idx] if torch.is_tensor(v) and v.ndim and v.shape[0] == B else v) for k, v in s.items()}
        obs = {k: (v[idx] if torch.is_tensor(v) and v.ndim and v.shape[0] == B else v) for k, v in self._last_obs.items()}
        chain = [{k: (v[idx] if torch.is_tensor(v) and v.ndim and v.shape[0] == B else v) for k, v in tok.items()}
                 for tok in self.chain]
        ctx = {"x": self.system.task_position(sub), "v": self.system.task_velocity(sub), "goal": goal[idx],
               "alive": alive[idx], "arrived": arrived[idx], "leg": leg[idx], "t": t, "state": sub, "chain": chain}
        ctx["spec"] = TaskSpec(hold.target.delta[idx], hold.target.alpha[idx], hold.target.gate[idx],
                               None if hold.target.yaw is None else hold.target.yaw[idx],
                               None if hold.target.yaw_gate is None else hold.target.yaw_gate[idx])
        if obs:
            ctx["obs"] = obs
        comp._rows = idx                              # so the composer's own per-row memory can scatter
        part = comp.emit(ctx)
        comp._rows = None
        full = hold.target.clone()
        if part.moved is not None:
            full.moved = torch.zeros(B, dtype=torch.bool, device=idx.device); full.moved[idx] = part.moved
        full.delta[idx] = part.delta; full.alpha[idx] = part.alpha; full.gate[idx] = part.gate
        if part.yaw is not None:
            if full.yaw is None:
                full.yaw = torch.zeros(B, dtype=part.yaw.dtype, device=part.yaw.device)
                full.yaw_gate = torch.zeros(B, dtype=part.yaw.dtype, device=part.yaw.device)
            full.yaw[idx] = part.yaw
            full.yaw_gate[idx] = part.yaw_gate if part.yaw_gate is not None else 1.0
        return full

    def _composer_start(self, s, goal=None):
        """The task layer's state for a batch: the hold, the drone's report
        stream, and the event clock that decides when the composer is asked.

        A token-emitting composer starts from an OPENING action: the straight
        placement at full reach, applied once at takeoff and written into its
        chain like any other token.  It is the state the old identity prior
        gave every flight; silent, the subgoal would be the goal itself, and
        measured, the low level flown at a 17 m target reaches nothing
        (0.000 / 0.777) where the reach-limited point reaches 0.26.  From
        here on every change is the composer's own."""
        comp, sysm = self.composer, self.system
        x = sysm.task_position(s); B = x.shape[0]
        every = int(getattr(comp, "every", 50))
        comp.reset(B); self.chain = []
        if hasattr(comp, "pair") and getattr(self, "_crn", None) is not None:
            comp.pair(*self._crn)
        cs = {"every": every, "m_every": int(getattr(comp, "measure_every", every)),
              "hold": self._hold_for(B), "x_int": x, "beam_min": None,
              "t_last": torch.full((B,), -every, dtype=torch.long, device=x.device),
              "n_sub": torch.zeros(B, dtype=torch.long, device=x.device), "leg_last": None,
              "tol": float(getattr(self.task, "tol", 1.0))}
        vocab = getattr(getattr(comp, "net", None), "vocab", None)
        if goal is not None and vocab is not None and hasattr(comp, "_apply"):
            ctx = self._context(s, goal, {}, torch.ones(B, dtype=torch.bool, device=x.device),
                                torch.zeros(B, dtype=torch.bool, device=x.device), torch.zeros(B, dtype=torch.long, device=x.device), 0)
            ctx["spec"] = cs["hold"].target
            with torch.no_grad():
                opening = comp._apply(torch.full((B,), vocab.straight, dtype=torch.long, device=x.device), ctx, comp.tokens(ctx))
            cs["hold"].set_target(opening)
            cs["n_sub"] += 1
        return cs

    def _composer_step(self, cs, s, goal, alive, arrived, leg, t, cost):
        """The task layer's turn at step `t`: the drone's report, then -- for
        the rows whose turn it is -- a decision.  Returns the spec in force.

        A decision is an EVENT, not a tick.  A row is asked again when the
        subgoal placed for it has been achieved (within the task's arrival
        tolerance, and the subgoal was not the goal itself -- a drone dwelling
        on the goal has nothing left to be told), when its leg changed (the
        placed offset is relative to the goal), or when the hold has run
        `every` steps -- and only at a report step, so the composer decides on
        a fresh report.  Every decision is one subgoal; `n_sub` counts them,
        and a co-training run charges the composer per subgoal so it reaches
        the goal with the fewest.  The report stream is unconditional: the
        composer keeps watching between decisions.
        """
        comp, sysm, hold = self.composer, self.system, cs["hold"]
        m_every, every = cs["m_every"], cs["every"]
        x = sysm.task_position(s)
        if t and t % m_every == 0:
            tok = self._token(cs["x_int"], x, goal + hold.target.delta, cs["beam_min"], alive, arrived,
                              speed=self.system.task_velocity(s).norm(dim=-1))
            tok["t"] = float(t)
            # the objective itself, so a learner above reads the same cost the
            # low level was evolved on and nothing shaped
            tok["cost"] = cost.clone()
            self.chain.append(tok)
            kc = int(getattr(comp, "k_chain", 64))
            if len(self.chain) > 4 * kc:
                self.chain = self.chain[-2 * kc:]
            cs["x_int"] = x; cs["beam_min"] = None
        # the placed subgoal is a WORLD point: when a row's goal changes (a leg
        # done) the offset the hold carries is re-expressed against the new goal
        # -- but only where something WAS placed.  `delta` is defined relative
        # to the goal, so a row whose subgoal is the goal itself (delta 0: the
        # identity composer, or a row that never placed) follows the goal to
        # the new leg exactly as the controller without a composer does; the
        # world-point rule is for subgoals the composer chose.
        if cs.get("goal_prev") is not None:
            jump = goal - cs["goal_prev"]
            # (a tolerance, not an exact zero: a straight placement that lands ON
            # the goal leaves an offset of ~1e-15 through the ego transform)
            placed = (hold.target.delta.norm(dim=-1, keepdim=True) > 1e-6) | (hold.realized.delta.norm(dim=-1, keepdim=True) > 1e-6)
            j = (jump != 0).any(-1) & placed.squeeze(-1)
            if bool(j.any()):
                shift = torch.where(placed, jump, torch.zeros_like(jump))
                hold.target.delta = hold.target.delta - shift; hold.realized.delta = hold.realized.delta - shift
                # ... and the next placement on such a row takes effect at
                # once (`SpecHold.snap`): the leg change is the task's step,
                # not the composer's move, so it is not slewed
                cs["jumped"] = j if cs.get("jumped") is None else (cs["jumped"] | j)
        cs["goal_prev"] = goal.clone()
        if t % m_every == 0:
            if cs["leg_last"] is None:
                cs["leg_last"] = leg.clone()
            live = alive & ~arrived
            if getattr(comp, "monitors", False):
                # a token-emitting composer is asked at EVERY report about every
                # live row, and decides for itself what, if anything, changes
                if bool(live.any()):
                    spec = self._emit_live(comp, s, goal, alive, arrived, leg, t, hold, rows=live)
                    hold.set_target(spec, rows=live)
                    moved = live if spec.moved is None else (live & spec.moved)
                    self._snap_jumped(cs, hold, moved)                  # the next PLACEMENT snaps, not the next report
                    cs["n_sub"] += moved.to(cs["n_sub"].dtype)
            else:
                placed = hold.target.delta
                achieved = ((x - (goal + placed)).norm(dim=-1) < cs["tol"]) & (placed.norm(dim=-1) > cs["tol"])
                due = live & (achieved | (t - cs["t_last"] >= every) | (leg != cs["leg_last"]))
                if bool(due.any()):
                    spec = self._emit_live(comp, s, goal, alive, arrived, leg, t, hold, rows=due)
                    hold.set_target(spec, rows=due)
                    self._snap_jumped(cs, hold, due if spec.moved is None else (due & spec.moved))
                    cs["t_last"] = torch.where(due, torch.full_like(cs["t_last"], t), cs["t_last"])
                    cs["n_sub"] += due.to(cs["n_sub"].dtype)
            cs["leg_last"] = leg.clone()
        return hold.step()

    @staticmethod
    def _snap_jumped(cs, hold, rows):
        """Rows placed on since their goal jumped realize the placement at once."""
        j = cs.get("jumped")
        if j is None:
            return
        snap = rows & j
        if bool(snap.any()):
            hold.snap(snap)
        cs["jumped"] = j & ~rows

    def _context(self, s, goal, obs, alive, arrived, leg, t):
        """What the composer sees once an interval: the raw pieces.  Ego-centric
        framing is the composer's own job."""
        sysm = self.system
        x = sysm.task_position(s)
        ctx = {"x": x, "v": sysm.task_velocity(s),
               "goal": goal, "alive": alive, "arrived": arrived, "leg": leg,
               "t": t, "state": s, "chain": self.chain}
        if obs:
            ctx["obs"] = obs
        if self.built_map is not None:
            # Read HERE rather than in `_observe`: the k nearest remembered
            # cells are wanted only when the composer is deciding (every ~20
            # steps), and computing them every step was 95% waste -- a top-k
            # over the whole grid, 1800 times an episode instead of ~90.
            ctx["obs"] = dict(obs or {})
            ctx["obs"]["map_built"] = self.built_map.read(x, now=float(t))
        return ctx

    @staticmethod
    def _to_go(goals_b: Tensor):
        """[B, n_legs] path length still to fly AFTER the current leg's
        waypoint, indexed by leg: the sum of the later legs (padded legs
        repeat their predecessor and add nothing).  None for a single goal."""
        if goals_b.ndim != 3 or goals_b.shape[1] < 2:
            return None
        seg = (goals_b[:, 1:] - goals_b[:, :-1]).norm(dim=-1)          # [B, n-1]: length of leg k >= 1
        suffix = seg.flip(1).cumsum(1).flip(1)                           # [B, n-1]: legs j+1 .. last
        return torch.cat([suffix, torch.zeros_like(suffix[:, :1])], 1)  # [B, n]: to go after leg j

    @staticmethod
    def _token(x0, x1, sub, beam_min, alive, arrived, speed=None):
        """The measurement token: what the last instruction actually did --
        progress toward the subgoal it was given, the closest any beam came,
        how fast it is going, and whether it is still flying.  Reported, never
        hoped."""
        e0 = (x0 - sub).norm(dim=-1); e1 = (x1 - sub).norm(dim=-1)
        return {"progress": e0 - e1, "remaining": e1, "min_beam": beam_min,
                "alive": alive.clone(), "arrived": arrived.clone(),
                "speed": None if speed is None else speed.clone()}

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
        # A task may also own where the episode BEGINS.  `reset` picks a start
        # before any goal exists, so on a map whose waypoints span the whole
        # window the first leg is drawn between two unrelated points -- measured
        # at 22-36 m against a `max_leg` of 10.  That is not a hard task, it is
        # an unreachable one, and it is invisible until the goal distances are
        # printed.
        if hasattr(self.task, "place_start"):
            s = self.task.place_start(s, goals_b)
        # learned-dynamics residual, if the plant declares one
        res_b = self.trainable.residual_slice(TH_b)
        return s, TH_b, goals_b, res_b, P, E

    # --- the hot path ------------------------------------------------------
    @torch.no_grad()
    def run(self, TH: Tensor, goals: Tensor, seed: int) -> RolloutResult:
        sysm, task, cfg = self.system, self.task, self.cfg
        T, dt = cfg.ep_steps, cfg.dt
        s, TH_b, goals_b, res_b, P, E = self._expand(TH, goals, seed)
        self._crn = (E, seed)                 # the composer pairs its token draws by episode across the population
        B = P * E

        cost = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)
        cost_sub = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)   # to the placed subgoal
        sat = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)
        eff_acc = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)
        shp_acc = torch.zeros(B, dtype=sysm.dtype, device=sysm.device)
        alive = sysm.alive(s)
        death_step = torch.full((B,), T, dtype=torch.long, device=sysm.device)   # when a row crashed, for the record
        leg_ends = task.leg_end_steps(T)
        leg_err = torch.zeros(B, task.n_legs, dtype=sysm.dtype, device=sysm.device)
        dead = torch.full((B,), cfg.dead_cost, dtype=sysm.dtype, device=sysm.device)
        arrival = getattr(task, "gating", "time") == "arrival"
        leg = torch.zeros(B, dtype=torch.long, device=sysm.device)
        # each episode's final leg -- a constant for most tasks, per-episode for
        # a tour whose length varies
        last_idx = task.last_leg(goals_b)
        # DISTANCE TO GO, not distance to the current waypoint.  Charging the
        # distance to the active waypoint means finishing a leg RAISES the
        # charge, from ~0 to the next leg's length, for every second left.  A
        # learned composer found that spot exactly: it placed its subgoal
        # 0.28 m short of the first waypoint (tolerance 0.25 m) and parked
        # there for the rest of the episode -- 0.000 reach at the best cost
        # of the run.  With the later legs' lengths added, the charge is
        # continuous through an arrival and only flying the tour lowers it.
        to_go_tab = self._to_go(goals_b) if arrival else None
        finish = torch.full((B,), float(T), dtype=sysm.dtype, device=sysm.device)
        # The final leg never advances `leg`, so its arrival test keeps firing for
        # every step the vehicle sits inside tol -- paying the bonus per step
        # would make hovering on the goal an unbounded reward.  Credit it once.
        paid_last = torch.zeros(B, dtype=torch.bool, device=sysm.device)
        # `arrived` freezes an episode that has finished the course, exactly as
        # `alive` freezes one that crashed.  Once nothing is still flying the
        # remaining steps cannot change any accumulator, so the loop can leave.
        arrived = torch.zeros(B, dtype=torch.bool, device=sysm.device)
        stop_early = bool(cfg.stop_on_arrival) and arrival
        # consecutive steps spent inside `tol` of the FINAL waypoint
        held = torch.zeros(B, dtype=torch.long, device=sysm.device)
        dwell = max(1, int(round(float(cfg.dwell_s) / dt)))
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

        comp = self.composer
        self._last_obs = {}
        cs = self._composer_start(s, task.goal_for_leg(goals_b, leg) if arrival else task.goal_at(goals_b, 0, T)) if comp is not None else None
        for t in range(T):
            goal = task.goal_for_leg(goals_b, leg) if arrival \
                else task.goal_at(goals_b, t, T)
            spec = None
            if comp is not None:
                # observe BEFORE asking: the composer's decision at this step is
                # made on this step's beams and pixels, never on none at all
                # (the first decision of every episode used to be blind)
                self._last_obs = self._observe(s, sgen, t, alive & ~arrived)
                spec = self._composer_step(cs, s, goal, alive, arrived, leg, t, cost)
            # Episodes whose sensors cannot see anything new, because their
            # state is frozen and will not change again: arrived OR crashed.
            # Both are frozen by the `tree_where` below, and a stateless sensor
            # is a pure function of the state, so its cached reading is the
            # reading a re-march would produce -- for a crashed episode as much
            # as for an arrived one (the sensing tests hold every acceptance
            # number bit-identical across this).  Crashed rows used to be
            # re-marched out of caution about their diverged state; on the
            # co-training rig, where two thirds of a training batch dies, that
            # was most of the camera's time spent looking from a wreck.
            awake = alive & ~arrived  # `live` below is a COST, not a mask
            # with a composer the observation for this step was taken above;
            # taking it twice would advance the sensor buffers twice
            obs = self._last_obs if comp is not None else self._observe(s, sgen, t, awake)
            self._last_obs = obs
            if comp is not None and obs:
                rng = next((v for k, v in obs.items() if k.startswith("range")), None)
                if rng is not None:
                    m = rng.reshape(rng.shape[0], -1).min(-1).values
                    cs["beam_min"] = m if cs["beam_min"] is None else torch.minimum(cs["beam_min"], m)
            u = self._u(TH_b, s, goal, obs, spec)
            s_new = sysm.step(s, u, dt, res_b)
            # crashed vehicles freeze; never integrate a diverged state
            moved = alive & ~arrived
            s = tree_where(moved, s_new, s)
            self._ver = self._ver + moved.long()   # see `_measure`

            err = sysm.task_position(s) - goal
            pos = task.position_cost(sysm.task_position(s), goal,
                                     cfg.pos_eps)
            if to_go_tab is not None:
                to_go = to_go_tab.gather(1, leg.clamp(0, to_go_tab.shape[1] - 1)[:, None]).squeeze(1)
                pos = pos + to_go
            eff = sysm.effort(u, s)
            shp = sysm.shaping_cost(s)
            live = pos + cfg.lambda_e * eff + cfg.lambda_s * shp
            if cfg.lambda_ttc:
                # charged on the SHORTFALL, so a comfortable margin costs
                # nothing and the term only speaks near an obstacle
                ttc = sysm.time_to_collision(s)
                short = (cfg.ttc_safe - ttc).clamp_min(0.0)
                live = live + cfg.lambda_ttc * short * short
            if cfg.lambda_los:
                live = live + cfg.lambda_los * sysm.sight_cost(s, goal)
            if cfg.lambda_occ:
                live = live + cfg.lambda_occ * (1.0 - sysm.visibility(s, goal))
            if res_b is not None:
                live = live + cfg.lambda_r * sysm.residual_penalty(res_b)
            if comp is not None:
                # the low level's own objective: the same charge, with the
                # distance measured to the subgoal the composer PLACED -- the
                # hold's target, not the slewed spec the low level is tracking
                # this step, and not the task goal.  Achieving what it was
                # asked is its job; where to ask for is the composer's.
                pos_sub = task.position_cost(sysm.task_position(s), goal + cs["hold"].target.delta, cfg.pos_eps)
                if to_go_tab is not None:
                    pos_sub = pos_sub + to_go
                live_sub = live - pos + pos_sub
            # in-place: this runs under no_grad, and the accumulators were each
            # allocating a fresh [B] tensor on every one of 250 steps
            # A dead vehicle is frozen at its crash site, so under "frozen" it
            # keeps paying the position term from there and needs no constant of
            # its own.  Effort and shaping are dropped: it is not actuating.
            charge = (pos if frozen_dead
                      else start_err if forfeit_dead else dead)
            step_cost = torch.where(alive, live, charge)
            if comp is not None:
                step_sub = torch.where(alive, live_sub, pos_sub if frozen_dead else charge)
            if stop_early:
                # a finished episode stops accruing; it is done, not hovering
                step_cost = torch.where(arrived, torch.zeros_like(step_cost),
                                        step_cost)
                if comp is not None:
                    step_sub = torch.where(arrived, torch.zeros_like(step_sub), step_sub)
            cost.add_(step_cost, alpha=dt)
            if comp is not None:
                cost_sub.add_(step_sub, alpha=dt)
            sat.add_(sysm.saturation(u, s))
            eff_acc.add_(eff)
            shp_acc.add_(shp)

            if arrival:
                # advance only on ARRIVAL, so reaching a waypoint early buys a
                # longer tail of low cost at the next one -- the incentive that
                # makes the fastest route the cheapest
                reached = (torch.linalg.vector_norm(err, dim=-1) < task.tol) & alive
                last = leg >= last_idx
                # DWELL: the final waypoint has to be HELD, not merely touched.
                # Intermediate waypoints stay pass-through -- a tour flies
                # through them, and asking for a hover at each would be a
                # different task.
                held = torch.where(reached & last, held + 1,
                                   torch.zeros_like(held))
                done = held >= dwell
                finish = torch.where(done & (finish >= T),
                                     torch.full_like(finish, float(t)), finish)
                if cfg.goal_bonus:
                    # An intermediate leg can only be credited once because
                    # reaching it advances `leg`; the last one needs `paid_last`.
                    # The last one is also credited on DWELL, not on contact, or
                    # the bonus pays for exactly the touch-and-go it is meant to
                    # rule out.
                    hit = (reached & ~last) | (done & ~paid_last)
                    credits.add_(hit.to(sysm.dtype))
                    paid_last = paid_last | done
                    # Paid into the stream WHEN it is earned, so the decisions
                    # that earned it see it inside their horizon (as a lump at
                    # the closing token it was invisible to a 2 s return).
                    # The episode total is unchanged: a row that dies later
                    # gives its credits back below.
                    cost.sub_(hit.to(sysm.dtype), alpha=cfg.goal_bonus)
                    if comp is not None:
                        cost_sub.sub_(hit.to(sysm.dtype), alpha=cfg.goal_bonus)
                leg = torch.where(reached & ~last, leg + 1, leg)
                if stop_early:
                    arrived = arrived | done
            elif t in leg_ends:
                leg_err[:, leg_ends.index(t)] = torch.linalg.vector_norm(err, dim=-1)
            was = alive
            alive = alive & sysm.alive(s)
            death_step = torch.where(was & ~alive, torch.full_like(death_step, t), death_step)
            q = float(cfg.stop_quantile)
            if q >= 1.0:
                # exact: wait for every episode to arrive or die
                enough = bool((arrived | ~alive).all())
            else:
                # ARRIVALS ONLY.  Counting crashes toward the quantile lets a
                # policy that crashes a lot reach the threshold sooner and end
                # the batch early -- truncating precisely the survivors who were
                # still flying, which rewards crashing with a shorter episode.
                # Arrivals-only also self-gates: if more than (1 - q) of the
                # batch dies, the threshold is unreachable and the batch runs to
                # full length, so the shortcut applies only once the policy is
                # already good.
                #
                # That self-gating needs the second clause, or it stalls the
                # batch on nothing.  With q = 0.9 and a 12.5% crash rate the
                # arrived fraction tops out at 0.875 and the quantile is never
                # met, so the loop kept stepping physics for hundreds of steps
                # after every episode had already arrived or died.  Measured on
                # `pillars`: 84 s/generation, against 30 s for the same
                # population once the batch is allowed to notice it is finished.
                # The clause is exact rather than an approximation -- with
                # nothing still flying, the hover charge below applies to no
                # episode -- so it changes cost by zero and only saves work.
                enough = bool(arrived.to(sysm.dtype).mean() >= q) or \
                    bool((arrived | ~alive).all())
            fin = float(getattr(cfg, "stop_finished", 0.0) or 0.0)
            if stop_early and fin > 0.0 and not enough:
                # The adaptive cap counts ARRIVALS ONLY, as a fraction of the
                # flights still alive: the batch ends once `fin` of the
                # survivors have arrived.  Crashes never bring the end closer
                # -- counting them let a batch that mostly died end within a
                # couple of seconds and score every survivor still in the air
                # as a failure (batch reach 0.03 against 0.26 judged).
                # ... among the rows the learner will read, when a composer is
                # recording a subset: the other rows fly the mean and arrive
                # sooner, and counting them ended the batch with 30% of the
                # explored survivors still in the air, charged as hovering
                rec = getattr(comp, "record_rows", None) if comp is not None else None
                if rec is not None:
                    of = torch.zeros(B, dtype=torch.bool, device=alive.device); of[rec.to(alive.device)] = True
                    n_alive = (alive & of).to(sysm.dtype).sum(); n_arr = (arrived & of).to(sysm.dtype).sum()
                else:
                    n_alive = alive.to(sysm.dtype).sum(); n_arr = arrived.to(sysm.dtype).sum()
                enough = bool(n_alive > 0) and bool(n_arr >= fin * n_alive)
            if stop_early and enough:
                # every episode has finished or died; the tail is all zeros for
                # the finished ones, and a constant rate for the dead ones, so
                # settle the dead in one go rather than stepping the physics
                # forward for nothing.
                if not frozen_dead and not forfeit_dead:
                    cost.add_((~alive).to(sysm.dtype),
                              alpha=cfg.dead_cost * dt * (T - 1 - t))
                    if comp is not None:
                        cost_sub.add_((~alive).to(sysm.dtype),
                                      alpha=cfg.dead_cost * dt * (T - 1 - t))
                elif frozen_dead or forfeit_dead:
                    tail = start_err if forfeit_dead else pos
                    cost.add_(torch.where(alive, torch.zeros_like(tail), tail),
                              alpha=dt * (T - 1 - t))
                    if comp is not None:
                        tail = start_err if forfeit_dead else pos_sub
                        cost_sub.add_(torch.where(alive, torch.zeros_like(tail), tail),
                                      alpha=dt * (T - 1 - t))
                if q < 1.0:
                    # Anything still flying is charged as if it hovered here for
                    # the rest of the episode.  This is the approximation the
                    # quantile buys its speed with: a vehicle that WOULD have
                    # arrived at step 700 is scored as though it never did, so
                    # the cut falls hardest on policies that are slow because
                    # they are careful.
                    flying = alive & ~arrived
                    cost.add_(torch.where(flying, pos, torch.zeros_like(pos)),
                              alpha=dt * (T - 1 - t))
                    if comp is not None:
                        cost_sub.add_(torch.where(flying, pos_sub, torch.zeros_like(pos_sub)),
                                      alpha=dt * (T - 1 - t))
                break

        self.n_subgoals = cs["n_sub"] if cs is not None else None   # decisions per row
        final_goal = task.goal_for_leg(goals_b, leg) if arrival \
            else task.goal_at(goals_b, T - 1, T)
        # Only survivors keep the bonus: the dead give their credits back.
        # (Crediting a crashed row would make "touch the goal, then crash"
        # score almost as well as completing the task -- fitness would improve
        # while `success`, which requires being alive, fell.)  The credit
        # itself was paid into the stream at the moment it was earned.
        if cfg.goal_bonus:
            cost.add_(credits * (~alive).to(sysm.dtype), alpha=cfg.goal_bonus)
            if comp is not None:
                cost_sub.add_(credits * (~alive).to(sysm.dtype), alpha=cfg.goal_bonus)
        if cs is not None:
            # The SETTLED cost closes the report stream.  The composer's return
            # is read off the stream's cost, and until this token it ended at
            # the last report before the batch stopped -- before the crashed
            # rows' remaining death charge, the still-flying rows' hover charge
            # and the arrival bonus were applied.  Measured on that stream the
            # composer's gradient was consistent (cos 0.85 between halves of a
            # batch) while the judged cost never moved: it was learning that
            # an early crash is cheap and arriving pays nothing.
            tok = self._token(cs["x_int"], sysm.task_position(s), final_goal + cs["hold"].target.delta,
                              cs["beam_min"], alive, arrived,
                              speed=sysm.task_velocity(s).norm(dim=-1))
            tok["t"] = float(T); tok["cost"] = cost.clone()
            self.chain.append(tok)
        done = (finish < T) if arrival else task.success(s, final_goal)
        if arrival:
            leg_err[:, -1] = torch.linalg.vector_norm(
                sysm.task_position(s) - final_goal, dim=-1)
        if comp is None:
            cost_sub = cost                                  # one objective: the task's
        fit = cost.view(P, E).mean(dim=1)
        fit_sub = cost_sub.view(P, E).mean(dim=1)
        if cfg.lambda_crash:
            # Per-GENOME, not per-episode: the point is to price the rate, and a
            # rate is not a property of one episode.  Shrunk toward a prior so
            # the log acts on a belief rather than on a count that is usually 0.
            died = (~alive).view(P, E).to(cost.dtype).sum(dim=1)
            a = cfg.crash_prior * cfg.crash_prior_n
            p_hat = (died + a) / (E + cfg.crash_prior_n)
            fit = fit + cfg.lambda_crash * torch.log(p_hat)
            fit_sub = fit_sub + cfg.lambda_crash * torch.log(p_hat)
        return RolloutResult(
            fitness=fit,
            cost=cost,
            cost_sub=cost_sub,
            fitness_sub=fit_sub,
            death_step=death_step,
            alive=alive,
            leg_err=leg_err,
            final_err=torch.linalg.vector_norm(
                sysm.task_position(s) - final_goal, dim=-1),
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
    def trace(self, TH: Tensor, goals: Tensor, seed: int, freeze_arrivals: bool = False) -> Trace:
        """Same dynamics, but keeps every state.  Separate from `run` so the hot
        path allocates no trace buffers.

        `freeze_arrivals`: hold a flight where it is once it has HELD the final
        goal for `dwell_s`, as `run` does.  Off by default (the trace shows
        what the controller does after arriving); on for renders with a
        composer, which otherwise keeps re-placing a vehicle that has
        finished and walks it away from the goal it reached."""
        sysm, task, cfg = self.system, self.task, self.cfg
        T, dt = cfg.ep_steps, cfg.dt
        s, TH_b, goals_b, res_b, P, E = self._expand(TH, goals, seed)
        self._crn = (E, seed)                 # the composer pairs its token draws by episode across the population

        alive = sysm.alive(s)
        sgen = make_gen(seed + 5_701_889)
        self._prime(s, sgen)
        # the replay has to advance goals exactly as training did, or a trace
        # shows a different flight from the one that was scored
        arrival = getattr(task, "gating", "time") == "arrival"
        leg = torch.zeros(goals_b.shape[0], dtype=torch.long, device=sysm.device)
        states, gs, us, al, lg = [s], [], [], [], []
        comp = self.composer
        self._last_obs = {}
        cs = self._composer_start(s, task.goal_for_leg(goals_b, leg) if arrival else task.goal_at(goals_b, 0, T)) if comp is not None else None
        never = torch.zeros_like(alive)                  # the trace does not freeze arrivals (unless asked)
        arrived = torch.zeros_like(alive); held = torch.zeros(goals_b.shape[0], dtype=torch.long, device=sysm.device)
        dwell = max(1, int(round(float(cfg.dwell_s) / dt))); last_idx = task.n_legs - 1
        no_cost = torch.zeros(goals_b.shape[0], dtype=sysm.dtype, device=sysm.device)
        specs = []
        for t in range(T):
            goal = task.goal_for_leg(goals_b, leg) if arrival \
                else task.goal_at(goals_b, t, T)
            spec = None
            # the same order as `run`: observe, report, decide, act
            obs = self._observe(s, sgen, t); self._last_obs = obs
            if comp is not None:
                spec = self._composer_step(cs, s, goal, alive, arrived if freeze_arrivals else never, leg, t, no_cost)
                specs.append(spec.delta.clone())
                rng = next((v for k, v in obs.items() if k.startswith("range")), None)
                if rng is not None:
                    m = rng.reshape(rng.shape[0], -1).min(-1).values
                    cs["beam_min"] = m if cs["beam_min"] is None else torch.minimum(cs["beam_min"], m)
            u = self._u(TH_b, s, goal, obs, spec)
            s = tree_where(alive & ~arrived, sysm.step(s, u, dt, res_b), s)
            gs.append(goal)
            us.append(u)
            al.append(alive)
            lg.append(leg.clone())
            states.append(s)
            if arrival:
                err = torch.linalg.vector_norm(sysm.task_position(s) - goal, dim=-1)
                reached = (err < task.tol) & alive
                if freeze_arrivals:
                    held = torch.where(reached & (leg >= last_idx), held + 1, torch.zeros_like(held))
                    arrived = arrived | (held >= dwell)
                leg = torch.where(reached & (leg < task.n_legs - 1), leg + 1, leg)
            alive = alive & sysm.alive(s)
        self.last_specs = torch.stack(specs) if specs else None
        self.n_subgoals = cs["n_sub"] if cs is not None else None
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
