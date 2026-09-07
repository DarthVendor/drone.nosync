"""Evaluate a population across processes.

Profiling says the rollout is already properly vectorized -- 512x the batch costs
3.8x the time, so per-element cost falls 134x and there is no hidden Python loop
over genomes.  What it is NOT is parallel across cores: torch's intra-op threads
do nothing here (1 thread 0.508 s, 10 threads 0.531 s) because the per-step
tensors are small and the loop is dominated by op dispatch, of which there are
~250 steps x tens of ops per rollout.

So the parallelism that pays is across the POPULATION, which is embarrassingly
parallel: genomes share the goals and the reset noise but never interact within a
generation.

**This is bit-identical to single-process evaluation, not merely equivalent.**
Common random numbers make it so: `Rollout._expand` draws the E shared initial
states from `make_gen(seed)` and tiles them, and sensor noise is drawn per episode
and tiled the same way -- neither depends on how many genomes a worker happens to
hold.  `test_parallel.py` asserts the equality rather than trusting the argument.
"""
from __future__ import annotations

import os
import subprocess
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import torch
from torch import Tensor

from .rollout import Rollout, RolloutResult

_RIG = None          # per-worker Rollout, built once by the initializer


def _init(spec: dict) -> None:
    global _RIG
    torch.set_num_threads(1)          # workers must not fight each other for cores
    from .es import build, build_sensors

    cfg = spec["cfg"]
    system, trainable, task = build(cfg)
    if spec.get("terms_fn") is not None:
        from .trainables import make_trainable
        trainable = make_trainable(cfg.trainable, system,
                                   terms=spec["terms_fn"](system))
    sensors = spec["sensors_fn"](system) if spec.get("sensors_fn") \
        else build_sensors(cfg, system)
    # NEVER compile inside a worker.  TorchInductor runs its own pool of compile
    # processes, and starting one from inside a `ProcessPoolExecutor` worker
    # deadlocks: measured, the whole run sat at 0% CPU across every process and
    # made no progress at all -- which looks like a slow generation rather than a
    # hang, and is the worst possible failure mode for an unattended run.
    #
    # `compile_forward` measured a genuine 3.16x, but only in a single process.
    # It stays available for that case; here it is forced off rather than left
    # to whoever writes the config.
    rc = cfg.rollout
    if getattr(rc, "compile_forward", False):
        from dataclasses import replace
        rc = replace(rc, compile_forward=False)
    _RIG = Rollout(system, trainable, task, rc, sensors)


def _work(payload):
    TH, goals, seed = payload
    r = _RIG.run(TH, goals, seed)
    return (r.fitness, r.cost, r.alive, r.leg_err, r.final_err, r.success,
            r.legs_done, r.finish_frac, r.saturation, r.effort, r.shaping, r.n_eps)


def _merge(parts) -> RolloutResult:
    cat = lambda i: torch.cat([p[i] for p in parts], dim=0)
    return RolloutResult(
        fitness=cat(0), cost=cat(1), alive=cat(2), leg_err=cat(3),
        final_err=cat(4), success=cat(5), legs_done=cat(6), finish_frac=cat(7),
        saturation=cat(8), effort=cat(9), shaping=cat(10), n_eps=parts[0][11])


def default_workers() -> int:
    """How many worker processes to run: PERFORMANCE cores, not logical ones.

    Workers are pinned to one thread each and the generation barrier waits on
    the slowest, so a worker that lands on an efficiency core sets the wall
    time for everyone.  `cpu_count() - 1` counts both kinds and oversubscribes
    badly on an asymmetric machine.

    Measured on a 10-logical / 4-performance Apple Silicon part, one generation
    of 48 genomes x 32 episodes x 1200 steps:

        2 workers 1.015s    4 workers 0.848s    8 workers 0.957s
        3 workers 0.885s    6 workers 0.925s   16 workers 1.387s

    The optimum is exactly the performance-core count, and the old default of 9
    was 1.13x slower than it.  Falls back to `cpu_count() - 1` where the split
    cannot be read, which is the honest answer for a symmetric machine.
    """
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                                 capture_output=True, text=True, timeout=2)
            n = int(out.stdout.strip())
            if n > 0:
                return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) - 1)


class ParallelRollout:
    """Drop-in replacement for `Rollout.run` that shards the population.

    Falls back to in-process evaluation when the population is too small to be
    worth the round trip -- below a few dozen genomes the pickling costs more
    than the rollout saves.
    """

    def __init__(self, spec: dict, workers: Optional[int] = None,
                 min_pop: int = 32):
        self.spec = spec
        self.workers = int(workers or default_workers())
        self.min_pop = int(min_pop)
        self._pool: Optional[ProcessPoolExecutor] = None
        self._local: Optional[Rollout] = None

    def _pool_up(self):
        if self._pool is None:
            # `fork` rather than the macOS default `spawn`: spawn re-imports the
            # caller's __main__, so any script without an `if __name__` guard
            # re-runs itself in every worker and the pool dies on the spot.  Fork
            # inherits the already-imported modules, needs no guard, and starts
            # far faster.  Workers are pinned to one thread each, and the parent
            # is single-threaded here anyway, so there is nothing to fork unsafely.
            torch.set_num_threads(1)
            try:
                ctx = mp.get_context("fork")
            except ValueError:                      # platform without fork
                ctx = mp.get_context("spawn")
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=ctx,
                initializer=_init, initargs=(self.spec,))
        return self._pool

    def _local_rig(self):
        if self._local is None:
            _init(self.spec)
            self._local = _RIG
        return self._local

    def run(self, TH: Tensor, goals: Tensor, seed: int) -> RolloutResult:
        P = TH.shape[0]
        if self.workers <= 1 or P < self.min_pop:
            return self._local_rig().run(TH, goals, seed)
        n = self._shards(P)
        step = P // n
        chunks = [(TH[i * step:(i + 1) * step].contiguous(), goals, seed)
                  for i in range(n)]
        return _merge(list(self._pool_up().map(_work, chunks)))

    def _shards(self, P: int) -> int:
        """Largest worker count <= `workers` that divides P EXACTLY.

        Uniform shards matter more here than using every core.  The barrier waits
        on the slowest worker, and per-element cost rises steeply as a shard
        shrinks (105 us/element at batch 8, 1.0 us at batch 2048), so one oversized
        shard sets the wall time while the undersized ones sit idle -- and one
        *undersized* shard is disproportionately inefficient on its own.  An exact
        split with fewer workers beats a ragged split with more.
        """
        for n in range(min(self.workers, P), 1, -1):
            if P % n == 0:
                return n
        return 1

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
