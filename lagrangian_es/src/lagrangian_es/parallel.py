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
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .rollout import Rollout, RolloutResult

_RIG = None          # per-worker Rollout, built once by the initializer


def _ipc(x):
    """Convert tensors to plain ndarray payloads before crossing a process pipe.

    PyTorch's multiprocessing reducer sends CPU tensors through shared-memory
    storage and may start `torch_shm_manager`.  The co-training record stream is
    many small tensors, so plain pickle copies are more predictable and avoid
    taking the whole pool down when the manager cannot be spawned.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _ipc(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_ipc(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_ipc(v) for v in x)
    return x


def _tensor(x):
    return torch.as_tensor(x) if isinstance(x, np.ndarray) else x


def _from_ipc(x):
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x)
    if isinstance(x, dict):
        return {k: _from_ipc(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_from_ipc(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_from_ipc(v) for v in x)
    return x


def _init(spec: dict) -> None:
    # `threads`: a composer in the loop is matmul-bound, and four single-
    # threaded workers leave the efficiency cores idle
    # (an unconditional reset to 1 used to follow this line, so every worker
    # ran single-threaded whatever the setting; measured ~97% CPU a worker)
    torch.set_num_threads(int(spec.get("threads", 1)))
    global _RIG
    from .es import build, build_sensors

    cfg = spec["cfg"]
    system, trainable, task = build(cfg)
    if spec.get("terms_fn") is not None:
        from .trainables import make_trainable
        trainable = make_trainable(cfg.trainable, system,
                                   terms=spec["terms_fn"](system))
    sensors = spec["sensors_fn"](system) if spec.get("sensors_fn") \
        else build_sensors(cfg, system)
    # NEVER compile inside a worker.
    #
    # Observed twice, on the 799-slot learned rig with 4 workers: a real run sat
    # at 0% CPU across every process and never produced a generation, and a
    # reduced reproduction dies with `BrokenProcessPool` -- a worker terminated
    # while the pool waited on it.  Both happen with Inductor's compile pool at
    # its default width AND pinned to one thread, so the nested pool is not the
    # whole story; the 52-slot hand-designed rig compiles in a worker quite
    # happily, which points at the size of the graph each worker has to build.
    #
    # The first failure mode is the dangerous one: 0% CPU and no output is
    # indistinguishable from a slow generation, and an unattended run can sit in
    # it for hours.
    #
    # `compile_forward` measured a genuine 3.16x SINGLE-PROCESS and stays
    # available for that.  Here it is forced off rather than left to whoever
    # writes the config, because the failure is silent.
    #
    # Sept 10 2026: that hang looked exactly like the swap thrash seen the same
    # day (load 40-100 with every worker at 0-30% CPU, 64 MB free, 5 GB in the
    # compressor).  Inductor's default compile pool is `compile_threads` = 10
    # SUBPROCESSES per worker, each a full torch import: eight workers spawn
    # eighty of them, and this 16 GB machine goes to swap.  So compilation in a
    # worker is opt-in (`spec["compile_workers"]`) and runs with ONE compile
    # thread, in-process, no pool; the shard batch is one fixed shape, so each
    # worker compiles once per shard size and the on-disk FX graph cache lets
    # later workers and relaunches skip the code generation.
    rc = cfg.rollout
    if getattr(rc, "compile_forward", False):
        if spec.get("compile_workers"):
            import torch._inductor.config as _ic
            _ic.compile_threads = 1
        else:
            from dataclasses import replace
            rc = replace(rc, compile_forward=False)
    from .es import build_composer
    # the task-level layer, if the config names one: a worker that flew
    # without it would rank a different controller from the one the parent
    # evaluates, and the mismatch would be silent
    _RIG = Rollout(system, trainable, task, rc, sensors,
                   composer=build_composer(cfg, system, trainable))


def _work(payload):
    TH, goals, seed = payload[:3]
    TH, goals = _tensor(TH), _tensor(goals)
    stochastic, record_frac, shard, difficulty = (payload[3:] + (False, 0.0, 0, None))[:4] if len(payload) > 3 \
        else (False, 0.0, 0, None)
    if difficulty is not None and hasattr(_RIG.system, "difficulty"):
        # a curriculum on the scene: the fraction of obstacles left active this
        # batch, set on the worker's own plant (the pool is forked once)
        _RIG.system.difficulty = float(difficulty)
    comp = getattr(_RIG, "composer", None)
    if comp is not None and hasattr(comp, "stochastic"):
        # Reload the composer's weights if the file changed since this worker
        # last read it.  The pool is forked ONCE, before the parent does any
        # multithreaded work: re-forking each iteration from a parent with a
        # live thread pool is how a worker "terminates abruptly" on the second
        # iteration, every time.
        _maybe_reload(comp)
        # exploration noise is seeded per shard, so a batch is reproducible
        # and two shards never draw the same noise
        comp.stochastic = bool(stochastic); comp.records = []
        torch.manual_seed(int(seed) * 1_000 + int(shard))
        # record only the rows that will be kept, from the first decision on
        B_rows = TH.shape[0] * goals.shape[0]
        comp.record_rows = (torch.arange(0, B_rows, max(1, int(round(1.0 / record_frac))))
                            if (stochastic and record_frac > 0) else None)
    _t0, _c0 = time.perf_counter(), time.process_time()
    r = _RIG.run(TH, goals, seed)
    if os.environ.get("LES_WORK_TIMING"):
        print(f"[work pid {os.getpid()} shard {shard} rows {TH.shape[0] * goals.shape[0]}: wall {time.perf_counter() - _t0:.1f}s cpu {time.process_time() - _c0:.1f}s]",
              file=sys.stderr, flush=True)
    base = (r.fitness, r.cost, r.alive, r.leg_err, r.final_err, r.success,
            r.legs_done, r.finish_frac, r.saturation, r.effort, r.shaping, r.n_eps,
            r.cost_sub, r.fitness_sub, r.death_step)
    if comp is None or record_frac <= 0 or not getattr(comp, "records", None):
        return _ipc(base + (None,))
    # Slim the composer's records to a fraction of rows, in float32: the
    # tokens are the bulk of what crosses the process boundary, and the
    # composer's update needs far fewer flights than the GA's ranking does.
    B = r.cost.shape[0]
    every = max(1, int(round(1.0 / record_frac)))
    sel = torch.arange(0, B, every)                       # full-batch rows kept for the update
    pos_of = torch.full((B,), -1, dtype=torch.long); pos_of[sel] = torch.arange(sel.numel())
    recs = []
    for rc in comp.records:
        # a record holds only the rows that were flying at that decision;
        # keep those among `sel`, and address them by POSITION within `sel`,
        # which is how the returns over the kept rows are indexed
        r = rc.get("rows", torch.arange(rc["act"].shape[0]))
        keep = pos_of[r] >= 0
        tk = rc.get("tok_keep")                            # the rows whose scene tokens were kept (see PolicyComposer.tok_frac)
        keep_t = keep if tk is None else keep[tk]          # the same filter, over the token rows
        recs.append({"t": rc["t"], "act": rc["act"][keep], "alive": rc["alive"][keep],
                     "rows": pos_of[r[keep]], "moved": rc["moved"][keep] if "moved" in rc else None,
                     "logits": rc["logits"][keep] if "logits" in rc else None,
                     "pi_logits": rc["pi_logits"][keep] if "pi_logits" in rc else None,   # the policy at collection: the update's trust region
                     "tok_keep": None if tk is None else tk[keep],
                     "tok": {k: (v[keep_t].float() if torch.is_tensor(v) and v.is_floating_point()
                                 else (v[keep_t] if torch.is_tensor(v) else v)) for k, v in rc["tok"].items()}})
    chain = [{k: (v[sel] if torch.is_tensor(v) else v) for k, v in tok.items()} for tok in _RIG.chain]
    comp.records = []
    n_sub = getattr(_RIG, "n_subgoals", None)
    return _ipc(base + ({"records": recs, "chain": chain, "rows": sel,
                         "n_sub": None if n_sub is None else float(n_sub.to(torch.float64).mean())},))


def _maybe_reload(comp) -> bool:
    """True if the composer's weights file changed and was reloaded."""
    import os
    path = getattr(comp, "weights_path", "")
    if not path or not os.path.exists(path):
        return False
    m = os.path.getmtime(path)
    if m == getattr(comp, "_weights_mtime", None):
        return False
    comp.net.load_state_dict(torch.load(path, map_location="cpu"))
    comp.net.eval(); comp._weights_mtime = m
    return True


def _work_local(rig, payload):
    """`_work` against a given rig, for the single-process fallback."""
    global _RIG
    saved = globals().get("_RIG"); _RIG = rig
    try:
        return _work(payload)
    finally:
        _RIG = saved


def _merge(parts) -> RolloutResult:
    parts = [_from_ipc(p) for p in parts]
    cat = lambda i: torch.cat([_tensor(p[i]) for p in parts], dim=0)
    opt = lambda i: None if any(p[i] is None for p in parts) else cat(i)
    return RolloutResult(
        fitness=cat(0), cost=cat(1), alive=cat(2), leg_err=cat(3),
        final_err=cat(4), success=cat(5), legs_done=cat(6), finish_frac=cat(7),
        saturation=cat(8), effort=cat(9), shaping=cat(10), n_eps=parts[0][11],
        cost_sub=opt(12), fitness_sub=opt(13), death_step=opt(14))


def _records(parts, step: int, n_eps: int):
    """Per-shard composer records with their rows re-based to the whole batch.
    Shards can end at different intervals (early exit), so their decision
    lists are kept separate rather than concatenated."""
    out = []
    parts = [_from_ipc(p) for p in parts]
    for i, p in enumerate(parts):
        if len(p) > 15 and p[15] is not None:
            rec = dict(p[15]); rec["rows"] = rec["rows"] + i * step * n_eps
            out.append(rec)
    return out


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
                 min_pop: int = 32, threads: Optional[int] = None, shards: Optional[int] = None):
        self.spec = dict(spec)
        if threads is None:
            # With a composer in the loop each worker is matmul-bound (the
            # transformer at every decision), so two threads per worker use the
            # efficiency cores the four single-threaded workers left idle.  A
            # pure ES run is physics-bound and keeps the one thread its
            # worker count was tuned for.
            cfg = spec.get("cfg")
            threads = 2 if (cfg is not None and getattr(cfg, "composer", "")) else 1
        self.spec["threads"] = int(threads)
        self.workers = int(workers or default_workers())
        self.min_pop = int(min_pop)
        # More shards than workers, dispatched one at a time: on a machine with
        # performance AND efficiency cores (this Mac: 4 + 6) equal static shards
        # wait on the slowest core, and eight workers bought nothing over four.
        # With P shards a fast worker takes several while a slow one takes one.
        self.shards = None if shards is None else int(shards)
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
            if self.spec.get("compile_workers"):
                # Compiling in a forked child crashed the child outright on
                # macOS (Sept 10 2026): "+[MPSGraphObject initialize] may have
                # been in progress in another thread when fork() was called ...
                # Crashing instead" -- the compiler's first use touches Metal's
                # graph classes, and Objective-C refuses to run a class
                # initializer in a forked child.  So the compiler is exercised
                # once HERE, before the fork, with one compile thread (no
                # subprocess pool to inherit); the children find it initialised.
                import os
                import torch._inductor.config as _ic
                _ic.compile_threads = 1
                # The generated kernels' OpenMP regions ignored `torch.set_num_threads(1)`
                # in a forked child (libomp re-initialises after fork): two compiled
                # workers ran 2x SLOWER than eager until OMP_NUM_THREADS pinned them
                # (Sept 10 2026: 27-32 s -> 12 s per batch).  Inherited by the children.
                os.environ.setdefault("OMP_NUM_THREADS", str(int(self.spec.get("threads", 1))))
                if self.spec.get("compile_warmup", True):
                    torch.compile(lambda x: x * 2.0 + 1.0, dynamic=False)(torch.ones(4))
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

    def _pool_map(self, chunks):
        try:
            return list(self._pool_up().map(_work, chunks))
        except BrokenProcessPool:
            print("ParallelRollout: worker pool broke; rerunning this batch in-process",
                  file=sys.stderr, flush=True)
            self.close()
            rig = self._local_rig()
            return [_work_local(rig, c) for c in chunks]

    def run(self, TH: Tensor, goals: Tensor, seed: int) -> RolloutResult:
        P = TH.shape[0]
        if self.workers <= 1 or P < self.min_pop:
            return self._local_rig().run(TH, goals, seed)
        n = self._shards(P)
        step = P // n
        goals_ipc = _ipc(goals)
        chunks = [(_ipc(TH[i * step:(i + 1) * step].contiguous()), goals_ipc, seed)
                  for i in range(n)]
        return _merge(self._pool_map(chunks))

    def run_with_records(self, TH: Tensor, goals: Tensor, seed: int,
                         stochastic: bool = True, record_frac: float = 0.125, difficulty=None):
        """`run`, and the composer's recorded decisions from every shard.

        For co-training: the GA ranks every flight from the merged result while
        the composer's update reads a fraction of rows -- each entry carries
        `records`, `chain` and the global `rows` they belong to."""
        P = TH.shape[0]
        n = self._shards(P) if (self.workers > 1 and P >= self.min_pop) else 1
        if self.shards and n > 1:
            n = max(d for d in range(1, min(P, self.shards) + 1) if P % d == 0)     # the largest exact split up to `shards`
        step = P // n
        goals_ipc = _ipc(goals)
        chunks = [(_ipc(TH[i * step:(i + 1) * step].contiguous()), goals_ipc, seed, stochastic, record_frac, i, difficulty)
                  for i in range(n)]
        parts = self._pool_map(chunks) if n > 1 else [_work_local(self._local_rig(), chunks[0])]
        return _merge(parts), _records(parts, step, goals.shape[0])

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
