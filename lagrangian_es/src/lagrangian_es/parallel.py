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
    stochastic, record_frac, shard, difficulty, noise, mute = \
        (tuple(payload[3:]) + (False, 0.0, 0, None, 0, False))[:6] if len(payload) > 3 \
        else (False, 0.0, 0, None, 0, False)
    if difficulty is not None and hasattr(_RIG.system, "difficulty"):
        # a curriculum on the scene: the fraction of obstacles left active this
        # batch, set on the worker's own plant (the pool is forked once)
        _RIG.system.difficulty = float(difficulty)
    comp = getattr(_RIG, "composer", None)
    if comp is not None:
        # THE CONTROL HALF of the paired counterfactual: the same task, the
        # same seed, the same initial state, the composer saying nothing.
        comp.mute = bool(mute)
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
        # `noise` moves ONLY the composer's draws.  The tasks, the initial
        # states and the sensor noise all come from `seed`, so re-flying a
        # batch with the same seed and a different `noise` gives independent
        # policy samples of the SAME tasks -- which is what a per-task baseline
        # needs, and what lets a hard task enter the update as soon as one of
        # its samples arrives.
        torch.manual_seed(int(seed) * 1_000 + int(shard) + int(noise) * 1_000_003)
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
            r.cost_sub, r.fitness_sub, r.death_step, r.soft_time)
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
                     # the continuous vocabulary's extra fields: the sampled
                     # argument, how many slots the token actually used, and
                     # the Gaussian that drew it.  Absent for the token
                     # composer, so this stays None there.
                     # `explore_mu` is the HELD exploration centre: without it
                     # here the parent scores the mixture with the UNIFORM
                     # component while the worker SAMPLED from the held
                     # Gaussian, so the density does not match the behaviour
                     # policy that drew the action.  Exactly the `pi_logits`
                     # failure again -- this list is by NAME and silently drops
                     # anything not in it.
                     **{k: (rc[k][keep] if torch.is_tensor(rc[k]) and rc[k].shape[:1] == r.shape[:1] else rc[k])
                        for k in ("u", "n_args", "mu", "log_std", "x", "goalw", "explore_mu")
                        if k in rc and rc[k] is not None},
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


def _ep_rows(P: int, E: int, lo: int, hi: int) -> Tensor:
    """Global flat rows held by the episode shard `[lo, hi)`.

    The batch is laid out `member * E + episode`, so a POPULATION shard owns a
    contiguous block and an EPISODE shard owns a stride -- one run of `hi - lo`
    rows per member.  Everything the caller gets back is addressed in these
    global rows, which is why the merge scatters instead of concatenating.
    """
    e = torch.arange(lo, hi)
    p = torch.arange(P)
    return (p[:, None] * E + e[None, :]).reshape(-1)


def _merge(parts, order=None) -> RolloutResult:
    parts = [_from_ipc(p) for p in parts]
    if order is None:
        cat = lambda i: torch.cat([_tensor(p[i]) for p in parts], dim=0)
    else:
        # Episode shards interleave, so each part is written to the global rows
        # it owns rather than appended.  The two PER-GENOME fields are the
        # exception: every shard reports the same P genomes, each already
        # averaged over its own (equal-sized) slice of episodes, so their mean
        # is the full-batch mean exactly.
        N = sum(int(o.numel()) for o in order)
        PER_GENOME = (0, 13)                                   # fitness, fitness_sub
        def cat(i):
            vs = [_tensor(p[i]) for p in parts]
            if i in PER_GENOME:
                return torch.stack(vs, 0).mean(0)
            out = torch.empty((N,) + tuple(vs[0].shape[1:]), dtype=vs[0].dtype)
            for o, v in zip(order, vs):
                out[o] = v
            return out
    opt = lambda i: None if any(p[i] is None for p in parts) else cat(i)
    return RolloutResult(
        fitness=cat(0), cost=cat(1), alive=cat(2), leg_err=cat(3),
        final_err=cat(4), success=cat(5), legs_done=cat(6), finish_frac=cat(7),
        saturation=cat(8), effort=cat(9), shaping=cat(10),
        n_eps=parts[0][11] if order is None else sum(p[11] for p in parts),
        cost_sub=opt(12), fitness_sub=opt(13), death_step=opt(14), soft_time=opt(15))


def _records(parts, step: int, n_eps: int, order=None):
    """Per-shard composer records with their rows re-based to the whole batch.
    Shards can end at different intervals (early exit), so their decision
    lists are kept separate rather than concatenated."""
    out = []
    parts = [_from_ipc(p) for p in parts]
    for i, p in enumerate(parts):
        if len(p) > 16 and p[16] is not None:
            rec = dict(p[16])
            rec["rows"] = (rec["rows"] + i * step * n_eps) if order is None else order[i][rec["rows"]]
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
                 min_pop: int = 32, threads: Optional[int] = None, shards: Optional[int] = None,
                 shard_axis: str = "auto"):
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
        # Which axis to split.  The population is the natural one for a GA, but
        # a FROZEN low level is a single genome, so there is nothing to split
        # and the whole batch ran in one process -- 576 flights, one core, the
        # other nine idle.  The EPISODES are independent too, so "auto" splits
        # those whenever the population is too small to fill the workers.
        self.shard_axis = str(shard_axis)
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

    #: an episode shard's slice must keep the recorded-row stride's PHASE.  The
    #: worker keeps every `1/record_frac`-th row of its OWN batch, and the
    #: trainer reads the complement as the policy rows by global index; both
    #: agree as long as each shard's episode count is a multiple of the stride.
    EP_ALIGN = 4

    def _ep_shards(self, E: int) -> int:
        """Largest worker count <= `workers` that splits E exactly and aligned."""
        for n in range(min(self.workers, E), 1, -1):
            if E % n == 0 and (E // n) % self.EP_ALIGN == 0:
                return n
        return 1

    def _plan(self, P: int, E: int, cap: bool) -> tuple:
        """`(axis, n)` -- how this batch is split, and into how many pieces."""
        if self.workers <= 1:
            return "pop", 1
        axis = self.shard_axis
        if axis == "auto":
            axis = "episode" if P < self.workers else "pop"
        if axis == "episode":
            n = self._ep_shards(E)
            return ("episode", n) if n > 1 else ("pop", 1)
        if P < self.min_pop:
            return "pop", 1
        n = self._shards(P)
        if cap and self.shards and n > 1:
            n = max(d for d in range(1, min(P, self.shards) + 1) if P % d == 0)   # the largest exact split up to `shards`
        return "pop", n

    def _chunks(self, TH: Tensor, goals: Tensor, seed: int, axis: str, n: int):
        """`(payloads, order, step)`: the `(TH, goals, seed)` each shard flies,
        the global rows it owns (None when the shards are contiguous), and the
        width of one shard along whichever axis was split."""
        P, E = TH.shape[0], goals.shape[0]
        if axis == "pop":
            step = P // n
            g = _ipc(goals)
            return ([(_ipc(TH[i * step:(i + 1) * step].contiguous()), g, seed) for i in range(n)],
                    None, step)
        step = E // n
        th = _ipc(TH)
        # a DIFFERENT reset seed per shard: `Rollout._expand` draws its initial
        # states from `make_gen(seed)`, so the same seed in every shard would
        # fly the same starts against different goals
        return ([(th, _ipc(goals[i * step:(i + 1) * step].contiguous()), seed + i * 104_729)
                 for i in range(n)],
                [_ep_rows(P, E, i * step, (i + 1) * step) for i in range(n)], step)

    def run(self, TH: Tensor, goals: Tensor, seed: int) -> RolloutResult:
        P, E = TH.shape[0], goals.shape[0]
        axis, n = self._plan(P, E, cap=False)
        if n <= 1:
            return self._local_rig().run(TH, goals, seed)
        chunks, order, _ = self._chunks(TH, goals, seed, axis, n)
        return _merge(self._pool_map(chunks), order)

    def run_with_records(self, TH: Tensor, goals: Tensor, seed: int,
                         stochastic: bool = True, record_frac: float = 0.125, difficulty=None,
                         noise: int = 0, mute: bool = False):
        """`run`, and the composer's recorded decisions from every shard.

        For co-training: the GA ranks every flight from the merged result while
        the composer's update reads a fraction of rows -- each entry carries
        `records`, `chain` and the global `rows` they belong to."""
        P, E = TH.shape[0], goals.shape[0]
        axis, n = self._plan(P, E, cap=True)
        chunks, order, step = self._chunks(TH, goals, seed, axis, max(n, 1))
        chunks = [c + (stochastic, record_frac, i, difficulty, noise, mute) for i, c in enumerate(chunks)]
        parts = self._pool_map(chunks) if n > 1 else [_work_local(self._local_rig(), chunks[0])]
        return _merge(parts, order), _records(parts, step, E, order)

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
