"""Sharding the population across processes.

Profiling motivated this: the rollout is already well vectorized -- 512x the
batch costs 3.8x the time, so per-element cost falls 134x and there is no hidden
Python loop over genomes -- but torch's intra-op threads do nothing (1 thread
0.508 s, 10 threads 0.531 s) because the per-step tensors are small and the loop
is dominated by op dispatch.  The parallelism that pays is across the population.
"""
import pytest
import torch

from lagrangian_es.config import Config, ESCfg, RolloutCfg
from lagrangian_es.es import build, train
from lagrangian_es.parallel import ParallelRollout
from lagrangian_es.rollout import Rollout
from lagrangian_es.tasks import make_task
from lagrangian_es.util import make_gen

DT = torch.float64


def _cfg(pop=64):
    return Config(system="quadrotor", trainable="quadrotor_agent",
                  task="waypoint_pair", seed=0,
                  rollout=RolloutCfg(ep_steps=60, n_eps=4),
                  es=ESCfg(pop=pop, gens=3, metric_every=2))


def _pop(tr, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return tr.init()[None] + 0.15 * torch.randn(n, tr.dim, generator=g, dtype=DT)


def test_shards_are_exactly_uniform():
    """The barrier waits on the slowest worker, and per-element cost rises steeply
    as a shard shrinks -- so one oversized shard sets the wall time while the
    undersized ones idle.  An exact split with fewer workers beats a ragged split
    with more."""
    pr = ParallelRollout({}, workers=8)
    for P in (48, 128, 250, 256, 384, 512, 1000, 1024):
        n = pr._shards(P)
        assert n >= 1 and P % n == 0, f"pop {P} splits {n} ways unevenly"
        assert n <= 8
    assert pr._shards(7) in (1, 7)          # prime population


def test_parallel_evaluation_is_bit_identical():
    """Not merely equivalent.  Common random numbers make it exact: the shared
    initial states and the sensor noise are drawn per EPISODE and tiled, so
    neither depends on how many genomes a worker happens to hold."""
    cfg = _cfg()
    s, tr, task = build(cfg)
    task = make_task("waypoint_pair", s, gating="arrival")
    goals = task.sample(cfg.rollout.n_eps, make_gen(0))
    TH = _pop(tr, cfg.es.pop)

    ref = Rollout(s, tr, task, cfg.rollout).run(TH, goals, 7)
    with ParallelRollout({"cfg": cfg}, workers=4) as pr:
        out = pr.run(TH, goals, 7)

    for field in ("fitness", "cost", "alive", "final_err", "success",
                  "legs_done", "finish_frac", "saturation", "effort"):
        a, b = getattr(ref, field), getattr(out, field)
        assert torch.equal(a, b), f"{field} differs between serial and sharded"
    assert out.n_eps == ref.n_eps


def test_small_populations_stay_in_process():
    """Below a few dozen genomes the round trip costs more than the rollout saves."""
    cfg = _cfg(pop=8)
    s, tr, task = build(cfg)
    goals = task.sample(cfg.rollout.n_eps, make_gen(0))
    TH = _pop(tr, 8)
    pr = ParallelRollout({"cfg": cfg}, workers=4, min_pop=32)
    out = pr.run(TH, goals, 3)
    assert pr._pool is None, "spun up a pool for a population too small to benefit"
    ref = Rollout(s, tr, task, cfg.rollout).run(TH, goals, 3)
    assert torch.equal(out.fitness, ref.fitness)


def test_training_through_a_sharded_evaluator_matches():
    """The whole loop, not just one rollout: selection, sigma adaptation and the
    metric all have to see identical numbers."""
    cfg = _cfg()
    a = train(cfg)
    with ParallelRollout({"cfg": cfg}, workers=4) as pr:
        b = train(cfg, evaluator=pr)
    assert torch.equal(a.theta, b.theta), "sharded training diverged"
    for ha, hb in zip(a.history, b.history):
        assert ha["fitness_elite"] == hb["fitness_elite"]
        assert ha["sigma"] == hb["sigma"]


@pytest.mark.parametrize("workers", [2, 3])
def test_result_is_independent_of_worker_count(workers):
    cfg = _cfg(pop=48)
    s, tr, task = build(cfg)
    goals = task.sample(cfg.rollout.n_eps, make_gen(2))
    TH = _pop(tr, 48, seed=5)
    ref = Rollout(s, tr, task, cfg.rollout).run(TH, goals, 11)
    with ParallelRollout({"cfg": cfg}, workers=workers) as pr:
        out = pr.run(TH, goals, 11)
    assert torch.equal(out.fitness, ref.fitness)


def test_default_workers_counts_performance_cores_not_logical_ones():
    """The generation barrier waits on the slowest worker.

    Each worker is pinned to one thread, so one that lands on an efficiency core
    sets the wall time for the whole population.  `cpu_count() - 1` counts both
    kinds; on a 10-logical / 4-performance part it returns 9, which measured
    1.13x slower than 4 -- and 16 workers was 1.64x slower.

    The property, not the number: never more than the machine has, and at least
    one.  The exact value is a machine fact, so it is only compared against the
    performance-core count where that can actually be read.
    """
    import os
    import subprocess
    import sys

    from lagrangian_es.parallel import default_workers

    n = default_workers()
    assert 1 <= n <= (os.cpu_count() or 1)
    if sys.platform == "darwin":
        try:
            got = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                                 capture_output=True, text=True, timeout=2)
            perf = int(got.stdout.strip())
        except Exception:
            pytest.skip("cannot read the performance-core count")
        assert n == perf, (n, perf)
        assert n <= (os.cpu_count() or 1)


def test_a_shard_split_does_not_change_the_answer():
    """Sharding is a compute trick: the fitness must not depend on how the
    population was divided across workers.

    `stop_quantile` below 1.0 breaks this, which is why it is not the default.
    Its exit test asks how many of THIS BATCH have arrived, and the batch is a
    worker's shard -- so a genome scores differently depending on who it was
    packed with.  Measured at 0.8: up to 3.95 of fitness between a 2-worker and
    a 4-worker split.  That is not a bias to trade against speed, it is an
    objective that does not have a single value.
    """
    import json
    import pathlib

    import torch

    from lagrangian_es.config import Config, ESCfg, RolloutCfg
    from lagrangian_es.es import build
    from lagrangian_es.parallel import ParallelRollout

    genome = pathlib.Path(__file__).parent.parent / "assets" / "nav99_genome.json"
    if not genome.exists():
        pytest.skip("prototype genome not present")
    cfg = Config(system="quadrotor_nav", trainable="nav_agent",
                 task="waypoint_pair", environment="pillars", sensors=("range",),
                 gating="arrival", seed=0, system_kw=(("prox_gain", 30.0),),
                 rollout=RolloutCfg(n_eps=8, ep_steps=300, dead_mode="constant",
                                    dead_cost=6.0, goal_bonus=15.0,
                                    stop_on_arrival=True),   # default quantile
                 es=ESCfg(pop=32, gens=1))
    system, tr, task = build(cfg)
    th = torch.tensor(json.loads(genome.read_text())["theta"], dtype=torch.float64)
    g = torch.Generator().manual_seed(7)
    TH = th[None] + 0.05 * torch.randn(32, tr.dim, generator=g, dtype=torch.float64)
    goals = task.sample(8, torch.Generator().manual_seed(11))
    out = []
    for w in (2, 4):
        par = ParallelRollout({"cfg": cfg}, workers=w, min_pop=8)
        try:
            out.append(par.run(TH, goals, 11).fitness)
        finally:
            par.close()
    # NOT bit-equality: a reduction over a shard of 16 associates differently
    # from two over 8, so the last bits of a sum depend on the split.  That is
    # inherent to batched floating point.  What must not happen is a genome's
    # score depending on WHO it was packed with, which is a different order of
    # magnitude entirely -- 3.95 of fitness at `stop_quantile` 0.8, against
    # 1e-12 here.
    assert float((out[0] - out[1]).abs().max()) < 1e-9, \
        float((out[0] - out[1]).abs().max())


def test_a_quantile_stop_makes_fitness_depend_on_the_shard_split():
    """Why `stop_quantile` is not the default, stated as a property.

    Its exit test asks how many of the current BATCH have arrived, and the batch
    is one worker's shard of the population -- so a genome batched with fast
    siblings is truncated earlier than the same genome batched with slow ones,
    and its fitness changes.  Measured at 0.8: up to 3.95 of fitness between a
    2-worker and a 4-worker split, where 1.0 agrees to 1e-12.

    Kept as a test rather than a comment because the setting is still available
    and someone may reach for it: this is the cost, and it is not a bias that
    trades against speed, it is an objective without a single value.
    """
    import json
    import pathlib

    import torch

    from lagrangian_es.config import Config, ESCfg, RolloutCfg
    from lagrangian_es.es import build
    from lagrangian_es.parallel import ParallelRollout

    genome = pathlib.Path(__file__).parent.parent / "assets" / "nav99_genome.json"
    if not genome.exists():
        pytest.skip("prototype genome not present")

    def spread(q):
        cfg = Config(system="quadrotor_nav", trainable="nav_agent",
                     task="waypoint_pair", environment="pillars",
                     sensors=("range",), gating="arrival", seed=0,
                     system_kw=(("prox_gain", 30.0),),
                     rollout=RolloutCfg(n_eps=8, ep_steps=300,
                                        dead_mode="constant", dead_cost=6.0,
                                        goal_bonus=15.0, stop_on_arrival=True,
                                        stop_quantile=q),
                     es=ESCfg(pop=32, gens=1))
        system, tr, task = build(cfg)
        th = torch.tensor(json.loads(genome.read_text())["theta"],
                          dtype=torch.float64)
        g = torch.Generator().manual_seed(7)
        TH = th[None] + 0.05 * torch.randn(32, tr.dim, generator=g,
                                           dtype=torch.float64)
        goals = task.sample(8, torch.Generator().manual_seed(11))
        out = []
        for w in (2, 4):
            par = ParallelRollout({"cfg": cfg}, workers=w, min_pop=8)
            try:
                out.append(par.run(TH, goals, 11).fitness)
            finally:
                par.close()
        return float((out[0] - out[1]).abs().max())

    assert spread(1.0) < 1e-9
    assert spread(0.8) > 0.1, "the coupling this test documents has gone away"
