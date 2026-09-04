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
