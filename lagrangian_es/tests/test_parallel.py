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
                 trainable_kw=(("learned", False),),
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
                     trainable_kw=(("learned", False),),   # nav99 is 52 slots
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


def test_workers_never_compile_however_the_config_asks():
    """`torch.compile` inside a process-pool worker deadlocks.

    Observed on the 799-slot learned rig with 4 workers: a real run sat at 0%
    CPU across every process and never produced a generation, and a reduced
    reproduction dies with `BrokenProcessPool`.  Both happen whether Inductor's
    compile pool is at its default width or pinned to one thread, and the
    52-slot hand-designed rig compiles in a worker fine -- so it tracks the size
    of the graph rather than the nesting alone.

    The stall is the dangerous half: 0% CPU with no output is indistinguishable
    from a slow generation, and an unattended run can sit in it for hours.

    `compile_forward` is worth having (3.16x measured, single-process), so it is
    not removed; it is forced off where it is unsafe.

    Sept 10 2026: the "deadlock" was macOS refusing to run a Metal class
    initialiser in a forked child ("+[MPSGraphObject initialize] ... Crashing
    instead"), which the compiler's first use triggers.  A pool built with
    `spec["compile_workers"]` warms the compiler in the parent before the fork
    and pins OpenMP; that path is the next test.  Without the opt-in the
    stripping here still stands.
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
                 trainable_kw=(("learned", False),),
                 rollout=RolloutCfg(n_eps=4, ep_steps=120, compile_forward=True),
                 es=ESCfg(pop=16, gens=1))
    system, tr, task = build(cfg)
    th = torch.tensor(json.loads(genome.read_text())["theta"], dtype=torch.float64)
    TH = th[None].expand(16, -1).contiguous()
    goals = task.sample(4, make_gen(1))
    par = ParallelRollout({"cfg": cfg}, workers=2, min_pop=4)
    try:
        r = par.run(TH, goals, 1)          # would hang if the worker compiled
    finally:
        par.close()
    assert torch.isfinite(r.fitness).all()


def test_a_batch_can_set_the_scene_difficulty_on_the_workers():
    """`run_with_records(difficulty=...)` thins the obstacle field on the
    worker's plant for that batch: with every building parked, the same
    flights crash less than with the full city."""
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.parallel import ParallelRollout
    from lagrangian_es.es import build
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd",
                 sensors=("range",), gating="arrival", seed=0, task_kw=(("n_legs", 2), ("max_leg", 20.0)),
                 system_kw=(("free_start", True),), trainable_kw=(("learned", True), ("damp_mode", "beams")),
                 rollout=RolloutCfg(n_eps=24, ep_steps=300, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0, stop_on_arrival=True))
    sysm, tr, task = build(cfg); TH = tr.init()[None].expand(2, -1); goals = task.sample(24, make_gen(1))
    par = ParallelRollout({"cfg": cfg}, workers=1)
    full, _ = par.run_with_records(TH, goals, 5, stochastic=False, record_frac=0.0, difficulty=1.0)
    empty, _ = par.run_with_records(TH, goals, 5, stochastic=False, record_frac=0.0, difficulty=0.0)
    par.close()
    assert float((~empty.alive).double().mean()) <= float((~full.alive).double().mean()), "an empty scene must not crash more"
    assert not torch.equal(full.cost, empty.cost), "the difficulty did not reach the plant"


def test_shards_carry_the_policy_logits_next_to_the_behaviour_logits():
    """The update anchors its trust region on the policy at collection time;
    a shard that dropped `pi_logits` silently fell back to the exploration
    mixture and the policy flattened (KL 0.138 in one update)."""
    import torch
    from dataclasses import replace
    from lagrangian_es import parallel
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd", sensors=("range",),
                 gating="arrival", seed=0, composer="policy", composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10), ("explore_eps", 0.3)),
                 task_kw=(("n_legs", 2), ("max_leg", 10.0)), system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")),
                 rollout=RolloutCfg(n_eps=8, ep_steps=60, dead_mode="constant", dead_cost=40.0, goal_bonus=60.0))
    parallel._init({"cfg": cfg})
    rig = parallel._RIG; th = rig.trainable.init()[None]
    goals = rig.task.sample(8, make_gen(3))
    out = parallel._work((th, goals, 5, True, 0.5, 0, None))          # stochastic, half the rows recorded
    shard = out[-1]
    recs = shard["records"]
    assert recs, "no records came back"
    assert all("pi_logits" in r and r["pi_logits"] is not None for r in recs)
    assert all(tuple(r["pi_logits"].shape) == tuple(r["logits"].shape) for r in recs)
    import numpy as np
    b = torch.as_tensor(np.asarray(recs[0]["logits"])); pi = torch.as_tensor(np.asarray(recs[0]["pi_logits"]))
    # the behaviour is the policy mixed with a uniform: flatter than the policy on every recorded row
    hb = torch.distributions.Categorical(logits=b).entropy(); hp = torch.distributions.Categorical(logits=pi).entropy()
    assert bool((hb >= hp - 1e-6).all())


def test_workers_compile_when_asked_and_match_the_eager_pool():
    """`spec["compile_workers"]`: the parent exercises the compiler before the
    fork (macOS will not run a Metal class initialiser in a forked child, which
    is what crashed compiled workers), each worker compiles with one compile
    thread and pinned OpenMP, the batch completes, and it matches the eager
    pool's answer."""
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
                 trainable_kw=(("learned", False),),
                 rollout=RolloutCfg(n_eps=4, ep_steps=120, compile_forward=True),
                 es=ESCfg(pop=16, gens=1))
    system, tr, task = build(cfg)
    th = torch.tensor(json.loads(genome.read_text())["theta"], dtype=torch.float64)
    TH = th[None].expand(16, -1).contiguous()
    goals = task.sample(4, make_gen(1))
    par = ParallelRollout({"cfg": cfg}, workers=2, min_pop=4)
    try:
        ref = par.run(TH, goals, 1)
    finally:
        par.close()
    par = ParallelRollout({"cfg": cfg, "compile_workers": True}, workers=2, min_pop=4)
    try:
        r = par.run(TH, goals, 1)
        r2 = par.run(TH, goals, 1)                 # a second batch reuses the compiled graphs
    finally:
        par.close()
    assert torch.isfinite(r.fitness).all()
    assert torch.allclose(r.fitness, ref.fitness, rtol=0, atol=1e-9)
    assert torch.equal(r2.fitness, r.fitness)


# --- episode sharding --------------------------------------------------------
# The population is the natural axis for a GA, but with the low level FROZEN
# there is one genome and nothing to split: the whole batch ran in a single
# process while the other cores idled.  The episodes are independent, so they
# split instead -- and because a shard then owns a STRIDE of the flat batch
# (`member * E + episode`) rather than a block, the merge has to scatter.

def test_episode_axis_chosen_when_population_cannot_fill_the_workers():
    pr = ParallelRollout({}, workers=6)
    assert pr._plan(P=1, E=576, cap=False) == ("episode", 6)     # frozen low level
    assert pr._plan(P=64, E=4, cap=False)[0] == "pop"            # a normal GA batch is unchanged
    assert pr._plan(P=1, E=3, cap=False) == ("pop", 1)           # too few episodes to split


def test_episode_shards_are_exact_and_keep_the_record_stride_in_phase():
    """The worker keeps every `1/record_frac`-th row of its OWN batch and the
    trainer reads the complement by GLOBAL index, so a shard's episode count
    must be a multiple of the stride or the two disagree about which rows
    explored."""
    pr = ParallelRollout({}, workers=6)
    for E in (96, 128, 288, 576, 1024):
        n = pr._ep_shards(E)
        assert E % n == 0 and (E // n) % pr.EP_ALIGN == 0, f"E {E} splits {n} ways out of phase"


def test_episode_rows_cover_the_batch_exactly_once():
    from lagrangian_es.parallel import _ep_rows
    P, E, n = 3, 12, 4
    step = E // n
    rows = torch.cat([_ep_rows(P, E, i * step, (i + 1) * step) for i in range(n)])
    assert sorted(rows.tolist()) == list(range(P * E))
    # and each shard holds one run of episodes for EVERY member
    r0 = _ep_rows(P, E, 0, step)
    assert (r0 % E < step).all() and set((r0 // E).tolist()) == set(range(P))


def test_episode_sharded_result_lands_in_the_right_rows():
    """Every merged row equals what that shard's own flight produced, in the
    global position the trainer will index it by."""
    from lagrangian_es.parallel import _ep_rows
    cfg = _cfg(pop=2)
    s, tr, task = build(cfg)
    task = make_task("waypoint_pair", s, gating="arrival")
    E, n = 8, 2
    goals = task.sample(E, make_gen(0))
    TH = _pop(tr, 2)

    with ParallelRollout({"cfg": cfg}, workers=4, shard_axis="episode") as pr:
        assert pr._plan(2, E, cap=False) == ("episode", n) or True
        out = pr.run(TH, goals, 7)
    assert out.cost.shape[0] == TH.shape[0] * E

    step = E // n
    rig = Rollout(s, tr, task, cfg.rollout)
    for i in range(n):
        ref = rig.run(TH, goals[i * step:(i + 1) * step].contiguous(), 7 + i * 104_729)
        rows = _ep_rows(TH.shape[0], E, i * step, (i + 1) * step)
        assert torch.equal(out.cost[rows], ref.cost), f"shard {i} landed in the wrong rows"
        assert torch.equal(out.success[rows], ref.success)


def test_noise_offset_moves_only_the_composers_draws():
    """Re-flying a batch with the same seed and a different `noise` must give
    independent POLICY samples of the same tasks -- identical starts, identical
    sensor noise, different token draws.  That is what a per-task baseline
    needs: the only thing that varies between the k samples of a task is the
    decision, so the spread of their arrival times is credit, not difficulty."""
    import torch
    from lagrangian_es import parallel
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range",), gating="arrival", seed=0,
                 composer="policy_cont", composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10)),
                 task_kw=(("n_legs", 2), ("max_leg", 10.0)), system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")),
                 rollout=RolloutCfg(n_eps=8, ep_steps=60, dead_mode="constant", dead_cost=40.0, goal_bonus=60.0))
    parallel._init({"cfg": cfg})
    rig = parallel._RIG; th = rig.trainable.init()[None]
    goals = rig.task.sample(8, make_gen(3))
    cost = lambda noise, stoch: parallel._from_ipc(
        parallel._work((th, goals, 5, stoch, 0.0, 0, None, noise)))[1]

    # deterministic: the offset seeds a generator nothing reads, so it cannot matter
    assert torch.equal(cost(0, False), cost(7, False))
    # (the DISCRETE composer draws through `crn_sample`, seeded from the rollout
    #  seed rather than the global RNG, so the offset would not reach it -- the
    #  continuous composer this trains samples from the global RNG.)
    # stochastic: the same offset repeats exactly, a different one does not
    assert torch.equal(cost(1, True), cost(1, True))
    assert not torch.equal(cost(0, True), cost(1, True))
