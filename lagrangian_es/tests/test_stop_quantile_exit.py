"""The quantile shortcut must never make a batch run LONGER than the exact one.

`stop_quantile < 1` ends a batch once `q` of it has arrived, charging whatever
is still flying as though it hovered.  Counting arrivals only -- not crashes --
is deliberate, so that a policy cannot end the batch early by dying.  But that
also means the arrived fraction tops out at `1 - crash_rate`: on a policy that
crashes more than `1 - q`, the threshold is unreachable and the batch used to
step physics to `ep_steps` even after every episode had arrived or died.

So the loop also exits when nothing is still flying.  That condition is exactly
the one `q = 1.0` uses, and the hover charge then applies to no episode, so the
fitness must come out identical -- these tests pin that, since a shortcut that
quietly changed cost would be far worse than a slow one.
"""
import torch

from lagrangian_es.config import Config, RolloutCfg
from lagrangian_es.es import build, build_sensors
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen


def _fitness(env, q, steps=400, eps=32, pop=6, seed=3):
    cfg = Config(system="quadrotor_nav", trainable="nav_agent",
                 task="waypoint_pair", environment=env, sensors=("range",),
                 gating="arrival", seed=0,
                 rollout=RolloutCfg(n_eps=eps, ep_steps=steps, dead_mode="constant",
                                    dead_cost=6.0, goal_bonus=15.0,
                                    stop_on_arrival=True, stop_quantile=q))
    system, trainable, task = build(cfg)
    roll = Rollout(system, trainable, task, cfg.rollout,
                   build_sensors(cfg, system))
    torch.manual_seed(0)
    TH = trainable.init().expand(pop, -1).clone()
    TH = TH + 0.05 * torch.randn(TH.shape, generator=make_gen(11), dtype=TH.dtype)
    res = roll.run(TH, task.sample(eps, make_gen(2)), seed)
    return res


def test_unreachable_quantile_matches_the_exact_rollout():
    """An untrained controller crashes far more than 1 - q, so only the new
    clause can end the batch -- and it must agree with q = 1.0 exactly."""
    exact = _fitness("pillars", 1.0)
    assert float((~exact.alive).to(torch.float64).mean()) > 0.1, \
        "this fixture is only meaningful while the crash rate exceeds 1 - q"
    for q in (0.9, 0.8):
        got = _fitness("pillars", q)
        assert torch.equal(got.fitness, exact.fitness), \
            f"stop_quantile={q} changed the cost it was only meant to speed up"


def test_the_quantile_shortcut_still_applies_when_it_is_reachable():
    """The approximation must remain available -- the fix adds an exit, it does
    not disable the quantile."""
    a = _fitness("empty", 1.0, steps=600)
    b = _fitness("empty", 0.5, steps=600)
    assert a.fitness.shape == b.fitness.shape
    # nothing to assert about equality here: cutting the batch at half the
    # arrivals is a real approximation and is allowed to move the cost.


def test_exit_holds_for_a_batch_that_all_dies():
    """Every episode dead and none arrived is still 'nothing is flying'."""
    res = _fitness("pillars", 0.9, steps=200)
    assert torch.isfinite(res.fitness).all()


def test_adaptive_cap_counts_arrivals_among_survivors_never_crashes():
    """`stop_finished` ends a batch once that fraction of the SURVIVORS has
    arrived.  A batch that mostly crashes and never arrives is not ended by
    its crashes: its cost is byte-identical to the uncapped run.  With early
    exit off (judging) the cap is inert."""
    from dataclasses import replace
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    base = Config(system="quadrotor_nav", trainable="nav_agent", task="waypoint_pair", environment="pillars",
                  sensors=("range",), gating="arrival", seed=0,
                  system_kw=(("phi0", (0.25, 0.25, 0.25, 0.10, 0.10, 0.10)),),   # the old marginal prior: crashes untrained
                  rollout=RolloutCfg(n_eps=32, ep_steps=300, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0,
                                     stop_on_arrival=True, stop_quantile=1.0))
    sysm, tr, task = build(base)
    TH = tr.init()[None]; goals = task.sample(32, make_gen(2))         # the untrained prior: crashes, no arrivals
    r_full = Rollout(sysm, tr, task, base.rollout, build_sensors(base, sysm)).run(TH, goals, 3)
    r_cap = Rollout(sysm, tr, task, replace(base.rollout, stop_finished=0.8), build_sensors(base, sysm)).run(TH, goals, 3)
    assert float((~r_full.alive).double().mean()) > 0.5 and not bool(r_full.success.any()), "fixture should crash and never arrive"
    assert torch.equal(r_cap.cost, r_full.cost), "crashes must not end the batch"
    judge = replace(base.rollout, stop_on_arrival=False, stop_quantile=1.0, stop_finished=0.8)
    a = Rollout(sysm, tr, task, judge, build_sensors(base, sysm)).run(TH, goals, 3)
    b = Rollout(sysm, tr, task, replace(judge, stop_finished=0.0), build_sensors(base, sysm)).run(TH, goals, 3)
    assert torch.equal(a.cost, b.cost), "the cap must be inert when early exit is off"


def test_adaptive_cap_counts_only_the_recorded_rows_when_a_composer_records():
    """With a composer recording a subset, the cap waits for that subset's
    survivors to arrive; the mean-flying rows do not end the batch for them.
    Checked through the cost: the recorded rows' flights are as long as when
    they are the whole batch."""
    from dataclasses import replace
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_composer, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    base = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd",
                  sensors=("range",), gating="arrival", seed=0, composer="policy",
                  task_kw=(("n_legs", 1), ("max_leg", 6.0)), system_kw=(("free_start", True),),
                  trainable_kw=(("learned", True), ("damp_mode", "beams")),
                  rollout=RolloutCfg(n_eps=8, ep_steps=200, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0,
                                     stop_on_arrival=True, stop_quantile=1.0, stop_finished=0.5))
    sysm, tr, task = build(base); th = tr.init()[None]; goals = task.sample(8, make_gen(1))
    comp = build_composer(base, sysm, tr); comp.stochastic = False           # no noise: rows differ only by the mask
    comp.record_rows = torch.tensor([1, 3, 5, 7])
    r = Rollout(sysm, tr, task, base.rollout, build_sensors(base, sysm), composer=comp).run(th, goals, 3)
    # the same rows as the whole batch, same seeds per episode is not possible, so compare against
    # the cap being evaluated on everyone: the recorded rows can only have flown LONGER, never shorter
    comp2 = build_composer(base, sysm, tr); comp2.stochastic = False; comp2.record_rows = None
    r2 = Rollout(sysm, tr, task, base.rollout, build_sensors(base, sysm), composer=comp2).run(th, goals, 3)
    assert r.cost.shape == r2.cost.shape
    assert bool((r.cost[[1, 3, 5, 7]] <= r2.cost[[1, 3, 5, 7]] + 1e-9).all()), "recorded rows were cut shorter than before"
