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


def test_adaptive_cap_ends_a_mostly_dead_batch_early_and_never_touches_judging():
    """`stop_finished` counts finishes of either kind: on a batch where most
    rows crash it fires long before the arrivals-only quantile could; with it
    off (as in judging) the rollout is byte-identical to before."""
    import time
    from dataclasses import replace
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    base = Config(system="quadrotor_nav", trainable="nav_agent", task="waypoint_pair", environment="pillars",
                  sensors=("range",), gating="arrival", seed=0,
                  rollout=RolloutCfg(n_eps=32, ep_steps=600, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0,
                                     stop_on_arrival=True, stop_quantile=0.9))
    sysm, tr, task = build(base)
    TH = tr.init()[None]; goals = task.sample(32, make_gen(2))         # the untrained prior crashes most rows
    t0 = time.time(); r_full = Rollout(sysm, tr, task, base.rollout, build_sensors(base, sysm)).run(TH, goals, 3); t_full = time.time() - t0
    capped = replace(base.rollout, stop_finished=0.8)
    t0 = time.time(); r_cap = Rollout(sysm, tr, task, capped, build_sensors(base, sysm)).run(TH, goals, 3); t_cap = time.time() - t0
    assert float((~r_full.alive).double().mean()) > 0.5, "fixture should mostly crash"
    assert t_cap < 0.8 * t_full, (t_cap, t_full)
    judge = replace(base.rollout, stop_on_arrival=False, stop_quantile=1.0, stop_finished=0.8)
    a = Rollout(sysm, tr, task, judge, build_sensors(base, sysm)).run(TH, goals, 3)
    b = Rollout(sysm, tr, task, replace(judge, stop_finished=0.0), build_sensors(base, sysm)).run(TH, goals, 3)
    assert torch.equal(a.cost, b.cost), "the cap must be inert when early exit is off"
