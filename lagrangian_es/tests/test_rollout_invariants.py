"""Invariants that must hold over any rollout, whatever the policy does.

Most of this session's real bugs were not wrong arithmetic in a function but
quantities that disagreed with each other: `finish_frac` reporting 1.0 for a
flight that was still airborne, a weight that silently went uniform, a genome
flown under a config it was not evolved with.  These pin the relationships
between what a rollout reports, so a future change that breaks one is caught
where it happens rather than eight experiments downstream.
"""
import torch

from lagrangian_es.config import Config, RolloutCfg
from lagrangian_es.es import build, build_composer, build_sensors
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen

STEPS, EPS = 400, 24


def _run(with_composer=True, difficulty=1.0, quantile=1.0):
    kw = dict(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
              environment="singapore_cbd", sensors=("range", "range_down"),
              gating="arrival", seed=0,
              task_kw=(("n_legs", 2), ("max_leg", 20.0)),
              system_kw=(("free_start", True), ("speed_limit", 5.0)),
              trainable_kw=(("learned", True), ("damp_mode", "beams")),
              rollout=RolloutCfg(n_eps=EPS, ep_steps=STEPS, dead_mode="constant",
                                 dead_cost=40.0, goal_bonus=60.0,
                                 stop_on_arrival=True, stop_quantile=quantile))
    if with_composer:
        kw["composer"] = "policy_cont"
        kw["composer_kw"] = (("reach", 10.0), ("every", 20), ("measure_every", 20))
    cfg = Config(**kw)
    system, trainable, task = build(cfg)
    system.difficulty = difficulty
    comp = build_composer(cfg, system, trainable) if with_composer else None
    if comp is not None:
        comp.stochastic = True
        comp.records = []
        comp.record_rows = None
        comp.reset(EPS)
        comp.pair(EPS, 1)
    rig = Rollout(system, trainable, task, cfg.rollout, build_sensors(cfg, system), composer=comp)
    torch.manual_seed(0)
    with torch.no_grad():
        return rig.run(trainable.init()[None], task.sample(EPS, make_gen(1)), 2)


def test_success_and_finish_frac_agree():
    """`finish_frac` is 1.0 exactly when the flight never arrived.  The whole
    arrival-time weighting rests on that, and soft_time was added because the
    sentinel made failures unrankable."""
    r = _run()
    arrived = r.success.bool()
    assert torch.equal(arrived, r.finish_frac < 1.0), \
        "success and finish_frac disagree about which flights arrived"
    assert float(r.finish_frac.max()) <= 1.0 + 1e-6
    assert float(r.finish_frac.min()) >= 0.0


def test_soft_time_is_bounded_and_agrees_with_arrival():
    """Soft time exists so a failure can be RANKED; it must stay in [0, 1] and
    a flight that arrived quickly must not score worse than one that never did."""
    r = _run()
    assert r.soft_time is not None
    assert float(r.soft_time.min()) >= 0.0 and float(r.soft_time.max()) <= 1.0 + 1e-6
    ok, bad = r.success.bool(), ~r.success.bool()
    if bool(ok.any()) and bool(bad.any()):
        assert float(r.soft_time[ok].mean()) < float(r.soft_time[bad].mean()), \
            "arrivals do not score better than non-arrivals on soft time"


def test_a_dead_episode_stays_dead():
    """`alive` must be monotone: an episode that crashes cannot come back, or
    every rate computed from it is meaningless."""
    r = _run()
    assert r.death_step is not None
    dead = ~r.alive.bool()
    assert torch.equal(dead, r.death_step < STEPS), \
        "alive disagrees with death_step about which episodes died"


def test_the_quantile_shortcut_does_not_change_arrival():
    """stop_quantile is a SPEED approximation.  At 0.9 it truncated the batch
    once 90% had arrived and scored everything still flying as failed, capping
    the reported rate near 0.90 -- the curriculum's 0.95 bar became unreachable
    and the composer sat at 0% buildings for 500 iterations.  q=1.0 is exact."""
    exact = _run(quantile=1.0)
    cut = _run(quantile=0.5)
    assert float(exact.success.double().mean()) >= float(cut.success.double().mean()), \
        "the quantile shortcut reported MORE arrivals than the exact rollout"


# NOTE -- a test for "silence strands the composer" was tried here and removed.
# The property is real and measured (with the evolved genome: speaking arrives
# 0.977 at 8 m / 10%, forced silence arrives 0.000 with exactly 1.0 subgoal --
# it flies to its single opening waypoint and halts).  But it only shows up
# with a low level good enough to reach that waypoint: on `trainable.init()`
# both arms stop ~13 m short and the composer changes nothing.  Reproducing it
# would mean depending on an evolved genome that is not a repo artefact, so it
# belongs in the diagnostics, not the suite.
#
# Worth recording because the first reading of it was wrong: `silent -> 0.000
# arrivals AND 0.000 crashes` is not "the vehicle never moves".  It moves once,
# to the opening placement, and stops -- nothing ever updates the subgoal again.
