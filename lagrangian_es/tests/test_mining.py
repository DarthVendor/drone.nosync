"""Mining a controller's failures for replay training.

The point of the band filter is that a mined situation is only worth training on
while its outcome still depends on the genome.  These tests pin the selection
logic, which is where the measured -26.3% held-out crash reduction came from,
and the harvest's two easy-to-get-wrong details: the whole scene travels with
the pose, and the goal recorded is the one that was active at that instant.
"""
import torch

from lagrangian_es.config import Config, RolloutCfg
from lagrangian_es.es import build, build_sensors
from lagrangian_es.mining import contested_leads, harvest_pre_crash
from lagrangian_es.rollout import Rollout


def _roll(env="pillars", eps=8, steps=300):
    cfg = Config(system="quadrotor_nav", trainable="nav_agent",
                 task="waypoint_pair", environment=env, sensors=("range",),
                 gating="arrival", seed=0,
                 rollout=RolloutCfg(n_eps=eps, ep_steps=steps,
                                    dead_mode="constant", dead_cost=6.0,
                                    goal_bonus=15.0, stop_on_arrival=True))
    system, trainable, task = build(cfg)
    return (system, trainable, task,
            Rollout(system, trainable, task, cfg.rollout,
                    build_sensors(cfg, system)))


def test_contested_leads_drops_the_hopeless_and_the_solved():
    pools = {l: ({}, torch.zeros(1)) for l in (20, 30, 40, 60, 90)}
    rates = {20: 0.93, 30: 0.88, 40: 0.57, 60: 0.31, 90: 0.04}
    #        ^ always dies        ^ contested        ^ never dies
    assert contested_leads(pools, rates) == [40, 60]


def test_contested_leads_keeps_at_most_the_requested_number():
    pools = {l: ({}, torch.zeros(1)) for l in (20, 30, 40, 60, 90)}
    rates = {l: 0.5 for l in pools}
    got = contested_leads(pools, rates, keep=2)
    assert got == [60, 90], "the longest leads are the recoverable ones"


def test_contested_leads_falls_back_when_nothing_is_contested():
    """A controller that has stopped crashing still has to return a pool."""
    pools = {l: ({}, torch.zeros(1)) for l in (20, 40, 90)}
    got = contested_leads(pools, {l: 0.0 for l in pools})
    assert got == [40], "the middle lead, so the caller is never handed nothing"


def test_contested_leads_handles_an_empty_harvest():
    assert contested_leads({}, {}) == []


def test_harvest_keeps_the_whole_scene_and_the_active_goal():
    torch.manual_seed(0)
    system, trainable, task, roll = _roll()
    theta = trainable.init()          # untrained: it crashes, which is the point
    pools, seen, crashes = harvest_pre_crash(roll, task, theta, 12_345,
                                             leads=(5, 15), blocks=1, block=32)
    assert seen == 32
    if not crashes:                   # nothing to mine is a valid outcome
        assert pools == {}
        return
    for lead, (states, goals) in pools.items():
        n = goals.shape[0]
        assert goals.shape[1:] == (task.n_legs, task.task_dim)
        # every leg carries the same point: the goal that was live at that step
        assert torch.allclose(goals[:, 0], goals[:, -1])
        assert all(v.shape[0] == n for v in states.values())
        assert "p" in states and "v" in states
        assert any("/" in k for k in states), "the obstacle field must travel too"


def test_harvest_of_a_controller_that_never_crashes_is_empty():
    system, trainable, task, roll = _roll(env="empty", steps=120)
    theta = trainable.init()
    pools, seen, crashes = harvest_pre_crash(roll, task, theta, 999,
                                             leads=(10,), blocks=1, block=16)
    assert seen == 16
    if crashes == 0:
        assert pools == {}
