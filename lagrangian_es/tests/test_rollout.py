

def test_the_charge_on_a_tour_is_the_distance_to_go_and_never_rises_for_finishing_a_leg():
    """On a multi-leg tour the position charge is the distance to the active
    waypoint PLUS the legs after it, so an arrival is continuous: standing on
    waypoint 1 costs the same the step before and the step after the leg
    advances.  A learned composer parked 0.28 m short of the first waypoint
    when the charge jumped there."""
    import torch
    from lagrangian_es.rollout import Rollout
    g = torch.tensor([[[0.0, 0.0, 1.0], [3.0, 4.0, 1.0], [3.0, 4.0, 13.0]],       # legs of 5 m and 12 m
                      [[0.0, 0.0, 1.0], [6.0, 8.0, 1.0], [6.0, 8.0, 1.0]]])       # 10 m, then a padded leg
    tab = Rollout._to_go(g)
    assert torch.allclose(tab, torch.tensor([[17.0, 12.0, 0.0], [10.0, 0.0, 0.0]])), tab
    assert Rollout._to_go(g[:, :1]) is None
    x = g[:, 0]                                                                       # standing on the first waypoint
    d0 = (x - g[:, 0]).norm(dim=-1) + tab[:, 0]                                       # charge while still on leg 0: 0 + 17
    d1 = (x - g[:, 1]).norm(dim=-1) + tab[:, 1]                                       # charge the step after the leg advances: 5 + 12
    assert torch.allclose(d0, d1), (d0, d1)
    assert torch.allclose(d0, torch.tensor([17.0, 10.0]))


def test_the_result_records_when_each_episode_crashed():
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="waypoint_pair", environment="pillars", sensors=("range",), gating="arrival",
                 rollout=RolloutCfg(n_eps=16, ep_steps=300, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0, stop_on_arrival=True))
    sysm, tr, task = build(cfg)
    r = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm)).run(tr.init()[None], task.sample(16, make_gen(1)), 3)
    assert r.death_step.shape == (16,)
    assert bool((r.death_step[r.alive] == 300).all()), "a survivor has no crash step"
    assert bool((r.death_step[~r.alive] < 300).all()), "a crashed row records the step it died"


def test_soft_time_ranks_the_flights_finish_frac_cannot():
    """`finish_frac` is 1.0 for EVERY flight that did not arrive, so a vehicle
    that crashed on takeoff and one that hovered 30 cm short are identical to
    it.  That is why the imitation filter could only give failures weight zero,
    and why nothing in the objective could push the failure rate down.

    `soft_time` accumulates sigmoid((|err| - tol)/tol) over the steps a flight
    was still going, so it falls as the vehicle spends time NEAR the goal
    whether or not it satisfies the dwell test.  Failures become rankable.
    """
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="waypoint_pair",
                 environment="pillars", sensors=("range",), gating="arrival", seed=0,
                 rollout=RolloutCfg(n_eps=24, ep_steps=300, dead_mode="constant",
                                    dead_cost=40.0, goal_bonus=60.0))
    system, trainable, task = build(cfg)
    roll = Rollout(system, trainable, task, cfg.rollout, build_sensors(cfg, system))
    torch.manual_seed(0)
    TH = trainable.init().expand(4, -1).clone()
    TH = TH + 0.05 * torch.randn(TH.shape, generator=make_gen(3), dtype=TH.dtype)
    r = roll.run(TH, task.sample(24, make_gen(2)), 5)

    assert r.soft_time is not None and r.soft_time.shape == r.finish_frac.shape
    assert float(r.soft_time.min()) >= 0.0 and float(r.soft_time.max()) <= 1.0 + 1e-6
    missed = r.finish_frac >= 1.0
    assert bool(missed.any()), "fixture is only meaningful with some non-arrivals"
    # finish_frac collapses them all onto one value; soft_time must not
    assert float(r.finish_frac[missed].std()) == 0.0
    assert float(r.soft_time[missed].std()) > 0.0, \
        "soft_time gives every failure the same score -- it ranks nothing"
    # and it must still agree with finish_frac about which flights were quick
    if bool((~missed).any()) and int((~missed).sum()) > 2:
        ok = ~missed
        c = torch.corrcoef(torch.stack([r.finish_frac[ok].double(), r.soft_time[ok].double()]))[0, 1]
        assert float(c) > 0.3, f"soft_time disagrees with arrival time on arrivals (r={float(c):.2f})"
