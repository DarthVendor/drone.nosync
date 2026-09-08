"""The task-level layer, and the promises that make it safe to put there.

`FixedWeights` is the identity, and it has to be exactly that: Stage A evolves
the low level with the composer in place, so a composer that changed the
controller by a rounding error would put that error into every fitness it ranks.
The hold is the part that keeps the certificate -- V_d = sum_i alpha_i V_i is
the gradient of a function only while alpha is constant in x -- so it must move
the realized spec at a bounded rate and never faster.  The two interlocks are
non-learned by design and are tested for the edges they guard.
"""
import torch

from lagrangian_es.composer import (FixedWeights, SpecHold, TaskSpec,
                                    ground_release, stale_fallback)
from lagrangian_es.config import Config, RolloutCfg
from lagrangian_es.es import build, build_composer, build_sensors
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen


def _cfg(composer=""):
    return Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                  environment="singapore_cbd", sensors=("range",),
                  gating="arrival", seed=0, composer=composer,
                  task_kw=(("n_legs", 2), ("max_leg", 10.0)),
                  system_kw=(("free_start", True),),
                  trainable_kw=(("learned", True), ("damp_mode", "beams")),
                  rollout=RolloutCfg(n_eps=6, ep_steps=120, dead_mode="constant",
                                     dead_cost=6.0, goal_bonus=15.0))


def test_fixed_weights_is_the_identity_through_a_real_rollout():
    a = _cfg(); b = _cfg("fixed")
    sysm, tr, task = build(a)
    th = tr.init()[None]
    goals = task.sample(6, make_gen(1))
    bare = Rollout(sysm, tr, task, a.rollout, build_sensors(a, sysm))
    comp = build_composer(b, sysm, tr)
    assert isinstance(comp, FixedWeights)
    held = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=comp)
    r0 = bare.run(th, goals, 7); r1 = held.run(th, goals, 7)
    assert torch.equal(r0.fitness, r1.fitness), "the identity spec changed the fitness"
    assert torch.equal(r0.cost, r1.cost)
    assert torch.equal(r0.alive, r1.alive)


def test_the_measurement_chain_reports_what_happened():
    b = _cfg("fixed")
    sysm, tr, task = build(b)
    comp = build_composer(b, sysm, tr); comp.every = 40
    roll = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=comp)
    roll.run(tr.init()[None], task.sample(6, make_gen(2)), 8)
    assert len(roll.chain) == 120 // 40 - 1, "one token per completed interval"
    tok = roll.chain[0]
    for k in ("progress", "remaining", "min_beam", "alive", "arrived"):
        assert k in tok and tok[k].shape[0] == 6
    assert torch.isfinite(tok["progress"]).all()


def test_the_hold_never_moves_faster_than_its_rate():
    B, d, n = 4, 3, 2
    hold = SpecHold(B, d, n, dt=0.02, dtype=torch.float64, device="cpu",
                    omega_n=2.0, reach=10.0)
    far = TaskSpec(delta=torch.full((B, d), 30.0, dtype=torch.float64),
                   alpha=torch.full((B, n), 5.0, dtype=torch.float64),
                   gate=torch.zeros(B, n, dtype=torch.float64))
    hold.set_target(far)
    prev = hold.realized.clone()
    for _ in range(50):
        cur = hold.step()
        assert float((cur.delta - prev.delta).norm(dim=-1).max()) <= hold.rate_m + 1e-12
        assert float((cur.alpha - prev.alpha).abs().max()) <= hold.rate_w + 1e-12
        assert float((cur.gate - prev.gate).abs().max()) <= hold.rate_w + 1e-12
        prev = cur.clone()
    for _ in range(5000):
        cur = hold.step()
    assert torch.allclose(cur.delta, far.delta) and torch.allclose(cur.alpha, far.alpha)
    assert torch.allclose(cur.gate, far.gate)


def test_the_hold_is_constant_between_composer_calls_when_settled():
    """d(alpha)/dx = 0 within an interval: once settled, stepping does nothing."""
    hold = SpecHold(2, 3, 1, dt=0.02, dtype=torch.float64, device="cpu")
    tgt = TaskSpec.identity(2, 3, 1, torch.float64, "cpu"); tgt.delta[:] = 1.0
    hold.set_target(tgt)
    for _ in range(2000):
        hold.step()
    a = hold.step().clone(); b = hold.step()
    assert torch.equal(a.delta, b.delta) and torch.equal(a.weight, b.weight)


def test_ground_release_needs_low_and_slow():
    z = torch.tensor([0.3, 0.3, 2.0, 0.3]); vz = torch.tensor([0.1, -0.9, 0.1, 0.49])
    got = ground_release(z, vz, z_release=0.6, vz_max=0.5)
    assert got.tolist() == [True, False, False, True]


def test_stale_fallback_only_touches_stale_rows():
    hold = SpecHold(3, 3, 2, dt=0.02, dtype=torch.float64, device="cpu")
    tgt = TaskSpec.identity(3, 3, 2, torch.float64, "cpu"); tgt.delta[:] = 4.0
    hold.set_target(tgt)
    hold.age = torch.tensor([0, 5, 12])
    stale = stale_fallback(hold, k=8)
    assert stale.tolist() == [False, False, True]
    assert torch.equal(hold.target.delta[2], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(hold.target.delta[0], torch.full((3,), 4.0, dtype=torch.float64))


def test_oracle_subgoals_stay_clear_of_buildings_and_within_reach():
    """The teacher must only ever ask for legs the low level is known to fly:
    the subgoal sits within `reach` of the vehicle and outside the margin."""
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range",), gating="arrival",
                 seed=0, composer="oracle",
                 composer_kw=(("reach", 10.0), ("margin", 0.9)),
                 task_kw=(("n_legs", 2), ("max_leg", 0.0)),
                 system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")))
    sysm, tr, task = build(cfg)
    comp = build_composer(cfg, sysm, tr)
    goals = task.sample(12, make_gen(3))
    s = sysm.reset(12, make_gen(4))
    x = sysm.task_position(s)
    spec = comp.emit({"x": x, "v": sysm.task_velocity(s), "goal": goals[:, 0],
                      "alive": torch.ones(12, dtype=torch.bool), "state": s,
                      "arrived": torch.zeros(12, dtype=torch.bool),
                      "leg": torch.zeros(12, dtype=torch.long), "t": 0})
    sub = goals[:, 0] + spec.delta
    hop = (sub[:, :2] - x[:, :2]).norm(dim=-1)
    assert float(hop.max()) <= 10.0 + 1e-6, "subgoal beyond the measured reach"
    clear = sysm.env.sdf(sub, s)
    assert float(clear.min()) > 0.9 - 0.5 * 2 ** 0.5, "subgoal inside the clearance margin"
    assert torch.equal(spec.weight, torch.ones_like(spec.weight)), "the oracle decides where, not how"
    # a goal already within reach is handed over as-is -- provided it is clear.
    # A goal inside the margin is relocated to the nearest free cell first,
    # which is the planner refusing to send the vehicle into a wall.
    near = x.clone(); near[:, :2] += 3.0
    spec2 = comp.emit({"x": x, "v": sysm.task_velocity(s), "goal": near,
                       "alive": torch.ones(12, dtype=torch.bool), "state": s,
                       "arrived": torch.zeros(12, dtype=torch.bool),
                       "leg": torch.zeros(12, dtype=torch.long), "t": 0})
    ok = sysm.env.sdf(near, s) > 0.9 + 0.5 * 2 ** 0.5
    assert bool(ok.any()), "fixture: no clear near goal to check"
    off = spec2.delta.norm(dim=-1)
    assert float(off[ok].max()) == 0.0, \
        "a clear goal within reach must be handed over EXACTLY -- a cell centre can sit outside the arrival tolerance"
    assert float(off[~ok].min()) >= 0.0


def test_line_of_sight_oracle_only_asks_for_what_the_vehicle_can_see():
    """The city measurement: an occluded leg is lost 91% of the time, a visible
    one 24%.  With `los=True` every subgoal must be visible from the vehicle
    at the clearance margin -- the straight line to it never enters the margin."""
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="occluded", sensors=("range",), gating="arrival",
                 seed=0, composer="oracle",
                 composer_kw=(("reach", 10.0), ("margin", 0.9), ("los", True)),
                 task_kw=(("n_legs", 2), ("max_leg", 0.0)),
                 system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")))
    sysm, tr, task = build(cfg)
    comp = build_composer(cfg, sysm, tr)
    B = 24
    goals = task.sample(B, make_gen(11)); s = sysm.reset(B, make_gen(12))
    x = sysm.task_position(s)
    spec = comp.emit({"x": x, "v": sysm.task_velocity(s), "goal": goals[:, 0],
                      "alive": torch.ones(B, dtype=torch.bool), "state": s,
                      "arrived": torch.zeros(B, dtype=torch.bool),
                      "leg": torch.zeros(B, dtype=torch.long), "t": 0, "chain": []})
    sub = goals[:, 0] + spec.delta
    n = 200
    ts = torch.linspace(0, 1, n, dtype=x.dtype)[None, :, None]
    pts = x[:, None] + ts * (sub - x)[:, None]
    d = torch.stack([sysm.env.sdf(pts[:, i], s) for i in range(n)], 1)
    # the vehicle's own start may sit inside the margin; judge the line past it
    assert float(d[:, n // 8:].min()) > 0.0, "a line-of-sight subgoal is hidden behind a wall"
    assert float((sub - x).norm(dim=-1).max()) <= 10.0 + 1e-9


def test_training_with_the_identity_composer_is_byte_identical():
    """Stage A runs with the composer in place and out of the way: a whole
    training run's fitness history must not move by a rounding error."""
    from dataclasses import replace
    from lagrangian_es.config import ESCfg
    from lagrangian_es.es import train
    base = _cfg()
    small = replace(base, rollout=replace(base.rollout, n_eps=4, ep_steps=60),
                    es=ESCfg(pop=6, gens=3, sigma0=0.05, elite_frac=0.5, strategy="ga",
                             whiten=False, metric_every=1000))
    with_comp = replace(small, composer="fixed")
    a = train(small, *build(small)); b = train(with_comp, *build(with_comp))
    assert torch.equal(a.theta, b.theta)
    assert [h["fitness_elite"] for h in a.history] == [h["fitness_elite"] for h in b.history]


def test_workers_build_the_composer_the_config_names():
    """The parallel path must fly the same controller the parent evaluates."""
    from lagrangian_es import parallel
    cfg = _cfg("fixed")
    parallel._init({"cfg": cfg})
    assert isinstance(parallel._RIG.composer, FixedWeights)
    parallel._init({"cfg": _cfg()})
    assert parallel._RIG.composer is None


def test_sensors_no_term_reads_follow_the_composers_cadence():
    """The depth camera is read by the composer alone, so it is cast at the
    composer's interval; the range fan the low level reads keeps its stride."""
    from dataclasses import replace
    cfg = replace(_cfg("fixed"), sensors=("range", "depth_camera"), composer="transformer",
                  composer_kw=(("reach", 10.0), ("every", 10)))
    sysm, tr, task = build(cfg); sens = build_sensors(cfg, sysm)
    before = {s.name: int(getattr(s, "update_every", 1)) for s in sens}
    Rollout(sysm, tr, task, cfg.rollout, sens, composer=build_composer(cfg, sysm, tr))
    after = {s.name: int(getattr(s, "update_every", 1)) for s in sens}
    assert after["range"] == before["range"], "the low level reads the fan; its stride is not the composer's to change"
    assert after["depth_camera"] == 10, after
