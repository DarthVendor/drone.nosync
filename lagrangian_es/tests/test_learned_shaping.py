

def test_split_terms_are_the_single_term_partitioned():
    """PULL and BRAKE laid end to end are the whole learned term: the same
    genome flies the same batch bit for bit at unit priorities, and only the
    split gives a composer's per-term priority a lever on the braking."""
    import torch
    from dataclasses import replace
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    tkw = (("learned", True), ("damp_mode", "beams"), ("extra_obs", (("range_down", 4), ("tilt", 3))))
    base = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd",
                  sensors=("range", "range_down", "tilt"), sensor_kw=(("range", (("spread", 2.0944),)),), gating="arrival", seed=0,
                  task_kw=(("n_legs", 2), ("max_leg", 15.0)), system_kw=(("free_start", True),), trainable_kw=tkw,
                  rollout=RolloutCfg(n_eps=8, ep_steps=150, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0))
    split = replace(base, trainable_kw=tkw + (("split_terms", True),))
    s1, t1, k1 = build(base); s2, t2, k2 = build(split)
    assert len(t1.terms) == 1 and len(t2.terms) == 2 and t1.dim == t2.dim
    th = t1.init(); th_split = t2.init()
    assert torch.equal(th, th_split), "the split's prior is the single term's prior"
    goals = k1.sample(8, make_gen(1))
    a = Rollout(s1, t1, k1, base.rollout, build_sensors(base, s1)).run(th[None], goals, 3)
    b = Rollout(s2, t2, k2, split.rollout, build_sensors(split, s2)).run(th_split[None], goals, 3)
    assert torch.equal(a.cost, b.cost) and torch.equal(a.alive, b.alive), "the split changed a flight"
    # the brake term on its own: no potential, only dissipation
    s = s2.reset(4, make_gen(0)); obs = {x.name: x.observe(s, make_gen(1)) for x in build_sensors(split, s2)}
    e = torch.randn(4, 3, dtype=torch.float64); v = torch.randn(4, 3, dtype=torch.float64)
    pull, brake = t2.terms; sl = list(t2.term_slices(th_split))
    assert torch.equal(brake.potential(sl[1], e, v, s["p"], obs), torch.zeros(4, dtype=torch.float64))
    assert torch.equal(pull.grad_potential(sl[0], e, torch.zeros_like(v), s["p"], obs),
                       pull.grad_potential(sl[0], e, v, s["p"], obs)), "the pull term must not depend on the velocity"


def test_the_potentials_yaw_gradient_matches_a_finite_difference_of_the_raycast():
    """Turning is implicit in the Lagrangian: dV/dpsi through the body-fixed
    beams must agree with rotating the body and re-casting the beams."""
    import math, torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.util import make_gen
    tkw = (("learned", True), ("damp_mode", "beams"), ("extra_obs", (("range_down", 4), ("tilt", 3))))
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd",
                 sensors=("range", "range_down", "tilt"), sensor_kw=(("range", (("spread", 2.0944),)),), gating="arrival", seed=0,
                 task_kw=(("n_legs", 2), ("max_leg", 15.0)), system_kw=(("free_start", True), ("yaw_mode", "lagrangian")), trainable_kw=tkw,
                 rollout=RolloutCfg(n_eps=64, ep_steps=10))
    sysm, tr, task = build(cfg); term = tr.terms[0]
    sens = {x.name: x for x in build_sensors(cfg, sysm)}
    th = tr.init() + 0.3 * torch.randn(tr.dim, generator=torch.Generator().manual_seed(4), dtype=torch.float64)   # a potential that reads its beams
    s = sysm.reset(64, make_gen(2)); goals = task.sample(64, make_gen(3))
    x = sysm.task_position(s); e = x - goals[:, 0]; v = sysm.task_velocity(s)
    def observe(st):
        r, J = sens["range"].measure(st)
        return {"range": r, "range/J": J, "range/dir": sens["range"]._dirs(st),
                "range_down": sens["range_down"].measure(st)[0], "tilt": sens["tilt"].observe(st, None)}
    def rotated(st, d):
        c, sn = math.cos(d), math.sin(d)
        Rz = torch.tensor([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        return dict(st, R=Rz @ st["R"])
    delta = 1e-5
    Vp = term.potential(th, e, v, x, observe(rotated(s, delta))); Vm = term.potential(th, e, v, x, observe(rotated(s, -delta)))
    fd = (Vp - Vm) / (2 * delta)
    an = term.grad_potential_yaw(th, e, v, x, observe(s))
    rel = (fd - an).abs() / fd.abs().clamp_min(1e-6)
    close = rel < 1e-2
    assert float(close.double().mean()) >= 0.8, (float(close.double().mean()), fd[:6], an[:6])   # edge-grazing beams aside
    assert float(rel[close].median()) < 1e-4


def test_lagrangian_yaw_drives_the_body_and_still_takes_a_command():
    """In `yaw_mode='lagrangian'` the heading has no rule: a yaw torque from
    the potential turns the body, a commanded heading still blends on top."""
    import math, torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd", sensors=("range",),
                 gating="arrival", seed=0, task_kw=(("n_legs", 2), ("max_leg", 15.0)), system_kw=(("free_start", True), ("yaw_mode", "lagrangian")),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")), rollout=RolloutCfg(n_eps=4, ep_steps=10))
    sysm, tr, task = build(cfg)
    assert sysm.allocator_dim == 6 and tr.dim == tr.terms[0].dim + 6
    s = sysm.reset(4, make_gen(0)); phi = sysm.allocator_init()
    F = sysm.gravity_force(s)
    u0 = sysm.allocate(F, s, phi.expand(4, -1))
    u1 = sysm.allocate(F, s, phi.expand(4, -1), yaw_torque=torch.full((4,), 0.05, dtype=torch.float64))
    assert torch.allclose(u0[:, :3], u1[:, :3]) and bool((u1[:, 3] - u0[:, 3] > 0.04).all()), "the yaw torque must land on the third torque channel only"
    cur = torch.atan2(s["R"][:, 1, 0], s["R"][:, 0, 0])
    u2 = sysm.allocate(F, s, phi.expand(4, -1), yaw=cur + 0.5, yaw_gate=torch.ones(4, dtype=torch.float64))
    assert not torch.allclose(u2[:, 3], u0[:, 3]), "a commanded heading must still act"


def test_the_fused_gradient_equals_the_two_gradients():
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.util import make_gen
    tkw = (("learned", True), ("damp_mode", "beams"), ("extra_obs", (("range_down", 4), ("tilt", 3))), ("split_terms", True))
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd",
                 sensors=("range", "range_down", "tilt"), sensor_kw=(("range", (("spread", 2.0944),)),), gating="arrival", seed=0,
                 task_kw=(("n_legs", 2), ("max_leg", 15.0)), system_kw=(("free_start", True), ("yaw_mode", "lagrangian")), trainable_kw=tkw,
                 rollout=RolloutCfg(n_eps=16, ep_steps=10))
    sysm, tr, task = build(cfg); sens = {x.name: x for x in build_sensors(cfg, sysm)}
    th = tr.init() + 0.3 * torch.randn(tr.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    s = sysm.reset(16, make_gen(2)); goals = task.sample(16, make_gen(3))
    x = sysm.task_position(s); e = x - goals[:, 0]; v = sysm.task_velocity(s)
    r, J = sens["range"].measure(s); rd, Jd = sens["range_down"].measure(s)
    obs = {"range": r, "range/J": J, "range/dir": sens["range"]._dirs(s), "range_down": rd, "range_down/J": Jd,
           "range_down/dir": sens["range_down"]._dirs(s), "tilt": sens["tilt"].observe(s, None)}
    for term, sl in zip(tr.terms, tr.term_slices(th)):
        g, gy = term.grad_potential_both(sl, e, v, x, obs)
        assert torch.allclose(g, term.grad_potential(sl, e, v, x, obs), atol=1e-12)
        assert torch.allclose(gy, term.grad_potential_yaw(sl, e, v, x, obs), atol=1e-12)


def test_the_plant_speed_limit_holds_in_flight():
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd", sensors=("range",),
                 gating="arrival", seed=0, task_kw=(("n_legs", 2), ("max_leg", 20.0)), system_kw=(("free_start", True), ("speed_limit", 5.0)),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")), rollout=RolloutCfg(n_eps=8, ep_steps=200))
    sysm, tr, task = build(cfg); sysm.difficulty = 0.0
    t = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm)).trace(tr.init()[None], task.sample(8, make_gen(1)), 3)
    sp = t.states["v"].norm(dim=-1)
    assert float(sp.max()) <= 5.0 + 1e-9, float(sp.max())
    assert float(sp.max()) > 4.0, "the fixture should reach the limit"


def test_the_fresh_lagrangian_flies_the_empty_map():
    """Bug test of the base controller: from its own init, with no composer and
    nothing to hit, it must not fall out of the sky.  Measured before the
    thrust envelope and the derived attitude prior: 29-31 of 32 on the floor
    inside two seconds."""
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    tkw = (("learned", True), ("damp_mode", "beams"), ("extra_obs", (("range_down", 4), ("tilt", 3))), ("split_terms", True))
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd",
                 sensors=("range", "range_down", "tilt"), sensor_kw=(("range", (("spread", 6.2831853),)),), gating="arrival", seed=0,
                 task_kw=(("n_legs", 2), ("max_leg", 15.0)), system_kw=(("prox_gain", 30.0), ("free_start", True), ("reset_yaw", 3.14159265), ("speed_limit", 5.0), ("yaw_mode", "lagrangian")),
                 trainable_kw=tkw, rollout=RolloutCfg(n_eps=32, ep_steps=300, dead_mode="constant", dead_cost=40.0, goal_bonus=15.0))
    sysm, tr, task = build(cfg); sysm.difficulty = 0.0
    t = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm)).trace(tr.init()[None], task.sample(32, make_gen(1)), 3)
    alive = t.alive[-1]
    assert float(alive.double().mean()) >= 0.9, f"the fresh controller crashed {int((~alive).sum())} of 32 on an empty map"
    tilt = torch.rad2deg(torch.acos(t.states["R"][..., 2, 2].clamp(-1, 1)))
    assert float(tilt.max()) < 90.0, "the body must never go past horizontal"
    x0 = t.states["p"][0]; xT = t.states["p"][-1]; g = t.goals[0]
    assert float(((x0 - g).norm(dim=-1) - (xT - g).norm(dim=-1)).mean()) > 3.0, "it must make progress toward the goal"


def test_the_thrust_envelope_keeps_the_vertical_and_bounds_the_force():
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd", sensors=("range",),
                 gating="arrival", seed=0, task_kw=(("n_legs", 2), ("max_leg", 15.0)), system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")), rollout=RolloutCfg(n_eps=4, ep_steps=10))
    sysm, tr, task = build(cfg); s = sysm.reset(4, make_gen(0)); phi = sysm.allocator_init().expand(4, -1)
    big = torch.tensor([[30.0, 0.0, 4.9]] * 4, dtype=torch.float64)                 # far beyond f_max, hover-weight vertical
    u = sysm.allocate(big, s, phi)
    assert float(u[:, 0].max()) <= sysm.f_max + 1e-9
    # the body is asked to tilt no further than the envelope allows with 4.9 N kept vertical
    import math
    assert math.degrees(math.acos(4.9 / sysm.f_max)) < 70.0


def test_the_pull_is_bounded_by_the_plants_budget_and_unchanged_near_the_goal():
    """V = |g|^2 near the goal, Huber beyond: the pull's force saturates at
    half the rotors' horizontal room at hover so the brake keeps the other
    half.  The quadratic bowl asked 20 N at 10 m against 9.6 N of room."""
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd", sensors=("range",),
                 gating="arrival", seed=0, task_kw=(("n_legs", 2), ("max_leg", 15.0)), system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams"), ("split_terms", True)), rollout=RolloutCfg(n_eps=4, ep_steps=10))
    sysm, tr, task = build(cfg); sysm.difficulty = 0.0
    s = sysm.reset(4, make_gen(0)); roll = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm)); roll._prime(s, make_gen(1))
    obs = roll._observe(s, make_gen(1), 0); th = tr.init(); x = sysm.task_position(s)
    pull = next(t for t in tr.terms if getattr(t, "part", "") == "potential"); sl = tr.term_slices(th)[tr.terms.index(pull)]
    budget = sysm.pull_budget(); assert 4.0 < budget < 5.5
    f = {}
    for d in (0.5, 1.0, 10.0):
        e = torch.tensor([d, 0.0, 0.0], dtype=torch.float64).expand(4, -1)
        f[d] = float(pull.grad_potential(sl, e, torch.zeros_like(e), x, obs).norm(dim=-1).mean())
    assert abs(f[0.5] - 1.0) < 0.15 and abs(f[1.0] - 2.0) < 0.3, f          # the bowl, 2 N/m, untouched near the goal
    assert f[10.0] <= budget + 1e-6 and f[10.0] > 0.9 * budget, f           # saturated at the budget, not vanishing
    e = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64).expand(4, -1)
    both, _ = pull.grad_potential_both(sl, e, torch.zeros_like(e), x, obs)
    assert torch.allclose(both, pull.grad_potential(sl, e, torch.zeros_like(e), x, obs))


def test_the_potentials_yaw_torque_acts_only_in_the_lagrangian_yaw_mode():
    """In the heading-commanding modes the attitude loop owns yaw; adding the
    potential's torque there fought it in every existing configuration."""
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build
    from lagrangian_es.util import make_gen
    out = {}
    for mode in ("world_x", "lagrangian"):
        cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd", sensors=("range",),
                     gating="arrival", seed=0, task_kw=(("n_legs", 2), ("max_leg", 15.0)), system_kw=(("free_start", True), ("yaw_mode", mode)),
                     trainable_kw=(("learned", True), ("damp_mode", "beams")), rollout=RolloutCfg(n_eps=4, ep_steps=10))
        sysm, tr, task = build(cfg); s = sysm.reset(4, make_gen(0)); phi = sysm.allocator_init().expand(4, -1)
        F = torch.tensor([[0.0, 0.0, 4.9]] * 4, dtype=torch.float64); yt = torch.full((4,), 0.2, dtype=torch.float64)
        u0 = sysm.allocate(F, s, phi); u1 = sysm.allocate(F, s, phi, yaw_torque=yt)
        out[mode] = float((u1 - u0).abs().max())
    assert out["world_x"] == 0.0, out
    assert out["lagrangian"] > 0.0, out
