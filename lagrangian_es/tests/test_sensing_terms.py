"""Lens models, the landmark camera, and dL as a Lagrangian term."""
import time

import pytest
import torch
from torch.func import jacrev, vmap

from lagrangian_es.config import RolloutCfg
from lagrangian_es.rollout import Rollout
from lagrangian_es.sensors import make_sensor
from lagrangian_es.sensors.lens import DoubleSphere, Pinhole, interaction_matrix
from lagrangian_es.sensors.landmarks import LandmarkCamera
from lagrangian_es.systems import make_system
from lagrangian_es.tasks import make_task
from lagrangian_es.trainables import make_trainable
from lagrangian_es.trainables.sensor_terms import (
    FovBarrier, SensorDissipation, goal_gate, goal_gate_grad,
)
from lagrangian_es.trainables.terms import DissipationTerm, GoalBowl
from lagrangian_es.util import make_gen

DT = torch.float64
LENSES = [Pinhole(), DoubleSphere()]


# --------------------------------------------------------------------------- #
# lenses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lens", LENSES, ids=lambda L: L.name)
def test_lens_round_trip(lens):
    """unproject(project(X)) parallel to X across the full field of view."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(4000, 3, generator=g, dtype=DT)
    X[:, 2] = X[:, 2].abs() + 0.2
    X = X / X.norm(dim=-1, keepdim=True)
    uv = lens.project(X)
    vis = lens.in_view(X, uv)
    assert int(vis.sum()) > 200, "test is vacuous if nothing is in view"
    cos = (lens.unproject(uv) * X).sum(-1).clamp(-1, 1)
    assert float((1.0 - cos[vis]).abs().max()) < 1e-5


@pytest.mark.parametrize("lens", LENSES, ids=lambda L: L.name)
def test_lens_jacobian_matches_finite_differences(lens):
    g = torch.Generator().manual_seed(1)
    X = torch.randn(300, 3, generator=g, dtype=DT)
    X[:, 2] = X[:, 2].abs() + 0.6
    J = lens.jacobian(X)
    h = 1e-6
    fd = torch.zeros_like(J)
    for i in range(3):
        d = torch.zeros_like(X)
        d[:, i] = h
        fd[:, :, i] = (lens.project(X + d) - lens.project(X - d)) / (2 * h)
    assert float((J - fd).abs().max()) < 1e-4


@pytest.mark.parametrize("lens", LENSES, ids=lambda L: L.name)
def test_lens_is_trace_safe(lens):
    g = torch.Generator().manual_seed(2)
    X = torch.randn(64, 3, generator=g, dtype=DT)
    X[:, 2] = X[:, 2].abs() + 0.3
    out = vmap(jacrev(lambda x: lens.project(x)))(X)
    assert torch.isfinite(out).all()
    assert torch.isfinite(vmap(jacrev(lambda x: lens.unproject(x)))(
        lens.project(X))).all()


def test_double_sphere_has_the_wider_field_of_view():
    """Wide FOV is the point: it makes the FOV barrier less binding."""
    g = torch.Generator().manual_seed(3)
    X = torch.randn(4000, 3, generator=g, dtype=DT)
    X = X / X.norm(dim=-1, keepdim=True)
    n = [int(L.in_view(X, L.project(X)).sum()) for L in LENSES]
    assert n[1] > n[0]


def test_interaction_matrix_shape_and_zero_translation_rows():
    xn = torch.randn(16, 2, dtype=DT)
    Z = torch.rand(16, dtype=DT) + 0.5
    L = interaction_matrix(xn, Z)
    assert L.shape == (16, 2, 6)
    assert torch.allclose(L[:, 0, 1], torch.zeros(16, dtype=DT))
    assert torch.allclose(L[:, 1, 0], torch.zeros(16, dtype=DT))


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def test_landmark_camera_shapes_and_visibility():
    sysm = make_system("quadrotor")
    cam = make_sensor("landmark_camera", sysm, n_landmarks=12)
    s = sysm.reset(8, make_gen(0))
    assert cam.obs_dim == 24
    assert cam.observe(s, make_gen(1)).shape == (8, 24)
    assert cam.jacobian(s).shape == (8, 24, 3)
    assert cam.valid(s).shape == (8, 24)
    assert cam.visible(s).shape == (8, 12)


def test_camera_jacobian_matches_finite_differences():
    """d(pixels)/d(camera position), closed form, against central differences."""
    sysm = make_system("quadrotor")
    cam = make_sensor("landmark_camera", sysm, n_landmarks=6, sigma_px=0.0,
                      quantize=False)
    x = torch.tensor([[0.4, -0.3, 1.6], [-0.5, 0.2, 2.0]], dtype=DT)
    s = sysm.nominal_state(x, torch.zeros_like(x))
    J = cam.jacobian(s)
    h = 1e-6
    fd = torch.zeros_like(J)
    for i in range(3):
        d = torch.zeros_like(x)
        d[:, i] = h
        sp = sysm.nominal_state(x + d, torch.zeros_like(x))
        sm = sysm.nominal_state(x - d, torch.zeros_like(x))
        fd[:, :, i] = (cam.observe(sp, make_gen(0))
                       - cam.observe(sm, make_gen(0))) / (2 * h)
    assert float((J - fd).abs().max()) < 1e-3


def test_camera_quantizes_and_reports_out_of_view():
    sysm = make_system("quadrotor")
    cam = make_sensor("landmark_camera", sysm, n_landmarks=10, sigma_px=0.0)
    s = sysm.reset(32, make_gen(0))
    obs = cam.observe(s, make_gen(0))
    assert torch.equal(obs, torch.round(obs)), "quantization to integer pixels"
    v = cam.valid(s)
    assert v.dtype == torch.bool and not bool(v.all()), "some landmarks must be out of view"


def test_camera_dropout_needs_a_generator():
    """Unseeded randomness would sit outside common random numbers."""
    sysm = make_system("quadrotor")
    cam = make_sensor("landmark_camera", sysm, n_landmarks=8, dropout=0.5)
    s = sysm.reset(64, make_gen(0))
    geo = cam.valid(s)
    assert torch.equal(geo, cam.valid(s)), "without a generator, valid is deterministic"
    drop = cam.valid(s, make_gen(1))
    assert int((~drop).sum()) >= int((~geo).sum())


def test_camera_runtime_overhead():
    """A coarse guard against a pathological regression -- NOT a benchmark.

    Section 7 asks for < 15% at B=192, K=32.  With the default sensor stride the
    camera now adds about 14% on an idle machine, but this is a wall-clock ratio
    and both halves move: speeding the BASELINE up raises the ratio without the
    camera costing any more, and other load on the machine inflates everything
    unevenly.  So the bound is deliberately loose and catches an order-of-
    magnitude regression, not a percentage.
    """
    sysm = make_system("quadrotor")
    tr = make_trainable("energy_shaping", sysm)
    task = make_task("waypoint_pair", sysm)
    cfg = RolloutCfg(n_eps=8)
    goals = task.sample(8, make_gen(0))
    TH = tr.init()[None].expand(24, -1)          # B = 192

    def bench(sensors):
        r = Rollout(sysm, tr, task, cfg, sensors)
        r.run(TH, goals, 1)
        best = min(_time(r, TH, goals) for _ in range(3))
        return best

    def _time(r, TH, goals):
        t0 = time.time()
        r.run(TH, goals, 1)
        return time.time() - t0

    base = bench(None)
    cam = bench([make_sensor("landmark_camera", sysm, n_landmarks=32)])
    assert cam / base < 4.0, f"camera overhead {(cam / base - 1) * 100:.0f}% regressed"


# --------------------------------------------------------------------------- #
# dL
# --------------------------------------------------------------------------- #
def test_goal_gate_is_exactly_zero_inside_the_goal_ball():
    e = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.349, 0, 0]], dtype=DT)
    assert float(goal_gate(e, 0.35, 0.35).abs().max()) == 0.0
    assert float(goal_gate_grad(e, 0.35, 0.35).abs().max()) == 0.0
    far = torch.tensor([[5.0, 0, 0]], dtype=DT)
    assert abs(float(goal_gate(far, 0.35, 0.35)) - 1.0) < 1e-12


def test_delta_V_has_no_support_near_the_goal():
    """1000 random genomes: the sensor term contributes literally nothing inside
    r_goal, which is what makes the equilibrium immune to sensor bias."""
    bar = FovBarrier(3, "cam", 320, 240, 8, r_goal=0.4, gate_width=0.3)
    g = torch.Generator().manual_seed(0)
    TH = 2.0 * torch.randn(1000, bar.dim, generator=g, dtype=DT)
    obs = {"cam": 400.0 * torch.randn(1000, 16, generator=g, dtype=DT),
           "cam/J": torch.randn(1000, 16, 3, generator=g, dtype=DT)}
    for scale in (0.0, 0.1, 0.39):
        e = scale * torch.nn.functional.normalize(
            torch.randn(1000, 3, generator=g, dtype=DT), dim=-1)
        assert float(bar.potential(TH, e, e, e, obs).abs().max()) == 0.0
        assert float(bar.grad_potential(TH, e, e, e, obs).abs().max()) == 0.0


def test_delta_V_is_nonnegative_and_bites_outside_the_ball():
    bar = FovBarrier(3, "cam", 320, 240, 6, r_goal=0.3, gate_width=0.2)
    th = bar.init(dtype=DT)
    e = torch.tensor([[2.0, 0.0, 0.0]], dtype=DT)
    border = torch.tensor([[1.0, 120.0] * 6], dtype=DT)
    obs = {"cam": border, "cam/J": torch.ones(1, 12, 3, dtype=DT)}
    assert float(bar.potential(th, e, e, e, obs)) > 0
    assert float(bar.grad_potential(th, e, e, e, obs).abs().max()) > 0
    centre = torch.tensor([[160.0, 120.0] * 6], dtype=DT)
    assert float(bar.potential(th, e, e, e, {"cam": centre, "cam/J": obs["cam/J"]})) == 0.0


def test_sensor_dissipation_is_psd():
    d = SensorDissipation(3, "flow", obs_dim=3)
    g = torch.Generator().manual_seed(1)
    TH = 3.0 * torch.randn(500, d.dim, generator=g, dtype=DT)
    eig = torch.linalg.eigvalsh(d.damping(TH))
    assert float(eig.min()) >= -1e-10


def test_sensor_terms_are_missing_obs_safe():
    """A term must degrade to zero when its channel is absent, not crash: the
    same genome has to survive being run without sensors."""
    bar = FovBarrier(3, "cam", 320, 240, 4)
    th = bar.init(dtype=DT)
    e = torch.ones(2, 3, dtype=DT)
    assert float(bar.potential(th, e, e, e, None).abs().max()) == 0.0
    assert float(bar.grad_potential(th, e, e, e, {"other": e}).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# THE claim
# --------------------------------------------------------------------------- #
def test_constant_sensor_bias_does_not_move_the_equilibrium():
    """Inject a constant additive bias into the observation and assert the
    closed-loop equilibrium is still the goal.

    This is the single test that validates the section 4 claim.  An estimator-based
    controller would settle wherever the bias vanishes; because dV has no support
    near the goal, no bias can move it.  If this fails, dV has support it should
    not have.
    """
    sysm = make_system("quadrotor")
    task = make_task("waypoint_pair", sysm)

    class BiasedCamera(LandmarkCamera):
        name = "cam"

        def observe(self, s, gen):
            return super().observe(s, gen) + 60.0      # a large constant bias

    cam = BiasedCamera(sysm, n_landmarks=8, sigma_px=0.0, dropout=0.0,
                       quantize=False, latency_steps=0)
    # Stable gains, not the generation-0 prior: that prior crashes ~69% of the
    # time BY DESIGN (section 6), and a crashed vehicle has no equilibrium to
    # measure.  What is under test is WHERE the closed loop settles, not whether
    # an untrained genome can fly.
    terms = ([GoalBowl(3, a0=2.0) for _ in range(3)]
             + [DissipationTerm(3, d0=2.0),
                FovBarrier(3, "cam", cam.lens.width, cam.lens.height, cam.K,
                           w0=0.5, r_goal=0.35, gate_width=0.35)])
    tr = make_trainable("energy_shaping", sysm, terms=terms)

    # A goal close to the reset pose, so the vehicle reaches the neighbourhood of
    # the goal and SETTLES.  The claim under test is about the equilibrium, and a
    # crashed or still-transiting vehicle has none to measure.
    cfg = RolloutCfg(ep_steps=1200, n_eps=1)
    roll = Rollout(sysm, tr, task, cfg, [cam])
    goal = torch.tensor([[0.15, -0.1, 0.8]], dtype=DT)
    goals = goal[:, None, :].expand(1, 2, 3).clone()

    theta = tr.init().clone()
    theta[tr.policy_dim:tr.policy_dim + 6] = torch.tensor(
        [1.0, 1.0, 1.0, 0.35, 0.35, 0.35], dtype=DT)     # faster attitude loop
    res = roll.run(theta[None], goals, seed=5)
    assert bool(res.alive.all()), "the vehicle must survive to have an equilibrium"
    assert float(res.final_err.max()) < 1e-3, (
        f"biased sensor moved the equilibrium by {float(res.final_err.max()):.2e} m")


# --- the range barrier must stay connected to the control -------------------
def _barrier():
    from lagrangian_es.trainables.sensor_terms import RangeBarrier
    return RangeBarrier(3, "range", 12)


def test_range_barrier_cannot_be_parameterised_into_inertness():
    """`safe`/`margin` must be bounded, or they are a free switch-me-off knob.

    Unbounded, pushing `safe` past the sensor's max range makes every beam sit
    below the bump's support: the potential goes constant, its gradient vanishes,
    and the term stops affecting u.  Its parameters are then fitness-flat, and
    BLX-alpha extrapolation walks them to infinity -- the obstacles stage really
    did end at safe = 2.8e29 with the range sensor contributing nothing.
    """
    b = _barrier()
    for v in (1e2, 1e6, 1e14, 1e29, -1e29):
        th = torch.tensor([1.0, v, v], dtype=torch.float64)
        _, safe, margin = b._params(th)
        assert b.safe_lo <= float(safe) <= b.safe_hi
        assert b.margin_lo <= float(margin) <= b.margin_hi
        # probe INSIDE whatever safe radius this parameterisation chose: a small
        # safe radius is a legitimate thing to learn, a zero gradient there is not
        close = 0.5 * safe
        _, d = b._bump((close - safe) / margin)
        assert abs(float(d)) > 1e-6, f"barrier went inert at theta={v}"


def test_range_barrier_still_repels_inside_the_safe_radius():
    """The plain cubic bump is FLAT below m=0 -- repulsion switches off exactly
    where a collision is imminent.  The gradient must stay at its maximum there,
    not fall to zero."""
    b = _barrier()
    _, safe, margin = b._params(b.init())
    deep = torch.tensor([0.02, 0.10, 0.25], dtype=torch.float64)
    _, d_deep = b._bump((deep - safe) / margin)
    _, d_edge = b._bump(torch.zeros(1, dtype=torch.float64))
    assert torch.all(d_deep.abs() > 0.0), "no repulsion inside the safe radius"
    assert torch.allclose(d_deep.abs(), d_edge.abs().expand_as(d_deep)), \
        "repulsion inside the safe radius must hold at the maximum"
    # ... and stay bounded, which is what the certificate claims
    wide = torch.linspace(-50.0, 5.0, 400, dtype=torch.float64)
    _, d_all = b._bump(wide)
    assert float(d_all.abs().max()) <= 3.0 + 1e-9


def test_range_barrier_potential_is_nonnegative_and_monotone_in_proximity():
    b = _barrier()
    _, safe, margin = b._params(b.init())
    obs = torch.linspace(0.01, 4.0, 300, dtype=torch.float64)
    psi, _ = b._bump((obs - safe) / margin)
    assert torch.all(psi >= 0.0)
    assert torch.all(psi[1:] <= psi[:-1] + 1e-12), "closer must never cost less"


# --- hard-wall (infinite) barrier mode --------------------------------------
def _wall_barrier():
    from lagrangian_es.trainables.sensor_terms import RangeBarrier
    return RangeBarrier(3, "range", 12, mode="log")


def test_wall_barrier_diverges_toward_the_obstacle_and_vanishes_in_open_space():
    b = _wall_barrier()
    th = b.init()
    _, safe, _ = b._params(th)
    near = torch.tensor([[float(safe) - 1e-3, b.d_min + 1e-3]], dtype=torch.float64)
    g = b.raw_grad(th, near)[0]
    assert abs(float(g[0])) < abs(float(g[1])) / 20.0, \
        "barrier must steepen sharply as the wall is approached"
    far = torch.tensor([[float(safe) + 1e-6, 4.0]], dtype=torch.float64)
    assert torch.allclose(b.raw_grad(th, far), torch.zeros(1, 2, dtype=torch.float64))


def test_wall_barrier_is_c1_where_it_meets_open_space():
    """A bare -ln z leaves a kink at the edge of its support; the shifted form
    must meet open space with both value and slope at zero."""
    b = _wall_barrier()
    th = b.init()
    _, safe, _ = b._params(th)
    edge = torch.tensor([[float(safe) - 1e-6]], dtype=torch.float64)
    assert abs(float(b.raw_potential(th, edge))) < 1e-8
    assert abs(float(b.raw_grad(th, edge)[0, 0])) < 1e-4


def test_wall_barrier_still_pushes_out_after_tunnelling_through():
    """At dt=0.02 and 4 m/s the vehicle covers 8 cm per step, so it CAN end up
    inside d_min.  Clamping the log's argument is unavoidable; letting the push
    vanish there is not -- nothing would drive it back out."""
    b = _wall_barrier()
    th = b.init()
    deep = torch.tensor([[b.d_min - 0.05, b.d_min * 0.1, 0.0]], dtype=torch.float64)
    g = b.raw_grad(th, deep)[0]
    assert torch.all(g < 0.0), "no restoring push once inside the wall"
    _, safe, _ = b._params(th)
    edge = torch.tensor([[b.d_min + 1e-4]], dtype=torch.float64)
    assert torch.all(g.abs() <= abs(float(b.raw_grad(th, edge)[0, 0])) + 1e-6), \
        "push inside the wall must saturate, not exceed the boundary value"


def test_wall_barrier_potential_is_nonnegative():
    b = _wall_barrier()
    th = b.init()
    obs = torch.linspace(0.0, 4.0, 400, dtype=torch.float64)[None]
    assert torch.all(b.raw_potential(th, obs) >= -1e-12)


def test_bump_mode_is_unchanged_and_is_still_the_default():
    from lagrangian_es.trainables.sensor_terms import RangeBarrier
    assert RangeBarrier(3, "range", 12).mode == "bump"
    assert RangeBarrier(3, "range", 12).certificate(None)["bounded_grad"] is True
    assert _wall_barrier().certificate(None)["bounded_grad"] is False


# --- closing damper (envelope-gated) -------------------------------------
def _damper(**kw):
    from lagrangian_es.trainables.sensor_terms import RangeDamper
    return RangeDamper(3, "range", 12, **kw)


def _obs(rng, beam_dir=(1.0, 0.0, 0.0), n=12):
    """One beam looking along `beam_dir`; J = d(range)/dx points the other way."""
    J = torch.zeros(1, n, 3, dtype=torch.float64)
    J[0, 0] = -torch.tensor(beam_dir, dtype=torch.float64)
    z = torch.full((1, n), 8.0, dtype=torch.float64)
    z[0, 0] = rng
    return {"range": z, "range/J": J}


def _force(t, v, rng):
    th = t.init(dtype=torch.float64)
    e = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    x = torch.zeros(1, 3, dtype=torch.float64)
    vv = torch.tensor([v], dtype=torch.float64)
    return -t.grad_potential(th, e, vv, x, _obs(rng))


def _v_safe(t, d):
    _, a = t._params(t.init(dtype=torch.float64))
    return float((2.0 * float(a) * d) ** 0.5)


def test_range_damper_is_dissipative_and_silent_when_receding():
    """It is a Rayleigh term, so it may only ever REMOVE energy.  If it could add
    any, `H = T + V_d` would stop being non-increasing and the LaSalle argument
    the controller rests on would go with it."""
    t = _damper()
    d = 0.3
    for vx in (_v_safe(t, d) + 0.5, _v_safe(t, d) + 3.0):
        F = _force(t, [vx, 0.0, 0.0], d)
        assert float((F * torch.tensor([[vx, 0.0, 0.0]],
                                       dtype=torch.float64)).sum()) < 0.0
    for vx in (-3.0, 0.0):
        F = _force(t, [vx, 0.0, 0.0], d)
        assert torch.allclose(F, torch.zeros_like(F), atol=1e-12)


def test_range_damper_is_silent_inside_the_safe_envelope():
    """The whole point of the envelope.  A damper keyed on distance alone brakes
    whenever anything is within reach, which in a pillar field is nearly always;
    it then pays a toll on every episode to rescue the few that would collide,
    and fitness correctly rejects the trade.  Cruising past an obstacle at a
    speed the remaining distance can absorb must cost exactly nothing.
    """
    t = _damper()
    for d in (0.3, 0.6, 1.2, 2.0):
        F = _force(t, [_v_safe(t, d) * 0.95, 0.0, 0.0], d)
        assert torch.allclose(F, torch.zeros_like(F), atol=1e-12), \
            f"damper fired inside the envelope at d={d}"
        F = _force(t, [_v_safe(t, d) + 1.0, 0.0, 0.0], d)
        assert float(F[0, 0]) < 0.0, f"damper silent OUTSIDE the envelope at d={d}"


def test_range_damper_resists_only_the_excess_over_the_safe_speed():
    t = _damper()
    d = 0.4
    vs = _v_safe(t, d)
    f1 = float(_force(t, [vs + 1.0, 0.0, 0.0], d)[0, 0])
    f2 = float(_force(t, [vs + 2.0, 0.0, 0.0], d)[0, 0])
    assert f2 == pytest.approx(2.0 * f1, rel=1e-9)


def test_range_damper_envelope_widens_with_distance():
    """sqrt(2 a d): more room means more speed is tolerated, which is what makes
    the support compact without a cutoff -- far away nothing triggers."""
    t = _damper()
    assert _v_safe(t, 2.0) > _v_safe(t, 0.5)
    v = 2.5
    near = _force(t, [v, 0.0, 0.0], 0.2)
    far = _force(t, [v, 0.0, 0.0], 4.0)
    assert float(near[0, 0]) < 0.0
    assert torch.allclose(far, torch.zeros_like(far), atol=1e-12)


def test_range_damper_leaves_tangential_motion_untouched():
    """Damping the approach must not forbid going AROUND."""
    t = _damper()
    F = _force(t, [0.0, 5.0, 0.0], 0.3)
    assert torch.allclose(F, torch.zeros_like(F), atol=1e-12)


def test_range_damper_accel_is_bounded_so_it_cannot_be_switched_off():
    t = _damper()
    for v in (1e3, 1e12, -1e12):
        _, a = t._params(torch.tensor([1.0, v], dtype=torch.float64))
        assert t.accel_lo <= float(a) <= t.accel_hi


def test_range_damper_is_silent_without_observations():
    t = _damper()
    th = t.init(dtype=torch.float64)
    e = torch.ones(2, 3, dtype=torch.float64)
    z = torch.zeros(2, 3, dtype=torch.float64)
    assert torch.equal(t.grad_potential(th, e, z, z, None), z)


def test_range_damper_certificate_declares_dissipative_not_bounded():
    c = _damper().certificate(None)
    assert c["dissipative"] and c["psd"] and c["zero_at_goal"]
    assert c["bounded_grad"] is False        # grows with closing speed, by design


# --- gyroscopic steering ----------------------------------------------------
def _vortex(**kw):
    from lagrangian_es.trainables.sensor_terms import RangeVortex
    return RangeVortex(3, "range", 12, **kw)


def test_range_vortex_does_no_work_whatever_the_state():
    """The entire justification for the term.  F perpendicular to v leaves
    `H = T + V_d` non-increasing, so the certificate survives -- while breaking
    the pure-gradient structure that forces the flow onto V_d's critical points.
    If it could do work, it would be an unconstrained energy source."""
    t = _vortex()
    th = t.init(dtype=torch.float64)
    e = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    x = torch.zeros(1, 3, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        v = torch.randn(1, 3, generator=g, dtype=torch.float64) * 2.0
        J = torch.nn.functional.normalize(
            torch.randn(1, 12, 3, generator=g, dtype=torch.float64), dim=-1)
        z = torch.rand(1, 12, generator=g, dtype=torch.float64) * 3.0
        F = -t.grad_potential(th, e, v, x, {"range": z, "range/J": J})
        assert abs(float((F * v).sum())) < 1e-9


def test_range_vortex_force_is_perpendicular_to_velocity():
    t = _vortex()
    th = t.init(dtype=torch.float64)
    e = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    x = torch.zeros(1, 3, dtype=torch.float64)
    J = torch.zeros(1, 12, 3, dtype=torch.float64)
    J[0, 0] = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
    z = torch.full((1, 12), 4.0, dtype=torch.float64)
    z[0, 0] = 0.4
    v = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    F = -t.grad_potential(th, e, v, x, {"range": z, "range/J": J})
    assert float(F.norm()) > 1e-6, "test is vacuous if the term is silent"
    cosang = float((F * v).sum() / (F.norm() * v.norm()))
    assert abs(cosang) < 1e-9


def test_range_vortex_is_silent_in_open_space_and_at_the_goal():
    t = _vortex()
    th = t.init(dtype=torch.float64)
    x = torch.zeros(1, 3, dtype=torch.float64)
    v = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    J = torch.zeros(1, 12, 3, dtype=torch.float64)
    J[0, 0] = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
    far = torch.full((1, 12), 4.0, dtype=torch.float64)
    e = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    F = -t.grad_potential(th, e, v, x, {"range": far, "range/J": J})
    assert torch.allclose(F, torch.zeros_like(F), atol=1e-12)
    near = far.clone(); near[0, 0] = 0.4
    at_goal = torch.tensor([[0.02, 0.0, 0.0]], dtype=torch.float64)
    F = -t.grad_potential(th, at_goal, v, x, {"range": near, "range/J": J})
    assert torch.allclose(F, torch.zeros_like(F), atol=1e-12)


def test_range_vortex_reach_is_bounded():
    t = _vortex()
    for v in (1e3, 1e12, -1e12):
        _, r = t._params(torch.tensor([1.0, v], dtype=torch.float64))
        assert t.reach_lo <= float(r) <= t.reach_hi


def test_range_vortex_certificate_declares_gyroscopic():
    c = _vortex().certificate(None)
    assert c["gyroscopic"] and c["workless"] and c["zero_at_goal"]
