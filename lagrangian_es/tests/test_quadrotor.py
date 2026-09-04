import copy

import torch

from lagrangian_es.systems import make_system
from lagrangian_es.systems.so3 import orthogonality_error, rodrigues
from lagrangian_es.util import make_gen

DT = torch.float64


def _sys(**kw):
    return make_system("quadrotor", **kw)


def _level_state(sys, B=4, z=1.0):
    p = torch.zeros(B, 3, dtype=DT)
    p[:, 2] = z
    return {
        "p": p,
        "v": torch.zeros(B, 3, dtype=DT),
        "R": torch.eye(3, dtype=DT).expand(B, 3, 3).clone(),
        "om": torch.zeros(B, 3, dtype=DT),
    }


def test_hover_is_an_equilibrium():
    """f = mg, tau = 0, R = I must hold station for a full episode."""
    sys = _sys()
    s = _level_state(sys, B=4, z=1.0)
    u = torch.zeros(4, 4, dtype=DT)
    u[:, 0] = sys.m * sys.g
    p0 = s["p"].clone()
    for _ in range(250):
        s = sys.step(s, u, 0.02)
    assert s["v"].norm(dim=-1).max() < 1e-6
    assert (s["p"] - p0).abs().max() < 1e-6
    assert s["om"].abs().max() < 1e-12
    assert orthogonality_error(s["R"]).max() < 1e-12


def test_unactuated_fall_conserves_energy():
    """u = 0 free fall: (1/2)mv^2 + mgz conserved to 1% at the production dt."""
    sys = _sys()
    dt, n = 0.02, 50
    s = _level_state(sys, B=3, z=20.0)
    u = torch.zeros(3, 4, dtype=DT)

    def energy(st):
        return 0.5 * sys.m * (st["v"] ** 2).sum(-1) + sys.m * sys.g * st["p"][:, 2]

    e0 = energy(s)
    for _ in range(n):
        s = sys.step(s, u, dt)
    rel = ((energy(s) - e0) / e0).abs().max()
    assert rel < 0.01, f"energy drift {rel:.4f}"
    # free fall really happened, i.e. the assertion is not vacuous
    assert (s["p"][:, 2] < 20.0 - 4.0).all()


def test_energy_drift_is_first_order_at_fixed_horizon():
    """Semi-implicit Euler drifts by O(dt^2) *per step*, hence O(dt) over a fixed
    horizon: halving dt must halve the error.  The closed form for free fall is
    (1/2) m g^2 dt T exactly, which pins the integrator down completely."""
    sys = _sys()

    def drift(dt, horizon=1.0):
        n = int(round(horizon / dt))
        s = _level_state(sys, B=1, z=50.0)
        u = torch.zeros(1, 4, dtype=DT)
        e = lambda st: 0.5 * sys.m * (st["v"] ** 2).sum(-1) + sys.m * sys.g * st["p"][:, 2]
        e0 = e(s)
        for _ in range(n):
            s = sys.step(s, u, dt)
        return float((e(s) - e0).abs())

    d1, d2 = drift(0.02), drift(0.01)
    assert abs(d1 / max(d2, 1e-18) - 2.0) < 0.05, f"ratio {d1 / d2:.3f} not ~2"
    closed_form = 0.5 * sys.m * sys.g**2 * 0.02 * 1.0
    assert abs(d1 - closed_form) / closed_form < 1e-6


def test_step_is_pure():
    sys = _sys()
    s = _level_state(sys, B=5, z=1.0)
    ref = copy.deepcopy(s)
    u = torch.randn(5, 4, dtype=DT)
    sys.step(s, u, 0.02)
    for k in s:
        assert torch.equal(s[k], ref[k]), f"step() mutated state key {k!r}"


def test_allocate_at_hover_returns_hover_thrust():
    """gravity_force in, level attitude, zero rate => f = mg and tau = 0."""
    sys = _sys()
    s = _level_state(sys, B=6, z=1.0)
    phi = torch.tensor([0.25, 0.25, 0.25, 0.10, 0.10, 0.10], dtype=DT).expand(6, 6)
    u = sys.allocate(sys.gravity_force(s), s, phi)
    assert torch.allclose(u[:, 0], torch.full((6,), sys.m * sys.g, dtype=DT), atol=1e-10)
    assert u[:, 1:].abs().max() < 1e-10


def test_allocate_tilts_toward_the_commanded_force():
    """A lateral force command must produce a torque that rolls/pitches the
    vehicle in the direction that tilts body z toward it."""
    sys = _sys()
    s = _level_state(sys, B=1, z=1.0)
    phi = torch.tensor([[0.5] * 3 + [0.2] * 3], dtype=DT)
    # R <- R @ exp(hat(om) dt), so +om_y rotates body z toward +x: a +x force
    # command must produce a POSITIVE torque about +y (and only about +y).
    F = sys.gravity_force(s) + torch.tensor([[2.0, 0.0, 0.0]], dtype=DT)
    tau = sys.allocate(F, s, phi)[:, 1:4]
    assert tau[0, 1] > 1e-3
    assert tau[0, 0].abs() < 1e-12 and tau[0, 2].abs() < 1e-12
    # ... and symmetrically, +y force tilts about -x
    Fy = sys.gravity_force(s) + torch.tensor([[0.0, 2.0, 0.0]], dtype=DT)
    tau_y = sys.allocate(Fy, s, phi)[:, 1:4]
    assert tau_y[0, 0] < -1e-3
    assert tau_y[0, 1].abs() < 1e-12 and tau_y[0, 2].abs() < 1e-12


def test_inv_mass_is_constant_and_correct():
    """The scope note in section 1, as an assertion: M is constant for a rigid
    body, so inv_mass must not depend on the state it is handed."""
    sys = _sys()
    a = sys.inv_mass(_level_state(sys, B=4, z=1.0))
    s2 = _level_state(sys, B=4, z=9.0)
    s2["R"] = rodrigues(torch.randn(4, 3, dtype=DT))
    s2["om"] = torch.randn(4, 3, dtype=DT)
    b = sys.inv_mass(s2)
    assert torch.equal(a, b)
    expect = torch.tensor(
        [1 / sys.m, 1 / float(sys.Jvec[0]), 1 / float(sys.Jvec[1]), 1 / float(sys.Jvec[2])],
        dtype=DT,
    )
    assert torch.allclose(a[0], expect)
    assert a.shape == (4, 4)


def test_gravity_and_shaping_cost():
    sys = _sys()
    s = _level_state(sys, B=2, z=1.0)
    g = sys.gravity_force(s)
    assert g.shape == (2, 3)
    assert torch.allclose(g[:, 2], torch.full((2,), sys.m * sys.g, dtype=DT))
    assert g[:, :2].abs().max() == 0
    assert sys.shaping_cost(s).abs().max() < 1e-12          # level => no tilt penalty
    s["R"] = rodrigues(torch.tensor([[0.0, torch.pi / 2, 0.0]], dtype=DT)).expand(2, 3, 3)
    assert (sys.shaping_cost(s) - 1.0).abs().max() < 1e-9   # 90 deg => 1 - 0


def test_alive_envelope():
    sys = _sys()
    s = _level_state(sys, B=5, z=1.0)
    s["p"][0, 2] = 0.0                       # below the floor
    s["v"][1] = torch.tensor([1e3, 0.0, 0.0], dtype=DT)
    s["om"][2] = torch.tensor([0.0, 1e3, 0.0], dtype=DT)
    s["p"][3] = torch.tensor([float("nan")] * 3, dtype=DT)
    ok = sys.alive(s)
    assert ok.tolist() == [False, False, False, False, True]


def test_reset_is_seed_reproducible():
    sys = _sys()
    a = sys.reset(16, make_gen(3))
    b = sys.reset(16, make_gen(3))
    c = sys.reset(16, make_gen(4))
    for k in a:
        assert torch.equal(a[k], b[k])
    assert not torch.equal(a["p"], c["p"])
    assert orthogonality_error(a["R"]).max() < 1e-12
    assert a["p"].shape == (16, 3) and a["R"].shape == (16, 3, 3)


def test_saturation_flags_bounds():
    sys = _sys()
    s = _level_state(sys, B=3, z=1.0)
    u = torch.zeros(3, 4, dtype=DT)
    u[0, 0] = sys.m * sys.g                  # mid-range thrust, no torque
    u[1, 0] = sys.f_max                      # thrust saturated
    u[2, 0] = sys.f_max
    u[2, 1:] = sys.tau_max                   # everything saturated
    sat = sys.saturation(u, s)
    assert float(sat[0]) == 0.0                      # mid-range on every channel
    assert abs(float(sat[1]) - 0.25) < 1e-12         # thrust only, 1 of 4
    assert abs(float(sat[2]) - 1.0) < 1e-12          # all four
