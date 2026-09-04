"""Connection joints: welds, cables, and compliant links."""
import pytest
import torch

from lagrangian_es.systems import make_system
from lagrangian_es.systems.connectors import (
    CONNECTORS, Cable, RigidLink, SpringCable, make_connector,
)
from lagrangian_es.util import make_gen

DT = torch.float64
KINDS = ["cable", "spring_cable", "rigid_link"]


def _flat(B=1):
    p = torch.zeros(B, 3, dtype=DT)
    R = torch.eye(3, dtype=DT).expand(B, 3, 3).clone()
    z = torch.zeros(B, 3, dtype=DT)
    return p, R, z, z


def test_registry():
    assert set(CONNECTORS) == set(KINDS)
    assert isinstance(make_connector("cable", length=0.4), Cable)
    with pytest.raises(KeyError):
        make_connector("string", length=1.0)


def test_cable_is_unilateral():
    """A rope pulls but never pushes: the row must engage only when taut.

    A bilateral distance constraint would hold the package UP when slack, which
    is not a small modelling error -- it is a different machine.
    """
    p, R, v, om = _flat(2)
    d = torch.zeros(2, 3, dtype=DT)
    pl = torch.tensor([[0.0, 0.0, -0.60], [0.0, 0.0, -0.30]], dtype=DT)   # taut, slack
    _, resid, act, _ = Cable(length=0.5).rows(p, R, v, om, pl, torch.zeros_like(pl), d)
    assert float(resid[0]) > 0 and float(resid[1]) < 0
    assert float(act[0]) > 0.99, "a taut rope must be fully engaged"
    assert float(act[1]) < 0.01, "a slack rope must be fully released"


def test_rigid_link_is_bilateral_and_three_rows():
    p, R, v, om = _flat(2)
    d = torch.zeros(2, 3, dtype=DT)
    pl = torch.tensor([[0.0, 0.0, -0.6], [0.0, 0.0, 0.3]], dtype=DT)
    J, resid, act, _ = RigidLink().rows(p, R, v, om, pl, torch.zeros_like(pl), d)
    assert J.shape == (2, 3, 9) and resid.shape == (2, 3)
    assert (act == 1.0).all(), "a weld is always enforced, either side"


def test_spring_cable_is_one_sided():
    p, R, v, om = _flat(2)
    d = torch.zeros(2, 3, dtype=DT)
    pl = torch.tensor([[0.0, 0.0, -0.60], [0.0, 0.0, -0.30]], dtype=DT)
    f = SpringCable(length=0.5, k=1000.0).force(p, R, v, om, pl, torch.zeros_like(pl), d)
    assert float(f[0, 2]) > 0, "stretched: pulls the package back up"
    assert float(f[1].abs().max()) == 0.0, "slack: no force at all"
    assert SpringCable(length=0.5).n_rows() == 0, "compliant connectors add no rows"


def test_rest_offsets_match_the_connector():
    assert Cable(length=0.45).rest_offset() == (0.0, 0.0, -0.45)
    assert SpringCable(length=0.3).rest_offset() == (0.0, 0.0, -0.3)
    assert RigidLink().rest_offset() == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# the payload plant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", KINDS)
def test_nominal_state_starts_on_the_constraint_manifold(kind):
    """An inconsistent initial state makes Baumgarte yank the payload into place,
    which visibly jolts the carrier -- so `rest_offset` must be honoured."""
    sysm = make_system("quadrotor_payload", connector=kind)
    x = torch.tensor([[0.0, 0.0, 1.0]], dtype=DT)
    s = sysm.nominal_state(x, torch.zeros_like(x))
    att = s["p"] + torch.einsum("...ij,...j->...i", s["R"], sysm.attach[0].expand(1, 3))
    d = float((s["pl"][0, 0] - att[0]).norm())
    want = 0.0 if kind == "rigid_link" else sysm.cable_length
    assert abs(d - want) < 1e-12


@pytest.mark.parametrize("kind", KINDS)
def test_hover_holds_both_drone_and_package(kind):
    """Thrust equal to the TOTAL weight must hold station.  A controller that
    compensates only the drone's own mass sags from the first step."""
    sysm = make_system("quadrotor_payload", connector=kind)
    x = torch.tensor([[0.0, 0.0, 1.0]], dtype=DT)
    s = sysm.nominal_state(x, torch.zeros_like(x))
    u = torch.zeros(1, 4, dtype=DT)
    u[:, 0] = sysm.total_mass * sysm.g
    for _ in range(1500):
        s = sysm.step(s, u, 0.002)
    assert abs(float(s["p"][0, 2]) - 1.0) < 5e-3, "the vehicle drifted"
    att = s["p"] + torch.einsum("...ij,...j->...i", s["R"], sysm.attach[0].expand(1, 3))
    d = float((s["pl"][0, 0] - att[0]).norm())
    want = 0.0 if kind == "rigid_link" else sysm.cable_length
    tol = 1e-3 if kind != "spring_cable" else 2e-3      # a spring genuinely stretches
    assert abs(d - want) < tol


def test_spring_stretch_matches_the_spring_constant():
    """mg/k, as a spring must."""
    sysm = make_system("quadrotor_payload", connector="spring_cable")
    conn = sysm.connectors[0]
    x = torch.tensor([[0.0, 0.0, 1.0]], dtype=DT)
    s = sysm.nominal_state(x, torch.zeros_like(x))
    u = torch.zeros(1, 4, dtype=DT)
    u[:, 0] = sysm.total_mass * sysm.g
    for _ in range(2000):
        s = sysm.step(s, u, 0.002)
    att = s["p"] + torch.einsum("...ij,...j->...i", s["R"], sysm.attach[0].expand(1, 3))
    stretch = float((s["pl"][0, 0] - att[0]).norm()) - conn.length
    assert abs(stretch - sysm.m_load * sysm.g / conn.k) < 2e-4


def test_taut_cable_is_a_workless_constraint():
    """Free fall with a swinging package: a constraint force does no work, so any
    energy change is the integrator's, not the connector's."""
    sysm = make_system("quadrotor_payload", connector="cable")

    def energy(st):
        ke = (0.5 * sysm.m * (st["v"] ** 2).sum(-1)
              + 0.5 * (st["om"] ** 2 * sysm.Jvec).sum(-1)
              + 0.5 * sysm.m_load * (st["vl"] ** 2).sum(-1).sum(-1))
        pe = sysm.m * sysm.g * st["p"][..., 2] \
            + sysm.m_load * sysm.g * st["pl"][..., 2].sum(-1)
        return float(ke + pe)

    x = torch.tensor([[0.0, 0.0, 5.0]], dtype=DT)
    s = sysm.nominal_state(x, torch.zeros_like(x))
    s["vl"] = torch.tensor([[[0.8, 0.3, 0.0]]], dtype=DT)
    e0 = energy(s)
    for _ in range(1500):
        s = sysm.step(s, torch.zeros(1, 4, dtype=DT), 0.001)
    assert abs(energy(s) - e0) / abs(e0) < 0.01


def test_slack_rope_lets_the_package_fall():
    sysm = make_system("quadrotor_payload", connector="cable")
    x = torch.tensor([[0.0, 0.0, 5.0]], dtype=DT)
    s = sysm.nominal_state(x, torch.zeros_like(x))
    s["pl"] = s["p"][:, None, :] + torch.tensor([[[0.0, 0.0, -0.20]]], dtype=DT)
    u = torch.zeros(1, 4, dtype=DT)
    u[:, 0] = sysm.total_mass * sysm.g
    d0 = float((s["pl"][0, 0] - s["p"][0]).norm())
    for _ in range(60):
        s = sysm.step(s, u, 0.002)
    d1 = float((s["pl"][0, 0] - s["p"][0]).norm())
    assert d1 > d0 + 0.05, "a slack rope must not hold the package up"
    assert d1 <= sysm.cable_length + 0.05


def test_step_is_pure_with_a_payload():
    import copy
    sysm = make_system("quadrotor_payload")
    s = sysm.reset(4, make_gen(0))
    ref = copy.deepcopy(s)
    sysm.step(s, torch.randn(4, 4, dtype=DT), 0.002)
    for k in s:
        assert torch.equal(s[k], ref[k]), f"step mutated {k!r}"


def test_swing_angle_and_shaping_cost():
    sysm = make_system("quadrotor_payload")
    x = torch.tensor([[0.0, 0.0, 1.0]], dtype=DT)
    s = sysm.nominal_state(x, torch.zeros_like(x))
    assert float(sysm.swing_angle(s).max()) < 1e-9      # hanging straight down
    s["pl"] = s["p"][:, None, :] + torch.tensor([[[0.45, 0.0, 0.0]]], dtype=DT)
    assert abs(float(sysm.swing_angle(s)[0, 0]) - torch.pi / 2) < 1e-9
    assert float(sysm.shaping_cost(s)) > 0


def test_gravity_feedforward_includes_the_package():
    sysm = make_system("quadrotor_payload", payload_mass=0.2)
    plain = make_system("quadrotor")
    s = sysm.reset(2, make_gen(0))
    g = sysm.gravity_force(s)
    assert abs(float(g[0, 2]) - sysm.total_mass * sysm.g) < 1e-9
    assert float(g[0, 2]) > plain.m * plain.g, "must lift more than the bare drone"


def test_render_outputs_are_well_formed():
    sysm = make_system("quadrotor_payload")
    s = sysm.reset(3, make_gen(0))
    spec = sysm.render_spec()
    assert spec["dim"] == 3 and len(spec["bodies"]) == 1 + sysm.n_pay
    assert sysm.render_poses(s).shape == (3, 1 + sysm.n_pay, 12)
    assert sysm.render_extras(s)["cable"].shape == (3, sysm.n_pay, 6)
