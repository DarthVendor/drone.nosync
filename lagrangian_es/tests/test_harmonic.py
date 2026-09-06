"""The harmonic obstacle field, and the charge memory that makes it harmonic."""
import math

import pytest
import torch

from lagrangian_es.config import Config, ESCfg, RolloutCfg
from lagrangian_es.es import build
from lagrangian_es.rollout import ChargeMemory, Rollout
from lagrangian_es.sensors.range_sensor import RangeSensor
from lagrangian_es.trainables import make_trainable
from lagrangian_es.trainables.harmonic import HarmonicField
from lagrangian_es.trainables.terms import DissipationTerm, GoalBowl

DT = torch.float64


# --- charge memory ----------------------------------------------------------
def test_charge_memory_is_a_ring_and_masks_misses():
    m = ChargeMemory(slots=6)
    a = torch.arange(2 * 3 * 3, dtype=DT).reshape(2, 3, 3)
    m.write(a, torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=DT))
    p, w = m.read()
    assert p.shape == (2, 6, 3) and w.shape == (2, 6)
    assert torch.equal(p[:, :3], a)
    assert float(w[0, 1]) == 0.0 and float(w[1, 2]) == 0.0
    # wrap: three more writes of 3 must overwrite the oldest, not grow
    for _ in range(3):
        m.write(a + 100.0, torch.ones(2, 3, dtype=DT))
    p2, _ = m.read()
    assert p2.shape == (2, 6, 3)
    assert torch.all(p2 >= 100.0), "ring did not wrap over the stale entries"


def test_charge_memory_resets_between_episodes():
    m = ChargeMemory(slots=4)
    m.write(torch.ones(1, 2, 3, dtype=DT), torch.ones(1, 2, dtype=DT))
    m.reset()
    assert m.read() == (None, None)


def test_charge_memory_broadcasts_a_held_reading_to_the_state_batch():
    """A strided sensor hands back a narrower batch than the state's; the weights
    have to line up with the points or the write silently misaligns."""
    m = ChargeMemory(slots=8)
    m.write(torch.zeros(5, 2, 3, dtype=DT), torch.ones(1, 2, dtype=DT))
    _, w = m.read()
    assert w.shape == (5, 8)


# --- the field --------------------------------------------------------------
def _field(**kw):
    return HarmonicField(3, **kw)


def _V(term, th, x, p, m):
    return term.raw(th, x, p, m)


def test_harmonic_field_laplacian_vanishes_away_from_its_charges():
    """The whole point.  lap V = tr(Hess V), so a critical point of a harmonic V
    has eigenvalues summing to zero and cannot be a local minimum.  The barrier
    it replaces is a field of SLIDING sources and carries a large positive
    Laplacian; frozen charges do not.
    """
    t = _field(soft0=0.02)
    th = t.init(dtype=DT)
    p = torch.tensor([[[1.5, 0.3, 1.2], [1.5, -0.3, 1.2], [1.7, 0.0, 1.2]]],
                     dtype=DT)
    m = torch.ones(1, 3, dtype=DT)
    h = 1e-3
    x0 = torch.tensor([[0.6, 0.05, 1.2]], dtype=DT)
    lap = 0.0
    for k in (0, 1):                      # the 2-D plane the log kernel lives in
        for sgn in (+1, -1):
            d = torch.zeros_like(x0); d[0, k] = sgn * h
            lap += float(_V(t, th, x0 + d, p, m)[0])
    lap -= 4 * float(_V(t, th, x0, p, m)[0])
    lap /= h ** 2
    assert abs(lap) < 0.5, f"field is not harmonic: lap = {lap}"


def test_harmonic_field_pushes_away_from_charges():
    t = _field()
    th = t.init(dtype=DT)
    p = torch.tensor([[[1.0, 0.0, 1.2]]], dtype=DT)
    m = torch.ones(1, 1, dtype=DT)
    x = torch.tensor([[0.6, 0.0, 1.2]], dtype=DT)
    _, g = t.raw(th, x, p, m)
    assert float(g[0, 0]) > 0.0, "grad must point toward the charge (force away)"


def test_harmonic_field_is_silent_without_a_charge_memory():
    t = _field()
    th = t.init(dtype=DT)
    e = torch.ones(2, 3, dtype=DT)
    z = torch.zeros(2, 3, dtype=DT)
    assert torch.equal(t.grad_potential(th, e, z, z, None), z)
    assert torch.equal(t.grad_potential(th, e, z, z, {}), z)


def test_harmonic_field_vanishes_inside_the_goal_ball():
    """goal_gate is exactly zero inside r_goal, so near the goal V_d is the bowl
    alone -- which is what keeps the goal the unique minimum with bounded force."""
    t = _field()
    th = t.init(dtype=DT)
    p = torch.tensor([[[0.2, 0.0, 1.2]]], dtype=DT)
    m = torch.ones(1, 1, dtype=DT)
    x = torch.zeros(1, 3, dtype=DT)
    e = torch.tensor([[0.05, 0.0, 0.0]], dtype=DT)     # well inside r_goal
    obs = {"charges": p, "charge_w": m}
    assert torch.allclose(t.grad_potential(th, e, x, x, obs),
                          torch.zeros(1, 3, dtype=DT))
    assert float(t.potential(th, e, x, x, obs)) == 0.0


def test_harmonic_field_softening_is_bounded_so_it_cannot_be_switched_off():
    t = _field()
    for v in (1e3, 1e12, -1e12):
        _, soft = t._params(torch.tensor([1.0, v], dtype=DT))
        assert t.soft_lo <= float(soft) <= t.soft_hi


def test_harmonic_field_certificate_claims_hold():
    t = _field()
    c = t.certificate(t.init(dtype=DT))
    assert c["psd"] and c["zero_at_goal"] and c["bounded_grad"] and c["harmonic"]


# --- wired through the rollout ---------------------------------------------
def test_rollout_only_builds_a_charge_memory_when_a_term_asks():
    cfg = Config(system="quadrotor_nav", trainable="energy_shaping",
                 task="waypoint_pair", environment="pillars", sensors=(),
                 gating="arrival", rollout=RolloutCfg(n_eps=1))
    sysm, _, task = build(cfg)
    sen = RangeSensor(sysm, n_beams=8, sigma=0.0)
    plain = make_trainable("energy_shaping", sysm,
                           terms=[GoalBowl(3), DissipationTerm(3)])
    assert Rollout(sysm, plain, task, cfg.rollout, (sen,)).charge_mem is None
    harm = make_trainable("energy_shaping", sysm,
                          terms=[GoalBowl(3), DissipationTerm(3),
                                 HarmonicField(3)])
    assert Rollout(sysm, harm, task, cfg.rollout, (sen,)).charge_mem is not None


def test_charges_land_on_the_obstacle_surface():
    """hit = x - d*J reconstructs the world point a beam returned from; if the
    arithmetic is wrong the charges sit somewhere else entirely and the field is
    repelling from nothing."""
    cfg = Config(system="quadrotor_nav", trainable="energy_shaping",
                 task="waypoint_pair", environment="pillars", sensors=(),
                 gating="arrival", rollout=RolloutCfg(n_eps=1))
    sysm, _, task = build(cfg)
    sen = RangeSensor(sysm, n_beams=16, sigma=0.0)
    tr = make_trainable("energy_shaping", sysm,
                        terms=[GoalBowl(3), DissipationTerm(3), HarmonicField(3)])
    roll = Rollout(sysm, tr, task, cfg.rollout, (sen,))
    s = sysm.reset(1, torch.Generator().manual_seed(0))
    c = torch.full((1, 6, 2), 90.0, dtype=DT)
    r = torch.full((1, 6), 0.05, dtype=DT)
    c[:, 0] = torch.tensor([1.2, 0.0], dtype=DT)
    r[:, 0] = 0.40
    s["pillars/c"], s["pillars/r"] = c, r
    s["p"] = torch.tensor([[0.0, 0.0, 1.2]], dtype=DT)
    s["v"] = torch.zeros(1, 3, dtype=DT)
    g = torch.Generator().manual_seed(1)
    roll._prime(s, g)
    obs = roll._observe(s, g, 0)
    p, w = obs["charges"], obs["charge_w"]
    live = w[0] > 0.5
    assert int(live.sum()) > 0, "no beam returned a hit; test is vacuous"
    d = (p[0][live][:, :2] - torch.tensor([1.2, 0.0], dtype=DT)).norm(dim=-1)
    assert torch.allclose(d, torch.full_like(d, 0.40), atol=2e-2), \
        f"charges are not on the pillar surface: radii {d.tolist()}"
