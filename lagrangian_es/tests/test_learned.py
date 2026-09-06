"""`LearnedShaping` -- the shape is learned, the certificate is structural."""
import pytest
import torch

from lagrangian_es.trainables.learned import LearnedShaping

DT = torch.float64


def _rig(B=6, n_obs=12, d=3, seed=0):
    t = LearnedShaping(d, n_obs=n_obs)
    th = t.init(dtype=DT)
    g = torch.Generator().manual_seed(seed)
    obs = {"range": torch.rand(B, n_obs, generator=g, dtype=DT) * 4.0}
    e = torch.randn(B, d, generator=g, dtype=DT)
    v = torch.randn(B, d, generator=g, dtype=DT)
    return t, th, e, v, torch.zeros(B, d, dtype=DT), obs


def _scramble(t, th, seed=3, scale=0.6):
    """A genome the search might actually reach, not just the prior."""
    g = torch.Generator().manual_seed(seed)
    return th + scale * torch.randn(th.shape, generator=g, dtype=DT)


def test_potential_is_nonnegative_for_any_weights():
    """V = ||h - h(0)||^2, so PSD holds by the FORM.  A learned potential that
    could go negative would let the closed loop pump energy and the whole
    stability argument would be a matter of luck in the weights."""
    t, th, e, v, x, obs = _rig()
    for s in range(6):
        w = _scramble(t, th, seed=s, scale=1.5)
        assert torch.all(t.potential(w, e, v, x, obs) >= 0.0)


def test_goal_is_stationary_for_any_weights():
    """Value AND gradient vanish at e = 0, so the goal is the closed-loop
    equilibrium however the weights come out -- the property `MLPPolicy`
    deliberately gives up."""
    t, th, e, v, x, obs = _rig()
    z = torch.zeros_like(e)
    zv = torch.zeros_like(v)
    for s in range(6):
        w = _scramble(t, th, seed=s, scale=1.5)
        assert float(t.potential(w, z, zv, x, obs).abs().max()) == 0.0
        assert float(t.grad_potential(w, z, zv, x, obs).abs().max()) == 0.0


def test_dissipation_head_can_only_remove_energy():
    t, th, e, v, x, obs = _rig()
    z = torch.zeros_like(e)
    for s in range(6):
        w = _scramble(t, th, seed=s, scale=1.5)
        force = -t.grad_potential(w, z, v, x, obs)      # at the goal: R only
        assert float((force * v).sum(-1).max()) <= 1e-12


def test_analytic_jacobian_matches_autograd():
    """The trunk's d h / d e is written out by hand so the term composes under
    vmap/jacrev without nesting transforms; if it drifts from the true gradient
    the force is silently wrong and nothing else would catch it."""
    t, th, e, v, x, obs = _rig(B=1)
    w = _scramble(t, th, seed=1)
    ee = e.clone().requires_grad_(True)
    V = t.potential(w, ee, v, x, obs).sum()
    (auto,) = torch.autograd.grad(V, ee)
    hand = t.grad_potential(w, e, torch.zeros_like(v), x, obs)
    assert torch.allclose(auto, hand, atol=1e-9), f"{auto} vs {hand}"


def test_prior_is_the_quadratic_bowl():
    """Without the identity skip a small-weight net is nearly flat and has no
    goal attraction, so the comparison against the hand-designed stack would be
    measuring the warm start instead of the parameterisation."""
    t = LearnedShaping(3)
    th = t.init(dtype=DT)
    obs = {"range": torch.full((3, 12), 4.0, dtype=DT)}
    e = torch.tensor([[0.5, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], dtype=DT)
    z = torch.zeros(3, 3, dtype=DT)
    V = t.potential(th, e, z, z, obs)
    assert torch.allclose(V, (e * e).sum(-1), rtol=0.05)


def test_is_silent_without_observations():
    t, th, e, v, x, obs = _rig()
    assert torch.equal(t.grad_potential(th, e, v, x, None), torch.zeros_like(e))
    assert torch.equal(t.grad_potential(th, e, v, x, {}), torch.zeros_like(e))


def test_certificate_claims_what_the_form_guarantees():
    t = LearnedShaping(3)
    c = t.certificate(t.init(dtype=DT))
    assert c["psd"] and c["zero_at_goal"] and c["learned"]
    assert c["params"] == t.dim
