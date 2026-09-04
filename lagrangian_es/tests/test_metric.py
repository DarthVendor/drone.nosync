import pytest
import torch

from lagrangian_es.config import RolloutCfg
from lagrangian_es.metric import identity_preconditioner, physics_metric
from lagrangian_es.rollout import Rollout
from lagrangian_es.systems import make_system
from lagrangian_es.tasks import make_task
from lagrangian_es.trainables import make_trainable
from lagrangian_es.util import make_gen

DT = torch.float64


@pytest.fixture(scope="module")
def rig():
    system = make_system("quadrotor")
    tr = make_trainable("energy_shaping", system)
    task = make_task("waypoint_pair", system)
    cfg = RolloutCfg(ep_steps=120, n_eps=2)
    roll = Rollout(system, tr, task, cfg)
    goals = task.sample(4, make_gen(11))
    return system, tr, task, cfg, roll, goals


def _metric(rig, **kw):
    system, tr, task, cfg, roll, goals = rig
    return physics_metric(system, tr, tr.init(), goals, cfg, seed=5, roll=roll, **kw)


def test_G_is_symmetric_and_psd(rig):
    m = _metric(rig)
    assert torch.allclose(m.G, m.G.T, atol=1e-6)
    assert (torch.linalg.eigvalsh(m.G) >= -1e-9).all()
    assert (m.eigs >= 0).all()
    assert torch.isfinite(m.G).all()


def test_G_is_normalized_so_sigma_keeps_its_meaning(rig):
    """G is divided by mean(diag(G)) so a fixed sigma means the same thing from
    one generation to the next even as the controller's scale drifts."""
    m = _metric(rig)
    assert abs(float(torch.diagonal(m.G).mean()) - 1.0) < 1e-9


def test_P_is_well_conditioned_after_the_ridge(rig):
    m = _metric(rig, ridge=1e-3)
    assert torch.allclose(m.P, m.P.T, atol=1e-9)
    eig = torch.linalg.eigvalsh(m.P)
    assert (eig > 0).all(), "preconditioner must be positive definite"
    assert torch.isfinite(m.P).all()
    assert float(eig.max() / eig.min()) < 1e6


def test_P_preserves_step_length_and_changes_only_shape(rig):
    """The rescale sets mean(lambda^-1/2) = 1, so whitening redirects the step
    without silently also making it bigger -- otherwise the ablation would be
    confounded by an effective step-size change."""
    m = _metric(rig)
    lam = torch.linalg.eigvalsh(m.P)
    assert abs(float(lam.mean()) - 1.0) < 1e-9


def test_whitened_and_isotropic_differ(rig):
    """Section 7's first silent failure mode: if the ridge dominates, P -> I and
    the whitened arm quietly becomes the baseline -- an arm compared to itself."""
    m = _metric(rig, ridge=1e-3)
    assert m.dist_from_identity > 0.5, f"||P - I|| = {m.dist_from_identity}"
    assert m.cond > 10.0


def test_a_large_ridge_collapses_P_toward_identity(rig):
    """The failure mode is real and this is what it looks like, so the diagnostic
    that detects it is doing something."""
    near = _metric(rig, ridge=1e6)
    assert near.dist_from_identity < 1e-3


def test_identity_preconditioner_is_exactly_the_baseline():
    m = identity_preconditioner(45, DT)
    assert torch.equal(m.P, torch.eye(45, dtype=DT))
    assert torch.equal(m.P_inv, torch.eye(45, dtype=DT))
    assert m.dist_from_identity == 0.0 and m.cond == 1.0


def test_P_inv_actually_inverts_P(rig):
    m = _metric(rig)
    n = m.P.shape[0]
    assert torch.allclose(m.P @ m.P_inv, torch.eye(n, dtype=DT), atol=1e-8)


def test_saturation_is_reported(rig):
    """jacrev through a saturated clamp returns zero, so a genome living at f_max
    yields a rank-deficient G in exactly the directions that matter -- and the
    ridge hides it.  The fraction has to be visible."""
    m = _metric(rig)
    assert 0.0 <= m.saturation <= 1.0
    assert 0.0 < m.rank_frac <= 1.0


def test_no_grad_leak_from_the_rollout(rig):
    """The method is gradient-free with respect to the dynamics.  Nothing may
    differentiate through the integrator, the liveness logic or the cost."""
    system, tr, task, cfg, roll, goals = rig
    theta = tr.init().clone().requires_grad_(True)
    with torch.enable_grad():
        res = roll.run(theta[None], goals, seed=3)
        assert not res.fitness.requires_grad
        assert res.fitness.grad_fn is None
        trace = roll.trace(theta[None], goals, seed=3)
        for k, v in trace.states.items():
            assert not v.requires_grad, f"trace state {k!r} carries a graph"
        assert not trace.us.requires_grad


def test_metric_is_seed_reproducible(rig):
    a, b = _metric(rig), _metric(rig)
    assert torch.equal(a.G, b.G) and torch.equal(a.P, b.P)


def test_dense_and_diagonal_paths_agree_on_a_diagonal_mass(rig):
    """The only place the weak and strong forms of the claim differ is one
    einsum.  Feeding a diagonal M through the dense path must reproduce the
    diagonal path exactly, or the arm's numbers are not comparable."""
    system, tr, task, cfg, roll, goals = rig
    from torch.func import jacrev, vmap
    trace = roll.trace(tr.init()[None], goals, 5)
    s = {k: v[10] for k, v in trace.states.items()}
    gl = trace.goals[10]
    J = vmap(jacrev(tr.forward), in_dims=(None, 0, 0))(tr.init(), s, gl)
    Minv = system.inv_mass(s)
    diag_path = torch.einsum("sid,si,sie->de", J, Minv, J)
    dense_path = torch.einsum("sid,sij,sje->de", J, torch.diag_embed(Minv), J)
    assert torch.allclose(diag_path, dense_path, atol=1e-10)


def test_arm_metric_is_state_dependent_but_quadrotor_metric_is_not():
    """The scope note of section 1, as an assertion.

    For a rigid body M is constant in the body frame, so on the quadrotor the
    trajectory dependence of G enters ONLY through the controller Jacobian.  The
    stronger claim -- that G tracks a configuration-dependent inertia no single
    global covariance can represent -- requires a plant where M(q) varies, and
    the arm is that plant.
    """
    g = torch.Generator().manual_seed(0)

    quad = make_system("quadrotor")
    sq = quad.reset(128, g)
    assert torch.equal(quad.inv_mass(sq)[0], quad.inv_mass(sq)[-1]), \
        "quadrotor M^-1 must be constant"

    arm = make_system("two_link_arm")
    sa = arm.reset(128, g)
    sa["q"] = sa["q"] + 3.0 * torch.randn(128, 2, generator=g, dtype=DT)
    Ma = arm.inv_mass(sa)
    assert Ma.shape == (128, 2, 2) and arm.dense_mass
    spread = (Ma.reshape(128, -1).std(0) / Ma.reshape(128, -1).abs().mean(0)).max()
    assert float(spread) > 0.1, "arm M^-1 should vary substantially with configuration"
    cond = arm.mass_condition(sa)
    assert float(cond.max() / cond.min()) > 3.0
