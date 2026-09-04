import torch
from torch.func import jacrev, vmap

from lagrangian_es.systems import make_system
from lagrangian_es.trainables import make_trainable

DT = torch.float64


def _tr(**kw):
    return make_trainable("energy_shaping", make_system("quadrotor"), **kw)


def test_potential_is_nonnegative_with_a_unique_minimum_at_the_goal():
    """The claim that every individual -- including a random one -- is an
    energy-shaping controller whose equilibrium is the goal, tested on 1000
    random genomes rather than on the prior."""
    tr = _tr()
    g = torch.Generator().manual_seed(0)
    TH = 2.0 * torch.randn(1000, tr.dim, generator=g, dtype=DT)
    zero = torch.zeros(1000, 3, dtype=DT)

    assert tr.potential(TH, zero).abs().max() < 1e-12          # V(0) = 0
    assert tr.grad_potential(TH, zero).abs().max() < 1e-12     # and it is a critical point

    for scale in [1e-3, 1e-1, 1.0, 10.0]:
        e = scale * torch.randn(1000, 3, generator=g, dtype=DT)
        V = tr.potential(TH, e)
        assert (V >= 0).all(), f"V < 0 at scale {scale}: min {V.min():.3e}"
        assert torch.isfinite(V).all()


def test_potential_is_strictly_positive_away_from_the_goal():
    """Nonnegativity alone would be satisfied by V == 0.  With w != 0 and A
    invertible the minimum must actually be unique."""
    tr = _tr()
    th = tr.init()
    for scale in [1e-2, 1.0, 10.0]:
        e = scale * torch.nn.functional.normalize(torch.randn(256, 3, dtype=DT), dim=-1)
        assert (tr.potential(th.expand(256, -1), e) > 0).all()


def test_grad_potential_is_bounded_as_error_diverges():
    """Pseudo-Huber bowls: the commanded force saturates through the potential's
    own geometry.  This is what makes actuator limits a property of the search
    space rather than a clip applied after the fact."""
    tr = _tr()
    g = torch.Generator().manual_seed(1)
    TH = torch.randn(64, tr.dim, generator=g, dtype=DT)

    # Directions are drawn ONCE and reused: the saturation value depends on the
    # direction, so redrawing per scale compares different limits.
    dirs = torch.nn.functional.normalize(torch.randn(64, 3, generator=g, dtype=DT), dim=-1)

    mags = []
    for scale in [1e0, 1e2, 1e4, 1e6, 1e8]:
        m = tr.grad_potential(TH, scale * dirs).norm(dim=-1)
        assert torch.isfinite(m).all()
        mags.append(m)

    # Along a fixed ray the magnitude must converge to the asymptote
    # ||sum_k w_k^2 A_k^T A_k d / ||A_k d|| ||, so successive scales stop moving.
    rel = ((mags[-1] - mags[-2]).abs() / mags[-2].clamp_min(1e-12)).max()
    assert rel < 1e-3, f"not converged along a fixed ray: rel change {rel:.3e}"

    # ... and the bound is uniform: a 1e8-fold increase in ||e|| must not move it.
    growth = (mags[-1] / mags[0].clamp_min(1e-12)).max()
    assert growth < 3.0, f"grad V grew {growth:.1f}x over 8 decades of ||e||"

    # The analytic bound sum_k w_k^2 sigma_max(A_k) must actually hold.
    w2, A, _, _ = tr.unpack(TH)
    bound = (w2 * torch.linalg.matrix_norm(A, ord=2)).sum(-1)
    assert (mags[-1] <= bound + 1e-8).all()


def test_grad_matches_autograd_of_the_potential():
    """The closed-form gradient is used because `forward` must be cheap, but it
    still has to be the gradient of the potential it claims to differentiate."""
    tr = _tr()
    g = torch.Generator().manual_seed(2)
    th = torch.randn(tr.dim, generator=g, dtype=DT)
    e = torch.randn(32, 3, generator=g, dtype=DT)
    auto = vmap(jacrev(lambda x: tr.potential(th, x)))(e)
    assert torch.allclose(auto, tr.grad_potential(th.expand(32, -1), e), atol=1e-10)


def test_damping_is_psd_by_construction():
    tr = _tr()
    g = torch.Generator().manual_seed(3)
    TH = 3.0 * torch.randn(1000, tr.dim, generator=g, dtype=DT)
    eig = torch.linalg.eigvalsh(tr.damping(TH))
    assert (eig >= -1e-10).all(), f"min eig {eig.min():.3e}"
    Kd = tr.damping(TH)
    assert torch.allclose(Kd, Kd.transpose(-1, -2), atol=1e-12)


def test_stiffness_is_psd_and_is_the_hessian_at_the_goal():
    tr = _tr()
    g = torch.Generator().manual_seed(4)
    th = torch.randn(tr.dim, generator=g, dtype=DT)
    K = tr.stiffness(th)
    assert torch.allclose(K, K.T, atol=1e-12)
    assert (torch.linalg.eigvalsh(K) >= -1e-10).all()
    hess = jacrev(lambda x: tr.grad_potential(th, x))(torch.zeros(3, dtype=DT))
    assert torch.allclose(hess, K, atol=1e-9)


def test_prior_matches_the_specified_physical_numbers():
    """w = 1, A_k = I, D = 1.2 I, kR = 0.25, kW = 0.10 -> stiffness 3 N/m and
    omega_n ~ 2.4 rad/s at 0.5 kg.  These are the prototype's numbers; drifting
    off them changes the difficulty of the whole benchmark."""
    tr = _tr()
    th = tr.init()
    K = tr.stiffness(th)
    assert torch.allclose(K, 3.0 * torch.eye(3, dtype=DT), atol=1e-12)
    assert torch.allclose(tr.damping(th), 1.44 * torch.eye(3, dtype=DT), atol=1e-12)
    d = tr.describe(th)
    assert abs(d["wn_pos"] - 2.449) < 1e-2
    assert abs(d["kR"] - 0.0625) < 1e-12 and abs(d["kW"] - 0.01) < 1e-12


def test_invariants_reported_for_the_conformance_contract():
    tr = _tr()
    inv = tr.invariants(tr.init())
    assert {"V_at_goal", "V_min", "Kd_eigs"} <= set(inv)
    assert float(inv["V_at_goal"]) == 0.0
    assert float(inv["V_min"]) >= 0.0
    assert (inv["Kd_eigs"] >= -1e-12).all()


def test_describe_exposes_bowl_anisotropy():
    """Three identical A_k is a valid but boring optimum; `describe` has to make
    that visible or you cannot tell whether NK = 1 would do just as well."""
    tr = _tr()
    iso = tr.describe(tr.init())
    assert abs(iso["K_aniso"] - 1.0) < 1e-9      # the prior IS isotropic

    th = tr.init().clone()
    th[tr._i_A[0]] = 3.0                          # stretch one bowl along one axis
    assert tr.describe(th)["K_aniso"] > 2.0


def test_unpack_is_batch_rank_agnostic():
    tr = _tr()
    for shape in [(), (5,), (2, 5)]:
        th = torch.randn(shape + (tr.dim,), dtype=DT)
        w2, A, D, phi = tr.unpack(th)
        assert w2.shape == shape + (tr.NK,)
        assert A.shape == shape + (tr.NK, 3, 3)
        assert D.shape == shape + (3, 3)
        assert phi.shape == shape + (6,)
        assert (w2 >= 0).all()
