import torch
from torch.func import jacrev, vmap

from lagrangian_es.systems.so3 import (
    hat, log_so3, orthogonality_error, renormalize, rodrigues, vee,
)

DT = torch.float64


def test_rodrigues_zero_is_identity():
    w = torch.zeros(7, 3, dtype=DT)
    R = rodrigues(w)
    assert torch.allclose(R, torch.eye(3, dtype=DT).expand_as(R), atol=1e-14)


def test_rodrigues_small_angle_matches_series():
    """The `torch.where` Taylor branch must agree with the trig branch across the
    switchover, or the metric sees a discontinuity that is not in the physics."""
    for mag in [1e-6, 1e-5, 1e-4, 1e-3]:
        w = torch.randn(64, 3, dtype=DT)
        w = w / w.norm(dim=-1, keepdim=True) * mag
        R = rodrigues(w)
        assert torch.isfinite(R).all()
        assert orthogonality_error(R).max() < 1e-12
        # first-order: R ~ I + hat(w)
        approx = torch.eye(3, dtype=DT) + hat(w)
        assert (R - approx).abs().max() < 10 * mag**2


def test_orthogonality_after_1000_increments():
    """1000 random composed increments must stay on SO(3) to 1e-5."""
    g = torch.Generator().manual_seed(0)
    R = torch.eye(3, dtype=DT).expand(8, 3, 3).clone()
    for _ in range(1000):
        dw = 0.05 * torch.randn(8, 3, generator=g, dtype=DT)
        R = R @ rodrigues(dw)
    assert orthogonality_error(R).max() < 1e-5
    det = torch.linalg.det(R)
    assert (det - 1.0).abs().max() < 1e-5


def test_vee_hat_roundtrip():
    v = torch.randn(32, 3, dtype=DT)
    assert torch.allclose(vee(hat(v)), v, atol=1e-14)
    S = hat(v)
    assert torch.allclose(S, -S.transpose(-1, -2), atol=1e-14)


def test_log_exp_roundtrip():
    w = torch.randn(64, 3, dtype=DT)
    w = w / w.norm(dim=-1, keepdim=True) * torch.rand(64, 1, dtype=DT) * 3.0
    assert torch.allclose(log_so3(rodrigues(w)), w, atol=1e-9)


def test_log_at_identity_is_finite():
    R = torch.eye(3, dtype=DT).expand(4, 3, 3)
    out = log_so3(R)
    assert torch.isfinite(out).all() and out.abs().max() < 1e-9


def test_renormalize_fixes_drift():
    R = rodrigues(torch.randn(16, 3, dtype=DT))
    R_bad = R + 1e-3 * torch.randn(16, 3, 3, dtype=DT)
    assert orthogonality_error(renormalize(R_bad)).max() < 1e-12


def test_vmap_and_jacrev_safe():
    """so3 sits inside `vmap(jacrev(forward))`; it must trace and stay finite
    even at the small-angle branch point, where a naive sin(t)/t gives NaN."""
    def f(w):
        return rodrigues(w).reshape(-1)

    w = torch.randn(16, 3, dtype=DT) * 1e-9        # deep inside the Taylor branch
    J = vmap(jacrev(f))(w)
    assert J.shape == (16, 9, 3)
    assert torch.isfinite(J).all()

    w2 = torch.randn(16, 3, dtype=DT)
    assert torch.isfinite(vmap(jacrev(f))(w2)).all()
    assert torch.isfinite(vmap(jacrev(lambda x: vee(hat(x))))(w2)).all()


def test_hat_batch_rank_agnostic():
    """Called both batched (rollout) and unbatched (inside vmap)."""
    v1 = torch.randn(3, dtype=DT)
    assert hat(v1).shape == (3, 3)
    assert hat(torch.randn(5, 3, dtype=DT)).shape == (5, 3, 3)
    assert hat(torch.randn(2, 5, 3, dtype=DT)).shape == (2, 5, 3, 3)
