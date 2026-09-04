"""Batched SO(3) primitives.

Every function here is written to survive `vmap` and `jacrev`:
no in-place writes, no Python branches on tensor values, and every division has
a clamped denominator so that both sides of a `torch.where` stay finite (a NaN
in the *unused* branch still poisons the gradient).

Shapes: a leading `...` means any number of batch dims, including none.
"""
from __future__ import annotations

import torch
from torch import Tensor

_SMALL = 1e-8      # theta^2 below this uses the Taylor branch
_EPS = 1e-24       # floor for the clamped theta^2 under the sqrt


def hat(v: Tensor) -> Tensor:
    """so(3) generator: [..., 3] -> skew-symmetric [..., 3, 3]."""
    z = torch.zeros_like(v[..., 0])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([z, -w, y], dim=-1)
    row1 = torch.stack([w, z, -x], dim=-1)
    row2 = torch.stack([-y, x, z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def vee(S: Tensor) -> Tensor:
    """Inverse of `hat` on the skew part: [..., 3, 3] -> [..., 3]."""
    return torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)


def rodrigues(w: Tensor) -> Tensor:
    """Exponential map: axis-angle [..., 3] -> rotation [..., 3, 3].

    R = I + a K + b K^2 with K = hat(w), a = sin(t)/t, b = (1-cos t)/t^2.
    The small-angle limits (a -> 1, b -> 1/2) are selected with `torch.where`
    on a *clamped* argument, never with a branch, so this is vmap/jacrev safe.
    """
    th2 = (w * w).sum(dim=-1, keepdim=True)                 # [..., 1]
    th2_safe = th2.clamp_min(_EPS)
    th = torch.sqrt(th2_safe)
    small = th2 < _SMALL
    a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / th2_safe)
    K = hat(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return eye + a[..., None] * K + b[..., None] * (K @ K)


def log_so3(R: Tensor) -> Tensor:
    """Logarithm map: [..., 3, 3] -> axis-angle [..., 3].  Valid for |angle| < pi."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_t = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    th = torch.acos(cos_t)[..., None]                       # [..., 1]
    sin_t = torch.sin(th)
    skew = vee(R - R.transpose(-1, -2))                     # 2 sin(t) * axis
    small = th.squeeze(-1) < 1e-4
    scale = torch.where(
        small[..., None],
        torch.full_like(th, 0.5),                           # lim t/(2 sin t) = 1/2
        th / (2.0 * sin_t.clamp_min(1e-12)),
    )
    return scale * skew


def renormalize(R: Tensor) -> Tensor:
    """Re-orthonormalize by modified Gram-Schmidt on the columns.

    Cheaper and more vmap-friendly than an SVD projection.  Not used on the hot
    integration path (float64 composition drifts by ~1e-15 per step); exposed for
    long-horizon replays and for tests.
    """
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    b0 = c0 / c0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    c1 = c1 - (b0 * c1).sum(dim=-1, keepdim=True) * b0
    b1 = c1 / c1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b2 = torch.cross(b0, b1, dim=-1)
    return torch.stack([b0, b1, b2], dim=-1)


def orthogonality_error(R: Tensor) -> Tensor:
    """max |R^T R - I| over the trailing 3x3 block.  [..., 3, 3] -> [...]."""
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    return (R.transpose(-1, -2) @ R - eye).abs().amax(dim=(-1, -2))
