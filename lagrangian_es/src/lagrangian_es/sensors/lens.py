"""Camera projection models.

`Pinhole` for a rectilinear lens, `DoubleSphere` (Usenko et al. 2018) for fisheye.
Double sphere is the fisheye model to use here because it is differentiable,
closed-form invertible, and needs no iterative undistortion -- and wide FOV is
worth having on its own, because it makes the field-of-view barrier term less
binding.

**Do not undistort before servoing.** It costs latency and is wasted work when
the Jacobian can be pushed through the distortion instead.  The pullback chain is
J = J_lens . J_project . J_pose, and `jacobian` below is the first factor.

Every Jacobian here is closed form rather than autograd, for the same reason the
potential's gradient is: `forward` is vmapped over a whole population every step,
and `jacrev` differentiates it with respect to theta, not to the pixel.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

EPS = 1e-6


class Lens(ABC):
    """Maps camera-frame points to pixels and back."""

    name: str = "lens"

    def __init__(self, fx: float = 240.0, fy: float = 240.0,
                 cx: float = 160.0, cy: float = 120.0,
                 width: int = 320, height: int = 240):
        self.fx, self.fy, self.cx, self.cy = map(float, (fx, fy, cx, cy))
        self.width, self.height = int(width), int(height)

    @abstractmethod
    def project(self, Xc: Tensor) -> Tensor:
        """[..., 3] camera-frame -> [..., 2] pixels."""

    @abstractmethod
    def unproject(self, uv: Tensor) -> Tensor:
        """[..., 2] pixels -> [..., 3] unit ray in the camera frame."""

    @abstractmethod
    def jacobian(self, Xc: Tensor) -> Tensor:
        """d(pixel)/d(camera-frame point), [..., 2, 3].  Closed form."""

    def in_view(self, Xc: Tensor, uv: Tensor) -> Tensor:
        """[...] bool -- in front of the camera and inside the image."""
        return ((Xc[..., 2] > EPS)
                & (uv[..., 0] >= 0) & (uv[..., 0] <= self.width)
                & (uv[..., 1] >= 0) & (uv[..., 1] <= self.height))

    def describe(self) -> dict:
        return {"lens": type(self).__name__, "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height}


class Pinhole(Lens):
    name = "pinhole"

    def project(self, Xc: Tensor) -> Tensor:
        z = Xc[..., 2].clamp_min(EPS)
        return torch.stack([self.fx * Xc[..., 0] / z + self.cx,
                            self.fy * Xc[..., 1] / z + self.cy], dim=-1)

    def unproject(self, uv: Tensor) -> Tensor:
        mx = (uv[..., 0] - self.cx) / self.fx
        my = (uv[..., 1] - self.cy) / self.fy
        r = torch.stack([mx, my, torch.ones_like(mx)], dim=-1)
        return r / r.norm(dim=-1, keepdim=True).clamp_min(EPS)

    def jacobian(self, Xc: Tensor) -> Tensor:
        x, y = Xc[..., 0], Xc[..., 1]
        z = Xc[..., 2].clamp_min(EPS)
        zero = torch.zeros_like(x)
        row_u = torch.stack([self.fx / z, zero, -self.fx * x / (z * z)], dim=-1)
        row_v = torch.stack([zero, self.fy / z, -self.fy * y / (z * z)], dim=-1)
        return torch.stack([row_u, row_v], dim=-2)


class DoubleSphere(Lens):
    """Usenko et al. (2018).

        d1 = |X|,  d2 = sqrt(x^2 + y^2 + (xi*d1 + z)^2)
        w  = alpha*d2 + (1-alpha)*(xi*d1 + z)
        u  = fx*x/w + cx,   v = fy*y/w + cy

    `w` is floored, because the model is singular behind the sphere and a NaN
    there poisons the whole batch through vmap.
    """

    name = "double_sphere"

    def __init__(self, fx: float = 130.0, fy: float = 130.0, cx: float = 160.0,
                 cy: float = 120.0, xi: float = -0.18, alpha: float = 0.59, **kw):
        super().__init__(fx=fx, fy=fy, cx=cx, cy=cy, **kw)
        self.xi, self.alpha = float(xi), float(alpha)

    def _w(self, Xc: Tensor):
        x, y, z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
        d1 = torch.sqrt((Xc * Xc).sum(-1).clamp_min(EPS * EPS))
        k = self.xi * d1 + z
        d2 = torch.sqrt((x * x + y * y + k * k).clamp_min(EPS * EPS))
        w = (self.alpha * d2 + (1.0 - self.alpha) * k)
        return d1, k, d2, w

    def project(self, Xc: Tensor) -> Tensor:
        _, _, _, w = self._w(Xc)
        w = w.clamp_min(EPS)
        return torch.stack([self.fx * Xc[..., 0] / w + self.cx,
                            self.fy * Xc[..., 1] / w + self.cy], dim=-1)

    def unproject(self, uv: Tensor) -> Tensor:
        mx = (uv[..., 0] - self.cx) / self.fx
        my = (uv[..., 1] - self.cy) / self.fy
        r2 = mx * mx + my * my
        a = self.alpha
        disc = (1.0 - (2.0 * a - 1.0) * r2).clamp_min(0.0)
        mz = (1.0 - a * a * r2) / (a * torch.sqrt(disc) + (1.0 - a)).clamp_min(EPS)
        num = mz * self.xi + torch.sqrt(
            (mz * mz + (1.0 - self.xi * self.xi) * r2).clamp_min(0.0))
        den = (mz * mz + r2).clamp_min(EPS)
        scale = (num / den)[..., None]
        v = scale * torch.stack([mx, my, mz], dim=-1)
        v = v - torch.tensor([0.0, 0.0, self.xi], dtype=uv.dtype, device=uv.device)
        return v / v.norm(dim=-1, keepdim=True).clamp_min(EPS)

    def jacobian(self, Xc: Tensor) -> Tensor:
        x, y, z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
        d1, k, d2, w = self._w(Xc)
        d1 = d1.clamp_min(EPS); d2 = d2.clamp_min(EPS); w = w.clamp_min(EPS)
        e3 = torch.zeros_like(Xc); e3[..., 2] = 1.0
        dd1 = Xc / d1[..., None]
        dk = self.xi * dd1 + e3
        xy0 = torch.stack([x, y, torch.zeros_like(z)], dim=-1)
        dd2 = (xy0 + k[..., None] * dk) / d2[..., None]
        dw = self.alpha * dd2 + (1.0 - self.alpha) * dk

        e1 = torch.zeros_like(Xc); e1[..., 0] = 1.0
        e2 = torch.zeros_like(Xc); e2[..., 1] = 1.0
        inv_w = (1.0 / w)[..., None]
        row_u = self.fx * (e1 * inv_w - (x / (w * w))[..., None] * dw)
        row_v = self.fy * (e2 * inv_w - (y / (w * w))[..., None] * dw)
        return torch.stack([row_u, row_v], dim=-2)

    def describe(self) -> dict:
        d = super().describe()
        d.update(xi=self.xi, alpha=self.alpha)
        return d


def interaction_matrix(xn: Tensor, Z: Tensor) -> Tensor:
    """Point-feature interaction matrix in NORMALIZED coordinates, [..., 2, 6].

        L = [[-1/Z,    0,  x/Z,     x*y, -(1+x^2),   y],
             [   0, -1/Z,  y/Z,  (1+y^2),    -x*y,  -x]]

    Provided for the image-based servoing path; the framework's own force
    pullback goes through `Sensor.jacobian` instead.
    """
    x, y = xn[..., 0], xn[..., 1]
    iZ = 1.0 / Z.clamp_min(EPS)
    zero = torch.zeros_like(x)
    row_u = torch.stack([-iZ, zero, x * iZ, x * y, -(1.0 + x * x), y], dim=-1)
    row_v = torch.stack([zero, -iZ, y * iZ, 1.0 + y * y, -x * y, -x], dim=-1)
    return torch.stack([row_u, row_v], dim=-2)


LENSES = {"pinhole": Pinhole, "double_sphere": DoubleSphere}


def make_lens(kind: str, **kw) -> Lens:
    if kind not in LENSES:
        raise KeyError(f"unknown lens {kind!r}; registered: {sorted(LENSES)}")
    return LENSES[kind](**kw)
