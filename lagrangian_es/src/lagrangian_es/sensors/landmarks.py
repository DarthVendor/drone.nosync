"""`LandmarkCamera` -- project, never render.

Rendering 192 parallel vehicles x 250 steps is 48,000 images per generation and
destroys the sub-minute training budget for no benefit.  The environment instead
carries K known 3-D points; the sensor transforms them into the camera frame,
projects them through the lens, and marks the ones that are behind the camera or
outside the image invalid.  The whole thing is a few batched matmuls, negligible
against a 0.3 s rollout.

Noise model: Gaussian pixel noise, Bernoulli detection dropout, and quantization
to integer pixels.  Dropout reports through `valid` rather than through a
sentinel value, because "no return" must read as UNKNOWN and not as clear.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from ..systems.base import LagrangianSystem, State
from .base import Sensor
from .lens import Lens, make_lens


class LandmarkCamera(Sensor):
    kind = "position_like"
    name = "landmark_camera"

    def __init__(self, system: LagrangianSystem,
                 landmarks: Optional[Sequence] = None, n_landmarks: int = 16,
                 lens: str = "double_sphere", lens_kw: Optional[dict] = None,
                 sigma_px: float = 0.5, dropout: float = 0.05,
                 quantize: bool = True, latency_steps: int = 3,
                 extent: Sequence = (3.0, 3.0, 2.5), seed: int = 0,
                 R_mount: Optional[Tensor] = None, t_mount: Sequence = (0.0, 0.0, 0.0)):
        self.system = system
        self.lens: Lens = make_lens(lens, **(lens_kw or {}))
        self.sigma_px, self.dropout = float(sigma_px), float(dropout)
        self.quantize = bool(quantize)
        self.latency_steps = int(latency_steps)

        dt, dev = system.dtype, system.device
        if landmarks is None:
            g = torch.Generator(device=str(dev)).manual_seed(int(seed))
            ext = torch.as_tensor(extent, dtype=dt, device=dev)
            lm = (torch.rand(int(n_landmarks), 3, generator=g, dtype=dt, device=dev)
                  * 2.0 - 1.0) * ext
            lm[:, 2] = lm[:, 2].abs() + 0.2            # keep them above the floor
        else:
            lm = torch.as_tensor(landmarks, dtype=dt, device=dev)
        self.landmarks = lm
        self.K = lm.shape[0]
        self.obs_dim = 2 * self.K

        # camera mounted looking forward along body +x, with world-z up mapping to
        # image -y: columns are the camera axes expressed in the body frame
        self.R_mount = (torch.as_tensor(R_mount, dtype=dt, device=dev)
                        if R_mount is not None
                        else torch.tensor([[0.0, 0.0, 1.0],
                                           [-1.0, 0.0, 0.0],
                                           [0.0, -1.0, 0.0]], dtype=dt, device=dev))
        self.t_mount = torch.as_tensor(t_mount, dtype=dt, device=dev)
        self._has_mount_offset = bool(self.t_mount.abs().max() > 0)

    # --- geometry -----------------------------------------------------------
    def _camera_frame(self, s: State) -> Tensor:
        """Landmarks in the camera frame, [..., K, 3]."""
        p, R = self.system.camera_pose(s)
        Rwc = R @ self.R_mount                                   # body -> camera
        origin = p
        if self._has_mount_offset:
            origin = p + torch.einsum("...ij,...j->...i", R,
                                      self.t_mount.expand_as(p))
        rel = self.landmarks - origin[..., None, :]              # [..., K, 3]
        return torch.einsum("...ji,...kj->...ki", Rwc, rel)

    def _pixels(self, s: State, visibility: bool = True):
        """`visibility=False` skips the in-view test.

        Worth the flag: this sensor sits on the hot path once per step, and at
        these tensor sizes the cost is dominated by per-op dispatch rather than
        arithmetic -- so computing a visibility mask that `observe` never reads
        is a real fraction of the rollout, not a rounding error.
        """
        Xc = self._camera_frame(s)
        uv = self.lens.project(Xc)
        vis = self.lens.in_view(Xc, uv) if visibility else None
        return Xc, uv, vis

    # --- Sensor -------------------------------------------------------------
    def observe(self, s: State, gen: torch.Generator) -> Tensor:
        _, uv, _ = self._pixels(s, visibility=False)
        flat = uv.reshape(uv.shape[:-2] + (self.obs_dim,))
        if self.sigma_px > 0:
            flat = flat + self.sigma_px * self.crn_noise(
                flat.shape, gen, flat.dtype, flat.device)
        if self.quantize:
            flat = torch.round(flat)
        return flat

    def jacobian(self, s: State) -> Tensor:
        """d(pixels)/d(camera position), [..., 2K, 3].

        The pullback chain J = J_lens . J_project . J_pose; here J_pose is the
        derivative of the camera-frame point with respect to the camera's own
        position, which is -R_wc^T.  Closed form throughout, so this stays cheap
        enough to sit inside a vmapped `forward`.
        """
        Xc = self._camera_frame(s)
        Jl = self.lens.jacobian(Xc)                              # [..., K, 2, 3]
        _, R = self.system.camera_pose(s)
        Rwc = R @ self.R_mount
        # dXc_j/dp_l = -(Rwc^T)[j, l] = -Rwc[l, j], so contract Jl's camera-axis
        # index j against Rwc's SECOND index.  Feeding Rwc already transposed and
        # then contracting "...lj" transposes it twice and cyclically permutes the
        # columns -- which still looks plausible and is caught only by the
        # finite-difference check.
        J = -torch.einsum("...kij,...lj->...kil", Jl, Rwc)
        return J.reshape(J.shape[:-3] + (self.obs_dim, 3))

    def valid(self, s: State, gen=None) -> Tensor:
        _, _, vis = self._pixels(s)
        if self.dropout > 0 and gen is not None:
            u = torch.rand(vis.shape, generator=gen, dtype=self.system.dtype,
                           device=vis.device)
            vis = vis & (u >= self.dropout)
        return vis[..., None].expand(vis.shape + (2,)).reshape(
            vis.shape[:-1] + (self.obs_dim,))

    def visible(self, s: State) -> Tensor:
        """[..., K] bool -- geometric visibility only, no dropout."""
        return self._pixels(s)[2]

    def describe(self) -> dict:
        d = super().describe()
        d.update(n_landmarks=self.K, sigma_px=self.sigma_px,
                 dropout=self.dropout, **self.lens.describe())
        return d

    # --- rendering ----------------------------------------------------------
    def render_spec(self) -> dict:
        return {"type": "landmark_camera", "n_landmarks": self.K,
                "landmarks": self.landmarks.reshape(-1).tolist(),
                "image": [self.lens.width, self.lens.height],
                "mount": self.R_mount.reshape(-1).tolist()}

    def render_frame(self, s: State) -> dict:
        Xc, uv, vis = self._pixels(s)
        return {"uv": uv, "visible": vis.to(uv.dtype), "depth": Xc[..., 2]}
