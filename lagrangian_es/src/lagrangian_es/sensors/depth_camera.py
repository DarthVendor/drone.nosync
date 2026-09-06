"""`DepthCamera` -- a body-mounted forward depth camera. Project, never render.

Why this and not `LandmarkCamera`
---------------------------------
`LandmarkCamera` projects KNOWN 3-D points and is a localisation sensor: it tells
the vehicle where it is, not what is in front of it.  This one casts a grid of
rays through the scene and returns depth, so it reports obstacles the way the
range fan does -- and the existing obstacle terms consume exactly that, a depth
vector plus its position Jacobian, so they work against it unchanged.

Why forward-facing, and why it has vertical extent
--------------------------------------------------
`RangeSensor` sweeps a full circle but every beam has `z = 0`, so it is blind to
anything off the vehicle's own altitude.  On the hoop scenes that is total: 0 of
192 beams return a hit at reset.  Adding elevation rings recovered +0.11 of
reach, which says the blindness is real but not the whole story -- 84% of hoop
impacts happen with the vehicle more than 45 deg off the ring's axis, i.e. flying
broadside into it.

A camera answers the part a fan cannot.  It is mounted in the BODY frame and
rotates with the vehicle, so its readings are organised around where the vehicle
is pointing rather than around a world compass, and a grid over both azimuth and
elevation resolves the SHAPE of what is ahead -- an aperture reads as a pocket of
far returns ringed by near ones, which a single horizontal line cannot express.

Cost
----
Rays, not images: `H*W` raycasts per step against the same SDF the fan uses.  A
16x12 grid is 192 rays, comparable to the 5-ring fan's 120, and no pixels are
ever shaded.  Depth is clamped at `max_range`, and a miss reads as far rather
than as a sentinel, so "nothing there" cannot be confused with "something close".
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor

from ..systems.base import LagrangianSystem, State
from .base import Sensor

EPS = 1e-9


class DepthCamera(Sensor):
    kind = "range"
    name = "depth_camera"
    update_every = 1        # an obstacle sensor blind for 0.1 s doubles crashes

    def __init__(self, system: LagrangianSystem, width: int = 16, height: int = 12,
                 hfov: float = 1.4, vfov: float = 1.05, max_range: float = 6.0,
                 sigma: float = 0.02, latency_steps: int = 1,
                 pitch: float = 0.0, forward: Sequence = (1.0, 0.0, 0.0)):
        self.system = system
        self.W, self.H = int(width), int(height)
        self.obs_dim = self.W * self.H
        self.hfov, self.vfov = float(hfov), float(vfov)
        self.max_range = float(max_range)
        self.sigma = float(sigma)
        self.latency_steps = int(latency_steps)
        self.pitch = float(pitch)
        self._fwd = tuple(float(x) for x in forward)
        self._body_dirs: Optional[Tensor] = None

    @classmethod
    def supports(cls, system) -> bool:
        return hasattr(system, "raycast")

    def _grid(self, dtype, device) -> Tensor:
        """Ray directions in the BODY frame, [obs_dim, 3]. Built once."""
        if self._body_dirs is not None and self._body_dirs.dtype == dtype:
            return self._body_dirs
        # pixel centres, so no ray sits exactly on the optical axis or the edge
        u = (torch.arange(self.W, dtype=dtype, device=device) + 0.5) / self.W - 0.5
        v = (torch.arange(self.H, dtype=dtype, device=device) + 0.5) / self.H - 0.5
        az = u * self.hfov
        el = v * self.vfov + self.pitch
        A, E = torch.meshgrid(az, el, indexing="xy")
        A, E = A.reshape(-1), E.reshape(-1)
        ce = torch.cos(E)
        # forward is body +x; azimuth swings in body xy, elevation out of plane
        d = torch.stack([torch.cos(A) * ce, torch.sin(A) * ce, torch.sin(E)], -1)
        f = torch.tensor(self._fwd, dtype=dtype, device=device)
        f = f / f.norm().clamp_min(EPS)
        if not torch.allclose(f, torch.tensor([1.0, 0.0, 0.0], dtype=dtype,
                                              device=device)):
            # rotate the boresight onto `forward` (Rodrigues about f x +x)
            x = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
            v_ = torch.cross(x, f, dim=-1)
            c = float((x * f).sum())
            k = v_.norm().clamp_min(EPS)
            K = torch.tensor([[0.0, -float(v_[2]), float(v_[1])],
                              [float(v_[2]), 0.0, -float(v_[0])],
                              [-float(v_[1]), float(v_[0]), 0.0]],
                             dtype=dtype, device=device)
            R = (torch.eye(3, dtype=dtype, device=device) + K
                 + K @ K * ((1.0 - c) / (k * k)))
            d = d @ R.T
        self._body_dirs = d
        return d

    def _dirs(self, s: State) -> Tensor:
        """Body grid carried into the world by the vehicle's attitude."""
        R = s["R"]
        d = self._grid(R.dtype, R.device)
        # [..., 3, 3] x [n, 3] -> [..., n, 3]; the camera turns with the vehicle
        return torch.einsum("...ij,nj->...ni", R, d)

    def observe(self, s: State, gen: torch.Generator) -> Tensor:
        rng, _ = self.system.raycast(s, self._dirs(s), self.max_range)
        if self.sigma > 0 and gen is not None:
            rng = rng + self.sigma * self.crn_noise(rng.shape, gen, rng.dtype,
                                                    rng.device)
        return rng.clamp(0.0, self.max_range)

    def jacobian(self, s: State) -> Tensor:
        _, grad = self.system.raycast(s, self._dirs(s), self.max_range)
        return grad

    def valid(self, s: State, gen=None) -> Tensor:
        rng, _ = self.system.raycast(s, self._dirs(s), self.max_range)
        return rng < self.max_range - EPS

    def describe(self) -> dict:
        d = super().describe()
        d.update(width=self.W, height=self.H, hfov=self.hfov, vfov=self.vfov,
                 max_range=self.max_range, sigma=self.sigma, pitch=self.pitch)
        return d

    def render_spec(self) -> dict:
        return {"type": "depth_camera", "width": self.W, "height": self.H,
                "hfov": self.hfov, "vfov": self.vfov,
                "max_range": self.max_range}

    def render_frame(self, s: State) -> dict:
        rng, _ = self.system.raycast(s, self._dirs(s), self.max_range)
        return {"depth": rng, "range": rng}
