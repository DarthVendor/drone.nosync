"""A carried map of the buildings -- the "Google Maps view".

This is the ONE place a prior over the world's geometry is allowed to enter,
and it enters as a declared sensor so that a config either carries it or does
not.  Everything else in the stack still reads only what the vehicle perceives;
see `composer/tokens.py`, whose contract is that the scene reaches the composer
through `obs` and never through `state["boxes/*"]`.

The model is a vector map, not a picture: real map data is building footprints,
and a raster fine enough to resolve a four-metre street would cost hundreds of
tokens where the footprints cost one apiece.  Each entry is one building --
where it is, how big it is, how tall -- and only those inside the viewport,
which is what a map application shows you at a given zoom.

What it deliberately does NOT know: anything not in the map.  It reports static
footprints, so a scene whose obstacles moved would be reported wrongly, exactly
as a stale map would be.  It is also noiseless and undelayed -- a map does not
shimmer -- which is why `sigma` and `latency_steps` default to zero.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..systems.base import LagrangianSystem, State
from .base import Sensor

#: features per building: centre x, centre y, half-width, half-depth, height,
#: yaw, confidence, age.  World frame -- the tokenizer puts them in the ego
#: frame, the same way it does the goal, so equivariance is one rule in one
#: place.  The age slot exists because both maps present as MEASUREMENT tokens;
#: a carried survey is never stale, so this one always reports zero, and that
#: contrast is itself informative to the composer.
FEATS = 8


class MapPrior(Sensor):
    """The `k` nearest building footprints within `max_range`, world frame."""

    kind = "map"
    name = "map_prior"
    update_every = 5        # static geometry; only the ranking moves with the vehicle
    stateless = True

    def __init__(self, system: LagrangianSystem, k: int = 12, max_range: float = 30.0,
                 sigma: float = 0.0, latency_steps: int = 0, **kw):
        self.system = system
        self.k = int(k)
        self.max_range = float(max_range)
        self.obs_dim = self.k * FEATS
        self.sigma = float(sigma)
        self.latency_steps = int(latency_steps)

    @classmethod
    def supports(cls, system) -> bool:
        return hasattr(system, "env") or hasattr(system, "environment")

    def observe(self, s: State, gen: torch.Generator) -> Tensor:
        c, h, a = s.get("boxes/c"), s.get("boxes/h"), s.get("boxes/a")
        p = self.system.task_position(s)
        B = p.shape[0]
        dt, dev = p.dtype, p.device
        if c is None or h is None:
            return torch.zeros(B, self.obs_dim, dtype=dt, device=dev)
        n = c.shape[1]
        if a is None:
            a = torch.zeros(c.shape[:2], dtype=dt, device=dev)
        # Rank by distance to the footprint's SURFACE, not its centre: a wide
        # block whose centre is far can still be the wall beside the vehicle,
        # and the near ones are what a map is consulted about.
        d = self._surface_distance(p, c, h, a)                      # [B, n]
        k = min(self.k, n)
        near, idx = torch.topk(d, k, dim=1, largest=False)
        gather = lambda t: torch.gather(t, 1, idx.reshape(B, k, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:]))
        cs, hs, as_ = gather(c), gather(h), torch.gather(a, 1, idx)
        valid = (near <= self.max_range).to(dt)                     # outside the viewport: reported as absent
        out = torch.cat([cs, hs[..., :2], hs[..., 2:3], as_[..., None], valid[..., None],
                         torch.zeros(B, k, 1, dtype=dt, device=dev)], -1)
        if k < self.k:                                              # pad to a fixed width
            out = torch.cat([out, torch.zeros(B, self.k - k, FEATS, dtype=dt, device=dev)], 1)
        return out.reshape(B, self.obs_dim)

    def measure(self, s: State):
        """Noiseless reading and (no) Jacobian.

        Split out so the rollout can skip frozen episodes; a map read is a pure
        function of the state, hence `stateless`.
        """
        return self.observe(s, None), None

    def perturb(self, x: Tensor, gen) -> Tensor:
        return x                      # a map does not shimmer

    @staticmethod
    def _surface_distance(p: Tensor, c: Tensor, h: Tensor, a: Tensor) -> Tensor:
        """Horizontal distance from each vehicle to each box's surface, [B, n]."""
        d = p[:, None, :2] - c                                      # [B, n, 2]
        ca, sa = torch.cos(a), torch.sin(a)
        loc = torch.stack([ca * d[..., 0] + sa * d[..., 1],
                           -sa * d[..., 0] + ca * d[..., 1]], -1)   # into each box's own frame
        q = loc.abs() - h[..., :2]
        return q.clamp(min=0.0).norm(dim=-1) + q.max(-1).values.clamp(max=0.0)

    def jacobian(self, s: State) -> Tensor:
        """Zero: the map is not a control input.

        It exists for the task-level composer, never for the low-level
        potential -- a controller that felt a gradient from remembered geometry
        would be steering on a prior instead of on what it can see.
        """
        p = self.system.task_position(s)
        return torch.zeros(p.shape[0], self.obs_dim, p.shape[-1], dtype=p.dtype, device=p.device)
