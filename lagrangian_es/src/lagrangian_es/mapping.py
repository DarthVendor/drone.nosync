"""The map the vehicle builds for itself, from its own beams.

The counterpart to `sensors/map_view.py`.  That one hands the composer a prior
it was given; this one hands it a record of what it has actually seen, and the
two are deliberately the same shape so an experiment can switch either on
without changing anything downstream.

Why it exists: the composer's only memory was a chain of scalars -- progress,
distance remaining, closest beam, alive, arrived.  Nothing in that has spatial
structure, so a vehicle could fly into a dead end, back out, and fly straight
back in, because the second approach looks exactly like the first.  A fifth of
corridor flights end alive but unfinished, which is the signature of that.

What it may use: ONLY what the range sensor returned -- the same noisy, strided,
delayed readings the rest of the stack sees.  It never consults the environment.
That is what separates it from the prior map, and the separation is the whole
point of running them against each other.

Occupancy only.  The beams are a horizontal fan at the vehicle's own altitude,
so they carry almost no information about how tall anything is; the height slot
a prior map fills is reported as zero here, and the confidence slot carries how
often a cell has been seen instead.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

#: matches `sensors.map_view.FEATS`, so both maps tokenize through one path:
#: centre x, centre y, half-width, half-depth, height, yaw, confidence, age.
#: The age is what makes these MEASUREMENT tokens rather than static facts --
#: "a wall here, last seen eight seconds ago" is the signal a vehicle needs to
#: know it is re-entering a corridor it already tried.
FEATS = 8

#: a cell needs this many hits to read as fully confident
FULL_CONFIDENCE = 3.0


class BuiltMap:
    """A per-episode occupancy grid accumulated from returned beam ranges."""

    def __init__(self, extent: float = 72.0, cell: float = 2.0, k: int = 12,
                 max_range: float = 30.0):
        self.extent = float(extent)         # side of the square the grid covers, metres
        self.cell = float(cell)
        self.k = int(k)
        self.max_range = float(max_range)   # the read viewport, matching the prior map's
        self.G = max(1, int(round(self.extent / self.cell)))
        self.hits: Optional[Tensor] = None
        self.seen_t: Optional[Tensor] = None      # simulation time each cell was last returned
        self._centres: Optional[Tensor] = None

    # --- lifecycle ---------------------------------------------------------
    def reset(self, B: int, device, dtype) -> None:
        """A fresh, empty map: the vehicle starts every episode knowing nothing."""
        self.hits = torch.zeros(B, self.G * self.G, dtype=torch.float32, device=device)
        self.seen_t = torch.zeros(B, self.G * self.G, dtype=torch.float32, device=device)
        self._centres = None

    def centres(self, dtype, device) -> Tensor:
        """World-frame centre of every cell, [G*G, 2]."""
        if self._centres is not None and self._centres.dtype == dtype and self._centres.device == device:
            return self._centres
        half = self.extent / 2.0
        ax = (torch.arange(self.G, dtype=dtype, device=device) + 0.5) * self.cell - half
        cx = ax[None, :].expand(self.G, -1).reshape(-1)
        cy = ax[:, None].expand(-1, self.G).reshape(-1)
        self._centres = torch.stack([cx, cy], -1)
        return self._centres

    # --- accumulate --------------------------------------------------------
    def update(self, p: Tensor, dirs: Tensor, rng: Tensor, max_range: float,
               live: Optional[Tensor] = None, t: float = 0.0) -> None:
        """Mark the cell each returning beam ended in.

        `dirs` are the beams' WORLD directions and `rng` what they returned, so
        a reading that was noisy or held over from an earlier step marks the
        cell it appears to have come from -- the vehicle's own belief, errors
        and all.  A beam that ran to its limit hit nothing and marks nothing.
        """
        if self.hits is None:
            return
        B, n = rng.shape
        hit = rng < 0.999 * max_range
        if live is not None:
            hit = hit & live[:, None]
        if not bool(hit.any()):
            return
        end = p[:, None, :2] + dirs[..., :2] * rng[..., None]        # [B, n, 2]
        half = self.extent / 2.0
        ij = ((end + half) / self.cell).floor().long()
        inside = (ij >= 0).all(-1) & (ij < self.G).all(-1) & hit
        flat = (ij[..., 1].clamp(0, self.G - 1) * self.G + ij[..., 0].clamp(0, self.G - 1))
        # one scatter over the whole batch; cells hit twice in a scan count twice
        flat = flat.clamp(0, self.G * self.G - 1)
        self.hits.scatter_add_(1, flat, inside.to(self.hits.dtype))
        # last-seen time: a later sighting refreshes the cell
        stamp = torch.where(inside, torch.full_like(flat, 0, dtype=self.seen_t.dtype) + float(t),
                            torch.zeros_like(flat, dtype=self.seen_t.dtype))
        self.seen_t.scatter_reduce_(1, flat, stamp, reduce="amax")

    # --- read --------------------------------------------------------------
    def read(self, p: Tensor, now: float = 0.0) -> Tensor:
        """The `k` nearest remembered cells, world frame, [B, k * FEATS].

        Same layout as the prior map: centre, half-extents, height, yaw,
        confidence, age.  Height is zero (a horizontal fan cannot see it), the
        confidence slot says how often the cell has been seen, and the age says
        how long ago -- which is the point of keeping the map at all.
        """
        B = p.shape[0]
        dt, dev = p.dtype, p.device
        if self.hits is None:
            return torch.zeros(B, self.k * FEATS, dtype=dt, device=dev)
        ctr = self.centres(dt, dev)                                  # [G*G, 2]
        d = (p[:, None, :2] - ctr[None]).norm(dim=-1)                # [B, G*G]
        seen = self.hits > 0
        d = torch.where(seen, d, torch.full_like(d, float("inf")))
        d = torch.where(d <= self.max_range, d, torch.full_like(d, float("inf")))
        k = min(self.k, d.shape[1])
        near, idx = torch.topk(d, k, dim=1, largest=False)
        cs = ctr[idx]                                                # [B, k, 2]
        cnt = torch.gather(self.hits.to(dt), 1, idx)
        conf = (cnt / FULL_CONFIDENCE).clamp(0.0, 1.0) * torch.isfinite(near).to(dt)
        h = torch.full((B, k, 1), self.cell / 2.0, dtype=dt, device=dev)
        age = (float(now) - torch.gather(self.seen_t.to(dt), 1, idx)).clamp(min=0.0)
        out = torch.cat([cs, h, h, torch.zeros(B, k, 2, dtype=dt, device=dev),
                         conf[..., None], age[..., None]], -1)
        out = out * torch.isfinite(near).to(dt)[..., None]           # nothing remembered: an absent entry
        if k < self.k:
            out = torch.cat([out, torch.zeros(B, self.k - k, FEATS, dtype=dt, device=dev)], 1)
        return out.reshape(B, self.k * FEATS)
