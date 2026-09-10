"""The composer's ACTION VOCABULARY: components, chained, ended by EOS.

The composer is a language model over the flight: at every report it reads
the chain -- the drone's measurement tokens and its own earlier action tokens
-- and emits a DECISION: a sequence of component tokens ended by EOS, like a
short program.  Nothing in here decides for it; this file only says what the
components do to the TaskSpec when the decision is applied.

    EOS                       the decision ends.  Alone, it is silence: nothing changes
                              and nothing is written to the chain.
    BEARING(b)                the pending placement's direction: the goal direction
                              rotated by b (NB steps around the circle)
    RANGE(r)                  the pending placement's distance: a fraction of the way
                              to the goal or of the reach, whichever is shorter
    RAISE i / LOWER i         the priority of constraint term i, x1.5 / /1.5 (immediate)
    TURN(d)                   command a heading d from the current one, taking the
                              heading over from the plant (gate 1) (immediate)
    LOOK                      hand the heading back to the plant's own look-at (immediate)

At EOS a pending placement is made if a bearing or a range was given: a
bearing without a range places at the full range, a range without a bearing
places straight ahead.  So "BEARING(0) EOS" is the old straight placement,
"RANGE(0.3) EOS" a short straight one, "TURN(+90) BEARING(+90) RANGE(0.6)
EOS" a turn-and-place, and a decision may carry up to `L_MAX` components.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor

from .spec import TaskSpec
from .tokens import to_world


class Vocab:
    NB, RANGES = 12, (0.3, 0.6, 1.0)
    TURNS = (-90.0, -45.0, 45.0, 90.0)
    L_MAX = 4                                  # components per decision, before EOS is forced

    def __init__(self, n_terms: int):
        self.n = int(n_terms); NR = len(self.RANGES)
        self.EOS = 0; self.HOLD = 0            # `HOLD` kept as an alias: silence is a bare EOS
        self.BEAR0 = 1
        self.RANGE0 = self.BEAR0 + self.NB
        self.RAISE0 = self.RANGE0 + NR
        self.LOWER0 = self.RAISE0 + self.n
        self.TURN0 = self.LOWER0 + self.n
        self.LOOK = self.TURN0 + len(self.TURNS)
        self.V = self.LOOK + 1
        self.straight = self.BEAR0 + self.NB // 2               # bearing 0: the goal direction
        # kept for readers of the old layout
        self.PLACE0 = self.BEAR0; self.n_place = self.NB + NR

    # --- classes of component -------------------------------------------------
    def is_bearing(self, tok: Tensor) -> Tensor:
        return (tok >= self.BEAR0) & (tok < self.RANGE0)

    def is_range(self, tok: Tensor) -> Tensor:
        return (tok >= self.RANGE0) & (tok < self.RAISE0)

    def is_place(self, tok: Tensor) -> Tensor:
        """A component that shapes a placement (bearing or range)."""
        return (tok >= self.BEAR0) & (tok < self.RAISE0)

    def name(self, tok: int) -> str:
        if tok == self.EOS: return "EOS"
        if self.BEAR0 <= tok < self.RANGE0: return f"BEARING({-180 + (tok - self.BEAR0) * 360 // self.NB:+d} deg)"
        if self.RANGE0 <= tok < self.RAISE0: return f"RANGE({self.RANGES[tok - self.RANGE0]:.1f})"
        if self.RAISE0 <= tok < self.LOWER0: return f"RAISE {tok - self.RAISE0}"
        if self.LOWER0 <= tok < self.TURN0: return f"LOWER {tok - self.LOWER0}"
        if self.TURN0 <= tok < self.LOOK: return f"TURN({self.TURNS[tok - self.TURN0]:+.0f} deg)"
        return "LOOK"

    def names(self) -> List[str]:
        return [self.name(t) for t in range(self.V)]

    # --- a decision, one component at a time -----------------------------------
    def begin(self, cur: TaskSpec, psi: Tensor):
        """The state of a decision for every row: the spec being edited and
        the pending placement (bearing index, range index; -1 = not given)."""
        out = cur.clone()
        if out.yaw is None:
            out.yaw = psi.clone().to(psi.dtype); out.yaw_gate = torch.zeros_like(psi)
        B = psi.shape[0]
        return out, torch.full((B,), -1, dtype=torch.long, device=psi.device), torch.full((B,), -1, dtype=torch.long, device=psi.device)

    def step(self, tok: Tensor, out: TaskSpec, pend_b: Tensor, pend_r: Tensor, psi: Tensor, rows: Optional[Tensor] = None):
        """Apply one component per row (immediate ones now, placement ones as
        pending).  `rows`: which rows of the decision state `tok` refers to."""
        idx = torch.arange(tok.shape[0], device=tok.device) if rows is None else rows
        b = self.is_bearing(tok); r = self.is_range(tok)
        if bool(b.any()):
            pend_b[idx[b]] = tok[b] - self.BEAR0
        if bool(r.any()):
            pend_r[idx[r]] = tok[r] - self.RANGE0
        for i in range(self.n):
            up = tok == self.RAISE0 + i; dn = tok == self.LOWER0 + i
            if bool(up.any()) or bool(dn.any()):
                a = out.alpha[idx, i]
                a = torch.where(up, (a * 1.5).clamp(max=20.0), a)
                a = torch.where(dn, (a / 1.5).clamp(min=0.05), a)
                out.alpha = out.alpha.clone(); out.alpha[idx, i] = a
        for k, d in enumerate(self.TURNS):
            m = tok == self.TURN0 + k
            if bool(m.any()):
                out.yaw[idx[m]] = psi[idx[m]] + math.radians(d)
                out.yaw_gate[idx[m]] = 1.0
        look = tok == self.LOOK
        if bool(look.any()):
            out.yaw_gate[idx[look]] = 0.0

    def finish(self, out: TaskSpec, pend_b: Tensor, pend_r: Tensor, x: Tensor, goal: Tensor, psi: Tensor,
               g_ego: Tensor, reach: float, z_min: float) -> TaskSpec:
        """EOS for every row: make the pending placements.  Returns the spec
        with `moved` marking the rows that placed."""
        dt = x.dtype
        place = (pend_b >= 0) | (pend_r >= 0)
        if bool(place.any()):
            NR = len(self.RANGES)
            bidx = torch.where(pend_b >= 0, pend_b, torch.full_like(pend_b, self.NB // 2))          # no bearing: straight
            ridx = torch.where(pend_r >= 0, pend_r, torch.full_like(pend_r, NR - 1))                 # no range: full
            theta = (-math.pi + bidx.to(dt) * (2 * math.pi / self.NB))
            frac = torch.tensor(self.RANGES, dtype=dt, device=x.device)[ridx]
            gxy = g_ego[:, :2].to(dt); L0 = gxy.norm(dim=-1)
            phi = torch.atan2(gxy[:, 1], gxy[:, 0]) + theta
            L = frac * L0.clamp(max=1.0)
            zf = (L / L0.clamp_min(1e-6)).clamp(max=1.0)
            sub = torch.stack([L * torch.cos(phi), L * torch.sin(phi), g_ego[:, 2].to(dt) * zf], -1)
            sub = sub / sub.norm(dim=-1, keepdim=True).clamp_min(1.0)       # the reach BALL, height included
            sub_world = x + to_world(sub * reach, psi)
            sub_world = torch.cat([sub_world[:, :2], sub_world[:, 2:].clamp_min(z_min)], -1)
            out.delta = torch.where(place[:, None], sub_world - goal, out.delta)
        out.moved = place
        return out

    def apply(self, tok: Tensor, cur: TaskSpec, x: Tensor, goal: Tensor, psi: Tensor, g_ego: Tensor,
              reach: float, z_min: float) -> TaskSpec:
        """A one-component decision per row followed by EOS (the opening
        placement, tests): `step` then `finish`."""
        out, pb, pr = self.begin(cur, psi.to(x.dtype))
        self.step(tok, out, pb, pr, psi.to(x.dtype))
        return self.finish(out, pb, pr, x, goal, psi.to(x.dtype), g_ego, reach, z_min)
