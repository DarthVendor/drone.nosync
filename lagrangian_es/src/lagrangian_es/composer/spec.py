"""The spec, and the non-learned machinery that makes it safe to apply.

`TaskSpec` is what crosses the interface: where the subgoal sits relative to the
task goal, and how much of each term the low level should use.  Nothing about
wrenches, masses or allocation appears here, which is what lets the same spec
mean something for a quadrotor and for an arm.

`SpecHold` is the part that keeps the certificate.  The composer's output may
jump between calls; the low level must see a spec that changes slowly, and --
more importantly -- that is CONSTANT with respect to the state within an
interval.  V_d = sum_i alpha_i V_i is the gradient of an actual function only
while d(alpha)/dx = 0; an alpha that depended on x would silently make the
field non-conservative.  The hold integrates toward the target at a fixed rate
per step, a rate tied to the closed loop's bandwidth so the low level is never
asked to track a spec faster than it can settle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class TaskSpec:
    """Per-episode: `delta` [B, d] subgoal offset from the task goal (world
    frame, metres); `alpha` [B, n] priorities >= 0; `gate` [B, n] in [0, 1]."""
    delta: Tensor
    alpha: Tensor
    gate: Tensor
    yaw: Optional[Tensor] = None      # [B] commanded heading, world frame; None = the plant's own look-at
    yaw_gate: Optional[Tensor] = None # [B] in [0, 1]: 0 = the plant's look-at, 1 = the command

    @property
    def weight(self) -> Tensor:
        """The conic coefficient the low level actually applies, alpha * gate."""
        return self.alpha * self.gate

    @staticmethod
    def identity(B: int, d: int, n_terms: int, dtype, device) -> "TaskSpec":
        """No subgoal shift, every term at unit priority and fully open.
        Compiling this reproduces the bare controller exactly."""
        return TaskSpec(delta=torch.zeros(B, d, dtype=dtype, device=device),
                        alpha=torch.ones(B, n_terms, dtype=dtype, device=device),
                        gate=torch.ones(B, n_terms, dtype=dtype, device=device))

    def clone(self) -> "TaskSpec":
        return TaskSpec(self.delta.clone(), self.alpha.clone(), self.gate.clone(),
                        None if self.yaw is None else self.yaw.clone(),
                        None if self.yaw_gate is None else self.yaw_gate.clone())

    def where(self, mask: Tensor, other: "TaskSpec") -> "TaskSpec":
        """Rows where `mask` is true from self, the rest from `other`."""
        m = mask.unsqueeze(-1)
        yaw = None if self.yaw is None or other.yaw is None else torch.where(mask, self.yaw, other.yaw)
        yg = None if self.yaw_gate is None or other.yaw_gate is None else torch.where(mask, self.yaw_gate, other.yaw_gate)
        return TaskSpec(torch.where(m, self.delta, other.delta),
                        torch.where(m, self.alpha, other.alpha),
                        torch.where(m, self.gate, other.gate), yaw, yg)


class SpecHold:
    """Zero-order hold with rate limiting, between composer and controller.

    `rate_m` metres per second the subgoal may move; `rate_w` per second for
    the weights.  Both are set from the low level's bandwidth (`omega_n`), not
    tuned: a subgoal that moves faster than the closed loop can follow is a
    step, and the whole point of the hold is that the low level sees none.
    """

    def __init__(self, B: int, d: int, n_terms: int, dt: float, dtype, device,
                 omega_n: float = 2.0, reach: float = 10.0):
        self.dt = float(dt)
        self.realized = TaskSpec.identity(B, d, n_terms, dtype, device)
        self.target = self.realized.clone()
        # one closed-loop time constant to sweep the reach; weights a bit faster
        self.rate_m = reach * omega_n * self.dt
        self.rate_w = 2.0 * omega_n * self.dt
        self.rate_psi = 2.0 * omega_n * self.dt          # rad per step, ~ the plant's own yaw slew
        self.age = torch.zeros(B, dtype=torch.long, device=device)

    def set_target(self, spec: TaskSpec, rows: Optional[Tensor] = None) -> None:
        if rows is None:
            self.target = spec.clone()
            self.age.zero_()
        else:
            self.target = spec.where(rows, self.target)
            self.age = torch.where(rows, torch.zeros_like(self.age), self.age)

    def step(self) -> TaskSpec:
        """Advance the realized spec one control step toward the target."""
        r = self.realized; t = self.target
        dd = t.delta - r.delta
        n = dd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        r.delta = r.delta + dd * (self.rate_m / n).clamp_max(1.0)
        r.alpha = r.alpha + (t.alpha - r.alpha).clamp(-self.rate_w, self.rate_w)
        r.gate = r.gate + (t.gate - r.gate).clamp(-self.rate_w, self.rate_w)
        if t.yaw is not None:
            cur = t.yaw if r.yaw is None else r.yaw
            d = torch.atan2(torch.sin(t.yaw - cur), torch.cos(t.yaw - cur))
            r.yaw = cur + d.clamp(-self.rate_psi, self.rate_psi)
            g0 = torch.zeros_like(t.yaw) if r.yaw_gate is None else r.yaw_gate
            tg = torch.ones_like(t.yaw) if t.yaw_gate is None else t.yaw_gate
            r.yaw_gate = g0 + (tg - g0).clamp(-self.rate_w, self.rate_w)
        else:
            r.yaw = None; r.yaw_gate = None
        self.age += 1
        return r


def ground_release(z: Tensor, vz: Tensor, z_release: float = 0.6,
                   vz_max: float = 0.5) -> Tensor:
    """Hard interlock: the ground barrier may be released only low and slow.

    Landing is the one skill that needs the barrier every other skill holds,
    so this is not a learned gate.  Returns a bool per episode."""
    return (z < z_release) & (vz.abs() < vz_max)


def stale_fallback(hold: SpecHold, k: int) -> Tensor:
    """Rows whose target spec is older than `k` intervals: hold position under
    local barriers rather than execute an instruction the world has moved on
    from.  Coordinator latency is routine, not exceptional.  Returns the rows
    and rewrites their target to 'no subgoal, barriers open'."""
    stale = hold.age > k
    if bool(stale.any()):
        safe = TaskSpec.identity(hold.target.delta.shape[0],
                                 hold.target.delta.shape[1],
                                 hold.target.alpha.shape[1],
                                 hold.target.delta.dtype, hold.target.delta.device)
        # "hold position" = subgoal at the current spot is the caller's job
        # (it knows x); here the delta is zeroed so the leg goal stops pulling
        safe.delta = hold.target.delta * 0.0
        hold.target = safe.where(stale, hold.target)
    return stale
