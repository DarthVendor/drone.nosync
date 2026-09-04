"""`MLPPolicy` -- the unstructured arm of the 2x2.

A two-hidden-layer tanh network mapping (task error, task velocity) to a
task-space force.

On width: the spec names "2x64" in prose but "~1-3k" parameters in the table, and
the two disagree -- 64 units give 4,803 policy slots.  The default here is 32
(1,379 slots), which lands in the stated band and keeps the ablation tractable:
the metric's `eigh` is O(dim^3), so a 4.8k genome makes each refresh ~40x more
expensive than a 1.4k one and the 2x2 stops being budget-matched in wall clock.
Pass `hidden=64` for the literal reading.
It exists so that the *structured genome* claim can be tested separately from the
*whitened operator* claim: the two are independent design choices, and the
ablation is only meaningful if each can be removed on its own.

It is deliberately not a strawman.  It gets the same `gravity_force` feedforward
every other trainable gets -- without it the network would spend its entire
budget learning to hover and the comparison would be worthless -- and the same
allocator seam, so the only thing that differs from `EnergyShaping` is where the
force comes from.

What it does NOT get, because that is the whole point of the comparison:
`equilibrium_exact = False`.  Nothing constrains a random network to vanish at
the goal, so an untrained individual is not an energy-shaping controller and its
closed-loop equilibrium is wherever the weights happen to put it.  Evolution has
to discover stability rather than being handed it.

Inputs are scaled before the first layer.  The network sees errors spanning
centimetres to metres, and an unscaled first layer saturates its tanh units
immediately -- which would also hand the metric a rank-deficient G for reasons
that have nothing to do with the physics.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..systems.base import LagrangianSystem, State
from .base import Trainable


class MLPPolicy(Trainable):
    equilibrium_exact = False

    def __init__(self, system: LagrangianSystem, hidden: int = 32,
                 e_scale: float = 2.0, v_scale: float = 2.0, out_scale: float = 4.0,
                 init_gain: float = 0.5,
                 phi0: tuple = (0.25, 0.25, 0.25, 0.10, 0.10, 0.10)):
        super().__init__(system)
        self.d = int(system.task_dim)
        self.h = int(hidden)
        self.n_in = 2 * self.d
        self.e_scale, self.v_scale, self.out_scale = e_scale, v_scale, out_scale
        self.init_gain = float(init_gain)
        self.phi0 = tuple(phi0)

        n = 0
        self._sl = {}
        for key, size in (
            ("W1", self.n_in * self.h), ("b1", self.h),
            ("W2", self.h * self.h),    ("b2", self.h),
            ("W3", self.h * self.d),    ("b3", self.d),
        ):
            self._sl[key] = (n, n + size)
            n += size
        self._policy_dim = n

    @property
    def policy_dim(self) -> int:
        return self._policy_dim

    def unpack(self, theta: Tensor):
        lead = theta.shape[:-1]
        g = lambda k, shape: theta[..., self._sl[k][0]: self._sl[k][1]].reshape(lead + shape)
        return (g("W1", (self.n_in, self.h)), g("b1", (self.h,)),
                g("W2", (self.h, self.h)), g("b2", (self.h,)),
                g("W3", (self.h, self.d)), g("b3", (self.d,)),
                self.split(theta)[1])

    def init(self) -> Tensor:
        """Small random weights: the net starts near zero output, so the initial
        controller is the gravity feedforward alone -- it hovers, badly, exactly
        like the energy-shaping prior does."""
        gen = torch.Generator(device=str(self.device)).manual_seed(0)
        kw = dict(generator=gen, dtype=self.dtype, device=self.device)
        parts = []
        for fan_in, fan_out in ((self.n_in, self.h), (self.h, self.h), (self.h, self.d)):
            s = self.init_gain / fan_in ** 0.5
            parts.append(s * torch.randn(fan_in * fan_out, **kw))
            parts.append(torch.zeros(fan_out, dtype=self.dtype, device=self.device))
        # interleave to match the slice order W1,b1,W2,b2,W3,b3
        flat = torch.cat(parts, dim=0)
        if self.system.allocator_dim:
            flat = torch.cat([flat, self.system.allocator_init()], dim=0)
        if self.system.residual_dim:
            flat = torch.cat([flat, self.system.residual_init()], dim=0)
        return flat

    def forward(self, theta: Tensor, s: State, goal: Tensor) -> Tensor:
        sysm = self.system
        W1, b1, W2, b2, W3, b3, phi = self.unpack(theta)
        e = (sysm.task_position(s) - goal) / self.e_scale
        v = sysm.task_velocity(s) / self.v_scale
        x = torch.cat([e, v], dim=-1)
        x = torch.tanh(torch.einsum("...i,...ij->...j", x, W1) + b1)
        x = torch.tanh(torch.einsum("...i,...ij->...j", x, W2) + b2)
        out = torch.einsum("...i,...ij->...j", x, W3) + b3
        F_des = sysm.gravity_force(s) + self.out_scale * out
        return sysm.allocate(F_des, s, phi)

    def describe(self, theta: Tensor) -> dict:
        with torch.no_grad():
            W1, _, W2, _, W3, _, phi = self.unpack(theta)
            out = {"W1_norm": float(W1.norm()), "W2_norm": float(W2.norm()),
                   "W3_norm": float(W3.norm()), "n_params": self.policy_dim}
            if phi.numel() >= 6:
                out["kR"] = float((phi[:3] ** 2).mean())
                out["kW"] = float((phi[3:6] ** 2).mean())
            return out
