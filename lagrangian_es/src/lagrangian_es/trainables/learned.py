"""A learned energy-shaping term: the SHAPE is a network, the guarantees are not.

Why this and not `MLPPolicy`
----------------------------
`MLPPolicy` maps state straight to force and declares `equilibrium_exact = False`
-- nothing makes a random network vanish at the goal, so it is the unstructured
control in the 2x2, not a controller with a certificate.  This term sits in
between: the network shapes `V_d` and `K_d`, while PSD-ness and stationarity at
the goal are enforced by the FORM rather than learned.  So the search is free to
discover what the potential should look like without being free to discard the
property that makes every individual an energy-shaping controller.

That matters here because the hand-designed stack -- three bowls, a constant
damper, a range barrier, a closing damper -- is a series of guesses, and several
were wrong in ways that took measurement to find.  This asks whether the search
can do better than the guesses when the structure stops constraining the shape.

The construction
----------------
    V(e, obs) = || h(e, obs) - h(0, obs) ||^2 ,   h = A e + mlp(e, obs)

is nonnegative for free, is exactly zero at `e = 0`, and has `grad_e V = 0` there
too (the gradient carries a factor of `h - h(0)`).  Subtracting the network's own
value at the goal is what buys the last two: without it, a random net puts the
closed-loop equilibrium wherever its weights happen to land.

`A` is a linear skip initialised to the identity, so the PRIOR is exactly the
quadratic bowl the hand-designed stack starts from.  Without it a small-weight
net is nearly flat and has no goal attraction at all, and the comparison would be
measuring the warm start rather than the parameterisation.

    R(v, obs) = 1/2 v^T L L^T v ,   L = reshape(g(obs))

is PSD by construction like `DissipationTerm`, so it can only ever remove energy,
and `H = T + V_d` stays non-increasing however the weights come out.

Both heads share one trunk over `obs`, because the thing they need to know --
where the obstacles are -- is the same.  The dissipation head is not optional
window dressing: a potential's force depends on position alone, so it delivers a
fixed deceleration and can arrest an approach only from `v <= sqrt(2 a d)`.
Measured on this plant that ceiling is 2.3 m/s while collisions happen at 3.09,
and no amount of shaping fixes it -- only a velocity-dependent term can.

Sizing
------
The genome is ~360 slots against 50 for the hand-designed stack.  ES scales badly
in dimension and the metric's `eigh` is O(dim^3), so this is deliberately one
hidden layer and narrow; it is a test of whether shape can be learned, not an
attempt to win on capacity.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from .terms import LagrangianTerm


class LearnedShaping(LagrangianTerm):
    """MLP-shaped `V_d` and `K_d`, with the certificate enforced structurally."""

    kind = "learned_shaping"
    uses_obs = True

    def __init__(self, d: int, sensor_name: str = "range", n_obs: int = 12,
                 hidden: int = 16, out: int = 6, obs_scale: float = 4.0,
                 e_scale: float = 2.0, v_gate: bool = True,
                 init_gain: float = 0.4, damp0: float = 1.2):
        super().__init__(d)
        self.sensor_name = sensor_name
        self.n_obs, self.h, self.out = int(n_obs), int(hidden), int(out)
        self.obs_scale, self.e_scale = float(obs_scale), float(e_scale)
        self.init_gain, self.damp0 = float(init_gain), float(damp0)
        self.n_in = self.d + self.n_obs
        n, self._sl = 0, {}
        for key, size in (("W1", self.n_in * self.h), ("b1", self.h),
                          ("W2", self.h * self.out), ("b2", self.out),
                          ("A", self.d * self.out),
                          ("Wd", self.n_obs * (self.d * self.d)),
                          ("bd", self.d * self.d)):
            self._sl[key] = (n, n + size)
            n += size
        self._dim = n

    @property
    def dim(self) -> int:
        return self._dim

    def _p(self, theta, key, shape):
        a, b = self._sl[key]
        return theta[..., a:b].reshape(theta.shape[:-1] + shape)

    def init(self, dtype=torch.float64, device="cpu") -> Tensor:
        g = torch.Generator(device="cpu").manual_seed(0)
        parts = []
        for key, shape in (("W1", (self.n_in, self.h)), ("b1", (self.h,)),
                           ("W2", (self.h, self.out)), ("b2", (self.out,))):
            if key.startswith("W"):
                fan = shape[0]
                parts.append(self.init_gain * torch.randn(*shape, generator=g,
                                                          dtype=dtype)
                             / fan ** 0.5)
            else:
                parts.append(torch.zeros(*shape, dtype=dtype))
        # linear skip = identity: the prior is the quadratic bowl
        A = torch.zeros(self.d, self.out, dtype=dtype)
        A[:, :self.d] = torch.eye(self.d, dtype=dtype)
        parts.append(A)
        # dissipation head starts at the hand-designed isotropic damper, so the
        # prior is a controller that already works rather than noise
        parts.append(torch.zeros(self.n_obs, self.d * self.d, dtype=dtype))
        parts.append((self.damp0 ** 0.5
                      * torch.eye(self.d, dtype=dtype)).reshape(-1))
        return torch.cat([p.reshape(-1) for p in parts]).to(device)

    # --- heads --------------------------------------------------------------
    def _h(self, theta, e, z):
        """Trunk. Returns (value, d(value)/d(e)) -- the Jacobian is written out
        rather than taken with autodiff so the term composes under vmap/jacrev
        without nesting transforms."""
        W1 = self._p(theta, "W1", (self.n_in, self.h))
        b1 = self._p(theta, "b1", (self.h,))
        W2 = self._p(theta, "W2", (self.h, self.out))
        b2 = self._p(theta, "b2", (self.out,))
        A = self._p(theta, "A", (self.d, self.out))
        inp = torch.cat([e / self.e_scale, z], dim=-1)
        a1 = torch.einsum("...i,...ih->...h", inp, W1) + b1
        t1 = torch.tanh(a1)
        y = torch.einsum("...h,...ho->...o", t1, W2) + b2 \
            + torch.einsum("...i,...io->...o", e, A)
        # d y / d e = A + W1[:d] * (1 - t1^2) * W2 , scaled
        dt = 1.0 - t1 * t1
        J = A + torch.einsum("...ih,...h,...ho->...io",
                             W1[..., :self.d, :], dt, W2) / self.e_scale
        return y, J

    def _L(self, theta, z):
        Wd = self._p(theta, "Wd", (self.n_obs, self.d * self.d))
        bd = self._p(theta, "bd", (self.d * self.d,))
        flat = torch.einsum("...i,...ij->...j", z, Wd) + bd
        return flat.reshape(flat.shape[:-1] + (self.d, self.d))

    def _read(self, obs):
        if obs is None or self.sensor_name not in obs:
            return None
        return (obs[self.sensor_name] / self.obs_scale).clamp(-4.0, 4.0)

    # --- contributions ------------------------------------------------------
    def potential(self, theta, e, v, x, obs=None):
        z = self._read(obs)
        if z is None:
            return torch.zeros_like(e[..., 0])
        y, _ = self._h(theta, e, z)
        y0, _ = self._h(theta, torch.zeros_like(e), z)
        g = y - y0
        return (g * g).sum(-1)

    def grad_potential(self, theta, e, v, x, obs=None):
        z = self._read(obs)
        if z is None:
            return torch.zeros_like(e)
        y, J = self._h(theta, e, z)
        y0, _ = self._h(theta, torch.zeros_like(e), z)
        g = y - y0
        # grad_e ||g||^2 = 2 J g ; vanishes at e = 0 because g does
        gradV = 2.0 * torch.einsum("...io,...o->...i", J, g)
        L = self._L(theta, z)
        Lv = torch.einsum("...ji,...j->...i", L, v)
        dRdv = torch.einsum("...ij,...j->...i", L, Lv)     # (L L^T) v
        return gradV + dRdv

    def damping(self, theta: Tensor) -> Tensor:
        """Reported at zero observation, for `describe`."""
        z = torch.zeros(theta.shape[:-1] + (self.n_obs,), dtype=theta.dtype,
                        device=theta.device)
        L = self._L(theta, z)
        return L @ L.transpose(-1, -2)

    def certificate(self, theta: Tensor, goal: Optional[Tensor] = None) -> Dict:
        return {"kind": self.kind, "psd": True, "zero_at_goal": True,
                "bounded_grad": False, "learned": True,
                "params": self.dim}

    def describe(self, theta: Tensor) -> Dict[str, float]:
        eig = torch.linalg.eigvalsh(self.damping(theta))
        return {"lrn_Kd_min": float(eig[0]), "lrn_Kd_max": float(eig[-1]),
                "lrn_wnorm": float(theta.norm())}
