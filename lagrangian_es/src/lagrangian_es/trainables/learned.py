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

import math
import torch
from torch import Tensor

from .terms import LagrangianTerm


class LearnedShaping(LagrangianTerm):
    """MLP-shaped `V_d` and `K_d`, with the certificate enforced structurally."""

    kind = "learned_shaping"
    uses_obs = True
    BEAM_H = 6          # hidden width of the per-beam damping net, `beams` mode
    HINGE = 0.1         # smoothing width of the closing-speed hinge, m/s

    def __init__(self, d: int, sensor_name: str = "range", n_obs: int = 12,
                 hidden: int = 16, out: int = 6, obs_scale: float = 4.0,
                 e_scale: float = 2.0, v_gate: bool = True,
                 init_gain: float = 0.4, damp0: float = 1.2,
                 gyro: bool = False, obs_transform: str = "linear",
                 prox_scale: float = 1.0, damp_mode: str = "full",
                 pull_max: float = 0.0,
                 n_beams: int = None, part: str = "all"):
        super().__init__(d)
        # `part`: "all" is the whole term; "potential" carries only the shaped
        # potential (W1, b1, W2, b2, A) and "damping" only the dissipation
        # head (Wd, bd, the gyro).  Two parts laid end to end are the whole
        # term's genome exactly, so a controller splits into a PULL term and a
        # BRAKE term with separate priorities without changing a flight.
        # "heading" is a third kind: a small net from the observation to a
        # heading OFFSET the allocator's look-at is rotated by, so the low
        # level can turn the body -- and the body-fixed sensors -- from what
        # its beams see.  Zero at init: the look-at prior is unchanged.
        if part not in ("all", "potential", "damping", "heading"):
            raise ValueError(f"unknown part {part!r}")
        self.part = part
        self.sensor_name = sensor_name
        self.n_obs, self.h, self.out = int(n_obs), int(hidden), int(out)
        # the first `n_beams` channels are ranges; anything after is another
        # sensor (a downward fan, the tilt) that reads through unscaled
        self.n_beams = int(n_obs if n_beams is None else n_beams)
        self.beam_name = sensor_name[0] if isinstance(sensor_name, (tuple, list)) else sensor_name
        self.obs_scale, self.e_scale = float(obs_scale), float(e_scale)
        self.init_gain, self.damp0 = float(init_gain), float(damp0)
        self.gyro = bool(gyro)
        if damp_mode not in ("full", "iso", "beams"):
            raise ValueError(f"unknown damp_mode {damp_mode!r}")
        self.damp_mode = damp_mode
        if obs_transform not in ("linear", "proximity"):
            raise ValueError(f"unknown obs_transform {obs_transform!r}")
        self.obs_transform = obs_transform
        self.prox_scale = float(prox_scale)
        # The pull's force budget (0 = unbounded).  V = |g|^2 near the goal and
        # Huber beyond: linear in |g| once 2|g| would exceed `pull_max`, so the
        # gradient's magnitude saturates at the budget instead of growing with
        # distance.  Same potential, same learned trunk; only the far field
        # changes, and it changes so that a brake CAN win -- see
        # `Quadrotor.pull_budget`.
        self.r0 = 0.5 * float(pull_max) if pull_max else None
        self.n_in = self.d + self.n_obs
        slots = []
        if self.part in ("all", "potential"):
            slots += [("W1", self.n_in * self.h), ("b1", self.h),
                      ("W2", self.h * self.out), ("b2", self.out),
                      ("A", self.d * self.out)]
        if self.part in ("all", "damping"):
            slots += [("Wd", {"iso": self.n_obs, "beams": 3 * self.BEAM_H + 1}
                             .get(damp_mode, self.n_obs * self.d * self.d)),
                      ("bd", 1 if damp_mode in ("iso", "beams") else self.d * self.d)]
        if self.part == "heading":
            slots += [("Wy", self.n_obs * self.BEAM_H), ("by", self.BEAM_H), ("Wy2", self.BEAM_H), ("by2", 1)]
        if self.gyro and self.part in ("all", "damping"):
            # LAST, so a genome trained without the head extends to one with it
            # by appending zeros -- which is the same controller exactly, since
            # a zero G gives a zero S.
            slots += [("Wg", self.n_obs * (self.d * self.d)),
                      ("bg", self.d * self.d)]
        n, self._sl = 0, {}
        for key, size in slots:
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
        if self.part == "heading":
            Wy = self.init_gain * torch.randn(self.n_obs, self.BEAM_H, generator=g, dtype=dtype) / self.n_obs ** 0.5
            out = torch.cat([Wy.reshape(-1), torch.zeros(self.BEAM_H, dtype=dtype),
                             torch.zeros(self.BEAM_H, dtype=dtype), torch.zeros(1, dtype=dtype)])   # output weights zero: no offset
            assert out.numel() == self.dim
            return out.to(device)
        if self.part in ("all", "potential"):
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
        if self.part == "potential":
            out = torch.cat([p.reshape(-1) for p in parts])
            assert out.numel() == self.dim, (out.numel(), self.dim)
            return out.to(device)
        # dissipation head starts at the hand-designed isotropic damper, so the
        # prior is a controller that already works rather than noise.  Laid
        # out per MODE: the full head is (n_obs x d^2) + d^2 with an identity
        # block, the isotropic and per-beam heads carry their own weight slot
        # count and ONE scalar read through a softplus.  Building the full
        # layout for every mode made `init()` 205 slots longer than `dim` in
        # the per-beam mode, and the slices read the first `dim` of them
        # without complaint -- a fresh genome that was silently the wrong one.
        import math
        wd = self._sl["Wd"][1] - self._sl["Wd"][0]
        bd = self._sl["bd"][1] - self._sl["bd"][0]
        parts.append(torch.zeros(wd, dtype=dtype))
        if self.damp_mode == "full":
            parts.append((self.damp0 ** 0.5
                          * torch.eye(self.d, dtype=dtype)).reshape(-1))
        else:
            # softplus(b) = damp0  ->  b = log(exp(damp0) - 1)
            parts.append(torch.full((bd,), math.log(math.expm1(self.damp0)), dtype=dtype))
        out = torch.cat([p.reshape(-1) for p in parts])
        if self.gyro:
            out = torch.cat([out, torch.zeros(self.dim - out.numel(), dtype=dtype)])
        assert out.numel() == self.dim, (out.numel(), self.dim)
        return out.to(device)

    # --- heads --------------------------------------------------------------
    def _weights(self, theta):
        """The trunk's parameters, sliced out once.

        `_h` runs TWICE per step -- for h(e, obs) and for h(0, obs), whose
        difference is what makes V vanish at the goal -- and each call was
        re-slicing and re-reshaping the same five tensors.  Profiled on the
        default rig that was 12 `_p` calls and 15 reshapes per step producing
        identical results.
        """
        return (self._p(theta, "W1", (self.n_in, self.h)),
                self._p(theta, "b1", (self.h,)),
                self._p(theta, "W2", (self.h, self.out)),
                self._p(theta, "b2", (self.out,)),
                self._p(theta, "A", (self.d, self.out)))

    def _h(self, theta, e, z, w=None):
        """Trunk. Returns (value, d(value)/d(e)) -- the Jacobian is written out
        rather than taken with autodiff so the term composes under vmap/jacrev
        without nesting transforms."""
        W1, b1, W2, b2, A = self._weights(theta) if w is None else w
        inp = torch.cat([e / self.e_scale, z], dim=-1)
        # matmul, not einsum: these are batched MATVECS and go straight to
        # bmm, where einsum pays its equation-parsing and backend-check
        # overhead on every call.  Measured on the trunk's four contractions:
        # 1.23x together, 2.09x on the smallest.  (The hand-designed terms
        # measured the other way round -- einsum won there -- so this is a
        # shape-by-shape fact, not a rule.)
        a1 = (inp.unsqueeze(-2) @ W1).squeeze(-2) + b1
        t1 = torch.tanh(a1)
        y = (t1.unsqueeze(-2) @ W2).squeeze(-2) + b2 \
            + (e.unsqueeze(-2) @ A).squeeze(-2)
        # d y / d e = A + W1[:d] * (1 - t1^2) * W2 , scaled
        dt = 1.0 - t1 * t1
        J = A + ((W1[..., :self.d, :] * dt.unsqueeze(-2)) @ W2) / self.e_scale
        return y, J

    def _h_z(self, theta, e, z, w=None):
        """Trunk value and its Jacobians with respect to the error AND the
        observation, one pass: (y, J_e [..., d, out], J_z [..., n_obs, out])."""
        W1, b1, W2, b2, A = self._weights(theta) if w is None else w
        inp = torch.cat([e / self.e_scale, z], dim=-1)
        a1 = (inp.unsqueeze(-2) @ W1).squeeze(-2) + b1
        t1 = torch.tanh(a1)
        y = (t1.unsqueeze(-2) @ W2).squeeze(-2) + b2 + (e.unsqueeze(-2) @ A).squeeze(-2)
        dt = (1.0 - t1 * t1).unsqueeze(-2)
        Je = A + ((W1[..., :self.d, :] * dt) @ W2) / self.e_scale
        Jz = (W1[..., self.d:, :] * dt) @ W2
        return y, Je, Jz

    def _h_z2(self, theta, e, z, w=None):
        """The trunk at e AND at e = 0 in ONE pass (the two rows share the
        weights, so they go through the batched matmuls together): returns
        (y, Je, Jz, y0, Jz0).  The potential is |y - y0|^2, so every hot path
        needed both, and two separate passes were 1.0 s of small-op overhead
        per 51k flight-steps."""
        W1, b1, W2, b2, A = self._weights(theta) if w is None else w
        zero = torch.zeros_like(e)
        inp = torch.stack([torch.cat([e / self.e_scale, z], dim=-1), torch.cat([zero, z], dim=-1)], dim=-2)   # [..., 2, n_in]
        a1 = inp @ W1 + b1.unsqueeze(-2)
        t1 = torch.tanh(a1)                                                                                  # [..., 2, h]
        lin = torch.stack([(e.unsqueeze(-2) @ A).squeeze(-2), torch.zeros_like(e[..., :1]).expand(*e.shape[:-1], A.shape[-1])], dim=-2)
        yy = t1 @ W2 + b2.unsqueeze(-2) + lin                                                               # [..., 2, out]
        dt = 1.0 - t1 * t1
        Je = A + ((W1[..., :self.d, :] * dt[..., 0:1, :]) @ W2) / self.e_scale
        Jz = (W1[..., self.d:, :] * dt[..., 0:1, :]) @ W2
        Jz0 = (W1[..., self.d:, :] * dt[..., 1:2, :]) @ W2
        return yy[..., 0, :], Je, Jz, yy[..., 1, :], Jz0

    def _pull_fac(self, g):
        """dH/d(s^2) for the Huber potential H(s), s = |g|: 1 inside r0, r0/s
        beyond -- multiplies g wherever V = |g|^2 would use it."""
        if self.r0 is None:
            return None
        s = torch.linalg.vector_norm(g, dim=-1, keepdim=True)
        return (self.r0 / s.clamp_min(1e-12)).clamp(max=1.0)

    def _dz_dpsi(self, obs):
        """dz/dpsi for every body-fixed observation, in `_obs_vec` order."""
        names = self.sensor_name if isinstance(self.sensor_name, (tuple, list)) else (self.sensor_name,)
        parts = []
        for k, name in enumerate(names):
            raw = obs[name]; width = raw.shape[-1]
            J = obs.get(name + "/J"); dirs = obs.get(name + "/dir")
            if J is not None and dirs is not None:
                if k == 0 and self.obs_transform == "proximity":
                    dzdr = -self.prox_scale / (raw.clamp_min(0.0) + self.prox_scale) ** 2
                elif k == 0:
                    dzdr = ((raw / self.obs_scale).abs() < 4.0).to(raw.dtype) / self.obs_scale
                else:
                    dzdr = (raw.abs() < 4.0).to(raw.dtype)
                parts.append(dzdr * raw * (J[..., 1] * dirs[..., 0] - J[..., 0] * dirs[..., 1]))
            elif width == 3:
                inside = (raw.abs() < 4.0).to(raw.dtype)
                parts.append(inside * torch.stack([-raw[..., 1], raw[..., 0], torch.zeros_like(raw[..., 2])], -1))
            else:
                parts.append(torch.zeros_like(raw))
        return torch.cat(parts, -1)

    def grad_potential_both(self, theta, e, v, x, obs=None):
        """(dV/de + dR/dv, dV/dpsi) from ONE pair of trunk passes -- the hot
        path calls this rather than the two gradients separately."""
        zero_psi = torch.zeros(theta.shape[:-1], dtype=theta.dtype, device=theta.device)
        z = self._read(obs)
        if z is None or self.part == "heading":
            return torch.zeros_like(e), zero_psi
        out = torch.zeros_like(e); dpsi = zero_psi
        if self.part in ("all", "potential"):
            w = self._weights(theta)
            y, Je, Jz, y0, Jz0 = self._h_z2(theta, e, z, w)
            g = y - y0
            fac = self._pull_fac(g)
            if fac is not None:
                g = g * fac
            out = out + 2.0 * (Je @ g.unsqueeze(-1)).squeeze(-1)
            if obs is not None and (self.beam_name + "/J") in obs:
                dVdz = 2.0 * ((Jz - Jz0) @ g.unsqueeze(-1)).squeeze(-1)
                dpsi = (dVdz * self._dz_dpsi(obs)).sum(-1)
        if self.part == "potential":
            return out, dpsi
        if self.damp_mode == "beams":
            dRdv = self._dRdv_beams(theta, z, v, obs)
        else:
            L = self._L(theta, z)
            Lv = (v.unsqueeze(-2) @ L).squeeze(-2)
            dRdv = (L @ Lv.unsqueeze(-1)).squeeze(-1)
        out = out + dRdv
        if self.gyro:
            out = out + (self._S(theta, z) @ v.unsqueeze(-1)).squeeze(-1)
        return out, dpsi

    def grad_potential_yaw(self, theta, e, v, x, obs=None):
        """dV/dpsi: how the potential changes when the body -- and with it the
        front fan -- rotates about world z.  Implicit in the Lagrangian: V reads
        the beams, the beams depend on the heading, so the same potential that
        pulls and brakes also turns.  Through the ray-cast's own Jacobian,
        dr_i/dpsi = r_i (J_i . (e_z x d_i)); nothing is added to V."""
        zero = torch.zeros(theta.shape[:-1], dtype=theta.dtype, device=theta.device)
        if self.part not in ("all", "potential") or obs is None:
            return zero
        z = self._read(obs)
        if z is None or (self.beam_name + "/J") not in obs:
            return zero
        w = self._weights(theta)
        y, _, Jz, y0, Jz0 = self._h_z2(theta, e, z, w)
        g = y - y0
        fac = self._pull_fac(g)
        if fac is not None:
            g = g * fac
        dVdz = 2.0 * ((Jz - Jz0) @ g.unsqueeze(-1)).squeeze(-1)                          # V = |y - y0|^2, [..., n_obs]
        return (dVdz * self._dz_dpsi(obs)).sum(-1)

    def _L(self, theta, z):
        """The dissipation factor, R = L L^T.

        `full` lets the beams shape a whole matrix, which is more expressive and
        turns out to be exploitable: R can only ever REMOVE energy, so an
        anisotropic R offers a cheap direction to travel in, and the search takes
        it.  Measured on the trained city controller, the velocity collects 19.9%
        of the damping available on the stiff axis where a randomly oriented
        heading would collect 46.5% -- below the null in 93% of samples, and the
        same before city training, so it is the parameterisation and not the map.
        The controller had learned enough braking to survive 94% of its crashes
        and was receiving 30% of it.

        `iso` makes R a scalar times the identity, so damping is the same in
        every direction and cannot be dodged.  It costs FEWER parameters, not
        more, which matters: added dimension has hurt more than added
        expressiveness here.
        """
        if self.damp_mode == "iso":
            Wd = self._p(theta, "Wd", (self.n_obs, 1))
            bd = self._p(theta, "bd", (1,))
            # softplus keeps it positive and smooth; sqrt because R = L L^T
            g = torch.nn.functional.softplus(
                (z.unsqueeze(-2) @ Wd).squeeze(-2) + bd)
            eye = torch.eye(self.d, dtype=theta.dtype, device=theta.device)
            return g.sqrt().unsqueeze(-1) * eye
        Wd = self._p(theta, "Wd", (self.n_obs, self.d * self.d))
        bd = self._p(theta, "bd", (self.d * self.d,))
        flat = (z.unsqueeze(-2) @ Wd).squeeze(-2) + bd
        return flat.reshape(flat.shape[:-1] + (self.d, self.d))

    def _obs_vec(self, obs):
        """The observation the network reads: one sensor, or several named
        sensors concatenated in the order given -- the front fan, a downward
        fan, the tilt -- so the low level flies on what it senses about the
        world AND about itself."""
        names = self.sensor_name if isinstance(self.sensor_name, (tuple, list)) else (self.sensor_name,)
        parts = [obs[n] for n in names if obs is not None and n in obs]
        if len(parts) != len(names):
            return None
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _read(self, obs):
        """Beam ranges, in the coordinate the network reasons in.

        `linear` divides by `obs_scale`, which spends the input range where the
        measurements are, not where they matter: over 0.2-6 m at obs_scale 4,
        the far half (2.5-6 m) occupies 0.875 of the input and the near half
        (0.2-1.2 m) only 0.05 -- seventeen times more resolution devoted to
        distances that cannot hurt the vehicle.

        `proximity` uses s / (d + s), which is 1 at contact and falls towards 0
        at range, inverting that ratio: the near half now gets 0.378 of the
        input against the far half's 0.143.  It is the coordinate a barrier is
        simple in, without prescribing a barrier -- the network still learns
        what to do with it, and the channel count is unchanged, which matters
        because dimension has measurably cost more than expressiveness here.
        """
        if obs is None or self._obs_vec(obs) is None:
            return None
        d = self._obs_vec(obs)
        beams, rest = d[..., :self.n_beams], d[..., self.n_beams:]
        if self.obs_transform == "proximity":
            beams = self.prox_scale / (beams.clamp_min(0.0) + self.prox_scale)
        else:
            beams = (beams / self.obs_scale).clamp(-4.0, 4.0)
        return beams if rest.shape[-1] == 0 else torch.cat([beams, rest.clamp(-4.0, 4.0)], -1)

    # --- contributions ------------------------------------------------------
    def heading(self, theta, obs=None):
        """A heading offset in radians from the observation, [...]; zero for
        parts without the head or with nothing observed."""
        z = self._read(obs) if self.part == "heading" else None
        if z is None:
            return torch.zeros(theta.shape[:-1], dtype=theta.dtype, device=theta.device)
        Wy = self._p(theta, "Wy", (self.n_obs, self.BEAM_H)); by = self._p(theta, "by", (self.BEAM_H,))
        Wy2 = self._p(theta, "Wy2", (self.BEAM_H,)); by2 = self._p(theta, "by2", (1,))
        h = torch.tanh((z.unsqueeze(-2) @ Wy).squeeze(-2) + by)
        return math.pi * torch.tanh((h * Wy2).sum(-1) + by2[..., 0])

    def potential(self, theta, e, v, x, obs=None):
        z = self._read(obs)
        if z is None or self.part in ("damping", "heading"):
            return torch.zeros_like(e[..., 0])
        w = self._weights(theta)
        y, _ = self._h(theta, e, z, w)
        y0, _ = self._h(theta, torch.zeros_like(e), z, w)
        g = y - y0
        s2 = (g * g).sum(-1)
        if self.r0 is None:
            return s2
        s = s2.clamp_min(0.0).sqrt()
        return torch.where(s <= self.r0, s2, 2.0 * self.r0 * s - self.r0 ** 2)

    def grad_potential(self, theta, e, v, x, obs=None):
        z = self._read(obs)
        if z is None or self.part == "heading":
            return torch.zeros_like(e)
        out = torch.zeros_like(e)
        if self.part in ("all", "potential"):
            w = self._weights(theta)
            y, J = self._h(theta, e, z, w)
            y0, _ = self._h(theta, torch.zeros_like(e), z, w)
            g = y - y0
            fac = self._pull_fac(g)
            if fac is not None:
                g = g * fac
            # grad_e ||g||^2 = 2 J g ; vanishes at e = 0 because g does
            out = out + 2.0 * (J @ g.unsqueeze(-1)).squeeze(-1)
        if self.part == "potential":
            return out
        if self.damp_mode == "beams":
            dRdv = self._dRdv_beams(theta, z, v, obs)
        else:
            L = self._L(theta, z)
            Lv = (v.unsqueeze(-2) @ L).squeeze(-2)
            dRdv = (L @ Lv.unsqueeze(-1)).squeeze(-1)      # (L L^T) v
        out = out + dRdv
        if self.gyro:
            # The workless head.  A gradient plus a Rayleigh dissipation flows
            # downhill into the nearest critical point, and on this map 94% of
            # the stalls ARE critical points -- saddles, with the vehicle pinned
            # 0.35 m off a building and the goal 16.5 m past it.  Going AROUND
            # needs a force across the motion, which no gradient can supply.
            #
            # S = G - G^T is skew for any G, so v . S v = 0 identically: the
            # head can steer but can never add energy, and H = T + V_d stays
            # non-increasing exactly as before.  That is the whole reason to
            # shape it this way rather than let the net emit a free force.
            out = out + (self._S(theta, z) @ v.unsqueeze(-1)).squeeze(-1)
        return out

    def _dRdv_beams(self, theta, z, v, obs):
        """One damping axis per beam, weighted by what that beam sees.

            R = 1/2 s0 |v|^2  +  1/2 sum_i w_i(z_i) h(-J_i . v)^2

        `J_i = d(range_i)/dx` is the world-frame direction away from whatever
        beam i sees, so `-J_i . v` is the closing speed on it and `h` -- a smooth
        hinge -- keeps the term one-sided: it resists approach and puts no drag
        on retreat.  The Jacobian is already computed by the rollout for any
        term with `uses_obs`, so reading it here costs nothing extra.

        Why this shape.  The full head R = L L^T points its stiff axis at the
        NEAREST wall (|cos| 0.80 to grad sdf, null 0.64), which is right, but in
        a tight pocket the nearest wall and the threatening one differ, and a
        single axis leaves the impact direction at the null share of the
        damping (21.3% of lambda_max).  The isotropic head fixes that by
        damping everything, and kills the tangential sliding that avoidance
        depends on (crash 0.038 -> 0.152).  Summing an axis per beam damps each
        close wall along its own normal and nothing else -- the structure of the
        hand-designed `RangeDamper`, with the per-beam weight LEARNED from the
        beam's own range through a net shared across beams, since no beam is
        special.  `s0` is a learned floor, because this stack has no other
        damper and free flight needs one.

        Dissipative for any weights: dR/dv . v = s0|v|^2 + sum w_i h h' c_i,
        and h, h' and c_i share a sign wherever h is non-zero.
        """
        H = self.BEAM_H
        wd = self._p(theta, "Wd", (3 * H + 1,))
        W1, b1, W2 = wd[..., :H], wd[..., H:2 * H], wd[..., 2 * H:3 * H]
        b2 = wd[..., 3 * H:3 * H + 1]
        s0 = torch.nn.functional.softplus(self._p(theta, "bd", (1,)))
        base = s0 * v
        J = obs.get(self.beam_name + "/J") if obs is not None else None
        if J is None:
            return base                     # no geometry in view: floor only
        z = z[..., :self.n_beams]                            # the beams, not the tilt
        t1 = torch.tanh(z.unsqueeze(-1) * W1 + b1)          # [..., n_beams, H]
        w = torch.nn.functional.softplus((t1 * W2).sum(-1) + b2)   # [..., n_obs]
        c = -(J * v.unsqueeze(-2)).sum(-1)                  # closing speed
        eps = self.HINGE
        root = torch.sqrt(c * c + eps * eps)
        h = 0.5 * (c + root)
        hp = 0.5 * (1.0 + c / root)
        # dR/dv = sum_i w_i h h' dc_i/dv = -sum_i w_i h h' J_i
        return base - ((w * h * hp).unsqueeze(-1) * J).sum(-2)

    def _S(self, theta, z):
        """Skew-symmetric gyroscopic matrix from the beams."""
        Wg = self._p(theta, "Wg", (self.n_obs, self.d * self.d))
        bg = self._p(theta, "bg", (self.d * self.d,))
        flat = (z.unsqueeze(-2) @ Wg).squeeze(-2) + bg
        G = flat.reshape(flat.shape[:-1] + (self.d, self.d))
        return G - G.transpose(-1, -2)

    def damping(self, theta: Tensor) -> Tensor:
        """Reported at zero observation, for `describe`."""
        if self.part in ("potential", "heading"):
            return torch.zeros(theta.shape[:-1] + (self.d, self.d), dtype=theta.dtype, device=theta.device)
        if self.damp_mode == "beams":
            s0 = torch.nn.functional.softplus(self._p(theta, "bd", (1,)))
            eye = torch.eye(self.d, dtype=theta.dtype, device=theta.device)
            return s0.unsqueeze(-1) * eye
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
