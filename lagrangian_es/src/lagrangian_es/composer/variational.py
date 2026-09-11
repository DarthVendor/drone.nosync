"""Probabilistic waypoints from a variational principle.

The composer's problem when it places a subgoal is the classical one: of all
the two-leg paths from where the vehicle is, through a subgoal, to the goal,
which is stationary?  Write the action of a candidate subgoal as its path
length through the sensed field,

    S[sub] = |sub - x| + |goal - sub| + lambda * P[sub]

where `P` is a barrier built from what the range beams actually returned.  With
no obstacles the first two terms are minimised by any point ON the straight
segment, so the action's stationary set IS the straight path -- nothing is
imposed, the geometry says it.  Put a wall in the way and the barrier bends the
stationary point around it, the same way Fermat's principle refracts a ray.

Waypoints are then SAMPLED rather than chosen: a candidate set drawn from the
policy's own Gaussian is scored by `S` and one is drawn with probability
proportional to `exp(-S / T)`.  At large `T` this is the policy untouched; at
small `T` it is the variational optimum over the candidates.

Why this exists: the composer's exploration was uniform over a 10 m ball in
(r, theta, phi), and measured on 256 paired flights at 20 m legs the advantage
distribution ran from -0.755 up to 0 with a p95 of EXACTLY zero -- fewer than
5% of flights were helped by anything it emitted, so the update, working
correctly, could only tell it to stop talking.  The gradient was not noisy
(disjoint halves of one batch agreed at cosine 0.93); there was simply nothing
positive in the data to learn from.  Concentrating the draws where the action
is low is what gives a random placement a chance of being useful.

Nothing here reads the map.  `P` comes from the vehicle's own beams, so this
stays inside the same constraint the rest of the composer obeys.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def beam_points(dirs: Tensor, rng: Tensor, max_range: float) -> Tuple[Tensor, Tensor]:
    """Where each beam struck, in the body frame, and whether it struck at all.

    `dirs` [n, 3] unit directions, `rng` [B, n] measured ranges.  A beam that
    returned its maximum saw nothing, and must not place an obstacle at the end
    of empty space.
    """
    hit = rng < (float(max_range) - 1e-3)
    pts = rng[..., None] * dirs[None]                      # [B, n, 3]
    return pts, hit


def barrier(sub: Tensor, dirs: Tensor, rng: Tensor, hit: Tensor, clear: float = 2.0,
            eps: float = 0.5, kappa: float = 0.034) -> Tensor:
    """Soft barrier from the beams, in two parts.

    OCCLUSION is the one that matters, and the one a proximity-only barrier
    misses.  A subgoal placed 8 m straight ahead of a wall seen at 4 m is
    BEHIND that wall -- unreachable -- yet it sits 4 m from the return point
    and so pays a proximity barrier almost nothing.  Measured on the first
    version: with a wall dead ahead the action still put its minimum at 0
    degrees.  So each beam charges a candidate that lies in roughly its
    direction and FARTHER than it saw: `softplus((|sub| - rng) / eps)`, gated
    by a soft angular window `exp((cos - 1) / kappa)` about the beam.  `kappa`
    0.034 is 1 - cos(15 deg), the ring's own beam spacing, so a candidate is
    charged by the beams that actually looked at it.

    CLEARANCE keeps a candidate off a wall it is not behind -- the subgoal
    should not sit inside the obstacle it is routing around.

    `sub` [B, K, 3], `dirs` [n, 3], `rng`/`hit` [B, n].
    """
    R = torch.linalg.vector_norm(sub, dim=-1).clamp_min(1e-6)              # [B,K]
    u = sub / R[..., None]
    cos = (u[:, :, None, :] * dirs[None, None]).sum(-1)                    # [B,K,n]
    w = torch.exp((cos - 1.0) / max(float(kappa), 1e-6))                   # angular window
    behind = torch.nn.functional.softplus(
        (R[:, :, None] - rng[:, None, :]) / max(float(eps), 1e-6))         # [B,K,n]
    occ = (w * behind)

    pts = rng[..., None] * dirs[None]                                      # [B,n,3]
    d = torch.linalg.vector_norm(sub[:, :, None, :] - pts[:, None, :, :], dim=-1)
    near = torch.nn.functional.softplus((float(clear) - d) / max(float(eps), 1e-6))

    return ((occ + near) * hit[:, None, :].to(occ.dtype)).sum(-1)          # [B,K]


def path_action(sub: Tensor, g: Tensor, dirs: Tensor, rng: Tensor, hit: Tensor,
                lam: float = 2.0, clear: float = 2.0) -> Tensor:
    """`S[sub] = |sub| + |g - sub| + lam * barrier`, in the body frame.

    The vehicle is at the origin of that frame, so `|sub|` is the first leg.
    Returns [B, K], in metres -- the barrier is scaled by `lam` into the same
    units so the trade-off between detouring and clearing a wall is explicit.
    """
    leg1 = torch.linalg.vector_norm(sub, dim=-1)
    leg2 = torch.linalg.vector_norm(g[:, None, :] - sub, dim=-1)
    return leg1 + leg2 + float(lam) * barrier(sub, dirs, rng, hit, clear=clear)


def boltzmann_pick(S: Tensor, temperature: float = 2.0,
                   gen: Optional[torch.Generator] = None) -> Tensor:
    """Index per row, drawn with probability proportional to exp(-S / T).

    `S` [B, K] -> [B].  The shift by each row's minimum keeps the exponential
    from underflowing when the candidates differ by many metres of action.
    """
    logits = -(S - S.min(dim=-1, keepdim=True).values) / max(float(temperature), 1e-6)
    p = torch.softmax(logits.to(torch.float64), dim=-1)
    return torch.multinomial(p, 1, generator=gen).squeeze(-1)
