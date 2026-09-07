"""Widen a `LearnedShaping` genome without changing the controller it encodes.

The capacity question -- does one 16-unit layer limit how low the crash rate can
go -- is only answerable if the wide controller STARTS as the narrow one.  A
freshly initialised wide net would be measuring the restart, not the capacity.

New hidden units are given live input weights but ZERO output weights, so they
contribute exactly nothing at first and the search can grow into them.  That is
an identity on the function and the test below checks it as one, to full double
precision rather than to a tolerance.
"""
import torch
from torch import Tensor

from lagrangian_es.trainables.learned import LearnedShaping


def widen(theta: Tensor, narrow: LearnedShaping, wide: LearnedShaping,
          gen: torch.Generator, scale: float = 0.1) -> Tensor:
    """Re-lay `theta` from `narrow`'s slot map into `wide`'s, function intact."""
    if wide.h < narrow.h:
        raise ValueError(f"widen only grows: {narrow.h} -> {wide.h}")
    if (narrow.n_in, narrow.out, narrow.d) != (wide.n_in, wide.out, wide.d):
        raise ValueError("only the hidden width may differ")
    out = torch.empty(theta.shape[:-1] + (wide.dim,), dtype=theta.dtype,
                      device=theta.device)

    def get(key):
        a, b = narrow._sl[key]
        return theta[..., a:b]

    def put(key, val):
        a, b = wide._sl[key]
        out[..., a:b] = val.reshape(val.shape[:-2] + (-1,)) \
            if val.dim() > theta.dim() - 1 else val

    h, H, n_in, o = narrow.h, wide.h, narrow.n_in, narrow.out
    lead = theta.shape[:-1]
    W1 = get("W1").reshape(lead + (n_in, h))
    b1 = get("b1")
    W2 = get("W2").reshape(lead + (h, o))
    extra = H - h

    W1n = torch.empty(lead + (n_in, H), dtype=theta.dtype, device=theta.device)
    W1n[..., :h] = W1
    b1n = torch.empty(lead + (H,), dtype=theta.dtype, device=theta.device)
    b1n[..., :h] = b1
    W2n = torch.zeros(lead + (H, o), dtype=theta.dtype, device=theta.device)
    W2n[..., :h, :] = W2
    if extra:
        # live inputs, dead outputs: the unit is wired up but silent
        W1n[..., h:] = scale * torch.randn(lead + (n_in, extra),
                                           generator=gen, dtype=theta.dtype)
        b1n[..., h:] = scale * torch.randn(lead + (extra,), generator=gen,
                                           dtype=theta.dtype)

    a, b = wide._sl["W1"]; out[..., a:b] = W1n.reshape(lead + (-1,))
    a, b = wide._sl["b1"]; out[..., a:b] = b1n
    a, b = wide._sl["W2"]; out[..., a:b] = W2n.reshape(lead + (-1,))
    for key in ("b2", "A", "Wd", "bd"):
        a, b = wide._sl[key]
        out[..., a:b] = get(key)
    return out


def extend_policy(theta: Tensor, old_policy_dim: int,
                  new_policy_dim: int) -> Tensor:
    """Grow the policy block of a genome, keeping the allocator where it belongs.

    A genome is laid out `[policy | allocator]`, so appending slots for a new
    head at the END silently shifts the allocator into them and leaves the
    allocator itself as whatever the padding was.  The gains that point the
    thrust are then zero, and the vehicle does not fly: measured, a warm start
    of 0.751 progress / 0.228 crash read back as 0.001 / 1.000 -- which looks
    exactly like a broken head rather than a misplaced pad.
    """
    if new_policy_dim < old_policy_dim:
        raise ValueError(f"policy only grows: {old_policy_dim} -> {new_policy_dim}")
    pad = new_policy_dim - old_policy_dim
    if pad == 0:
        return theta.clone()
    zeros = torch.zeros(theta.shape[:-1] + (pad,), dtype=theta.dtype,
                        device=theta.device)
    return torch.cat([theta[..., :old_policy_dim], zeros,
                      theta[..., old_policy_dim:]], dim=-1)


def to_isotropic_damping(theta: Tensor, full, iso, target: float) -> Tensor:
    """Re-lay a genome from a full dissipation head onto an isotropic one.

    The trunk -- everything shaping V -- is copied verbatim, so the potential is
    untouched and only the damper changes character.  The new head is given zero
    weights and a bias chosen so that R = target * I, matching the average
    damping the full head was producing; the comparison is then about the SHAPE
    of the dissipation rather than its size.
    """
    import math

    if theta.shape[-1] < full.dim:
        raise ValueError("genome smaller than the full term")
    out = torch.zeros(theta.shape[:-1] + (theta.shape[-1] - full.dim + iso.dim,),
                      dtype=theta.dtype, device=theta.device)
    # everything before the damping head is laid out identically
    cut = full._sl["Wd"][0]
    if iso._sl["Wd"][0] != cut:
        raise ValueError("the two terms disagree about where the damper starts")
    out[..., :cut] = theta[..., :cut]
    # softplus(b) = target  ->  b = log(exp(target) - 1), stable for large target
    t = float(target)
    b = t if t > 20 else math.log(math.expm1(t))
    a0, b0 = iso._sl["bd"]
    out[..., a0:b0] = b
    # the allocator, and anything else that followed the term, keeps its place
    out[..., iso.dim:] = theta[..., full.dim:]
    return out


def to_beam_damping(theta: Tensor, full, beams, floor: float,
                    gen: torch.Generator, scale: float = 0.1) -> Tensor:
    """Re-lay a genome from a full dissipation head onto the per-beam one.

    The trunk is copied verbatim.  The floor `s0` is set to `floor` -- pass the
    full head's smallest eigenvalue, the omnidirectional damping it was already
    applying -- and the per-beam net starts small and random with a zero output
    bias, so every beam begins with the same modest weight and the search
    decides what a close beam is worth.
    """
    import math

    cut = full._sl["Wd"][0]
    if beams._sl["Wd"][0] != cut:
        raise ValueError("the two terms disagree about where the damper starts")
    out = torch.zeros(theta.shape[:-1] + (theta.shape[-1] - full.dim + beams.dim,),
                      dtype=theta.dtype, device=theta.device)
    out[..., :cut] = theta[..., :cut]
    a, b = beams._sl["Wd"]
    H = beams.BEAM_H
    net = scale * torch.randn(theta.shape[:-1] + (3 * H + 1,), generator=gen,
                              dtype=theta.dtype)
    net[..., 3 * H] = 0.0                       # output bias
    out[..., a:b] = net
    t = float(floor)
    a0, b0 = beams._sl["bd"]
    out[..., a0:b0] = t if t > 20 else math.log(math.expm1(t))
    out[..., beams.dim:] = theta[..., full.dim:]
    return out
