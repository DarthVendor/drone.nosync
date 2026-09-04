"""The mechanical metric G(theta) and the preconditioner P it induces.

A generalized-force perturbation du costs energy du^T M(q)^-1 du.  Pulling that
quadratic form back through the controller map gives a Riemannian metric on
genome space,

    G(theta) = E_traj[ (du/dtheta)^T M(q)^-1 (du/dtheta) ] ,

and sampling offspring as dtheta ~ N(0, sigma^2 G^-1) makes variation isotropic in
*energy* rather than in parameter coordinates: large steps where they are
energetically cheap, small steps where a small genetic change produces a violent
actuator kick.

The expectation is taken over states the controller actually visits, not over a
uniform box: the metric has to describe where the vehicle operates.

Crucially, only the controller map is differentiated.  The integrator, the
liveness logic and (later) the contact model are never touched by autograd, so the
method stays gradient-free with respect to the dynamics -- which is the entire
reason to use evolutionary search on such systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor
from torch.func import jacrev, vmap

from .rollout import Rollout
from .util import make_gen, tree_index


@dataclass
class MetricResult:
    G: Tensor                   # [dim, dim] normalized, symmetric, PSD
    P: Tensor                   # [dim, dim] preconditioner, mean(lambda^-1/2) = 1
    P_inv: Tensor               # [dim, dim] its inverse -- maps genomes INTO the
                                # whitened frame, where GA crossover is performed
    eigs: Tensor                # [dim] eigenvalues of G after clamping (pre-ridge)
    cond: float                 # lambda_max / lambda_min after the ridge
    dist_from_identity: float   # ||P - I||_F
    saturation: float           # fraction of sampled actuator channels at a bound
    rank_frac: float            # fraction of eigenvalues above 1e-8
    n_states: int
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "metric_cond": self.cond,
            "metric_dist_I": self.dist_from_identity,
            "metric_sat": self.saturation,
            "metric_rank_frac": self.rank_frac,
            "metric_eig_max": float(self.eigs.max()),
            "metric_eig_min": float(self.eigs.min()),
        }


def identity_preconditioner(dim: int, dtype=torch.float64, device="cpu") -> MetricResult:
    """The isotropic arm.  Both arms flow through the same code path; they differ
    only in P, which is exactly what makes the ablation a controlled comparison."""
    eye = torch.eye(dim, dtype=dtype, device=device)
    return MetricResult(
        G=eye.clone(), P=eye.clone(), P_inv=eye.clone(),
        eigs=torch.ones(dim, dtype=dtype, device=device),
        cond=1.0, dist_from_identity=0.0, saturation=0.0, rank_frac=1.0, n_states=0,
    )


@torch.no_grad()
def _subsample(trace, n_states: int, gen: torch.Generator):
    """Pick n_states random (t, b) pairs, preferring states the vehicle was
    actually alive in -- a frozen crashed state is not part of the operating
    region the metric is meant to describe."""
    T, B = trace.goals.shape[0], trace.goals.shape[1]
    flat_alive = trace.alive.reshape(-1)
    idx_all = torch.arange(T * B, device=flat_alive.device)
    pool = idx_all[flat_alive]
    if pool.numel() < max(8, n_states // 8):      # nearly everything crashed
        pool = idx_all
    pick = pool[torch.randint(pool.numel(), (n_states,), generator=gen)]
    t_idx, b_idx = pick // B, pick % B

    # states has T+1 entries; trace.alive[t] is alive entering step t
    sub = {k: v[t_idx, b_idx] for k, v in trace.states.items()}
    return sub, trace.goals[t_idx, b_idx], trace.us[t_idx, b_idx]


@torch.no_grad()
def _precondition(G: Tensor, ridge: float):
    G = 0.5 * (G + G.transpose(-1, -2))
    scale = torch.diagonal(G).mean().clamp_min(1e-30)
    G = G / scale                                  # sigma keeps a stable meaning
    lam, V = torch.linalg.eigh(G)
    lam_c = lam.clamp_min(0.0)
    inv_sqrt = (lam_c + ridge).rsqrt()
    inv_sqrt = inv_sqrt / inv_sqrt.mean()          # step SHAPE changes, not length
    P = (V * inv_sqrt) @ V.transpose(-1, -2)
    P = 0.5 * (P + P.transpose(-1, -2))
    P_inv = (V * (1.0 / inv_sqrt)) @ V.transpose(-1, -2)
    P_inv = 0.5 * (P_inv + P_inv.transpose(-1, -2))
    lam_r = lam_c + ridge
    return G, P, P_inv, lam_c, float(lam_r.max() / lam_r.min())


def physics_metric(
    system,
    trainable,
    theta: Tensor,
    goals: Tensor,
    cfg,
    seed: int,
    n_states: int = 96,
    ridge: float = 1e-3,
    task=None,
    roll: Optional[Rollout] = None,
) -> MetricResult:
    """Estimate G at `theta` and return the whitening preconditioner.

    Warmup: the first `jacrev` traces the controller map and costs ~1 s; later
    calls are essentially free.  Do one throwaway call before timing anything.
    """
    if roll is None:
        from .tasks import WaypointPair
        roll = Rollout(system, trainable, task or WaypointPair(system), cfg)

    # 1. where the controller actually operates, under the current mean genome
    trace = roll.trace(theta[None], goals, seed)
    gen = make_gen(seed + 999_331)
    s_sub, goal_sub, u_sub = _subsample(trace, n_states, gen)

    # 2. differentiate the CONTROLLER MAP ALONE -- never the integrator
    Jac = vmap(jacrev(trainable.forward), in_dims=(None, 0, 0))(theta, s_sub, goal_sub)
    if not torch.isfinite(Jac).all():
        Jac = torch.nan_to_num(Jac, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. contract with the (possibly state-dependent) inverse mass.  This einsum
    #    is the only place the weak and strong forms of the claim differ, and they
    #    differ only by which system is plugged in.
    Minv = system.inv_mass(s_sub)
    S = Jac.shape[0]
    if system.dense_mass:
        G = torch.einsum("sid,sij,sje->de", Jac, Minv, Jac) / S
    else:
        G = torch.einsum("sid,si,sie->de", Jac, Minv, Jac) / S

    G, P, P_inv, lam, cond = _precondition(G, ridge)

    dim = theta.shape[-1]
    eye = torch.eye(dim, dtype=P.dtype, device=P.device)
    sat = float(system.saturation(u_sub, s_sub).mean())
    return MetricResult(
        G=G, P=P, P_inv=P_inv, eigs=lam, cond=cond,
        dist_from_identity=float((P - eye).norm()),
        saturation=sat,
        rank_frac=float((lam > 1e-8).to(torch.float64).mean()),
        n_states=S,
        diagnostics={"alive_frac": float(trace.alive.to(torch.float64).mean())},
    )
