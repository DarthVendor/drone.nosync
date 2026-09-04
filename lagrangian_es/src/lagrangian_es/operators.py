"""Variation and selection.

`theta` is an opaque [trainable.dim] vector here; this module touches neither the
system's nor the trainable's internals.

The whitened and isotropic arms flow through exactly the same code path and differ
only in `P`.  With `P = I`, `mirrored_offspring` *is* the isotropic baseline --
not a reimplementation of it -- which is what makes the 2x2 a controlled
comparison rather than two separately-tuned algorithms.
"""
from __future__ import annotations

import torch
from torch import Tensor


def mirrored_offspring(mean: Tensor, sigma: float, P: Tensor, pop: int,
                       gen: torch.Generator) -> Tensor:
    """[pop, dim] genomes, antithetically paired.

    delta = sigma * (z @ P) with z ~ N(0, I); since P is symmetric with
    P^2 = (G + ridge)^-1 up to the length rescale, delta ~ N(0, sigma^2 G^-1):
    variation isotropic in energy rather than in parameter coordinates.
    """
    if pop % 2:
        raise ValueError(f"pop must be even for mirrored sampling, got {pop}")
    dim = mean.shape[-1]
    z = torch.randn(pop // 2, dim, generator=gen, dtype=mean.dtype, device=mean.device)
    d = sigma * (z @ P)
    return mean + torch.cat([d, -d], dim=0)


def rank_weights(mu: int, dtype=torch.float64, device="cpu") -> Tensor:
    """CMA-style log weights w_i ~ log(mu + 1/2) - log(i), normalized."""
    i = torch.arange(1, mu + 1, dtype=dtype, device=device)
    w = torch.log(torch.tensor(mu + 0.5, dtype=dtype, device=device)) - torch.log(i)
    return w / w.sum()


def recombine(TH: Tensor, fitness: Tensor, elite_frac: float):
    """Rank-based recombination over the top `elite_frac`.  Lower fitness = better.

    Returns (new_mean, elite_idx, elite_fitness).
    """
    pop = TH.shape[0]
    mu = max(1, int(round(elite_frac * pop)))
    order = torch.argsort(fitness)
    elite = order[:mu]
    w = rank_weights(mu, dtype=TH.dtype, device=TH.device)
    return (w[:, None] * TH[elite]).sum(dim=0), elite, fitness[elite]


def update_sigma(sigma: float, improved: bool, grow: float = 1.06, shrink: float = 0.97,
                 lo: float = 1e-3, hi: float = 0.5) -> float:
    return float(min(max(sigma * (grow if improved else shrink), lo), hi))


# --------------------------------------------------------------------------- #
# Genetic operators: a persistent population rather than a single moving mean.
#
# These share the SAME whitened variation operator as the ES arm, so "structured
# genome", "whitened variation" and "population scheme" stay three independent
# design choices that can be ablated separately.
# --------------------------------------------------------------------------- #
def tournament_select(fitness: Tensor, n: int, k: int, gen: torch.Generator) -> Tensor:
    """[n] indices, each the best of k uniformly drawn contenders (lower = better)."""
    N = fitness.shape[0]
    cand = torch.randint(N, (n, k), generator=gen, device=fitness.device)
    return cand.gather(1, fitness[cand].argmin(dim=1, keepdim=True)).squeeze(1)


def whitened_crossover(a: Tensor, b: Tensor, alpha: float, P: Tensor, P_inv: Tensor,
                       gen: torch.Generator) -> Tensor:
    """BLX-alpha, performed in the whitened frame.

    Plain coordinate-wise BLX draws its box along the *parameter* axes, which
    reintroduces exactly the coordinate dependence the metric exists to remove.
    Mapping the parents through P^-1 first aligns the box with the principal axes
    of G, so recombination mixes parents along energetically meaningful directions.
    With P = I this reduces to textbook BLX-alpha, keeping the arms comparable.
    """
    ya, yb = a @ P_inv, b @ P_inv
    u = torch.rand(ya.shape, generator=gen, dtype=ya.dtype, device=ya.device)
    u = u * (1.0 + 2.0 * alpha) - alpha          # U(-alpha, 1 + alpha)
    return (ya + u * (yb - ya)) @ P


def whitened_mutation(TH: Tensor, sigma: float, P: Tensor, gen: torch.Generator) -> Tensor:
    """delta ~ N(0, sigma^2 G^-1), applied per individual (no mirroring)."""
    z = torch.randn(TH.shape, generator=gen, dtype=TH.dtype, device=TH.device)
    return TH + sigma * (z @ P)


def ga_step(TH: Tensor, fitness: Tensor, sigma: float, P: Tensor, P_inv: Tensor,
            gen: torch.Generator, elitism: int = 2, tournament_k: int = 3,
            crossover_rate: float = 0.7, blx_alpha: float = 0.5):
    """One generational GA step: elitism + tournament + crossover + mutation.

    Fully vectorized -- a Python loop over individuals would dominate the runtime,
    since a whole population rollout costs well under a second.
    """
    N = TH.shape[0]
    elitism = max(0, min(int(elitism), N - 2))
    order = torch.argsort(fitness)
    elites = TH[order[:elitism]]

    n_child = N - elitism
    pa = tournament_select(fitness, n_child, tournament_k, gen)
    pb = tournament_select(fitness, n_child, tournament_k, gen)

    crossed = whitened_crossover(TH[pa], TH[pb], blx_alpha, P, P_inv, gen)
    do_cross = torch.rand(n_child, generator=gen, dtype=TH.dtype,
                          device=TH.device) < crossover_rate
    children = torch.where(do_cross[:, None], crossed, TH[pa])
    children = whitened_mutation(children, sigma, P, gen)
    return torch.cat([elites, children], dim=0), order


def segment_crossover(A: Tensor, B: Tensor, segments, gen: torch.Generator,
                      rate: float = 0.5) -> Tensor:
    """Exchange whole genome segments between paired parents.

    This is the GP-style structural recombination that weight-level crossover
    cannot provide.  Swapping raw coordinates between two networks is close to
    meaningless because their weights carry competing conventions -- unit 7 in one
    parent has nothing to do with unit 7 in the other.  Lagrangian terms have no
    such convention: each is an independently valid, independently certified
    contribution to L_d, so a child assembled from either parent's terms is still
    a well-formed energy-shaping controller.

    Safety here is exactly the composition invariant: because every term
    preserves nonnegativity and the equilibrium on its own, any conic mixture of
    terms drawn from two valid parents is itself valid.
    """
    n = A.shape[0]
    segs = list(segments)
    take_b = torch.rand(n, len(segs), generator=gen, dtype=A.dtype,
                        device=A.device) < rate
    out = A.clone()
    for j, sl in enumerate(segs):
        m = take_b[:, j: j + 1]
        out[:, sl] = torch.where(m, B[:, sl], A[:, sl])
    return out


def ga_step_structured(TH: Tensor, fitness: Tensor, sigma: float, P: Tensor,
                       P_inv: Tensor, gen: torch.Generator, segments,
                       elitism: int = 2, tournament_k: int = 3,
                       crossover_rate: float = 0.7, blx_alpha: float = 0.5,
                       segment_rate: float = 0.5, mode: str = "mixed"):
    """A GA step that can recombine structurally as well as numerically.

    `mode`:
      "blx"     -- whitened BLX-alpha only (numeric interpolation)
      "segment" -- whole-term exchange only (structural)
      "mixed"   -- half the children each way, which is usually what you want:
                   BLX refines within a structure, segment swaps explore across
                   structures, and neither alone does both.
    """
    N = TH.shape[0]
    elitism = max(0, min(int(elitism), N - 2))
    order = torch.argsort(fitness)
    elites = TH[order[:elitism]]
    n_child = N - elitism

    pa = tournament_select(fitness, n_child, tournament_k, gen)
    pb = tournament_select(fitness, n_child, tournament_k, gen)
    A, B = TH[pa], TH[pb]

    blx = whitened_crossover(A, B, blx_alpha, P, P_inv, gen)
    seg = segment_crossover(A, B, segments, gen, rate=segment_rate)
    if mode == "blx":
        crossed = blx
    elif mode == "segment":
        crossed = seg
    elif mode == "mixed":
        use_seg = torch.rand(n_child, 1, generator=gen, dtype=TH.dtype,
                             device=TH.device) < 0.5
        crossed = torch.where(use_seg, seg, blx)
    else:
        raise ValueError(f"unknown crossover mode {mode!r}")

    do_cross = torch.rand(n_child, generator=gen, dtype=TH.dtype,
                          device=TH.device) < crossover_rate
    children = torch.where(do_cross[:, None], crossed, A)
    children = whitened_mutation(children, sigma, P, gen)
    return torch.cat([elites, children], dim=0), order
