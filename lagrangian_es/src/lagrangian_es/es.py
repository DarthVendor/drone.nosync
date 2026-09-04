"""Training loops.

Two search schemes over the same genome and the same variation operator:

  * `"es"` -- distribution based.  One moving mean, mirrored sampling, rank-weighted
    recombination.  This is the loop the paper's ablation runs.
  * `"ga"` -- a persistent population with tournament selection, whitened BLX
    crossover, elitism and per-individual mutation.  Batch-evaluated exactly like
    the ES arm.

Both take a system and a trainable and touch neither's internals; `theta` is an
opaque [trainable.dim] vector throughout.  Because they share `operators.py`, the
whitening claim can be tested under either scheme rather than being entangled
with one particular population model.

Two things worth not rediscovering the hard way:

  * Common random numbers.  Every member of a generation faces identical goals and
    identical reset noise (enforced inside `Rollout`).  Without it, ES variance
    swamps the effect the ablation measures.
  * Step-size adaptation must compare like with like.  Goals are resampled every
    generation, so "did the elite beat its own best-ever score" is confounded by
    task difficulty and decays sigma monotonically no matter how well search is
    going.  Both loops instead use a 1/5th-rule signal measured WITHIN a
    generation, against a reference genome evaluated on the very same goals -- the
    ES arm appends the parent to the batch, the GA arm re-evaluates its elites.
    That costs no extra rollout call: batch size is nearly free.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import Tensor

from .config import Config
from .metric import identity_preconditioner, physics_metric
from .operators import (
    ga_step_structured, mirrored_offspring, rank_weights, recombine,
    update_sigma, whitened_mutation,
)
from .rollout import Rollout
from .sensors import make_sensor
from .systems import make_system
from .tasks import make_task
from .trainables import make_trainable
from .util import gen_seed, make_gen, torch_dtype


@dataclass
class TrainResult:
    theta: Tensor                     # final genome to evaluate
    history: List[dict] = field(default_factory=list)
    cfg: Optional[Config] = None
    wall_time: float = 0.0
    population: Optional[Tensor] = None   # ga only

    def curve(self, key: str = "fitness_elite") -> List[float]:
        return [h[key] for h in self.history]

    @property
    def final(self) -> dict:
        return self.history[-1] if self.history else {}


def build(cfg: Config):
    """(system, trainable, task) from names -- one place, so scripts agree."""
    dt = torch_dtype(cfg.dtype)
    kw = {"environment": cfg.environment} if cfg.environment else {}
    system = make_system(cfg.system, dtype=dt, **kw)
    trainable = make_trainable(cfg.trainable, system)
    task = make_task(cfg.task, system, gating=cfg.gating)
    return system, trainable, task


def build_sensors(cfg: Config, system):
    """Sensor instances named by the config, or () for the full-state path."""
    return tuple(make_sensor(n, system) for n in (cfg.sensors or ()))


def _maybe_metric(cfg, g, theta, system, trainable, task, roll, P):
    """Refresh the preconditioner, or keep P = I for the isotropic arm."""
    es, rc = cfg.es, cfg.rollout
    if not es.whiten or (g % es.metric_every):
        return P, {}
    mgoals = task.sample(max(4, rc.n_eps), make_gen(gen_seed(cfg.seed, g, 3)))
    P = physics_metric(system, trainable, theta, mgoals, rc,
                       seed=gen_seed(cfg.seed, g, 4), n_states=es.metric_states,
                       ridge=es.ridge, null_mode=es.null_mode,
                       sign=es.metric_sign, roll=roll)
    return P, P.summary()


def _log(rec, verbose, tag):
    if verbose:
        print(f"  [{tag}] gen {rec['gen']:3d}  elite {rec['fitness_elite']:8.3f}  "
              f"crash {rec['crash_rate']:.2f}  legB {rec.get('legB_err', float('nan')):.3f}  "
              f"sigma {rec['sigma']:.4f}  hit {rec['success_frac']:.2f}"
              + (f"  cond {rec['metric_cond']:.3g}" if "metric_cond" in rec else ""))


# --------------------------------------------------------------------------- #
# distribution-based ES
# --------------------------------------------------------------------------- #
def train_es(cfg, system, trainable, task, callback=None, verbose=False,
             constraints=None, sensors=None, evaluator=None,
             theta0=None) -> TrainResult:
    es, rc = cfg.es, cfg.rollout
    roll = Rollout(system, trainable, task, rc,
                   sensors if sensors is not None else build_sensors(cfg, system))
    # `evaluator` shards the population across processes; the metric and the
    # traces still use the in-process rollout, which is what keeps the whole
    # thing bit-identical either way
    ev = evaluator or roll
    # A warm start carries a genome in from an earlier curriculum stage.  It only
    # means anything if the architecture is identical across stages -- the genome
    # is an opaque vector, so a mismatched dimension is a silent reinterpretation
    # rather than an error.
    theta = trainable.init() if theta0 is None else theta0.clone()
    if theta.shape[-1] != trainable.dim:
        raise ValueError(f"warm start has {theta.shape[-1]} slots but this "
                         f"trainable needs {trainable.dim}")
    sigma = es.sigma0
    P = identity_preconditioner(theta.shape[-1], theta.dtype, theta.device)
    history, t0 = [], time.time()

    for g in range(es.gens):
        P, minfo = _maybe_metric(cfg, g, theta, system, trainable, task, roll, P)
        goals = task.sample(rc.n_eps, make_gen(gen_seed(cfg.seed, g, 1)))
        TH = mirrored_offspring(theta, sigma, P.P, es.pop, make_gen(gen_seed(cfg.seed, g, 2)))

        # The parent rides along in the same batch: it gives a same-goals
        # reference for the 1/5th rule and a mean-genome learning curve, without
        # spending a second rollout on it.
        res = ev.run(torch.cat([TH, theta[None]], 0), goals,
                     seed=gen_seed(cfg.seed, g, 5))
        off = res.genome_slice(slice(0, es.pop))
        off_fit, parent_fit = res.fitness[: es.pop], res.fitness[es.pop]
        success_frac = float((off_fit < parent_fit).to(torch.float64).mean())

        # Selection acts on the AUGMENTED objective; lambda lives out here, never
        # in theta, so the metric never sees a coordinate with du/dlambda = 0.
        cinfo = {}
        sel_fit = off_fit
        if constraints is not None and len(constraints):
            sel_fit = constraints.augment(off_fit, off)
            cinfo = constraints.update(off)

        theta, elite, elite_fit = recombine(TH, sel_fit, es.elite_frac)
        sigma = update_sigma(sigma, success_frac > es.success_target,
                             es.grow, es.shrink, es.sigma_min, es.sigma_max)

        rec = {"gen": g, "fitness_elite": float(elite_fit.mean()),
               "fitness_parent": float(parent_fit), "sigma": sigma,
               "success_frac": success_frac, "n_elite": int(elite.numel()),
               **off.summary(), **minfo, **cinfo,
               **{f"th_{k}": v for k, v in trainable.describe(theta).items()}}
        history.append(rec)
        _log(rec, verbose, "es")
        if callback is not None:
            callback(rec, theta)

    return TrainResult(theta=theta, history=history, cfg=cfg, wall_time=time.time() - t0)


# --------------------------------------------------------------------------- #
# genetic algorithm over a persistent population
# --------------------------------------------------------------------------- #
def train_ga(cfg, system, trainable, task, callback=None, verbose=False,
             constraints=None, sensors=None, evaluator=None,
             theta0=None) -> TrainResult:
    es, rc = cfg.es, cfg.rollout
    roll = Rollout(system, trainable, task, rc,
                   sensors if sensors is not None else build_sensors(cfg, system))
    ev = evaluator or roll
    theta0 = trainable.init() if theta0 is None else theta0.clone()
    if theta0.shape[-1] != trainable.dim:
        raise ValueError(f"warm start has {theta0.shape[-1]} slots but this "
                         f"trainable needs {trainable.dim}")
    sigma = es.sigma0
    P = identity_preconditioner(theta0.shape[-1], theta0.dtype, theta0.device)

    # seed the population around the physical prior; keep one unmutated copy so
    # the GA can never start worse than the prior it was handed
    g0 = make_gen(gen_seed(cfg.seed, 0, 11))
    TH = whitened_mutation(theta0.expand(es.pop, -1).clone(), es.sigma0, P.P, g0)
    TH[0] = theta0

    best, best_fit, history, t0 = theta0.clone(), float("inf"), [], time.time()
    n_el = max(0, min(int(es.elitism), es.pop - 2))

    for g in range(es.gens):
        P, minfo = _maybe_metric(cfg, g, best, system, trainable, task, roll, P)
        goals = task.sample(rc.n_eps, make_gen(gen_seed(cfg.seed, g, 1)))
        res = ev.run(TH, goals, seed=gen_seed(cfg.seed, g, 5))
        fit = res.fitness
        cinfo = {}
        if constraints is not None and len(constraints):
            fit = constraints.augment(fit, res)
            cinfo = constraints.update(res)

        # After the first step, slots [0:n_el] are last generation's elites and
        # [n_el:] are its children -- both scored on today's goals, so their
        # comparison is a clean within-generation improvement signal.
        if g > 0 and n_el > 0:
            success_frac = float((fit[n_el:] < fit[:n_el].min()).to(torch.float64).mean())
        else:
            success_frac = es.success_target

        order = torch.argsort(fit)
        if float(fit[order[0]]) < best_fit:
            best_fit, best = float(fit[order[0]]), TH[order[0]].clone()

        mu = max(1, int(round(es.elite_frac * es.pop)))
        w = rank_weights(mu, dtype=TH.dtype, device=TH.device)
        centroid = (w[:, None] * TH[order[:mu]]).sum(0)

        rec = {"gen": g, "fitness_elite": float(fit[order[:mu]].mean()),
               "fitness_parent": float(fit[order[0]]), "sigma": sigma,
               "success_frac": success_frac, "n_elite": mu,
               "diversity": float(TH.std(dim=0).mean()),
               **res.summary(), **minfo, **cinfo,
               **{f"th_{k}": v for k, v in trainable.describe(centroid).items()}}
        history.append(rec)
        _log(rec, verbose, "ga")
        if callback is not None:
            callback(rec, TH[order[0]])

        TH, _ = ga_step_structured(
            TH, fit, sigma, P.P, P.P_inv, make_gen(gen_seed(cfg.seed, g, 2)),
            segments=trainable.segments(), elitism=n_el,
            tournament_k=es.tournament_k, crossover_rate=es.crossover_rate,
            blx_alpha=es.blx_alpha, segment_rate=es.segment_rate,
            mode=es.crossover_mode)
        sigma = update_sigma(sigma, success_frac > es.success_target,
                             es.grow, es.shrink, es.sigma_min, es.sigma_max)

    return TrainResult(theta=best, history=history, cfg=cfg,
                       wall_time=time.time() - t0, population=TH)


_STRATEGIES = {"es": train_es, "ga": train_ga}


def train(cfg: Config, system=None, trainable=None, task=None,
          callback: Optional[Callable] = None, verbose: bool = False,
          constraints=None, sensors=None, evaluator=None,
          theta0=None) -> TrainResult:
    """`constraints` is an optional `ConstraintSet`.  It is passed here rather
    than folded into the genome on purpose -- see `constraints.py` for why a
    multiplier inside theta silently rank-deficits the metric."""
    if system is None or trainable is None or task is None:
        system, trainable, task = build(cfg)
    if cfg.es.strategy not in _STRATEGIES:
        raise KeyError(f"unknown strategy {cfg.es.strategy!r}; "
                       f"expected one of {sorted(_STRATEGIES)}")
    return _STRATEGIES[cfg.es.strategy](cfg, system, trainable, task, callback,
                                        verbose, constraints, sensors, evaluator,
                                        theta0)
