"""Frozen configuration dataclasses.

This module holds NO logic: no validation, no IO, no derived quantities that
require computation.  Anything that needs to *do* something with a config lives
in `util.py` (loading) or in the module that consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RolloutCfg:
    """Episode integration and cost weights."""

    dt: float = 0.02
    ep_steps: int = 250
    n_eps: int = 2                # episodes per genome; shared across population
    lambda_e: float = 0.02        # effort weight
    lambda_s: float = 0.10        # plant-specific shaping weight
    lambda_los: float = 0.0       # weight on `sight_cost`: keeping the
                                  # target in the camera's view.  Free on a
                                  # quadrotor -- yaw does not disturb
                                  # position tracking -- but only if the
                                  # allocator is in yaw_mode='learned',
                                  # otherwise heading is a fixed constant
                                  # and this weight buys nothing.
    lambda_occ: float = 0.0       # weight on losing sight of the target to
                                  # an obstacle.  Distinct from lambda_los:
                                  # that one is fixed by TURNING, this one
                                  # only by MOVING, since no heading sees
                                  # through a pillar.
    lambda_r: float = 0.05        # learned-residual regularizer weight
    stop_on_arrival: bool = False  # freeze an episode once it reaches the FINAL
                                  # waypoint, and leave the loop once every
                                  # episode is finished or dead.  The vehicle
                                  # arrives after ~6% of a 2400-step episode, so
                                  # the rest is pure waste -- both in training
                                  # time and in a replay that runs on long after
                                  # the task is over.
    charge_slots: int = 96        # world-frame sensor returns remembered by a
                                  # harmonic field term; 8 updates of a 12-beam
                                  # fan.  Ignored unless a term asks for them.
    pos_eps: float = 1e-2         # smoothing floor in sqrt(|e|^2 + eps).
                                  # MUST be << the task's error scale: at 1e-2
                                  # a plant with 0.1 m errors has a nearly flat
                                  # objective (0.100 -> 0.141 across the task).
    dead_cost: float = 5.0        # per-second cost accrued by a crashed vehicle
                                  # under dead_mode="constant"
    dead_mode: str = "constant"   # "constant" | "frozen" | "forfeit".
                                  # "constant" charges a flat dead_cost/s, which
                                  # is a free parameter that has to be tuned
                                  # against the position scale and gets it wrong
                                  # in both directions: at 5.0 crashing cost more
                                  # than the whole task was worth and the
                                  # population froze; at 0.625 crashing was
                                  # cheaper than hovering and it dove.
                                  # "frozen" removes the parameter entirely --
                                  # a crashed vehicle is already frozen in place
                                  # by `tree_where`, so it simply keeps accruing
                                  # the POSITION term from where it stopped.
                                  # Crashing at the start then costs exactly what
                                  # never moving costs, with nothing to tune.
                                  # But it leaves a loophole at the other end:
                                  # crashing NEAR the goal is nearly free, so a
                                  # sprint-and-crash outscores a careful arrival
                                  # and selection dismantles obstacle avoidance.
                                  # Measured: safe policy (reach 0.904) fitness
                                  # 3.9703 vs crashing policy (0.777) at 3.7952.
                                  #
                                  # "forfeit" charges a dead vehicle the distance
                                  # it started with rather than the one it died
                                  # at, so a crash gives back all its progress.
                                  # Crashing then costs the same as never having
                                  # moved -- no incentive to sit still, which is
                                  # what a raised dead_cost creates -- while a
                                  # late crash stops being a bargain.
    goal_bonus: float = 0.0       # cost SUBTRACTED once per waypoint reached.
                                  # Arrival gating only.  Without it the sole
                                  # reward for arriving is a shorter tail of
                                  # position cost, which is worth under 1.0 in
                                  # an objective where standing still costs 10 --
                                  # i.e. the goal is barely an incentive at all.
                                  # 0.0 keeps the pure integral-cost objective.


@dataclass(frozen=True)
class ESCfg:
    """Evolution-strategy hyperparameters."""

    pop: int = 16                 # must be even (mirrored sampling)
    gens: int = 6
    sigma0: float = 0.08
    sigma_min: float = 1e-3
    sigma_max: float = 0.5
    grow: float = 1.06            # step-size multiplier on improvement
    shrink: float = 0.97          # ... and on stagnation.  The asymmetry makes a
                                  # growth step ~1.9x a shrink step in log space,
                                  # which is what actually sets sigma's fixed
                                  # point -- see success_target below.
    elite_frac: float = 0.25

    # --- search strategy --------------------------------------------------
    #: "es" = distribution-based (one mean, mirrored sampling, rank recombination)
    #: "ga" = persistent population with tournament selection + crossover
    strategy: str = "es"
    success_target: float = 0.20   # 1/5th rule: grow sigma above this hit rate.
                                   # NOTE this is the trigger, not the fixed
                                   # point.  sigma equilibrates where the drift
                                   # p*ln(grow) + (1-p)*ln(shrink) vanishes, i.e.
                                   # where the FRACTION OF GENERATIONS above the
                                   # trigger is ln(1/shrink)/ln(grow/shrink) =
                                   # 0.343 for the defaults -- not 0.20.  Raising
                                   # success_target alone does not move that
                                   # balance point; grow/shrink set it.
    elitism: int = 2               # ga: individuals copied through unchanged
    tournament_k: int = 3          # ga: tournament size
    crossover_rate: float = 0.7    # ga: probability a child is a recombination
    blx_alpha: float = 0.5         # ga: BLX-alpha interpolation slack
    crossover_mode: str = "mixed"  # ga: "blx" | "segment" | "mixed"
    segment_rate: float = 0.5      # ga: per-segment probability of taking parent B

    whiten: bool = True           # False => P = I, the isotropic baseline
    metric_every: int = 5         # generations between metric refreshes
    metric_states: int = 96       # states subsampled per refresh
    ridge: float = 1e-3           # added to normalized eigenvalues
    null_mode: str = "cap"        # "ridge" | "cap" -- see metric._precondition.
                                  # "cap" is the default because a small ridge
                                  # amplifies G's null space and measurably
                                  # HURTS; see README "What the ablation found".
    metric_sign: float = -1.0     # -1 = G^-1/2 (specified); +1 inverts it


@dataclass(frozen=True)
class Config:
    """Top-level experiment description."""

    system: str = "quadrotor"
    trainable: str = "energy_shaping"
    task: str = "waypoint_pair"
    sensors: tuple = ()           # sensor registry names; () = full-state path
    environment: str = ""         # scene preset for plants that take one
    task_kw: tuple = ()           # extra task kwargs as (name, value) pairs.
                                  # A bigger arena is not one number: the scene's
                                  # extent, the task's reach and the sensor's
                                  # range all have to move together, and without
                                  # this the task stays the size it always was.
    system_kw: tuple = ()         # extra plant kwargs as (name, value) pairs;
                                  # a tuple, not a dict, so Config stays frozen
                                  # and hashable
    gating: str = "arrival"       # "time" | "arrival"; see tasks.Task.gating
    seed: int = 0
    dtype: str = "float64"
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    es: ESCfg = field(default_factory=ESCfg)
