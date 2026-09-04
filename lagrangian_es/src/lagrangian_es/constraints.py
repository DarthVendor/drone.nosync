"""Episode-level constraints and their multipliers.

Constraints act at two different levels, and the design needs both.

**Path level.** Folded into V_d as a barrier term (`trainables/terms.py`).  These
change the commanded force, so they live in theta and are whitened by G along with
everything else.  This is the good case: the constraint is part of the controller.

**Episode level.** Attached to fitness rather than to the wrench:

    L(theta, lambda) = J(theta) + sum_i lambda_i * (c_i(theta) - budget_i)

These change only selection pressure.

The non-obvious consequence, and the reason this module exists separately from
the genome: **lambda must not go inside theta.**  A multiplier affects fitness,
not the commanded force, so du/dlambda = 0 identically -- it contributes zero rows
and columns to

    G(theta) = E_traj[(du/dtheta)^T M(q)^-1 (du/dtheta)] .

Put lambda in the genome and the metric is exactly singular there, with the ridge
silently doing all the work in precisely the coordinates that are supposed to be
searched.  Worse, it is a *silent* singularity: the ridge keeps `eigh` well posed,
so nothing errors and nothing looks wrong -- the multiplier just stops being
searched meaningfully.  `test_constraints.py` demonstrates the zero block rather
than asserting it in a comment.

So the multipliers are updated by dual ascent in the OUTER loop instead.  Expect
the usual primal-dual oscillation: the multiplier chases a violation that its own
growth then removes, overshoots, and rings.  `PIDMultiplier` (Stooke et al. 2020,
"Responsive Safety in Reinforcement Learning by PID Lagrangian Methods") is the
standard fix -- the classic dual ascent is exactly its integral term, and the
proportional term is what damps the ringing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence

from torch import Tensor

from .rollout import RolloutResult


# --------------------------------------------------------------------------- #
# constraints
# --------------------------------------------------------------------------- #
class EpisodeConstraint(ABC):
    """A budget on a genome's whole behaviour, evaluated after the episode."""

    name: str = "constraint"

    def __init__(self, budget: float):
        self.budget = float(budget)

    @abstractmethod
    def value(self, res: RolloutResult) -> Tensor:
        """[P] per-genome constraint value."""

    def violation(self, res: RolloutResult) -> Tensor:
        """[P], positive when the budget is exceeded."""
        return self.value(res) - self.budget

    def __repr__(self):
        return f"{type(self).__name__}(budget={self.budget})"


class CrashBudget(EpisodeConstraint):
    """Fraction of a genome's episodes that ended outside the flight envelope."""

    name = "crash"

    def value(self, res):
        return res.per_genome((~res.alive).to(res.cost.dtype))


class EffortBudget(EpisodeConstraint):
    """Mean counterforce in the M^-1 metric, gravity feedforward removed."""

    name = "effort"

    def value(self, res):
        return res.per_genome(res.effort)


class SaturationBudget(EpisodeConstraint):
    """Mean fraction of actuator channels sitting at a bound.

    Worth constraining for its own sake: jacrev through a saturated clamp returns
    zero, so a population that drifts into saturation quietly rank-deficits G.
    """

    name = "saturation"

    def value(self, res):
        return res.per_genome(res.saturation)


class ShapingBudget(EpisodeConstraint):
    """Mean plant-specific regularizer -- tilt on a quadrotor."""

    name = "shaping"

    def value(self, res):
        return res.per_genome(res.shaping)


CONSTRAINTS = {c.name: c for c in
               (CrashBudget, EffortBudget, SaturationBudget, ShapingBudget)}


def make_constraint(name: str, budget: float) -> EpisodeConstraint:
    if name not in CONSTRAINTS:
        raise KeyError(f"unknown constraint {name!r}; registered: {sorted(CONSTRAINTS)}")
    return CONSTRAINTS[name](budget)


# --------------------------------------------------------------------------- #
# multipliers  (outer loop -- never part of theta)
# --------------------------------------------------------------------------- #
class Multiplier(ABC):
    def __init__(self, lam0: float = 0.0, lam_max: float = 1e3):
        self.lam = float(lam0)
        self.lam_max = float(lam_max)

    @abstractmethod
    def update(self, violation: float) -> float:
        """Advance one dual step and return the new multiplier."""

    def _clip(self, v: float) -> float:
        return float(min(max(v, 0.0), self.lam_max))


class DualAscent(Multiplier):
    """lambda <- clamp_min(lambda + eta * violation, 0).

    The textbook update, and the one that oscillates: the multiplier only ever
    responds to accumulated violation, so it keeps growing through the period in
    which the population has already complied, then has to unwind.
    """

    def __init__(self, eta: float = 0.05, lam0: float = 0.0, lam_max: float = 1e3):
        super().__init__(lam0, lam_max)
        self.eta = float(eta)

    def update(self, violation: float) -> float:
        self.lam = self._clip(self.lam + self.eta * violation)
        return self.lam


class PIDMultiplier(Multiplier):
    """Stooke et al. (2020) PID Lagrangian.

    lambda = clamp_min(kp*v + I + kd*(v - v_prev), 0),  I <- clamp_min(I + ki*v, 0)

    The integral term alone IS dual ascent, so setting kp = kd = 0 recovers it
    exactly -- which is what makes the two comparable rather than two different
    algorithms.  The proportional term responds to the violation now instead of to
    its history, which is what damps the ringing; the derivative term anticipates.
    """

    def __init__(self, kp: float = 0.10, ki: float = 0.02, kd: float = 0.05,
                 lam0: float = 0.0, lam_max: float = 1e3):
        super().__init__(lam0, lam_max)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self._I = float(lam0)
        self._prev: Optional[float] = None

    def update(self, violation: float) -> float:
        self._I = max(0.0, self._I + self.ki * violation)
        d = 0.0 if self._prev is None else (violation - self._prev)
        self._prev = violation
        self.lam = self._clip(self.kp * violation + self._I + self.kd * d)
        return self.lam


MULTIPLIERS: Dict[str, Callable[..., Multiplier]] = {
    "dual_ascent": DualAscent, "pid": PIDMultiplier,
}


# --------------------------------------------------------------------------- #
class ConstraintSet:
    """Holds the constraints, their multipliers, and the augmented objective.

    Deliberately separate from the genome.  `augment` never touches theta and
    `update` never sees it: the primal search moves theta, the dual step moves
    lambda, and nothing in the metric ever learns that lambda exists.
    """

    def __init__(self, constraints: Sequence[EpisodeConstraint],
                 multiplier: str = "pid", **mkw):
        self.constraints: List[EpisodeConstraint] = list(constraints)
        if multiplier not in MULTIPLIERS:
            raise KeyError(f"unknown multiplier {multiplier!r}; "
                           f"registered: {sorted(MULTIPLIERS)}")
        self.multipliers = [MULTIPLIERS[multiplier](**mkw) for _ in self.constraints]
        self.kind = multiplier

    def __len__(self):
        return len(self.constraints)

    def augment(self, fitness: Tensor, res: RolloutResult) -> Tensor:
        """J(theta) + sum_i lambda_i (c_i(theta) - budget_i), per genome."""
        out = fitness
        for c, m in zip(self.constraints, self.multipliers):
            out = out + m.lam * c.violation(res)
        return out

    def update(self, res: RolloutResult) -> Dict[str, float]:
        """One dual step per constraint, driven by the POPULATION MEAN violation.

        The population mean rather than the elite's: the multiplier prices the
        constraint for the whole distribution being sampled, and pricing off the
        elite alone lets the population drift into violation between updates.
        """
        info = {}
        for c, m in zip(self.constraints, self.multipliers):
            v = float(c.violation(res).mean())
            m.update(v)
            info[f"viol_{c.name}"] = v
            info[f"lam_{c.name}"] = m.lam
        return info

    def state(self) -> Dict[str, float]:
        return {f"lam_{c.name}": m.lam for c, m in zip(self.constraints, self.multipliers)}

    def __repr__(self):
        inner = ", ".join(f"{c.name}<={c.budget}" for c in self.constraints)
        return f"ConstraintSet({inner}; {self.kind})"
