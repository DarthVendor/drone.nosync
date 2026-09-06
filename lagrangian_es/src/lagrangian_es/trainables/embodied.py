"""Plant-specific agents: robots that carry their own Lagrangian contributions.

`EnergyShaping` is the composition machinery -- it knows how to sum terms, split
a genome, and certify the result, but nothing about any particular robot.  What
terms a given robot *should* have is a property of that robot, and until now it
had to be assembled at the call site, which meant a quadruped-appropriate
controller existed only as an argument someone remembered to pass.

These subclasses declare it once instead:

    Trainable                      the evolvable-object interface
      +- EnergyShaping             composition of LagrangianTerms
           +- EmbodiedAgent        an agent that knows its robot
                +- QuadrotorAgent  bowls + damping + ground-clearance barrier
                +- QuadrupedAgent  ... + kinetic shaping + a base-pose envelope
                +- ArmAgent        ... + joint-limit barriers

A subclass is allowed to read its own plant's attributes -- `z_floor`,
`stand_height`, joint ranges -- because that is exactly what the specialization
buys.  `supports()` records which plants an agent is meant for, so the
conformance sweep skips pairs that were never intended rather than failing them.

Note what the envelope barriers do to the equilibrium claim.  A barrier is
compactly supported, so it withdraws the *unconditional* promise
(`equilibrium_exact` is False) while still keeping it for any goal that clears
the barrier's margin (`equilibrium_exact_for(theta, goal)`).  That is the honest
answer: an envelope really does move the equilibrium for a goal pressed against
it, and the certificate says so rather than quietly being wrong.
"""
from __future__ import annotations

from typing import List

from ..systems.base import LagrangianSystem
from .energy_shaping import EnergyShaping
from .terms import (
    DissipationTerm, GoalBowl, JointLimitBarrier, KineticShaping, LagrangianTerm,
)

BIG = 50.0          # stands in for an unbounded side of an envelope


class EmbodiedAgent(EnergyShaping):
    """An `EnergyShaping` controller whose term list is declared by the subclass."""

    #: systems this agent is designed for, by registry name
    for_systems: tuple = ()

    def __init__(self, system: LagrangianSystem, n_bowls: int = 3, terms=None, **kw):
        if terms is None:
            terms = self.build_terms(system, n_bowls=n_bowls, **kw)
        super().__init__(system, terms=terms)

    @classmethod
    def supports(cls, system) -> bool:
        return type(system).__name__ in cls.for_systems

    @classmethod
    def build_terms(cls, system, n_bowls: int = 3, **kw) -> List[LagrangianTerm]:
        raise NotImplementedError

    # --- shared helpers -----------------------------------------------------
    @staticmethod
    def _core(system, n_bowls: int) -> List[LagrangianTerm]:
        """The bowls-plus-damping core, sized by the plant's own suggested gains."""
        d = system.task_dim
        a0 = (float(system.potential_scale()) / max(n_bowls, 1)) ** 0.5
        d0 = float(system.damping_scale()) ** 0.5
        return [GoalBowl(d, a0=a0) for _ in range(n_bowls)] + [DissipationTerm(d, d0=d0)]


class QuadrotorAgent(EmbodiedAgent):
    """SE(3) quadrotor: the standard core plus a ground-clearance barrier.

    The barrier is the interesting addition.  Crashing into the floor is the
    dominant failure of the untrained prior, and a compactly supported repulsion
    just above `z_floor` makes clearance a property of the search space rather
    than something evolution has to discover by killing individuals.  It is
    exactly zero more than `margin` above the floor, so it costs nothing in
    normal flight.
    """

    for_systems = ("QuadrotorSE3", "QuadrotorPayload", "QuadrotorNav")

    @classmethod
    def build_terms(cls, system, n_bowls: int = 3, ground_margin: float = 0.35,
                    ground_weight: float = 2.0, **kw):
        terms = cls._core(system, n_bowls)
        floor = float(getattr(system, "z_floor", 0.0))
        terms.append(JointLimitBarrier(
            system.task_dim,
            lo=[-BIG, -BIG, floor], hi=[BIG, BIG, BIG],
            w0=ground_weight, margin0=ground_margin))
        return terms


class QuadrupedAgent(EmbodiedAgent):
    """Planar quadruped: core, kinetic shaping, and a base-pose envelope.

    `KineticShaping` earns its place here and not on the drone: the base-space
    inertia of a legged robot is genuinely anisotropic and configuration
    dependent -- vertical and pitch motion see very different effective masses --
    so shaping M_d is a real degree of freedom rather than a reparameterization.

    The envelope barrier keeps trunk height and pitch inside the region where the
    legs can actually deliver a wrench, which is what a fall looks like before it
    becomes a fall.
    """

    for_systems = ("PlanarQuadruped",)

    @classmethod
    def build_terms(cls, system, n_bowls: int = 3, envelope_weight: float = 6.0,
                    envelope_margin: float = 0.02, kinetic: bool = True, **kw):
        # The margin must be narrower than the gap between the nominal stance and
        # the envelope, or the barrier is active while the robot merely stands and
        # spends the whole episode fighting the controller.  Nominal height is
        # ~0.351 m against an upper bound of 0.98*reach = 0.392 m, so 0.05 would
        # overlap and 0.02 clears it.
        terms = cls._core(system, n_bowls)
        if kinetic:
            terms.append(KineticShaping(system.task_dim))
        reach = float(getattr(system, "l_thigh", 0.2) + getattr(system, "l_shank", 0.2))
        h = float(getattr(system, "stand_height", 0.35))
        terms.append(JointLimitBarrier(
            system.task_dim,
            lo=[-BIG, 0.55 * h, -0.45],          # don't collapse, don't tip
            hi=[BIG, 0.98 * reach, 0.45],        # don't reach the straight-leg singularity
            w0=envelope_weight, margin0=envelope_margin))
        return terms


class NavAgent(EmbodiedAgent):
    """Quadrotor navigating an obstacle field, avoiding from RANGE MEASUREMENTS.

    Exists as a registered class rather than a hand-assembled term list so the
    whole rig is expressible from a `Config` -- which is what lets worker
    processes rebuild it, since a closure over terms is not picklable.

    `RangeBarrier` is handed beams and never told where anything is, which is the
    difference that makes transfer to an unseen layout mean something: an
    `ObstacleBarrier` is given geometry at construction and cannot transfer.
    """

    for_systems = ("QuadrotorNav",)

    @classmethod
    def build_terms(cls, system, n_bowls: int = 3, n_beams: int = 24,
                    sensor_name: str = "range", barrier_weight: float = 3.0,
                    damper: bool = True, vortex: bool = True, **kw):
        """Core, plus a barrier on POSITION and a damper on CLOSING SPEED.

        Both are needed, and for a reason that is structural rather than a matter
        of tuning.  Stopping takes `v^2 / 2a` of room, so a potential -- whose
        force depends only on position -- delivers a fixed `a` and can arrest an
        approach only from `v <= sqrt(2 a d)`.  Measured here, the barrier gives
        about 2 m/s^2 over 1.35 m, good for 2.3 m/s, while every collision
        happened at a mean of 3.09 m/s.  Raising its gain fixes the speed limit
        and wrecks everything else (reach 0.21 at high gain).  The missing
        dependence is on velocity, and in a Lagrangian that is a Rayleigh term.

        Paired over three seeds on the pillar field, adding the damper takes
        reach 0.840 -> 0.867 and crashes 0.137 -> 0.109; against the original
        barrier-only baseline of 0.779 / 0.221 that is +0.088 reach at half the
        crash rate.
        """
        from .sensor_terms import RangeBarrier, RangeDamper, RangeVortex
        terms = cls._core(system, n_bowls) + [
            RangeBarrier(system.task_dim, sensor_name, n_beams, w0=barrier_weight)]
        if damper:
            terms.append(RangeDamper(system.task_dim, sensor_name, n_beams))
        if vortex:
            # workless: F perpendicular to v, so the certificate is untouched.
            # Redirects the stragglers that slide around a pillar at the barrier's
            # standoff netting only ~0.08 m/s of progress.
            terms.append(RangeVortex(system.task_dim, sensor_name, n_beams,
                                     k0=6.0))
        return terms


class ArmAgent(EmbodiedAgent):
    """Articulated arm in joint space: core plus joint-limit barriers."""

    for_systems = ("TwoLinkArm", "MaximalChain")

    @classmethod
    def build_terms(cls, system, n_bowls: int = 3, joint_limit: float = 2.6,
                    limit_weight: float = 3.0, limit_margin: float = 0.30, **kw):
        terms = cls._core(system, n_bowls)
        terms.append(JointLimitBarrier(
            system.task_dim, lo=-joint_limit, hi=joint_limit,
            w0=limit_weight, margin0=limit_margin))
        return terms


class PlanarQuadrotorAgent(QuadrotorAgent):
    """The cheap 2-D quadrotor: same structure, its own floor."""

    for_systems = ("PlanarQuadrotor",)

    @classmethod
    def build_terms(cls, system, n_bowls: int = 3, ground_margin: float = 0.35,
                    ground_weight: float = 2.0, **kw):
        terms = cls._core(system, n_bowls)
        floor = float(getattr(system, "z_floor", 0.0))
        terms.append(JointLimitBarrier(
            system.task_dim, lo=[-BIG, floor], hi=[BIG, BIG],
            w0=ground_weight, margin0=ground_margin))
        return terms


AGENTS = {
    "quadrotor_agent": QuadrotorAgent,
    "nav_agent": NavAgent,
    "planar_quadrotor_agent": PlanarQuadrotorAgent,
    "quadruped_agent": QuadrupedAgent,
    "arm_agent": ArmAgent,
}
