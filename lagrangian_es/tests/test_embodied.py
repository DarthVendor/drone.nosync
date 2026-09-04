"""Plant-specific agents: each robot declares its own Lagrangian contributions."""
import pytest
import torch
from torch.func import jacrev, vmap

from lagrangian_es.systems import SYSTEMS, make_system
from lagrangian_es.trainables import (
    TRAINABLES, ArmAgent, EmbodiedAgent, EnergyShaping, QuadrotorAgent,
    QuadrupedAgent, make_trainable,
)
from lagrangian_es.util import make_gen

DT = torch.float64
AGENTS = ["quadrotor_agent", "planar_quadrotor_agent", "quadruped_agent", "arm_agent"]


def _pairs():
    out = []
    for a in AGENTS:
        for s in sorted(SYSTEMS):
            if TRAINABLES[a].supports(SYSTEMS[s]()):
                out.append((s, a))
    return out


def test_every_agent_has_at_least_one_supported_plant():
    covered = {a for _, a in _pairs()}
    assert covered == set(AGENTS), f"unsupported agents: {set(AGENTS) - covered}"


def test_agents_are_energy_shaping_subclasses():
    """The hierarchy is a specialization, not a parallel implementation: agents
    inherit the composition machinery and only choose the terms."""
    for name in AGENTS:
        cls = TRAINABLES[name]
        assert issubclass(cls, EmbodiedAgent)
        assert issubclass(cls, EnergyShaping)


@pytest.mark.parametrize("sysname,agent", _pairs())
def test_agent_builds_and_traces_on_its_own_plant(sysname, agent):
    system = make_system(sysname)
    tr = make_trainable(agent, system)
    theta = tr.init()
    assert theta.shape == (tr.dim,)
    assert torch.isfinite(theta).all()
    assert tr.dim == tr.policy_dim + system.allocator_dim + system.residual_dim

    s = system.reset(6, make_gen(0))
    goal = system.task_position(s)
    u = vmap(tr.forward, in_dims=(None, 0, 0))(theta, s, goal)
    assert u.shape == (6, system.n_force) and torch.isfinite(u).all()
    J = vmap(jacrev(tr.forward), in_dims=(None, 0, 0))(theta, s, goal)
    assert J.shape == (6, system.n_force, tr.dim) and torch.isfinite(J).all()


@pytest.mark.parametrize("sysname,agent", _pairs())
def test_agent_declares_a_superset_of_the_generic_core(sysname, agent):
    """Every agent keeps the bowls-plus-damping core and adds to it."""
    system = make_system(sysname)
    tr = make_trainable(agent, system)
    kinds = [t.kind for t in tr.terms]
    assert kinds[:4] == ["goal_bowl"] * 3 + ["dissipation"]
    assert len(kinds) > 4, "an agent that adds nothing needs no subclass"
    generic = make_trainable("energy_shaping", system)
    assert tr.policy_dim > generic.policy_dim


def test_supports_narrows_the_conformance_sweep():
    """A quadruped agent is not a drone controller, and says so rather than
    failing a test it was never meant to pass."""
    assert QuadrotorAgent.supports(make_system("quadrotor"))
    assert not QuadrotorAgent.supports(make_system("quadruped"))
    assert QuadrupedAgent.supports(make_system("quadruped"))
    assert not QuadrupedAgent.supports(make_system("quadrotor"))
    assert ArmAgent.supports(make_system("two_link_arm"))
    assert ArmAgent.supports(make_system("maximal_chain"))
    # generic trainables stay universal
    assert TRAINABLES["energy_shaping"].supports(make_system("quadruped"))


@pytest.mark.parametrize("sysname,agent", _pairs())
def test_envelope_barrier_withdraws_only_the_unconditional_claim(sysname, agent):
    """A barrier really does move the equilibrium for a goal pressed against it,
    so the unconditional promise must lapse while the conditional one survives
    for goals that clear the margin."""
    system = make_system(sysname)
    tr = make_trainable(agent, system)
    assert tr.equilibrium_exact is False
    theta = tr.init()
    # Ask the barrier where its own interior is, rather than tabulating a safe
    # goal per plant.  Note the resting pose is NOT automatically clear: the
    # drone sits 0.45 m above its floor with a 0.35 m barrier margin, so takeoff
    # legitimately happens inside the barrier -- that is what it is for.
    bar = tr.terms[-1]
    sl = tr.term_slices(theta)[-1]
    margin = float(bar._margin(sl[..., 1]))
    lo = torch.tensor(bar.lo_t, dtype=DT)
    hi = torch.tensor(bar.hi_t, dtype=DT)
    rest = system.task_position(system.reset(1, make_gen(0)))[0]
    safe = torch.max(torch.min(rest, hi - 2 * margin), lo + 2 * margin)
    assert tr.equilibrium_exact_for(theta, safe) is True, (
        f"{sysname}: goal {safe.tolist()} should clear "
        f"lo={bar.lo_t} hi={bar.hi_t} margin={margin:.3f}")


def test_quadrotor_agent_ground_barrier_is_inert_in_normal_flight():
    """It must cost nothing where it is not needed: compact support means exactly
    zero more than `margin` above the floor, not merely small."""
    system = make_system("quadrotor")
    tr = make_trainable("quadrotor_agent", system)
    generic = make_trainable("energy_shaping", system)
    high = torch.tensor([[0.3, -0.2, 3.0]], dtype=DT)
    s = system.nominal_state(high, torch.zeros_like(high))
    goal = torch.tensor([[0.0, 0.0, 2.5]], dtype=DT)
    ua = vmap(tr.forward, in_dims=(None, 0, 0))(tr.init(), s, goal)
    ug = vmap(generic.forward, in_dims=(None, 0, 0))(generic.init(), s, goal)
    assert torch.allclose(ua, ug, atol=1e-10), "barrier is active far from the ground"


def test_quadrotor_agent_ground_barrier_pushes_up_near_the_floor():
    system = make_system("quadrotor")
    tr = make_trainable("quadrotor_agent", system)
    low = torch.tensor([[0.0, 0.0, system.z_floor + 0.05]], dtype=DT)
    barrier = tr.terms[-1]
    e = torch.zeros(1, 3, dtype=DT)
    g = barrier.grad_potential(tr.term_slices(tr.init())[-1], e, e, low)
    assert g[0, 2] < -1e-6, "barrier must push the vehicle away from the floor"


def test_quadruped_agent_includes_kinetic_shaping():
    """Kinetic shaping belongs on the legged robot and not the drone: base-space
    inertia is genuinely anisotropic there, so M_d is a real degree of freedom."""
    kinds = [t.kind for t in make_trainable(
        "quadruped_agent", make_system("quadruped")).terms]
    assert "kinetic_shaping" in kinds
    drone = [t.kind for t in make_trainable(
        "quadrotor_agent", make_system("quadrotor")).terms]
    assert "kinetic_shaping" not in drone


def test_agents_size_their_priors_to_the_plant():
    """A stiffness sized for a 0.5 kg drone is meaningless on an 11 kg robot."""
    drone = make_trainable("quadrotor_agent", make_system("quadrotor"))
    dog = make_trainable("quadruped_agent", make_system("quadruped"))
    kd = drone.describe(drone.init())
    kq = dog.describe(dog.init())
    assert abs(kd["K_max"] - 3.0) < 1e-9, "the drone's calibration must not drift"
    assert abs(kd["Kd_max"] - 1.44) < 1e-9
    assert kq["K_max"] > 20 * kd["K_max"]


@pytest.mark.parametrize("sysname,agent", _pairs())
def test_agent_segments_cover_the_whole_genome(sysname, agent):
    system = make_system(sysname)
    tr = make_trainable(agent, system)
    segs = tr.segments()
    covered = sum(s.stop - s.start for s in segs)
    assert covered == tr.dim
    assert segs[0].start == 0 and segs[-1].stop == tr.dim
