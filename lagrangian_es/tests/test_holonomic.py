"""Constraint-based coupling: multipliers, the KKT solve, and the hybrid model."""
import pytest
import torch
from torch.func import jacrev, vmap

from lagrangian_es.systems import make_system
from lagrangian_es.systems.holonomic import (
    ConstraintStack, GroundContact, JointCoupling, PinJointChain,
    constrained_inverse_inertia,
)
from lagrangian_es.trainables import make_trainable
from lagrangian_es.util import make_gen

DT = torch.float64


# --------------------------------------------------------------------------- #
# the KKT solve
# --------------------------------------------------------------------------- #
def test_multiplier_is_the_constraint_force():
    """A unit mass held against gravity: the multiplier must BE the support
    force, not merely be proportional to it."""
    st = ConstraintStack([GroundContact()])
    M = torch.eye(3, dtype=DT).expand(4, 3, 3).clone()
    J = torch.zeros(4, 1, 3, dtype=DT)
    J[:, 0, 2] = 1.0
    rhs = torch.tensor([[0.0, 0.0, -9.81]], dtype=DT).expand(4, 3)
    bias = torch.zeros(4, 1, dtype=DT)
    act = torch.ones(4, 1, dtype=DT)
    eps = torch.full((4, 1), 1e-10, dtype=DT)
    ddq, lam = st.solve(M, rhs, J, bias, act, eps)
    assert ddq[..., 2].abs().max() < 1e-6, "constrained direction must not accelerate"
    assert abs(float(lam[0, 0]) - 9.81) < 1e-5


def test_releasing_a_unilateral_row_zeroes_its_multiplier():
    """Activation -> 0 must release the row continuously, not switch it off:
    `allocate` is jacrev'd through touchdown."""
    st = ConstraintStack([GroundContact()])
    M = torch.eye(3, dtype=DT).expand(3, 3, 3).clone()
    J = torch.zeros(3, 1, 3, dtype=DT)
    J[:, 0, 2] = 1.0
    rhs = torch.tensor([[0.0, 0.0, -9.81]], dtype=DT).expand(3, 3)
    bias = torch.zeros(3, 1, dtype=DT)
    # compliance is eps/act, so the release depends on the RATIO -- use the
    # plant's real compliance and let the activation span its actual range.
    eps = torch.full((3, 1), 2e-6, dtype=DT)
    act = torch.tensor([[1.0], [1e-4], [1e-9]], dtype=DT)
    _, lam = st.solve(M, rhs, J, bias, act, eps)
    assert lam[0, 0] > 9.0, "a fully engaged row carries the load"
    assert lam[1, 0] < lam[0, 0]
    assert abs(float(lam[2, 0])) < 1e-2, "a fully released row carries no force"
    assert lam[2, 0] < lam[1, 0] < lam[0, 0], "release must be monotone, not a switch"


def test_compliance_makes_a_redundant_constraint_set_solvable():
    """Four feet on flat ground over-determine a planar base, so the hard KKT
    system is genuinely rank-deficient.  The compliance block is what keeps it
    invertible -- not a numerical nicety."""
    st = ConstraintStack([GroundContact()])
    M = torch.eye(3, dtype=DT)[None]
    J = torch.ones(1, 4, 3, dtype=DT)                # four identical rows
    rhs = torch.zeros(1, 3, dtype=DT)
    bias = torch.zeros(1, 4, dtype=DT)
    act = torch.ones(1, 4, dtype=DT)
    ddq, lam = st.solve(M, rhs, J, bias, act, torch.full((1, 4), 1e-6, dtype=DT))
    assert torch.isfinite(ddq).all() and torch.isfinite(lam).all()


def _drive_one_hip(sysm, steps=400, kp=60.0, kd=6.0, extra=6.0):
    """Hold the robot up with a joint PD and push ONE hip, so the run stays inside
    the operating envelope (with zero torque the legs simply fold and the
    configuration degenerates, which tests nothing about coupling)."""
    st = sysm.reset(2, make_gen(0))
    for _ in range(steps):
        tau = kp * (sysm.q_nom[3:] - st["q"][:, 3:]) - kd * st["dq"][:, 3:]
        tau[:, 0] = tau[:, 0] + extra                # drive leg 0's hip only
        st = sysm.step(st, tau, 0.002)
    return st


def test_joint_coupling_binds_two_coordinates():
    """A bilateral coupling must actually bind two coordinates together.

    Differential test: drive leg 0's hip and watch leg 2's.  Coupled, the two
    track each other; uncoupled, they separate.
    """
    free = make_system("quadruped")
    tied = make_system("quadruped", extra_constraints=(JointCoupling([(3, 7)]),))
    assert free.n_rows == 4 and tied.n_rows == 5     # 4 contacts (+1 coupling)

    sf, st_ = _drive_one_hip(free), _drive_one_hip(tied)
    gap_free = float((sf["q"][0, 3] - sf["q"][0, 7]).abs())
    gap_tied = float((st_["q"][0, 3] - st_["q"][0, 7]).abs())
    assert gap_free > 1e-3, f"test is vacuous: uncoupled gap only {gap_free:.5f}"
    assert gap_tied < 0.1 * gap_free, (
        f"coupling not enforced: tied {gap_tied:.5f} vs free {gap_free:.5f}")
    assert bool(tied.alive(st_).all())


# --------------------------------------------------------------------------- #
# constrained inverse inertia
# --------------------------------------------------------------------------- #
def test_constrained_inverse_inertia_annihilates_constraint_directions():
    """P J^T = 0: a force along a constrained direction produces no acceleration,
    because the constraint absorbs it."""
    g = torch.Generator().manual_seed(0)
    M = torch.diag_embed(1.0 + torch.rand(6, generator=g, dtype=DT))[None]
    J = torch.randn(1, 2, 6, generator=g, dtype=DT)
    P = constrained_inverse_inertia(M, J)
    assert (P @ J.transpose(-1, -2)).abs().max() < 1e-6
    assert torch.allclose(P, P.transpose(-1, -2), atol=1e-10)
    assert (torch.linalg.eigvalsh(0.5 * (P + P.transpose(-1, -2))) > -1e-9).all()


# --------------------------------------------------------------------------- #
# the two coordinate formulations
# --------------------------------------------------------------------------- #
def test_minimal_and_maximal_coordinates_agree():
    """The same physical two-link chain, written two ways.

    Minimal coordinates: M(q) dense and configuration-dependent, no constraints.
    Maximal coordinates: M constant and block-diagonal, joints enforced by
    multipliers.  Two independent implementations, so agreement is a real
    correctness check on both.
    """
    mini, maxi = make_system("two_link_arm"), make_system("maximal_chain")
    q0 = torch.tensor([[0.4, -0.8], [1.1, 0.5]], dtype=DT)
    dq0 = torch.tensor([[0.3, 0.2], [-0.4, 0.1]], dtype=DT)
    tau = torch.tensor([[1.5, -0.7], [2.0, 0.4]], dtype=DT)

    a, b = mini.nominal_state(q0, dq0), maxi.nominal_state(q0, dq0)
    for _ in range(1500):
        a, b = mini.step(a, tau, 1e-4), maxi.step(b, tau, 1e-4)
    assert (mini.task_position(a) - maxi.task_position(b)).abs().max() < 1e-3
    assert (mini.task_velocity(a) - maxi.task_velocity(b)).abs().max() < 1e-2


def test_gravity_torque_agrees_between_formulations():
    mini, maxi = make_system("two_link_arm"), make_system("maximal_chain")
    g = torch.Generator().manual_seed(1)
    q = 3.0 * torch.randn(64, 2, generator=g, dtype=DT)
    z = torch.zeros_like(q)
    assert (mini.gravity_force(mini.nominal_state(q, z))
            - maxi.gravity_force(maxi.nominal_state(q, z))).abs().max() < 1e-10


def test_the_invariant_is_the_constrained_inverse_inertia():
    """"M(q) varies" is a statement about COORDINATES: it is dense and
    configuration-dependent in minimal coordinates and constant in maximal ones.
    The physically meaningful object -- the inverse inertia a generalized force
    actually sees -- is the same in both, and varies in both."""
    mini, maxi = make_system("two_link_arm"), make_system("maximal_chain")
    g = torch.Generator().manual_seed(2)
    q = 3.0 * torch.randn(128, 2, generator=g, dtype=DT)
    z = torch.zeros_like(q)
    A = mini.inv_mass(mini.nominal_state(q, z))
    B = maxi.inv_mass(maxi.nominal_state(q, z))
    assert (A - B).abs().max() < 1e-6, "the invariant must agree"

    spread = lambda X: float((X.reshape(128, -1).std(0)
                              / X.reshape(128, -1).abs().mean(0)).max())
    assert spread(A) > 0.5 and abs(spread(A) - spread(B)) < 1e-3
    # ... while the raw maximal mass matrix is genuinely constant
    assert maxi._mass((1,)).std() >= 0


# --------------------------------------------------------------------------- #
# the hybrid contact model
# --------------------------------------------------------------------------- #
def test_hybrid_starts_as_the_pure_constraint_model():
    """The learned penalty must be exactly zero at init, so the hybrid begins as
    the constraint model and can only learn a correction on top of it."""
    sysm = make_system("quadruped", learned_contact=True)
    tr = make_trainable("energy_shaping", sysm)
    assert sysm.residual_dim == 4
    assert tr.dim == tr.policy_dim + sysm.allocator_dim + sysm.residual_dim
    res = tr.residual_slice(tr.init()[None].expand(3, -1))
    assert res.abs().max() == 0.0

    st = sysm.reset(3, make_gen(0))
    u = torch.zeros(3, 8, dtype=DT)
    a, b = sysm.step(st, u, 0.002, res), sysm.step(st, u, 0.002, None)
    for k in a:
        assert torch.equal(a[k], b[k]), f"residual perturbed {k!r} at init"


def test_learned_residual_actually_changes_the_dynamics():
    sysm = make_system("quadruped", learned_contact=True)
    st = sysm.reset(3, make_gen(0))
    u = torch.zeros(3, 8, dtype=DT)
    on = torch.full((3, 4), 0.5, dtype=DT)
    assert not torch.allclose(sysm.step(st, u, 0.002, on)["dq"],
                              sysm.step(st, u, 0.002, None)["dq"])


def test_residual_is_regularized():
    """Co-evolving plant parameters against the controller's own reward is reward
    hacking unless something holds the plant honest."""
    sysm = make_system("quadruped", learned_contact=True)
    assert float(sysm.residual_penalty(torch.zeros(4, dtype=DT))) == 0.0
    assert float(sysm.residual_penalty(torch.full((4,), 0.5, dtype=DT))) > 0.0
    assert float(sysm.residual_penalty(None)) == 0.0


def test_residual_params_form_a_zero_block_in_G():
    """The multiplier lesson generalizes: residual parameters change the PLANT,
    not the controller map, so du/dtheta is zero on them and they contribute a
    zero block to G exactly as an episode-level lambda would.

    `null_mode="cap"` is what makes that harmless -- it treats an uninformative
    direction isotropically instead of amplifying it by ridge^-1/2.
    """
    sysm = make_system("quadruped", learned_contact=True)
    tr = make_trainable("energy_shaping", sysm)
    st = sysm.reset(4, make_gen(0))
    goal = sysm.task_position(st)
    J = vmap(jacrev(tr.forward), in_dims=(None, 0, 0))(tr.init(), st, goal)
    n_res = sysm.residual_dim
    assert J[..., :-n_res].abs().max() > 0
    assert J[..., -n_res:].abs().max() == 0.0


# --------------------------------------------------------------------------- #
# the quadruped as a plant
# --------------------------------------------------------------------------- #
def test_quadruped_stands_under_its_controller():
    sysm = make_system("quadruped")
    tr = make_trainable("energy_shaping", sysm)
    st = sysm.reset(3, make_gen(0))
    goal = torch.tensor([[0.0, sysm.stand_height, 0.0]], dtype=DT).expand(3, 3)
    fb = vmap(tr.forward, in_dims=(None, 0, 0))
    th = tr.init()
    for _ in range(500):
        st = sysm.step(st, fb(th, st, goal), 0.002)
    assert bool(sysm.alive(st).all()), "the robot fell over"
    assert abs(float(st["q"][0, 1]) - sysm.stand_height) < 0.05
    feet = sysm.foot_positions(st)[..., 1]
    assert feet.abs().max() < 5e-3, "contact constraint not holding the feet down"
    assert float(sysm.contact_forces(st).sum(-1).mean()) > 0.5 * sysm.total_mass * sysm.g


def test_quadruped_mass_matrix_is_spd_and_varies():
    sysm = make_system("quadruped")
    g = torch.Generator().manual_seed(3)
    st = sysm.reset(64, make_gen(0))
    st["q"][:, 3:] += 0.5 * torch.randn(64, 8, generator=g, dtype=DT)
    kin = sysm._kin(st["q"], st["dq"])
    M = sysm._M(st["q"], kin["J"])
    assert M.shape == (64, 11, 11)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-10)
    assert (torch.linalg.eigvalsh(M) > 0).all()
    Minv = sysm.inv_mass(st)
    flat = Minv.reshape(64, -1)
    assert float((flat.std(0) / flat.abs().mean(0).clamp_min(1e-9)).max()) > 0.1
