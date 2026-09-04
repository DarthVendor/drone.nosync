import torch
from torch.func import jacrev, vmap

from lagrangian_es.systems import make_system
from lagrangian_es.trainables import make_trainable

DT = torch.float64


def _tr(**kw):
    return make_trainable("energy_shaping", make_system("quadrotor"), **kw)


def test_potential_is_nonnegative_with_a_unique_minimum_at_the_goal():
    """The claim that every individual -- including a random one -- is an
    energy-shaping controller whose equilibrium is the goal, tested on 1000
    random genomes rather than on the prior."""
    tr = _tr()
    g = torch.Generator().manual_seed(0)
    TH = 2.0 * torch.randn(1000, tr.dim, generator=g, dtype=DT)
    zero = torch.zeros(1000, 3, dtype=DT)

    assert tr.potential(TH, zero).abs().max() < 1e-12          # V(0) = 0
    assert tr.grad_potential(TH, zero).abs().max() < 1e-12     # and it is a critical point

    for scale in [1e-3, 1e-1, 1.0, 10.0]:
        e = scale * torch.randn(1000, 3, generator=g, dtype=DT)
        V = tr.potential(TH, e)
        assert (V >= 0).all(), f"V < 0 at scale {scale}: min {V.min():.3e}"
        assert torch.isfinite(V).all()


def test_potential_is_strictly_positive_away_from_the_goal():
    """Nonnegativity alone would be satisfied by V == 0.  With w != 0 and A
    invertible the minimum must actually be unique."""
    tr = _tr()
    th = tr.init()
    for scale in [1e-2, 1.0, 10.0]:
        e = scale * torch.nn.functional.normalize(torch.randn(256, 3, dtype=DT), dim=-1)
        assert (tr.potential(th.expand(256, -1), e) > 0).all()


def test_grad_potential_is_bounded_as_error_diverges():
    """Pseudo-Huber bowls: the commanded force saturates through the potential's
    own geometry.  This is what makes actuator limits a property of the search
    space rather than a clip applied after the fact."""
    tr = _tr()
    g = torch.Generator().manual_seed(1)
    TH = torch.randn(64, tr.dim, generator=g, dtype=DT)

    # Directions are drawn ONCE and reused: the saturation value depends on the
    # direction, so redrawing per scale compares different limits.
    dirs = torch.nn.functional.normalize(torch.randn(64, 3, generator=g, dtype=DT), dim=-1)

    mags = []
    for scale in [1e0, 1e2, 1e4, 1e6, 1e8]:
        m = tr.grad_potential(TH, scale * dirs).norm(dim=-1)
        assert torch.isfinite(m).all()
        mags.append(m)

    # Along a fixed ray the magnitude must converge to the asymptote
    # ||sum_k w_k^2 A_k^T A_k d / ||A_k d|| ||, so successive scales stop moving.
    rel = ((mags[-1] - mags[-2]).abs() / mags[-2].clamp_min(1e-12)).max()
    assert rel < 1e-3, f"not converged along a fixed ray: rel change {rel:.3e}"

    # ... and the bound is uniform: a 1e8-fold increase in ||e|| must not move it.
    growth = (mags[-1] / mags[0].clamp_min(1e-12)).max()
    assert growth < 3.0, f"grad V grew {growth:.1f}x over 8 decades of ||e||"

    # The analytic bound sum_k w_k^2 sigma_max(A_k) must hold; each bowl now
    # publishes its own share of it through its certificate.
    from lagrangian_es.trainables.terms import GoalBowl
    bound = torch.zeros(64, dtype=DT)
    for term, sl in zip(tr.terms, tr.term_slices(TH)):
        if isinstance(term, GoalBowl):
            w2, A = term._wA(sl)
            bound = bound + w2 * torch.linalg.matrix_norm(A, ord=2)
    assert (mags[-1] <= bound + 1e-8).all()


def test_grad_matches_autograd_of_the_potential():
    """The closed-form gradient is used because `forward` must be cheap, but it
    still has to be the gradient of the potential it claims to differentiate."""
    tr = _tr()
    g = torch.Generator().manual_seed(2)
    th = torch.randn(tr.dim, generator=g, dtype=DT)
    e = torch.randn(32, 3, generator=g, dtype=DT)
    auto = vmap(jacrev(lambda x: tr.potential(th, x)))(e)
    assert torch.allclose(auto, tr.grad_potential(th.expand(32, -1), e), atol=1e-10)


def test_damping_is_psd_by_construction():
    tr = _tr()
    g = torch.Generator().manual_seed(3)
    TH = 3.0 * torch.randn(1000, tr.dim, generator=g, dtype=DT)
    eig = torch.linalg.eigvalsh(tr.damping(TH))
    assert (eig >= -1e-10).all(), f"min eig {eig.min():.3e}"
    Kd = tr.damping(TH)
    assert torch.allclose(Kd, Kd.transpose(-1, -2), atol=1e-12)


def test_stiffness_is_psd_and_is_the_hessian_at_the_goal():
    tr = _tr()
    g = torch.Generator().manual_seed(4)
    th = torch.randn(tr.dim, generator=g, dtype=DT)
    K = tr.stiffness(th)
    assert torch.allclose(K, K.T, atol=1e-12)
    assert (torch.linalg.eigvalsh(K) >= -1e-10).all()
    hess = jacrev(lambda x: tr.grad_potential(th, x))(torch.zeros(3, dtype=DT))
    assert torch.allclose(hess, K, atol=1e-9)


def test_prior_matches_the_specified_physical_numbers():
    """w = 1, A_k = I, D = 1.2 I, kR = 0.25, kW = 0.10 -> stiffness 3 N/m and
    omega_n ~ 2.4 rad/s at 0.5 kg.  These are the prototype's numbers; drifting
    off them changes the difficulty of the whole benchmark."""
    tr = _tr()
    th = tr.init()
    K = tr.stiffness(th)
    assert torch.allclose(K, 3.0 * torch.eye(3, dtype=DT), atol=1e-12)
    assert torch.allclose(tr.damping(th), 1.44 * torch.eye(3, dtype=DT), atol=1e-12)
    d = tr.describe(th)
    assert abs(d["wn_pos"] - 2.449) < 1e-2
    assert abs(d["kR"] - 0.0625) < 1e-12 and abs(d["kW"] - 0.01) < 1e-12


def test_invariants_reported_for_the_conformance_contract():
    tr = _tr()
    inv = tr.invariants(tr.init())
    assert {"V_at_goal", "V_min", "Kd_eigs"} <= set(inv)
    assert float(inv["V_at_goal"]) == 0.0
    assert float(inv["V_min"]) >= 0.0
    assert (inv["Kd_eigs"] >= -1e-12).all()


def test_describe_exposes_bowl_anisotropy():
    """Three identical A_k is a valid but boring optimum; `describe` has to make
    that visible or you cannot tell whether NK = 1 would do just as well."""
    tr = _tr()
    iso = tr.describe(tr.init())
    assert abs(iso["K_aniso"] - 1.0) < 1e-9      # the prior IS isotropic

    th = tr.init().clone()
    th[1] = 3.0            # bowl 0 is [w, A(9)]; th[1] is A[0,0]
    assert tr.describe(th)["K_aniso"] > 2.0


def test_term_slices_are_batch_rank_agnostic():
    tr = _tr()
    for shape in [(), (5,), (2, 5)]:
        th = torch.randn(shape + (tr.dim,), dtype=DT)
        slices = tr.term_slices(th)
        assert len(slices) == len(tr.terms)
        for term, sl in zip(tr.terms, slices):
            assert sl.shape == shape + (term.dim,)
        assert tr.damping(th).shape == shape + (3, 3)
        assert tr.stiffness(th).shape == shape + (3, 3)


def test_default_terms_reproduce_the_original_fixed_potential():
    """The composed default -- three bowls plus dissipation -- must be the same
    39-slot controller the flat implementation was, or every measured number in
    the repo silently shifts."""
    tr = _tr()
    assert tr.policy_dim == 39 and tr.dim == 45
    assert [t.kind for t in tr.terms] == ["goal_bowl"] * 3 + ["dissipation"]
    th = tr.init()
    assert torch.allclose(tr.stiffness(th), 3.0 * torch.eye(3, dtype=DT), atol=1e-12)
    assert torch.allclose(tr.damping(th), 1.44 * torch.eye(3, dtype=DT), atol=1e-12)


def test_terms_compose_additively():
    """grad(sum_i V_i) = sum_i grad V_i -- the algebra the whole design rests on."""
    from lagrangian_es.trainables.terms import GoalBowl
    sys_ = make_system("quadrotor")
    one = make_trainable("energy_shaping", sys_, terms=[GoalBowl(3)])
    three = make_trainable("energy_shaping", sys_, terms=[GoalBowl(3) for _ in range(3)])
    e = torch.randn(64, 3, dtype=DT)
    th1, th3 = one.init(), three.init()
    assert torch.allclose(three.potential(th3, e), 3.0 * one.potential(th1, e), atol=1e-12)
    assert torch.allclose(three.grad_potential(th3, e),
                          3.0 * one.grad_potential(th1, e), atol=1e-12)


def test_composition_preserves_nonnegativity_and_the_equilibrium():
    """Any conic combination of terms that each vanish at the goal still vanishes
    there, and stays nonnegative -- so a random genome remains an energy-shaping
    controller no matter which terms are stacked."""
    from lagrangian_es.trainables.terms import DissipationTerm, GoalBowl, KineticShaping
    sys_ = make_system("quadrotor")
    tr = make_trainable("energy_shaping", sys_,
                        terms=[GoalBowl(3), GoalBowl(3), KineticShaping(3),
                               DissipationTerm(3)])
    g = torch.Generator().manual_seed(5)
    TH = 2.0 * torch.randn(300, tr.dim, generator=g, dtype=DT)
    zero = torch.zeros(300, 3, dtype=DT)
    assert tr.potential(TH, zero).abs().max() < 1e-12
    assert tr.grad_potential(TH, zero).abs().max() < 1e-12
    e = 2.0 * torch.randn(300, 3, generator=g, dtype=DT)
    assert (tr.potential(TH, e) >= 0).all()
    assert tr.equilibrium_exact


def test_a_barrier_overlapping_the_goal_withdraws_the_equilibrium_claim():
    """A barrier genuinely moves the equilibrium, so the claim must lapse rather
    than quietly become false.  This is what `certificate` is for."""
    from lagrangian_es.trainables.terms import DissipationTerm, GoalBowl, ObstacleBarrier
    sys_ = make_system("quadrotor")
    tr = make_trainable("energy_shaping", sys_,
                        terms=[GoalBowl(3), DissipationTerm(3),
                               ObstacleBarrier(3, center=(1.0, 0.0, 1.5), radius=0.4)])
    assert tr.equilibrium_exact is False, "unconditional claim must lapse"

    th = tr.init()
    far = torch.tensor([-3.0, -3.0, 2.0], dtype=DT)
    near = torch.tensor([1.0, 0.0, 1.5], dtype=DT)
    assert tr.equilibrium_exact_for(th, far) is True      # goal clear of the barrier
    assert tr.equilibrium_exact_for(th, near) is False     # goal inside it


def test_barriers_vanish_identically_in_the_free_region():
    """Compact support is what lets the equilibrium claim be conditional rather
    than forfeit: in the region the barrier does not police, both value and
    gradient are EXACTLY zero, not merely small.

    "Free" means different things for the two barriers: far from the sphere for
    an obstacle, and in the middle of the box for a joint limit.
    """
    from lagrangian_es.trainables.terms import JointLimitBarrier, ObstacleBarrier
    cases = [
        (ObstacleBarrier(3, center=(0.0, 0.0, 0.0), radius=0.5, margin0=0.6),
         torch.full((16, 3), 50.0, dtype=DT)),
        (JointLimitBarrier(3, lo=-2.0, hi=2.0, margin0=0.3),
         torch.zeros(16, 3, dtype=DT)),
    ]
    for term, free in cases:
        th = term.init(dtype=DT)
        z = torch.zeros_like(free)
        assert term.potential(th, free, z, free).abs().max() == 0.0
        assert term.grad_potential(th, free, z, free).abs().max() == 0.0


def test_barrier_force_stays_bounded_even_when_the_limit_is_violated():
    """A cubed compact bump grows without bound once the limit is actually
    crossed -- commanding an unbounded force exactly when the barrier is doing its
    job.  The linear extension caps |dpsi/ds| at 3, so the force stays bounded
    while still pushing back harder the further outside you are.
    """
    from lagrangian_es.trainables.terms import JointLimitBarrier, ObstacleBarrier
    for term in (ObstacleBarrier(3, center=(0.0, 0.0, 1.0), radius=0.5, margin0=0.6),
                 JointLimitBarrier(3, lo=-2.0, hi=2.0, margin0=0.3)):
        th = term.init(dtype=DT)
        mags = []
        for scale in (5.0, 50.0, 500.0, 5000.0):
            x = scale * torch.nn.functional.normalize(
                torch.randn(256, 3, dtype=DT), dim=-1)
            z = torch.zeros_like(x)
            p = term.potential(th, x, z, x)
            gr = term.grad_potential(th, x, z, x)
            assert (p >= 0).all() and torch.isfinite(p).all()
            assert torch.isfinite(gr).all()
            mags.append(float(gr.norm(dim=-1).max()))
        # deep violation must not escalate the commanded force without bound
        assert mags[-1] < 1.05 * mags[0] + 1e-9, f"force escalating: {mags}"

        # ... and it must still push back monotonically (nonzero restoring force)
        inside = torch.zeros(1, 3, dtype=DT)
        if isinstance(term, ObstacleBarrier):
            inside = torch.tensor([[0.0, 0.0, 1.0 + 0.1]], dtype=DT)
        else:
            inside = torch.tensor([[3.0, 0.0, 0.0]], dtype=DT)
        gi = term.grad_potential(th, inside, torch.zeros_like(inside), inside)
        assert gi.norm() > 1e-6, "no restoring force when the limit is violated"


def test_barrier_gradient_matches_autograd():
    from torch.func import jacrev, vmap
    from lagrangian_es.trainables.terms import JointLimitBarrier, ObstacleBarrier
    for term in (ObstacleBarrier(3, center=(0.3, -0.2, 1.0), radius=0.5),
                 JointLimitBarrier(3, lo=-1.5, hi=1.5)):
        th = term.init(dtype=DT)
        x = 1.2 * torch.randn(48, 3, dtype=DT)
        z = torch.zeros_like(x)
        auto = vmap(jacrev(lambda p: term.potential(th, p, torch.zeros(3, dtype=DT), p)))(x)
        assert torch.allclose(auto, term.grad_potential(th, x, z, x), atol=1e-9)


def test_kinetic_shaping_is_identity_when_absent_and_changes_force_when_present():
    from lagrangian_es.trainables.terms import DissipationTerm, GoalBowl, KineticShaping
    sys_ = make_system("quadrotor")
    base = make_trainable("energy_shaping", sys_, terms=[GoalBowl(3), DissipationTerm(3)])
    kin = make_trainable("energy_shaping", sys_,
                         terms=[GoalBowl(3), DissipationTerm(3), KineticShaping(3)])
    x = torch.tensor([[0.4, -0.3, 1.2]], dtype=DT)
    s = sys_.nominal_state(x, torch.full_like(x, 0.2))
    goal = torch.zeros(1, 3, dtype=DT)
    from torch.func import vmap
    u_base = vmap(base.forward, in_dims=(None, 0, 0))(base.init(), s, goal)

    th0 = kin.init()                      # W = 0 => M_d = M => A = I
    assert torch.allclose(vmap(kin.forward, in_dims=(None, 0, 0))(th0, s, goal),
                          u_base, atol=1e-10)
    th1 = th0.clone()
    th1[kin._bounds[2][0]:kin._bounds[2][1]] = 0.6     # a real M_d contribution
    assert not torch.allclose(vmap(kin.forward, in_dims=(None, 0, 0))(th1, s, goal),
                              u_base, atol=1e-6)


def test_segments_are_the_term_boundaries():
    tr = _tr()
    segs = tr.segments()
    assert len(segs) == len(tr.terms) + 1          # terms + allocator
    assert [s.stop - s.start for s in segs] == [t.dim for t in tr.terms] + [6]
    assert segs[-1].stop == tr.dim
