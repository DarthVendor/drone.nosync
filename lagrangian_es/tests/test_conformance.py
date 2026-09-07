"""Parameterized over the full Cartesian product SYSTEMS x TRAINABLES.

Every check here is a property the abstractions promise, so a new system or a new
trainable is conformant the moment it registers -- and if adding one requires
touching anything downstream, the seam is in the wrong place.
"""
import copy

import pytest
import torch
from torch.func import jacrev, vmap

from lagrangian_es.environments import GROUPS, make_group
from lagrangian_es.environments.base import ObstacleGroup
from lagrangian_es.systems import SYSTEMS, make_system
from lagrangian_es.trainables import TRAINABLES, make_trainable
from lagrangian_es.util import make_gen, tree_where

PAIRS = [(s, t) for s in sorted(SYSTEMS) for t in sorted(TRAINABLES)
         if TRAINABLES[t].supports(SYSTEMS[s]())]
IDS = [f"{s}+{t}" for s, t in PAIRS]
B = 8


@pytest.fixture(params=PAIRS, ids=IDS)
def pair(request):
    sname, tname = request.param
    system = make_system(sname)
    return system, make_trainable(tname, system)


def _goal(system, B=B, scale=1.0):
    g = torch.zeros(B, system.task_dim, dtype=system.dtype)
    return g + scale * torch.linspace(0.2, 0.8, system.task_dim, dtype=system.dtype)


# --------------------------------------------------------------------------- #
def test_declared_dims_are_consistent(pair):
    system, tr = pair
    assert system.n_force > 0 and system.task_dim > 0
    assert system.allocator_dim >= 0
    assert tr.policy_dim > 0
    assert tr.dim == tr.policy_dim + system.allocator_dim
    assert tr.init().shape == (tr.dim,)
    assert torch.isfinite(tr.init()).all()


def test_shapes_across_the_whole_interface(pair):
    system, tr = pair
    s = system.reset(B, make_gen(0))
    assert system.task_position(s).shape == (B, system.task_dim)
    assert system.task_velocity(s).shape == (B, system.task_dim)
    assert system.gravity_force(s).shape == (B, system.task_dim)

    Minv = system.inv_mass(s)
    want = (B, system.n_force, system.n_force) if system.dense_mass else (B, system.n_force)
    assert Minv.shape == want, f"inv_mass shape {Minv.shape} contradicts dense_mass"
    assert torch.isfinite(Minv).all()

    phi = tr.init()[tr.policy_dim :].expand(B, system.allocator_dim)
    u = system.allocate(system.gravity_force(s), s, phi)
    assert u.shape == (B, system.n_force)

    s2 = system.step(s, u, 0.02)
    assert set(s2) == set(s)
    for k in s:
        assert s2[k].shape == s[k].shape
    assert system.alive(s).shape == (B,) and system.alive(s).dtype == torch.bool
    assert system.effort(u, s).shape == (B,)
    assert system.shaping_cost(s).shape == (B,)
    assert system.saturation(u, s).shape == (B,)


def test_nominal_state_round_trips(pair):
    system, _ = pair
    x = _goal(system)
    v = 0.3 * torch.ones_like(x)
    s = system.nominal_state(x, v)
    assert torch.allclose(system.task_position(s), x, atol=1e-12)
    assert torch.allclose(system.task_velocity(s), v, atol=1e-12)


def test_step_is_pure(pair):
    """`step` must not mutate its input: the rollout freezes crashed entries by
    keeping the *old* state object, so an in-place integrator silently advances
    vehicles that are supposed to be dead."""
    system, tr = pair
    s = system.reset(B, make_gen(1))
    ref = copy.deepcopy(s)
    phi = tr.init()[tr.policy_dim :].expand(B, system.allocator_dim)
    system.step(s, system.allocate(system.gravity_force(s), s, phi), 0.02)
    for k in s:
        assert torch.equal(s[k], ref[k]), f"step() mutated {k!r}"


def test_forward_traces_under_vmap_jacrev(pair):
    """The metric is estimated by differentiating the controller map alone.  If
    this does not trace, the whitened arm cannot exist."""
    system, tr = pair
    theta = tr.init()
    s = system.reset(B, make_gen(2))
    goal = _goal(system)

    u = vmap(tr.forward, in_dims=(None, 0, 0))(theta, s, goal)
    assert u.shape == (B, system.n_force)
    assert torch.isfinite(u).all()

    J = vmap(jacrev(tr.forward), in_dims=(None, 0, 0))(theta, s, goal)
    assert J.shape == (B, system.n_force, tr.dim), J.shape
    assert torch.isfinite(J).all(), "non-finite controller Jacobian"
    assert J.abs().sum() > 0, "controller Jacobian is identically zero"


def test_no_grad_leak_through_forward(pair):
    """`forward` is differentiated w.r.t. theta on purpose and w.r.t. nothing
    else.  The state must not carry graph out of the controller."""
    system, tr = pair
    theta = tr.init().clone().requires_grad_(True)
    s = system.reset(4, make_gen(3))
    goal = _goal(system, 4)
    with torch.enable_grad():
        u = vmap(tr.forward, in_dims=(None, 0, 0))(theta, s, goal)
        u.sum().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    for k, v in s.items():
        assert not v.requires_grad, f"state key {k!r} acquired requires_grad"


def test_equilibrium_at_the_goal_is_by_construction(pair):
    """At the goal with zero velocity, an energy-shaping controller must emit
    exactly the gravity feedforward, for EVERY genome -- not just the trained one.

    Stated plant-agnostically: `forward` reduces to `allocate(gravity_force)`.
    On a fully-actuated plant, where `allocate` is the identity, that is literally
    "u == gravity_force".  On the quadrotor it is the same statement pushed
    through the underactuation seam.
    """
    system, tr = pair
    if not tr.equilibrium_exact:
        pytest.skip(f"{type(tr).__name__} makes no equilibrium claim")

    x = _goal(system)
    s = system.nominal_state(x, torch.zeros_like(x))
    gen = make_gen(4)
    for trial in range(8):
        theta = tr.init() if trial == 0 else torch.randn(
            tr.dim, generator=gen, dtype=system.dtype)
        u = vmap(tr.forward, in_dims=(None, 0, 0))(theta, s, x)
        phi = theta[tr.policy_dim :].expand(B, system.allocator_dim)
        u_ff = system.allocate(system.gravity_force(s), s, phi)
        assert torch.allclose(u, u_ff, atol=1e-4), (
            f"trial {trial}: max dev {(u - u_ff).abs().max():.2e}")


def test_output_finite_far_from_the_goal(pair):
    """||e|| = 1e4 and ||v|| = 1e3.  The pseudo-Huber bowls bound the commanded
    force through the potential's own geometry, so this must not blow up."""
    system, tr = pair
    theta = tr.init()
    d = system.task_dim
    for scale_e, scale_v in [(1e4, 0.0), (0.0, 1e3), (1e4, 1e3)]:
        x = torch.full((B, d), scale_e / d**0.5, dtype=system.dtype)
        v = torch.full((B, d), scale_v / d**0.5, dtype=system.dtype)
        s = system.nominal_state(x, v)
        u = vmap(tr.forward, in_dims=(None, 0, 0))(theta, s, torch.zeros_like(x))
        assert torch.isfinite(u).all(), f"non-finite u at |e|={scale_e} |v|={scale_v}"


def test_bounded_command_far_from_the_goal(pair):
    """Specifically for the equilibrium-shaping family: the *potential* gradient
    saturates, so the force does not grow without bound in ||e||."""
    system, tr = pair
    if not tr.equilibrium_exact or not hasattr(tr, "grad_potential"):
        pytest.skip("no shaped potential to bound")
    theta = tr.init()
    d = system.task_dim
    mags = []
    for scale in [1e1, 1e2, 1e3, 1e4]:
        e = torch.full((4, d), scale / d**0.5, dtype=system.dtype)
        mags.append(float(tr.grad_potential(theta.expand(4, -1), e).norm(dim=-1).max()))
    assert mags[-1] < 2 * mags[0] + 1e-6, f"grad V not saturating: {mags}"


def test_freeze_leaves_state_bit_identical(pair):
    """An all-False mask is what a fully crashed population looks like."""
    system, _ = pair
    s = system.reset(B, make_gen(5))
    advanced = {k: v + 1.0 for k, v in s.items()}
    frozen = tree_where(torch.zeros(B, dtype=torch.bool), advanced, s)
    for k in s:
        assert torch.equal(frozen[k], s[k]), f"freeze altered {k!r}"
    thawed = tree_where(torch.ones(B, dtype=torch.bool), advanced, s)
    for k in s:
        assert torch.equal(thawed[k], advanced[k])


def test_reset_is_reproducible(pair):
    system, _ = pair
    a = system.reset(B, make_gen(7))
    b = system.reset(B, make_gen(7))
    for k in a:
        assert torch.equal(a[k], b[k])


# --- every raycast agrees with the SDF that defines its surface ---------------

def _free_origins(grp, B, seed, extent):
    import torch

    from lagrangian_es.util import make_gen

    f = grp.sample(B, make_gen(seed), torch.float64, "cpu")
    o = torch.rand(B, 3, generator=make_gen(seed + 31), dtype=torch.float64)
    o = (o * 2 - 1) * extent
    o[:, 2] = 1.0
    keep = grp.sdf(o, f) > 0.15
    return o[keep], {k: v[keep] for k, v in f.items()}


def _fan(n, M, tilt):
    import torch

    th = torch.linspace(0.0, 6.283185307, M, dtype=torch.float64)
    d = torch.zeros(n, M, 3, dtype=torch.float64)
    d[..., 0], d[..., 1] = torch.cos(th), torch.sin(th)
    d[..., 2] = tilt
    return d / d.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize("kind", sorted(GROUPS))
@pytest.mark.parametrize("tilt", [0.0, 0.35])
def test_every_raycast_agrees_with_its_own_sdf(kind, tilt):
    """A raycast is a second implementation of a surface the SDF already defines.

    Three shipped primitives disagreed with their own definition and none of the
    disagreements was visible from any other test:

      * `march` never terminated, so rays walked THROUGH obstacles and out the
        far side, and were then reported as misses -- worse with more steps.
      * `Pillars` and `Gate` solved the xy intersection with the unit-direction
        closed form, which is wrong by 1/|d_xy| for any ray that is not
        horizontal.  The shipped beam fan is horizontal, so only the depth
        camera saw it.
      * `Walls` subtracted its thickness off the centreline distance, but its
        SDF is a capsule with round caps.

    So the check is now structural and runs over the registry: whatever a
    raycast reports, the SDF must agree that the point is ON the surface
    (soundness) and that nothing was crossed before it (completeness).  `tilt`
    is parameterised because a horizontal fan is exactly the case that hid two
    of these.
    """
    import torch

    grp = make_group(kind)
    extent = float(getattr(grp, "extent", 2.4))
    o, f = _free_origins(grp, 48, 3, extent)
    n = o.shape[0]
    if n < 4:
        pytest.skip(f"{kind}: too few free origins to test")
    M = 64
    d = _fan(n, M, tilt)
    rng, grad = grp.raycast(o, d, f, 6.0)
    assert torch.isfinite(rng).all() and torch.isfinite(grad).all()
    hit = rng < 6.0 - 1e-9
    if not bool(hit.any()):
        pytest.skip(f"{kind}: no hits at tilt {tilt}")

    exp = lambda k: {kk: vv.repeat_interleave(k, 0) for kk, vv in f.items()}
    # SOUNDNESS -- the reported point is on the surface.  A marched primitive
    # halts within `tol` of it by construction, so it gets that slack; a
    # closed-form one has no excuse.
    p = o[:, None, :] + d * rng[..., None]
    sd = grp.sdf(p.reshape(-1, 3), exp(M)).reshape(rng.shape)
    analytic = type(grp).raycast is not ObstacleGroup.raycast
    tol = 1e-9 if analytic else 6e-3
    assert float(sd[hit].abs().max()) < tol, (kind, tilt, float(sd[hit].abs().max()))

    # COMPLETENESS -- nothing is crossed before it
    S = 120
    ts = torch.linspace(0.0, 0.999, S, dtype=torch.float64)
    pre = o[:, None, None, :] + d[:, :, None, :] * (rng[..., None, None]
                                                    * ts[None, None, :, None])
    sp = grp.sdf(pre.reshape(-1, 3), exp(M * S))
    assert float(sp.min()) > -1e-6, (kind, tilt, float(sp.min()))


@pytest.mark.parametrize("kind", sorted(GROUPS))
def test_every_sdf_is_a_true_distance_function(kind):
    """|grad sdf| = 1 outside, or it is not a distance and everything built on
    it is subtly wrong.

    Sphere marching steps by the SDF and is only guaranteed not to overshoot a
    surface when the value is a true distance; `RangeBarrier` and the proximity
    penalty differentiate it, so a gradient of the wrong magnitude is a force of
    the wrong magnitude.  Worth stating as a property because it is the
    assumption every other piece of geometry code makes silently.

    Measured at the same time as the raycast cross-check, and all five
    primitives pass at 100% -- which is the useful half of that result: the
    definitions were right and the hand-written intersections that disagreed
    with them were the bugs.
    """
    import torch

    from lagrangian_es.util import make_gen

    grp = make_group(kind)
    B, P, h = 16, 256, 1e-5
    f = grp.sample(B, make_gen(2), torch.float64, "cpu")
    ext = float(getattr(grp, "extent", 2.4))
    p = (torch.rand(B, P, 3, generator=make_gen(8), dtype=torch.float64) * 2 - 1)
    p = p * ext * 1.3
    p[..., 2] = p[..., 2].abs() * 1.5 + 0.05
    flat = {k: v[:, None].expand((B, P) + v.shape[1:]).reshape((-1,) + v.shape[1:])
            for k, v in f.items()}

    def sdf(q):
        return grp.sdf(q.reshape(-1, 3), flat).reshape(B, P)

    base = sdf(p)
    g = []
    for i in range(3):
        d = torch.zeros(3, dtype=torch.float64)
        d[i] = h
        g.append((sdf(p + d) - sdf(p - d)) / (2 * h))
    norm = torch.stack(g, -1).norm(dim=-1)
    outside = base > 0.05           # away from the surface and from any cusp
    assert bool(outside.any()), kind
    q = norm[outside]
    frac = float(((q - 1.0).abs() < 0.01).to(torch.float64).mean())
    assert frac > 0.98, (kind, frac, float(q.median()))
