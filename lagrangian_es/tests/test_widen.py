"""Growing the learned term must not move the controller it encodes.

Capacity is only measurable against a wide net that STARTS as the narrow one --
otherwise the comparison reports the re-initialisation.  New units get live
input weights and zero output weights, which is an identity on the function, so
these check equality rather than closeness.
"""
import torch

from lagrangian_es.trainables.learned import LearnedShaping
from lagrangian_es.widen import widen


def _pair(h=16, H=32, d=3, n_obs=24):
    return (LearnedShaping(d, n_obs=n_obs, hidden=h),
            LearnedShaping(d, n_obs=n_obs, hidden=H))


def test_widening_preserves_the_potential_and_its_gradient():
    narrow, wide = _pair()
    gen = torch.Generator().manual_seed(0)
    theta = 0.3 * torch.randn(narrow.dim, generator=gen, dtype=torch.float64)
    big = widen(theta, narrow, wide, gen)
    assert big.shape[-1] == wide.dim

    e = torch.randn(7, 3, generator=gen, dtype=torch.float64)
    v = torch.randn(7, 3, generator=gen, dtype=torch.float64)
    x = torch.randn(7, 3, generator=gen, dtype=torch.float64)
    obs = {"range": torch.rand(7, 24, generator=gen, dtype=torch.float64)}
    for name in ("potential", "grad_potential"):
        a = getattr(narrow, name)(theta, e, v, x, obs)
        b = getattr(wide, name)(big, e, v, x, obs)
        assert torch.equal(a, b), f"{name} moved when the net was widened"


def test_widening_preserves_the_dissipation_head():
    """The R head reads `obs` through its own weights, which widening does not
    touch -- but it shares the genome, so a bad re-lay would corrupt it."""
    narrow, wide = _pair(h=16, H=48)
    gen = torch.Generator().manual_seed(1)
    theta = 0.3 * torch.randn(narrow.dim, generator=gen, dtype=torch.float64)
    big = widen(theta, narrow, wide, gen)
    assert torch.equal(narrow.damping(theta), wide.damping(big))
    z = torch.rand(5, 24, generator=gen, dtype=torch.float64)
    assert torch.equal(narrow._L(theta, z), wide._L(big, z))


def test_widening_is_batched_over_a_population():
    narrow, wide = _pair()
    gen = torch.Generator().manual_seed(2)
    TH = 0.3 * torch.randn(11, narrow.dim, generator=gen, dtype=torch.float64)
    big = widen(TH, narrow, wide, gen)
    assert big.shape == (11, wide.dim)
    e = torch.randn(11, 3, generator=gen, dtype=torch.float64)
    obs = {"range": torch.rand(11, 24, generator=gen, dtype=torch.float64)}
    assert torch.equal(narrow.potential(TH, e, e, e, obs),
                       wide.potential(big, e, e, e, obs))


def test_widen_refuses_to_shrink():
    wide, narrow = _pair(h=32, H=16)
    gen = torch.Generator().manual_seed(3)
    theta = torch.randn(wide.dim, generator=gen, dtype=torch.float64)
    try:
        widen(theta, wide, narrow, gen)
    except ValueError as e:
        assert "grows" in str(e)
    else:
        raise AssertionError("shrinking should be refused")


def test_extend_policy_keeps_the_allocator_after_the_padding():
    """`[policy | allocator]`: padding at the end would shift the allocator into
    the new slots and zero the gains that point the thrust."""
    from lagrangian_es.widen import extend_policy
    theta = torch.arange(10, dtype=torch.float64)          # policy 0..6, alloc 7..9
    got = extend_policy(theta, old_policy_dim=7, new_policy_dim=10)
    assert torch.equal(got[:7], theta[:7])
    assert torch.equal(got[7:10], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(got[10:], theta[7:]), "the allocator must survive intact"


def test_extend_policy_is_a_no_op_when_nothing_grows():
    from lagrangian_es.widen import extend_policy
    theta = torch.randn(9, dtype=torch.float64)
    assert torch.equal(extend_policy(theta, 5, 5), theta)


def test_extend_policy_refuses_to_shrink():
    from lagrangian_es.widen import extend_policy
    try:
        extend_policy(torch.zeros(6, dtype=torch.float64), 5, 3)
    except ValueError as e:
        assert "grows" in str(e)
    else:
        raise AssertionError("shrinking the policy should be refused")


def test_extending_a_real_genome_for_the_gyro_head_is_the_same_controller():
    """End to end: the trainable's own forward map must not move."""
    from lagrangian_es.config import Config
    from lagrangian_es.es import build
    from lagrangian_es.widen import extend_policy

    def mk(gyro):
        cfg = Config(system="quadrotor_nav", trainable="nav_agent",
                     task="waypoint_pair", environment="pillars",
                     sensors=("range",), gating="arrival", seed=0,
                     trainable_kw=(("learned", True), ("gyro", gyro)))
        return build(cfg)

    (system, narrow, task) = mk(False)
    (_, wide, _) = mk(True)
    gen = torch.Generator().manual_seed(4)
    theta = 0.2 * torch.randn(narrow.dim, generator=gen, dtype=torch.float64)
    big = extend_policy(theta, narrow.policy_dim, wide.policy_dim)
    assert big.shape[-1] == wide.dim

    state = system.reset(6, torch.Generator().manual_seed(5))
    goal = task.sample(6, torch.Generator().manual_seed(6))[:, 0]
    obs = {"range": torch.rand(6, 24, generator=gen, dtype=torch.float64)}
    assert torch.equal(narrow.forward(theta, state, goal, obs),
                       wide.forward(big, state, goal, obs))


def test_isotropic_transfer_keeps_the_potential_and_sets_the_damping():
    """Only the damper changes character; V must come through untouched, and the
    new R must equal target * I so the A/B is about shape, not magnitude."""
    import torch

    from lagrangian_es.trainables.learned import LearnedShaping
    from lagrangian_es.widen import to_isotropic_damping

    full = LearnedShaping(3, n_obs=24, hidden=16, damp_mode="full")
    iso = LearnedShaping(3, n_obs=24, hidden=16, damp_mode="iso")
    gen = torch.Generator().manual_seed(7)
    # the genome carries 6 allocator slots after the term, as the trainable does
    theta = torch.cat([0.3 * torch.randn(full.dim, generator=gen,
                                         dtype=torch.float64),
                       torch.arange(6, dtype=torch.float64)])
    TARGET = 6.5
    got = to_isotropic_damping(theta, full, iso, TARGET)
    assert got.shape[-1] == iso.dim + 6

    e = torch.randn(9, 3, generator=gen, dtype=torch.float64)
    v = torch.randn(9, 3, generator=gen, dtype=torch.float64)
    obs = {"range": torch.rand(9, 24, generator=gen, dtype=torch.float64)}
    assert torch.equal(full.potential(theta[..., :full.dim], e, v, e, obs),
                       iso.potential(got[..., :iso.dim], e, v, e, obs)), \
        "the potential must survive the damper swap"

    z = iso._read(obs)
    L = iso._L(got[..., :iso.dim], z)
    R = L @ L.transpose(-1, -2)
    ev = torch.linalg.eigvalsh(R)
    assert torch.allclose(ev, torch.full_like(ev, TARGET))

    assert torch.equal(got[..., iso.dim:], theta[..., full.dim:]), \
        "the allocator must keep its values"
