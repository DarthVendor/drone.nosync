"""The learned gyroscopic head: it may steer, it may not add energy.

A gradient potential plus a Rayleigh dissipation flows downhill into the nearest
critical point of `V_d`, and by LaSalle that is where it stops -- saddles
included.  On the city map 94% of stalls are exactly that: the vehicle pinned
0.35 m off a building with the goal 16.5 m past it, held by a balance normal to
the wall while the way out runs along it.  No gradient can push along the wall,
so the term is given a workless one.

`S = G - G^T` is skew for every G, so `v . S v = 0` identically and the head
cannot inject energy however the weights come out.  That is the property the
certificate rests on, so it is tested as an identity, not to a tolerance.
"""
import torch

from lagrangian_es.trainables.learned import LearnedShaping


def _pair(d=3, n_obs=24, hidden=16):
    return (LearnedShaping(d, n_obs=n_obs, hidden=hidden, gyro=False),
            LearnedShaping(d, n_obs=n_obs, hidden=hidden, gyro=True))


def _inputs(n=9, d=3, n_obs=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g, dtype=torch.float64),
            torch.randn(n, d, generator=g, dtype=torch.float64),
            {"range": torch.rand(n, n_obs, generator=g, dtype=torch.float64)}, g)


def test_the_head_is_exactly_skew_symmetric():
    _, wide = _pair()
    e, v, obs, g = _inputs()
    theta = 0.4 * torch.randn(wide.dim, generator=g, dtype=torch.float64)
    z = (obs["range"] / wide.obs_scale).clamp(-4.0, 4.0)
    S = wide._S(theta, z)
    assert torch.equal(S, -S.transpose(-1, -2))


def test_the_head_does_no_work():
    """v . S v = 0 is what keeps H = T + V_d non-increasing."""
    _, wide = _pair()
    e, v, obs, g = _inputs()
    theta = 0.8 * torch.randn(wide.dim, generator=g, dtype=torch.float64)
    z = (obs["range"] / wide.obs_scale).clamp(-4.0, 4.0)
    S = wide._S(theta, z)
    power = torch.einsum("bi,bij,bj->b", v, S, v)
    assert float(power.abs().max()) < 1e-12


def test_zero_weights_reproduce_the_term_without_the_head():
    """A genome trained without the head extends to one with it by appending
    zeros, so the head can be switched on as a warm start rather than a restart."""
    narrow, wide = _pair()
    e, v, obs, g = _inputs()
    theta = 0.3 * torch.randn(narrow.dim, generator=g, dtype=torch.float64)
    big = torch.cat([theta, torch.zeros(wide.dim - narrow.dim,
                                        dtype=torch.float64)])
    assert torch.equal(narrow.grad_potential(theta, e, v, e, obs),
                       wide.grad_potential(big, e, v, e, obs))
    assert torch.equal(narrow.potential(theta, e, v, e, obs),
                       wide.potential(big, e, v, e, obs))


def test_the_head_actually_changes_the_force_when_it_is_on():
    """Guard against a head that is quietly inert."""
    narrow, wide = _pair()
    e, v, obs, g = _inputs()
    theta = 0.3 * torch.randn(narrow.dim, generator=g, dtype=torch.float64)
    live = torch.cat([theta, 0.7 * torch.randn(wide.dim - narrow.dim,
                                               generator=g, dtype=torch.float64)])
    a = narrow.grad_potential(theta, e, v, e, obs)
    b = wide.grad_potential(live, e, v, e, obs)
    assert not torch.allclose(a, b)


def test_the_head_vanishes_with_the_velocity():
    """S v is proportional to v, so a stalled vehicle gets nothing from it --
    the head steers a moving vehicle around, it does not unstick a stopped one."""
    _, wide = _pair()
    e, v, obs, g = _inputs()
    theta = 0.5 * torch.randn(wide.dim, generator=g, dtype=torch.float64)
    zero_v = torch.zeros_like(v)
    with_v = wide.grad_potential(theta, e, v, e, obs)
    without = wide.grad_potential(theta, e, zero_v, e, obs)
    z = (obs["range"] / wide.obs_scale).clamp(-4.0, 4.0)
    S = wide._S(theta, z)
    assert torch.allclose(with_v - without,
                          (S @ v.unsqueeze(-1)).squeeze(-1)
                          + (wide._L(theta, z) @ (v.unsqueeze(-2)
                             @ wide._L(theta, z)).squeeze(-2).unsqueeze(-1)
                             ).squeeze(-1))
