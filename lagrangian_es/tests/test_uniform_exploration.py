"""Uniform-in-ACTION exploration, with an exact mixture density.

WHY. The Gaussian's sigma is 0.12, about +-21 degrees of bearing; routing around
a block needs ~+-90. A policy cannot learn what it never samples. MEASURED on
the trained router: the placement moves 1.6% of its own spread when every beam
and pixel is shuffled, while the beams ARE informative (53% of them hit, 15% of
returns inside 3 m, every decision has hits) and the gradient path IS alive
(d|mu|/d(beams) = 0.70 x d|mu|/d(goal)), and the read-out is cross-attention so
it can focus on a single beam. Information present, path open, architecture
capable -- and the detour never proposed.

WHY IT WAS OFF, and why that no longer applies:
  * "Cross-entropy has no importance ratio to correct for the distribution its
    samples came from" -- measured, 30% uniform walked the speak rate to exactly
    0.324, the mixture's own value. PPO HAS that ratio.
  * "A uniform mixture over an unbounded variable does not [keep the density
    exact]" -- true of the pre-squash u, false of the bounded action a = tanh(u),
    whose uniform density is (1/2)^k * prod(1 - tanh^2(u_i)).
"""
import math

import torch

from lagrangian_es.composer.actions_cont import ContVocab, log_prob

V = ContVocab(1)


def _args(B=4096, K=3):
    return (torch.zeros(B, 2), torch.zeros(B, K), torch.full((K,), math.log(0.12)),
            torch.full((B,), V.WAYPOINT, dtype=torch.long),
            torch.full((B,), V.n_arg_of(V.WAYPOINT), dtype=torch.long))


def test_the_mixture_density_is_finite_on_draws_the_gaussian_calls_impossible():
    torch.manual_seed(0)
    lg, mu, lsd, acts, na = _args()
    a = (torch.rand_like(mu) * 2 - 1).clamp(-0.999, 0.999)
    u = torch.atanh(a)                                  # a uniform ACTION draw
    gauss = log_prob(lg, mu, lsd, acts, u, na, type_is_action=False)
    mix = log_prob(lg, mu, lsd, acts, u, na, type_is_action=False, explore_eps=0.3)
    assert torch.isfinite(mix).all()
    assert float((mix - gauss).mean()) > 20.0, (
        "the Gaussian assigns ~e^-80 to a uniform draw; without the mixture its "
        "importance ratio is meaningless")


def test_it_barely_changes_draws_near_the_mean():
    torch.manual_seed(0)
    lg, mu, lsd, acts, na = _args()
    u = mu + lsd.exp() * torch.randn_like(mu)
    gauss = log_prob(lg, mu, lsd, acts, u, na, type_is_action=False)
    mix = log_prob(lg, mu, lsd, acts, u, na, type_is_action=False, explore_eps=0.3)
    d = float((mix - gauss).mean())
    assert abs(d - math.log(0.7)) < 0.05, f"expected ~log(1-eps) = {math.log(0.7):.3f}, got {d:.3f}"


def test_the_uniform_piece_is_a_proper_density():
    """Normalisable is the whole reason this works over the bounded action and
    not over the unbounded pre-squash variable.

    p_u(u) = (1/2)^K * prod(1 - tanh^2(u_i)), so in one dimension
    integral 0.5*sech^2(u) du = 0.5*[tanh u] = 1 over the whole line. Checked by
    quadrature, per dimension -- the product then integrates to 1 in any K.
    (An earlier version of this test importance-weighted against the density in
    a-space while sampling in u-space, which is not the same integral.)
    """
    u = torch.linspace(-8.0, 8.0, 200001, dtype=torch.float64)
    p = 0.5 * (1.0 - torch.tanh(u) ** 2)
    integral = float(torch.trapz(p, u))
    assert abs(integral - 1.0) < 1e-6, integral


def test_exploration_is_off_by_default():
    src = open("scripts/composer/cotrain_v11.py").read()
    # `__import__("os")`, not `_os`: that name is bound far below this line and
    # reading it early is the launch-time crash test_trainer_script_lint.py
    # guards -- which it caught when this knob was first added.
    assert 'EXPLORE_EPS = float(__import__("os").environ.get("LES_EXPLORE", "0.0"))' in src
    assert "explore_eps=EXPLORE_EPS" in src, "must reach the update, or the ratio is wrong"


def test_the_widening_is_gone():
    """`s * (1 + eps)` widened sigma at sampling time while the recorded
    `log_std` never included the widening -- an explored row's ratio was scored
    at the wrong sigma. The mixture replaces it and is scored exactly."""
    src = open("src/lagrangian_es/composer/policy_cont.py").read()
    assert "s * (1.0 + eps)" not in src
    assert 'getattr(self, "explore_eps", 0.0)' in src
