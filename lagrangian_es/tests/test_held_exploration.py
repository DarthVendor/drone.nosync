"""Held exploration: a mixture whose SECOND component is a held Gaussian.

Escapes are already sampled -- 12.6% of decisions point past 90 degrees at
eps 0.25 -- and never reinforced, because clearing a U-shaped pocket needs a
consistent RUN of detours and i.i.d. draws give a run of k with probability
eps^k. Holding the exploration CENTRE for `noise_hold` decisions makes a run
cost eps instead.

The density must stay a density, or PPO's ratio is silently wrong -- which is
the failure mode this project has paid for repeatedly.
"""
import math
import torch
from lagrangian_es.composer.actions_cont import ContVocab, log_prob


def _pieces(k=1, eps=0.3, sd=0.5):
    V = ContVocab(2)
    mu = torch.zeros(1, V.n_args)
    ex = torch.zeros(1, V.n_args); ex[0, 0] = 1.4          # a HELD centre, off mu
    ls = torch.full((V.n_args,), math.log(sd))
    tokl = torch.zeros(1, V.V)
    na = torch.tensor([k])
    return V, mu, ex, ls, tokl, na


def test_held_mixture_is_a_density():
    """Numerically integrate it. A mixture that does not integrate to 1 makes
    every importance ratio wrong by a factor that no test of the mean catches."""
    V, mu, ex, ls, tokl, na = _pieces(k=1, eps=0.3, sd=0.5)
    grid = torch.linspace(-14.0, 14.0, 60001)
    dx = float(grid[1] - grid[0])
    u = torch.zeros(grid.shape[0], V.n_args); u[:, 0] = grid
    lp = log_prob(tokl.expand(grid.shape[0], -1), mu.expand(grid.shape[0], -1), ls,
                  torch.zeros(grid.shape[0], dtype=torch.long), u,
                  na.expand(grid.shape[0]), type_is_action=False,
                  explore_eps=0.3, explore_mu=ex.expand(grid.shape[0], -1))
    mass = float(lp.exp().sum() * dx)
    assert abs(mass - 1.0) < 2e-3, f"held mixture integrates to {mass}, not 1"


def test_it_is_the_mixture_it_claims_to_be():
    """(1-eps)N(mu,sd) + eps*N(explore_mu,sd), exactly, at a few points."""
    V, mu, ex, ls, tokl, na = _pieces(k=1, eps=0.3, sd=0.5)
    for x in (-1.0, 0.0, 0.7, 1.4, 3.0):
        u = torch.zeros(1, V.n_args); u[0, 0] = x
        got = float(log_prob(tokl, mu, ls, torch.zeros(1, dtype=torch.long), u, na,
                             type_is_action=False, explore_eps=0.3,
                             explore_mu=ex).exp())
        n = lambda c: math.exp(-0.5 * ((x - c) / 0.5) ** 2) / (0.5 * math.sqrt(2 * math.pi))
        want = 0.7 * n(0.0) + 0.3 * n(1.4)
        assert abs(got - want) < 1e-5, f"at {x}: {got} vs {want}"


def test_zero_eps_is_the_plain_gaussian():
    """The held path must vanish when exploration is off, or every muted
    control and every judge flight silently changes."""
    V, mu, ex, ls, tokl, na = _pieces()
    u = torch.zeros(1, V.n_args); u[0, 0] = 0.6
    a = log_prob(tokl, mu, ls, torch.zeros(1, dtype=torch.long), u, na,
                 type_is_action=False, explore_eps=0.0, explore_mu=ex)
    b = log_prob(tokl, mu, ls, torch.zeros(1, dtype=torch.long), u, na,
                 type_is_action=False, explore_eps=0.0)
    assert torch.equal(a, b)


def test_an_escape_still_carries_gradient():
    """The whole reason for a held MEAN rather than a held ACTION: a sample
    drawn from the exploration component must still push mu_theta."""
    V, mu, ex, ls, tokl, na = _pieces(k=1, eps=0.25, sd=0.12)
    mu = mu.clone().requires_grad_(True)
    u = torch.zeros(1, V.n_args); u[0, 0] = 1.4          # drawn at the held centre
    lp = log_prob(tokl, mu, ls, torch.zeros(1, dtype=torch.long), u, na,
                  type_is_action=False, explore_eps=0.25, explore_mu=ex)
    lp.backward()
    assert mu.grad is not None and float(mu.grad.abs().sum()) > 0.0, \
        "a held ACTION would give exactly zero here; a held MEAN must not"


def test_noise_hold_actually_reaches_the_composer():
    """The knob must BE somewhere, not merely be accepted.

    `noise_hold` was a documented config key for the whole life of this project
    and `grep -rn noise_hold src/` returned ZERO hits: the trainer passed it,
    `composer_kw` carried it, and nothing ever read it, so every run that set
    it explored i.i.d. anyway. The first attempt at held exploration measured
    runs of consecutive detours reaching 3+ at 0.014 against an i.i.d.
    prediction of p^2 = 0.0146 -- identical, because `getattr(self,
    "noise_hold", 1)` was falling through to the default.
    """
    import inspect
    from lagrangian_es.composer.policy import PolicyComposer
    from lagrangian_es.composer import policy_cont
    init = inspect.getsource(PolicyComposer.__init__)
    assert 'kw.pop("noise_hold"' in init, \
        "PolicyComposer must consume noise_hold; nothing else setattrs kwargs"
    assert "self.noise_hold" in init, "and must bind it on the instance"
    choose = inspect.getsource(policy_cont.ContComposer)
    assert 'getattr(self, "noise_hold"' in choose, "the sampler must read it"


def test_the_base_class_does_not_setattr_kwargs():
    """Why the test above is needed at all: unknown composer_kw are SILENTLY
    dropped, so a new knob is inert until it is popped explicitly."""
    import inspect
    from lagrangian_es.composer.base import Composer
    src = inspect.getsource(Composer.__init__)
    assert "setattr" not in src and "self.__dict__.update" not in src, \
        "if this ever changes, the explicit pops above can be relaxed"
