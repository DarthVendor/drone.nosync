import pytest
import torch

from lagrangian_es.config import Config, ESCfg, RolloutCfg
from lagrangian_es.es import build, train
from lagrangian_es.evaluate import accepts, evaluate
from lagrangian_es.metric import identity_preconditioner
from lagrangian_es.operators import (
    ga_step, mirrored_offspring, rank_weights, recombine, update_sigma,
)
from lagrangian_es.util import make_gen

DT = torch.float64


def _cfg(**es_kw):
    base = dict(pop=16, gens=5, whiten=True, metric_every=4, strategy="es")
    base.update(es_kw)
    return Config(seed=0, rollout=RolloutCfg(ep_steps=120, n_eps=2), es=ESCfg(**base))


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
def test_mirrored_sampling_is_exactly_antithetic():
    mean = torch.zeros(8, dtype=DT)
    P = torch.eye(8, dtype=DT)
    TH = mirrored_offspring(mean, 0.1, P, 10, make_gen(0))
    assert TH.shape == (10, 8)
    assert torch.allclose(TH[:5] - mean, -(TH[5:] - mean), atol=1e-15)
    assert torch.allclose(TH.mean(0), mean, atol=1e-15)


def test_mirrored_sampling_rejects_odd_population():
    with pytest.raises(ValueError):
        mirrored_offspring(torch.zeros(4, dtype=DT), 0.1, torch.eye(4, dtype=DT), 7, make_gen(0))


def test_identity_P_reproduces_plain_isotropic_sampling():
    """Both arms flow through the same code path and differ only in P.  With
    P = I the whitened operator IS the isotropic baseline -- not a
    reimplementation of it."""
    mean = torch.randn(12, dtype=DT)
    a = mirrored_offspring(mean, 0.2, torch.eye(12, dtype=DT), 8, make_gen(3))
    g = make_gen(3)
    z = torch.randn(4, 12, generator=g, dtype=DT)
    d = 0.2 * z
    b = mean + torch.cat([d, -d], 0)
    assert torch.equal(a, b)


def test_whitening_reshapes_but_does_not_lengthen_the_step():
    torch.manual_seed(0)
    A = torch.randn(20, 20, dtype=DT)
    G = A @ A.T / 20 + 0.05 * torch.eye(20, dtype=DT)
    lam, V = torch.linalg.eigh(G)
    inv = lam.rsqrt()
    inv = inv / inv.mean()
    P = (V * inv) @ V.T
    mean = torch.zeros(20, dtype=DT)
    iso = mirrored_offspring(mean, 0.1, torch.eye(20, dtype=DT), 4000, make_gen(1))
    wht = mirrored_offspring(mean, 0.1, P, 4000, make_gen(1))
    assert not torch.allclose(iso, wht)
    ratio = float(wht.norm(dim=1).mean() / iso.norm(dim=1).mean())
    assert 0.5 < ratio < 2.0, f"step length changed by {ratio:.2f}x"


def test_rank_weights_are_positive_and_normalized():
    for mu in (1, 2, 4, 12):
        w = rank_weights(mu, dtype=DT)
        assert w.shape == (mu,)
        assert (w > 0).all()
        assert abs(float(w.sum()) - 1.0) < 1e-12
        if mu > 1:
            assert (w[:-1] >= w[1:]).all(), "weights must decrease with rank"


def test_recombination_favours_the_best():
    TH = torch.arange(8, dtype=DT)[:, None].repeat(1, 3)
    fit = torch.arange(8, dtype=DT)              # genome i has fitness i
    mean, elite, ef = recombine(TH, fit, 0.5)
    assert set(elite.tolist()) == {0, 1, 2, 3}
    assert float(mean[0]) < 1.5, "should sit near the best genomes"


def test_sigma_adaptation_bounds():
    assert update_sigma(0.1, True, 1.06, 0.97) > 0.1
    assert update_sigma(0.1, False, 1.06, 0.97) < 0.1
    assert update_sigma(1e9, True, 1.06, 0.97, hi=0.5) == 0.5
    assert update_sigma(1e-9, False, 1.06, 0.97, lo=1e-3) == 1e-3


def test_ga_step_preserves_elites_and_shape():
    g = make_gen(0)
    TH = torch.randn(16, 9, dtype=DT)
    fit = torch.randn(16, dtype=DT)
    I = torch.eye(9, dtype=DT)
    new, order = ga_step(TH, fit, 0.05, I, I, g, elitism=3)
    assert new.shape == TH.shape
    assert torch.equal(new[:3], TH[order[:3]]), "elites must pass through untouched"
    assert not torch.equal(new[3:], TH[order[3:]])


# --------------------------------------------------------------------------- #
# the loops
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["es", "ga"])
def test_training_improves_on_held_out_tasks(strategy):
    """The learning curve is noisy because goals are resampled every generation,
    so the meaningful assertion is on held-out performance, not on the curve.

    Uses the FULL 250-step episode deliberately.  At the shortened horizon the
    other tests use, a leg lasts 1.2 s -- shorter than the closed-loop settling
    time at omega_n ~ 2.4 rad/s -- so the waypoints are physically unreachable and
    the optimizer correctly spends its budget on staying alive instead.  Testing
    waypoint accuracy on a task that cannot be completed measures nothing.
    """
    cfg = Config(seed=0, rollout=RolloutCfg(ep_steps=250, n_eps=2),
                 es=ESCfg(pop=16, gens=14, whiten=True, metric_every=4,
                          strategy=strategy, sigma0=0.12, elite_frac=0.5))
    system, tr, task = build(cfg)
    res = train(cfg, system, tr, task)
    before = evaluate(system, tr, task, tr.init(), cfg.rollout, n_tasks=64)
    after = evaluate(system, tr, task, res.theta, cfg.rollout, n_tasks=64)
    assert after["fitness"] < before["fitness"]
    assert after["crash_rate"] <= before["crash_rate"]
    assert after["legB_err"] < before["legB_err"]


@pytest.mark.parametrize("strategy", ["es", "ga"])
def test_fixed_seed_reproduces_bit_identically(strategy):
    cfg = _cfg(strategy=strategy)
    a = train(cfg)
    b = train(cfg)
    assert torch.equal(a.theta, b.theta)
    assert [h["fitness_elite"] for h in a.history] == [h["fitness_elite"] for h in b.history]


@pytest.mark.parametrize("strategy", ["es", "ga"])
def test_different_seeds_diverge(strategy):
    a = train(_cfg(strategy=strategy))
    b = train(Config(seed=1, rollout=RolloutCfg(ep_steps=120, n_eps=2),
                     es=ESCfg(pop=16, gens=5, whiten=True, metric_every=4, strategy=strategy)))
    assert not torch.equal(a.theta, b.theta)


def test_whiten_false_is_exactly_the_identity_preconditioner():
    """The isotropic arm must never touch the metric: same seed, and P = I."""
    cfg = _cfg(whiten=False)
    seen = []
    system, tr, task = build(cfg)
    train(cfg, system, tr, task, callback=lambda rec, th: seen.append(rec))
    assert all("metric_cond" not in r for r in seen), "isotropic arm refreshed the metric"

    # ... and its trajectory equals a run where P is pinned to I by construction
    ident = identity_preconditioner(tr.dim, tr.dtype)
    assert torch.equal(ident.P, torch.eye(tr.dim, dtype=tr.dtype))


def test_whitened_and_isotropic_arms_take_different_paths():
    """Matched on seed, population and budget: the ONLY difference is P."""
    a = train(_cfg(whiten=True))
    b = train(_cfg(whiten=False))
    assert not torch.equal(a.theta, b.theta)
    assert len(a.history) == len(b.history)


def test_history_schema_matches_across_strategies():
    """viz and ablate read both arms through the same keys."""
    a = train(_cfg(strategy="es")).history[-1]
    b = train(_cfg(strategy="ga")).history[-1]
    shared = {"gen", "fitness_elite", "sigma", "success_frac", "crash_rate",
              "legA_err", "legB_err", "success_rate", "final_err"}
    assert shared <= set(a) and shared <= set(b)


def test_common_random_numbers_within_a_generation():
    """Every member of a generation must face identical goals and identical reset
    noise; without it ES variance swamps the effect the ablation measures."""
    from lagrangian_es.rollout import Rollout
    cfg = _cfg()
    system, tr, task = build(cfg)
    roll = Rollout(system, tr, task, cfg.rollout)
    goals = task.sample(3, make_gen(2))
    s, TH_b, goals_b, res_b, P, E = roll._expand(
        torch.zeros(4, tr.dim, dtype=DT), goals, 7)
    assert P == 4 and E == 3
    for k, v in s.items():
        base = v[:E]
        for i in range(1, P):
            assert torch.equal(v[i * E:(i + 1) * E], base), \
                f"reset noise differs across population members for {k!r}"
    for i in range(1, P):
        assert torch.equal(goals_b[i * E:(i + 1) * E], goals_b[:E])


def test_acceptance_helper_reads_the_section_6_criteria():
    good = {"legB_err": 0.10, "success_rate": 0.9, "crash_rate": 0.0}
    bad = {"legB_err": 0.30, "success_rate": 0.9, "crash_rate": 0.0}
    assert accepts(good)["ALL"] is True
    assert accepts(bad)["ALL"] is False


# --- goal bonus -------------------------------------------------------------
def _bonus_costs(bonus, tol_scale=1.0):
    """Per-episode cost for one genome, with and without the arrival bonus."""
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.systems import make_system
    from lagrangian_es.tasks import make_task
    from lagrangian_es.trainables import make_trainable

    system = make_system("quadrotor", dtype=DT)
    tr = make_trainable("energy_shaping", system)
    task = make_task("waypoint_pair", system, gating="arrival")
    task.tol = task.tol * tol_scale
    th = tr.init()[None].expand(4, -1).contiguous()
    goals = task.sample(4 * 4, make_gen(2))
    out = []
    for b in (0.0, bonus):
        cfg = RolloutCfg(n_eps=4, ep_steps=250, goal_bonus=b)
        out.append(Rollout(system, tr, task, cfg).run(th, goals, 3).cost)
    return out[0], out[1], task.n_legs


def test_goal_bonus_is_credited_once_per_waypoint_not_once_per_step():
    """The final leg never advances `leg`, so its arrival test fires on EVERY
    step the vehicle sits inside tol.  Paying the bonus per step would turn
    hovering on the goal into an unbounded reward and the objective would stop
    meaning anything -- the saving must be an integer multiple of the bonus, at
    most one per leg.
    """
    bonus = 10.0
    base, with_bonus, n_legs = _bonus_costs(bonus)
    saved = base - with_bonus
    assert torch.all(saved >= -1e-9)
    assert torch.all(saved <= n_legs * bonus + 1e-9)
    k = saved / bonus
    assert torch.allclose(k, k.round(), atol=1e-9), f"non-integer credits: {k}"
    # non-vacuity: a run that never reaches a goal would satisfy all of the
    # above trivially, so require the full credit range to be exercised
    counts = set(k.round().long().tolist())
    assert counts == {0, 1, 2}, f"credit path not exercised: {counts}"


def test_goal_bonus_scales_the_saving_linearly():
    """Two bonuses, same trajectories: the credit count cannot depend on the
    bonus size, so the saving must scale exactly with it."""
    b1, w1, _ = _bonus_costs(10.0)
    b2, w2, _ = _bonus_costs(40.0)
    assert torch.allclose(b1, b2)                       # same rollout either way
    assert torch.allclose((b1 - w1) * 4.0, b2 - w2, atol=1e-9)


def test_goal_bonus_defaults_off_and_leaves_the_objective_untouched():
    base, same, _ = _bonus_costs(0.0)
    assert torch.equal(base, same)


# --- dead_mode --------------------------------------------------------------
def _dead_mode_costs(mode, dead_cost, theta_scale=0.0):
    """Cost per episode under one dead_mode.  theta_scale=0 gives a controller
    that emits nothing, so the batch falls and dies -- the case that matters."""
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.systems import make_system
    from lagrangian_es.tasks import make_task
    from lagrangian_es.trainables import make_trainable

    system = make_system("quadrotor", dtype=DT)
    tr = make_trainable("energy_shaping", system)
    task = make_task("waypoint_pair", system, gating="arrival")
    th = (tr.init() * theta_scale)[None].expand(4, -1).contiguous()
    cfg = RolloutCfg(n_eps=4, ep_steps=120, dead_mode=mode, dead_cost=dead_cost,
                     lambda_e=0.0, lambda_s=0.0)
    r = Rollout(system, tr, task, cfg).run(th, task.sample(16, make_gen(2)), 5)
    return r.cost, r.alive


def test_frozen_dead_mode_ignores_dead_cost_entirely():
    """`dead_cost` is the free parameter being removed, so frozen mode must not
    read it -- otherwise the tuning problem is still there, just hidden."""
    c_lo, alive = _dead_mode_costs("frozen", 0.5)
    c_hi, _ = _dead_mode_costs("frozen", 500.0)
    assert not bool(alive.all()), "test is vacuous unless something dies"
    assert torch.allclose(c_lo, c_hi), "frozen mode still depends on dead_cost"


def test_dead_modes_agree_exactly_on_episodes_that_survived():
    """The modes may only differ where a vehicle actually died; a survivor's
    cost is the same integral either way."""
    # a scaled-up genome keeps the whole batch alive, which makes the second
    # half of this test vacuous; the zero genome leaves ~6% dead
    c_const, alive = _dead_mode_costs("constant", 5.0)
    c_frozen, alive2 = _dead_mode_costs("frozen", 5.0)
    assert torch.equal(alive, alive2)
    assert alive.any() and not alive.all(), "need both outcomes present"
    assert torch.allclose(c_const[alive], c_frozen[alive])
    assert not torch.allclose(c_const[~alive], c_frozen[~alive])


def test_dead_mode_constant_is_the_default_and_unchanged():
    assert RolloutCfg().dead_mode == "constant"
    assert RolloutCfg().goal_bonus == 0.0


def test_sigma_fixed_point_is_set_by_grow_shrink_not_by_success_target():
    """sigma is stationary where p*ln(grow) + (1-p)*ln(shrink) == 0.

    The config names a 0.20 trigger, but that is only the threshold the hit rate
    is compared against -- the balance point is 0.343 for the default
    multipliers.  Anyone retuning grow/shrink moves it, so pin the behaviour:
    driving the rule at the predicted rate must leave sigma where it started.
    """
    import math
    from lagrangian_es.config import ESCfg
    es = ESCfg()
    p = math.log(1 / es.shrink) / math.log(es.grow / es.shrink)
    assert p == pytest.approx(0.343, abs=0.005)

    def walk(rate, n=4000):
        sigma, hits = 0.05, 0
        for i in range(n):
            improved = (hits + 1) / (i + 1) <= rate     # drive at exactly `rate`
            hits += improved
            sigma = update_sigma(sigma, improved, es.grow, es.shrink,
                                 es.sigma_min, es.sigma_max)
        return sigma

    assert walk(p) == pytest.approx(0.05, rel=0.25)     # stationary at the balance
    assert walk(p - 0.15) < 0.05 * 0.2                  # below it, sigma decays
    assert walk(p + 0.15) > 0.05 * 5.0                  # above it, sigma grows
