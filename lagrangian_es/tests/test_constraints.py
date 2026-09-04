"""Episode-level constraints, their multipliers, and why lambda stays out of theta."""
import pytest
import torch
from torch.func import jacrev, vmap

from lagrangian_es.config import Config, ESCfg, RolloutCfg
from lagrangian_es.constraints import (
    ConstraintSet, CrashBudget, DualAscent, EffortBudget, PIDMultiplier,
    SaturationBudget, ShapingBudget, make_constraint,
)
from lagrangian_es.es import build, train
from lagrangian_es.metric import physics_metric
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen

DT = torch.float64


@pytest.fixture(scope="module")
def rig():
    cfg = Config(rollout=RolloutCfg(ep_steps=120, n_eps=4))
    system, tr, task = build(cfg)
    roll = Rollout(system, tr, task, cfg.rollout)
    goals = task.sample(4, make_gen(3))
    res = roll.run(tr.init()[None].expand(6, -1), goals, seed=1)
    return system, tr, task, cfg, roll, goals, res


# --------------------------------------------------------------------------- #
# THE structural point
# --------------------------------------------------------------------------- #
def test_a_multiplier_inside_theta_would_zero_a_block_of_G(rig):
    """A multiplier affects fitness, not the wrench, so du/dlambda = 0 identically.

    Padding the genome with slots `forward` ignores -- which is exactly what
    putting lambda in theta does -- produces a metric with an exactly-zero block.
    The failure is silent: the ridge keeps `eigh` well posed and nothing errors,
    so the multiplier simply stops being searched in any meaningful way while the
    ridge does all the work in those coordinates.

    This is why `ConstraintSet` is passed alongside the genome rather than
    concatenated into it.
    """
    system, tr, task, cfg, roll, goals, _ = rig
    n_lam = 3

    class PaddedWithMultipliers:
        """A deliberately wrong design, built only so its metric can be measured."""
        def __init__(self, inner):
            self.inner, self.system = inner, inner.system
            self.dim = inner.dim + n_lam

        def forward(self, theta, s, goal):
            return self.inner.forward(theta[..., : self.inner.dim], s, goal)

    padded = PaddedWithMultipliers(tr)
    theta = torch.cat([tr.init(), torch.full((n_lam,), 0.5, dtype=DT)])

    trace = roll.trace(tr.init()[None], goals, 5)
    s = {k: v[20] for k, v in trace.states.items()}
    gl = trace.goals[20]
    J = vmap(jacrev(padded.forward), in_dims=(None, 0, 0))(theta, s, gl)

    assert J[..., : tr.dim].abs().max() > 0, "sanity: real params must move u"
    assert J[..., tr.dim:].abs().max() == 0.0, "du/dlambda must be identically zero"

    G = torch.einsum("sid,si,sie->de", J, system.inv_mass(s), J) / J.shape[0]
    assert G[tr.dim:, :].abs().max() == 0.0
    assert G[:, tr.dim:].abs().max() == 0.0
    lam = torch.linalg.eigvalsh(0.5 * (G + G.T))
    assert int((lam.abs() < 1e-12).sum()) >= n_lam, "a zero block, silently ridged over"


def test_constraints_do_not_change_the_genome_dimension(rig):
    """The primal search space is untouched by adding a constraint."""
    system, tr, task, cfg, roll, goals, _ = rig
    before = tr.dim
    cs = ConstraintSet([CrashBudget(0.0), EffortBudget(2.0)], multiplier="pid")
    assert len(cs) == 2
    assert tr.dim == before
    assert tr.init().shape == (before,)


# --------------------------------------------------------------------------- #
# constraints
# --------------------------------------------------------------------------- #
def test_constraint_values_are_per_genome(rig):
    system, tr, task, cfg, roll, goals, res = rig
    for c in (CrashBudget(0.0), EffortBudget(1.0), SaturationBudget(0.1), ShapingBudget(0.1)):
        v = c.value(res)
        assert v.shape == (6,), f"{c.name} must aggregate to one value per genome"
        assert torch.isfinite(v).all()
        assert torch.allclose(c.violation(res), v - c.budget)


def test_crash_budget_matches_the_rollout(rig):
    system, tr, task, cfg, roll, goals, res = rig
    expect = (~res.alive).to(DT).view(6, res.n_eps).mean(1)
    assert torch.allclose(CrashBudget(0.0).value(res), expect)


def test_augmentation_prices_violations_and_is_identity_at_zero_lambda(rig):
    system, tr, task, cfg, roll, goals, res = rig
    cs = ConstraintSet([EffortBudget(0.0)], multiplier="dual_ascent", eta=0.5)
    assert torch.allclose(cs.augment(res.fitness, res), res.fitness)   # lambda starts at 0
    cs.update(res)
    aug = cs.augment(res.fitness, res)
    assert not torch.allclose(aug, res.fitness)
    lam = cs.multipliers[0].lam
    assert torch.allclose(aug, res.fitness + lam * EffortBudget(0.0).violation(res))


def test_make_constraint_registry():
    assert isinstance(make_constraint("crash", 0.0), CrashBudget)
    with pytest.raises(KeyError):
        make_constraint("nope", 1.0)


# --------------------------------------------------------------------------- #
# multipliers
# --------------------------------------------------------------------------- #
def test_dual_ascent_rises_on_violation_and_stays_nonnegative():
    m = DualAscent(eta=0.1)
    for _ in range(10):
        m.update(0.5)
    assert m.lam > 0
    for _ in range(100):
        m.update(-1.0)
    assert m.lam == 0.0, "multiplier must not go negative"


def test_pid_with_only_the_integral_term_is_exactly_dual_ascent():
    """The classic update IS the integral term, which is what makes the PID
    version a strict generalization rather than a different algorithm."""
    d = DualAscent(eta=0.05)
    p = PIDMultiplier(kp=0.0, ki=0.05, kd=0.0)
    for v in [0.4] * 8 + [-0.2] * 8 + [0.1] * 4:
        d.update(v)
        p.update(v)
        assert abs(d.lam - p.lam) < 1e-12


def test_pid_responds_to_the_violation_now_not_only_to_its_history():
    """The proportional term is the whole point: on the first step of a fresh
    violation, pure dual ascent can only contribute ki*v, while PID also
    contributes kp*v immediately."""
    d = DualAscent(eta=0.05)
    p = PIDMultiplier(kp=0.30, ki=0.05, kd=0.0)
    assert p.update(0.4) > d.update(0.4)


def test_pid_damps_the_primal_dual_overshoot():
    """Closed-loop check in a regime where the oscillation is real.

    The violation responds to the multiplier `lag` steps late and without
    saturation, which is exactly the condition that makes pure dual ascent
    overshoot: it keeps integrating through the dead time in which the population
    has already complied, then has to unwind.
    """
    def simulate(m, steps=600, lag=20, k=0.5, v0=0.5):
        lam, queue = [], [v0] * lag
        for _ in range(steps):
            v = queue.pop(0)
            m.update(v)
            queue.append(v0 - k * m.lam)     # more pricing -> less violation, later
            lam.append(m.lam)
        return lam

    di = simulate(DualAscent(eta=0.08))
    pid = simulate(PIDMultiplier(kp=0.32, ki=0.08, kd=0.16))
    over = lambda h: (max(h) - h[-1]) / max(h[-1], 1e-9)

    # the test would be vacuous if dual ascent did not actually ring here
    assert over(di) > 0.15, f"regime does not exhibit overshoot: {over(di):.3f}"
    assert over(pid) < over(di), (
        f"PID overshoot {over(pid):.3f} should beat dual ascent {over(di):.3f}")
    assert max(pid) < max(di)
    # both must still converge to the same price
    assert abs(pid[-1] - di[-1]) < 0.05 * max(di[-1], 1e-9)


def test_multiplier_is_capped():
    m = DualAscent(eta=10.0, lam_max=2.0)
    for _ in range(50):
        m.update(100.0)
    assert m.lam == 2.0


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["es", "ga"])
def test_training_with_constraints_runs_and_logs_multipliers(strategy):
    cfg = Config(seed=0, rollout=RolloutCfg(ep_steps=150, n_eps=2),
                 es=ESCfg(pop=16, gens=8, strategy=strategy, metric_every=4))
    system, tr, task = build(cfg)
    cs = ConstraintSet([SaturationBudget(0.02)], multiplier="pid")
    res = train(cfg, system, tr, task, constraints=cs)
    assert "lam_saturation" in res.history[-1]
    assert "viol_saturation" in res.history[-1]
    assert res.theta.shape == (tr.dim,), "constraints must not resize the genome"
    assert all(h["lam_saturation"] >= 0 for h in res.history)


def test_a_binding_constraint_actually_changes_the_search():
    """A budget the population violates must alter selection; one it never
    approaches must leave the run bit-identical."""
    cfg = Config(seed=0, rollout=RolloutCfg(ep_steps=150, n_eps=2),
                 es=ESCfg(pop=16, gens=8, strategy="es", metric_every=4))
    system, tr, task = build(cfg)

    free = train(cfg, *build(cfg), constraints=None)
    slack = train(cfg, *build(cfg),
                  constraints=ConstraintSet([EffortBudget(1e9)], multiplier="dual_ascent"))
    tight = train(cfg, *build(cfg),
                  constraints=ConstraintSet([EffortBudget(0.0)], multiplier="dual_ascent",
                                            eta=0.5))
    assert torch.equal(free.theta, slack.theta), "a slack budget must not perturb search"
    assert not torch.equal(free.theta, tight.theta), "a binding budget must bite"
    assert tight.history[-1]["lam_effort"] > 0
