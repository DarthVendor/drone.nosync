"""PPO's importance ratio must be 1 for an unchanged policy -- at any temperature.

`choose` draws actions with `log_std.exp() * temperature`. The records used to
store the BARE `log_std`, and `ppo_update_cont` evaluated the new policy at the
bare value too, so both densities described a distribution the actions were
never drawn from.

MEASURED consequence at temp 8 (sampled std 0.96, assumed 0.12): with var
0.0144 and draws ~1.0 from the mean, d(logp)/d(mu) = (u-mu)/var ~ 69, so a mu
shift of 0.1 moves the log-ratio ~7 and the ratio ~1000. A live run hit
kl 100.97 against a 0.02 cap, clipfrac 0.97, policy loss 3e5 -- and that starved
the critic, because clip_grad_norm_(1.0) rescales the whole gradient: EV sat at
-0.006, so the state baseline was inert.
"""
import math

import torch

from lagrangian_es.composer.actions_cont import ContVocab, log_prob as cont_log_prob

V = ContVocab(1)


def test_an_unchanged_policy_has_ratio_one_at_every_temperature():
    torch.manual_seed(0)
    B, K = 64, 3
    logits = torch.randn(B, 2)
    mu = torch.randn(B, K)
    base = torch.full((K,), math.log(0.12))
    acts = torch.full((B,), V.WAYPOINT, dtype=torch.long)
    nargs = torch.full((B,), V.n_arg_of(V.WAYPOINT), dtype=torch.long)
    for temp in (1.0, 2.0, 8.0):
        lsd = base + math.log(temp)
        u = mu + lsd.exp() * torch.randn(B, K)          # drawn at the SAMPLING width
        old = cont_log_prob(logits, mu, lsd, acts, u, nargs)
        new = cont_log_prob(logits, mu, lsd, acts, u, nargs)
        r = (new - old).exp()
        assert torch.allclose(r, torch.ones_like(r), atol=1e-5), f"temp {temp}: {r.mean():.4f}"


def test_a_width_mismatch_makes_the_ratio_explode():
    """The bug, reproduced: score draws taken at temp 8 against the bare width
    and a small mean shift blows the ratio up by orders of magnitude."""
    torch.manual_seed(0)
    B, K, temp = 512, 3, 8.0
    logits = torch.randn(B, 2)
    mu = torch.zeros(B, K)
    base = torch.full((K,), math.log(0.12))
    sampling = base + math.log(temp)
    u = mu + sampling.exp() * torch.randn(B, K)
    acts = torch.full((B,), V.WAYPOINT, dtype=torch.long)
    nargs = torch.full((B,), V.n_arg_of(V.WAYPOINT), dtype=torch.long)
    shifted = mu + 0.1                                   # a small policy step

    wrong = (cont_log_prob(logits, shifted, base, acts, u, nargs)
             - cont_log_prob(logits, mu, base, acts, u, nargs))
    right = (cont_log_prob(logits, shifted, sampling, acts, u, nargs)
             - cont_log_prob(logits, mu, sampling, acts, u, nargs))
    assert float(wrong.abs().mean()) > 20 * float(right.abs().mean()), \
        f"the mismatch must dominate ({float(wrong.abs().mean()):.2f} vs {float(right.abs().mean()):.2f})"
    assert float(right.abs().mean()) < 1.0, "at the true width a 0.1 step is a small log-ratio"


def test_the_update_scores_both_sides_at_the_sampling_width():
    import inspect

    from lagrangian_es.composer.policy_cont import ContComposer, ppo_update_cont
    src = inspect.getsource(ppo_update_cont)
    assert "lsd_now = net.log_std + math.log" in src, "new policy at the sampling width"
    assert "cont_log_prob(logits, mu, lsd_now" in src
    rec = inspect.getsource(ContComposer.emit)
    assert 'getattr(self, "temperature", 1.0)' in rec.split('"log_std"')[1][:400], \
        "records must store the width the action was drawn at"


def test_the_trust_region_is_enforced_before_the_step_not_after():
    """A cap that is measured but not enforced is not a cap.

    It used to be checked on the RUNNING MEAN across minibatches and broke only
    once that mean was already over. MEASURED with target_kl 0.02: reported kl
    0.14-0.40 (7-20x the cap) while arrival fell 0.383 -> 0.273 over five
    iterations. At sigma 0.12 the log-density sensitivity is (u-mu)/sigma^2 ~ 69,
    so one std of mean shift moves the log-ratio by ~1 and holding KL at 0.02
    needs ~0.002 sigma steps.
    """
    import inspect

    from lagrangian_es.composer.policy_cont import ppo_update_cont
    src = inspect.getsource(ppo_update_cont)
    i_check = src.index("kl_mb = float(")
    i_step = src.index("(loss + vcoef * v_loss - ent * h).backward()")
    assert i_check < i_step, "the KL must be measured before the policy step"
    assert "if target_kl > 0 and kl_mb > target_kl:" in src
    assert "(vcoef * v_loss).backward()" in src, \
        "the critic must keep training even when the policy step is skipped"


def test_the_cap_bounds_this_updates_drift_not_the_stale_offset():
    """The trainer pipelines: update k-1 runs while rollout k flies, so records
    are one update stale and the policy already sits at KL ~0.13 from them
    before anything changes. That staleness is what the importance ratio is
    FOR -- but `target_kl` is a step-size control, and capping the total spent
    the budget before the first step: measured, loss 0.000 and clipfrac 0.00
    every iteration, the policy never moving at all.
    """
    import inspect

    from lagrangian_es.composer.policy_cont import ppo_update_cont
    src = inspect.getsource(ppo_update_cont)
    assert "ratio = (lp - old_lp[idx])" in src, \
        "the importance ratio must reference the BEHAVIOUR policy, for correctness"
    assert "lp_start[idx]" in src and "_lr = (lp - lp_start[idx])" in src, \
        "the cap must bound drift from the START of this update, for step size"
    i0 = src.index("lp_start = torch.cat")
    i1 = src.index("for _ in range(epochs)")
    assert i0 < i1, "lp_start must be captured before any step is taken"


def test_the_trust_region_is_symmetric():
    """`mean(lp_start - lp)` is a signed log-ratio, not a KL. A policy moving to
    RAISE the likelihood of its sampled actions drove it NEGATIVE, so the cap
    never fired in that direction.

    MEASURED: iterations reporting -0.19, -0.21, -0.15 against a 0.02 cap with
    clipfrac 0.50-0.60, and arrival falling 0.351 -> 0.277 across exactly those
    steps. Schulman's k3, mean((r-1) - log r), is unbiased and non-negative.
    """
    torch.manual_seed(0)
    lp0 = torch.randn(4096)
    for delta in (-0.5, -0.1, 0.0, 0.1, 0.5):
        lp = lp0 + delta
        lr = (lp - lp0).clamp(-20.0, 20.0)
        k3 = float((lr.exp() - 1.0 - lr).mean())
        signed = float((lp0 - lp).mean())
        assert k3 >= -1e-9, f"k3 must be non-negative, got {k3}"
        if delta < 0:
            assert signed > 0 and k3 > 0
        if delta > 0:
            assert signed < 0 < k3, \
                f"the signed mean hides a real move (signed {signed:+.3f}, k3 {k3:.4f})"


def test_the_update_uses_the_k3_estimator():
    import inspect

    from lagrangian_es.composer.policy_cont import ppo_update_cont
    src = inspect.getsource(ppo_update_cont)
    assert "_lr.exp() - 1.0 - _lr" in src, "k3 estimator"
    assert "kl_mb = float((lp_start[idx] - lp).mean())" not in src, \
        "the signed mean must not be the cap"


def test_there_is_exactly_one_stopping_condition():
    """The cap was enforced TWICE -- a per-minibatch check before the step and a
    running-mean check after it -- and the second bit first.

    k3 is non-negative and grows as the policy drifts, so `acc['kl']/acc['nb']`
    crossed `target_kl` after a couple of minibatches and killed every update.
    MEASURED over 90 iterations (~2700 possible steps): |mu| moved 0.0097 ->
    0.0277, a bearing change of 1.8 -> 3.1 degrees, std 0.120 -> 0.118. The
    policy never moved, so the apparent arrival trend (t +4.43 then t -4.22) was
    a fit to task-draw noise.

    A running mean bounds the AVERAGE of steps already taken, so it tightens
    without limit as the update proceeds; only the per-step check bounds a step.
    """
    import inspect

    from lagrangian_es.composer.policy_cont import ppo_update_cont
    src = inspect.getsource(ppo_update_cont)
    assert src.count("stop = True") == 1, "exactly one stopping condition"
    assert 'acc["kl"] / acc["nb"] > target_kl' not in src, "no running-mean stop"
    assert "if target_kl > 0 and kl_mb > target_kl:" in src, "the per-step check remains"
    assert '"nb": acc["nb"]' in src, "the step count must be observable"
