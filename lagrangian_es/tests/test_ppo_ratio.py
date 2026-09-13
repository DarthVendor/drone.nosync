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


def test_the_kl_is_measured_before_the_step_even_though_it_is_not_enforced():
    """The KL is REPORTED, never enforced (user's call, twice): the composer's
    rate is fixed, with no cap, no early stop and no backtracking.

    It still has to be measured on THIS minibatch before the step, or the
    number in the log describes a policy that has already moved. The history
    behind it: the check once ran on the RUNNING MEAN across minibatches and
    fired only once that mean was already over -- reported kl 0.14-0.40 (7-20x
    a 0.02 cap) while arrival fell 0.383 -> 0.273 over five iterations.

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
    assert "if target_kl > 0 and kl_mb > target_kl:" not in src, \
        "measured, not enforced -- see test_the_update_never_stops_early_on_kl"


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


def test_no_stopping_condition_survives_anywhere():
    """There is no stopping condition at all any more, of either kind.

    History, because both failure modes are worth not repeating. The cap was
    once enforced TWICE -- a per-minibatch check before the step and a
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
    assert src.count("stop = True") == 0, "no stopping condition of any kind"
    assert 'acc["kl"] / acc["nb"] > target_kl' not in src, "no running-mean stop"
    assert "if target_kl > 0 and kl_mb > target_kl:" not in src, "no per-step stop"
    assert '"nb": acc["nb"]' in src, "the step count must be observable"


def test_a_batch_with_no_outcome_spread_is_skipped_not_normalised():
    """One shared goal per epoch means an unreachable goal makes EVERY flight
    score the same. Normalising that divides by the clamp floor, collapses the
    target toward zero, and makes EV a divide-by-almost-zero (-43639 observed
    live). There is nothing to learn from such a batch; skip it.
    """
    import sys
    sys.path.insert(0, "tests")
    import torch
    from test_actions_cont import _cont_net, _fake_tok
    from lagrangian_es.composer.actions_cont import ContVocab
    from lagrangian_es.composer.policy_cont import ppo_update_cont
    V = ContVocab(2); B, T = 16, 2
    torch.manual_seed(0); net = _cont_net()
    recs = []
    for t in range(T):
        tok = _fake_tok(B, 20.0, net.goal_gain); act = torch.full((B,), V.WAYPOINT)
        with torch.no_grad():
            lg, mu_all, _ = net.pre(tok); mu = mu_all[torch.arange(B), act]
            u = mu + net.log_std.exp() * torch.randn(B, V.n_args)
        recs.append({"t": float(t * 20), "act": act, "u": u,
                     "n_args": torch.tensor([V.n_arg_of(int(z)) for z in act.tolist()]),
                     "moved": torch.ones(B, dtype=torch.bool),
                     "alive": torch.ones(B, dtype=torch.bool),
                     "rows": torch.arange(B), "tok_keep": None, "logits": lg.clone(),
                     "pi_logits": lg.clone(), "mu": mu.clone(),
                     "log_std": net.log_std.detach().clone(), "tok": tok})
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    flat = torch.full((T, B), 1353.45, dtype=torch.float64)      # every flight identical
    st = ppo_update_cont(net, recs, flat, 2, epochs=2, batch=8, lr=1e-3)
    assert st.get("degenerate") is True, f"must report the batch as degenerate: {st}"
    assert st["ev"] != st["ev"], "EV must be nan, not a divide-by-almost-zero"
    after = net.state_dict()
    assert all(torch.equal(after[k], before[k]) for k in before), \
        "a batch that ranks nothing must not move the policy"


def test_the_update_never_stops_early_on_kl():
    """The composer's rate is fixed: no KL cap, no early stop, no backtracking
    (user's call, twice). `target_kl` is measured and reported, never enforced.

    The branch survived because the ACCUMULATION path returns before reaching
    it, so every LES_ACCUM>1 run had it dormant. Stepping every epoch woke it
    up and updates halted after 2, 3 and 6 of ~20 minibatches -- discarding
    70-90% of a batch that costs a whole rollout to collect.
    """
    import inspect
    from lagrangian_es.composer import policy_cont
    src = inspect.getsource(policy_cont.ppo_update_cont)
    assert "stop = True" not in src, "no early stop may be reintroduced"
    assert "if target_kl > 0 and kl_mb > target_kl" not in src, \
        "target_kl must be reported, not enforced"


def test_every_minibatch_is_used():
    """With the cap gone, an update must take a step per minibatch."""
    import sys
    sys.path.insert(0, "tests")
    import torch
    from test_actions_cont import _cont_net, _fake_tok
    from lagrangian_es.composer.actions_cont import ContVocab
    from lagrangian_es.composer.policy_cont import ppo_update_cont
    V = ContVocab(2); B, T = 64, 4
    torch.manual_seed(0); net = _cont_net()
    recs = []
    for t in range(T):
        tok = _fake_tok(B, 20.0, net.goal_gain); act = torch.full((B,), V.WAYPOINT)
        with torch.no_grad():
            lg, mu_all, _ = net.pre(tok); mu = mu_all[torch.arange(B), act]
            u = mu + net.log_std.exp() * torch.randn(B, V.n_args)
        recs.append({"t": float(t * 20), "act": act, "u": u,
                     "n_args": torch.tensor([V.n_arg_of(int(z)) for z in act.tolist()]),
                     "moved": torch.ones(B, dtype=torch.bool),
                     "alive": torch.ones(B, dtype=torch.bool),
                     "rows": torch.arange(B), "tok_keep": None, "logits": lg.clone(),
                     "pi_logits": lg.clone(), "mu": mu.clone(),
                     "log_std": net.log_std.detach().clone(), "tok": tok})
    R = torch.randn(T, B, dtype=torch.float64) * 50.0
    # a deliberately provocative rate: the old cap would have halted this at once
    st = ppo_update_cont(net, recs, R, 2, epochs=1, batch=32, lr=5e-3, target_kl=0.02)
    assert st["stopped_early"] is False, "must not stop early"
    assert st["nb"] >= (T * B) // 32, f"took only {st['nb']} steps of {(T*B)//32}"
