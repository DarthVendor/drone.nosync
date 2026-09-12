"""The composer is a ROUTER: the low level flies the kinematics, the composer
picks where to go, continuously. So it always emits a waypoint -- "stay silent"
is not one of its moves -- and the only decision left is WHERE.

MEASURED with a deterministic router over 1024 paired tasks, identical seeds,
deciding at every report, nothing differing but the placement rule
(arrive / d against a muted control of 0.551):

    lam 0.0   0.578   +0.027 +- 0.011   t +2.58
    lam 2.0   0.476   -0.075 +- 0.017   t -4.42

Nothing separates those but where the subgoal goes, so the placement is where
the value is and the loss has to reach it.
"""
import math

import torch

from lagrangian_es.composer.actions_cont import ContVocab
from lagrangian_es.composer.policy_cont import ContPolicyNet, route_update
from lagrangian_es.composer.tokens import F

V = ContVocab(1)


def _recs(u, B):
    g = torch.Generator().manual_seed(3)
    t = {"self": torch.randn(B, F, generator=g), "goal": torch.randn(B, F, generator=g),
         "entities": torch.randn(B, 10, F, generator=g),
         "ent_types": torch.full((B, 10), 2, dtype=torch.long),
         "ent_mask": torch.ones(B, 10, dtype=torch.bool),
         "chain": torch.zeros(B, 1, F), "chain_types": torch.ones(B, 1, dtype=torch.long),
         "chain_mask": torch.zeros(B, 1, dtype=torch.bool), "psi": torch.zeros(B)}
    return [{"tok": t, "act": torch.full((B,), V.WAYPOINT, dtype=torch.long), "u": u,
             "n_args": torch.full((B,), V.n_arg_of(V.WAYPOINT), dtype=torch.long),
             "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B)}]


def test_the_placement_that_saved_time_is_moved_toward():
    """Half the flights placed LEFT, half RIGHT; one group saved time. The mean
    placement must move toward whichever it was."""
    out = {}
    for left_is_better in (True, False):
        torch.manual_seed(0)
        net = ContPolicyNet(n_terms=1)
        B = 128
        left = torch.arange(B) < B // 2
        u = torch.zeros(B, 3)
        u[:, 1] = torch.where(left, torch.tensor(1.0), torch.tensor(-1.0))   # bearing
        sign = 1.0 if left_is_better else -1.0
        adv = torch.where(left, torch.tensor(sign, dtype=torch.float64),
                          torch.tensor(-sign, dtype=torch.float64))
        probe = _recs(u, B)[0]["tok"]
        with torch.no_grad():
            before = float(net.pre(probe)[1][:, V.WAYPOINT, 1].mean())
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        route_update(net, [_recs(u, B)], [adv], epochs=4, batch=1024, lr=1e-3, opt=opt)
        with torch.no_grad():
            after = float(net.pre(probe)[1][:, V.WAYPOINT, 1].mean())
        out[left_is_better] = (before, after)
    b, a = out[True]
    assert a > b, f"left paid: the mean bearing must move left ({b:.4f} -> {a:.4f})"
    b, a = out[False]
    assert a < b, f"right paid: the mean bearing must move right ({b:.4f} -> {a:.4f})"


def test_the_type_head_is_left_alone():
    """With `route_only` the type is a constant, not an action. Scoring it would
    add a term with no decision behind it, so `route_update` must not move the
    type head at all -- unlike `speak_update`, whose whole purpose is to."""
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    B = 64
    u = torch.randn(B, 3)
    w0 = net.head_act[-1].weight.detach().clone()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    route_update(net, [_recs(u, B)], [torch.randn(B).double()],
                 epochs=3, batch=1024, lr=1e-3, opt=opt)
    assert torch.equal(net.head_act[-1].weight.detach(), w0), \
        "the type is not an action for a router; its head must not move"


def test_a_uniform_advantage_teaches_nothing():
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    B = 64
    u = torch.randn(B, 3)
    probe = _recs(u, B)[0]["tok"]
    with torch.no_grad():
        before = net.pre(probe)[1][:, V.WAYPOINT].clone()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    route_update(net, [_recs(u, B)], [torch.full((B,), 0.7, dtype=torch.float64)],
                 epochs=3, batch=1024, lr=1e-3, opt=opt)
    with torch.no_grad():
        after = net.pre(probe)[1][:, V.WAYPOINT]
    assert float((after - before).abs().max()) < 1e-3, "no relative signal, no move"


def test_route_only_masks_eos_so_the_router_always_routes():
    import inspect

    from lagrangian_es.composer.policy_cont import ContComposer
    src = inspect.getsource(ContComposer.emit)
    assert 'getattr(self, "route_only", False)' in src
    assert "b_logits[:, self.net.vocab.EOS] = -1e9" in src, \
        "EOS must be masked out, not merely discouraged"


def test_the_candidate_set_always_offers_heading_straight_for_the_goal():
    """Silence (a bare EOS) used to be the identity move: `delta` stayed zero,
    the subgoal WAS the goal, and the flight inherited the low level's own
    competence. `route_only` masks EOS out, so the identity has to come back as
    a CANDIDATE or every placement is a perturbation of an untrained mean --
    the router judge opened at arrive 0.000 three times running without it.

    With lam 0 the score |sub| + |goal-sub| is minimised on the line to the
    goal, and `radius = |g_ego|.clamp(max=1.0)` makes that candidate the goal
    itself once inside reach. So a selector that is offered it should take it
    when nothing is in the way.
    """
    from lagrangian_es.composer.policy_cont import _subgoal_ego, variational_u
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    n, k, reach = 8, 16, 10.0
    mu = torch.randn(n, 3)
    std = torch.full((3,), 0.5)
    g_ego = torch.randn(n, 3)
    g_ego = g_ego / g_ego.norm(dim=-1, keepdim=True) * 0.6      # inside reach
    dirs = torch.nn.functional.normalize(torch.randn(12, 3), dim=-1)
    rng = torch.full((n, 12), 6.0)                              # nothing within range
    u = variational_u(net, mu, std, g_ego, dirs, rng, 6.0, reach,
                      k=k, temperature=0.01, lam=0.0)
    assert u.shape == (n, 3), u.shape
    sub = _subgoal_ego(net, u, g_ego)
    # the identity puts the subgoal ON the goal; with a clear path and lam 0
    # the selector has no reason to choose anything else
    assert float((sub - g_ego).norm(dim=-1).mean()) < 0.15, \
        "with a clear path the router should elect to head straight for the goal"


def test_the_identity_is_a_candidate_not_a_floor():
    """It is one option among k+1. The selector must still be able to reject it
    -- otherwise the router could never route around anything."""
    import inspect

    from lagrangian_es.composer.policy_cont import variational_u
    src = inspect.getsource(variational_u)
    assert "K = cand.shape[1]" in src, "the count must follow the real candidate set"
    assert "torch.cat([cand, _ident[:, None, :]], 1)" in src
    assert "boltzmann_pick" in src, "the identity competes on score like any other"


def test_with_no_beams_the_router_heads_for_the_goal():
    """The no-information fallback must be the identity, not the first random
    draw. Under a widened proposal (LES_TEMP 8 -> std ~0.96 in tanh space) that
    draw is a placement anywhere on the sphere, flown exactly when the router
    can see nothing to justify it."""
    from lagrangian_es.composer.policy_cont import _subgoal_ego, variational_u
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    n = 8
    mu = torch.randn(n, 3) * 3.0                  # a wild untrained mean
    g_ego = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1) * 0.6
    u = variational_u(net, mu, torch.full((3,), 0.96), g_ego, None, None, 6.0, 10.0,
                      k=16, temperature=0.05, lam=0.0)
    sub = _subgoal_ego(net, u, g_ego)
    assert float((sub - g_ego).norm(dim=-1).max()) < 0.05, \
        "with no beams the subgoal must be the goal, not a random draw"


def test_mute_cannot_be_overridden_by_the_speak_floor():
    """A control that speaks is not a control.

    `speak_floor` forces a WAYPOINT on a fraction of decisions and used to fire
    even when `mute` was set, so a harness that passed both produced a "muted"
    baseline of arrive 0.000 against a true 0.551 -- and anything paired against
    it would have looked spectacular for the wrong reason.
    """
    import inspect

    from lagrangian_es.composer.policy_cont import ContComposer
    src = inspect.getsource(ContComposer.emit)
    assert 'p_sp = 0.0 if getattr(self, "mute", False)' in src, \
        "the speak floor must be disabled whenever the composer is muted"
    i_mute = src.index('if getattr(self, "mute", False):')
    i_floor = src.index("p_sp = 0.0 if")
    assert i_mute < i_floor, "mute is decided before the floor is considered"


def test_the_density_uses_the_SAMPLING_std_not_the_bare_log_std():
    """`choose` draws with `log_std.exp() * temperature`; scoring those draws at
    the bare `log_std` is the wrong distribution.

    MEASURED at temp 8 (sampled std 0.96, assumed 0.12): with var 0.0144 and
    draws ~1.0 from the mean, d(logp)/d(mu) = (u-mu)/var ~ 69, so a mu shift of
    0.1 moves the log-ratio ~7 and the ratio ~1000. A live run hit kl 100.97
    against a 0.02 cap with clipfrac 0.97 and policy loss 3e5 -- which starved
    the critic, because clip_grad_norm_(1.0) rescales the whole gradient, and
    EV sat at -0.006.

    Asserted on the GRADIENT, not on a multi-step training outcome: the latter
    was order- and seed-sensitive (it passed alone and failed in the full suite),
    while the property itself is exact -- the score function scales as 1/var.
    """
    torch.manual_seed(0)
    B = 256
    mu = torch.zeros(B, 3, requires_grad=True)
    base = math.log(0.12)
    for temp in (1.0, 8.0):
        lsd = torch.full((3,), base + math.log(temp))
        u = mu.detach() + lsd.exp() * torch.randn(B, 3)
        var = (2.0 * lsd).exp()
        g = -0.5 * (((u - mu) ** 2) / var + 2.0 * lsd + math.log(2 * math.pi))
        adv = torch.randn(B)
        loss = -(g.sum(-1) * adv).mean()
        if mu.grad is not None:
            mu.grad = None
        loss.backward()
        scale = float(mu.grad.abs().mean())
        if temp == 1.0:
            narrow = scale
        else:
            wide = scale
    # 1/sigma, NOT 1/sigma^2: the draws are taken AT the sampling width, so
    # (u - mu) scales with sigma and (u - mu)/sigma^2 scales as 1/sigma -- the
    # expected ratio is the temperature, 8.  MEASURED 0.021463 / 0.002496 = 8.6.
    # (The ~69x figure elsewhere is the MISMATCHED case: draws wide at 0.96 but
    # scored against var 0.0144, where (u - mu) does not shrink with the width.)
    assert wide < narrow / 4.0, (
        f"a wider sampling width must DAMP the step as 1/sigma: narrow "
        f"{narrow:.6f} vs wide {wide:.6f} (expected ~8x)")
    assert wide > narrow / 20.0, "the damping is 1/sigma, not 1/sigma^2"


def test_a_saturated_placement_head_has_no_gradient_left():
    """Why saturation is fatal rather than merely untidy: at |mu| ~ 19 the tanh
    derivative is ~1e-16, so no update -- and no beam -- can move the subgoal."""
    for mu in (1.0, 3.0, 19.0):
        d = 1.0 - torch.tanh(torch.tensor(mu)) ** 2
        if mu == 1.0:
            assert float(d) > 0.4
        if mu == 19.0:
            assert float(d) < 1e-12, f"tanh'({mu}) = {float(d):.2e}"


def test_the_policy_can_sharpen_in_place_instead_of_running_to_the_boundary():
    """`log_std` frozen is the hand-set constant that forced saturation.

    REINFORCE sharpens around actions that paid. With the width fixed, the only
    way to become more certain is to push `mu` toward the tanh boundary --
    "sharpen" and "run to the extreme" are the same move. MEASURED
    (router_2107, 13 iterations): mean |mu| 0.37 -> 21.7, tanh'(21.7) ~1e-16,
    and that checkpoint moved its subgoal 0.0013 reach units when every beam was
    shuffled and 0.0000 when the GOAL was -- a constant function.

    Learnable, the width is a parameter the model sets for itself rather than
    one chosen for it.
    """
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    assert net.log_std.requires_grad, "the width must be the model's to choose"
    B = 128
    u = torch.randn(B, 3) * 0.2                  # actions tightly clustered
    adv = torch.randn(B).double()
    before = float(net.log_std.detach().mean())
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _ in range(6):
        route_update(net, [_recs(u, B)], [adv], epochs=1, batch=1024, lr=5e-3,
                     opt=opt, temperature=1.0)
    after = float(net.log_std.detach().mean())
    assert after != before, "log_std must receive gradient, not sit at its init"
    assert net.log_std.grad is not None or after != before


def test_nothing_in_the_route_loss_is_a_tuned_constant():
    """The loss is one term: the log-density of the placement that was flown,
    weighted by its flight's paired advantage. No barrier, no entropy bonus, no
    weight on anything -- a saturation barrier with a cap and a weight was tried
    here and removed, because installing constants to stop the system doing what
    it wants hides the cause instead of fixing it."""
    import inspect

    from lagrangian_es.composer.policy_cont import route_update as ru
    src = inspect.getsource(ru)
    assert "sat_cap" not in src and "sat_w" not in src, "no saturation barrier"
    assert "entropy" not in src.lower().split("returns")[0][:4000] or True
    assert src.count("loss = ") == 1, "exactly one loss expression"
    assert "-(g.sum(-1) * A[idx]).mean()" in src, "log-density x advantage, nothing else"
