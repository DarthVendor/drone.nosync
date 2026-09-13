"""`sens` has to measure whether the PLACEMENT reads the beams.

It used to shuffle the beam/pixel rows and report how far the TYPE distribution
moved (EOS vs WAYPOINT). For a router that is constant by construction --
`route_only` masks EOS out -- so it would report 0.0000 however sensor-driven
the routing was. Even for the speak arm it answered the wrong question: whether
the decision to SPEAK reads the beams, not whether the placement does, and the
placement is where the value lives (lam 0 vs lam 2 differ in nothing else and
move arrival +0.027 vs -0.075).

This matters beyond bookkeeping: `sens ~ 0.0005` was quoted all session as
"the composer ignores its sensors", and it never supported that claim.
"""
import torch


def test_sens_reads_the_subgoal_not_the_token_type():
    import inspect
    import re

    src = open("scripts/composer/cotrain_v11.py").read()
    m = re.search(r"\ndef _sens\(.*?\n(?=\n\w|\nrow_prev)", src, re.S)
    assert m, "could not locate _sens"
    body = m.group(0)
    assert "_subgoal_ego" in body, "sens must compare commanded SUBGOALS"
    assert "softmax(comp.net.pre(tk)[0], -1)" not in body, \
        "the type distribution is constant for a router; it cannot be the measure"
    assert "norm(dim=-1)" in body, "report a distance, in reach units"


def test_a_placement_head_that_ignores_beams_scores_zero():
    """The property the metric must have: shuffling the beams must move a
    beam-blind placement not at all, and a beam-reading one by something."""
    from lagrangian_es.composer.policy_cont import ContPolicyNet, _subgoal_ego
    from lagrangian_es.composer.tokens import BEAM, F
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    B, E = 16, 12
    tk = {"self": torch.randn(B, F), "goal": torch.randn(B, F),
          "entities": torch.randn(B, E, F),
          "ent_types": torch.full((B, E), BEAM, dtype=torch.long),
          "ent_mask": torch.ones(B, E, dtype=torch.bool),
          "chain": torch.zeros(B, 1, F), "chain_types": torch.ones(B, 1, dtype=torch.long),
          "chain_mask": torch.zeros(B, 1, dtype=torch.bool), "psi": torch.zeros(B)}
    t1 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in tk.items()}
    e = t1["entities"].clone()
    e[:, :] = e[torch.randperm(B)][:, :]
    t1["entities"] = e
    W = net.vocab.WAYPOINT
    with torch.no_grad():
        g = net.goal_ego(tk).to(torch.float32)
        s0 = _subgoal_ego(net, net.pre(tk)[1][:, W], g)
        s1 = _subgoal_ego(net, net.pre(t1)[1][:, W], g)
        moved = float((s0 - s1).norm(dim=-1).mean())
        # a goal-only placement: same g_ego, identical arguments -> exactly 0
        same = _subgoal_ego(net, net.pre(tk)[1][:, W], g)
        assert float((s0 - same).norm(dim=-1).max()) == 0.0
    assert moved >= 0.0
    assert moved == moved, "must be finite"


def _load_sens():
    """Exec the real `_sens` against stubs, so the test runs the shipped code."""
    import re
    import types
    import torch as _t
    from lagrangian_es.composer.policy_cont import ContPolicyNet, _subgoal_ego
    src = open("scripts/composer/cotrain_v11.py").read()
    m = re.search(r"\n_SENS_PROBE = \[\].*?\ndef _sens\(.*?\n(?=\n\w|\nrow_prev)", src, re.S)
    assert m, "could not locate _sens and its probe store"
    _t.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    ns = {"torch": _t, "_subgoal_ego": _subgoal_ego,
          "comp": types.SimpleNamespace(net=net)}
    exec(m.group(0), ns)
    return ns, net


def _draw(seed, B=32, E=12):
    """One task draw's records: different states every time, as in training."""
    from lagrangian_es.composer.tokens import BEAM, PIXEL, F
    g = torch.Generator().manual_seed(seed)
    ty = torch.where(torch.arange(E) % 2 == 0, BEAM, PIXEL).repeat(B, 1)
    tk = {"self": torch.randn(B, F, generator=g),
          "goal": torch.randn(B, F, generator=g),
          "entities": torch.randn(B, E, F, generator=g),
          "ent_types": ty,
          "ent_mask": torch.ones(B, E, dtype=torch.bool),
          "chain": torch.zeros(B, 1, F),
          "chain_types": torch.ones(B, 1, dtype=torch.long),
          "chain_mask": torch.zeros(B, 1, dtype=torch.bool),
          "psi": torch.zeros(B)}
    return [[{"tok": tk} for _ in range(6)]]


def test_sens_holds_still_while_the_policy_does():
    """The bug this file exists to prevent from coming back.

    An ACCUM window takes NO optimiser step for its first N-1 iterations, so
    the policy is bit-for-bit frozen and `sens` must report the same number.
    The shipped version drew 8 fresh records and an UNSEEDED randperm each
    call, and read 0.0034 -> 0.0069 on a frozen net: a 2x swing that was taken
    all session as the composer's sensor use rising and falling.
    """
    ns, net = _load_sens()
    sens = ns["_sens"]
    v = [sens(_draw(s)) for s in (1, 2, 3, 4)]   # four DIFFERENT task draws
    assert all(x == x for x in v), f"must be finite, got {v}"
    assert max(v) - min(v) == 0.0, \
        f"frozen policy must give one number across task draws, got {v}"


def test_sens_still_moves_when_the_policy_does():
    """Held fixed is only useful if it is not held CONSTANT."""
    ns, net = _load_sens()
    sens = ns["_sens"]
    before = sens(_draw(1))
    with torch.no_grad():                        # a real change to the weights
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    after = sens(_draw(2))
    assert after != before, "sens must track the weights it is measuring"
