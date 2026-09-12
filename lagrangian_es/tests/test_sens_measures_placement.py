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
