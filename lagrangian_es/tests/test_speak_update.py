"""The composer's only real decision is WHETHER to speak, so that is what the
loss has to reach.

MEASURED Sept 11 2026, 256 paired tasks, identical seeds, a WAYPOINT forced at
full reach on every decision:

    muted (no tokens at all)      arrive 0.539
    forced WAYPOINT   +90 deg     arrive 0.281
    forced WAYPOINT  +180 deg     arrive 0.285
    forced WAYPOINT    0 deg      arrive 0.285

Emitting costs a quarter of the arrival rate and three bearings spanning 180
degrees do identical damage, so the placement carries nothing and the type
decision carries everything.  `time_update` trained only the placement.
"""
import torch

from lagrangian_es.composer.actions_cont import ContVocab
from lagrangian_es.composer.policy_cont import ContPolicyNet, speak_update
from lagrangian_es.composer.tokens import F

V = ContVocab(1)


def _recs(acts, B):
    g = torch.Generator().manual_seed(3)
    t = {"self": torch.randn(B, F, generator=g), "goal": torch.randn(B, F, generator=g),
         "entities": torch.randn(B, 10, F, generator=g),
         "ent_types": torch.full((B, 10), 2, dtype=torch.long),
         "ent_mask": torch.ones(B, 10, dtype=torch.bool),
         "chain": torch.zeros(B, 1, F), "chain_types": torch.ones(B, 1, dtype=torch.long),
         "chain_mask": torch.zeros(B, 1, dtype=torch.bool), "psi": torch.zeros(B)}
    return [{"tok": t, "act": acts, "u": torch.zeros(B, 3),
             "n_args": torch.zeros(B, dtype=torch.long),
             "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B)}]


def _move(speaking_is_better, B=128):
    """Half the flights spoke, half stayed silent; one group saved time."""
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    spoke = torch.arange(B) < B // 2
    acts = torch.where(spoke, torch.full((B,), V.WAYPOINT), torch.full((B,), V.EOS))
    sign = 1.0 if speaking_is_better else -1.0
    adv = torch.where(spoke, torch.tensor(sign, dtype=torch.float64),
                      torch.tensor(-sign, dtype=torch.float64))
    probe = _recs(acts, B)[0]["tok"]
    with torch.no_grad():
        before = float(torch.softmax(net.pre(probe)[0], -1).mean(0)[V.WAYPOINT])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    speak_update(net, [_recs(acts, B)], [adv], epochs=4, batch=1024, lr=1e-3, opt=opt)
    with torch.no_grad():
        after = float(torch.softmax(net.pre(probe)[0], -1).mean(0)[V.WAYPOINT])
    return before, after


def test_saving_time_raises_the_decision_to_speak():
    before, after = _move(speaking_is_better=True)
    assert after > before + 0.05, f"speaking saved time; P(WAYPOINT) {before:.4f} -> {after:.4f}"


def test_costing_time_lowers_the_decision_to_speak():
    before, after = _move(speaking_is_better=False)
    assert after < before - 0.05, f"speaking cost time; P(WAYPOINT) {before:.4f} -> {after:.4f}"


def test_time_update_cannot_reach_the_token_type_head_at_all():
    """The bug this loss replaces, pinned so it cannot come back.

    `time_update`'s terms are `T_hat(q, g.detach())` and `T_hat(q, g)`, where
    `g` is built from the WAYPOINT ARGUMENTS (`mu`).  The type logits come out
    of `head_act`, which appears in NEITHER -- calling `pre()` runs it forward
    but the logits are discarded, so no gradient ever reaches its weights.

    Live consequence over 23 iterations: `nll` fell 0.113 -> 0.047 while
    `speak` sat at 0.509-0.515 and `arrive` never left the binomial floor
    (observed sd 0.0155 against 0.0133 at 1152 episodes).  `speak` drifted at
    all only because `head_act`'s INPUT moved as the shared encoder trained --
    and it went flat exactly when `nll` plateaued at iteration 11.
    """
    from lagrangian_es.composer.policy_cont import time_update
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    B = 64
    acts = torch.full((B,), V.WAYPOINT)
    w0 = net.head_act[-1].weight.detach().clone()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    st = time_update(net, [_recs(acts, B)], [torch.rand(B).double()], ep_steps=200,
                     epochs=4, batch=1024, lr=1e-3, opt=opt)
    assert st["n"] > 0, "the fixture must produce samples, or this proves nothing"
    assert torch.equal(net.head_act[-1].weight.detach(), w0), \
        "time_update moved the token-type head; it is not supposed to be able to"
    assert net.head_act[-1].weight.grad is None or \
        float(net.head_act[-1].weight.grad.abs().max()) == 0.0


def test_speak_update_does_reach_that_same_head():
    """The contrast that makes the test above meaningful."""
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    B = 128
    spoke = torch.arange(B) < B // 2
    acts = torch.where(spoke, torch.full((B,), V.WAYPOINT), torch.full((B,), V.EOS))
    adv = torch.where(spoke, torch.tensor(1.0, dtype=torch.float64),
                      torch.tensor(-1.0, dtype=torch.float64))
    w0 = net.head_act[-1].weight.detach().clone()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    speak_update(net, [_recs(acts, B)], [adv], epochs=2, batch=1024, lr=1e-3, opt=opt)
    assert not torch.equal(net.head_act[-1].weight.detach(), w0), \
        "the whole point of this loss is that it moves the type head"


def test_a_uniform_advantage_teaches_nothing():
    """Every flight equally good is no information: the paired control is the
    baseline, so a constant difference must not push the policy anywhere."""
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    B = 64
    acts = torch.full((B,), V.WAYPOINT)
    probe = _recs(acts, B)[0]["tok"]
    with torch.no_grad():
        before = float(torch.softmax(net.pre(probe)[0], -1).mean(0)[V.WAYPOINT])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    speak_update(net, [_recs(acts, B)], [torch.full((B,), 0.7, dtype=torch.float64)],
                 epochs=3, batch=1024, lr=1e-3, opt=opt)
    with torch.no_grad():
        after = float(torch.softmax(net.pre(probe)[0], -1).mean(0)[V.WAYPOINT])
    assert abs(after - before) < 1e-3, f"{before:.5f} -> {after:.5f}"
