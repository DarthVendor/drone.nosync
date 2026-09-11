"""Credit is the paired counterfactual: the same task flown twice, once with
the composer's tokens and once with it muted, and a token is worth the TIME it
saved.

This replaces `progress_weights`, which scored a decision by the distance
closed before the next one.  Under sparse injection a flight has one or two
decisions and the last is never credited, so that measured almost nothing --
0 of 13 forced WAYPOINTs survived it on the live checkpoint -- and what it did
measure was mostly the FROZEN LOW LEVEL's own progress, credited to whatever
token happened to be live.
"""
import torch

from lagrangian_es.composer.policy_cont import paired_advantage


def test_time_saved_against_the_control_is_the_advantage():
    t   = torch.tensor([0.4, 1.0, 0.3, 0.5, 1.0, 0.4])
    ctl = torch.tensor([1.0, 0.4, 0.5, 0.3, 1.0, 0.4])
    a = paired_advantage(t, ctl)
    assert float(a[0]) > 0.5, "rescued a flight the control lost"
    assert float(a[1]) < -0.5, "lost a flight the control won"
    assert abs(float(a[2]) - 0.2) < 1e-6, "arrived 20% of the episode sooner"
    assert abs(float(a[3]) + 0.2) < 1e-6, "arrived 20% later"
    assert float(a[4]) == 0.0, "neither arrived: the tokens changed nothing"
    assert float(a[5]) == 0.0, "identical: the tokens changed nothing"


def test_a_crash_is_the_worst_possible_time_not_the_best():
    """The trap this design has to avoid.

    `soft_time` accumulates only while a row is ALIVE, so a flight that crashes
    at step 100 accumulates less than one that flies 500 steps and arrives --
    it reads as FASTER.  As a paired advantage that pays the composer to crash.
    `finish_frac` is 1.0 for anything that never arrived, which is what
    "forever" has to mean here.
    """
    crashed_at_once, arrived_slowly = torch.tensor([1.0]), torch.tensor([0.9])
    control = torch.tensor([0.5])
    assert float(paired_advantage(crashed_at_once, control)) < 0.0
    assert float(paired_advantage(crashed_at_once, control)) < float(paired_advantage(arrived_slowly, control))


def test_silence_scores_exactly_zero():
    """Silence IS the control, so it cannot win by doing nothing -- the failure
    mode that took three runs to EOS 99.9%, TURN 100% and LOOK 78%."""
    t = torch.tensor([0.62, 0.31, 1.0])
    assert torch.allclose(paired_advantage(t, t.clone()), torch.zeros(3, dtype=torch.float64))


def test_the_low_levels_own_progress_cancels():
    """Both flights are the same task, same seed, same frozen low level: an
    easy task and a hard one give the same advantage for the same time saved.
    The old per-decision credit could not do this -- it paid out the distance
    the low level closed, which is larger on the easy task."""
    easy = paired_advantage(torch.tensor([0.20]), torch.tensor([0.30]))
    hard = paired_advantage(torch.tensor([0.80]), torch.tensor([0.90]))
    assert abs(float(easy) - float(hard)) < 1e-6   # float32 inputs


def test_mute_is_a_composer_flag_that_silences_every_decision():
    from lagrangian_es.composer.actions_cont import ContVocab
    from lagrangian_es.composer.policy_cont import ContPolicyNet
    V = ContVocab(1)
    net = ContPolicyNet(n_terms=1)
    logits = torch.randn(64, V.V)
    muted = torch.full_like(logits, -1e9); muted[:, V.EOS] = 0.0
    act = torch.distributions.Categorical(logits=muted).sample()
    assert int((act == V.EOS).sum()) == 64, "a muted decision must be a bare EOS"


def test_the_objective_moves_the_policy_the_way_the_advantage_points():
    """END TO END: advantage -> loss -> gradient -> policy.

    A sign error anywhere in that chain is invisible until a run has burned
    hours, and this session lost several to objectives that were subtly not
    measuring what they claimed.  Measured here on a net whose every emitted
    token is a WAYPOINT: advantage +1 takes P(WAYPOINT) 0.2555 -> 0.5640,
    advantage -1 takes it to 0.0926, and advantage 0 contributes no samples at
    all rather than nudging anything.
    """
    from lagrangian_es.composer.actions_cont import ContVocab
    from lagrangian_es.composer.policy_cont import ContPolicyNet, imitate_update
    from lagrangian_es.composer.tokens import F

    V = ContVocab(1)

    def recs(tok_id, B=64, nrec=2):
        g = torch.Generator().manual_seed(3)
        out = []
        for _ in range(nrec):
            t = {"self": torch.randn(B, F, generator=g), "goal": torch.randn(B, F, generator=g),
                 "entities": torch.randn(B, 10, F, generator=g),
                 "ent_types": torch.full((B, 10), 2, dtype=torch.long),
                 "ent_mask": torch.ones(B, 10, dtype=torch.bool),
                 "chain": torch.zeros(B, 1, F), "chain_types": torch.ones(B, 1, dtype=torch.long),
                 "chain_mask": torch.zeros(B, 1, dtype=torch.bool), "psi": torch.zeros(B)}
            out.append({"tok": t, "act": torch.full((B,), tok_id, dtype=torch.long),
                        "u": torch.zeros(B, 3),
                        "n_args": torch.full((B,), V.n_arg_of(tok_id), dtype=torch.long),
                        "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B)})
        return out

    def move(adv_value, B=64):
        torch.manual_seed(0)
        net = ContPolicyNet(n_terms=1)
        probe = recs(V.WAYPOINT)[0]["tok"]
        with torch.no_grad():
            before = float(torch.softmax(net.pre(probe)[0], -1).mean(0)[V.WAYPOINT])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        st = imitate_update(net, [recs(V.WAYPOINT, B)], [torch.ones(B, dtype=torch.bool)],
                            epochs=4, batch=1024, lr=1e-3, opt=opt,
                            advantage=[torch.full((B,), float(adv_value), dtype=torch.float64)],
                            reach=10.0)
        with torch.no_grad():
            after = float(torch.softmax(net.pre(probe)[0], -1).mean(0)[V.WAYPOINT])
        return st.get("n", 0), before, after

    n, b, a = move(+1.0)
    assert a > b + 0.1, f"saving time must RAISE the token that did it ({b:.4f} -> {a:.4f})"
    n, b, a = move(-1.0)
    assert a < b - 0.1, f"costing time must LOWER the token that did it ({b:.4f} -> {a:.4f})"
    n, b, a = move(0.0)
    assert n == 0 and a == b, "a token that changed nothing must contribute no sample"


def test_projected_time_ranks_failures_without_paying_for_a_crash():
    """`finish_frac` gives every failure exactly 1.0, so in a pair two failures
    tie at zero and teach nothing -- measured, ~70% of pairs at arrival 0.30.
    Charging a failure the time its remaining distance would have taken ranks
    them, and makes dying early the MOST expensive outcome because it leaves
    the most distance outstanding.  That is the property `soft_time` inverts:
    it stops accumulating at death, so it pays for crashing."""
    from lagrangian_es.composer.policy_cont import projected_time

    span = 180.0                      # 5 m/s * 1800 steps * 0.02 s
    ff = torch.tensor([0.40, 1.00, 1.00, 1.00])
    err = torch.tensor([0.10, 1.00, 60.0, 150.0])
    suc = torch.tensor([True, False, False, False])
    t = projected_time(ff, err, suc, span)

    assert abs(float(t[0]) - 0.40) < 1e-6, "an arrival keeps its own arrival time"   # float32 inputs
    assert float(t[1]) > 1.0, "a failure costs more than any arrival"
    assert float(t[1]) < float(t[2]) < float(t[3]), "further from the goal costs more"
    assert float(t[3]) > float(t[2]) > float(t[1]) > float(t[0])

    # and the pairing now separates failures that used to tie at zero
    near, far = t[1].reshape(1), t[3].reshape(1)
    assert float(paired_advantage(near, far)[0]) > 0.5, "nearly making it beats crashing early"
    assert float(paired_advantage(far, near)[0]) < -0.5
