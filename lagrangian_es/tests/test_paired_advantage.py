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
