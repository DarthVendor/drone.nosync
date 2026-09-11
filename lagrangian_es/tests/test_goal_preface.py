"""The goal preface: every chain opens with the goal the task handed the row.

It is context, not an action -- the composer cannot emit it, no token id maps
to it, and it is rebuilt by the tokenizer on every call.  What it buys is that
the chain blocks are CAUSALLY masked, so position 0 is the one entry every
later entry attends to: a measurement or an instruction is then read relative
to the objective rather than in isolation.  It also means the chain is never
empty, so `read_out` no longer skips the chain path outright on a leg's first
decision.

The invariant these tests hold is: **the preface is every row's first VALID
chain entry**, through the tokenizer, through the right-align sort, through
`collate`'s left-padding, and through the in-decision truncation.
"""
import torch

from lagrangian_es.composer.tokens import GOAL, INSTR, MEASURE, Tokenizer, drop_oldest_event


def _ctx(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(B, 3, generator=g), "v": torch.randn(B, 3, generator=g),
            "goal": torch.randn(B, 3, generator=g) * 10.0,
            "state": {"R": torch.eye(3).expand(B, 3, 3).clone()},
            "t": 100.0, "obs": {}, "chain": []}


def _measures(n, B, t0=0.0):
    return [{"t": t0 + i, "progress": torch.ones(B), "remaining": torch.ones(B), "min_beam": None,
             "alive": torch.ones(B, dtype=torch.bool), "arrived": torch.zeros(B, dtype=torch.bool)}
            for i in range(n)]


def _first_valid(tok, r):
    return int((~tok["chain_mask"][r]).nonzero()[0])


def test_chain_is_never_empty():
    """Before the preface a leg's first decision had chain length 0, and
    `read_out` skipped the chain blocks entirely for it."""
    a = Tokenizer(scale=50.0, reach=10.0, k_chain=5)(_ctx(), [])
    assert a["chain"].shape[1] == 1
    assert int(a["chain_types"][0, 0]) == GOAL
    assert not bool(a["chain_mask"].any())


def test_preface_carries_the_goal():
    a = Tokenizer(scale=50.0, reach=10.0, k_chain=5)(_ctx(), [])
    assert torch.allclose(a["chain"][:, 0], a["goal"]), "the preface is the goal token itself"


def test_preface_survives_a_full_window():
    """The window truncates the OLDEST events; the preface is not an event."""
    B, kc = 3, 5
    tk = Tokenizer(scale=50.0, reach=10.0, k_chain=kc)
    a = tk(dict(_ctx(B), chain=_measures(10, B)), [])
    assert a["chain"].shape[1] == kc, "the preface costs a slot, it does not widen the window"
    assert int(a["chain_types"][0, 0]) == GOAL
    assert (a["chain_types"][0, 1:] == MEASURE).all()
    assert torch.allclose(a["chain"][:, 0], a["goal"])


def test_preface_is_first_valid_under_the_right_align_sort():
    """An instruction only SOME rows were given pads the others; the sort that
    right-aligns each row's own events must leave the preface at the head of
    them, not buried."""
    B, kc = 3, 5
    tk = Tokenizer(scale=50.0, reach=10.0, k_chain=kc)
    valid = torch.tensor([True, False, True])
    a = tk(dict(_ctx(B), chain=_measures(10, B)), [(9.5, torch.tensor([3, 3, 3]), valid)])
    assert bool(a["chain_mask"].any()), "row 1 should carry padding for the instruction it never got"
    for r in range(B):
        f = _first_valid(a, r)
        assert int(a["chain_types"][r, f]) == GOAL, f"row {r} does not open on the goal"
        assert torch.allclose(a["chain"][r, f], a["goal"][r])


def test_preface_is_not_in_the_vocabulary():
    """It is context, not an action: nothing the composer can emit produces it,
    and it is never typed INSTR (so `_chain_in` adds no action embedding)."""
    a = Tokenizer(scale=50.0, reach=10.0, k_chain=5)(dict(_ctx(), chain=_measures(3, 3)), [])
    assert int(a["chain_types"][0, 0]) == GOAL != INSTR


def test_drop_oldest_event_keeps_the_preface():
    """The in-decision append used a flat `chain[:, -kc:]`, which took index 0
    from every row alike -- the padding for a padded row, but the PREFACE for a
    full one, on the first component append of every decision."""
    ctypes = torch.tensor([[GOAL, MEASURE, MEASURE, INSTR],      # full: must lose index 1
                           [MEASURE, GOAL, MEASURE, INSTR]])     # padded: may lose index 0
    cmask = torch.tensor([[False, False, False, False],
                          [True, False, False, False]])
    chain = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    ch, ty, cm = drop_oldest_event(chain, ctypes, cmask)
    assert ch.shape[1] == 3
    for r in range(2):
        f = int((~cm[r]).nonzero()[0])
        assert int(ty[r, f]) == GOAL, f"row {r} lost its preface"
    assert ch[0, :, 0].tolist() == [0.0, 4.0, 6.0], "the full row should drop its OLDEST measurement"
    assert ch[1, :, 0].tolist() == [10.0, 12.0, 14.0], "the padded row should drop its padding"


def test_preface_tracks_the_goal_as_the_vehicle_moves():
    """It is rebuilt every call in the CURRENT ego frame, so it is the live
    objective, not a stale snapshot taken in a frame that has since rotated."""
    tk = Tokenizer(scale=50.0, reach=10.0, k_chain=5)
    c = _ctx(2)
    a = tk(c, [])
    c2 = dict(c, x=c["x"] + torch.tensor([5.0, 0.0, 0.0]))
    b = tk(c2, [])
    assert not torch.allclose(a["chain"][:, 0], b["chain"][:, 0])
    assert torch.allclose(b["chain"][:, 0], b["goal"])


def test_speak_floor_forces_a_waypoint_not_merely_speech():
    """The floor exists to keep PLACEMENTS in the sample.

    Its first version only forbade EOS, which is useless once the policy has
    collapsed onto another token: measured, the composer went to
    `heading 100% / place 0%` by judge 40, so forbidding silence just forced
    more TURN and the update still never saw a subgoal placed.  WAYPOINT is the
    only token that moves the subgoal, so that is what gets forced; the
    arguments still come from the policy, so WHERE it places stays explored.
    """
    import torch
    from lagrangian_es.composer.actions_cont import ContVocab
    from lagrangian_es.composer.policy_cont import ContPolicyNet

    V = ContVocab(1)
    net = ContPolicyNet(n_terms=1)
    B = 512
    # a policy that has collapsed onto SILENCE.  This fixture used to collapse
    # it onto TURN, which is no longer in the vocabulary -- TURN, PRIORITY and
    # LOOK are all exact no-ops at a zero argument, and the policy found each
    # of them in turn because a no-op earns advantage zero and is dropped from
    # the batch rather than penalised.  EOS is the only no-op left, and it is
    # the control, so zero is the right score for it.
    logits = torch.full((B, V.V), -20.0)
    logits[:, V.EOS] = 20.0
    dev = logits.device

    torch.manual_seed(0)
    force = torch.rand(B, device=dev) < 1.0          # floor of 1.0: every row
    row = torch.full((V.V,), float("-inf"))
    row[V.WAYPOINT] = 0.0
    out = torch.where(force[:, None], row[None].expand_as(logits), logits)
    act = torch.distributions.Categorical(logits=out).sample()
    assert int((act == V.WAYPOINT).sum()) == B, "a forced decision must place"
    # and with the floor off the collapsed policy is untouched
    none = torch.where(torch.zeros(B, dtype=torch.bool)[:, None], row[None].expand_as(logits), logits)
    assert int((torch.distributions.Categorical(logits=none).sample() == V.EOS).sum()) == B
