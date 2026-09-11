"""A row's output must not depend on what it was batched with.

The number of perception tokens CHANGES during an episode -- the built map
contributes nothing until the vehicle has seen something, so early decisions
carry ~90 entity tokens and later ones ~102.  Collating them pads to the batch
maximum, and the padded positions are given type 0 (which is SELF) with zero
features.  If the mask polarity were wrong, or the padding were attended to,
every row's answer would depend on its batch-mates -- silently, and in a way
that looks like noise rather than a bug.
"""
import torch

from lagrangian_es.composer.policy import collate_tok
from lagrangian_es.composer.policy_cont import ContPolicyNet
from lagrangian_es.composer.transformer import F as NF


def _sample(n_ent, n_chain, seed):
    g = torch.Generator().manual_seed(seed)
    return {"self": torch.randn(NF, generator=g),
            "goal": torch.randn(NF, generator=g),
            "entities": torch.randn(n_ent, NF, generator=g),
            "ent_types": torch.randint(2, 8, (n_ent,), generator=g),
            "ent_mask": torch.ones(n_ent, dtype=torch.bool),
            "chain": torch.randn(n_chain, NF, generator=g),
            "chain_types": torch.randint(4, 6, (n_chain,), generator=g),
            "chain_mask": torch.zeros(n_chain, dtype=torch.bool),
            "psi": torch.zeros(())}


def test_padding_does_not_change_a_rows_output():
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1).eval()
    short, long = _sample(90, 40, 1), _sample(102, 64, 2)

    with torch.no_grad():
        alone_s = net.pre(collate_tok([short]))
        alone_l = net.pre(collate_tok([long]))
        together = net.pre(collate_tok([short, long]))

    for i, alone in enumerate((alone_s, alone_l)):
        for k, (a, b) in enumerate(zip(alone, together)):
            assert torch.allclose(a[0], b[i], atol=1e-5), (
                f"row {i} output {k} changed when batched with a differently sized row: "
                f"max diff {float((a[0] - b[i]).abs().max()):.2e} -- the padding is being attended to")


def test_masked_padding_carries_no_information():
    """Padded positions take type 0 (SELF) and zero features.  Changing what is
    written into them must be invisible: if it is not, they are being read."""
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1).eval()
    batch = collate_tok([_sample(90, 40, 1), _sample(102, 64, 2)])
    with torch.no_grad():
        before = net.pre(batch)
    poisoned = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    pad = ~poisoned["ent_mask"]
    assert bool(pad.any()), "fixture produced no padding"
    poisoned["entities"][pad] = 37.0                  # nonsense in the padded slots
    poisoned["ent_types"][pad] = 3
    with torch.no_grad():
        after = net.pre(poisoned)
    for k, (a, b) in enumerate(zip(before, after)):
        assert torch.allclose(a, b, atol=1e-5), (
            f"output {k} moved when the PADDING changed: max diff "
            f"{float((a - b).abs().max()):.2e}")


def test_the_chain_summary_comes_from_the_rows_own_last_entry():
    """`read_out` takes `ch[:, -1:]` as "the latest chain state".  The chain is
    LEFT-padded so that position is always the row's real most recent event.

    Right-padding made it a padded slot for any row shorter than the batch
    maximum, so a short row's chain summary was computed from a zero embedding
    instead of its own last event -- a constant 7.1e-04 shift in the logits,
    independent of how much padding, which is what distinguishes a structural
    error from leaked content.  Rollouts never exposed it because every row
    carries the same chain length there; only the update, which collates
    records from different times, pads at all.  So the network saw one
    representation in flight and a different one while learning from it.
    """
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1).eval()
    base = _sample(90, 40, 1)
    with torch.no_grad():
        alone = net.pre(collate_tok([base]))[0][0]
    for n_chain in (41, 64, 128):
        with torch.no_grad():
            together = net.pre(collate_tok([base, _sample(90, n_chain, 2)]))[0][0]
        d = float((alone - together).abs().max())
        assert d < 1e-5, (
            f"a row batched with a {n_chain}-long chain moved by {d:.2e}; its chain "
            "summary is being read from a padded position")


def test_the_chain_is_left_padded():
    """Stated directly, because `read_out`'s `ch[:, -1:]` depends on it and the
    dependency is not visible from either site alone."""
    batch = collate_tok([_sample(90, 40, 1), _sample(90, 64, 2)])
    cm = batch["chain_mask"]                      # True = padding
    kc = cm.shape[1]
    assert bool(cm[0, :kc - 40].all()), "the short row's padding is not at the FRONT"
    assert not bool(cm[0, kc - 40:].any()), "the short row's real entries are not at the END"
    assert not bool(cm[1].any()), "the longest row should carry no padding"


def test_every_rows_chain_ends_with_its_own_newest_event():
    """`read_out` takes `ch[:, -1:]` as "the latest chain state", so that slot
    must be a REAL event for every row.

    An instruction belongs only to the rows that were asked for it, so the
    newest event in the shared timeline is padding for everyone else.  Measured
    on a real rollout: with decisions confined to the report clock 0% of rows
    had that slot padded, but once a beam interrupt could fire for a SUBSET of
    rows it was 58.4% -- most rows summarising their history from a masked
    slot.  Mid-history holes were 97-100% throughout, from before any of that.

    The tokenizer now right-aligns each row's valid events (stable sort on
    validity, so time order survives), which closes both.
    """
    import torch as _t
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_composer, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.composer import tokens as TK
    from lagrangian_es.util import make_gen

    seen = {"rows": 0, "last_padded": 0, "holes": 0}
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range",), gating="arrival", seed=0,
                 composer="policy_cont",
                 composer_kw=(("reach", 10.0), ("every", 20), ("measure_every", 20)),
                 task_kw=(("n_legs", 2), ("max_leg", 20.0)),
                 system_kw=(("free_start", True), ("speed_limit", 5.0)),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")),
                 rollout=RolloutCfg(n_eps=16, ep_steps=200, dead_mode="constant",
                                    dead_cost=40.0, goal_bonus=60.0))
    system, trainable, task = build(cfg)
    system.difficulty = 1.0
    comp = build_composer(cfg, system, trainable)
    comp.beam_trigger = 1.5                  # the setting that exposed it
    comp.stochastic = True; comp.records = []; comp.record_rows = None
    comp.reset(16); comp.pair(16, 1)

    orig = TK.Tokenizer.__call__

    def spy(self, ctx, chain_instr):
        out = orig(self, ctx, chain_instr)
        cm = out["chain_mask"]
        if cm.numel() and cm.shape[1] > 1:
            seen["rows"] += cm.shape[0]
            seen["last_padded"] += int(cm[:, -1].sum())
            for b in range(cm.shape[0]):
                v = (~cm[b]).nonzero().flatten()
                if v.numel() and int(cm[b, int(v[0]):int(v[-1]) + 1].sum()) > 0:
                    seen["holes"] += 1
        return out

    TK.Tokenizer.__call__ = spy
    try:
        _t.manual_seed(0)
        with _t.no_grad():
            rig = Rollout(system, trainable, task, cfg.rollout,
                          build_sensors(cfg, system), composer=comp)
            rig.run(trainable.init()[None], task.sample(16, make_gen(1)), 2)
    finally:
        TK.Tokenizer.__call__ = orig

    assert seen["rows"] > 0, "no chains were built"
    assert seen["last_padded"] == 0, (
        f"{seen['last_padded']} of {seen['rows']} row-views end on a padded slot")
    assert seen["holes"] == 0, (
        f"{seen['holes']} of {seen['rows']} row-views have padding mid-history")
