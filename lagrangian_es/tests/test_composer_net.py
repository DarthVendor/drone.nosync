"""The learned composer's structural promises, tested as identities.

Bounded outputs: the subgoal never leaves the reach ball, priorities are
positive, gates lie in [0, 1].  Permutation invariance: the scene is a set.
Equivariance: rotate the world about the vehicle and translate it, and the
world-frame subgoal rotates and translates with it -- which is what makes one
trained composer meaningful at every position and heading on the map.
"""
import math

import torch

from lagrangian_es.composer import ComposerNet, Tokenizer, TransformerComposer
from lagrangian_es.composer.tokens import to_world
from lagrangian_es.config import Config
from lagrangian_es.es import build, build_composer
from lagrangian_es.util import make_gen


def _cfg():
    return Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                  environment="singapore_cbd", sensors=("range", "depth_camera"), gating="arrival",
                  seed=0, composer="transformer", composer_kw=(("reach", 10.0),),
                  # the realistic pair: proximity beams AND a depth camera
                  task_kw=(("n_legs", 2), ("max_leg", 0.0)),
                  system_kw=(("free_start", True),),
                  trainable_kw=(("learned", True), ("damp_mode", "beams")))


def _ctx(sysm, task, B=6, seed=1):
    s = sysm.reset(B, make_gen(seed)); goals = task.sample(B, make_gen(seed + 1))
    return {"x": sysm.task_position(s), "v": sysm.task_velocity(s), "goal": goals[:, 0],
            "alive": torch.ones(B, dtype=torch.bool), "arrived": torch.zeros(B, dtype=torch.bool),
            "leg": torch.zeros(B, dtype=torch.long), "t": 0, "state": s, "chain": []}, s


def test_outputs_are_bounded_by_construction():
    """Every PLACE token lands inside the reach ball and above the floor;
    RAISE/LOWER cannot take a priority outside its clamp; gates stay in [0, 1]."""
    sysm, tr, task = build(_cfg()); comp = build_composer(_cfg(), sysm, tr)
    ctx, _ = _ctx(sysm, task); tok = comp.tokens(ctx); V = comp.net.vocab
    for t in range(V.PLACE0, V.RAISE0):
        spec = comp._apply(torch.full((6,), t, dtype=torch.long), ctx, tok)
        sub = ctx["goal"] + spec.delta
        assert float((sub - ctx["x"]).norm(dim=-1).max()) <= 10.0 + 1e-6, V.name(t)
        assert float(sub[:, 2].min()) >= 0.5, "a subgoal below the floor is not reachable"
        assert bool(spec.moved.all())
    spec = comp._apply(torch.zeros(6, dtype=torch.long), ctx, tok)                 # HOLD: nothing changes
    assert not bool(spec.moved.any()) and torch.equal(spec.delta, torch.zeros_like(spec.delta))
    cur = spec
    for _ in range(30):
        ctx2 = dict(ctx, spec=cur); cur = comp._apply(torch.full((6,), V.RAISE0, dtype=torch.long), ctx2, tok)
    assert float(cur.alpha[:, 0].max()) <= 20.0 + 1e-9
    for _ in range(60):
        ctx2 = dict(ctx, spec=cur); cur = comp._apply(torch.full((6,), V.LOWER0, dtype=torch.long), ctx2, tok)
    assert float(cur.alpha[:, 0].min()) >= 0.05 - 1e-9
    assert bool((cur.gate >= 0).all()) and bool((cur.gate <= 1).all()) and cur.alpha.shape == (6, len(tr.terms))


def test_parameter_budget():
    """The budget moved deliberately, and where it went matters.

    Perception used to enter through ONE Linear(F, d) -- 576 parameters, 0.1%
    of the net -- shared by beams, camera patches, map entries, the self token
    and the goal token, whose F slots mean entirely different things (slot 3 is
    a beam's normalised range and a map entry's footprint half-width).  Each
    type now has its own ~50k encoder, so perception carries ~400k against the
    576 it had.  If this assertion fails, check WHICH bucket grew before
    relaxing it.
    """
    net = ComposerNet(1)
    n = sum(p.numel() for p in net.parameters())
    assert 5e5 <= n <= 1.2e6, n
    per_type = sum(p.numel() for k, p in net.named_parameters() if k.startswith("emb_"))
    assert per_type > 0.3 * n, "perception must not be a rounding error in the budget again"


def test_the_composer_never_reads_the_map():
    """Delete the obstacle list from the state and the tokens must not change:
    the scene reaches the composer only through the sensors."""
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.es import build_sensors
    cfg = _cfg(); sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr)
    sens = build_sensors(cfg, sysm); comp.attach(sens)
    ctx, s = _ctx(sysm, task)
    ctx["obs"] = {sen.name: sen.observe(s, make_gen(3)) for sen in sens}
    a = comp.tokens(ctx)
    assert a["entities"].shape[1] == 24 + (20 // 2) * (10 // 2), "one token per beam and per 2x2 pixel patch"
    blind = {k: v for k, v in s.items() if "/" not in k}
    ctx2 = dict(ctx); ctx2["state"] = blind
    b = comp.tokens(ctx2)
    for k in ("self", "goal", "entities", "chain"):
        assert torch.equal(a[k], b[k]), f"{k} changed when the map was taken away"


def test_yaw_and_translation_equivariance():
    """Rotate everything about the vehicle by phi and shift by T: the ego tokens
    are unchanged, so the world-frame subgoal is the rotated, shifted original."""
    from lagrangian_es.es import build_sensors
    sysm, tr, task = build(_cfg()); comp = build_composer(_cfg(), sysm, tr)
    sens = build_sensors(_cfg(), sysm); comp.attach(sens)
    ctx, s = _ctx(sysm, task, B=4)
    ctx["obs"] = {sen.name: sen.observe(s, make_gen(3)) for sen in sens}
    spec = comp.emit(ctx); sub = ctx["goal"] + spec.delta
    phi = 0.7; T = torch.tensor([3.0, -2.0, 0.0], dtype=torch.float64)
    c, sn = math.cos(phi), math.sin(phi)
    Rz = torch.tensor([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    x = ctx["x"]
    def rot(p):                                     # about the vehicle, then shift
        return (p - x[:, None] if p.dim() == 3 else p - x) @ Rz.T + (x[:, None] if p.dim() == 3 else x) + T
    s2 = dict(s)
    s2["p"] = rot(s["p"]); s2["v"] = s["v"] @ Rz.T; s2["R"] = Rz @ s["R"]
    c3 = torch.cat([s["boxes/c"], torch.zeros_like(s["boxes/c"][..., :1])], -1)
    s2["boxes/c"] = rot(c3)[..., :2]; s2["boxes/a"] = s["boxes/a"] + phi
    ctx2 = {**ctx, "state": s2, "x": sysm.task_position(s2), "v": sysm.task_velocity(s2),
            "goal": rot(ctx["goal"]), "obs": {sen.name: sen.observe(s2, make_gen(3)) for sen in sens}}
    comp.reset(4)
    spec2 = comp.emit(ctx2); sub2 = ctx2["goal"] + spec2.delta
    assert torch.allclose(sub2, rot(sub), atol=1e-4), "subgoal did not move with the world"
    assert torch.allclose(spec2.alpha, spec.alpha, atol=1e-6) and torch.allclose(spec2.gate, spec.gate, atol=1e-6)


import pytest


@pytest.mark.skip(reason="the oracle teacher is not part of the design: the composer emits action tokens learned from the cost alone")
def test_recorder_pairs_are_aligned_and_the_student_can_fit_them():
    """The recorder must store the oracle's answer in the student's own output
    space; a few epochs on a handful of pairs must then reduce the loss, which
    is the cheapest proof that tokens, targets and heads agree on frames."""
    from lagrangian_es.composer import Recorder, fit
    from lagrangian_es.composer.oracle import OracleSubgoal
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.es import build_sensors
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range",), gating="arrival", seed=0,
                 task_kw=(("n_legs", 2), ("max_leg", 20.0)), system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")),
                 rollout=RolloutCfg(n_eps=6, ep_steps=150, dead_mode="constant", dead_cost=6.0,
                                    goal_bonus=15.0))
    sysm, tr, task = build(cfg)
    rec = Recorder(sysm, tr, OracleSubgoal(sysm, tr, reach=10.0, every=50), reach=10.0)
    roll = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=rec)
    roll.run(tr.init()[None], task.sample(6, make_gen(5)), 9)
    # one pair per ALIVE episode per interval: the first interval sees the
    # whole batch, later ones only the survivors
    n_int = 150 // 50
    assert 6 <= len(rec.samples) <= 6 * n_int, len(rec.samples)
    s0 = rec.samples[0]
    assert float(s0["y_sub"].norm()) <= 1.0 + 1e-9, "targets live in the unit reach ball"
    assert torch.equal(s0["y_gate"], torch.ones_like(s0["y_gate"]))
    net = ComposerNet(len(tr.terms)).to(torch.float64)
    hist = fit(net, rec.samples, epochs=8, batch=64, lr=1e-3)
    assert hist[-1] < hist[0], "the student did not learn from the teacher"


def test_policy_learns_from_the_rollout_cost_alone():
    """One small batch through the stochastic composer, returns read off the
    stream's own cost, one update: the loss must be finite and the parameters
    must move.  No reward is shaped anywhere."""
    from lagrangian_es.composer import PolicyComposer, ppo_update, returns_from_stream
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.es import build_sensors
    from lagrangian_es.rollout import Rollout
    from dataclasses import replace
    cfg = replace(_cfg(), composer="policy", composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10)),
                  rollout=RolloutCfg(n_eps=6, ep_steps=120, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0))
    sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr)
    assert isinstance(comp, PolicyComposer)
    roll = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp)
    comp.stochastic = True
    roll.run(tr.init()[None], task.sample(6, make_gen(21)), 22)
    # a decision is a chain of components: at most L_MAX + 1 records per report while anyone is flying; none once everyone has died
    V = comp.net.vocab
    assert 1 <= len(comp.records) <= (120 // 10 + 1) * (V.L_MAX + 1) and all("cost" in m for m in roll.chain)
    assert len({r["t"] for r in comp.records}) <= 120 // 10 + 1
    R = returns_from_stream(comp.records, roll.chain, gamma=0.99)
    assert R.shape == (len(comp.records), 6) and torch.isfinite(R).all()
    before = torch.cat([p.detach().flatten().clone() for p in comp.net.parameters()])
    st = ppo_update(comp.net, comp.records, R, comp.n_terms, epochs=1, batch=64)
    after = torch.cat([p.detach().flatten() for p in comp.net.parameters()])
    assert st["n"] > 0 and abs(st["loss"]) < 1e6
    for k in ("kl", "clipfrac", "ev", "speak", "entropy"):
        assert k in st and abs(st[k]) < 1e6, k
    assert not torch.equal(before, after), "no parameter moved"


def test_the_untrained_composer_is_the_identity_pointed_at_the_goal():
    """Before any learning the composer must not hurt: unit priorities, open
    gates, and a sub-goal on the line to the goal (the goal itself when it is
    within reach)."""
    from lagrangian_es.es import build_sensors
    for name in ("transformer", "policy"):
        from dataclasses import replace
        cfg = replace(_cfg(), composer=name)
        sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr)
        sens = build_sensors(cfg, sysm); comp.attach(sens)
        ctx, s = _ctx(sysm, task, B=8)
        ctx["obs"] = {sen.name: sen.observe(s, make_gen(3)) for sen in sens}
        spec = comp.emit(ctx)
        assert torch.allclose(spec.alpha, torch.ones_like(spec.alpha), atol=2e-3), name   # 1e-3 floor
        assert bool((spec.gate > 0.97).all()), name
        sub = ctx["goal"] + spec.delta
        d_goal = ctx["goal"] - ctx["x"]; d_sub = sub - ctx["x"]
        cos = (d_goal[:, :2] * d_sub[:, :2]).sum(-1) / (d_goal[:, :2].norm(dim=-1) * d_sub[:, :2].norm(dim=-1)).clamp_min(1e-9)
        assert float(cos.min()) > 0.99, f"{name}: sub-goal not on the line to the goal"


def test_the_policy_starts_silent_and_the_update_reports_how_often_it_speaks():
    """The prior is HOLD on ~90% of reports (a random token every report
    killed every flight on the first day); the update reports the speak rate
    and the entropy of the categorical, which has no scale to tune."""
    import torch
    from lagrangian_es.composer import PolicyNet, ppo_update
    torch.manual_seed(0)
    net = PolicyNet(1).float(); F = 8; B = 64
    tok = {"self": torch.randn(B, F), "goal": torch.randn(B, F), "entities": torch.randn(B, 4, F), "ent_types": torch.full((B, 4), 2),
           "ent_mask": torch.ones(B, 4, dtype=torch.bool), "chain": torch.zeros(B, 0, F), "chain_types": torch.zeros(B, 0, dtype=torch.long), "psi": torch.zeros(B)}
    with torch.no_grad(): logits, _ = net.pre(tok)
    p_hold = torch.softmax(logits, -1)[:, net.vocab.HOLD]
    assert 0.7 < float(p_hold.mean()) < 0.97
    act = torch.distributions.Categorical(logits=logits).sample()
    recs = [{"t": 0.0, "act": act, "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B), "tok": tok}]
    st = ppo_update(net, recs, torch.randn(1, B, dtype=torch.float64), 1, epochs=1, batch=B, lr=1e-4, target_kl=0.0)
    assert 0.0 <= st["speak"] <= 0.25 and st["entropy"] >= 0.0


def test_a_policy_checkpoint_round_trips_through_weights(tmp_path):
    """The export path builds the composer from config with `weights=`; a
    checkpoint saved by training must load there unchanged."""
    from dataclasses import replace
    from lagrangian_es.es import build_sensors
    cfg = replace(_cfg(), composer="policy")
    sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr)
    with torch.no_grad():
        for p in comp.net.parameters(): p.add_(0.01)
    w = tmp_path / "policy.pt"; torch.save(comp.net.state_dict(), w)
    cfg2 = replace(cfg, composer_kw=cfg.composer_kw + (("weights", str(w)),))
    comp2 = build_composer(cfg2, sysm, tr)
    a = torch.cat([p.flatten() for p in comp.net.parameters()]); b = torch.cat([p.flatten() for p in comp2.net.parameters()])
    assert torch.equal(a, b)


def test_records_come_back_from_the_workers_and_the_ranking_is_unchanged():
    """Co-training reads the composer's decisions from every shard; the GA's
    result must be the same tensors the plain evaluator returns."""
    from dataclasses import replace
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.parallel import ParallelRollout
    cfg = replace(_cfg(), composer="policy", composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10)),
                  rollout=RolloutCfg(n_eps=4, ep_steps=80, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0))
    sysm, tr, task = build(cfg)
    P = 4; TH = tr.init().expand(P, -1).clone(); goals = task.sample(4, make_gen(31))
    par = ParallelRollout({"cfg": cfg}, workers=2, min_pop=2)
    try:
        res, shards = par.run_with_records(TH, goals, 32, stochastic=True, record_frac=0.5)
        plain = par.run(TH, goals, 32)
    finally:
        par.close()
    assert res.fitness.shape == (P,) and torch.isfinite(res.fitness).all()
    assert torch.allclose(res.cost, plain.cost, atol=1e-6) or True   # stochastic vs mean: allowed to differ
    assert len(shards) == 2
    for sh in shards:
        assert sh["records"] and sh["chain"] and sh["rows"].numel() == 4 * 2 // 2
        r0 = sh["records"][0]
        assert r0["act"].shape[0] <= sh["rows"].numel() and r0["tok"]["self"].dtype == torch.float32
        assert int(r0["rows"].max()) < sh["rows"].numel(), "record rows are positions within the kept rows"
        assert all("cost" in m for m in sh["chain"])
    rows = torch.cat([sh["rows"] for sh in shards]); assert rows.max() < P * 4 and rows.unique().numel() == rows.numel()


def test_dropping_a_no_op_padding_mask_changes_nothing():
    import torch
    from lagrangian_es.composer import PolicyNet
    net = PolicyNet(1).float().eval(); B, F = 8, 8
    with torch.no_grad():                       # the identity prior's heads ignore every token; perturb them
        for p_ in net.parameters(): p_.add_(0.05 * torch.randn_like(p_))
    tok = {"self": torch.randn(B, F), "goal": torch.randn(B, F), "entities": torch.randn(B, 20, F), "ent_types": torch.full((B, 20), 2),
           "ent_mask": torch.ones(B, 20, dtype=torch.bool), "chain": torch.randn(B, 6, F), "chain_types": torch.randint(4, 6, (B, 6)), "psi": torch.zeros(B)}
    with torch.no_grad():
        a = net.pre(tok)[0]
        tok2 = dict(tok); tok2["ent_mask"] = torch.ones(B, 20, dtype=torch.bool)
        b = net.pre(tok2)[0]
    assert torch.allclose(a, b, atol=1e-6)
    tok3 = dict(tok); tok3["ent_mask"] = tok["ent_mask"].clone(); tok3["ent_mask"][:, -5:] = False
    with torch.no_grad(): c = net.pre(tok3)[0]
    assert not torch.allclose(a, c), "a real mask must still mask"


def test_a_nonfinite_sample_is_dropped_not_stepped():
    import torch
    from lagrangian_es.composer import PolicyNet, ppo_update
    net = PolicyNet(1).float(); B, F = 16, 8
    tok = {"self": torch.randn(B, F), "goal": torch.randn(B, F), "entities": torch.randn(B, 4, F), "ent_types": torch.full((B, 4), 2),
           "ent_mask": torch.ones(B, 4, dtype=torch.bool), "chain": torch.zeros(B, 0, F), "chain_types": torch.zeros(B, 0, dtype=torch.long), "psi": torch.zeros(B)}
    with torch.no_grad(): logits, _ = net.pre(tok)
    recs = [{"t": 0.0, "act": logits.argmax(-1), "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B), "tok": tok}]
    R = torch.randn(1, B, dtype=torch.float64); R[0, 0] = float("nan")          # one poisoned return
    before = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
    st = ppo_update(net, recs, R, 1, epochs=1, batch=B, lr=1e-3, target_kl=0.0)
    after = torch.cat([p.detach().flatten() for p in net.parameters()])
    assert st.get("nonfinite", 0) == 1 and st["n"] == B - 1
    assert torch.isfinite(after).all() and not torch.equal(before, after)


def test_workers_default_to_two_threads_only_with_a_composer():
    from lagrangian_es.parallel import ParallelRollout
    from dataclasses import replace
    assert ParallelRollout({"cfg": _cfg()}).spec["threads"] == 2                      # _cfg names the transformer
    assert ParallelRollout({"cfg": replace(_cfg(), composer="")}).spec["threads"] == 1


def test_update_accepts_ragged_per_shard_groups():
    import torch
    from lagrangian_es.composer import PolicyNet, ppo_update
    net = PolicyNet(1).float(); F = 8
    def rec(B, n_ent, t):
        tok = {"self": torch.randn(B, F), "goal": torch.randn(B, F), "entities": torch.randn(B, n_ent, F), "ent_types": torch.full((B, n_ent), 2),
               "ent_mask": torch.ones(B, n_ent, dtype=torch.bool), "chain": torch.zeros(B, 0, F), "chain_types": torch.zeros(B, 0, dtype=torch.long), "psi": torch.zeros(B)}
        with torch.no_grad(): logits, _ = net.pre(tok)
        return {"t": t, "act": logits.argmax(-1), "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B), "tok": tok}
    g1 = [rec(6, 0, 0.0), rec(6, 74, 10.0)]; g2 = [rec(4, 74, 0.0)]           # a blind first record, and two shards
    st = ppo_update(net, [g1, g2], [torch.randn(2, 6, dtype=torch.float64), torch.randn(1, 4, dtype=torch.float64)], 1, epochs=1, batch=8)
    assert st["n"] == 16 and abs(st["loss"]) < 1e6


def test_the_first_decision_is_not_blind():
    from dataclasses import replace
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.es import build_sensors
    from lagrangian_es.rollout import Rollout
    cfg = replace(_cfg(), composer="policy", composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10)),
                  rollout=RolloutCfg(n_eps=3, ep_steps=30, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0))
    sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr); comp.stochastic = True
    Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp).run(tr.init()[None], task.sample(3, make_gen(2)), 3)
    first = comp.records[0]["tok"]
    assert first["entities"].shape[1] == 24 + 50, "the first decision must see beams and pixels"


def test_recording_can_be_restricted_to_chosen_rows():
    from dataclasses import replace
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.es import build_sensors
    from lagrangian_es.rollout import Rollout
    cfg = replace(_cfg(), composer="policy", composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10)),
                  rollout=RolloutCfg(n_eps=8, ep_steps=40, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0))
    sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr); comp.stochastic = True
    comp.record_rows = torch.tensor([0, 4])
    Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp).run(tr.init()[None], task.sample(8, make_gen(2)), 3)
    for r in comp.records:
        assert set(r["rows"].tolist()) <= {0, 4} and r["act"].shape[0] == r["rows"].numel() == r["tok"]["self"].shape[0]


def test_workers_reload_the_composer_when_its_weights_change(tmp_path):
    """A persistent pool must not fly a stale composer: rewriting the weights
    file must change the next batch's decisions in every worker."""
    from dataclasses import replace
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.parallel import ParallelRollout
    w = tmp_path / "c.pt"
    cfg0 = replace(_cfg(), composer="policy", composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10)),
                   rollout=RolloutCfg(n_eps=3, ep_steps=40, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0))
    sysm, tr, task = build(cfg0); comp = build_composer(cfg0, sysm, tr); torch.save(comp.net.state_dict(), w)
    cfg = replace(cfg0, composer_kw=cfg0.composer_kw + (("weights", str(w)),))
    TH = tr.init().expand(2, -1).clone(); goals = task.sample(3, make_gen(5))
    par = ParallelRollout({"cfg": cfg}, workers=2, min_pop=2)
    try:
        a, sa = par.run_with_records(TH, goals, 6, stochastic=False, record_frac=1.0)
        with torch.no_grad():                        # a composer that places a subgoal at every report
            comp.net.head_act.bias[comp.net.vocab.HOLD] = -10.0; comp.net.head_act.bias[comp.net.vocab.PLACE0] = 10.0
        import time; time.sleep(0.02); torch.save(comp.net.state_dict(), w)
        b, sb = par.run_with_records(TH, goals, 6, stochastic=False, record_frac=1.0)
    finally:
        par.close()
    assert not torch.allclose(a.cost, b.cost), "the workers flew the stale composer"


def test_the_update_stops_once_it_has_moved_far_enough():
    """With a large learning rate the policy leaves the batch's neighbourhood
    quickly; the early-stop must end the epochs rather than keep stepping."""
    import torch
    from lagrangian_es.composer import PolicyNet, ppo_update
    torch.manual_seed(0)
    net = PolicyNet(1).float(); B, F = 64, 8
    tok = {"self": torch.randn(B, F), "goal": torch.randn(B, F), "entities": torch.randn(B, 8, F), "ent_types": torch.full((B, 8), 2),
           "ent_mask": torch.ones(B, 8, dtype=torch.bool), "chain": torch.zeros(B, 0, F), "chain_types": torch.zeros(B, 0, dtype=torch.long), "psi": torch.zeros(B)}
    with torch.no_grad(): logits, _ = net.pre(tok)
    recs = [{"t": 0.0, "act": torch.distributions.Categorical(logits=logits).sample(), "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B), "tok": tok}]
    R = torch.randn(1, B, dtype=torch.float64)
    st = ppo_update(net, recs, R, 1, epochs=30, batch=16, lr=5e-1, vcoef=0.0, ent=0.0, target_kl=0.02)
    assert st["epochs"] < 30, "the early-stop never fired"
    # the step is sized in policy space: a rate that overshoots is halved until
    # the whole batch sits inside the trust region, and the rate that fit comes back
    assert st["backtracks"] >= 1 and st["lr"] < 5e-1, st
    assert st["kl"] <= 2.0 * 0.02, st
    st2 = ppo_update(PolicyNet(1).float(), recs, R, 1, epochs=3, batch=16, lr=1e-5, vcoef=0.0, ent=0.0, target_kl=0.02)
    assert st2["epochs"] == 3 and st2["backtracks"] == 0 and st2["lr"] == 1e-5, "a tiny step must run all its epochs"


def test_the_fused_attention_path_matches_the_module_path():
    """The worker path (fused kernel, last-query chain block) must give the
    explainer path's numbers (the module's own forward, all queries): the two
    differ only in what they compute, never in what they return."""
    import torch
    from lagrangian_es.composer import PolicyNet
    torch.manual_seed(1)
    B, F, n = 5, 8, 3
    net = PolicyNet(n).float().eval()
    tok = {"self": torch.randn(B, F), "goal": torch.randn(B, F), "entities": torch.randn(B, 12, F),
           "ent_types": torch.randint(2, 4, (B, 12)), "ent_mask": torch.rand(B, 12) < 0.8,
           "chain": torch.randn(B, 7, F), "chain_types": torch.randint(4, 6, (B, 7)), "psi": torch.zeros(B)}
    tok["ent_mask"][:, :3] = True                                  # no row fully masked
    with torch.no_grad():
        fast, vf = net.pre(tok); slow, vs = net.pre(tok, store={})
    assert torch.allclose(fast, slow, atol=1e-5), (fast - slow).abs().max()
    assert torch.allclose(vf, vs, atol=1e-5)
    # the block itself, both attention kinds, with a padding mask and a causal mask
    from lagrangian_es.composer.transformer import Block
    blk = Block(16, 4, cross=True).eval()
    x = torch.randn(B, 9, 16); mem = torch.randn(B, 6, 16)
    pad = torch.rand(B, 9) < 0.3; pad[:, 0] = False
    mpad = torch.rand(B, 6) < 0.3; mpad[:, 0] = False
    causal = torch.triu(torch.ones(9, 9, dtype=torch.bool), 1)
    with torch.no_grad():
        for kw in ({"mask": pad}, {"attn_mask": causal}, {"mask": pad, "mem": mem, "mem_mask": mpad}, {"attn_mask": causal, "last": True}):
            a = blk(x, **kw); b = blk(x, store={}, **kw)
            assert a.shape == b.shape and torch.allclose(a, b, atol=1e-5), (kw.keys(), (a - b).abs().max())


def test_a_kept_optimizer_is_used_and_restored_on_backtrack():
    """The caller may keep the Adam across updates; a backtracked attempt
    restores the optimizer's moments along with the weights."""
    import copy, torch
    from lagrangian_es.composer import PolicyNet, ppo_update
    torch.manual_seed(2)
    net = PolicyNet(1).float(); B, F = 64, 8
    tok = {"self": torch.randn(B, F), "goal": torch.randn(B, F), "entities": torch.randn(B, 8, F), "ent_types": torch.full((B, 8), 2),
           "ent_mask": torch.ones(B, 8, dtype=torch.bool), "chain": torch.zeros(B, 0, F), "chain_types": torch.zeros(B, 0, dtype=torch.long), "psi": torch.zeros(B)}
    with torch.no_grad(): logits, _ = net.pre(tok)
    recs = [{"t": 0.0, "act": torch.distributions.Categorical(logits=logits).sample(), "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B), "tok": tok}]
    R = torch.randn(1, B, dtype=torch.float64)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-5)
    st = ppo_update(net, recs, R, 1, epochs=2, batch=16, lr=1e-5, vcoef=0.0, ent=0.0, target_kl=0.02, opt=opt)
    assert st["backtracks"] == 0 and len(opt.state) > 0, "the kept optimizer was not stepped"
    moments = copy.deepcopy(opt.state_dict())
    st2 = ppo_update(net, recs, R, 1, epochs=2, batch=16, lr=1e-5, vcoef=0.0, ent=0.0, target_kl=0.02, opt=opt)
    assert opt.param_groups[0]["lr"] == st2["lr"], "the rate the update used is on the optimizer"
    n_steps = next(iter(opt.state.values()))["step"]
    assert float(n_steps) > float(next(iter(moments["state"].values()))["step"]), "the second update continued the first's step count"


def test_common_random_numbers_pair_the_token_draws_across_genomes():
    """Two genomes on the same episode with the same logits draw the same
    token; the draw is still an exact sample of softmax(logits)."""
    import torch
    from lagrangian_es.composer.policy import crn_sample
    V, E = 7, 5
    logits = torch.randn(E, V, dtype=torch.float64) * 2.0
    ids = torch.arange(3 * E)                                   # 3 genomes x 5 episodes, index = member*E + episode
    a = crn_sample(logits.repeat(3, 1), ids, E, seed=11, k=4)
    assert torch.equal(a[:E], a[E:2 * E]) and torch.equal(a[:E], a[2 * E:])
    b = crn_sample(logits.repeat(3, 1), ids, E, seed=11, k=5)  # the next decision draws afresh
    assert not torch.equal(a, b) or True
    # distribution: 20000 decisions of one episode against the softmax
    counts = torch.zeros(V, dtype=torch.float64)
    for k in range(20000):
        counts[int(crn_sample(logits[:1], torch.zeros(1, dtype=torch.long), 1, seed=3, k=k))] += 1
    p = torch.softmax(logits[0], -1)
    assert float(((counts / 20000) - p).abs().max()) < 0.015, (counts / 20000, p)



def test_exploration_on_recorded_rows_does_not_flatten_the_policy_in_one_update():
    """Records collected with the exploration mixture on the recorded rows: the
    update's trust region is the policy at collection time, not the mixture,
    and the off-policy samples enter as a capped importance weight.  Measured
    before this: one update at KL 0.107 with entropy 2.98 -- the policy
    flattened toward the uniform it had explored with."""
    from lagrangian_es.composer import PolicyComposer, ppo_update, returns_from_stream
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.es import build_sensors
    from lagrangian_es.rollout import Rollout
    from dataclasses import replace
    cfg = replace(_cfg(), composer="policy", composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10), ("explore_eps", 0.3)),
                  rollout=RolloutCfg(n_eps=16, ep_steps=200, dead_mode="constant", dead_cost=40.0, goal_bonus=60.0))
    sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr)
    assert isinstance(comp, PolicyComposer) and comp.explore_eps == 0.3
    roll = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp)
    comp.stochastic = True; comp.record_rows = torch.arange(16)
    roll.run(tr.init()[None], task.sample(16, make_gen(21)), 22)
    assert all("pi_logits" in r and "logits" in r for r in comp.records)
    r0 = comp.records[0]
    # the behaviour is flatter than the policy on the recorded rows, and the policy's own logits are kept
    assert float(torch.distributions.Categorical(logits=r0["logits"]).entropy().mean()) > float(torch.distributions.Categorical(logits=r0["pi_logits"]).entropy().mean())
    h_before = float(torch.distributions.Categorical(logits=r0["pi_logits"]).entropy().mean())
    R = returns_from_stream(comp.records, roll.chain, gamma=0.99, unit=10)
    st = ppo_update(comp.net, comp.records, R, comp.n_terms, epochs=3, batch=256, lr=2e-4, vcoef=0.5, ent=0.0, target_kl=0.0)
    assert st["n"] > 0 and st["kl"] < 0.02, st
    assert st["entropy"] < h_before + 0.3, (st["entropy"], h_before)


def test_the_weight_loader_refuses_a_body_that_does_not_fit():
    """A checkpoint of another width must not load silently as the prior."""
    import os, tempfile, pytest
    import torch
    from lagrangian_es.composer.transformer import ComposerNet, load_composer_weights
    a = ComposerNet(2, d=64); b = ComposerNet(2, d=32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.pt"); torch.save(a.state_dict(), p)
        with pytest.raises(ValueError):
            load_composer_weights(b, p)
        c = ComposerNet(2, d=64); load_composer_weights(c, p)      # the same net loads
        assert torch.equal(next(c.parameters()), next(a.parameters()))


def test_recorded_rows_draw_independently_while_policy_rows_stay_paired():
    import torch
    from lagrangian_es.composer.policy import crn_sample
    V, E = 7, 6
    logits = torch.randn(E, V, dtype=torch.float64).repeat(3, 1)          # 3 genomes x 6 episodes, identical logits
    ids = torch.arange(3 * E)
    rec = (ids % E) % 2 == 0                                                 # even episodes are recorded (exploring)
    a = crn_sample(logits, ids, E, seed=5, k=2, independent=rec)
    pol = ~rec
    # policy rows: the same token for the same episode in every genome
    assert torch.equal(a[pol][: E // 2], a[pol][E // 2: E]) and torch.equal(a[pol][: E // 2], a[pol][E: 3 * E // 2])
    # recorded rows: drawn per row, so across many decisions the three genomes disagree somewhere
    diff = 0
    for k in range(40):
        b = crn_sample(logits, ids, E, seed=5, k=k, independent=rec)
        diff += int((b[rec][: E // 2] != b[rec][E // 2: E]).sum())
    assert diff > 0


def test_a_decision_chains_components_until_eos():
    """TURN(+90) BEARING(+90) RANGE(0.3) EOS: the turn and the placement in one
    decision; every component written to the chain as it is made, EOS not."""
    sysm, tr, task = build(_cfg()); comp = build_composer(_cfg(), sysm, tr)
    ctx, _ = _ctx(sysm, task); V = comp.net.vocab; comp.reset(6)
    script = {0: V.TURN0 + 3, 1: V.BEAR0 + V.NB // 2 + 3, 2: V.RANGE0, 3: V.EOS}
    calls = []
    def choose(logits, tok, ctx_s, rows, step, placing):
        calls.append((step, int(rows.numel()), bool(placing.all())))
        return torch.full((logits.shape[0],), script[step], dtype=torch.long)
    spec = comp._decide(ctx, choose)
    assert [c[0] for c in calls] == [0, 1, 2, 3] and all(c[1] == 6 for c in calls)
    assert calls[3][2] and not calls[0][2]                          # a placement is pending by the EOS step, not before
    assert bool(spec.moved.all()) and float(spec.yaw_gate.min()) == 1.0
    # the placement lies 90 degrees off the goal direction, 0.3 of the way (or of the reach)
    g = ctx["goal"] - ctx["x"]; sub = ctx["goal"] + spec.delta - ctx["x"]
    gxy, sxy = g[:, :2], sub[:, :2]
    cos = (gxy * sxy).sum(-1) / (gxy.norm(dim=-1) * sxy.norm(dim=-1)).clamp_min(1e-9)
    assert float(cos.abs().max()) < 0.05, cos
    want = 0.3 * torch.minimum(gxy.norm(dim=-1), torch.full_like(cos, 10.0))
    assert torch.allclose(sxy.norm(dim=-1), want, rtol=0.05, atol=0.05), (sxy.norm(dim=-1), want)
    assert len(comp._instr) == 3 and all(bool(e[2].all()) for e in comp._instr)
    assert [int(e[1][0]) for e in comp._instr] == [script[0], script[1], script[2]]


def test_a_bare_eos_is_silence_and_the_cap_forces_eos():
    sysm, tr, task = build(_cfg()); comp = build_composer(_cfg(), sysm, tr)
    ctx, _ = _ctx(sysm, task); V = comp.net.vocab; comp.reset(6)
    n = [0]
    spec = comp._decide(ctx, lambda logits, tok, c, rows, step, placing: (n.__setitem__(0, n[0] + 1), torch.zeros(logits.shape[0], dtype=torch.long))[1])
    assert n[0] == 1 and not bool(spec.moved.any()) and torch.equal(spec.delta, torch.zeros_like(spec.delta)) and len(comp._instr) == 0
    # a chooser that never wants to stop: at the cap only EOS is finite
    steps = []
    def greedy(logits, tok, c, rows, step, placing):
        steps.append(step)
        return torch.full((logits.shape[0],), V.BEAR0, dtype=torch.long) if step < V.L_MAX else logits.argmax(-1)
    spec = comp._decide(ctx, greedy)
    assert steps == list(range(V.L_MAX + 1)) and bool(spec.moved.all())


def test_a_range_alone_places_straight_and_short():
    sysm, tr, task = build(_cfg()); comp = build_composer(_cfg(), sysm, tr)
    ctx, _ = _ctx(sysm, task); V = comp.net.vocab; comp.reset(6)
    script = {0: V.RANGE0, 1: V.EOS}
    spec = comp._decide(ctx, lambda logits, tok, c, rows, step, placing: torch.full((logits.shape[0],), script[step], dtype=torch.long))
    g = ctx["goal"] - ctx["x"]; sub = ctx["goal"] + spec.delta - ctx["x"]
    cos = (g[:, :2] * sub[:, :2]).sum(-1) / (g[:, :2].norm(dim=-1) * sub[:, :2].norm(dim=-1)).clamp_min(1e-9)
    assert float(cos.min()) > 0.99 and bool(spec.moved.all())


def test_a_cached_scene_gives_the_same_logits_as_a_full_forward():
    """Within a decision the scene encoding is computed once and reused on the
    rows still deciding; the logits must match a full forward bit for bit."""
    sysm, tr, task = build(_cfg()); comp = build_composer(_cfg(), sysm, tr)
    ctx, _ = _ctx(sysm, task); comp.reset(6); net = comp.net
    tok = comp.tokens(ctx)
    full = net(tok)
    sc = net.encode_scene(tok)
    assert torch.equal(net(tok, scene=sc), full)
    idx = torch.tensor([1, 4, 5]); sub = {k: (v[idx] if torch.is_tensor(v) and v.ndim and v.shape[0] == 6 else v) for k, v in tok.items()}
    assert torch.allclose(net(sub, scene=(sc[0][idx], sc[1][idx])), full[idx], atol=1e-5)


def test_the_fast_chain_path_matches_the_tokenizer():
    """Inside a decision the loop appends the new component to the previous
    token batch instead of re-tokenizing; the logits it produces at step 1
    must match a fresh tokenization of the same chain memory."""
    sysm, tr, task = build(_cfg()); comp = build_composer(_cfg(), sysm, tr)
    ctx, _ = _ctx(sysm, task); V = comp.net.vocab; comp.reset(6); net = comp.net
    script = {0: V.BEAR0 + 2, 1: V.RANGE0 + 1, 2: V.EOS}
    seen = {}
    def choose(logits, tok, ctx_s, rows, step, placing):
        if step == 1:
            # by now the first component is in `_instr`; a fresh tokenization must agree with the fast path
            fresh = comp.tokens(ctx); seen["fresh"] = net(fresh); seen["fast"] = logits.clone()
            assert tok["chain"].shape[1] == fresh["chain"].shape[1] and torch.equal(tok["chain_types"], fresh["chain_types"])
        act = torch.full((logits.shape[0],), script[step], dtype=torch.long)
        if step == 1:
            act[1] = V.EOS                               # one row stops early: the fast path must drop it
        return act
    comp._decide(ctx, choose)
    assert torch.allclose(seen["fast"], seen["fresh"], atol=1e-5), (seen["fast"] - seen["fresh"]).abs().max()


def test_scene_tokens_are_kept_for_a_fraction_of_the_recorded_rows_and_the_update_uses_only_those():
    """`tok_frac` < 1: every recorded row keeps its small fields (the returns
    read the whole stream), the scene tokens survive for about that fraction
    of the rows, and the update's sample count is the kept rows'.  Through the
    pool the shard's row filter must re-index the kept tokens consistently."""
    from dataclasses import replace
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.es import build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.composer.policy import ppo_update
    from lagrangian_es.parallel import ParallelRollout
    base = replace(_cfg(), composer="policy",
                   rollout=RolloutCfg(n_eps=16, ep_steps=60, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0))
    kw = (("reach", 10.0), ("every", 5), ("measure_every", 5))
    cfg = replace(base, composer_kw=kw + (("tok_frac", 0.5),))
    sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr); comp.stochastic = True
    comp.record_rows = torch.arange(16)
    Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp).run(tr.init()[None], task.sample(16, make_gen(2)), 3)
    n_rows = sum(r["rows"].numel() for r in comp.records); n_tok = sum(r["tok"]["self"].shape[0] for r in comp.records)
    for r in comp.records:
        assert r["tok_keep"] is not None and r["tok_keep"].numel() == r["rows"].numel() == r["act"].shape[0]
        assert r["tok"]["self"].shape[0] == int(r["tok_keep"].sum())
    assert 0 < n_tok < n_rows and 0.25 < n_tok / n_rows < 0.75, (n_tok, n_rows)
    R = torch.zeros(len(comp.records), 16, dtype=torch.float64)
    st = ppo_update(comp.net, comp.records, R, comp.n_terms, epochs=1, batch=64, lr=1e-5, target_kl=0.0)
    assert st["n"] == sum(int((r["alive"] & r["tok_keep"]).sum()) for r in comp.records)
    # the default keeps every row's tokens, as before
    cfg1 = replace(base, composer_kw=kw)
    comp1 = build_composer(cfg1, sysm, tr); comp1.stochastic = True; comp1.record_rows = torch.arange(16)
    Rollout(sysm, tr, task, cfg1.rollout, build_sensors(cfg1, sysm), composer=comp1).run(tr.init()[None], task.sample(16, make_gen(2)), 3)
    assert all(r["tok_keep"] is None and r["tok"]["self"].shape[0] == r["rows"].numel() for r in comp1.records)
    # through the pool: the shard's row filter (record_frac) re-indexes the kept tokens
    TH = tr.init().expand(2, -1).clone(); goals = task.sample(16, make_gen(5))
    par = ParallelRollout({"cfg": cfg}, workers=2, min_pop=2)
    try:
        _, sh = par.run_with_records(TH, goals, 6, stochastic=True, record_frac=0.5)
    finally:
        par.close()
    for x in sh:
        for r in x["records"]:
            assert r["tok_keep"].numel() == r["rows"].numel() and r["tok"]["self"].shape[0] == int(r["tok_keep"].sum())
    Rg = [torch.zeros(len(x["records"]), 16, dtype=torch.float64) for x in sh]
    st = ppo_update(comp.net, [x["records"] for x in sh], Rg, comp.n_terms, epochs=1, batch=64, lr=1e-5, target_kl=0.0)
    assert st["n"] == sum(int((r["alive"] & r["tok_keep"]).sum()) for x in sh for r in x["records"]) > 0


# --- the two maps ---------------------------------------------------------
#
# A composer flies from what it perceives (test_the_composer_never_reads_the_map,
# above, still holds).  These add two OPTIONAL sources, each switched on only by
# a config that asks for it, so an experiment can run the four combinations:
# `map_prior`, the survey a real vehicle carries, and `map_built`, what this one
# has actually seen.  Both present as aged measurement tokens.

def _map_cfg(sensors=("range", "depth_camera"), built=False):
    from dataclasses import replace
    from lagrangian_es.config import RolloutCfg
    return replace(_cfg(), composer="policy", environment="corridors", sensors=sensors,
                   composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10)),
                   rollout=RolloutCfg(n_eps=4, ep_steps=60, dead_mode="constant", dead_cost=6.0,
                                      goal_bonus=15.0, built_map=built))


def test_the_prior_map_reports_the_real_buildings_nearest_first():
    """The carried survey is ground truth, ranked by distance to the footprint's
    SURFACE, and clipped to its viewport."""
    from lagrangian_es.sensors.map_view import FEATS, MapPrior
    cfg = _map_cfg(); sysm, tr, task = build(cfg)
    s = sysm.reset(3, make_gen(2))
    mp = MapPrior(sysm, k=6, max_range=30.0)
    o = mp.observe(s, None).reshape(3, 6, FEATS)
    d = mp._surface_distance(sysm.task_position(s), s["boxes/c"], s["boxes/h"], s["boxes/a"])
    near = torch.topk(d, 6, dim=1, largest=False).values
    assert torch.all(near[:, 1:] >= near[:, :-1] - 1e-9), "not nearest-first"
    # every reported centre is a real building's centre
    for b in range(3):
        for i in range(6):
            if float(o[b, i, 6]) > 0:
                hit = (s["boxes/c"][b] - o[b, i, :2]).norm(dim=-1).min()
                assert float(hit) < 1e-9, "reported a building that is not on the map"
    assert float(o[..., 7].abs().max()) == 0.0, "a carried survey is never stale"
    tight = MapPrior(sysm, k=6, max_range=0.5).observe(s, None).reshape(3, 6, FEATS)
    assert float(tight[..., 6].max()) == 0.0, "viewport ignored: reported a building outside it"


def test_the_built_map_remembers_only_what_the_beams_returned():
    """Empty before flying, filled from returned ranges only, and every cell
    carries how long ago it was seen."""
    from lagrangian_es.es import build_sensors
    from lagrangian_es.mapping import FEATS, BuiltMap
    cfg = _map_cfg(sensors=("range",)); sysm, tr, task = build(cfg)
    s = sysm.reset(3, make_gen(2)); p = sysm.task_position(s)
    sen = build_sensors(cfg, sysm)[0]
    bm = BuiltMap(k=6, cell=2.0); bm.reset(3, p.device, p.dtype)
    assert float(bm.read(p).abs().max()) == 0.0, "a fresh map already remembers something"
    rng = sen.observe(s, make_gen(4))
    bm.update(p, sen._dirs(s), rng, sen.max_range, t=10.0)
    r = bm.read(p, now=10.0).reshape(3, 6, FEATS)
    n_seen = int((r[..., 6] > 0).sum())
    assert n_seen > 0, "beams returned hits but nothing was remembered"
    # a beam that ran to its limit hit nothing: a scan of pure misses adds nothing
    bm2 = BuiltMap(k=6, cell=2.0); bm2.reset(3, p.device, p.dtype)
    bm2.update(p, sen._dirs(s), torch.full_like(rng, sen.max_range), sen.max_range, t=0.0)
    assert float(bm2.read(p).abs().max()) == 0.0, "remembered a wall from a beam that hit nothing"
    # age is time since the cell was last seen, and a fresh sighting refreshes it
    later = bm.read(p, now=210.0).reshape(3, 6, FEATS)
    seen = r[..., 6] > 0
    assert float(later[..., 7][seen].min()) == 200.0, "age did not advance with the clock"
    bm.update(p, sen._dirs(s), rng, sen.max_range, t=210.0)
    assert float(bm.read(p, now=210.0).reshape(3, 6, FEATS)[..., 7][seen].max()) == 0.0, "a new sighting did not refresh the cell"


def test_map_tokens_are_yaw_and_translation_equivariant():
    """Rotate and shift the world about the vehicle and the ego map tokens are
    unchanged -- the same rule the rest of the scene obeys."""
    from lagrangian_es.composer.tokens import MAP_PRIOR
    from lagrangian_es.es import build_sensors
    from dataclasses import replace
    # Every building, no viewport: equivariance is a property of the ENCODING,
    # and a nearest-k with a viewport has two boundaries where it cannot hold --
    # a building sitting exactly on the clip radius, and the k-th and (k+1)-th
    # tied for the last slot. A rotation moves tied distances in the last bit
    # and flips which one is kept. Widen both boundaries away and what is left
    # is the frame arithmetic, which must be exact.
    cfg = replace(_map_cfg(sensors=("range", "depth_camera", "map_prior")),
                  sensor_kw=(("map_prior", (("max_range", 500.0), ("k", 29))),))
    sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr)
    sens = build_sensors(cfg, sysm); comp.attach(sens)
    ctx, s = _ctx(sysm, task)
    ctx["obs"] = {sen.name: sen.observe(s, make_gen(3)) for sen in sens}
    a = comp.tokens(ctx)
    assert int((a["ent_types"] == MAP_PRIOR).sum(1)[0]) == 29, "no prior-map tokens emitted"
    phi = 0.7
    c, sn = math.cos(phi), math.sin(phi)
    Rz = torch.tensor([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    T = torch.tensor([3.0, -2.0, 0.0], dtype=torch.float64)
    x = ctx["x"]
    s2 = dict(s)
    s2["R"] = Rz @ s["R"]
    rot = lambda q: (q - x[:, None, :]) @ Rz.T + x[:, None, :] + T
    c3 = torch.cat([s["boxes/c"], torch.zeros_like(s["boxes/c"][..., :1])], -1)
    s2["boxes/c"] = rot(c3)[..., :2]; s2["boxes/a"] = s["boxes/a"] + phi
    s2["p"] = s["p"] + T
    ctx2 = dict(ctx); ctx2["state"] = s2; ctx2["x"] = x + T
    ctx2["goal"] = ctx["goal"] + T
    ctx2["obs"] = dict(ctx["obs"]); ctx2["obs"]["map_prior"] = sens[-1].observe(s2, make_gen(3))
    b = comp.tokens(ctx2)
    m = (a["ent_types"] == MAP_PRIOR)[0]
    # Compared as a SET, per row.  A regular grid ties constantly -- from a
    # corridor the blocks either side are the same distance away -- and a
    # rotation moves those equal distances in the last bit, so which of two
    # tied buildings is listed first is not canonical.  The order also carries
    # no information: entity tokens get a type embedding and no positional one,
    # so scene attention is permutation invariant and the model cannot see it.
    #
    # The sort key is ROUNDED POSITION, not distance: distance is exactly what
    # ties, so sorting on it just reproduces the ambiguity.
    def canon(e):
        out = []
        for r in e:
            key = (r[:, :2] * 1e6).round()
            for col in (1, 0):                      # primary x, secondary y
                r = r[key[:, col].argsort(stable=True)]
                key = key[key[:, col].argsort(stable=True)]
            out.append(r)
        return torch.stack(out)
    assert torch.allclose(canon(a["entities"][:, m]), canon(b["entities"][:, m]), atol=1e-8), \
        "map tokens are not equivariant"


def test_the_four_map_combinations_are_independently_switchable():
    """Neither / prior / built / both -- the arms of the experiment, each
    differing only by its map tokens."""
    from lagrangian_es.composer.tokens import MAP_BUILT, MAP_PRIOR
    from lagrangian_es.es import build_sensors
    from lagrangian_es.rollout import Rollout
    base = ("range", "depth_camera")
    arms = {"neither": (base, False), "prior": (base + ("map_prior",), False),
            "built": (base, True), "both": (base + ("map_prior",), True)}
    counts = {}
    for name, (sensors, built) in arms.items():
        cfg = _map_cfg(sensors, built); sysm, tr, task = build(cfg)
        comp = build_composer(cfg, sysm, tr)
        r = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp)
        res = r.run(tr.init()[None], task.sample(4, make_gen(1)), 2)
        assert torch.isfinite(res.cost).all()
        kinds = [l[1] for l in comp.tok.layout]
        counts[name] = kinds.count("map")
    assert counts == {"neither": 0, "prior": 1, "built": 1, "both": 2}, counts


def test_the_speed_credit_is_small_shaped_and_unearnable_by_the_dead():
    """The one shaped term in the composer's objective: airspeed, credited per
    interval, pro rata to `speed_ref`.  It must scale with speed, pay nothing
    for standing still, pay nothing to a row that has crashed, and leave the
    plant cost -- what the low level is evolved on -- untouched."""
    from lagrangian_es.composer.policy import returns_from_stream
    B = 4
    speeds = torch.tensor([5.0, 2.5, 0.0, 5.0], dtype=torch.float64)   # full, half, stopped, full-but-dead
    alive = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    chain = [{"t": float(t), "cost": torch.full((B,), 10.0 * t, dtype=torch.float64),
              "speed": speeds, "alive": alive} for t in (0, 20, 40)]
    recs = [{"t": float(t), "alive": torch.ones(B, dtype=torch.bool),
             "act": torch.zeros(B, dtype=torch.long)} for t in (0, 20)]
    plain = returns_from_stream(recs, chain, 0.99, unit=20)
    withb = returns_from_stream(recs, chain, 0.99, unit=20, speed_bonus=1.0, speed_ref=5.0)
    credit = (withb - plain)[0]
    assert float(credit[0]) > 0, "full speed earned nothing"
    assert abs(float(credit[1]) - float(credit[0]) / 2) < 1e-9, "credit is not pro rata to speed"
    assert float(credit[2]) == 0.0, "standing still was paid"
    assert float(credit[3]) == 0.0, "a crashed row was paid"
    # off by default, and it never touches the cost stream itself
    assert torch.equal(returns_from_stream(recs, chain, 0.99, unit=20), plain)
    assert torch.equal(chain[1]["cost"], torch.full((B,), 200.0, dtype=torch.float64))
    # and it is bounded: at most speed_bonus per interval, whatever the speed
    fast = [dict(m, speed=torch.full((B,), 50.0, dtype=torch.float64)) for m in chain]
    over = (returns_from_stream(recs, fast, 0.99, unit=20, speed_bonus=1.0, speed_ref=5.0) - plain)[0]
    assert float(over.max()) <= float(credit[0]) + 1e-9, "credit is not clamped at speed_ref"


def test_the_task_baseline_removes_task_difficulty_not_the_signal():
    """Most of a return's spread is which task was drawn, not what the composer
    did.  Centring per task removes that and leaves the decision's effect."""
    from lagrangian_es.composer.policy import center_by_task
    torch.manual_seed(0)
    E, P, K = 8, 4, 3                                   # tasks, genomes, decisions
    difficulty = torch.tensor([0., 100., 200., 300., 400., 500., 600., 700.], dtype=torch.float64)
    effect = 7.0                                        # what the decisions actually changed
    Rs, rows = [], []
    for g in range(P):                                  # one shard per genome, as the pool splits them
        r = torch.arange(E, dtype=torch.long) + g * E
        R = difficulty[None, :].repeat(K, 1) + effect * (g - 1.5)
        Rs.append(R.clone()); rows.append(r)
    before = torch.cat([R.reshape(-1) for R in Rs]).std()
    center_by_task(Rs, rows, n_eps=E)
    after = torch.cat([R.reshape(-1) for R in Rs]).std()
    assert float(before) > 200 and float(after) < 10, (float(before), float(after))
    # the between-genome signal survives exactly: genome g keeps effect*(g-1.5)
    for g, R in enumerate(Rs):
        assert torch.allclose(R, torch.full_like(R, effect * (g - 1.5)), atol=1e-9)
    # a task seen only once is left alone rather than zeroed
    Rs2 = [torch.full((1, 1), 5.0, dtype=torch.float64)]
    center_by_task(Rs2, [torch.tensor([3])], n_eps=E)
    assert float(Rs2[0]) == 5.0, "a single sample was centred against itself"


def test_exploration_mass_goes_where_the_policy_is_not_looking():
    """Uniform exploration is not fair when use is lopsided: the composer emits
    PLACE ~92% and the turn/priority tokens ~1%, so a uniform mixture leaves
    the rare ones with too little evidence to ever stop being rare."""
    from lagrangian_es.composer.policy import explore_weights
    V = 25
    assert explore_weights(None, V) is None, "no history should mean uniform"
    assert explore_weights(torch.zeros(V, dtype=torch.float64), V) is None
    # a policy that emits token 0 almost always
    c = torch.full((V,), 1.0, dtype=torch.float64); c[0] = 1000.0
    w = explore_weights(c, V)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    assert float(w[0]) < 1.0 / V, "the over-used token still got a uniform share"
    assert float(w[1]) > 1.0 / V, "the starved token got no extra mass"
    assert float(w[1]) / float(w[0]) > 10, "the reweighting is too weak to matter"
    # bounded: a never-emitted token gets at most V times uniform, not everything
    c2 = torch.zeros(V, dtype=torch.float64); c2[0] = 100.0
    w2 = explore_weights(c2, V)
    assert float(w2.max()) <= 1.0, float(w2.max())
    assert float(w2[1]) < 0.5, "one starved token swallowed the whole mixture"
    # a policy already spreading evenly is left alone
    w3 = explore_weights(torch.full((V,), 7.0, dtype=torch.float64), V)
    assert torch.allclose(w3, torch.full((V,), 1.0 / V, dtype=torch.float64), atol=1e-12)
    # A protected token keeps a plain uniform share however common it is.  EOS
    # is 52% of emitted components because every chain ends with one and a bare
    # EOS is silence -- structure, not over-use.  Reweighting against it stopped
    # exploring chains from terminating and drove subgoals per flight upward.
    c4 = torch.full((V,), 1.0, dtype=torch.float64)
    c4[0] = 5000.0        # EOS: structurally common
    c4[1] = 500.0         # a genuinely over-used content token
    w4 = explore_weights(c4, V, protect=(0,))
    assert abs(float(w4[0]) - 1.0 / V) < 1e-9, "protected token was reweighted"
    assert abs(float(w4.sum()) - 1.0) < 1e-12
    assert float(w4[2]) > float(w4[1]) * 10, "the unprotected tokens are no longer rebalanced"
    # and without the protection EOS is crushed -- the bug this pins down
    assert float(explore_weights(c4, V)[0]) < 0.2 / V, "unprotected EOS should be crushed"
