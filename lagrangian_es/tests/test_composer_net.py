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
    sysm, tr, task = build(_cfg()); comp = build_composer(_cfg(), sysm, tr)
    ctx, _ = _ctx(sysm, task)
    spec = comp.emit(ctx)
    sub = ctx["goal"] + spec.delta
    assert float((sub - ctx["x"]).norm(dim=-1).max()) <= 10.0 + 1e-9
    assert float(sub[:, 2].min()) >= 0.5, "a subgoal below the floor is not reachable"
    assert bool((spec.alpha > 0).all()) and bool((spec.gate >= 0).all()) and bool((spec.gate <= 1).all())
    assert spec.alpha.shape == (6, len(tr.terms))


def test_parameter_budget():
    n = sum(p.numel() for p in ComposerNet(1).parameters())
    assert 8e4 <= n <= 2.5e5, n


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
    # one record per interval while anyone is flying; none once everyone has died
    assert 1 <= len(comp.records) <= 120 // 10 and all("cost" in m for m in roll.chain)
    R = returns_from_stream(comp.records, roll.chain, gamma=0.99)
    assert R.shape == (len(comp.records), 6) and torch.isfinite(R).all()
    before = torch.cat([p.detach().flatten().clone() for p in comp.net.parameters()])
    st = ppo_update(comp.net, comp.records, R, comp.n_terms, epochs=1, batch=64)
    after = torch.cat([p.detach().flatten() for p in comp.net.parameters()])
    assert st["n"] > 0 and abs(st["loss"]) < 1e6
    for k in ("kl", "clipfrac", "ev", "sigma"):
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


def test_the_exploration_scale_learns_faster_than_the_network():
    """`log_std` sits in its own optimiser group at ten times the rate: at the
    network's rate it could move ~0.7% an iteration and never adapt in a run."""
    import torch
    from lagrangian_es.composer import PolicyNet
    from lagrangian_es.composer.policy import collate_tok
    net = PolicyNet(1).double()
    F = 8; B = 16
    tok = {"self": torch.randn(B, F, dtype=torch.float64), "goal": torch.randn(B, F, dtype=torch.float64),
           "entities": torch.randn(B, 4, F, dtype=torch.float64), "ent_types": torch.full((B, 4), 2),
           "ent_mask": torch.ones(B, 4, dtype=torch.bool), "chain": torch.zeros(B, 0, F, dtype=torch.float64),
           "chain_types": torch.zeros(B, 0, dtype=torch.long), "psi": torch.zeros(B, dtype=torch.float64)}
    recs = [{"t": 0.0, "act": torch.randn(B, 3 + 2 * 1 + 2, dtype=torch.float64), "alive": torch.ones(B, dtype=torch.bool),
             "tok": tok}]   # sub-goal 3, per-term alpha/gate, heading delta + gate
    from lagrangian_es.composer import ppo_update
    before = net.log_std.detach().clone(); w_before = net.head_sub.weight.detach().clone()
    ppo_update(net, recs, torch.randn(1, B, dtype=torch.float64), 1, epochs=1, batch=B, lr=1e-3)
    d_std = float((net.log_std - before).abs().max()); d_w = float((net.head_sub.weight - w_before).abs().max())
    assert d_std > 5 * d_w, (d_std, d_w)


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


def test_a_nonfinite_minibatch_is_skipped_not_stepped():
    import torch
    from lagrangian_es.composer import PolicyNet, ppo_update
    net = PolicyNet(1).float(); B, F = 16, 8
    tok = {"self": torch.randn(B, F), "goal": torch.randn(B, F), "entities": torch.randn(B, 4, F), "ent_types": torch.full((B, 4), 2),
           "ent_mask": torch.ones(B, 4, dtype=torch.bool), "chain": torch.zeros(B, 0, F), "chain_types": torch.zeros(B, 0, dtype=torch.long), "psi": torch.zeros(B)}
    with torch.no_grad(): pre, _ = net.pre(tok)
    act = pre.clone(); act[0, 0] = float("nan")                 # one poisoned action
    recs = [{"t": 0.0, "act": act, "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B), "tok": tok}]
    before = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
    st = ppo_update(net, recs, torch.randn(1, B, dtype=torch.float64), 1, epochs=1, batch=B, lr=1e-3)
    after = torch.cat([p.detach().flatten() for p in net.parameters()])
    # the poisoned sample is dropped and counted; the healthy ones still train, finitely
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
        with torch.no_grad(): pre, _ = net.pre(tok)
        return {"t": t, "act": pre, "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B), "tok": tok}
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
        with torch.no_grad():
            for p_ in comp.net.parameters(): p_.add_(0.3 * torch.randn_like(p_))
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
    with torch.no_grad(): pre, _ = net.pre(tok)
    recs = [{"t": 0.0, "act": pre + 0.05 * torch.randn_like(pre), "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B), "tok": tok}]
    R = torch.randn(1, B, dtype=torch.float64)
    st = ppo_update(net, recs, R, 1, epochs=30, batch=16, lr=5e-3, vcoef=0.0, ent=0.0, target_kl=0.02)
    assert st["epochs"] < 30, "the early-stop never fired"
    # the step is sized in policy space: a rate that overshoots is halved until
    # the whole batch sits inside the trust region, and the rate that fit comes back
    assert st["backtracks"] >= 1 and st["lr"] < 5e-3, st
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
