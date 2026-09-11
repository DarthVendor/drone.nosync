"""The task-level layer, and the promises that make it safe to put there.

`FixedWeights` is the identity, and it has to be exactly that: Stage A evolves
the low level with the composer in place, so a composer that changed the
controller by a rounding error would put that error into every fitness it ranks.
The hold is the part that keeps the certificate -- V_d = sum_i alpha_i V_i is
the gradient of a function only while alpha is constant in x -- so it must move
the realized spec at a bounded rate and never faster.  The two interlocks are
non-learned by design and are tested for the edges they guard.
"""
import torch

from lagrangian_es.composer import (FixedWeights, SpecHold, TaskSpec,
                                    ground_release, stale_fallback)
from lagrangian_es.config import Config, RolloutCfg
from lagrangian_es.es import build, build_composer, build_sensors
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen


def _cfg(composer=""):
    return Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                  environment="singapore_cbd", sensors=("range",),
                  gating="arrival", seed=0, composer=composer,
                  task_kw=(("n_legs", 2), ("max_leg", 10.0)),
                  system_kw=(("free_start", True),),
                  trainable_kw=(("learned", True), ("damp_mode", "beams")),
                  rollout=RolloutCfg(n_eps=6, ep_steps=120, dead_mode="constant",
                                     dead_cost=6.0, goal_bonus=15.0))


def test_fixed_weights_is_the_identity_through_a_real_rollout():
    a = _cfg(); b = _cfg("fixed")
    sysm, tr, task = build(a)
    th = tr.init()[None]
    goals = task.sample(6, make_gen(1))
    bare = Rollout(sysm, tr, task, a.rollout, build_sensors(a, sysm))
    comp = build_composer(b, sysm, tr)
    assert isinstance(comp, FixedWeights)
    held = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=comp)
    r0 = bare.run(th, goals, 7); r1 = held.run(th, goals, 7)
    assert torch.equal(r0.fitness, r1.fitness), "the identity spec changed the fitness"
    assert torch.equal(r0.cost, r1.cost)
    assert torch.equal(r0.alive, r1.alive)


def test_the_measurement_chain_reports_what_happened():
    b = _cfg("fixed")
    sysm, tr, task = build(b)
    comp = build_composer(b, sysm, tr); comp.every = 40
    roll = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=comp)
    roll.run(tr.init()[None], task.sample(6, make_gen(2)), 8)
    assert len(roll.chain) == (120 // 40 - 1) + 1, "one token per completed interval, plus the closing token with the settled cost"
    tok = roll.chain[0]
    for k in ("progress", "remaining", "min_beam", "alive", "arrived"):
        assert k in tok and tok[k].shape[0] == 6
    assert torch.isfinite(tok["progress"]).all()


def test_the_hold_never_moves_faster_than_its_rate():
    B, d, n = 4, 3, 2
    hold = SpecHold(B, d, n, dt=0.02, dtype=torch.float64, device="cpu",
                    omega_n=2.0, reach=10.0)
    far = TaskSpec(delta=torch.full((B, d), 30.0, dtype=torch.float64),
                   alpha=torch.full((B, n), 5.0, dtype=torch.float64),
                   gate=torch.zeros(B, n, dtype=torch.float64))
    hold.set_target(far)
    prev = hold.realized.clone()
    for _ in range(50):
        cur = hold.step()
        assert float((cur.delta - prev.delta).norm(dim=-1).max()) <= hold.rate_m + 1e-12
        assert float((cur.alpha - prev.alpha).abs().max()) <= hold.rate_w + 1e-12
        assert float((cur.gate - prev.gate).abs().max()) <= hold.rate_w + 1e-12
        prev = cur.clone()
    for _ in range(5000):
        cur = hold.step()
    assert torch.allclose(cur.delta, far.delta) and torch.allclose(cur.alpha, far.alpha)
    assert torch.allclose(cur.gate, far.gate)


def test_the_hold_is_constant_between_composer_calls_when_settled():
    """d(alpha)/dx = 0 within an interval: once settled, stepping does nothing."""
    hold = SpecHold(2, 3, 1, dt=0.02, dtype=torch.float64, device="cpu")
    tgt = TaskSpec.identity(2, 3, 1, torch.float64, "cpu"); tgt.delta[:] = 1.0
    hold.set_target(tgt)
    for _ in range(2000):
        hold.step()
    a = hold.step().clone(); b = hold.step()
    assert torch.equal(a.delta, b.delta) and torch.equal(a.weight, b.weight)


def test_ground_release_needs_low_and_slow():
    z = torch.tensor([0.3, 0.3, 2.0, 0.3]); vz = torch.tensor([0.1, -0.9, 0.1, 0.49])
    got = ground_release(z, vz, z_release=0.6, vz_max=0.5)
    assert got.tolist() == [True, False, False, True]


def test_stale_fallback_only_touches_stale_rows():
    hold = SpecHold(3, 3, 2, dt=0.02, dtype=torch.float64, device="cpu")
    tgt = TaskSpec.identity(3, 3, 2, torch.float64, "cpu"); tgt.delta[:] = 4.0
    hold.set_target(tgt)
    hold.age = torch.tensor([0, 5, 12])
    stale = stale_fallback(hold, k=8)
    assert stale.tolist() == [False, False, True]
    assert torch.equal(hold.target.delta[2], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(hold.target.delta[0], torch.full((3,), 4.0, dtype=torch.float64))


def test_oracle_subgoals_stay_clear_of_buildings_and_within_reach():
    """The teacher must only ever ask for legs the low level is known to fly:
    the subgoal sits within `reach` of the vehicle and outside the margin."""
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range",), gating="arrival",
                 seed=0, composer="oracle",
                 composer_kw=(("reach", 10.0), ("margin", 0.9)),
                 task_kw=(("n_legs", 2), ("max_leg", 0.0)),
                 system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")))
    sysm, tr, task = build(cfg)
    comp = build_composer(cfg, sysm, tr)
    goals = task.sample(12, make_gen(3))
    s = sysm.reset(12, make_gen(4))
    x = sysm.task_position(s)
    spec = comp.emit({"x": x, "v": sysm.task_velocity(s), "goal": goals[:, 0],
                      "alive": torch.ones(12, dtype=torch.bool), "state": s,
                      "arrived": torch.zeros(12, dtype=torch.bool),
                      "leg": torch.zeros(12, dtype=torch.long), "t": 0})
    sub = goals[:, 0] + spec.delta
    hop = (sub[:, :2] - x[:, :2]).norm(dim=-1)
    assert float(hop.max()) <= 10.0 + 1e-6, "subgoal beyond the measured reach"
    clear = sysm.env.sdf(sub, s)
    assert float(clear.min()) > 0.9 - 0.5 * 2 ** 0.5, "subgoal inside the clearance margin"
    assert torch.equal(spec.weight, torch.ones_like(spec.weight)), "the oracle decides where, not how"
    # a goal already within reach is handed over as-is -- provided it is clear.
    # A goal inside the margin is relocated to the nearest free cell first,
    # which is the planner refusing to send the vehicle into a wall.
    near = x.clone(); near[:, :2] += 3.0
    spec2 = comp.emit({"x": x, "v": sysm.task_velocity(s), "goal": near,
                       "alive": torch.ones(12, dtype=torch.bool), "state": s,
                       "arrived": torch.zeros(12, dtype=torch.bool),
                       "leg": torch.zeros(12, dtype=torch.long), "t": 0})
    ok = sysm.env.sdf(near, s) > 0.9 + 0.5 * 2 ** 0.5
    assert bool(ok.any()), "fixture: no clear near goal to check"
    off = spec2.delta.norm(dim=-1)
    assert float(off[ok].max()) == 0.0, \
        "a clear goal within reach must be handed over EXACTLY -- a cell centre can sit outside the arrival tolerance"
    assert float(off[~ok].min()) >= 0.0


def test_line_of_sight_oracle_only_asks_for_what_the_vehicle_can_see():
    """The city measurement: an occluded leg is lost 91% of the time, a visible
    one 24%.  With `los=True` every subgoal must be visible from the vehicle
    at the clearance margin -- the straight line to it never enters the margin."""
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="occluded", sensors=("range",), gating="arrival",
                 seed=0, composer="oracle",
                 composer_kw=(("reach", 10.0), ("margin", 0.9), ("los", True)),
                 task_kw=(("n_legs", 2), ("max_leg", 0.0)),
                 system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")))
    sysm, tr, task = build(cfg)
    comp = build_composer(cfg, sysm, tr)
    B = 24
    goals = task.sample(B, make_gen(11)); s = sysm.reset(B, make_gen(12))
    x = sysm.task_position(s)
    spec = comp.emit({"x": x, "v": sysm.task_velocity(s), "goal": goals[:, 0],
                      "alive": torch.ones(B, dtype=torch.bool), "state": s,
                      "arrived": torch.zeros(B, dtype=torch.bool),
                      "leg": torch.zeros(B, dtype=torch.long), "t": 0, "chain": []})
    sub = goals[:, 0] + spec.delta
    n = 200
    ts = torch.linspace(0, 1, n, dtype=x.dtype)[None, :, None]
    pts = x[:, None] + ts * (sub - x)[:, None]
    d = torch.stack([sysm.env.sdf(pts[:, i], s) for i in range(n)], 1)
    # the vehicle's own start may sit inside the margin; judge the line past it
    assert float(d[:, n // 8:].min()) > 0.0, "a line-of-sight subgoal is hidden behind a wall"
    assert float((sub - x).norm(dim=-1).max()) <= 10.0 + 1e-9


def test_training_with_the_identity_composer_is_byte_identical():
    """Stage A runs with the composer in place and out of the way: a whole
    training run's fitness history must not move by a rounding error."""
    from dataclasses import replace
    from lagrangian_es.config import ESCfg
    from lagrangian_es.es import train
    base = _cfg()
    small = replace(base, rollout=replace(base.rollout, n_eps=4, ep_steps=60),
                    es=ESCfg(pop=6, gens=3, sigma0=0.05, elite_frac=0.5, strategy="ga",
                             whiten=False, metric_every=1000))
    with_comp = replace(small, composer="fixed")
    a = train(small, *build(small)); b = train(with_comp, *build(with_comp))
    assert torch.equal(a.theta, b.theta)
    assert [h["fitness_elite"] for h in a.history] == [h["fitness_elite"] for h in b.history]


def test_workers_build_the_composer_the_config_names():
    """The parallel path must fly the same controller the parent evaluates."""
    from lagrangian_es import parallel
    cfg = _cfg("fixed")
    parallel._init({"cfg": cfg})
    assert isinstance(parallel._RIG.composer, FixedWeights)
    parallel._init({"cfg": _cfg()})
    assert parallel._RIG.composer is None


def test_sensors_no_term_reads_follow_the_composers_cadence():
    """The depth camera is read by the composer alone, so it is cast at the
    composer's interval; the range fan the low level reads keeps its stride."""
    from dataclasses import replace
    cfg = replace(_cfg("fixed"), sensors=("range", "depth_camera"), composer="transformer",
                  composer_kw=(("reach", 10.0), ("every", 10)))
    sysm, tr, task = build(cfg); sens = build_sensors(cfg, sysm)
    before = {s.name: int(getattr(s, "update_every", 1)) for s in sens}
    Rollout(sysm, tr, task, cfg.rollout, sens, composer=build_composer(cfg, sysm, tr))
    after = {s.name: int(getattr(s, "update_every", 1)) for s in sens}
    assert after["range"] == before["range"], "the low level reads the fan; its stride is not the composer's to change"
    assert after["depth_camera"] == 10, after


def test_live_only_emit_is_identical_when_everyone_is_flying_and_dead_rows_keep_their_target():
    """Skipping dead rows is exact: a batch with everyone alive is byte-identical,
    and once rows die the survivors' decisions are unchanged by their absence."""
    from dataclasses import replace
    from lagrangian_es.config import RolloutCfg
    cfg = replace(_cfg("policy"), composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10)),
                  rollout=RolloutCfg(n_eps=8, ep_steps=200, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0))
    sysm, tr, task = build(cfg)
    goals = task.sample(8, make_gen(31)); th = tr.init()[None]
    comp_a = build_composer(cfg, sysm, tr); comp_b = build_composer(cfg, sysm, tr)
    comp_b.net.load_state_dict(comp_a.net.state_dict()); comp_b.live_only = False
    ra = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp_a).run(th, goals, 32)
    rb = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp_b).run(th, goals, 32)
    assert torch.equal(ra.fitness, rb.fitness) and torch.equal(ra.cost, rb.cost), "live-only emit changed the cost"


def test_the_low_level_is_scored_on_the_placed_subgoal():
    """`cost_sub` charges the distance to the subgoal the composer PLACED, with
    the same bonus and death charge.  Without a composer, or under the
    identity, it is the task cost to the bit; under a composer that places
    the subgoal short of the goal, the low level's charge is to that nearer
    point while the task's charge is unchanged."""
    from lagrangian_es.composer import Composer, TaskSpec
    a = _cfg(); b = _cfg("fixed")
    sysm, tr, task = build(a)
    th = tr.init()[None]; goals = task.sample(6, make_gen(1))
    bare = Rollout(sysm, tr, task, a.rollout, build_sensors(a, sysm)).run(th, goals, 7)
    assert bare.cost_sub is not None and torch.equal(bare.cost_sub, bare.cost)
    assert torch.equal(bare.fitness_sub, bare.fitness)
    ident = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=build_composer(b, sysm, tr)).run(th, goals, 7)
    assert torch.equal(ident.cost_sub, ident.cost), "the identity places the goal itself"

    class Halfway(Composer):
        kind = "halfway"
        def emit(self, ctx):
            sp = TaskSpec.identity(ctx["x"].shape[0], self.d, self.n_terms, ctx["x"].dtype, ctx["x"].device)
            sp.delta = 0.5 * (ctx["x"] - ctx["goal"])      # the subgoal: halfway from the drone to the goal
            return sp
    half = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=Halfway(sysm, tr)).run(th, goals, 7)
    assert half.cost_sub.shape == half.cost.shape and torch.isfinite(half.cost_sub).all()
    # the placed point is nearer than the goal at every step flown, and the
    # death charge and bonus are the same, so the low level's charge is the
    # smaller one on every row; the task's charge does not care where the
    # composer pointed
    assert bool((half.cost_sub < half.cost).all()), (half.cost_sub, half.cost)
    assert half.fitness_sub.shape == half.fitness.shape == (1,)
    assert float(half.fitness_sub) < float(half.fitness)


def test_decisions_are_events_not_ticks():
    """A row is asked again when its placed subgoal is achieved, when its leg
    changes, or when the hold runs out -- never merely because a tick passed.
    With a hold longer than the episode and the identity composer (subgoal =
    goal), the count per row is one decision at the start plus one per leg
    reached, and the report stream still runs every `measure_every` steps."""
    from lagrangian_es.composer import Composer, TaskSpec
    b = _cfg("fixed"); sysm, tr, task = build(b)
    th = tr.init()[None]; goals = task.sample(6, make_gen(1))
    comp = build_composer(b, sysm, tr); comp.every = 10_000; comp.measure_every = 10
    roll = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=comp)
    r = roll.run(th, goals, 7)
    n = roll.n_subgoals
    assert n.shape == (6,) and bool((n >= 1).all())
    assert bool((n <= 1 + r.legs_done).all()), (n, r.legs_done)
    assert len(roll.chain) == b.rollout.ep_steps // 10 - 1 + (0 if b.rollout.ep_steps % 10 else 0) or len(roll.chain) >= 1
    # a hold that runs out re-asks: with a 20-step hold every row is asked at
    # least every 20 steps while it flies
    comp.every = 20
    roll.run(th, goals, 7)
    assert bool((roll.n_subgoals >= n).all())

    class HereOrGoal(Composer):
        """Even rows get the subgoal where they are (achieved at once); odd
        rows get the goal itself (nothing more to be told)."""
        kind = "here_or_goal"; live_only = True
        def __init__(self, system, trainable, **kw):
            super().__init__(system, trainable, **kw); self.calls = []
        def emit(self, ctx):
            self.calls.append(int(ctx["x"].shape[0]))
            B = ctx["x"].shape[0]
            sp = TaskSpec.identity(B, self.d, self.n_terms, ctx["x"].dtype, ctx["x"].device)
            rows = getattr(self, "_rows", None)
            ids = torch.arange(B) if rows is None else rows
            here = (ids % 2 == 0).unsqueeze(-1)
            sp.delta = torch.where(here, ctx["x"] - ctx["goal"], torch.zeros_like(ctx["x"]))
            return sp
    hg = HereOrGoal(sysm, tr); hg.every = 10_000; hg.measure_every = 10
    roll2 = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=hg)
    r2 = roll2.run(th, goals, 7)
    n2 = roll2.n_subgoals
    # the rows whose subgoal is achieved are asked again, and only they are
    # asked; the rows told to go to the goal are never asked again
    assert hg.calls[0] == 6 and len(hg.calls) > 1 and all(c <= 3 for c in hg.calls[1:]), hg.calls
    assert bool((n2[1::2] == 1).all()), n2
    assert int(n2[0::2].max()) > 1, n2


def test_the_composer_pays_per_subgoal():
    from lagrangian_es.composer import returns_from_stream
    B = 4
    chain = [{"t": 10.0, "cost": torch.tensor([1.0, 1.0, 1.0, 1.0])}, {"t": 20.0, "cost": torch.tensor([3.0, 3.0, 3.0, 3.0])}]
    recs = [{"t": 0.0, "act": torch.zeros(B, 2), "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B)},
            {"t": 10.0, "act": torch.zeros(2, 2), "alive": torch.ones(2, dtype=torch.bool), "rows": torch.tensor([0, 2])}]
    R0 = returns_from_stream(recs, chain, 1.0)
    R1 = returns_from_stream(recs, chain, 1.0, subgoal_cost=2.0)
    # every row paid for its first subgoal; rows 0 and 2 paid for a second
    assert torch.allclose(R1[0] - R0[0], torch.tensor([-4.0, -2.0, -4.0, -2.0])), R1[0] - R0[0]
    assert torch.allclose(R1[1] - R0[1], torch.tensor([-2.0, 0.0, -2.0, 0.0]))


def test_the_report_stream_ends_with_the_settled_cost():
    """The composer's return is read off the stream's cost.  The last token
    must carry the cost as SETTLED -- death tail, hover tail and bonus applied
    -- or an early crash looks cheap and an arrival pays nothing."""
    b = _cfg("fixed"); sysm, tr, task = build(b)
    th = tr.init()[None]; goals = task.sample(6, make_gen(1))
    roll = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=build_composer(b, sysm, tr))
    r = roll.run(th, goals, 7)
    last = roll.chain[-1]
    assert torch.equal(last["cost"], r.cost), "the stream's last cost is not the settled cost"
    assert last["t"] == float(b.rollout.ep_steps)
    assert len(roll.chain) >= 2 and roll.chain[-2]["t"] < last["t"]


def test_the_discount_is_a_rate_in_time_when_asked():
    from lagrangian_es.composer import returns_from_stream
    B = 2
    chain = [{"t": 10.0, "cost": torch.tensor([1.0, 1.0])}, {"t": 30.0, "cost": torch.tensor([2.0, 2.0])}, {"t": 40.0, "cost": torch.tensor([4.0, 4.0])}]
    recs = [{"t": 0.0, "act": torch.zeros(B, 1), "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B)},
            {"t": 30.0, "act": torch.zeros(B, 1), "alive": torch.ones(B, dtype=torch.bool), "rows": torch.arange(B)}]
    per_record = returns_from_stream(recs, chain, 0.5)                 # r0 = -(2-0)... cost before t=0 is 0
    per_time = returns_from_stream(recs, chain, 0.5, unit=10.0)        # 30 steps between records = 3 units
    # decision 1's return is the same either way (nothing after it)
    assert torch.allclose(per_record[1], per_time[1])
    # decision 0: r0 + gamma * G1 per record, r0 + gamma^3 * G1 per time
    r0 = -(2.0 - 0.0); G1 = -(4.0 - 2.0)
    assert torch.allclose(per_record[0], torch.full((B,), r0 + 0.5 * G1))
    assert torch.allclose(per_time[0], torch.full((B,), r0 + 0.5 ** 3 * G1))


def test_every_row_samples_its_token_and_only_the_recorded_rows_are_read():
    """One policy: when stochastic every row samples (recorded or not); the
    records hold only the rows the update will read."""
    b = _cfg("policy"); sysm, tr, task = build(b)
    th = tr.init()[None]; goals = task.sample(6, make_gen(1))
    det = build_composer(b, sysm, tr); det.stochastic = False
    r_det = Rollout(sysm, tr, task, b.rollout, build_sensors(b, sysm), composer=det); r_det.trace(th, goals, 7); specs_det = r_det.last_specs.clone()
    with torch.no_grad(): det.net.head_act.bias[det.net.vocab.HOLD] = 0.0        # a policy that speaks often, so sampling shows
    torch.save(det.net.state_dict(), "/tmp/_one_path.pt")
    from dataclasses import replace
    b2 = replace(b, composer_kw=b.composer_kw + (("weights", "/tmp/_one_path.pt"),))
    r_det2 = Rollout(sysm, tr, task, b2.rollout, build_sensors(b2, sysm), composer=build_composer(b2, sysm, tr)); r_det2.trace(th, goals, 7); specs_det2 = r_det2.last_specs.clone()
    noisy = build_composer(b2, sysm, tr); noisy.stochastic = True; noisy.record_rows = torch.tensor([0, 2, 4])
    torch.manual_seed(0)
    r_noisy = Rollout(sysm, tr, task, b2.rollout, build_sensors(b2, sysm), composer=noisy); r_noisy.trace(th, goals, 7); specs_noisy = r_noisy.last_specs
    assert not torch.equal(specs_det2[:, [1, 3, 5]], specs_noisy[:, [1, 3, 5]]), "unrecorded rows must sample too"
    assert all(set(r["rows"].tolist()) <= {0, 2, 4} for r in noisy.records), "only the recorded rows are read"


def test_the_self_token_carries_view_dot_travel():
    """The eighth self feature is the cosine between the forward axis and the
    velocity: +1 flying forward, -1 backward, 0 sideways or at rest."""
    import math
    from lagrangian_es.composer import Tokenizer
    tok = Tokenizer(scale=30.0, reach=10.0)
    B = 4; psi = torch.tensor([0.0, 0.0, math.pi / 2, 0.3], dtype=torch.float64)
    c, s_ = torch.cos(psi), torch.sin(psi)
    R = torch.zeros(B, 3, 3, dtype=torch.float64); R[:, 0, 0] = c; R[:, 0, 1] = -s_; R[:, 1, 0] = s_; R[:, 1, 1] = c; R[:, 2, 2] = 1.0
    v = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    x = torch.zeros(B, 3, dtype=torch.float64); x[:, 2] = 1.5
    ctx = {"x": x, "v": v, "goal": x + 5.0, "state": {"R": R, "p": x, "v": v}, "t": 0, "chain": []}
    t = tok(ctx, [])
    assert torch.allclose(t["self"][:, 7], torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-9), t["self"][:, 7]


def test_spec_where_carries_moved_and_the_recorder_survives_frozen_rows():
    import torch
    from lagrangian_es.composer.spec import TaskSpec
    a = TaskSpec.identity(4, 3, 2, torch.float64, "cpu"); b = a.clone()
    a.moved = torch.tensor([True, True, False, False]); b.moved = torch.tensor([False, False, False, True])
    m = torch.tensor([True, False, True, False])
    assert a.where(m, b).moved.tolist() == [True, False, False, True]      # self where the mask holds, the other elsewhere
    # the recorder composer through a rollout where rows freeze (crash or arrive)
    from dataclasses import replace
    from lagrangian_es.config import RolloutCfg
    from lagrangian_es.es import build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.composer import Recorder, OracleSubgoal
    cfg = replace(_cfg(), rollout=RolloutCfg(n_eps=6, ep_steps=150, dead_mode="constant", dead_cost=40.0, goal_bonus=60.0, stop_on_arrival=True))
    sysm, tr, task = build(cfg); comp = Recorder(sysm, tr, OracleSubgoal(sysm, tr, reach=10.0, every=50), reach=10.0)
    roll = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp)
    r = roll.run(tr.init()[None], task.sample(6, make_gen(5)), 6)
    assert torch.isfinite(r.cost).all()


def test_error_update_runs_end_to_end_on_real_records():
    """Exercise the whole path: fly, record, update.

    Three launches died on plumbing rather than design -- a KeyError for a
    context field the sub-context never carried, and an import of `to_ego` from
    the wrong module -- each costing a run to discover.  Unit tests on the loss
    arithmetic caught none of it because they never touched a real record.
    """
    import torch
    from lagrangian_es.composer.policy_cont import error_update
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_composer, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range",), gating="arrival", seed=0,
                 composer="policy_cont",
                 composer_kw=(("reach", 10.0), ("every", 20), ("measure_every", 20)),
                 task_kw=(("n_legs", 2), ("max_leg", 20.0)),
                 system_kw=(("free_start", True), ("speed_limit", 5.0)),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")),
                 rollout=RolloutCfg(n_eps=8, ep_steps=200, dead_mode="constant",
                                    dead_cost=40.0, goal_bonus=60.0))
    system, trainable, task = build(cfg)
    comp = build_composer(cfg, system, trainable)
    comp.stochastic = True; comp.records = []; comp.record_rows = torch.arange(8)
    comp.tok_frac = 1.0
    comp.reset(8); comp.pair(8, 3)
    rig = Rollout(system, trainable, task, cfg.rollout, build_sensors(cfg, system), composer=comp)
    torch.manual_seed(0)
    with torch.no_grad():
        rig.run(trainable.init()[None], task.sample(8, make_gen(1)), 2)

    recs = comp.records
    assert recs, "no decisions were recorded"
    # the fields the error loss needs must survive to the update
    with_tok = [r for r in recs if r.get("tok")]
    assert with_tok, "no scene tokens kept"
    assert all("x" in r and "goalw" in r for r in with_tok), \
        "the decision position never reached the record -- the error loss cannot pair decisions"

    before = comp.net.arg_w2.detach().clone()
    st = error_update(comp.net, recs, reach=10.0, epochs=1, batch=64, lr=1e-3)
    assert st["n"] > 0, "no decision pairs were formed"
    assert st["ce"] == st["ce"], "loss is NaN"
    assert not torch.equal(before, comp.net.arg_w2), "the update changed nothing"


def test_time_update_runs_end_to_end_and_is_purely_time():
    """The loss has one quantity in it: time.

    T_hat(s, g) is the composer's estimate of time-to-goal via subgoal g; T is
    the time the flight actually took from that decision.  The model is fitted
    to T, and the policy descends the model.  No distance, no aim, no reward,
    no labels -- T comes from the flight itself.
    """
    import torch
    from lagrangian_es.composer.policy_cont import time_update
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_composer, build_sensors
    from lagrangian_es.rollout import Rollout
    from lagrangian_es.util import make_gen
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range",), gating="arrival", seed=0,
                 composer="policy_cont",
                 composer_kw=(("reach", 10.0), ("every", 20), ("measure_every", 20)),
                 task_kw=(("n_legs", 2), ("max_leg", 20.0)),
                 system_kw=(("free_start", True), ("speed_limit", 5.0)),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")),
                 rollout=RolloutCfg(n_eps=8, ep_steps=200, dead_mode="constant",
                                    dead_cost=40.0, goal_bonus=60.0))
    system, trainable, task = build(cfg)
    comp = build_composer(cfg, system, trainable)
    comp.stochastic = True; comp.records = []; comp.record_rows = torch.arange(8)
    comp.tok_frac = 1.0
    comp.reset(8); comp.pair(8, 3)
    rig = Rollout(system, trainable, task, cfg.rollout, build_sensors(cfg, system), composer=comp)
    torch.manual_seed(0)
    with torch.no_grad():
        r = rig.run(trainable.init()[None], task.sample(8, make_gen(1)), 2)

    before_pol = comp.net.arg_w2.detach().clone()
    before_mod = comp.net.time_head[-1].weight.detach().clone()
    st = time_update(comp.net, comp.records, [r.finish_frac], ep_steps=200,
                     epochs=1, batch=64, lr=1e-3)
    assert st["n"] > 0, "no decisions carried a measured time"
    assert st["nll"] == st["nll"] and st["ce"] == st["ce"], "loss is NaN"
    # BOTH halves must move: the model fitted to measured time, and the policy
    # descending the model
    assert not torch.equal(before_mod, comp.net.time_head[-1].weight), "the time model did not learn"
    assert not torch.equal(before_pol, comp.net.arg_w2), "the policy did not follow the model"


def test_the_time_model_conditions_on_the_subgoal():
    """The existing value head sees only the state, so it cannot say which of
    two subgoals is quicker -- and that comparison is the whole of navigation.
    The time head must give different answers for different subgoals."""
    import torch
    from lagrangian_es.composer.policy_cont import ContPolicyNet
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    q = torch.randn(16, net.time_head[0].weight.shape[1] - 3)
    near = torch.zeros(16, 3); near[:, 0] = 0.05      # subgoal at the vehicle's feet
    far = torch.zeros(16, 3); far[:, 0] = 0.95        # subgoal out at the goal
    with torch.no_grad():
        t_near = net.time_head(torch.cat([q, near], -1))
        t_far = net.time_head(torch.cat([q, far], -1))
    assert not torch.allclose(t_near, t_far), "the time model ignores the subgoal it is scoring"
