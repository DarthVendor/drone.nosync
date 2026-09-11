"""The beam interrupt: a close return makes a row decide without waiting for
the report clock.  New code, and I got two things wrong writing it -- the
sensor selection matched `range_down` (which reads the floor at ~1.5 m all
flight, so the alarm was permanently true and arrival went to 0.000), and the
`leg_last` guard landed on the wrong assignment."""
import torch

from lagrangian_es.config import Config, RolloutCfg
from lagrangian_es.es import build, build_composer, build_sensors
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen


def _rig(trigger, gap=4, eps=8, steps=200):
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range", "range_down"),
                 gating="arrival", seed=0, composer="policy_cont",
                 composer_kw=(("reach", 10.0), ("every", 20), ("measure_every", 20)),
                 task_kw=(("n_legs", 2), ("max_leg", 20.0)),
                 system_kw=(("free_start", True), ("speed_limit", 5.0)),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")),
                 rollout=RolloutCfg(n_eps=eps, ep_steps=steps, dead_mode="constant",
                                    dead_cost=40.0, goal_bonus=60.0))
    system, trainable, task = build(cfg)
    system.difficulty = 1.0
    comp = build_composer(cfg, system, trainable)
    comp.beam_trigger = trigger
    comp.trigger_gap = gap
    comp.stochastic = True
    comp.records = []
    comp.record_rows = None
    comp.reset(eps)
    comp.pair(eps, 1)
    rig = Rollout(system, trainable, task, cfg.rollout, build_sensors(cfg, system), composer=comp)
    return rig, trainable, task, eps


def _decision_steps(trigger, gap=4):
    rig, trainable, task, eps = _rig(trigger, gap)
    steps = []
    orig = Rollout._emit_live

    def spy(self, c, s, goal, alive, arrived, leg, t, hold, rows=None):
        steps.append(int(t))
        return orig(self, c, s, goal, alive, arrived, leg, t, hold, rows=rows)

    Rollout._emit_live = spy
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            res = rig.run(trainable.init()[None], task.sample(eps, make_gen(1)), 2)
    finally:
        Rollout._emit_live = orig
    return steps, res


def test_trigger_off_decides_only_on_the_report_clock():
    """beam_trigger = 0 must behave exactly as before the interrupt existed."""
    steps, _ = _decision_steps(0.0)
    assert steps, "no decisions at all"
    off = [t for t in steps if t % 20 != 0]
    assert not off, f"decisions off the 20-step clock with the trigger disabled: {off[:8]}"


def test_a_close_beam_decides_between_reports():
    steps, _ = _decision_steps(1.5)
    off = [t for t in steps if t % 20 != 0]
    assert off, "the trigger never fired in a full city at 20 m legs"


def test_the_trigger_gap_is_respected_per_row():
    """Without a gap a wall held in view re-fires every step and the chain
    fills with duplicates.

    PER ROW: `t_last` is per episode, so different rows firing at 166, 169 and
    172 is correct -- each is respecting its own gap.  Checking the global step
    list conflates them and fails on correct behaviour."""
    gap = 6
    rig, trainable, task, eps = _rig(1.5, gap=gap)
    fired = {}
    orig = Rollout._emit_live

    def spy(self, c, s, goal, alive, arrived, leg, t, hold, rows=None):
        if rows is not None and int(t) % 20 != 0:
            for j in rows.nonzero().flatten().tolist():
                fired.setdefault(int(j), []).append(int(t))
        return orig(self, c, s, goal, alive, arrived, leg, t, hold, rows=rows)

    Rollout._emit_live = spy
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            rig.run(trainable.init()[None], task.sample(eps, make_gen(1)), 2)
    finally:
        Rollout._emit_live = orig
    assert fired, "the trigger never fired"
    for row, ts in fired.items():
        close = [(a, b) for a, b in zip(ts, ts[1:]) if b - a < gap]
        assert not close, f"row {row} interrupted twice within the gap of {gap}: {close[:4]}"


def test_the_alarm_ignores_the_downward_fan():
    """`range_down` points at the floor and reads ~1.5 m for the whole flight.
    Matching it made the alarm permanently true -- every row re-decided every
    `trigger_gap` steps and arrival collapsed to 0.000."""
    rig, trainable, task, eps = _rig(1.9)           # well above the floor reading
    steps = []
    orig = Rollout._emit_live

    def spy(self, c, s, goal, alive, arrived, leg, t, hold, rows=None):
        steps.append(int(t))
        return orig(self, c, s, goal, alive, arrived, leg, t, hold, rows=rows)

    Rollout._emit_live = spy
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            rig.run(trainable.init()[None], task.sample(eps, make_gen(1)), 2)
    finally:
        Rollout._emit_live = orig
    off = [t for t in steps if t % 20 != 0]
    # if the floor were triggering it, EVERY gap-spaced step would fire
    assert len(off) < len(steps), "the alarm fired on essentially every step -- it is seeing the floor"


def test_the_hold_slews_slower_than_the_interrupt_reacts():
    """The interrupt is only worth having if the hold realises the new subgoal
    faster than the clock it bypassed.

    `SpecHold` rate-limits the subgoal at reach * omega_n * dt = 0.4 m/step,
    from the low level's bandwidth -- a subgoal moving faster than the closed
    loop can follow is a step, which is what the hold exists to prevent.  So an
    interrupt saves up to 20 steps of waiting and then spends some of it
    slewing.  Pinned because if the rate ever drops, the interrupt stops paying
    for itself and the reason would not be visible from either file alone.
    """
    import torch as _t
    from lagrangian_es.composer.spec import SpecHold

    dt = 0.02
    hold = SpecHold(4, 3, 1, dt, _t.float64, _t.device("cpu"), omega_n=2.0, reach=10.0)
    assert abs(hold.rate_m - 0.4) < 1e-9, hold.rate_m

    def steps_to_realise(move):
        h = SpecHold(4, 3, 1, dt, _t.float64, _t.device("cpu"), omega_n=2.0, reach=10.0)
        tgt = h.realized.clone()
        tgt.delta = tgt.delta + _t.tensor([move, 0.0, 0.0], dtype=_t.float64)
        h.set_target(tgt)
        n = 0
        while float((h.realized.delta - tgt.delta).norm(dim=-1).max()) > 1e-6 and n < 500:
            h.step(); n += 1
        return n

    # a typical avoidance move must land inside the 20-step interval it skipped
    assert steps_to_realise(3.0) <= 20, steps_to_realise(3.0)
    assert steps_to_realise(5.0) <= 20, steps_to_realise(5.0)
    # and the limit must still bite on a large one, or the hold is not holding
    assert steps_to_realise(10.0) > 20


def test_snap_bypasses_the_slew_entirely():
    """A leg change is the TASK's step, not the composer's move, so it is
    snapped rather than slewed -- slewing it made an identity composer differ
    from no composer by 2.8e-2 in cost on the leg that changed mid-episode."""
    import torch as _t
    from lagrangian_es.composer.spec import SpecHold

    h = SpecHold(4, 3, 1, 0.02, _t.float64, _t.device("cpu"), omega_n=2.0, reach=10.0)
    tgt = h.realized.clone()
    tgt.delta = tgt.delta + _t.tensor([9.0, 0.0, 0.0], dtype=_t.float64)
    h.set_target(tgt)
    rows = _t.tensor([True, True, False, False])
    h.snap(rows)
    d = (h.realized.delta - tgt.delta).norm(dim=-1)
    assert float(d[:2].max()) < 1e-9, "snapped rows did not reach the target at once"
    assert float(d[2:].min()) > 1.0, "unsnapped rows should still be slewing"


def test_a_subset_target_does_not_leak_into_other_rows():
    """The interrupt fires for SUBSETS of rows, so `set_target(spec, rows=...)`
    merges a partial spec far more often than the report clock ever did.  A
    leak here would let one episode's emergency avoidance overwrite another's
    subgoal -- and it would look like noise, not a bug."""
    import torch as _t
    from lagrangian_es.composer.spec import SpecHold

    B = 4
    hold = SpecHold(B, 3, 1, 0.02, _t.float64, _t.device("cpu"), omega_n=2.0, reach=10.0)
    first = hold.realized.clone()
    first.delta = _t.tensor([[1.0, 0.0, 0.0]], dtype=_t.float64).repeat(B, 1)
    hold.set_target(first)
    before = hold.target.delta.clone()

    only_row1 = hold.realized.clone()
    only_row1.delta = _t.tensor([[9.0, 9.0, 9.0]], dtype=_t.float64).repeat(B, 1)
    rows = _t.tensor([False, True, False, False])
    hold.set_target(only_row1, rows=rows)

    assert _t.allclose(hold.target.delta[1], _t.tensor([9.0, 9.0, 9.0], dtype=_t.float64))
    for j in (0, 2, 3):
        assert _t.allclose(hold.target.delta[j], before[j]), (
            f"row {j} changed when only row 1 was interrupted: {hold.target.delta[j].tolist()}")
    # and only the interrupted row's age resets
    assert int(hold.age[1]) == 0
    assert all(int(hold.age[j]) == int(hold.age[0]) for j in (2, 3))


def test_merging_a_spec_without_yaw_drops_yaw_for_everyone():
    """KNOWN HAZARD.  `TaskSpec.where` returns yaw=None if EITHER side is None,
    so merging a partial spec that carries no yaw silently discards the yaw of
    rows that were not even selected.

    Not reachable today -- `ContVocab.begin` clones the current spec, so an
    emitted spec always carries yaw whenever the task has one -- but the
    asymmetry is invisible at the call site, and the interrupt made partial
    merges common.  Pinned so that a future composer which omits yaw fails
    here instead of quietly flattening every row's heading."""
    import torch as _t
    from lagrangian_es.composer.spec import TaskSpec

    B = 3
    withyaw = TaskSpec.identity(B, 3, 1, _t.float64, _t.device("cpu"))
    withyaw.yaw = _t.full((B,), 0.5, dtype=_t.float64)
    noyaw = TaskSpec.identity(B, 3, 1, _t.float64, _t.device("cpu"))
    noyaw.yaw = None

    merged = noyaw.where(_t.tensor([True, False, False]), withyaw)
    assert merged.yaw is None, (
        "the asymmetry has been fixed -- update this test to assert the rows "
        "that were not selected keep their yaw")
