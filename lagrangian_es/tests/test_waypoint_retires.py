"""A waypoint is somewhere to go THROUGH, not a new goal.

The controller's target is `goal + hold.target.delta`. EOS means "no change"
and so never clears `delta`, and under sparse injection (`inject_at`) the
injected steps are the ONLY ones a row decides at -- so once a WAYPOINT was
placed, nothing could put `delta` back and the vehicle flew to `goal + delta`
and parked there for the rest of the episode.

MEASURED before the fix, 256 paired tasks, identical seeds, 20 m legs / 25%
buildings, a WAYPOINT forced at full reach on every decision:

    muted 0.539  |  +90 deg 0.281  |  +180 deg 0.285  |  0 deg 0.285

A quarter of the arrival rate at EVERY bearing -- the bearing only chose where
to park -- with final_err 5.46 -> 8.5 m, parked inside the 10 m reach ball. The
action space had no beneficial region, so a bandit on the type decision
correctly learned silence (speak 0.370 -> 0.000 in 13 iterations).
"""
import torch

from lagrangian_es.composer.spec import SpecHold


def _hold(B=4, delta=None):
    h = SpecHold(B, 3, 1, dt=0.02, dtype=torch.float32, device=torch.device("cpu"))
    if delta is not None:
        h.target.delta = delta.clone()
        h.realized.delta = delta.clone()
    return h


def _retire(hold, x, goal, alive, arrived, tol):
    """The retirement rule exactly as `Rollout._composer_step` applies it.

    Returns (reached, dropped): `reached` is the waypoint-arrival event that
    also earns the row another decision; `dropped` additionally covers the
    goal-in-hand case, which zeroes the delta but is not an event.
    """
    pl = hold.target.delta
    held = pl.norm(dim=-1) > tol
    reached = alive & ~arrived & held & ((x - (goal + pl)).norm(dim=-1) < tol)
    at_goal = alive & ~arrived & held & ((x - goal).norm(dim=-1) < tol)
    drop = reached | at_goal
    if bool(drop.any()):
        hold.target.delta = torch.where(drop[:, None], torch.zeros_like(pl), pl)
    return reached, drop


def test_reaching_the_waypoint_hands_the_vehicle_back_to_the_goal():
    goal = torch.zeros(4, 3)
    delta = torch.tensor([[8.0, 0.0, 0.0]] * 4)
    hold = _hold(4, delta)
    # row 0 is standing on the waypoint; row 1 is still on its way
    x = torch.tensor([[8.0, 0.0, 0.0], [2.0, 0.0, 0.0], [8.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    alive = torch.tensor([True, True, False, True])
    arrived = torch.tensor([False, False, False, True])
    reached, _ = _retire(hold, x, goal, alive, arrived, tol=1.0)
    assert bool(reached[0]), "standing on the waypoint must retire it"
    assert not bool(reached[1]), "still flying toward it: it stands"
    assert not bool(reached[2]), "a dead row retires nothing"
    assert not bool(reached[3]), "an arrived row retires nothing"
    assert float(hold.target.delta[0].norm()) == 0.0, "retired: the goal is the target again"
    assert torch.allclose(hold.target.delta[1], delta[1]), "untouched"


def test_a_waypoint_at_the_goal_is_not_retired_as_if_it_were_reached():
    """`delta` below tolerance IS the goal -- silence. Zeroing it would be a
    no-op, but the guard keeps `reached` meaning what it says."""
    goal = torch.zeros(2, 3)
    hold = _hold(2, torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]))
    x = torch.zeros(2, 3)
    reached, _ = _retire(hold, x, goal, torch.ones(2, dtype=torch.bool),
                         torch.zeros(2, dtype=torch.bool), tol=1.0)
    assert not bool(reached.any()), "a subgoal that already IS the goal is not a waypoint"


def test_only_the_target_is_zeroed_so_the_low_level_sees_no_step():
    """`SpecHold.step` slews `realized` toward `target` at `rate_m`. Zeroing
    the target lets the hold walk the subgoal home; zeroing `realized` too
    would hand the controller a step input, which is the one thing the hold
    exists to prevent."""
    goal = torch.zeros(1, 3)
    delta = torch.tensor([[8.0, 0.0, 0.0]])
    hold = _hold(1, delta)
    _retire(hold, torch.tensor([[8.0, 0.0, 0.0]]), goal,
            torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), tol=1.0)
    assert float(hold.target.delta.norm()) == 0.0
    assert float(hold.realized.delta.norm()) > 0.0, "realized must still be slewing, not snapped"
    moved = [float(hold.realized.delta.norm())]
    for _ in range(200):
        hold.step(); moved.append(float(hold.realized.delta.norm()))
    assert moved[-1] < 1e-3, f"the hold must walk it home, got {moved[-1]:.4f}"
    steps = [moved[i] - moved[i + 1] for i in range(len(moved) - 1) if moved[i + 1] > 0]
    assert max(steps) <= hold.rate_m + 1e-6, "never faster than the hold's rate limit"


def test_the_retirement_rule_is_the_one_the_rollout_actually_runs():
    """Guards against this test drifting from the implementation."""
    import inspect

    from lagrangian_es.rollout import Rollout
    src = inspect.getsource(Rollout._composer_step)
    assert "_reached" in src and "A WAYPOINT IS CONSUMED WHEN IT IS REACHED" in src
    assert "hold.target.delta = torch.where(_reached[:, None]" in src, \
        "the rollout must zero the TARGET delta on reached rows"
    assert "hold.realized.delta = torch.where(_reached" not in src, \
        "realized must be left to the hold's slew, not snapped"


def test_the_goal_in_hand_rule_is_deliberately_absent():
    """It looked like a second bug of the same family and is NOT one.

    Arrival is scored at the real goal and must be held for `dwell_s`, so a
    live subgoal ought to park the vehicle beside it and block the dwell. But
    `_subgoal_ego` sets `radius = L0.clamp(max=1.0)` -- the subgoal shrinks
    with the goal distance -- so |delta| is already inside `tol` by the time it
    could matter. Adding the rule moved the every-report arm -0.070 -> -0.105
    (+-0.036): nothing. Recorded so it is not "fixed" again.
    """
    import inspect

    from lagrangian_es.rollout import Rollout
    src = inspect.getsource(Rollout._composer_step)
    assert "_at_goal" not in src, "measured at zero effect; do not reintroduce without a measurement"
    assert "NOT ALSO" in src, "the reason it is absent must stay written down"
