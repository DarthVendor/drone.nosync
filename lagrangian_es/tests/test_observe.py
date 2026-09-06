"""Watching a target that obstacles can hide.

Three pieces, each of which fails differently:
  StandoffBowl   an equilibrium that is a SHELL, so there is somewhere to watch
                 from that is not on top of the thing being watched
  visibility     a PENUMBRA, so the gradient points out of the shadow
  place_occluders  a scene where the shadow is actually in the way
"""
import math

import pytest
import torch

from lagrangian_es.systems import make_system
from lagrangian_es.tasks import make_task
from lagrangian_es.trainables.terms import StandoffBowl
from lagrangian_es.util import make_gen

DT = torch.float64


# --- the shell --------------------------------------------------------------
def test_standoff_bowl_settles_on_a_shell_not_on_the_target():
    """Every other potential pulls onto the goal, which is the wrong equilibrium
    for an observation: you cannot watch what you are standing on, and occlusion
    is meaningless at zero range."""
    t = StandoffBowl(3, r0=1.8)
    th = t.init(dtype=DT)
    z = torch.zeros(1, 3, dtype=DT)
    for n, want in ((0.5, "out"), (1.0, "out"), (2.5, "in"), (3.5, "in")):
        e = torch.tensor([[n, 0.0, 0.0]], dtype=DT)
        g = float(t.grad_potential(th, e, z, z)[0, 0])
        assert (g < 0) if want == "out" else (g > 0), f"|e|={n}"
    at = torch.tensor([[1.8, 0.0, 0.0]], dtype=DT)
    assert abs(float(t.grad_potential(th, at, z, z)[0, 0])) < 1e-9
    assert float(t.potential(th, at, z, z)) < 1e-12


def test_standoff_bowl_is_psd_and_says_the_goal_is_not_its_equilibrium():
    t = StandoffBowl(3)
    th = t.init(dtype=DT)
    z = torch.zeros(8, 3, dtype=DT)
    g = torch.Generator().manual_seed(0)
    e = torch.randn(8, 3, generator=g, dtype=DT) * 2.0
    assert torch.all(t.potential(th, e, z, z) >= 0.0)
    c = t.certificate(th)
    assert c["psd"] is True
    # claiming the goal is the equilibrium would be false: the shell is
    assert c["zero_at_goal"] is False


def test_standoff_radius_is_bounded_so_the_shell_cannot_leave_the_scene():
    t = StandoffBowl(3)
    for v in (1e3, 1e12, -1e12):
        _, r = t._params(torch.tensor([1.0, v], dtype=DT))
        assert t.r_lo <= float(r) <= t.r_hi


# --- the penumbra -----------------------------------------------------------
def _blocked_rig(lat, n=1):
    system = make_system("quadrotor_nav", environment="pillars", dtype=DT)
    s = system.reset(n, make_gen(0))
    c = torch.full((n, 6, 2), 90.0, dtype=DT)
    r = torch.full((n, 6), 0.05, dtype=DT)
    c[:, 0] = torch.tensor([0.0, 0.0], dtype=DT)
    r[:, 0] = 0.35
    s["pillars/c"], s["pillars/r"] = c, r
    goal = torch.tensor([[1.6, 0.0, 1.2]], dtype=DT).expand(n, -1).clone()
    s["p"] = torch.tensor([[-1.6, 0.0, 1.2]], dtype=DT).expand(n, -1).clone()
    s["p"] = s["p"].clone()
    s["p"][:, 1] = torch.as_tensor(lat, dtype=DT)
    return system, s, goal


def test_visibility_is_a_penumbra_with_a_lateral_gradient():
    """The whole reason for integrating over the target's extent.  A single ray's
    overshoot varies along the LINE OF SIGHT, so smoothing it leaves the umbra
    flat sideways -- and sideways is the only direction that recovers the view."""
    lat = torch.linspace(-1.6, 1.6, 21, dtype=DT)
    system, s, goal = _blocked_rig(lat, n=21)
    v = system.visibility(s, goal)
    assert float(v[0]) > 0.95 and float(v[-1]) > 0.95      # clear at the edges
    assert float(v[10]) < 0.05                             # umbra in the middle
    step = (v[1:] - v[:-1]).abs()
    assert 0.02 < float(step.max()) < 0.9, "hard edge, not a penumbra"
    # monotone out of the shadow on each side
    left = v[:11]
    assert torch.all(left[1:] <= left[:-1] + 1e-9)


def test_visibility_is_one_with_nothing_in_the_way():
    system, s, goal = _blocked_rig(0.0)
    s["pillars/c"] = torch.full_like(s["pillars/c"], 90.0)
    assert float(system.visibility(s, goal)) > 0.95


# --- the scene --------------------------------------------------------------
def _shell_blocked(n_occ, n=96, radius=1.8, k=24):
    system = make_system("quadrotor_nav", environment="pillars", dtype=DT,
                         n_occluders=n_occ)
    task = make_task("observe", system)
    s = system.reset(n, make_gen(7))
    goals = task.sample(n, make_gen(7))
    s = system.place_course(s, goals)
    g = goals[:, 0]
    blocked = torch.zeros(n, dtype=DT)
    for i in range(k):
        a = 2 * math.pi * i / k
        st = dict(s)
        st["p"] = g + torch.tensor([math.cos(a) * radius, math.sin(a) * radius,
                                    0.0], dtype=DT)
        blocked += (system.visibility(st, g) < 0.5).to(DT)
    return float((blocked / k).mean())


def test_occluders_actually_shadow_the_viewing_shell():
    """Without them the stock field blocks ~10% of the shell and the task never
    asks the vehicle to reposition."""
    bare, placed = _shell_blocked(0), _shell_blocked(3)
    assert placed > bare + 0.10, f"bare {bare:.3f} vs placed {placed:.3f}"


def test_occluders_never_bury_the_target_itself():
    """They go NEAR the target, not on it -- `clear_points` runs first, so a
    clear vantage always exists and the task stays solvable."""
    system = make_system("quadrotor_nav", environment="pillars", dtype=DT,
                         n_occluders=4)
    task = make_task("observe", system)
    s = system.reset(128, make_gen(3))
    goals = task.sample(128, make_gen(3))
    s = system.place_course(s, goals)
    assert torch.all(system.env.sdf(goals[:, 0], s) > 0.0)


def test_observe_success_needs_all_three_conditions():
    system = make_system("quadrotor_nav", environment="pillars", dtype=DT)
    task = make_task("observe", system)
    s = system.reset(4, make_gen(0))
    goal = system.task_position(s).clone()
    goal[:, 0] += 0.1                       # standing on it: in sight, out of band
    assert not bool(task.success(s, goal).any())


def test_observe_pays_for_the_BAND_not_for_closing_the_distance():
    """The objective has to agree with the controller's equilibrium.

    `StandoffBowl` holds a shell, but while the cost still paid for closing the
    distance the cheapest behaviour was to fly onto the target -- where sight and
    aim are trivially satisfied.  It did exactly that: `band` fell 0.918 -> 0.012
    over 100 generations while visibility stayed near 0.85.  A contradiction
    between what the potential holds and what the cost pays for is resolved in
    favour of the cost.
    """
    from lagrangian_es.tasks import make_task
    system = make_system("quadrotor_nav", environment="pillars", dtype=DT)
    task = make_task("observe", system)
    g = torch.zeros(1, 3, dtype=DT)
    mid = 0.5 * (task.r_near + task.r_far)
    at_target = float(task.position_cost(torch.tensor([[0.05, 0, 0]], dtype=DT),
                                         g, 1e-2))
    in_band = float(task.position_cost(torch.tensor([[mid, 0, 0]], dtype=DT),
                                       g, 1e-2))
    too_far = float(task.position_cost(
        torch.tensor([[task.r_far + 1.0, 0, 0]], dtype=DT), g, 1e-2))
    assert in_band < at_target, "sitting on the target must not be the cheapest"
    assert in_band < too_far
    # flat across the band: no gradient pulling it to one edge
    for r in (task.r_near + 0.1, mid, task.r_far - 0.1):
        c = float(task.position_cost(torch.tensor([[r, 0, 0]], dtype=DT), g, 1e-2))
        assert abs(c - in_band) < 1e-6


def test_waypoint_position_cost_is_unchanged():
    """The hook exists for the observation task; every waypoint task must still
    pay plain distance."""
    from lagrangian_es.tasks import make_task
    system = make_system("quadrotor_nav", environment="pillars", dtype=DT)
    task = make_task("waypoint_pair", system)
    g = torch.zeros(4, 3, dtype=DT)
    x = torch.tensor([[0.3, 0, 0], [1.2, 0, 0], [3.0, 0, 0], [0, 2.0, 0]],
                     dtype=DT)
    want = torch.sqrt((x * x).sum(-1) + 1e-2)
    assert torch.allclose(task.position_cost(x, g, 1e-2), want)
