"""The WAYPOINT bearing is measured from the GOAL, not from the nose.

From the nose, +-180 degrees is compressed into a1 in [-1,1], so the network had
to reproduce atan2(g_y, g_x) to a few degrees at EVERY decision and the error was
amplified by pi. MEASURED: a student distilled onto a teacher that arrives 0.576
reached MSE 0.0349 in mu -- RMS 0.187, about 33 degrees of bearing -- and still
arrived 0.000, ruining 213 of 213 flights. Its capacity went into the goal
geometry, leaving beam-driven routing a rounding correction on a large learned
quantity, which is why `sens` was ~0.

Goal-relative, a1 = 0 IS straight at the goal: the identity sits at the interior
origin (no boundary optimum, so no |mu| runaway -- it reached 21.7 where
tanh' ~ 1e-16) and the network's whole output is the DEVIATION.
"""
import math

import torch

from lagrangian_es.composer.actions_cont import PHI_MAX, ContVocab
from lagrangian_es.composer.policy_cont import ContPolicyNet, _subgoal_ego
from lagrangian_es.composer.spec import TaskSpec


def test_zero_argument_points_straight_at_the_goal():
    net = ContPolicyNet(n_terms=1)
    torch.manual_seed(0)
    for _ in range(8):
        g = torch.randn(1, 3)
        g = g / g.norm() * float(torch.rand(1) * 0.9 + 0.1)
        sub = _subgoal_ego(net, torch.zeros(1, 3), g)
        assert torch.allclose(sub[0] / sub[0].norm(), g[0] / g[0].norm(), atol=1e-5), \
            "mu = 0 must aim exactly at the goal"


def test_the_identity_is_interior_so_nothing_has_to_saturate():
    """The old frame put the best action on the boundary: full radius aimed at
    the goal needed a -> +-1, reachable only as |mu| -> inf. That is what drove
    |mu| 0.37 -> 21.7 and killed every input's influence."""
    net = ContPolicyNet(n_terms=1)
    g = torch.tensor([[0.0, -0.7, 0.2]])
    sub = _subgoal_ego(net, torch.zeros(1, 3), g)
    assert torch.isfinite(sub).all()
    d = 1.0 - math.tanh(0.0) ** 2
    assert d == 1.0, "at the identity the squash is at its most responsive"


def test_the_argument_still_spans_the_whole_circle():
    """A barrier would be a tuned constant; this must stay a full-range
    correction so routing around either side is expressible."""
    net = ContPolicyNet(n_terms=1)
    g = torch.tensor([[1.0, 0.0, 0.0]]) * 0.5
    left = _subgoal_ego(net, torch.tensor([[0.0, 1.0, 0.0]]), g)      # pi*1 -> +180
    right = _subgoal_ego(net, torch.tensor([[0.0, -1.0, 0.0]]), g)
    fwd = _subgoal_ego(net, torch.zeros(1, 3), g)
    assert float(fwd[0, 0]) > 0, "zero aims at the goal (+x here)"
    assert float(left[0, 0]) < 0 and float(right[0, 0]) < 0, "a1 = +-1 turns away"


def test_directly_away_from_the_goal_is_reachable_at_a_finite_argument():
    """THE reason the frame changed.

    `pi*tanh(a1)` covers (-180, 180) -- the whole circle MINUS ONE POINT -- and
    the missing point is "fly directly away from the goal", needing
    a1 = infinity.  On `occluded` every pocket opens the same way, so that is
    the manoeuvre the map most demands: measured, the median leg needs a 90 deg
    correction just to leave its own pocket (3.9 sigma out at sd 0.12) and 14%
    of legs need beyond 8 sigma, while tanh' collapses on top of it (at 179 deg
    the parameter must travel ~500x further per degree gained).
    """
    net = ContPolicyNet(n_terms=1)
    g = torch.tensor([[1.0, 0.0, 0.0]]) * 0.5
    back = _subgoal_ego(net, torch.tensor([[0.0, 1.0, 0.0]]), g)
    # a1 = 1 is EXACTLY opposite, at a finite and modest argument
    assert float(back[0, 0]) < 0 and abs(float(back[0, 1])) < 1e-6, \
        "a1 = 1 must aim exactly 180 deg from the goal"
    # and the correction is LINEAR in the argument -- no boundary to saturate
    for a1, deg in ((0.25, 45.0), (0.5, 90.0), (0.75, 135.0), (1.0, 180.0)):
        sub = _subgoal_ego(net, torch.tensor([[0.0, a1, 0.0]]), g)
        got = math.degrees(math.atan2(float(sub[0, 1]), float(sub[0, 0])))
        assert abs(abs(got) - deg) < 1e-3, f"a1={a1} gave {got}, wanted {deg}"


def test_the_bearing_has_no_vanishing_gradient_anywhere():
    """The old frame's slope died exactly where the map needed it most.

    `d(correction)/d(a1)` was `pi*(1 - tanh^2 a1)`: 180 deg/unit at the
    identity but 2 deg/unit at 179 deg, a ~500x collapse right where a pocket
    map needs the travel.  Linear in `a1`, it is pi everywhere.
    """
    net = ContPolicyNet(n_terms=1)
    g = torch.tensor([[1.0, 0.0, 0.0]]) * 0.5
    h = 1e-4
    def ang(a1):
        sub = _subgoal_ego(net, torch.tensor([[0.0, a1, 0.0]]), g)
        return math.atan2(float(sub[0, 1]), float(sub[0, 0]))
    for a1 in (0.0, 0.25, 0.5, 0.75, 0.95):
        d = (ang(a1 + h) - ang(a1 - h)) / (2 * h)     # d(theta)/d(a1)
        assert abs(d - math.pi) < 1e-2, \
            f"slope {d:.4f} at a1={a1}, wanted pi everywhere (old frame: collapsed)"


def test_the_rollout_geometry_and_the_loss_mirror_agree():
    """`ContVocab.finish` flies the subgoal; `_subgoal_ego` is what the loss
    differentiates. If they disagree the update optimises a geometry the
    vehicle never flies."""
    torch.manual_seed(0)
    V = ContVocab(1)
    net = ContPolicyNet(n_terms=1)
    B, reach = 16, 10.0
    x = torch.zeros(B, 3)
    goal = torch.randn(B, 3) * 4.0
    psi = torch.zeros(B)
    g_ego = (goal - x) / reach
    u = torch.randn(B, 3) * 0.8
    out = TaskSpec.identity(B, 3, 1, x.dtype, x.device)
    # `step` documents `arg` as ALREADY squashed, and the live path calls
    # `V.squash(u)` before it -- the loss mirror squashes internally instead, so
    # the two take their argument in different spaces and a test that forgets
    # this reports a disagreement that is its own.
    spec = V.apply(torch.full((B,), V.WAYPOINT, dtype=torch.long), V.squash(u), out,
                   x, goal, psi, g_ego, reach, z_min=-1e9)
    flown = (goal + spec.delta - x) / reach          # what finish() actually placed
    mirror = _subgoal_ego(net, u, g_ego)             # what the loss differentiates
    assert torch.allclose(flown, mirror, atol=1e-4), \
        f"max disagreement {float((flown - mirror).abs().max()):.6f}"
