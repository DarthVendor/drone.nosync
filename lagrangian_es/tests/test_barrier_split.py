"""`barrier` is two terms doing OPPOSITE jobs, and one `lam` scaled them together.

    OCCLUSION  charges a subgoal lying BEHIND what a beam saw -- unreachable.
               A reactive low level cannot know its target is unreachable, so
               this is genuine ROUTING, and it reads the beams.
    CLEARANCE  charges a subgoal near a wall. The low level already does this
               with the same beams and prox_gain 30: AVOIDANCE, duplicated.

The dose-response that set `var_lam` to 0 moved both at once (lam 0 +0.027
t+2.58, lam 2 -0.075 t-4.42), and the winning arm consults NO SENSOR at all --
`|sub| + |goal-sub|` is pure geometry and always elects straight-to-goal. So
that measurement never established that routing beats silence. Weighting the
halves separately is how a sensor-USING rule gets measured on its own.
"""
import torch

from lagrangian_es.composer.variational import barrier, path_action


def _scene(n_beam=16):
    dirs = torch.zeros(n_beam, 3)
    ang = torch.linspace(-torch.pi, torch.pi, n_beam + 1)[:n_beam]
    dirs[:, 0] = ang.cos(); dirs[:, 1] = ang.sin()
    return dirs


def test_the_two_halves_sum_to_the_whole():
    torch.manual_seed(0)
    d = _scene()
    rng = torch.rand(4, 16) * 5 + 1
    hit = torch.ones(4, 16, dtype=torch.bool)
    sub = torch.randn(4, 7, 3) * 3
    both = barrier(sub, d, rng, hit)
    occ = barrier(sub, d, rng, hit, w_near=0.0)
    near = barrier(sub, d, rng, hit, w_occ=0.0)
    assert torch.allclose(both, occ + near, atol=1e-5), "the split must be exact"
    assert float(occ.abs().sum()) > 0 and float(near.abs().sum()) > 0


def test_occlusion_charges_a_subgoal_behind_a_wall_and_clearance_does_not():
    """A wall dead ahead at 4 m, a subgoal 8 m dead ahead. It is UNREACHABLE --
    occlusion must charge it heavily. Clearance barely notices: the subgoal sits
    4 m from the return point, which is nowhere near it. That asymmetry is the
    whole reason the two cannot share a weight."""
    d = _scene()
    rng = torch.full((1, 16), 6.0)
    rng[0, 0] = 4.0                                   # a wall straight ahead
    hit = torch.ones(1, 16, dtype=torch.bool)
    behind = torch.tensor([[[8.0, 0.0, 0.0]]])        # past the wall, same bearing
    occ = float(barrier(behind, d, rng, hit, w_near=0.0))
    near = float(barrier(behind, d, rng, hit, w_occ=0.0))
    assert occ > 1.0, f"a subgoal behind a wall must be charged as unreachable ({occ:.3f})"
    assert occ > near, f"occlusion must dominate here, not clearance ({occ:.3f} vs {near:.3f})"


def test_lam_zero_reads_no_sensor_at_all():
    """Why the lam 0 'win' proves nothing about routing: with the barrier off
    the score is pure geometry, so the beams cannot change the ranking. The
    same candidates score identically whether the scene is empty or a wall is
    dead ahead."""
    torch.manual_seed(0)
    d = _scene()
    sub = torch.randn(1, 9, 3) * 3
    g = torch.tensor([[6.0, 0.0, 0.0]])
    hit = torch.ones(1, 16, dtype=torch.bool)
    empty = torch.full((1, 16), 6.0)
    walled = torch.full((1, 16), 6.0); walled[0, 0] = 1.0
    s_empty = path_action(sub, g, d, empty, hit, lam=0.0)
    s_walled = path_action(sub, g, d, walled, hit, lam=0.0)
    assert torch.allclose(s_empty, s_walled), \
        "lam 0 is sensor-blind: a wall dead ahead must not change a single score"
    # and with occlusion on, it must
    o_empty = path_action(sub, g, d, empty, hit, lam=2.0, w_near=0.0)
    o_walled = path_action(sub, g, d, walled, hit, lam=2.0, w_near=0.0)
    assert not torch.allclose(o_empty, o_walled), \
        "occlusion must make the scene matter, or it is not a sensor-using rule"
