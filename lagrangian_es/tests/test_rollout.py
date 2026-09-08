

def test_the_charge_on_a_tour_is_the_distance_to_go_and_never_rises_for_finishing_a_leg():
    """On a multi-leg tour the position charge is the distance to the active
    waypoint PLUS the legs after it, so an arrival is continuous: standing on
    waypoint 1 costs the same the step before and the step after the leg
    advances.  A learned composer parked 0.28 m short of the first waypoint
    when the charge jumped there."""
    import torch
    from lagrangian_es.rollout import Rollout
    g = torch.tensor([[[0.0, 0.0, 1.0], [3.0, 4.0, 1.0], [3.0, 4.0, 13.0]],       # legs of 5 m and 12 m
                      [[0.0, 0.0, 1.0], [6.0, 8.0, 1.0], [6.0, 8.0, 1.0]]])       # 10 m, then a padded leg
    tab = Rollout._to_go(g)
    assert torch.allclose(tab, torch.tensor([[17.0, 12.0, 0.0], [10.0, 0.0, 0.0]])), tab
    assert Rollout._to_go(g[:, :1]) is None
    x = g[:, 0]                                                                       # standing on the first waypoint
    d0 = (x - g[:, 0]).norm(dim=-1) + tab[:, 0]                                       # charge while still on leg 0: 0 + 17
    d1 = (x - g[:, 1]).norm(dim=-1) + tab[:, 1]                                       # charge the step after the leg advances: 5 + 12
    assert torch.allclose(d0, d1), (d0, d1)
    assert torch.allclose(d0, torch.tensor([17.0, 10.0]))
