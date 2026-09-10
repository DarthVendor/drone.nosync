"""The corridor city: every waypoint is inside a corridor, nothing is open ground."""
import math
import torch


def _rig():
    from lagrangian_es.systems import make_system
    from lagrangian_es.sensors import make_sensor
    sysm = make_system("quadrotor_nav", environment="corridors", free_start=True, dtype=torch.float64)
    return sysm, make_sensor("range", sysm, spread=2 * math.pi)


def test_every_waypoint_is_clear_of_the_walls():
    from lagrangian_es.util import make_gen
    sysm, _ = _rig()
    wps = torch.as_tensor(sysm.env.waypoints, dtype=torch.float64)
    assert wps.shape[0] >= 90, wps.shape
    s = sysm.reset(wps.shape[0], make_gen(0))
    f = {k: v for k, v in s.items() if k not in ("p", "v", "R", "om")}
    d = sysm.env.sdf(wps, f)
    assert float(d.min()) > 1.5, f"a waypoint is {float(d.min()):.2f} m from a wall; streets are 4 m wide"


def test_every_waypoint_is_enclosed():
    """At least half of a 360-degree fan returns inside 6 m from every
    waypoint: a junction sees walls on four corners, a street on two sides."""
    from lagrangian_es.util import make_gen
    sysm, fan = _rig()
    wps = torch.as_tensor(sysm.env.waypoints, dtype=torch.float64)
    s = sysm.reset(wps.shape[0], make_gen(1))
    s = dict(s); s["p"] = wps.clone(); s["v"] = torch.zeros_like(wps)
    s["R"] = torch.eye(3, dtype=torch.float64).expand(wps.shape[0], 3, 3).clone(); s["om"] = torch.zeros_like(wps)
    r = fan.measure(s)[0]
    hit = (r < 0.98 * fan.max_range).double().mean(dim=-1)
    assert float(hit.min()) >= 0.5, f"a waypoint sees open ground: {float(hit.min()):.2f} of its beams return"
    assert float(hit.mean()) >= 0.6, float(hit.mean())          # junctions 0.625, street midpoints 0.75


def test_the_tour_has_straight_legs_and_legs_round_a_corner():
    from lagrangian_es.tasks import make_task
    from lagrangian_es.util import make_gen
    sysm, _ = _rig()
    task = make_task("city_tour", sysm, gating="arrival", n_legs=2, max_leg=20.0)
    g = task.sample(512, make_gen(3))
    d = (g[:, 1] - g[:, 0]).abs()
    straight = ((d[:, 0] < 1e-6) | (d[:, 1] < 1e-6)).double().mean()
    assert 0.2 < float(straight) < 0.8, float(straight)
