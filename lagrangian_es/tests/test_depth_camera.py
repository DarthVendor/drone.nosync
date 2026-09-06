"""`DepthCamera` -- a body-mounted forward depth camera."""
import math

import pytest
import torch

from lagrangian_es.sensors import make_sensor
from lagrangian_es.sensors.range_sensor import RangeSensor
from lagrangian_es.systems import make_system
from lagrangian_es.util import make_gen

DT = torch.float64


def _rig(env="pillars", **kw):
    system = make_system("quadrotor_nav", environment=env, dtype=DT)
    return system, make_sensor("depth_camera", system, sigma=0.0, **kw)


def test_camera_looks_where_the_vehicle_points():
    """Body-mounted, so its readings are organised around the vehicle's heading
    rather than a world compass.  If it did not rotate with the body it would be
    a differently-shaped fan, not a camera."""
    system, cam = _rig()
    s = system.reset(1, make_gen(0))
    s["R"] = torch.eye(3, dtype=DT)[None]
    # averaging unit rays over a wide FOV shortens the mean (0.88 here), so
    # check its DIRECTION, not its length
    fwd = cam._dirs(s)[0].mean(0)
    fwd = fwd / fwd.norm()
    assert float(fwd[0]) > 0.99, "boresight is not along body +x"
    # yaw by 90 deg: the boresight must follow
    c, sn = math.cos(math.pi / 2), math.sin(math.pi / 2)
    s["R"] = torch.tensor([[[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]],
                          dtype=DT)
    fwd2 = cam._dirs(s)[0].mean(0)
    fwd2 = fwd2 / fwd2.norm()
    assert float(fwd2[1]) > 0.99 and abs(float(fwd2[0])) < 0.05


def test_camera_has_vertical_extent_where_the_fan_has_none():
    """The whole reason it exists.  Every RangeSensor beam has z = 0, so the fan
    is blind off the vehicle's own altitude -- on the hoop scenes it returns 0 of
    192 hits."""
    system, cam = _rig()
    s = system.reset(1, make_gen(0))
    s["R"] = torch.eye(3, dtype=DT)[None]
    z = cam._dirs(s)[0][:, 2]
    assert float(z.max()) > 0.2 and float(z.min()) < -0.2
    fan = RangeSensor(system, n_beams=24, sigma=0.0)
    assert torch.allclose(fan._dirs(s)[0][:, 2], torch.zeros(24, dtype=DT))


def test_camera_sees_hoops_that_the_horizontal_fan_misses():
    system, cam = _rig("hoops")
    fan = RangeSensor(system, n_beams=24, sigma=0.0)
    s = system.reset(16, make_gen(0))
    cam_hits = int((cam.observe(s, None) < cam.max_range * 0.98).sum())
    fan_hits = int((fan.observe(s, None) < fan.max_range * 0.98).sum())
    assert cam_hits > fan_hits, f"camera {cam_hits} vs fan {fan_hits}"


def test_rays_are_unit_length_and_span_the_declared_field_of_view():
    system, cam = _rig(hfov=1.4, vfov=1.05)
    s = system.reset(1, make_gen(0))
    s["R"] = torch.eye(3, dtype=DT)[None]
    d = cam._dirs(s)[0]
    assert torch.allclose(d.norm(dim=-1), torch.ones(d.shape[0], dtype=DT))
    az = torch.atan2(d[:, 1], d[:, 0])
    el = torch.asin(d[:, 2].clamp(-1, 1))
    assert float(az.abs().max()) <= cam.hfov / 2 + 1e-9
    assert float(el.abs().max()) <= cam.vfov / 2 + 1e-9


def test_jacobian_shape_and_finiteness():
    system, cam = _rig()
    s = system.reset(8, make_gen(0))
    J = cam.jacobian(s)
    assert J.shape == (8, cam.obs_dim, system.task_dim)
    assert torch.isfinite(J).all()


def test_a_miss_reads_as_far_not_as_a_sentinel():
    """'Nothing there' must never be confusable with 'something close'."""
    system, cam = _rig("empty")
    s = system.reset(8, make_gen(0))
    d = cam.observe(s, None)
    assert torch.all(d >= cam.max_range - 1e-6)


def test_camera_refreshes_every_step_like_the_range_fan():
    """Striding an obstacle sensor doubled the crash rate (0.027 -> 0.058)."""
    assert make_sensor("depth_camera",
                       make_system("quadrotor_nav")).update_every == 1


def test_obs_dim_matches_the_grid():
    system, cam = _rig(width=8, height=6)
    assert cam.obs_dim == 48
    s = system.reset(4, make_gen(0))
    assert cam.observe(s, None).shape == (4, 48)


def test_camera_is_usable_by_the_obstacle_terms_unchanged():
    """The terms consume a depth vector plus its position Jacobian, which is
    exactly what this provides -- so no term needed editing to accept it."""
    from lagrangian_es.trainables.sensor_terms import RangeBarrier, RangeDamper
    system, cam = _rig()
    s = system.reset(4, make_gen(0))
    obs = {cam.name: cam.observe(s, None), cam.name + "/J": cam.jacobian(s)}
    e = torch.ones(4, 3, dtype=DT)
    v = torch.zeros(4, 3, dtype=DT)
    x = system.task_position(s)
    for term in (RangeBarrier(3, cam.name, cam.obs_dim),
                 RangeDamper(3, cam.name, cam.obs_dim)):
        g = term.grad_potential(term.init(dtype=DT), e, v, x, obs)
        assert g.shape == (4, 3) and torch.isfinite(g).all()
