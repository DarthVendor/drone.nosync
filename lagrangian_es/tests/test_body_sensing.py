"""Sensors mounted on the body, and a low level that reads itself.

The range fans turn with yaw and dip with tilt because they are bolted to the
airframe; the tilt sensor reports the body z-axis, which is what an IMU gives a
flight controller; a downward fan reads the ground.  The learned term reads all
of them as one observation, and a genome that never saw the new channels must
fly EXACTLY as before once extended -- extension is a warm start, not a restart.
"""
import math

import torch

from lagrangian_es.config import Config
from lagrangian_es.es import build, build_sensors
from lagrangian_es.sensors import make_sensor
from lagrangian_es.trainables.learned import LearnedShaping
from lagrangian_es.util import make_gen
from lagrangian_es.widen import extend_inputs


def _sys(sensors=("range",), sensor_kw=(), extra=()):
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="waypoint_pair",
                 environment="pillars", sensors=sensors, sensor_kw=sensor_kw, gating="arrival", seed=0,
                 trainable_kw=(("learned", True), ("damp_mode", "beams"), ("extra_obs", extra)))
    sysm, tr, task = build(cfg)
    return cfg, sysm, tr, task


def _rot_y(pitch):
    c, s = math.cos(pitch), math.sin(pitch)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


def test_beams_dip_with_pitch():
    """A nose-down pitch tilts the forward beam into the ground."""
    cfg, sysm, tr, task = _sys(sensor_kw=(("range", (("spread", 2.0944),)),))
    sen = build_sensors(cfg, sysm)[0]
    s = sysm.reset(1, make_gen(0)); s = dict(s); s["R"] = torch.eye(3, dtype=torch.float64)[None]
    level = sen._dirs(s)[0]
    s["R"] = _rot_y(0.4)[None]
    pitched = sen._dirs(s)[0]
    mid = level.shape[0] // 2                       # the central, forward-most beam
    assert abs(float(level[mid, 2])) < 1e-9
    assert float(pitched[mid, 2]) < -0.3, "forward beam did not dip with the nose"
    assert torch.allclose(pitched.norm(dim=-1), torch.ones(pitched.shape[0], dtype=torch.float64))


def test_front_fan_covers_only_the_front():
    cfg, sysm, tr, task = _sys(sensor_kw=(("range", (("spread", 2.0944),)),))
    sen = build_sensors(cfg, sysm)[0]
    s = sysm.reset(1, make_gen(0)); s = dict(s); s["R"] = torch.eye(3, dtype=torch.float64)[None]
    d = sen._dirs(s)[0]
    assert float(d[:, 0].min()) > 0.4, "a 120-degree fan must not look behind"


def test_downward_fan_and_tilt_sensor():
    cfg, sysm, tr, task = _sys(sensors=("range", "range_down", "tilt"))
    sens = build_sensors(cfg, sysm)
    names = [x.name for x in sens]
    assert names == ["range", "range_down", "tilt"], names
    s = sysm.reset(3, make_gen(1))
    down = sens[1]._dirs(s)
    assert float(down[..., 2].max()) < -0.7, "the downward fan must point down"
    assert sens[1].obs_dim == 4 and sens[1].max_range == 2.0
    z = sens[2].observe(s, make_gen(2))
    assert z.shape == (3, 3) and torch.allclose(z, s["R"][..., :, 2])


def test_extended_genome_flies_identically_with_the_new_channels():
    """Extra channels with zero input rows: same forces, bit for bit, on the
    same state and beams -- and a different output only once they are used."""
    cfg0, sysm, tr0, task = _sys()
    cfg1, _, tr1, _ = _sys(sensors=("range", "range_down", "tilt"),
                           extra=(("range_down", 4), ("tilt", 3)))
    t0, t1 = tr0.terms[0], tr1.terms[0]
    assert t1.n_obs == 31 and t1.n_beams == 24 and t1.beam_name == "range"
    th0 = tr0.init(); th1 = extend_inputs(th0, t0, t1)
    assert th1.shape[-1] == tr1.dim
    s = sysm.reset(5, make_gen(3)); goal = task.sample(5, make_gen(4))[:, 0]
    sens = build_sensors(cfg1, sysm)
    obs = {x.name: x.observe(s, make_gen(5)) for x in sens}
    obs0 = {"range": obs["range"]}
    # to rounding: a wider W1 changes the matmul's summation order, nothing else
    assert torch.allclose(tr0.forward(th0, s, goal, obs0), tr1.forward(th1, s, goal, obs), atol=1e-10, rtol=0)
    th2 = th1.clone(); a, b = t1._sl["W1"]
    th2[a:b] = th2[a:b] + 0.05                       # give the new rows something to say
    assert not torch.equal(tr1.forward(th1, s, goal, obs), tr1.forward(th2, s, goal, obs))


def test_heading_gate_zero_is_the_plants_own_lookat_and_one_follows_the_command():
    cfg, sysm, tr, task = _sys()
    s = sysm.reset(4, make_gen(6)); goal = task.sample(4, make_gen(7))[:, 0]
    obs = {"range": build_sensors(cfg, sysm)[0].observe(s, make_gen(8))}
    th = tr.init()
    B = 4; zero = torch.zeros(B, 3, dtype=torch.float64); one = torch.ones(B, 1, dtype=torch.float64)
    psi = torch.atan2(s["R"][:, 1, 0], s["R"][:, 0, 0])
    cmd = psi + 1.2
    u_plain = tr.forward(th, s, goal, obs)
    u_gate0 = tr.forward(th, s, goal, obs, spec=(zero, one, cmd, torch.zeros(B, dtype=torch.float64)))
    u_gate1 = tr.forward(th, s, goal, obs, spec=(zero, one, cmd, torch.ones(B, dtype=torch.float64)))
    assert torch.allclose(u_plain, u_gate0, atol=1e-12), "gate 0 must leave the look-at untouched"
    assert not torch.allclose(u_plain, u_gate1), "gate 1 must turn the reference toward the command"


def test_the_command_works_in_the_learned_yaw_mode_too():
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="waypoint_pair", environment="pillars",
                 sensors=("range",), gating="arrival", seed=0, system_kw=(("yaw_mode", "learned"),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")))
    sysm, tr, task = build(cfg)
    assert tr.dim - tr.policy_dim == 9, "learned yaw mode carries three look-at weights"
    s = sysm.reset(4, make_gen(6)); goal = task.sample(4, make_gen(7))[:, 0]
    obs = {"range": build_sensors(cfg, sysm)[0].observe(s, make_gen(8))}
    th = tr.init(); B = 4
    zero = torch.zeros(B, 3, dtype=torch.float64); one = torch.ones(B, 1, dtype=torch.float64)
    psi = torch.atan2(s["R"][:, 1, 0], s["R"][:, 0, 0]); cmd = psi + 1.2
    u_plain = tr.forward(th, s, goal, obs)
    u_g0 = tr.forward(th, s, goal, obs, spec=(zero, one, cmd, torch.zeros(B, dtype=torch.float64)))
    u_g1 = tr.forward(th, s, goal, obs, spec=(zero, one, cmd, torch.ones(B, dtype=torch.float64)))
    assert torch.allclose(u_plain, u_g0, atol=1e-12)
    assert not torch.allclose(u_plain, u_g1)


def test_reset_yaw_spreads_headings_over_the_circle():
    from lagrangian_es.systems import make_system
    sysm = make_system("quadrotor_nav", dtype=torch.float64, environment="pillars", reset_yaw=3.14159265)
    s = sysm.reset(512, make_gen(9))
    R = s["R"]
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(512, 3, 3), atol=1e-9)
    psi = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    hist = torch.histc(psi, bins=8, min=-3.1416, max=3.1416)
    assert float(hist.min()) > 512 / 8 * 0.5, hist        # every octant populated
    sysm0 = make_system("quadrotor_nav", dtype=torch.float64, environment="pillars")
    psi0 = torch.atan2(sysm0.reset(64, make_gen(9))["R"][:, 1, 0], sysm0.reset(64, make_gen(9))["R"][:, 0, 0])
    assert float(psi0.abs().max()) < 0.2, "the default keeps the old small attitude noise"


def test_the_plants_lookat_prior_turns_the_nose_toward_the_goal():
    """With the look-at slots at the plant's prior (1, 0, 0), a vehicle facing
    away from its goal gets a yaw reference toward it; with them at zero it
    held heading -- which, with a forward fan, was flying blind."""
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="waypoint_pair", environment="pillars",
                 sensors=("range",), gating="arrival", seed=0, system_kw=(("yaw_mode", "learned"), ("reset_yaw", 3.14159265)),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")))
    sysm, tr, task = build(cfg)
    s = sysm.reset(16, make_gen(21)); goal = task.sample(16, make_gen(22))[:, 0]
    obs = {"range": build_sensors(cfg, sysm)[0].observe(s, make_gen(23))}
    th = tr.init(); assert torch.equal(th[-3:], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    th0 = th.clone(); th0[-3:] = 0.0
    u_prior = tr.forward(th, s, goal, obs); u_zero = tr.forward(th0, s, goal, obs)
    # the two differ only in the yaw reference; a prior that looks at the goal must change the command
    assert not torch.allclose(u_prior, u_zero)
    # and the change is a yaw torque, not a thrust change
    assert torch.allclose(u_prior[:, 0], u_zero[:, 0], atol=1e-9)
