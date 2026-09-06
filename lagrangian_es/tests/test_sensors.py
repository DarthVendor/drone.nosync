"""Sensing seam: conformance, delay, common random numbers, and the identity gate."""
import pytest
import torch

from lagrangian_es.config import Config, ESCfg, RolloutCfg
from lagrangian_es.es import build, train
from lagrangian_es.evaluate import evaluate
from lagrangian_es.rollout import Rollout
from lagrangian_es.sensors import SENSORS, DelayBuffer, make_sensor
from lagrangian_es.systems import make_system
from lagrangian_es.tasks import make_task
from lagrangian_es.trainables import make_trainable
from lagrangian_es.util import make_gen

DT = torch.float64


def _rig(sensors=None, n_eps=4):
    system = make_system("quadrotor")
    tr = make_trainable("energy_shaping", system)
    task = make_task("waypoint_pair", system)
    cfg = RolloutCfg(ep_steps=120, n_eps=n_eps)
    sens = [make_sensor(n, system) for n in (sensors or [])]
    return system, tr, task, cfg, Rollout(system, tr, task, cfg, sens)


# --------------------------------------------------------------------------- #
# conformance
# --------------------------------------------------------------------------- #
SENSOR_PAIRS = [(s, n) for n in sorted(SENSORS)
                for s in ("quadrotor", "quadrotor_nav")
                if SENSORS[n].supports(make_system(s))]


@pytest.mark.parametrize("sysname,name", SENSOR_PAIRS)
def test_sensor_conformance(sysname, name):
    system = make_system(sysname)
    sen = make_sensor(name, system)
    s = system.reset(6, make_gen(0))

    obs = sen.observe(s, make_gen(1))
    assert obs.shape == (6, sen.obs_dim), "declared obs_dim must match what is reported"
    assert torch.isfinite(obs).all()

    J = sen.jacobian(s)
    assert J.shape == (6, sen.obs_dim, system.task_dim)
    assert torch.isfinite(J).all()

    v = sen.valid(s)
    assert v.shape == (6, sen.obs_dim) and v.dtype == torch.bool
    assert sen.kind in ("position_like", "velocity_like", "range")
    assert sen.latency_steps >= 0


@pytest.mark.parametrize("sysname,name", SENSOR_PAIRS)
def test_sensor_jacobian_is_finite(sysname, name):
    """The pullback must be finite everywhere the vehicle can be, including where
    a beam misses entirely -- a clamped range has zero derivative, not NaN."""
    system = make_system(sysname)
    sen = make_sensor(name, system)
    s = system.reset(16, make_gen(0))
    J = sen.jacobian(s)
    assert torch.isfinite(J).all()
    assert J.shape == (16, sen.obs_dim, system.task_dim)


def test_default_strides_reflect_what_each_sensor_costs_and_buys():
    """A 10 Hz refresh against a 50 Hz loop is realistic and 4-7x cheaper on
    ray-traced scenes, and it is still the default for the expensive sensors.

    `range` is the exception, and it was measured: striding it leaves the vehicle
    blind for 0.1 s, which is 30 cm at 3 m/s, and that doubles the crash rate
    (0.027 -> 0.058 on the pillar field with everything else held fixed).  For an
    obstacle sensor that is the wrong trade, so it refreshes every step.

    `FullState` is pinned to every step for a different reason: it is the
    identity baseline the sensor-free path has to reproduce bit-for-bit, and
    striding it would make it something else."""
    system = make_system("quadrotor")
    assert make_sensor("range", make_system("quadrotor_nav")).update_every == 1
    assert make_sensor("landmark_camera", system).update_every == 5
    assert make_sensor("noisy_position", system).update_every == 5
    assert make_sensor("full_state", system).update_every == 1
    assert make_sensor("full_state_velocity", system).update_every == 1


def test_stride_holds_the_last_measurement():
    """Between refreshes the controller sees the previous reading, not a fresh
    one -- and that staleness has to actually change the flight, or the stride is
    not doing anything."""
    from lagrangian_es.tasks import make_task as _mt
    system = make_system("quadrotor_nav", environment="pillars")
    tr = make_trainable("nav_agent", system)
    task = _mt("waypoint_pair", system, gating="arrival")
    goals = task.sample(4, make_gen(0))
    TH = tr.init()[None]
    out = {}
    for k in (1, 5):
        sen = make_sensor("range", system, n_beams=12)
        sen.update_every = k
        out[k] = float(Rollout(system, tr, task, RolloutCfg(n_eps=4),
                               [sen]).run(TH, goals, 3).fitness)
    assert out[1] != out[5], "striding the sensor changed nothing"


def test_range_sensor_needs_an_environment():
    assert not SENSORS["range"].supports(make_system("quadrotor"))
    assert SENSORS["range"].supports(make_system("quadrotor_nav"))


def test_full_state_is_the_identity():
    system = make_system("quadrotor")
    sen = make_sensor("full_state", system)
    s = system.reset(5, make_gen(2))
    assert torch.equal(sen.observe(s, make_gen(0)), system.task_position(s))
    assert torch.equal(sen.jacobian(s)[0], torch.eye(3, dtype=DT))
    assert sen.latency_steps == 0


# --------------------------------------------------------------------------- #
# delay
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("L", [0, 1, 3, 7])
def test_delay_buffer_returns_the_observation_from_exactly_k_steps_prior(L):
    b = DelayBuffer(L)
    b.reset(torch.full((1,), -1.0, dtype=DT))
    got = [float(b.push(torch.full((1,), float(t), dtype=DT))) for t in range(12)]
    for t in range(12):
        want = float(t - L) if t >= L else -1.0
        assert got[t] == want, f"latency {L} at step {t}: {got[t]} != {want}"
    assert len(b) == L + 1


def test_zero_latency_buffer_is_a_passthrough():
    b = DelayBuffer(0)
    x = torch.randn(4, 3, dtype=DT)
    b.reset(x)
    assert torch.equal(b.push(x), x)


def test_delay_actually_changes_the_closed_loop():
    """Delay costs omega*tau of phase margin.  If a buffer of depth k did not
    change the trajectory, it would not be protecting against anything."""
    system, tr, task, cfg, r0 = _rig(["full_state"])
    sen = make_sensor("full_state", system, latency_steps=6)
    r6 = Rollout(system, tr, task, cfg, [sen])
    goals = task.sample(cfg.n_eps, make_gen(3))
    TH = tr.init()[None]
    a = r0.run(TH, goals, seed=1)
    b = r6.run(TH, goals, seed=1)
    # the controller here ignores obs, so fitness must match; the buffer itself
    # must still be delivering different values
    assert torch.equal(a.fitness, b.fitness)
    s = system.reset(4, make_gen(0))
    buf = DelayBuffer(6)
    buf.reset(sen.observe(s, make_gen(0)))
    first = buf.push(torch.full_like(sen.observe(s, make_gen(0)), 99.0))
    assert not torch.allclose(first, torch.full_like(first, 99.0))


# --------------------------------------------------------------------------- #
# THE regression gate
# --------------------------------------------------------------------------- #
def test_full_state_reproduces_the_sensorless_path_bit_identically():
    """With `FullState` and latency 0, every acceptance number must reproduce
    BIT-identically.  If it does not, the sensor seam is in the wrong place and
    nothing else in the sensing addendum should be built until it is."""
    system, tr, task, cfg, plain = _rig(None)
    _, _, _, _, sensed = _rig(["full_state"])
    goals = task.sample(cfg.n_eps, make_gen(7))
    TH = tr.init()[None].expand(6, -1)

    a, b = plain.run(TH, goals, seed=11), sensed.run(TH, goals, seed=11)
    assert torch.equal(a.fitness, b.fitness), "fitness drifted"
    for f in ("cost", "alive", "leg_err", "final_err", "saturation", "effort"):
        assert torch.equal(getattr(a, f), getattr(b, f)), f"{f} drifted"

    ta, tb = plain.trace(TH[:1], goals, 12), sensed.trace(TH[:1], goals, 12)
    for k in ta.states:
        assert torch.equal(ta.states[k], tb.states[k]), f"trace state {k!r} drifted"
    assert torch.equal(ta.us, tb.us)


def test_full_state_training_is_bit_identical():
    base = Config(seed=0, rollout=RolloutCfg(ep_steps=120, n_eps=2),
                  es=ESCfg(pop=16, gens=6, metric_every=3))
    sensed = Config(seed=0, sensors=("full_state",),
                    rollout=RolloutCfg(ep_steps=120, n_eps=2),
                    es=ESCfg(pop=16, gens=6, metric_every=3))
    a, b = train(base), train(sensed)
    assert torch.equal(a.theta, b.theta), "training diverged with the identity sensor"
    assert [h["fitness_elite"] for h in a.history] == [h["fitness_elite"] for h in b.history]

    system, tr, task = build(sensed)
    ev_a = evaluate(system, tr, task, a.theta, sensed.rollout, n_tasks=64)
    ev_b = evaluate(system, tr, task, b.theta, sensed.rollout, n_tasks=64)
    assert ev_a == ev_b


# --------------------------------------------------------------------------- #
# common random numbers
# --------------------------------------------------------------------------- #
def test_sensor_noise_is_shared_across_the_population():
    """Same noise realization for every genome in a generation, exactly as with
    goals and reset noise.  Otherwise sensor stochasticity becomes fitness
    ranking variance, and ES is already variance-limited."""
    system = make_system("quadrotor")
    sen = make_sensor("noisy_position", system, sigma=0.05)
    P, E = 5, 4
    sen.crn_group = E
    n = sen.crn_noise((P * E, 3), make_gen(0), DT, "cpu")
    base = n[:E]
    for i in range(1, P):
        assert torch.equal(n[i * E:(i + 1) * E], base), \
            "noise differs across population members"
    assert not torch.equal(base[0], base[1]), "noise must still vary across episodes"


def test_noise_free_sensor_ignores_the_generator():
    system = make_system("quadrotor")
    sen = make_sensor("noisy_position", system, sigma=0.0)
    s = system.reset(4, make_gen(0))
    assert torch.equal(sen.observe(s, make_gen(1)), sen.observe(s, make_gen(2)))


def test_noisy_sensor_is_reproducible_under_a_fixed_seed():
    system, tr, task, cfg, _ = _rig()
    sen = lambda: make_sensor("noisy_position", system, sigma=0.03, latency_steps=2)
    r1 = Rollout(system, tr, task, cfg, [sen()])
    r2 = Rollout(system, tr, task, cfg, [sen()])
    goals = task.sample(cfg.n_eps, make_gen(4))
    TH = tr.init()[None].expand(3, -1)
    assert torch.equal(r1.run(TH, goals, 9).fitness, r2.run(TH, goals, 9).fitness)


def test_dropout_reports_invalid_channels():
    system = make_system("quadrotor")
    sen = make_sensor("noisy_position", system, dropout=0.5)
    s = system.reset(2000, make_gen(0))
    assert bool(sen.valid(s).all()), "without a generator, dropout is not drawn"
    v = sen.valid(s, make_gen(4))
    frac = float((~v).to(DT).mean())
    assert 0.4 < frac < 0.6, f"dropout rate {frac:.3f} is not ~0.5"


def test_per_sensor_latency_is_independent():
    """Flow and IMU run at ~2 ms, ToF at 5-20 ms, vision at 30-80 ms; one global
    lag would erase exactly the timescale separation that matters."""
    system, tr, task, cfg, _ = _rig()
    fast = make_sensor("full_state", system, latency_steps=0)
    slow = make_sensor("noisy_position", system, sigma=0.0, latency_steps=5)
    r = Rollout(system, tr, task, cfg, [fast, slow])
    assert [b.latency for b in r.buffers] == [0, 5]
    assert {s.name for s in r.sensors} == {"full_state", "noisy_position"}


def test_sensors_receive_the_crn_group_from_the_rollout():
    system, tr, task, cfg, _ = _rig(n_eps=3)
    sen = make_sensor("noisy_position", system, sigma=0.01)
    Rollout(system, tr, task, cfg, [sen])
    assert sen.crn_group == cfg.n_eps == 3
