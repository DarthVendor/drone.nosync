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
    # "map" is the carried survey (sensors/map_view.py).  It is a real sensor --
    # declared in the config, delivered through `obs` -- but it feeds the
    # task-level composer and never the potential, so its Jacobian is zero by
    # design rather than by omission.
    assert sen.kind in ("position_like", "velocity_like", "range", "attitude_like", "map")
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
    ray-traced scenes, and it is the default for the expensive sensors.

    `range` used to be the exception, pinned to every step because striding it
    doubled the crash rate (0.027 -> 0.058 on the pillar field).  That finding
    still holds and has since been measured more precisely: nav99, trained at
    stride 1, goes from 0.9923/0.0015 to 0.9840/0.0111 when strided to 8 under
    it.  But that measures TRANSFER to an input the genome never saw.  The
    march is 97% of rollout time, so the stride is the single biggest compute
    dial in the project, and a policy that trains on 0.16 s-old readings can
    stand further off and damp harder -- options a policy handed staleness only
    at test time does not have.

    So the default is now 8 and the cost is paid where it belongs: the >99%
    prototype pins `PROTOTYPE_STRIDE = 1` in tests/test_prototype.py, because a
    published number is a claim about a configuration and must not follow a
    compute dial around.

    `FullState` is pinned to every step for a different reason: it is the
    identity baseline the sensor-free path has to reproduce bit-for-bit, and
    striding it would make it something else."""
    system = make_system("quadrotor")
    assert make_sensor("range", make_system("quadrotor_nav")).update_every == 8
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
    # `n_beams` has to match the sensor now.  The hand-designed terms sliced
    # whatever width they were given; the learned potential's first layer is
    # sized `task_dim + n_obs`, so a 12-beam sensor under a 24-beam trainable is
    # an einsum shape error rather than a silent mis-read.  That coupling is a
    # feature -- it fails loudly -- but it is new, so it is stated here.
    tr = make_trainable("nav_agent", system, n_beams=12)
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


# --- one march, not two ------------------------------------------------------

def test_observe_with_jacobian_matches_the_two_calls_it_replaces():
    """`system.raycast` returns the range AND its gradient from one march.

    `observe` was taking the range and discarding the gradient; `jacobian` then
    repeated the identical march and discarded the range.  Profiled on the
    imported city, that march was 97% of rollout time and ran twice per sensor
    update.  Combining them is only legitimate if it is bit-exact, including the
    common-random-numbers draw -- a sensor whose noise stream shifted would
    change every fitness comparison in the search.
    """
    import torch

    from lagrangian_es.sensors import make_sensor
    from lagrangian_es.systems import make_system
    from lagrangian_es.util import make_gen

    for env, name in (("pillars", "range"), ("singapore_cbd", "range"),
                      ("pillars", "depth_camera")):
        sysm = make_system("quadrotor_nav", environment=env)
        sen = make_sensor(name, sysm)
        s = sysm.reset(32, make_gen(0))
        for sigma in (0.0, 0.05):
            sen.sigma = sigma
            r_old = sen.observe(s, make_gen(7))
            j_old = sen.jacobian(s)
            r_new, j_new = sen.observe_with_jacobian(s, make_gen(7))
            assert torch.equal(r_old, r_new), (env, name, sigma)
            assert torch.equal(j_old, j_new), (env, name, sigma)


def test_a_sensor_that_shares_nothing_still_works():
    """The base implementation just calls both, so overriding is optional."""
    import torch

    from lagrangian_es.sensors import make_sensor
    from lagrangian_es.systems import make_system
    from lagrangian_es.util import make_gen

    sysm = make_system("quadrotor", environment=None) \
        if False else make_system("quadrotor")
    sen = make_sensor("landmark_camera", sysm, n_landmarks=4)
    s = sysm.reset(8, make_gen(0))
    r, j = sen.observe_with_jacobian(s, make_gen(1))
    assert torch.equal(r, sen.observe(s, make_gen(1)))
    assert j.shape[0] == 8


def test_token_bearings_match_the_rays_the_sensor_actually_casts():
    """The tokenizer used to rebuild the fan from n_beams/spread/elevations
    instead of deriving it from the sensor's own construction, and the two
    disagreed for any fan outside the horizontal plane.

    `RangeSensor._dirs` forms [cos(off)cos(el), sin(off)cos(el), sin(el)], so at
    el = -pi/2 the cos(el) factor is ZERO and the azimuth collapses:
    `range_down`'s four "slightly splayed" beams are four identical
    straight-down rays and its spread does nothing.  The tokenizer labelled them
    -17.2, -8.6, 0.0 and +8.6 degrees, so a quarter of the beam tokens carried
    directions the sensor never looked in -- range attached to the wrong
    bearing, which no amount of training can undo.

    Compared as VECTORS, because azimuth is undefined for a straight-down ray
    and comparing two arbitrary values of an undefined quantity proves nothing.
    """
    import math
    import torch
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.composer.tokens import Tokenizer

    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range", "range_down", "depth_camera"),
                 gating="arrival", seed=0, task_kw=(("n_legs", 2), ("max_leg", 20.0)),
                 system_kw=(("free_start", True),), trainable_kw=(("learned", True),),
                 rollout=RolloutCfg(n_eps=4, ep_steps=100))
    system, trainable, task = build(cfg)
    sensors = build_sensors(cfg, system)
    tok = Tokenizer(scale=30.0, reach=10.0, sensors=sensors)

    checked = 0
    for sen in sensors:
        if not (hasattr(sen, "n_beams") and hasattr(sen, "spread")):
            continue
        lay = [l for l in tok.layout if l[0] == sen.name]
        assert lay, f"{sen.name} is not in the layout: its readings never become tokens"
        bear = lay[0][2]
        n = int(sen.n_beams)
        off = (torch.arange(n, dtype=torch.float64) / n - 0.5) * float(sen.spread)
        i = 0
        for e in getattr(sen, "elevations", (0.0,)):
            ce, se = math.cos(float(e)), math.sin(float(e))
            truth = torch.stack([torch.cos(off) * ce, torch.sin(off) * ce,
                                 torch.full_like(off, se)], -1)
            for b in range(n):
                az, el = float(bear[i, 0]), float(bear[i, 1])
                got = torch.tensor([math.cos(az) * math.cos(el),
                                    math.sin(az) * math.cos(el), math.sin(el)],
                                   dtype=torch.float64)
                ang = math.degrees(math.acos(min(1.0, float((got * truth[b]).sum()))))
                assert ang < 1.0, (
                    f"{sen.name} beam {i}: the token says {[round(float(v), 3) for v in got]} "
                    f"but the sensor casts {[round(float(v), 3) for v in truth[b]]} ({ang:.1f} deg apart)")
                checked += 1
                i += 1
    assert checked >= 28, f"only {checked} beams checked"


def test_the_built_map_uses_a_fan_that_can_actually_map():
    """`RangeDown` subclasses `RangeSensor`, so it reports kind "range" too --
    and the built map used to take the FIRST such sensor, making the choice a
    function of config order rather than geometry.

    It matters because a straight-down beam has no horizontal component:
    `end = p + dirs[..., :2] * rng` is the vehicle's own position, so a
    downward fan would stamp the drone's own cell as occupied on every scan and
    the map it builds of the world would be a trail of itself.
    """
    import math
    from lagrangian_es.config import Config, RolloutCfg
    from lagrangian_es.es import build, build_sensors
    from lagrangian_es.rollout import Rollout

    for order in (("range", "range_down"), ("range_down", "range")):
        cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                     environment="singapore_cbd", sensors=order, gating="arrival", seed=0,
                     task_kw=(("n_legs", 2), ("max_leg", 20.0)),
                     system_kw=(("free_start", True),), trainable_kw=(("learned", True),),
                     rollout=RolloutCfg(n_eps=4, ep_steps=50))
        system, trainable, task = build(cfg)
        rig = Rollout(system, trainable, task, cfg.rollout, build_sensors(cfg, system))
        assert getattr(rig._beam_sen, "name", None) == "range", (
            f"with sensors declared {order} the built map chose "
            f"{getattr(rig._beam_sen, 'name', None)}")


def test_a_vertical_fan_marks_the_vehicles_own_cell():
    """The geometry behind the test above, stated directly: a beam straight
    down ends where the vehicle is, in the horizontal plane the map lives in."""
    import torch
    from lagrangian_es.mapping import BuiltMap

    m = BuiltMap()
    m.reset(1, torch.device("cpu"), torch.float32)      # the grid is allocated lazily
    p = torch.zeros(1, 3)
    down = torch.tensor([[[0.0, 0.0, -1.0]]])          # one straight-down beam
    rng = torch.tensor([[1.2]])                         # a return well inside range
    m.update(p, down, rng, max_range=2.0, live=torch.ones(1, dtype=torch.bool), t=1.0)
    assert m.hits is not None
    G = m.G
    centre = (G // 2) * G + (G // 2)
    assert float(m.hits[0, centre]) > 0.0, (
        "a straight-down beam did not mark the vehicle's own cell -- the fixture "
        "no longer demonstrates why the mapping fan must not be vertical")
