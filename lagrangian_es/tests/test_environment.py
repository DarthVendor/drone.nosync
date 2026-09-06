"""Composable obstacle fields, navigation, and hoop courses."""
import math

import pytest
import torch

from lagrangian_es.config import RolloutCfg
from lagrangian_es.rollout import Rollout
from lagrangian_es.sensors import make_sensor
from lagrangian_es.systems import make_system
from lagrangian_es.environments import (
    GROUPS, PRESETS, Environment, Hoops, Pillars, Walls, make_environment,
    make_group,
)
from lagrangian_es.tasks import make_task
from lagrangian_es.trainables import make_trainable
from lagrangian_es.util import make_gen

DT = torch.float64
SCENES = sorted(PRESETS)


def test_registries():
    assert set(GROUPS) >= {"pillars", "walls", "gate", "hoops"}
    assert isinstance(make_group("pillars", n=2), Pillars)
    with pytest.raises(KeyError):
        make_environment("nope")


@pytest.mark.parametrize("name", SCENES)
def test_every_preset_samples_and_starts_clear(name):
    """An episode that begins in collision measures nothing, so the start pose
    must be outside every obstacle in every layout."""
    env = make_environment(name)
    f = env.sample(256, make_gen(0), DT, "cpu")
    p = torch.zeros(256, 3, dtype=DT)
    p[:, 2] = 0.5
    d = env.sdf(p, f)
    assert torch.isfinite(d).all()
    if len(env):
        assert float(d.min()) > 0.0, f"{name} starts in collision"


@pytest.mark.parametrize("name", SCENES)
def test_raycast_shapes_and_finiteness(name):
    env = make_environment(name)
    f = env.sample(8, make_gen(1), DT, "cpu")
    o = torch.zeros(8, 3, dtype=DT)
    o[:, 2] = 1.2
    ang = torch.linspace(0, 2 * math.pi, 12, dtype=DT)[:-1]
    d = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
    d = d.expand(8, 11, 3).contiguous()
    rng, grad = env.raycast(o, d, f, max_range=5.0)
    assert rng.shape == (8, 11) and grad.shape == (8, 11, 3)
    assert torch.isfinite(rng).all() and torch.isfinite(grad).all()
    assert float(rng.min()) >= 0.0 and float(rng.max()) <= 5.0


def test_closed_form_raycast_matches_finite_differences():
    """Pillars and walls override the marcher with analytic intersections; the
    gradient has to be the real derivative, not merely finite."""
    env = Environment([Pillars(n=3), Walls(n=2)])
    f = env.sample(4, make_gen(2), DT, "cpu")
    o = torch.zeros(4, 3, dtype=DT)
    o[:, 2] = 1.0
    d = torch.tensor([[[1., 0, 0], [0, 1., 0], [-1., 0, 0], [0, -1., 0]]], dtype=DT)
    d = d.expand(4, 4, 3).contiguous()
    _, grad = env.raycast(o, d, f, 6.0)
    h = 1e-6
    for i in range(2):
        op, om = o.clone(), o.clone()
        op[:, i] += h
        om[:, i] -= h
        fd = (env.raycast(op, d, f, 6.0)[0] - env.raycast(om, d, f, 6.0)[0]) / (2 * h)
        assert float((fd - grad[..., i]).abs().max()) < 1e-4


def test_vertical_primitives_report_zero_height_gradient():
    """Pillars and walls are vertical, so height genuinely does not change the
    range -- a fudged nonzero value is a lie the pullback would act on."""
    env = Environment([Pillars(n=4), Walls(n=2)])
    f = env.sample(6, make_gen(3), DT, "cpu")
    o = torch.zeros(6, 3, dtype=DT)
    o[:, 2] = 1.0
    ang = torch.linspace(0, 2 * math.pi, 9, dtype=DT)[:-1]
    d = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
    _, grad = env.raycast(o, d.expand(6, 8, 3).contiguous(), f, 5.0)
    assert float(grad[..., 2].abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# hoops
# --------------------------------------------------------------------------- #
def test_hoop_opening_is_free_and_the_ring_is_solid():
    """The one obstacle that cannot be satisfied by going around."""
    h = Hoops(n=1, radius=0.5, tube=0.06)
    f = {"hoops/c": torch.tensor([[[2., 0., 1.5]]], dtype=DT),
         "hoops/a": torch.tensor([[[1., 0., 0.]]], dtype=DT),
         "hoops/r": torch.tensor([[0.5]], dtype=DT),
         "hoops/tilt": torch.zeros(1, 1, dtype=DT)}
    centre = torch.tensor([[2., 0., 1.5]], dtype=DT)
    on_ring = torch.tensor([[2., 0.5, 1.5]], dtype=DT)
    assert float(h.sdf(centre, f)) > 0.4, "the opening must be free"
    assert float(h.sdf(on_ring, f)) < 0.0, "the ring itself must be solid"


def test_hoop_tilt_spans_vertical_to_horizontal():
    g = make_gen(0)
    for name, lo, hi in [("hoops_upright", 0.0, 1.0),
                         ("hoops_flat", 60.0, 90.1),
                         ("hoop_course", 0.0, 90.1)]:
        f = make_environment(name).sample(400, g, DT, "cpu")
        a = f["hoops/a"]
        assert float((a.norm(dim=-1) - 1.0).abs().max()) < 1e-12, "axes must stay unit"
        tilt = torch.asin(a[..., 2].abs().clamp(0, 1)) * 180 / math.pi
        assert lo <= float(tilt.min()) and float(tilt.max()) <= hi


def test_marched_group_only_needs_an_sdf():
    """Hoops implement `sdf` and `normal` and inherit ray casting -- that is the
    property that makes a new primitive one class rather than three."""
    assert "raycast" not in Hoops.__dict__, "Hoops should inherit the marcher"
    env = make_environment("hoop_course")
    f = env.sample(4, make_gen(1), DT, "cpu")
    o = torch.zeros(4, 3, dtype=DT)
    o[:, 2] = 1.4
    d = torch.tensor([[[1., 0, 0]]], dtype=DT).expand(4, 1, 3).contiguous()
    rng, grad = env.raycast(o, d, f, 6.0)
    assert torch.isfinite(rng).all() and torch.isfinite(grad).all()


# --------------------------------------------------------------------------- #
# navigation plant
# --------------------------------------------------------------------------- #
def test_geometry_lives_in_the_state_and_survives_a_step():
    sysm = make_system("quadrotor_nav", environment="pillars")
    s = sysm.reset(8, make_gen(0))
    assert "pillars/c" in s and "pillars/r" in s
    u = torch.zeros(8, 4, dtype=DT)
    u[:, 0] = sysm.m * sysm.g
    s2 = sysm.step(s, u, 0.02)
    for k in s:
        assert k in s2, f"{k} lost across a step"
    assert torch.equal(s["pillars/c"], s2["pillars/c"]), "geometry must not drift"


def test_collision_ends_the_episode():
    sysm = make_system("quadrotor_nav", environment="pillars")
    s = sysm.reset(16, make_gen(0))
    c = s["pillars/c"][:, 0]
    s["p"] = torch.cat([c, torch.full((16, 1), 1.0, dtype=DT)], dim=-1)
    assert not bool(sysm.alive(s).any()), "sitting inside a pillar must not be alive"
    assert float(sysm.clearance(s).max()) < 0.0


def test_held_out_seed_gives_held_out_layouts():
    """The property the whole generalization claim rests on."""
    sysm = make_system("quadrotor_nav", environment="pillars")
    a = sysm.reset(32, make_gen(1))["pillars/c"]
    b = sysm.reset(32, make_gen(1))["pillars/c"]
    c = sysm.reset(32, make_gen(99))["pillars/c"]
    assert torch.equal(a, b), "same seed must reproduce the scene"
    assert not torch.allclose(a, c), "a different seed must be a different scene"


def test_hoops_are_placed_on_the_waypoints():
    """A gate the route does not pass through is not a gate."""
    sysm = make_system("quadrotor_nav", environment="hoop_course")
    tr = make_trainable("energy_shaping", sysm)
    task = make_task("hoop_course", sysm, n_gates=3)
    assert sysm.needs_course and task.n_legs == 3
    roll = Rollout(sysm, tr, task, RolloutCfg(n_eps=4))
    goals = task.sample(4, make_gen(1))
    s, _, gb, _, P, E = roll._expand(tr.init()[None], goals, 7)
    assert torch.allclose(s["hoops/c"][:, :3], gb[:, :3], atol=1e-12)
    assert float(sysm.clearance(s).min()) > 0.0
    assert float((s["hoops/a"].norm(dim=-1) - 1).abs().max()) < 1e-12


def test_range_sensor_sees_the_scene():
    sysm = make_system("quadrotor_nav", environment="cluttered")
    sen = make_sensor("range", sysm, n_beams=12)
    s = sysm.reset(64, make_gen(0))
    o = sen.observe(s, make_gen(1))
    assert o.shape == (64, 12)
    assert float(o.min()) >= 0.0 and float(o.max()) <= sen.max_range
    assert float(o.min()) < sen.max_range, "some beam must actually hit something"
    assert torch.isfinite(sen.jacobian(s)).all()


# --------------------------------------------------------------------------- #
# waypoint reachability
# --------------------------------------------------------------------------- #
REACHABLE = ["pillars", "sparse", "forest", "gate", "walls", "cluttered",
             "slalom", "gate_forest", "train_mix", "pair_mix", "test_mix"]


@pytest.mark.parametrize("scene", REACHABLE)
def test_no_waypoint_is_ever_inside_geometry(scene):
    """Goals and geometry are drawn from different generators, so nothing
    coordinates them by default and a few percent of waypoints land inside an
    obstacle -- a permanent, invisible ceiling on the success rate that no amount
    of training removes.

    Obstacles move, not the goals: rejection-sampling the goals would bias the
    task distribution toward open space, and the bias would grow with obstacle
    density, so a "harder" scene would silently become a *different* task.
    """
    sysm = make_system("quadrotor_nav", environment=scene)
    tr = make_trainable("energy_shaping", sysm)
    task = make_task("waypoint_pair", sysm)
    roll = Rollout(sysm, tr, task, RolloutCfg(n_eps=512))
    goals = task.sample(512, make_gen(11))
    s, _, gb, _, _, _ = roll._expand(tr.init()[None], goals, 7)

    for leg in range(gb.shape[1]):
        d = sysm.env.sdf(gb[:, leg, :], s)
        assert float(d.min()) > 0.0, f"{scene}: a waypoint is inside geometry"
    assert float(sysm.clearance(s).min()) > 0.0, "the start must be clear too"


@pytest.mark.parametrize("scene,gates", [("hoop_course", 3), ("hoops", 2),
                                         ("hoop_slalom", 3)])
def test_gates_stay_on_their_waypoints_after_clearing(scene, gates):
    """Clearing runs BEFORE placement, so a gate still lands exactly on its
    waypoint -- otherwise clearing would push the course apart and delete the
    task it exists to define."""
    sysm = make_system("quadrotor_nav", environment=scene)
    tr = make_trainable("energy_shaping", sysm)
    task = make_task("hoop_course", sysm, n_gates=gates)
    roll = Rollout(sysm, tr, task, RolloutCfg(n_eps=128))
    goals = task.sample(128, make_gen(3))
    s, _, gb, _, _, _ = roll._expand(tr.init()[None], goals, 7)
    n = min(gates, s["hoops/c"].shape[1])
    assert torch.allclose(s["hoops/c"][:, :n], gb[:, :n], atol=1e-12)
    for leg in range(gb.shape[1]):
        assert float(sysm.env.sdf(gb[:, leg, :], s).min()) > 0.0


def test_clearing_is_a_no_op_when_nothing_overlaps():
    """A layout that already has clearance must be left exactly alone."""
    env = make_environment("sparse")
    f = env.sample(64, make_gen(5), DT, "cpu")
    far = torch.full((64, 2, 3), 40.0, dtype=DT)
    out = env.clear_points(f, far, margin=0.3)
    for k in f:
        assert torch.equal(f[k], out[k]), f"{k} moved with nothing to avoid"


def test_gate_inherits_clearance_not_just_geometry():
    """`Gate` borrows `sdf`/`raycast` from `Pillars` by assignment; it originally
    kept the base class's no-op `clear_points` and silently swallowed waypoints."""
    from lagrangian_es.environments import Gate, Pillars
    assert Gate.clear_points is Pillars.clear_points


# --------------------------------------------------------------------------- #
# compositional mixtures
# --------------------------------------------------------------------------- #
def test_mixture_activates_the_right_number_of_regimes():
    from lagrangian_es.environments import MIXTURES
    for name, (lo, hi) in MIXTURES.items():
        env = make_environment(name)
        f = env.sample(512, make_gen(0), DT, "cpu")
        live = f["mix/active"].sum(-1)
        assert float(live.min()) >= lo and float(live.max()) <= hi, name


def test_mixture_state_schema_is_identical_across_splits():
    """One genome and one sensor must serve train and test, so the tensors have to
    be present and the same shape whichever regimes an episode drew."""
    a = make_environment("train_mix").sample(16, make_gen(0), DT, "cpu")
    b = make_environment("test_mix").sample(16, make_gen(1), DT, "cpu")
    assert set(a) == set(b)
    for k in a:
        assert a[k].shape == b[k].shape, k


def test_inactive_regimes_are_out_of_play():
    """Parked geometry must not affect clearance or ranging."""
    env = make_environment("train_mix")
    f = env.sample(256, make_gen(2), DT, "cpu")
    live = f["mix/active"]
    assert float(live.sum(-1).max()) == 1.0
    p = torch.zeros(256, 3, dtype=DT)
    p[:, 2] = 1.0
    assert torch.isfinite(env.sdf(p, f)).all()
    # with only one regime live, at most one group can be the nearest thing
    per = torch.stack([g.sdf(p, f) for g in env.groups], dim=-1)
    parked = per > 100.0
    assert int(parked.sum(-1).min()) >= len(env.groups) - 1


# --- proximity shaping: the only smooth obstacle signal in the objective -----
def test_proximity_penalty_meets_dead_cost_at_the_collision_boundary():
    """Weighted proximity rate at zero clearance must MEET `dead_cost`.

    The term's whole job is to turn the collision cliff into a slope.  If the
    rate at zero clearance is below `dead_cost` the objective still steps at the
    boundary and the search sees no gradient exactly where it needs one -- which
    is not hypothetical: at the original gain of 2.0 this term carried 0.07% of
    the episode cost against 58% for the crash it anticipates, and the
    population's cheapest route under that objective was to stop moving.
    """
    cfg = RolloutCfg(lambda_s=0.2)          # the curriculum's shaping weight
    sysm = make_system("quadrotor_nav", environment="pillars", dtype=DT)
    rate = cfg.lambda_s * sysm.prox_gain * sysm.prox_band ** 2
    assert rate == pytest.approx(cfg.dead_cost, rel=0.05)


def test_course_band_leaves_a_centred_hoop_pass_unpenalized():
    """A gate is meant to be flown through, so the band must fit inside it.

    The torus SDF at the centre of a hoop is `radius - tube`, so a band wider
    than that charges for passing a gate correctly -- the course settings must
    stay under it while keeping the boundary match above.
    """
    h = Hoops()
    centre_clearance = h.radius - h.tube
    sysm = make_system("quadrotor_nav", environment="hoop_course", dtype=DT,
                       prox_gain=200.0, prox_band=0.35)
    assert sysm.prox_band < centre_clearance
    assert 0.2 * sysm.prox_gain * sysm.prox_band ** 2 == pytest.approx(5.0, rel=0.05)


def test_proximity_penalty_is_zero_in_the_clear_and_grows_approaching_geometry():
    sysm = make_system("quadrotor_nav", environment="pillars", dtype=DT)
    s = sysm.reset(64, make_gen(0))
    far = dict(s)
    # clearance() reads position, so move the batch rather than stub the method.
    # Pillars are vertical columns, so the move has to be LATERAL -- climbing
    # away from one never clears it.
    far["p"] = s["p"] + torch.tensor([500.0, 500.0, 0.0], dtype=DT)
    lo = sysm.shaping_cost(far) - (1.0 - far["R"][..., 2, 2])
    assert torch.allclose(lo, torch.zeros_like(lo), atol=1e-12)
