"""The imported city: real geometry, and the seams that make it flyable.

The Singapore DXF is an OSM export with roads, water, railways and coastline and
NO buildings, so `scripts/dxf_city.py` recovers the built form as the complement
of the street network.  What is asserted here is not that the rasteriser is
correct in the abstract -- it is that the result is a scene a controller can
actually be asked to fly, which is a different and more falsifiable claim.
"""
import json

import pytest
import torch

from lagrangian_es.environments import MAPS, make_environment
from lagrangian_es.systems import make_system
from lagrangian_es.tasks import make_task
from lagrangian_es.util import make_gen

DT = torch.float64
CITY = MAPS / "singapore_cbd.json"


@pytest.fixture(scope="module")
def env():
    return make_environment("singapore_cbd")


@pytest.fixture(scope="module")
def meta():
    return json.loads(CITY.read_text())["meta"]


def test_the_map_ships_with_the_package(env, meta):
    assert CITY.exists()
    assert meta["n_blocks"] > 20 and meta["n_waypoints"] > 10
    assert len(env.waypoints) == meta["n_waypoints"]
    assert env.span == pytest.approx(meta["half_m"] / meta["scale"])


def test_blocks_never_move_for_a_waypoint(env):
    """`clear_points` is a no-op on imported geometry.

    A random pillar field is nudged aside to keep a waypoint reachable; a city
    block may not be, or it stops being a simulation of that city.
    """
    f = env.sample(2, make_gen(0), DT, "cpu")
    pts = torch.zeros(2, 3, 3, dtype=DT)
    out = env.clear_points(f, pts, margin=2.0)
    for k in f:
        assert torch.equal(out[k], f[k]), k


def test_geometry_is_identical_across_episodes_and_seeds(env):
    a = env.sample(3, make_gen(1), DT, "cpu")
    b = env.sample(3, make_gen(999), DT, "cpu")
    for k in a:
        assert torch.equal(a[k], b[k]), k
        assert torch.equal(a[k][0], a[k][2]), k


def test_streets_are_wider_than_the_standoff_the_controller_flies_with(meta):
    """The scale is the whole ballgame.

    Divide a real city by too much and every street closes to under the 0.60 m
    proximity band the controller keeps, so the task is not hard, it is
    impossible; divide by too little and the vehicle cannot cross a junction
    inside an episode.  This pins the end that silently produces a scene nothing
    can fly.
    """
    p50, p90 = meta["corridor_halfwidth"]
    assert p50 > 0.60, f"median corridor half-width {p50} is inside the standoff"
    assert p90 > p50
    assert meta["flyable_fraction"] > 0.25


def test_every_waypoint_has_room_to_hover(env):
    """A goal inside a building is a goal the vehicle is scored for missing."""
    f = env.sample(1, make_gen(0), DT, "cpu")
    w = torch.as_tensor(env.waypoints, dtype=DT)
    assert env.sdf(w, f).min() >= 0.90


def test_the_tour_only_proposes_legs_that_can_be_flown():
    """Waypoints span the whole window; an episode covers metres of it.

    Drawn independently, two waypoints are usually further apart than the
    episode is long -- which does not make the task harder, it makes it
    unfinishable.
    """
    sysm = make_system("quadrotor_nav", environment="singapore_cbd")
    task = make_task("city_tour", sysm, n_legs=3, max_leg=10.0)
    g = task.sample(4000, make_gen(7))
    d = (g[:, 1:, :2] - g[:, :-1, :2]).norm(dim=-1)
    assert d.max() <= 10.0 + 1e-9
    assert d.min() > 0.0, "a leg of zero length: a waypoint is its own neighbour"


def test_no_waypoint_is_its_own_neighbour():
    """`cdist`'s diagonal is not exactly zero.

    Computed through the same expansion as every other entry, it comes back as a
    few times 1e-9 on some rows and as 0.0 on others, so a `d > 0` self-test
    admits the point itself as a neighbour -- and a tour that lands on such a
    node never leaves it.  Measured before the index mask: 349 of 8000 legs had
    zero length.
    """
    sysm = make_system("quadrotor_nav", environment="singapore_cbd")
    task = make_task("city_tour", sysm, max_leg=10.0)
    for i in range(task.pool.shape[0]):
        assert not bool((task.nbr[i, :task.cnt[i]] == i).any()), i


def test_the_tour_reaches_the_whole_map():
    """A neighbour graph that has fallen into a component is still a valid
    graph, and would quietly train on one junction."""
    sysm = make_system("quadrotor_nav", environment="singapore_cbd")
    task = make_task("city_tour", sysm, n_legs=4, max_leg=10.0)
    g = task.sample(3000, make_gen(2)).reshape(-1, 3)
    seen = {tuple(round(v, 6) for v in p) for p in g.tolist()}
    assert len(seen) == task.pool.shape[0]


def test_free_start_uses_the_map_and_not_a_three_metre_box():
    """The start pool's default extent is the arena the random fields use.

    On a +/-31 m city that samples a pool entirely inside one block -- or throws,
    because there is no free space within 3 m of the origin.
    """
    sysm = make_system("quadrotor_nav", environment="singapore_cbd",
                       free_start=True)
    assert sysm.start_pool is not None
    assert sysm.start_pool[:, :2].abs().max() > 3.0
    f = sysm.env.sample(1, make_gen(0), DT, "cpu")
    assert sysm.env.sdf(sysm.start_pool, f).min() > 0.0


def test_culling_is_exact_on_the_city():
    """`local_field` must not change what a ray hits, only what it costs.

    The first version of this test used four points and eight rays and passed
    against a cull that was genuinely unsound: the city's greedy block
    decomposition produces one rectangle 15 m across, `local_field` padded its
    cull radius by the LARGEST half-extent, and at a 28.5 m radius 51 of 60
    blocks qualified while `cull_k` was 40.  So it dropped blocks that rays
    could reach, and did it rarely enough that a handful of samples missed it.

    Hence: every waypoint, a full fan, and a bit-for-bit comparison.
    """
    sysm = make_system("quadrotor_nav", environment="singapore_cbd")
    env = sysm.env
    grp = env.groups[0]
    assert grp.cull_k, "the city ships with culling on; this test is the reason"
    n = len(env.waypoints)
    f = env.sample(n, make_gen(0), DT, "cpu")
    p = torch.as_tensor(env.waypoints, dtype=DT)
    B = 64
    th = torch.linspace(0.0, 6.283185307179586, B, dtype=DT)
    dirs = torch.zeros(n, B, 3, dtype=DT)
    dirs[..., 0], dirs[..., 1] = torch.cos(th), torch.sin(th)
    culled, _ = env.raycast(p, dirs, f, 6.0)
    saved, grp.cull_k = grp.cull_k, 0
    try:
        exact, _ = env.raycast(p, dirs, f, 6.0)
    finally:
        grp.cull_k = saved
    err = (culled - exact).abs()
    assert err.max() < 1e-9, (float(err.max()), int((err > 1e-9).sum()),
                              f"{n * B} rays")


def test_cull_k_covers_every_block_that_can_reach_the_sensor():
    """The soundness condition, stated where it can be checked.

    Culling is exact iff `cull_k` is at least the number of primitives whose
    SURFACE lies within reach -- which is what `chunk_occupancy` counts.
    """
    sysm = make_system("quadrotor_nav", environment="singapore_cbd")
    env, grp = sysm.env, sysm.env.groups[0]
    f = env.sample(1, make_gen(0), DT, "cpu")
    f1 = {k: v[0] for k, v in f.items()}
    w = torch.as_tensor(env.waypoints, dtype=DT)
    g = make_gen(3)
    jit = (torch.rand(4000, 3, generator=g, dtype=DT) - 0.5) * torch.tensor(
        [4.0, 4.0, 0.8], dtype=DT)
    pts = w[torch.randint(len(w), (4000,), generator=g)] + jit
    need = int(grp.chunk_occupancy(pts, f1, 6.0).max())
    assert grp.cull_k >= need, (grp.cull_k, need)


def test_culling_ranks_by_surface_distance_not_centre_distance():
    """A big far primitive can be nearer than a small close one.

    Centre-ranking gets that backwards, and no amount of padding fixes it -- the
    pad has to cover the largest primitive, so on a mixed field it swells the
    cull radius until nearly everything qualifies.
    """
    from lagrangian_es.environments import StaticBoxes
    grp = StaticBoxes(centres=[[0.0, 30.0], [0.0, 8.0]],
                      halves=[[15.0, 15.0, 5.0], [0.2, 0.2, 5.0]],
                      cull_k=1)
    f = grp.sample(1, make_gen(0), DT, "cpu")
    origin = torch.zeros(1, 3, dtype=DT)
    origin[0, 2] = 1.0
    out = grp.local_field(origin, f, 6.0)
    # the wall's face is at y = 15, the speck's at y = 7.8: the speck is nearer
    kept = out[grp._k("h")][0, 0]
    assert float(kept[0]) < 1.0, "kept the 15 m block over the 0.2 m one"
