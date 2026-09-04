"""Importing CAD drawings as simulation environments."""
import pytest
import torch

from lagrangian_es.environments import Environment, load_dxf
from lagrangian_es.environments.cad import (
    StaticPillars, StaticWalls, dxf_to_environment, read_dxf,
)
from lagrangian_es.sensors import make_sensor
from lagrangian_es.systems import make_system
from lagrangian_es.tasks import make_task
from lagrangian_es.util import make_gen

DT = torch.float64
ezdxf = pytest.importorskip("ezdxf")


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    """A small floor plan in millimetres: outer room, partitions, columns, an arc."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (8000, 0), (8000, 6000), (0, 6000)], close=True)
    msp.add_line((3000, 0), (3000, 3500))
    msp.add_line((5000, 6000), (5000, 2500))
    for x, y in [(1500, 4500), (6500, 1500), (6500, 4500)]:
        msp.add_circle((x, y), 350)
    msp.add_arc((4000, 3000), 800, 0, 180)
    p = tmp_path_factory.mktemp("cad") / "plan.dxf"
    doc.saveas(p)
    return p


def test_reads_every_supported_entity(plan):
    segs, circles = read_dxf(plan)
    assert len(circles) == 3
    # 4 closed-polyline spans + 2 lines + 16 arc spans
    assert len(segs) >= 20
    assert all(len(s) == 2 and len(s[0]) == 2 for s in segs)


def test_blocks_are_expanded_not_dropped(tmp_path):
    """INSERT entities are everywhere in real drawings; importing them as nothing
    silently yields an empty building."""
    doc = ezdxf.new()
    blk = doc.blocks.new(name="COL")
    blk.add_circle((0, 0), 200)
    msp = doc.modelspace()
    msp.add_blockref("COL", (1000, 1000))
    msp.add_blockref("COL", (2000, 1500))
    p = tmp_path / "blocks.dxf"
    doc.saveas(p)
    _, circles = read_dxf(p)
    assert len(circles) == 2, "block references were dropped"


def test_units_are_normalized(plan):
    """A plan drawn in millimetres has to become metres without the caller
    knowing its units."""
    env = dxf_to_environment(plan, fit=8.0)
    assert abs(env.imported["extent"] - 8.0) < 1e-9
    assert abs(env.imported["scale"] - 0.001) < 1e-9
    f = env.sample(2, make_gen(0), DT, "cpu")
    for v in f.values():
        assert float(v.abs().max()) < 20.0, "geometry left the arena after scaling"


def test_explicit_scale_overrides_fit(plan):
    env = dxf_to_environment(plan, fit=None, scale=0.002)
    assert abs(env.imported["scale"] - 0.002) < 1e-12
    assert abs(env.imported["extent"] - 16.0) < 1e-9


def test_imported_geometry_is_identical_every_episode(plan):
    """Static, unlike the sampled fields: the building does not get redrawn."""
    env = dxf_to_environment(plan, fit=8.0)
    f = env.sample(8, make_gen(0), DT, "cpu")
    for v in f.values():
        assert torch.equal(v[0], v[-1]), "imported geometry varied across episodes"
    g = env.sample(8, make_gen(999), DT, "cpu")
    for k in f:
        assert torch.equal(f[k], g[k]), "imported geometry varied with the seed"


def test_a_building_never_moves_for_a_waypoint(plan):
    """`clear_points` is a no-op on imported geometry: a building that dodges the
    drone is not that building any more."""
    env = dxf_to_environment(plan, fit=8.0)
    f = env.sample(4, make_gen(0), DT, "cpu")
    pts = torch.zeros(4, 3, 3, dtype=DT)          # right on the origin
    out = env.clear_points(f, pts, margin=1.0)
    for k in f:
        assert torch.equal(f[k], out[k]), f"{k} moved to clear a waypoint"
    assert StaticWalls.clear_points is not None
    assert StaticPillars.clear_points is not None


def test_free_space_sampling_respects_the_walls(plan):
    env = dxf_to_environment(plan, fit=8.0)
    pts = env.free_points(2000, make_gen(0), extent=3.5, margin=0.4,
                          z_range=(1.0, 2.0), dtype=DT)
    assert pts.shape == (2000, 3)
    f = env.sample(1, make_gen(0), DT, "cpu")
    assert float(env.sdf(pts, f).min()) >= 0.4


def test_free_space_failure_is_explained(plan):
    """An impossible clearance should say why, not return an empty tensor."""
    env = dxf_to_environment(plan, fit=8.0)
    with pytest.raises(ValueError, match="no free space"):
        env.free_points(64, make_gen(0), extent=3.0, margin=50.0)


def test_simulating_inside_an_imported_plan(plan):
    """The whole point: a drawing becomes a plant you can actually fly in."""
    env = dxf_to_environment(plan, fit=8.0)
    sysm = make_system("quadrotor_nav", env=env, free_start=True)
    task = make_task("free_space", sysm, n_legs=2, margin=0.4, extent=3.5)

    s = sysm.reset(128, make_gen(0))
    assert float(sysm.clearance(s).min()) > 0.0, "an episode began inside a wall"

    goals = task.sample(128, make_gen(1))
    for leg in range(goals.shape[1]):
        assert float(env.sdf(goals[:, leg, :], s).min()) > 0.0

    sen = make_sensor("range", sysm, n_beams=12)
    obs = sen.observe(s, make_gen(2))
    assert obs.shape == (128, 12)
    assert float(obs.min()) < sen.max_range, "no beam found a wall"
    assert torch.isfinite(sen.jacobian(s)).all()


def test_empty_drawing_is_rejected(tmp_path):
    doc = ezdxf.new()
    doc.saveas(tmp_path / "empty.dxf")
    with pytest.raises(ValueError, match="no usable geometry"):
        dxf_to_environment(tmp_path / "empty.dxf")


def test_load_dxf_is_reachable_from_the_package(plan):
    env = load_dxf(plan, fit=5.0)
    assert isinstance(env, Environment) and len(env) >= 1
