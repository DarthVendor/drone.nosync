"""The visualizer's render path, checked headlessly against the real schema.

Two bugs motivated this, and both killed the whole page rather than one overlay,
because an exception inside `requestAnimationFrame` stops the loop:

  * the sensing overlay was gated on `R.sensor` being truthy rather than on its
    TYPE, so a range sensor had a camera's fields (landmarks, mount, image) read
    off it;
  * a sprite kind was handled in `paint`'s draw loop but not its collection loop.

Neither is visible from Python, and neither shows up until a scene contains
something other than the vehicle.  The payload here is built from the real
`render_spec` / `render_poses` seam rather than hand-written, so it cannot drift
away from what the exporter actually produces.
"""
import json
import shutil
import subprocess

import pytest
import torch

from lagrangian_es.config import RolloutCfg
from lagrangian_es.rollout import Rollout
from lagrangian_es.sensors import make_sensor
from lagrangian_es.systems import make_system
from lagrangian_es.tasks import make_task
from lagrangian_es.trainables import make_trainable
from lagrangian_es.trainables.sensor_terms import FovBarrier, RangeBarrier
from lagrangian_es.trainables.terms import DissipationTerm, GoalBowl
from lagrangian_es.util import make_gen

DT = torch.float64
node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node not available")

#: one entry per overlay combination the page has to survive
CASES = [
    ("plain", dict(system="quadrotor", task="waypoint_pair")),
    ("camera", dict(system="quadrotor", task="waypoint_pair", sensor="camera")),
    ("obstacles", dict(system="quadrotor_nav", task="waypoint_pair",
                       env="pillars", sensor="range")),
    ("hoops", dict(system="quadrotor_nav", task="hoop_course", env="hoop_course",
                   sensor="range", task_kw=dict(n_gates=3))),
    ("payload", dict(system="quadrotor_payload", task="waypoint_pair")),
    # the planar drone is task_dim 2, so it needs a 2-D goal task
    ("planar", dict(system="planar_quad", task="joint_pair",
                    task_kw=dict(lo=0.5, hi=2.0))),
    ("legged", dict(system="quadruped", task="base_pose")),
    ("arm", dict(system="two_link_arm", task="joint_pair")),
]


def _r(x, nd=4):
    return [round(float(v), nd) for v in x.reshape(-1)]


def _build(spec, steps=6):
    kw = {"environment": spec["env"]} if spec.get("env") else {}
    sysm = make_system(spec["system"], **kw)
    sensor, sensors, terms = None, (), None
    if spec.get("sensor") == "camera":
        sensor = make_sensor("landmark_camera", sysm, n_landmarks=5, latency_steps=0)
        terms = ([GoalBowl(sysm.task_dim) for _ in range(3)]
                 + [DissipationTerm(sysm.task_dim),
                    FovBarrier(sysm.task_dim, sensor.name, sensor.lens.width,
                               sensor.lens.height, sensor.K)])
        sensors = (sensor,)
    elif spec.get("sensor") == "range":
        sensor = make_sensor("range", sysm, n_beams=8, latency_steps=0)
        terms = ([GoalBowl(sysm.task_dim) for _ in range(3)]
                 + [DissipationTerm(sysm.task_dim),
                    RangeBarrier(sysm.task_dim, sensor.name, sensor.n_beams)])
        sensors = (sensor,)
    tr = make_trainable("energy_shaping", sysm, terms=terms)
    task = make_task(spec["task"], sysm, **spec.get("task_kw", {}))
    cfg = RolloutCfg(ep_steps=steps, n_eps=2)
    roll = Rollout(sysm, tr, task, cfg, sensors)
    goals = task.sample(2, make_gen(0))
    tr_ = roll.trace(tr.init()[None], goals, 1)

    poses = sysm.render_poses(tr_.states)
    tpos = sysm.task_position(tr_.states)
    extras = sysm.render_extras(tr_.states)
    static = sysm.render_static(tr_.states)
    spec_r = sysm.render_spec()
    runs = {}
    for who in ("trained", "prior"):
        eps = []
        for b in range(2):
            ghost = sysm.render_poses(
                sysm.nominal_state(goals[b], torch.zeros_like(goals[b])))
            rec = {"pose": _r(poses[:, b]), "task": _r(tpos[:, b]),
                   "goal_pose": _r(ghost), "goal_task": _r(goals[b]),
                   "alive": [int(v) for v in tr_.alive[:, b]]}
            for k, v in extras.items():
                rec[k] = _r(v[:, b])
            for g in static.get("obstacles", []):
                grp = {"kind": g["kind"]}
                for k, v in g.items():
                    if k != "kind":
                        grp[k] = (_r(v[0, b]) if isinstance(v, torch.Tensor)
                                  else float(v))
                rec.setdefault("obstacles", []).append(grp)
            if sensor is not None:
                fr = sensor.render_frame(tr_.states)
                if "uv" in fr:
                    rec["uv"] = _r(fr["uv"][:, b])
                    rec["seen"] = [int(v) for v in fr["visible"][:, b].reshape(-1)]
                if "range" in fr:
                    rec["beams"] = _r(fr["range"][:, b])
            eps.append(rec)
        runs[who] = eps

    robot = {
        "key": spec["system"], "label": spec["system"],
        "system": spec["system"], "trainable": "energy_shaping", "task": spec["task"],
        "dim": spec_r["dim"], "ground": spec_r["ground"], "scale": spec_r["scale"],
        "bodies": spec_r["bodies"],
        **{k: v for k, v in spec_r.items()
           if k not in ("dim", "ground", "scale", "bodies")},
        "dt": cfg.dt, "n_frames": steps + 1,
        "switch_frac": 1.0 / task.n_legs, "n_legs": task.n_legs,
        "task_labels": list(sysm.labels()), "tol": 0.25, "runs": runs,
        "eval": {w: {"crash_rate": 0.0, "success_rate": 1.0, "final_err_mean": 0.1}
                 for w in ("trained", "prior")},
        "curve": {"fitness_elite": [1.0, 0.5], "crash_rate": [0.1, 0.0],
                  "final_err": [1.0, 0.2]},
        "genome": {"dim": tr.dim, "policy": tr.policy_dim,
                   "terms": [t.kind for t in tr.terms]},
    }
    if sensor is not None:
        rs = sensor.render_spec()
        if rs:
            robot["sensor"] = {k: (v if not isinstance(v, torch.Tensor) else _r(v))
                               for k, v in rs.items()}
    return robot


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    robots = [_build(spec) for _, spec in CASES]
    p = tmp_path_factory.mktemp("viz") / "data.json"
    p.write_text(json.dumps({"robots": robots}, separators=(",", ":")))
    return p


def test_every_overlay_combination_renders(payload, pytestconfig):
    """Drives every robot through every mode, episode and several frames, and
    asserts each declared overlay actually produces sprites -- rendering without
    throwing is not the same as drawing anything."""
    root = pytestconfig.rootpath
    r = subprocess.run(
        [node, str(root / "scripts" / "check_visualizer.js"),
         str(root / "scripts" / "visualizer.html"), str(payload)],
        capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr


def test_payload_covers_every_overlay(payload):
    """The check is only worth as much as its coverage."""
    robots = json.loads(payload.read_text())["robots"]
    kinds = set()
    for r in robots:
        ep = r["runs"]["trained"][0]
        if r.get("sensor"):
            kinds.add(r["sensor"]["type"])
        for key in ("obstacles", "cable", "feet", "beams", "uv"):
            if ep.get(key):
                kinds.add(key)
        kinds.add(f"dim{r['dim']}")
    for need in ("landmark_camera", "range", "obstacles", "cable", "feet",
                 "dim2", "dim3"):
        assert need in kinds, f"no case exercises {need}"
