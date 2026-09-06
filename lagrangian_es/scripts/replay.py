#!/usr/bin/env python3
"""Export flight/motion trajectories for the visualizer.

Plant-agnostic: everything drawable comes from `system.render_spec()` and
`system.render_poses()`, so a new robot becomes viewable by describing itself
rather than by editing the exporter or the renderer.

Goals are exported as a *ghost pose* -- the goal pushed through `nominal_state`
and rendered -- so the target is drawn as a translucent copy of the robot on
every plant, instead of needing a per-plant goal marker.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lagrangian_es.config import Config, ESCfg, RolloutCfg   # noqa: E402
from lagrangian_es.es import train                            # noqa: E402
from lagrangian_es.evaluate import evaluate                   # noqa: E402
from lagrangian_es.rollout import Rollout                     # noqa: E402
from lagrangian_es.sensors import make_sensor                 # noqa: E402
from lagrangian_es.trainables.sensor_terms import FovBarrier  # noqa: E402
from lagrangian_es.trainables.terms import DissipationTerm, GoalBowl  # noqa: E402
from lagrangian_es.systems import make_system                 # noqa: E402
from lagrangian_es.tasks import make_task                     # noqa: E402
from lagrangian_es.trainables import make_trainable           # noqa: E402
from lagrangian_es.util import make_gen                       # noqa: E402

PRESETS = {
    "quadrotor": dict(system="quadrotor", trainable="quadrotor_agent",
                      task="waypoint_pair", label="SE(3) quadrotor",
                      rollout=dict(n_eps=8), gens=60, pop=48, stride=1),
    "quadrotor_vision": dict(system="quadrotor", trainable="energy_shaping",
                            task="waypoint_pair", label="Quadrotor + landmark camera",
                            rollout=dict(n_eps=8), gens=60, pop=48, stride=1,
                            sensor=dict(kind="landmark_camera", n_landmarks=14,
                                        sigma_px=0.4, dropout=0.03,
                                        latency_steps=3)),
    "quadrotor_nav": dict(system="quadrotor_nav", trainable="energy_shaping",
                         task="waypoint_pair", label="Quadrotor + obstacle field",
                         rollout=dict(n_eps=8, lambda_s=0.2), gens=60, pop=48,
                         stride=1, environment="pillars",
                         sensor=dict(kind="range", n_beams=12, sigma=0.02,
                                     latency_steps=1)),
    "quadrotor_hoops": dict(system="quadrotor_nav", trainable="energy_shaping",
                            task="hoop_course", task_kw=dict(n_gates=3),
                            label="Quadrotor + hoop course",
                            rollout=dict(n_eps=8, lambda_s=0.2, ep_steps=340),
                            gens=60, pop=48, stride=1, environment="hoop_course",
                            sensor=dict(kind="range", n_beams=12, sigma=0.02,
                                        latency_steps=1)),
    "quadruped": dict(system="quadruped", trainable="quadruped_agent",
                      task="base_pose", label="Planar quadruped",
                      rollout=dict(dt=0.002, ep_steps=500, n_eps=4, lambda_e=2e-4,
                                   lambda_s=0.05, lambda_r=0.05, pos_eps=1e-6,
                                   dead_cost=3.0),
                      gens=40, pop=32, stride=2),
    "quadrotor_payload": dict(system="quadrotor_payload", trainable="quadrotor_agent",
                             task="waypoint_pair", label="Quadrotor + slung package",
                             rollout=dict(dt=0.008, ep_steps=625, n_eps=8,
                                          lambda_s=0.08),
                             gens=60, pop=48, stride=3),
    "two_link_arm": dict(system="two_link_arm", trainable="arm_agent",
                         task="joint_pair", label="Two-link arm (minimal coords)",
                         rollout=dict(dt=0.01, ep_steps=300, n_eps=8, lambda_e=0.01,
                                      lambda_s=0.0, pos_eps=1e-4),
                         gens=40, pop=48, stride=1),
    "maximal_chain": dict(system="maximal_chain", trainable="arm_agent",
                          task="joint_pair", label="Two-link chain (maximal coords)",
                          rollout=dict(dt=0.005, ep_steps=600, n_eps=8, lambda_e=0.01,
                                       lambda_s=0.0, pos_eps=1e-4),
                          gens=40, pop=48, stride=2),
}


def _r(x, nd=3):
    return [round(float(v), nd) for v in x.reshape(-1)]


def capture(system, roll, theta, goals, seed, stride=1, sensor=None):
    tr = roll.trace(theta[None], goals, seed)
    T, B = tr.goals.shape[0], tr.goals.shape[1]
    poses = system.render_poses(tr.states)                    # [T+1, B, nb, K]
    task = system.task_position(tr.states)                    # [T+1, B, task_dim]
    extras = system.render_extras(tr.states)
    static = system.render_static(tr.states)
    sel = list(range(0, T + 1, stride))
    alive_sel = [min(i, T - 1) for i in sel]
    out = []
    for b in range(B):
        ghost = system.render_poses(
            system.nominal_state(goals[b], torch.zeros_like(goals[b])))
        rec = {"pose": _r(poses[sel, b]),
               "task": _r(task[sel, b], 4),
               "alive": [int(tr.alive[i, b]) for i in alive_sel],
               # per-frame target index, so the ghost advances when the drone
               # actually reaches a waypoint rather than on a timer
               "leg": [int(tr.legs[i, b]) for i in alive_sel],
               "goal_pose": _r(ghost),
               "goal_task": _r(goals[b], 4)}
        for k, v in extras.items():
            rec[k] = _r(v[sel, b], 4)
        for g in static.get("obstacles", []):
            grp = {"kind": g["kind"]}
            for k, v in g.items():
                if k == "kind":
                    continue
                grp[k] = (_r(v[0, b], 4) if isinstance(v, torch.Tensor)
                          else float(v))          # frame 0: geometry is constant
            rec.setdefault("obstacles", []).append(grp)
        if sensor is not None:
            f = sensor.render_frame(tr.states)
            if "uv" in f:
                rec["uv"] = _r(f["uv"][sel, b], 1)
                rec["seen"] = [int(x) for x in f["visible"][sel, b].reshape(-1)]
            if "range" in f:
                rec["beams"] = _r(f["range"][sel, b], 3)
        out.append(rec)
    return out


def build_one(name, args):
    pre = PRESETS[name]
    rc = RolloutCfg(**pre["rollout"])
    cfg = Config(system=pre["system"], trainable=pre["trainable"], task=pre["task"],
                 sensors=tuple(), seed=args.seed, rollout=rc,
                 es=ESCfg(pop=pre["pop"], gens=args.gens or pre["gens"], sigma0=0.15,
                          elite_frac=0.5, strategy="ga", whiten=True,
                          metric_every=5, null_mode="cap"))
    skw = {"environment": pre["environment"]} if pre.get("environment") else {}
    system = make_system(pre["system"], **skw)
    sensor, sensors = None, ()
    if pre.get("sensor"):
        sk = dict(pre["sensor"])
        sensor = make_sensor(sk.pop("kind"), system, **sk)
        sensors = (sensor,)
        # a vision genome: the usual core plus a barrier on the image border
        if sensor.kind == "range":
            from lagrangian_es.trainables.sensor_terms import RangeBarrier
            extra = RangeBarrier(system.task_dim, sensor.name, sensor.n_beams,
                                 w0=1.2)
        else:
            extra = FovBarrier(system.task_dim, sensor.name, sensor.lens.width,
                               sensor.lens.height, sensor.K, w0=0.5)
        terms = ([GoalBowl(system.task_dim, a0=1.0) for _ in range(3)]
                 + [DissipationTerm(system.task_dim, d0=1.2), extra])
        trainable = make_trainable(pre["trainable"], system, terms=terms)
    else:
        trainable = make_trainable(pre["trainable"], system)
    task = make_task(pre["task"], system, **pre.get("task_kw", {}))
    print(f"  training {name} ({trainable.dim} genome slots) ...", flush=True)
    res = train(cfg, system, trainable, task, sensors=sensors)
    theta0 = trainable.init()

    tol = 0.25 if pre["system"].startswith("quadrotor") else 0.05
    ev_t = evaluate(system, trainable, task, res.theta, rc, n_tasks=128, tol=tol,
                    roll=Rollout(system, trainable, task, rc, sensors))
    ev_0 = evaluate(system, trainable, task, theta0, rc, n_tasks=128, tol=tol,
                    roll=Rollout(system, trainable, task, rc, sensors))

    roll = Rollout(system, trainable, task, rc, sensors)
    goals = task.sample(args.episodes, make_gen(4242))
    spec = system.render_spec()
    stride = pre["stride"]
    return {
        "key": name, "label": pre["label"],
        "system": pre["system"], "trainable": pre["trainable"], "task": pre["task"],
        "dim": spec["dim"], "ground": spec["ground"], "scale": spec["scale"],
        "bodies": spec["bodies"],
        # optional keys a plant may add (e.g. "cables"); copied generically so a
        # new render hint does not need an exporter edit
        **{k: v for k, v in spec.items()
           if k not in ("dim", "ground", "scale", "bodies")},
        "dt": rc.dt * stride, "n_frames": len(range(0, rc.ep_steps + 1, stride)),
        "switch_frac": 1.0 / task.n_legs,
        "n_legs": task.n_legs, "task_labels": list(system.labels()),
        "tol": tol,
        "runs": {"trained": capture(system, roll, res.theta, goals, 4243, stride,
                                    sensor),
                 "prior": capture(system, roll, theta0, goals, 4243, stride,
                                  sensor)},
        **({"sensor": sensor.render_spec()} if sensor is not None else {}),
        "eval": {"trained": ev_t, "prior": ev_0},
        "curve": {k: [round(h[k], 5) for h in res.history]
                  for k in ("fitness_elite", "crash_rate", "final_err")},
        "genome": {"dim": trainable.dim, "policy": trainable.policy_dim,
                   "terms": [t.kind for t in getattr(trainable, "terms", [])]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="trajectories.json")
    # Drone family by default; the quadruped and arms are still selectable by
    # name but are not trained unless asked for.
    ap.add_argument("--robots", nargs="+",
                    default=["quadrotor", "quadrotor_vision", "quadrotor_nav",
                             "quadrotor_hoops", "quadrotor_payload"],
                    choices=list(PRESETS))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--html", default=None,
                    help="also write a standalone page using scripts/visualizer.html")
    args = ap.parse_args()

    payload = {"robots": [build_one(n, args) for n in args.robots]}
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(payload['robots'])} robots)")
    if args.html:
        tpl = pathlib.Path(__file__).resolve().parent / "visualizer.html"
        page = pathlib.Path(args.html)
        page.write_text(tpl.read_text().replace("__DATA__", out.read_text()))
        print(f"wrote {page}  ({page.stat().st_size / 1024:.0f} KB)")
    for r in payload["robots"]:
        e0, e1 = r["eval"]["prior"], r["eval"]["trained"]
        print(f"  {r['label']:36s} err {e0['final_err_mean']:.3f} -> "
              f"{e1['final_err_mean']:.3f}   fail {e0['crash_rate']:.2f} -> "
              f"{e1['crash_rate']:.2f}")


if __name__ == "__main__":
    main()
