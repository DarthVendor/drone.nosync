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
from lagrangian_es.systems import make_system                 # noqa: E402
from lagrangian_es.tasks import make_task                     # noqa: E402
from lagrangian_es.trainables import make_trainable           # noqa: E402
from lagrangian_es.util import make_gen                       # noqa: E402

PRESETS = {
    "quadrotor": dict(system="quadrotor", trainable="quadrotor_agent",
                      task="waypoint_pair", label="SE(3) quadrotor",
                      rollout=dict(n_eps=8), gens=60, pop=48, stride=1),
    "quadruped": dict(system="quadruped", trainable="quadruped_agent",
                      task="base_pose", label="Planar quadruped",
                      rollout=dict(dt=0.002, ep_steps=500, n_eps=4, lambda_e=2e-4,
                                   lambda_s=0.05, lambda_r=0.05, pos_eps=1e-6,
                                   dead_cost=3.0),
                      gens=40, pop=32, stride=2),
    "two_link_arm": dict(system="two_link_arm", trainable="arm_agent",
                         task="joint_pair", label="Two-link arm (minimal coords)",
                         rollout=dict(dt=0.01, ep_steps=300, n_eps=8, lambda_e=0.01,
                                      lambda_s=0.0, pos_eps=1e-4),
                         gens=40, pop=48, stride=1),
    "maximal_chain": dict(system="maximal_chain", trainable="arm_agent",
                          task="joint_pair", label="Two-link chain (maximal coords)",
                          rollout=dict(dt=0.002, ep_steps=400, n_eps=8, lambda_e=0.01,
                                       lambda_s=0.0, pos_eps=1e-4),
                          gens=40, pop=48, stride=2),
}


def _r(x, nd=3):
    return [round(float(v), nd) for v in x.reshape(-1)]


def capture(system, roll, theta, goals, seed, stride=1):
    tr = roll.trace(theta[None], goals, seed)
    T, B = tr.goals.shape[0], tr.goals.shape[1]
    poses = system.render_poses(tr.states)                    # [T+1, B, nb, K]
    extras = system.render_extras(tr.states)
    sel = list(range(0, T + 1, stride))
    alive_sel = [min(i, T - 1) for i in sel]
    out = []
    for b in range(B):
        ghost = system.render_poses(
            system.nominal_state(goals[b], torch.zeros_like(goals[b])))
        rec = {"pose": _r(poses[sel, b]),
               "alive": [int(tr.alive[i, b]) for i in alive_sel],
               "goal_pose": _r(ghost),
               "goal_task": _r(goals[b], 4)}
        for k, v in extras.items():
            rec[k] = _r(v[sel, b], 4)
        out.append(rec)
    return out


def build_one(name, args):
    pre = PRESETS[name]
    rc = RolloutCfg(**pre["rollout"])
    cfg = Config(system=pre["system"], trainable=pre["trainable"], task=pre["task"],
                 seed=args.seed, rollout=rc,
                 es=ESCfg(pop=pre["pop"], gens=args.gens or pre["gens"], sigma0=0.15,
                          elite_frac=0.5, strategy="ga", whiten=True,
                          metric_every=5, null_mode="cap"))
    system = make_system(pre["system"])
    trainable = make_trainable(pre["trainable"], system)
    task = make_task(pre["task"], system)
    print(f"  training {name} ({trainable.dim} genome slots) ...", flush=True)
    res = train(cfg, system, trainable, task)
    theta0 = trainable.init()

    tol = 0.25 if pre["system"] == "quadrotor" else 0.05
    ev_t = evaluate(system, trainable, task, res.theta, rc, n_tasks=128, tol=tol)
    ev_0 = evaluate(system, trainable, task, theta0, rc, n_tasks=128, tol=tol)

    roll = Rollout(system, trainable, task, rc)
    goals = task.sample(args.episodes, make_gen(4242))
    spec = system.render_spec()
    stride = pre["stride"]
    return {
        "key": name, "label": pre["label"],
        "system": pre["system"], "trainable": pre["trainable"], "task": pre["task"],
        "dim": spec["dim"], "ground": spec["ground"], "scale": spec["scale"],
        "bodies": spec["bodies"],
        "dt": rc.dt * stride, "n_frames": len(range(0, rc.ep_steps + 1, stride)),
        "switch_frac": 0.5 if task.n_legs > 1 else 1.0,
        "n_legs": task.n_legs, "task_labels": _labels(pre["system"]),
        "tol": tol,
        "runs": {"trained": capture(system, roll, res.theta, goals, 4243, stride),
                 "prior": capture(system, roll, theta0, goals, 4243, stride)},
        "eval": {"trained": ev_t, "prior": ev_0},
        "curve": {k: [round(h[k], 5) for h in res.history]
                  for k in ("fitness_elite", "crash_rate", "final_err")},
        "genome": {"dim": trainable.dim, "policy": trainable.policy_dim,
                   "terms": [t.kind for t in getattr(trainable, "terms", [])]},
    }


def _labels(system_name):
    return {"quadrotor": ["x", "y", "z"],
            "planar_quad": ["x", "z"],
            "quadruped": ["x", "height", "pitch"]}.get(system_name, ["q1", "q2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="trajectories.json")
    ap.add_argument("--robots", nargs="+", default=list(PRESETS),
                    choices=list(PRESETS))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    payload = {"robots": [build_one(n, args) for n in args.robots]}
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(payload['robots'])} robots)")
    for r in payload["robots"]:
        e0, e1 = r["eval"]["prior"], r["eval"]["trained"]
        print(f"  {r['label']:36s} err {e0['final_err_mean']:.3f} -> "
              f"{e1['final_err_mean']:.3f}   fail {e0['crash_rate']:.2f} -> "
              f"{e1['crash_rate']:.2f}")


if __name__ == "__main__":
    main()
