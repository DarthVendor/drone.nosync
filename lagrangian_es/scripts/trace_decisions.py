#!/usr/bin/env python3
"""Trace the composer's decisions on a few flights -- crashes first.

    python scripts/trace_decisions.py SP composer.pt genome.json out.json [n_crashes] [n_arrived]

Finds crashed flights in a deterministic batch on the v2 rig, then re-flies
each chosen episode with the `Tracer` explaining every decision of it.
"""
import json, sys, torch
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parents[1].joinpath("src").as_posix())
from lagrangian_es.config import Config, RolloutCfg
from lagrangian_es.es import build, build_composer, build_sensors
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen

SP, W, G, OUT = sys.argv[1:5]
N_CRASH = int(sys.argv[5]) if len(sys.argv) > 5 else 3
N_ARR = int(sys.argv[6]) if len(sys.argv) > 6 else 1
torch.set_num_threads(4); N = 32

def cfg_for(composer, watch=0):
    ckw = (("reach", 10.0), ("every", 10), ("measure_every", 10), ("weights", W)) + ((("watch", watch),) if composer == "tracer" else ())
    return Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd",
                  sensors=("range", "range_down", "depth_camera", "tilt"), sensor_kw=(("range", (("spread", 2.0944),)),),
                  gating="arrival", seed=0, composer=composer, composer_kw=ckw,
                  task_kw=(("n_legs", 2), ("max_leg", 20.0)),
                  system_kw=(("prox_gain", 30.0), ("free_start", True), ("yaw_mode", "learned")),
                  trainable_kw=(("learned", True), ("damp_mode", "beams"), ("extra_obs", (("range_down", 4), ("tilt", 3)))),
                  rollout=RolloutCfg(n_eps=N, ep_steps=1800, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0,
                                     stop_on_arrival=False, stop_quantile=1.0))

th = torch.tensor(json.load(open(G))["theta"], dtype=torch.float64)
# --- which flights crash --------------------------------------------------------
cfg = cfg_for("policy"); sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr)
roll = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp)
goals = task.sample(N, make_gen(9_900_001))
res = roll.run(th[None], goals, 9_900_002)
crashed = (~res.alive.bool()).nonzero().flatten().tolist(); arrived = res.success.bool().nonzero().flatten().tolist()
pick = crashed[:N_CRASH] + arrived[:N_ARR]
print(f"  batch of {N}: {len(crashed)} crashed, {len(arrived)} arrived; tracing {pick}", flush=True)
# --- trace each ------------------------------------------------------------------
flights = []
for b in pick:
    c2 = cfg_for("tracer", watch=b); s2, t2, k2 = build(c2); cp = build_composer(c2, s2, t2)
    r2 = Rollout(s2, t2, k2, c2.rollout, build_sensors(c2, s2), composer=cp)
    t = r2.trace(th[None], goals, 9_900_002)
    al = t.alive[:, b].bool(); died = bool((~al).any()); end = int((~al).nonzero()[0]) if died else t.alive.shape[0] - 1
    P = t.states["p"][:end + 1, b]; R = torch.atan2(t.states["R"][:end + 1, b, 1, 0], t.states["R"][:end + 1, b, 0, 0])
    f = {"episode": b, "outcome": "crash" if died else ("arrived" if bool(t.legs[end - 1, b] == 1) else "timeout"),
         "end_step": end, "path": [[round(float(x), 2), round(float(y), 2)] for x, y in P[:, :2]],
         "yaw": [round(float(v), 3) for v in R], "goals": [[round(float(g[0]), 2), round(float(g[1]), 2)] for g in goals[b]],
         "decisions": []}
    for d in cp.trace:
        tk = d["tokens"]; o = d["outputs"]; ty = tk["ent_types"].tolist(); ent = tk["entities"]; nb = sum(1 for v in ty if v == 2)
        f["decisions"].append({
            "t": d["t"], "beams": [round(float(v), 3) for v in ent[:nb, 3]], "beam_az": [round(float(v), 3) for v in torch.atan2(ent[:nb, 1], ent[:nb, 0])],
            "patches": [round(float(v), 3) for v in ent[nb:, 3]], "chain_len": int(tk["chain"].shape[0]),
            "goal_ego": [round(float(v), 2) for v in o["goal_ego"]], "sub_ego": [round(float(v), 2) for v in o["sub_ego"]],
            "sub_world": [round(float(v), 2) for v in o["sub_world"]], "alpha": [round(float(v), 3) for v in o["alpha"]],
            "gate": [round(float(v), 3) for v in o["gate"]], "heading_delta": round(float(o["heading_delta"]), 3),
            "heading_gate": round(float(o["heading_gate"]), 3), "value": round(float(o["value"]), 3),
            "attn_sub": [round(v, 4) for v in d["attention"]["subgoal_query"]] if d["attention"] else None,
            "attn_con": [[round(v, 4) for v in q] for q in d["attention"]["constraint_queries"]] if d["attention"] else None,
            "sal": {"self": round(d["saliency"]["self"], 4), "goal": round(d["saliency"]["goal"], 4),
                    "entities": [round(v, 4) for v in d["saliency"]["entities"]], "chain": [round(v, 4) for v in d["saliency"]["chain"]]},
            "cf": {k: round(v, 3) for k, v in d["counterfactual_shift_m"].items()}})
    f["n_beams"] = nb; flights.append(f)
    print(f"  episode {b}: {f['outcome']} at {end*0.02:.1f} s, {len(f['decisions'])} decisions", flush=True)
env = json.load(open(__import__("pathlib").Path(__file__).resolve().parents[1] / "src/lagrangian_es/environments/maps/singapore_cbd.json"))
json.dump({"flights": flights, "boxes": env["boxes"], "patch_grid": [10, 5], "dt": 0.02, "reach": 10.0}, open(OUT, "w"))
print("TRACE DONE", flush=True)
