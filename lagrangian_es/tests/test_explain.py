"""The decision explainer's promises: attention rows are distributions over
the keys it names, saliency is a gradient norm, counterfactuals are probability shifts,
and the tracer explains exactly the decisions it flew."""
import torch

from lagrangian_es.composer import Tracer, explain_decision
from lagrangian_es.config import Config, RolloutCfg
from lagrangian_es.es import build, build_composer, build_sensors
from lagrangian_es.rollout import Rollout
from lagrangian_es.util import make_gen


def _cfg(comp="policy"):
    return Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment="singapore_cbd",
                  sensors=("range", "depth_camera"), gating="arrival", seed=0, composer=comp,
                  composer_kw=(("reach", 10.0), ("every", 10), ("measure_every", 10)),
                  task_kw=(("n_legs", 2), ("max_leg", 10.0)), system_kw=(("free_start", True),),
                  trainable_kw=(("learned", True), ("damp_mode", "beams")),
                  rollout=RolloutCfg(n_eps=3, ep_steps=60, dead_mode="constant", dead_cost=6.0, goal_bonus=15.0))


def test_explain_one_decision():
    cfg = _cfg(); sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr)
    sens = build_sensors(cfg, sysm); comp.attach(sens)
    s = sysm.reset(3, make_gen(1)); goals = task.sample(3, make_gen(2))
    ctx = {"x": sysm.task_position(s), "v": sysm.task_velocity(s), "goal": goals[:, 0],
           "alive": torch.ones(3, dtype=torch.bool), "arrived": torch.zeros(3, dtype=torch.bool),
           "leg": torch.zeros(3, dtype=torch.long), "t": 0, "state": s, "chain": [],
           "obs": {x.name: x.observe(s, make_gen(3)) for x in sens}}
    ex = explain_decision(comp, ctx, b=1)
    a = ex["attention"]; assert a is not None
    # self + 24 beams + 50 patches + the stream memory.  The stream key is
    # there even on this FIRST decision, with `ctx["chain"]` empty: every chain
    # now opens with the goal preface, so it is never zero-length and
    # `read_out` no longer skips the chain blocks outright.  It used to be 76.
    assert len(a["keys"]) == 2 + 24 + 50 + 1 and abs(sum(a["action_query"]) - 1.0) < 1e-6
    assert a["keys"][-1] == "stream memory"
    assert all(abs(sum(q) - 1.0) < 1e-6 for q in a["constraint_queries"])
    sal = ex["saliency"]; assert len(sal["entities"]) == 74 and min(sal["entities"]) >= 0.0
    cf = ex["counterfactual_shift_p"]
    assert set(cf) == {"no_beams", "no_camera", "no_stream", "no_goal"} and all(-1.0 <= v <= 1.0 for v in cf.values())
    o = ex["outputs"]; V = comp.net.vocab
    assert o["probs"].shape == (V.V,) and abs(float(o["probs"].sum()) - 1.0) < 1e-5
    assert isinstance(o["chosen_name"], str) and len(o["top"]) == 5 and o["top"][0][1] >= o["top"][1][1]


def test_tracer_explains_every_decision_it_flew():
    cfg = _cfg("tracer"); sysm, tr, task = build(cfg); comp = build_composer(cfg, sysm, tr)
    assert isinstance(comp, Tracer)
    roll = Rollout(sysm, tr, task, cfg.rollout, build_sensors(cfg, sysm), composer=comp)
    roll.run(tr.init()[None], task.sample(3, make_gen(4)), 5)
    assert 1 <= len(comp.trace) <= 60 // 10 + 1
    assert comp.trace[-1]["t"] > comp.trace[0]["t"] or len(comp.trace) == 1
    assert "attention" in comp.trace[0] and "counterfactual_shift_p" in comp.trace[0]
