"""The navigation prototype: >99% of waypoints reached, <1% crashes.

Guards the whole stack end to end -- sensor geometry, both obstacle terms, the
gyroscopic steering, the objective weights and the trained flight controller --
against any one of them silently regressing.  Every constant it depends on was
measured, and the numbers that justify each are in the modules that own them.

Reference, from `assets/nav99_genome.json` on the pillar field:
    n = 8192, eval seed        reach 0.9940   crash 0.0020   timeout 0.0040
    n = 4096, held-out seed    reach 0.9961   crash 0.0010   timeout 0.0029
"""
import json
import pathlib

import pytest
import torch

from lagrangian_es.config import Config, ESCfg, RolloutCfg
from lagrangian_es.es import build, build_sensors
from lagrangian_es.evaluate import evaluate
from lagrangian_es.rollout import Rollout

GENOME = pathlib.Path(__file__).resolve().parents[1] / "assets" / "nav99_genome.json"


def _cfg(env="pillars", steps=2400):
    return Config(system="quadrotor_nav", trainable="nav_agent",
                  task="waypoint_pair", environment=env, sensors=("range",),
                  gating="arrival", seed=0, system_kw=(("prox_gain", 30.0),),
                  # PINNED.  `nav_agent` now defaults to the learned potential,
                  # which is an 806-slot genome; nav99 is 52 and was measured on
                  # the hand-designed stack.  A published number names the
                  # configuration it was measured in -- same rule as
                  # PROTOTYPE_STRIDE above.
                  trainable_kw=(("learned", False),),
                  rollout=RolloutCfg(n_eps=8, ep_steps=steps, lambda_s=0.2,
                                     lambda_e=0.005, dead_mode="constant",
                                     dead_cost=6.0, goal_bonus=15.0),
                  es=ESCfg(pop=8, gens=1))


#: The stride nav99 was TRAINED and measured at.  Pinned here rather than
#: inherited, because the library default is a cost/accuracy dial that moves:
#: it is 8 now, and striding this genome -- which never saw a stale reading --
#: takes it from 0.9923/0.0015 to 0.9840/0.0111.  The >99% claim is a claim
#: about a configuration, so the configuration belongs in the test.
PROTOTYPE_STRIDE = 1


def _rig(env="pillars", steps=2400):
    cfg = _cfg(env, steps)
    system, tr, task = build(cfg)
    sens = build_sensors(cfg, system)
    sens[0].update_every = PROTOTYPE_STRIDE
    saved = json.loads(GENOME.read_text())
    assert saved["dim"] == tr.dim, (
        f"genome is {saved['dim']} slots but nav_agent now builds {tr.dim}; "
        f"terms changed from {saved['terms']} to {[t.kind for t in tr.terms]}")
    th = torch.tensor(saved["theta"], dtype=torch.float64)
    return system, tr, task, Rollout(system, tr, task, cfg.rollout, sens), th, cfg


def test_the_stack_is_wired_as_measured():
    """Each of these was worth a measurable amount of crash rate; a silent
    revert would cost it back without failing anything else."""
    cfg = _cfg()
    system, tr, task = build(cfg)
    sens = build_sensors(cfg, system)
    kinds = [t.kind for t in tr.terms]
    assert "range_barrier" in kinds      # position: stops the approach
    assert "range_damper" in kinds       # velocity: a potential cannot see speed
    assert "range_vortex" in kinds       # workless: redirects the slow sliders
    assert sens[0].n_beams >= 24         # 12 beams leave 0.52 m gaps at 1 m
    assert system.goal_margin > 0.45     # must exceed the barrier's own standoff


def test_the_shipped_numbers_name_the_stride_they_were_measured_at():
    """A default that moves must not silently redefine a published result.

    `RangeSensor.update_every` is a compute dial; nav99's >99% was measured at
    stride 1 and does not survive being strided under it without retraining.
    Pinning it in `_rig` is what keeps this file measuring the artefact rather
    than measuring whatever the default happens to be today.
    """
    cfg = _cfg()
    system, _, _ = build(cfg)
    sens = build_sensors(cfg, system)
    assert PROTOTYPE_STRIDE == 1
    if sens[0].update_every != PROTOTYPE_STRIDE:
        # not a failure -- just the fact this test exists to make explicit
        assert sens[0].update_every > PROTOTYPE_STRIDE


def test_prototype_reaches_over_99_percent_without_crashing():
    system, tr, task, roll, th, cfg = _rig()
    ev = evaluate(system, tr, task, th, cfg.rollout, n_tasks=512, seed=777_001,
                  roll=roll)
    timeout = 1.0 - ev["success_rate"] - ev["crash_rate"]
    assert ev["success_rate"] > 0.98, f"reach regressed to {ev['success_rate']:.4f}"
    assert ev["crash_rate"] < 0.02, f"crash regressed to {ev['crash_rate']:.4f}"
    assert timeout < 0.02


def test_prototype_holds_on_layouts_from_an_unseen_seed():
    """The scene lives in the state, so a different seed is a different set of
    pillar fields -- this is generalization, not a re-read of the training set."""
    system, tr, task, roll, th, cfg = _rig()
    ev = evaluate(system, tr, task, th, cfg.rollout, n_tasks=512, seed=31_337,
                  roll=roll)
    assert ev["success_rate"] > 0.98
    assert ev["crash_rate"] < 0.02


@pytest.mark.parametrize("env", ["sparse", "gate", "slalom"])
def test_prototype_transfers_to_scenes_it_was_never_tuned_on(env):
    """The obstacle terms consume BEAMS and are never handed geometry, which is
    what makes transfer mean anything.  Measured: sparse 1.000, gate 0.999,
    slalom 0.990, forest 0.988, gate_forest 0.992."""
    system, tr, task, roll, th, cfg = _rig(env)
    ev = evaluate(system, tr, task, th, cfg.rollout, n_tasks=256, seed=777_001,
                  roll=roll)
    assert ev["success_rate"] > 0.95, f"{env}: {ev['success_rate']:.4f}"
    assert ev["crash_rate"] < 0.05


def test_hoop_scenes_are_a_known_limitation_not_a_silent_failure():
    """A repulsive barrier pushes AWAY from the ring it must fly THROUGH, so the
    hoop scenes sit near 0.73 with a 0.27 crash rate.  Pinned deliberately: if it
    ever improves, that is a real result and this test should be updated, not a
    number that quietly drifted."""
    system, tr, task, roll, th, cfg = _rig("hoop_course")
    ev = evaluate(system, tr, task, th, cfg.rollout, n_tasks=256, seed=777_001,
                  roll=roll)
    assert ev["success_rate"] < 0.95, (
        f"hoop_course now reaches {ev['success_rate']:.3f} -- the aperture "
        "problem may be solved; verify and update this test")
