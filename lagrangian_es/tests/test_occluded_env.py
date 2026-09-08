"""The occluded map: no waypoint can see any other, and every waypoint is clear.

This is the environment for the one failure the city singled out -- a wall on
the straight line between checkpoints -- so the property it exists for is
tested as an identity over ALL pairs, not sampled.
"""
import itertools

import torch

from lagrangian_es.config import Config
from lagrangian_es.es import build
from lagrangian_es.util import make_gen


def _env():
    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="occluded", sensors=("range",), gating="arrival", seed=0,
                 task_kw=(("n_legs", 2), ("max_leg", 0.0)), system_kw=(("free_start", True),),
                 trainable_kw=(("learned", True), ("damp_mode", "beams")))
    sysm, tr, task = build(cfg)
    s = sysm.reset(1, make_gen(0))
    wps = torch.tensor(sysm.env.waypoints, dtype=torch.float64)
    return sysm, task, s, wps


def _crosses_wall(sysm, s, a, b, step=0.1):
    # sample finer than a wall is thick, or a shallow crossing steps over it
    n = int(float((b - a).norm()) / step) + 2
    ts = torch.linspace(0, 1, n, dtype=a.dtype)[:, None]
    pts = a[None] + ts * (b - a)[None]
    st = {k: v.expand(n, *v.shape[1:]) for k, v in s.items()}
    return bool((sysm.env.sdf(pts, st) < 0.0).any())


def test_every_waypoint_pair_is_occluded():
    sysm, task, s, wps = _env()
    assert wps.shape[0] == 16
    for i, j in itertools.combinations(range(wps.shape[0]), 2):
        assert _crosses_wall(sysm, s, wps[i], wps[j]), f"waypoints {i} and {j} can see each other"


def test_every_waypoint_is_clear_and_inside_its_pocket():
    sysm, task, s, wps = _env()
    st = {k: v.expand(wps.shape[0], *v.shape[1:]) for k, v in s.items()}
    clear = sysm.env.sdf(wps, st)
    assert float(clear.min()) > 1.5, "a waypoint sits too close to its own pocket walls"


def test_the_task_samples_legs_on_this_map():
    sysm, task, s, wps = _env()
    g = task.sample(8, make_gen(1))
    assert g.shape == (8, 2, 3)
    assert not torch.equal(g[:, 0], g[:, 1])


def test_the_cull_covers_every_block_within_reach():
    """Exact iff cull_k >= the blocks whose surface lies within a ray's reach,
    checked over many free positions on both maps at the fan's 6 m."""
    for env in ("singapore_cbd", "occluded"):
        cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour", environment=env, sensors=("range",),
                     gating="arrival", seed=0, task_kw=(("n_legs", 2), ("max_leg", 0.0)), system_kw=(("free_start", True),),
                     trainable_kw=(("learned", True), ("damp_mode", "beams")))
        sysm, tr, task = build(cfg); s = sysm.reset(1024, make_gen(11)); grp = sysm.env.groups[0]
        c, h, a = s["boxes/c"][0], s["boxes/h"][0], s["boxes/a"][0]; p = s["p"]
        d = p[:, None, :2] - c[None]; ca, sa = torch.cos(a), torch.sin(a)
        lx = d[..., 0] * ca + d[..., 1] * sa; ly = -d[..., 0] * sa + d[..., 1] * ca
        qx = lx.abs() - h[None, :, 0]; qy = ly.abs() - h[None, :, 1]
        surf = torch.sqrt(qx.clamp_min(0) ** 2 + qy.clamp_min(0) ** 2) + torch.maximum(qx, qy).clamp_max(0)
        within = int((surf <= 6.0).sum(1).max())
        assert within <= grp.cull_k, (env, within, grp.cull_k)
