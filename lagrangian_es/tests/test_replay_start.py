"""`ReplayStart` restarts episodes in situations a controller previously died in.

The pool reaches the task through `task_kw`, which is config data and therefore
JSON-able -- nested lists, not tensors.  Two things have to survive that trip:
the pool must end up indexable by a tensor of row indices, and float states must
carry the system's dtype.  A pool that silently became float32 would start the
replay somewhere the vehicle never actually was.
"""
import torch

from lagrangian_es.config import Config
from lagrangian_es.es import build
from lagrangian_es.tasks import make_task


def _fixture(n=6):
    cfg = Config(system="quadrotor_nav", trainable="nav_agent",
                 task="waypoint_pair", environment="sparse",
                 sensors=("range",), gating="arrival", seed=0)
    system, _, task = build(cfg)
    goals = task.sample(n, torch.Generator().manual_seed(0))
    state = system.reset(n, torch.Generator().manual_seed(1))
    pool = {k: v.tolist() for k, v in state.items()
            if torch.is_tensor(v) and v.ndim and v.shape[0] == n}
    replay = make_task("replay_start", system, base="waypoint_pair", mix=0.5,
                       states=pool, goals=goals.reshape(-1).tolist(),
                       gating="arrival")
    return system, replay, state, n


def test_pool_arrives_as_tensors_in_the_system_dtype():
    system, replay, state, _ = _fixture()
    assert replay.states, "the pool should not be empty"
    for key, val in replay.states.items():
        assert torch.is_tensor(val), f"{key} stayed a list"
        if state[key].is_floating_point():
            assert val.dtype == system.dtype
            assert torch.allclose(val, state[key])


def test_replayed_rows_land_in_the_recorded_situation():
    system, replay, _, n = _fixture()
    E, reps = 8, 3
    goals = replay.sample(E, torch.Generator().manual_seed(2))
    fresh = system.reset(E * reps, torch.Generator().manual_seed(3))
    out = replay.place_start(fresh, goals.repeat(reps, 1, 1))
    _, k, idx = replay._picked
    assert 0 < k < E
    for j in range(reps):                      # one block per population member
        lo = j * E
        assert torch.allclose(out["p"][lo:lo + k], replay.states["p"][idx])
        assert torch.equal(out["p"][lo + k:lo + E], fresh["p"][lo + k:lo + E])


def test_the_obstacle_field_is_replayed_with_the_pose():
    """A recovery in a different scene is a different problem."""
    system, replay, _, _ = _fixture()
    scene = [k for k in replay.states if "/" in k]
    assert scene, "the obstacle field should ride in the replayed state"
    E, goals = 8, None
    goals = replay.sample(E, torch.Generator().manual_seed(4))
    fresh = system.reset(E, torch.Generator().manual_seed(5))
    out = replay.place_start(fresh, goals)
    _, k, idx = replay._picked
    for key in scene:
        assert torch.allclose(out[key][:k], replay.states[key][idx])


def test_an_empty_pool_leaves_every_episode_where_the_base_put_it():
    cfg = Config(system="quadrotor_nav", trainable="nav_agent",
                 task="waypoint_pair", environment="sparse",
                 sensors=("range",), gating="arrival", seed=0)
    system, _, _ = build(cfg)
    replay = make_task("replay_start", system, base="waypoint_pair", mix=0.5,
                       gating="arrival")
    assert replay.n_pool == 0
    goals = replay.sample(8, torch.Generator().manual_seed(6))
    fresh = system.reset(8, torch.Generator().manual_seed(7))
    # `waypoint_pair` places no starts of its own, so nothing should move
    out = replay.place_start(fresh, goals)
    assert torch.equal(out["p"], fresh["p"])
    assert torch.equal(out["v"], fresh["v"])


def test_the_base_tasks_start_placement_is_kept_for_unreplayed_rows():
    """`ReplayStart` overwrites the replayed rows only.  Everything else must
    start where the base task would have put it -- `city_tour` begins a tour on
    a street beside its first waypoint, and losing that ruins most of the batch.
    """
    import torch

    from lagrangian_es.config import Config
    from lagrangian_es.es import build
    from lagrangian_es.tasks import make_task

    cfg = Config(system="quadrotor_nav", trainable="nav_agent", task="city_tour",
                 environment="singapore_cbd", sensors=("range",),
                 gating="arrival", seed=0,
                 task_kw=(("n_legs", 2), ("max_leg", 10.0)),
                 system_kw=(("prox_gain", 30.0), ("free_start", True)))
    system, _, city = build(cfg)

    n = 6
    goals = city.sample(n, torch.Generator().manual_seed(0))
    state = system.reset(n, torch.Generator().manual_seed(1))
    placed = city.place_start(state, goals)
    pool = {k: v.tolist() for k, v in placed.items()
            if torch.is_tensor(v) and v.ndim and v.shape[0] == n}
    replay = make_task("replay_start", system, base="city_tour", mix=0.5,
                       states=pool, goals=goals.reshape(-1).tolist(),
                       gating="arrival", n_legs=2, max_leg=10.0)

    E = 8
    g2 = replay.sample(E, torch.Generator().manual_seed(2))
    fresh = system.reset(E, torch.Generator().manual_seed(3))
    out = replay.place_start(fresh, g2)
    _, k, _ = replay._picked
    expected = city.place_start(fresh, g2)
    assert torch.allclose(out["p"][k:], expected["p"][k:]), \
        "rows that are not replays must keep the base task's placement"
    assert not torch.allclose(out["p"][:k], fresh["p"][:k]), \
        "replayed rows should have been overwritten"
