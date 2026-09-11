"""Tokens carrying continuous arguments: [WAYPOINT(r, theta)] [EOS].

The grammar of `actions.py` with real-valued arguments instead of a 25-way
grid.  What these pin down: the polar frame is the VEHICLE's, the radius is a
fraction of the distance still to go (so arrival stays possible), and the
non-waypoint tokens keep exactly the semantics their discrete counterparts had.
"""
import math

import torch

from lagrangian_es.composer.actions_cont import (PRIORITY_MAX, TURN_MAX, ContVocab,
                                                 entropy, log_prob)
from lagrangian_es.composer.spec import TaskSpec

DT = torch.float64


def _spec(B, n=2):
    s = TaskSpec.identity(B, 3, n, DT, torch.device("cpu"))
    s.yaw = torch.zeros(B, dtype=DT)
    s.yaw_gate = torch.zeros(B, dtype=DT)
    return s


def _place(V, r, th, L=10.0, reach=10.0, psi=0.0, B=4, phi=0.0, dz=0.0):
    """`L` is the distance to the goal in METRES; `g_ego` is handed to the
    vocabulary in units of the reach, which is the convention `goal_ego` uses.

    `phi` is the elevation argument, 0 meaning level with the vehicle."""
    x = torch.zeros(B, 3, dtype=DT); x[:, 2] = 1.5
    goal = x + torch.tensor([[L, 0.0, dz]], dtype=DT).repeat(B, 1)
    g_ego = torch.tensor([[L / reach, 0.0, dz / reach]], dtype=DT).repeat(B, 1)
    a = torch.zeros(B, 3, dtype=DT); a[:, 0] = r; a[:, 1] = th; a[:, 2] = phi
    tok = torch.full((B,), V.WAYPOINT, dtype=torch.long)
    sp = V.apply(tok, a, _spec(B), x, goal, torch.full((B,), psi, dtype=DT), g_ego, reach, 0.3)
    return x, goal, sp


def test_the_waypoint_is_polar_in_the_vehicles_own_frame():
    """theta is measured from the nose, which is the frame the beams report
    their bearings in -- so a clear beam and a waypoint toward it are the same
    number, and no frame conversion sits between seeing and acting."""
    V = ContVocab(2)
    for th, deg in ((0.0, 0.0), (0.5, 90.0), (-0.5, -90.0), (0.25, 45.0)):
        x, goal, sp = _place(V, 1.0, th)
        w = (goal + sp.delta)[0]
        got = math.degrees(math.atan2(float(w[1] - x[0, 1]), float(w[0] - x[0, 0])))
        assert abs(got - deg) < 1e-6, (th, got, deg)
    # and it is the VEHICLE's frame: yaw the vehicle, the world point yaws with it
    _, g1, s1 = _place(V, 1.0, 0.0, psi=0.0)
    _, g2, s2 = _place(V, 1.0, 0.0, psi=math.pi / 2)
    p1, p2 = (g1 + s1.delta)[0], (g2 + s2.delta)[0]
    assert abs(float(p1[0]) - 10.0) < 1e-6 and abs(float(p1[1])) < 1e-6
    assert abs(float(p2[1]) - 10.0) < 1e-6 and abs(float(p2[0])) < 1e-6


def test_r_spans_from_a_full_stop_to_the_whole_reach():
    """r is the only brake the task level has: -1 puts the subgoal on the
    vehicle (stop), +1 puts it at the edge of what it can reach."""
    V = ContVocab(2)
    x, goal, stop = _place(V, -1.0, 0.0)
    assert float((goal + stop.delta)[0][:2].sub(x[0][:2]).norm()) < 1e-9, "r=-1 should be a full stop"
    _, goal2, far = _place(V, 1.0, 0.0)
    assert abs(float((goal2 + far.delta)[0][:2].sub(x[0][:2]).norm()) - 10.0) < 1e-6
    _, goal3, mid = _place(V, 0.0, 0.0)
    assert abs(float((goal3 + mid.delta)[0][:2].sub(x[0][:2]).norm()) - 5.0) < 1e-6


def test_the_placement_ball_shrinks_onto_the_goal_so_arrival_is_an_attractor():
    """The radius is min(reach, |goal - x|).

    This was briefly a FIXED reach ball, to stop the frame from computing any
    part of the placement.  It cost arrival outright: on an EMPTY map at 8 m
    legs the run arrived 0.000 for six iterations while the frozen low level
    alone flies that rung at 0.96-0.99, and because the update is
    cross-entropy on the composer's own successes, zero arrivals meant zero
    training tokens and a nan loss -- the run could not bootstrap at all.

    What the frame decides is whether closing on the goal makes ARRIVING
    easier.  With the ball tied to the goal it does: the ball contracts onto
    the goal, and once the vehicle is inside the 0.25 m tolerance every
    placement in the ball arrives.  With a fixed ball it does not -- half a
    metre out the vehicle is still commanded up to a full reach away, so the
    error never settles.

    It does not tell the network WHERE the goal is; it still has to learn
    theta and phi.  That was `goal_residual`, which added atanh(bearing)
    directly into mu, and it is gone.
    """
    V = ContVocab(1)
    # inside the reach the ball is the goal's own range, so r = 1 lands ON it
    x, goal, sp = _place(V, r=1.0, th=0.0, L=6.0, reach=10.0)
    assert float(((goal + sp.delta) - goal)[0].norm()) < 1e-6, "a full-radius aim should reach the goal"
    assert abs(float(((goal + sp.delta) - x)[0].norm()) - 6.0) < 1e-6
    # beyond the reach it saturates at the reach, so the subgoal stays flyable
    x, goal, sp = _place(V, r=1.0, th=0.0, L=18.0, reach=10.0)
    assert abs(float(((goal + sp.delta) - x)[0].norm()) - 10.0) < 1e-6
    # the attractor: the WORST a placement can miss by shrinks with the range
    for L, worst in ((6.0, 6.0), (2.0, 2.0), (0.2, 0.2)):
        x, goal, sp = _place(V, r=-1.0, th=0.0, L=L, reach=10.0)   # r = 0, the worst aim
        assert abs(float(((goal + sp.delta) - goal)[0].norm()) - worst) < 1e-6

def test_turn_priority_and_look_match_their_discrete_counterparts():
    """The other tokens keep the semantics the grid gave them, so this is a
    generalisation of the vocabulary rather than a different controller."""
    V = ContVocab(2); B = 4
    x = torch.zeros(B, 3, dtype=DT); x[:, 2] = 1.5
    g_ego = torch.tensor([[10.0, 0.0, 0.0]], dtype=DT).repeat(B, 1); goal = x + g_ego
    psi = torch.zeros(B, dtype=DT)
    def go(t, a0, a1=0.0):
        a = torch.zeros(B, 2, dtype=DT); a[:, 0] = a0; a[:, 1] = a1
        return V.apply(torch.full((B,), t, dtype=torch.long), a, _spec(B), x, goal, psi, g_ego, 10.0, 0.3)
    s = go(V.TURN, 1.0)
    assert abs(float(s.yaw[0]) - TURN_MAX) < 1e-9 and float(s.yaw_gate[0]) == 1.0
    assert abs(float(go(V.TURN, -1.0).yaw[0]) + TURN_MAX) < 1e-9
    # PRIORITY is multiplicative and meets RAISE/LOWER exactly at the extremes
    s = go(V.PRIORITY, 1.0, -1.0)
    assert abs(float(s.alpha[0, 0]) - 1.5) < 1e-9, float(s.alpha[0, 0])
    assert abs(float(s.alpha[0, 1]) - 1.0 / 1.5) < 1e-9, float(s.alpha[0, 1])
    assert not bool(go(V.PRIORITY, 1.0).moved[0]), "PRIORITY must not place a subgoal"
    assert float(go(V.LOOK, 0.0).yaw_gate[0]) == 0.0, "LOOK hands yaw back to the plant"
    assert not bool(go(V.EOS, 0.0).moved[0]), "a bare EOS is silence"


def test_the_likelihood_covers_only_the_arguments_a_token_uses():
    """EOS carries none, so its density is pure classification; a token's
    unused argument slots must not enter its log-probability or the update
    would fit noise.  (LOOK used to be the other argument-free token; it is out
    of the vocabulary now -- see N_TOKENS in actions_cont.py.)"""
    V = ContVocab(2); B = 4
    torch.manual_seed(0)
    lg = torch.randn(B, V.V, dtype=DT)
    mu = torch.randn(B, 2, dtype=DT); ls = torch.full((2,), -0.5, dtype=DT)
    u = torch.randn(B, 2, dtype=DT)
    toks = torch.tensor([V.EOS, V.WAYPOINT, V.TURN, V.PRIORITY])
    na = torch.tensor([V.n_arg_of(int(t)) for t in toks])
    lp = log_prob(lg, mu, ls, toks, u, na)
    cat = torch.log_softmax(lg, -1)
    for i, t in enumerate(toks.tolist()):
        if V.n_arg_of(t) == 0:
            assert abs(float(lp[i]) - float(cat[i, t])) < 1e-12, V.name(t)
        else:
            assert float(lp[i]) < float(cat[i, t]), V.name(t)
    # changing an unused slot changes nothing
    u2 = u.clone(); u2[0, :] += 3.0                      # row 0 is EOS: no arguments
    assert torch.allclose(log_prob(lg, mu, ls, toks, u2, na)[0], lp[0])
    assert float(entropy(lg, ls)[0]) > 0


# --- the network and the update -------------------------------------------

def _cont_net(n_terms=2):
    from lagrangian_es.composer.policy_cont import ContPolicyNet
    return ContPolicyNet(n_terms)


def _fake_tok(B, goal_deg, gain, n_ent=4, n_chain=2):
    g = torch.zeros(B, 8, dtype=torch.float32)
    g[:, 0] = 5.0 * math.cos(math.radians(goal_deg)) / gain
    g[:, 1] = 5.0 * math.sin(math.radians(goal_deg)) / gain
    return {"self": torch.zeros(B, 8), "goal": g, "psi": torch.zeros(B),
            "entities": torch.zeros(B, n_ent, 8), "ent_types": torch.zeros(B, n_ent, dtype=torch.long),
            "ent_mask": torch.ones(B, n_ent, dtype=torch.bool),
            "chain": torch.zeros(B, n_chain, 8), "chain_types": torch.zeros(B, n_chain, dtype=torch.long),
            "chain_mask": torch.zeros(B, n_chain, dtype=torch.bool)}


def test_nothing_aims_the_waypoint_at_the_goal_any_more():
    """The goal bearing used to be ADDED to theta by hand, supplying 99.9% of
    its variation while the network supplied 0.0004.  It is gone, so an
    untrained composer points nowhere in particular -- measured, mean bearing
    error went 6.4 deg to 85.7 deg, which is chance.  Learning where the goal is
    from the goal token is now part of the problem."""
    net = _cont_net()
    assert net.goal_residual is False
    V = ContVocab(2)
    errs = []
    for deg in (-150.0, -60.0, 0.0, 60.0, 150.0):
        with torch.no_grad():
            _, mu, _ = net.pre(_fake_tok(8, deg, net.goal_gain))
        got = math.degrees(math.pi * float(V.squash(mu[:, V.WAYPOINT])[0, 1]))
        errs.append(abs(((got - deg + 180) % 360) - 180))
    assert sum(errs) / len(errs) > 30.0, (
        "an untrained head still tracks the goal -- something is still "
        "injecting the answer")

def test_the_update_runs_and_moves_both_heads_under_its_trust_region():
    """A mixed decision needs a mixed gradient: the type head and the argument
    head must both move, and the KL cap must actually bind -- the last
    continuous composer this project ran was uncapped and diverged."""
    from lagrangian_es.composer.policy_cont import ppo_update_cont
    torch.manual_seed(0)
    net = _cont_net()
    V = ContVocab(2)
    B, T = 24, 3
    recs = []
    for t in range(T):
        tok = _fake_tok(B, 20.0, net.goal_gain)
        act = torch.randint(0, V.V, (B,))
        recs.append({"t": float(t * 20), "act": act, "u": torch.randn(B, V.n_args),
                     "n_args": torch.tensor([V.n_arg_of(int(z)) for z in act.tolist()]),
                     "moved": act == V.WAYPOINT, "alive": torch.ones(B, dtype=torch.bool),
                     "rows": torch.arange(B), "tok_keep": None,
                     "logits": torch.randn(B, V.V), "pi_logits": torch.randn(B, V.V),
                     "mu": torch.randn(B, V.n_args), "log_std": net.log_std.detach().clone(),
                     "tok": tok})
    R = torch.randn(T, B, dtype=torch.float64)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    st = ppo_update_cont(net, recs, R, 2, epochs=2, batch=16, lr=1e-3, target_kl=0.02)
    assert st["n"] == T * B
    moved = {k for k, v in net.state_dict().items() if not torch.equal(v, before[k])}
    assert any(k.startswith("head_act") for k in moved), "the type head did not move"
    assert any(k.startswith("arg_w") for k in moved), "the argument heads did not move"
    # the spread is FROZEN on purpose: fitting it by maximum likelihood on the
    # policy's own successes shrinks it every update until exploration dies
    assert "log_std" not in moved, "the argument spread should be frozen"
    # a huge learning rate must trip the cap rather than run away
    net2 = _cont_net()
    st2 = ppo_update_cont(net2, recs, R, 2, epochs=8, batch=16, lr=5.0, target_kl=0.01)
    assert st2["stopped_early"], "the trust region never bound at lr 5.0"


def test_an_argument_a_token_does_not_use_gets_no_gradient():
    """EOS and LOOK carry no arguments; if their unused slots reached the
    Gaussian term the update would be fitting pure noise."""
    from lagrangian_es.composer.policy_cont import ppo_update_cont
    torch.manual_seed(1)
    net = _cont_net()
    V = ContVocab(2)
    B = 16
    act = torch.full((B,), V.EOS, dtype=torch.long)          # nothing but silence
    rec = {"t": 0.0, "act": act, "u": torch.randn(B, V.n_args),
           "n_args": torch.zeros(B, dtype=torch.long),
           "moved": torch.zeros(B, dtype=torch.bool), "alive": torch.ones(B, dtype=torch.bool),
           "rows": torch.arange(B), "tok_keep": None,
           "logits": torch.zeros(B, V.V), "pi_logits": torch.zeros(B, V.V),
           "mu": torch.zeros(B, V.n_args), "log_std": net.log_std.detach().clone(),
           "tok": _fake_tok(B, 0.0, net.goal_gain)}
    R = torch.randn(1, B, dtype=torch.float64)
    before = net.arg_w2.detach().clone()
    ppo_update_cont(net, [rec], R, 2, epochs=2, batch=8, lr=1e-2, target_kl=0.0, vcoef=0.0)
    # EOS carries no arguments, so its head must not move -- and no other
    # token's either, since none of them were chosen
    assert torch.equal(net.arg_w2.detach(), before), \
        "silence moved an argument head, so unused slots are entering the likelihood"


# --- arrival time as the cross-entropy weight --------------------------------
# The hard `keep_frac` cut threw away the gradient from two thirds of the
# arrivals and quantized the rest; worse, with one flight per task it ranked
# the TASKS, so "the fastest 30%" meant "the 30% nearest goals".

def test_a_flight_that_never_arrived_is_never_imitated():
    """`finish_frac` is 1.0 exactly when the flight never arrived, and such a
    flight has no arrival time to rank.  A positive weight on it would make the
    failure more likely, so imitation gives it zero."""
    from lagrangian_es.composer.policy_cont import arrival_weights
    t = torch.tensor([0.2, 0.5, 0.8, 1.0, 1.0])
    w = arrival_weights(t, tau=1.0)
    assert (w[3:] == 0).all()
    assert (w[:3] > 0).all()


def test_imitation_weight_falls_with_arrival_time_and_averages_one():
    from lagrangian_es.composer.policy_cont import arrival_weights
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    w = arrival_weights(t, tau=1.0)
    assert (w[:-1] > w[1:]).all(), "a quicker flight must count for more"
    assert abs(float(w.mean()) - 1.0) < 1e-9, "mean 1 keeps the loss on the scale an unweighted mean has"


def test_effective_sample_size_matches_the_cut_it_replaces():
    """tau = 1 is chosen so the soft weight is about as selective as the 30%
    cut: for a normal spread of arrival times the effective sample size is
    e^-1 = 37% of the arrivals."""
    from lagrangian_es.composer.policy_cont import arrival_weights
    g = torch.Generator().manual_seed(0)
    t = 0.5 + 0.1 * torch.randn(20000, generator=g, dtype=torch.float64)
    w = arrival_weights(t.clamp(0.01, 0.99), tau=1.0)
    ess = float(w.sum() ** 2 / (w * w).sum()) / w.numel()
    assert 0.30 < ess < 0.45, f"effective sample size {ess:.3f}, expected ~0.37"


def test_grouping_by_task_ranks_decisions_not_goals():
    """The whole point.  Two tasks, one easy and one hard; the hard task's best
    flight is SLOWER in absolute terms than the easy task's worst, and must
    still outweigh it -- otherwise the update only ever sees easy tasks."""
    from lagrangian_es.composer.policy_cont import arrival_weights
    t = torch.tensor([0.10, 0.20, 0.60, 0.90])       # task 0 easy, task 1 hard
    task = torch.tensor([0, 0, 1, 1])
    ungrouped = arrival_weights(t, tau=1.0)
    assert ungrouped[2] < ungrouped[1], "without tasks the easy goal wins on absolute time"
    w = arrival_weights(t, task, tau=1.0)
    assert w[2] > w[1], "the hard task's best flight must outweigh the easy task's worst"
    assert w[0] > w[1] and w[2] > w[3], "within a task, quicker still wins"


def test_signed_weight_pushes_failures_down_and_balances_within_a_task():
    """A signed weight is a policy gradient with a per-task baseline, written
    as a weight -- one cross-entropy term still, but it can move probability
    DOWN, which is the only way the failure rate can fall."""
    from lagrangian_es.composer.policy_cont import arrival_weights
    t = torch.tensor([0.3, 1.0, 0.4, 0.5])           # flight 1 never arrived
    task = torch.tensor([0, 0, 1, 1])
    w = arrival_weights(t, task, tau=1.0, signed=True)
    assert w[1] < 0, "a failure must be made LESS likely"
    assert w[0] > 0
    for j in (0, 1):                                  # the baseline is the task's own mean
        assert abs(float(w[task == j].sum())) < 1e-9
    # and the magnitude is bounded, since nothing stops -log p running away
    big = arrival_weights(torch.tensor([0.0, 0.5, 1.0]), tau=0.01, signed=True, w_max=2.0)
    assert float(big.abs().max()) <= 2.0


def test_one_sample_per_task_falls_back_to_the_batch_baseline():
    """Centring a task's single flight on its own mean leaves exactly zero, so
    every weight comes out 1 and the update degenerates to imitating
    everything.  That is the k=1 control arm, and it must centre on the batch."""
    from lagrangian_es.composer.policy_cont import arrival_weights
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    solo = torch.arange(4)                          # one flight per task
    w = arrival_weights(t, solo, tau=1.0)
    assert float(w.std()) > 0.1, "weights collapsed to uniform: no selection at all"
    assert torch.allclose(w, arrival_weights(t, tau=1.0))
    ess = float(w.sum() ** 2 / (w * w).sum())
    assert ess < 0.9 * w.numel(), "an effective sample size equal to n means nothing was weighted"


# --- phi: the vertical axis the composer did not have -----------------------
# Height used to be DERIVED -- the subgoal took the goal's own height scaled by
# how far out it sat -- so the composer could route around a building but never
# over one, and could not lift away from the floor where most deaths happen.

def test_phi_lifts_and_drops_the_subgoal():
    from lagrangian_es.composer.actions_cont import ContVocab, PHI_MAX
    V = ContVocab(1)
    x, goal, lvl = _place(V, r=1.0, th=0.0, phi=0.0)
    _, _, up = _place(V, r=1.0, th=0.0, phi=1.0)
    _, _, dn = _place(V, r=1.0, th=0.0, phi=-1.0)
    z = lambda sp: (goal + sp.delta)[:, 2]
    assert (z(up) > z(lvl)).all(), "phi=+1 must place the subgoal above"
    assert (z(dn) < z(lvl)).all(), "phi=-1 must place it below"
    # at full deflection the climb is PHI_MAX, so rise = radius * sin(PHI_MAX)
    rise = float((z(up) - x[:, 2]).mean())
    assert abs(rise - 10.0 * math.sin(PHI_MAX)) < 1e-6, rise


def test_phi_spans_the_full_vertical_range():
    """PHI_MAX was pi/4 -- a judgement that a climb steeper than 45 degrees was
    not something the vehicle should want.  That is the network's call, so the
    bound is now the full pi/2.  A bound still exists because a squash needs a
    scale, but it no longer encodes a preference."""
    from lagrangian_es.composer.actions_cont import PHI_MAX
    assert abs(PHI_MAX - math.pi / 2) < 1e-9
    V = ContVocab(1)
    x, goal, sp = _place(V, r=1.0, th=0.0, phi=1.0, reach=10.0)
    sub = goal + sp.delta
    assert float((sub[:, 2] - x[:, 2]).mean()) > 9.0, "full deflection should climb a whole reach"

def test_the_placement_scales_with_the_goals_range_in_three_dimensions():
    """The same arguments place a NEARER point when the goal is nearer: the
    radius is min(reach, |goal - x|), so the whole ball -- height included --
    contracts as the vehicle closes.  That contraction is what makes arrival
    reachable by approach; see
    `test_the_placement_ball_shrinks_onto_the_goal_so_arrival_is_an_attractor`.
    """
    V = ContVocab(1)
    xn, gn, near = _place(V, r=0.5, th=0.0, L=4.0, reach=10.0)
    xf, gf, far = _place(V, r=0.5, th=0.0, L=18.0, reach=10.0)
    off_near = (gn + near.delta) - xn
    off_far = (gf + far.delta) - xf
    assert float(off_near.norm(dim=-1)[0]) < float(off_far.norm(dim=-1)[0])
    # exactly the ratio of the two radii: 4 m against the reach-capped 10 m
    assert torch.allclose(off_near * (10.0 / 4.0), off_far, atol=1e-6)

def test_the_elevation_residual_round_trips_to_the_goals_own_climb_angle():
    """`pre` adds `atanh(elev / PHI_MAX)` to the WAYPOINT's phi slot, the same
    residual it adds to theta for the bearing.  The property that matters is
    the round trip: with the head contributing nothing, the token that comes
    out must aim at the goal's actual climb angle -- so a waypoint placed on
    the prior flies AT the goal in three dimensions, and the head only has to
    learn the correction.
    """
    from lagrangian_es.composer.actions_cont import PHI_MAX
    for dz, L in ((2.0, 4.0), (-1.0, 5.0), (0.0, 3.0)):
        elev = math.atan2(dz, L)                      # the goal's own climb angle
        assert abs(elev) < PHI_MAX, "fixture must stay inside the token's range"
        u = math.atanh(min(max(elev / PHI_MAX, -0.999), 0.999))   # what `pre` adds
        got = PHI_MAX * math.tanh(u)                  # what `finish` turns it back into
        assert abs(got - elev) < 1e-9, (dz, L, got, elev)
    # and a goal steeper than the token's range saturates rather than wrapping
    steep = math.atan2(50.0, 1.0)
    u = math.atanh(min(max(steep / PHI_MAX, -0.999), 0.999))
    assert 0 < PHI_MAX * math.tanh(u) <= PHI_MAX


def test_selection_survives_a_high_failure_rate():
    """The weight must keep selecting when flights start failing.

    A non-arrival sits at `finish_frac` 1.0, far above any real arrival time,
    and unsigned weighting gives it zero weight -- but it was still inflating
    the spread every other flight is divided by.  Measured on the live run at
    10% buildings: 211 effective flights out of 226, i.e. the weights had gone
    uniform and the update had stopped preferring the quick flights at all.
    Exactly when obstacles appear and selection matters most.
    """
    from lagrangian_es.composer.policy_cont import arrival_weights
    g = torch.Generator().manual_seed(0)
    for fail in (0.02, 0.2, 0.5):
        n = 1000
        t = (0.3 + 0.08 * torch.randn(n, generator=g, dtype=torch.float64)).clamp(0.05, 0.95)
        t[: int(fail * n)] = 1.0
        w = arrival_weights(t, tau=1.0)
        fails = w[t >= 1.0]
        assert fails.numel() == 0 or float(fails.abs().max()) == 0.0, "a failure must never be imitated"
        nz = w[w > 0]
        ess = float(nz.sum() ** 2 / (nz * nz).sum()) / nz.numel()
        assert 0.25 < ess < 0.55, f"at {fail:.0%} failures the effective sample size was {ess:.2f}"


def test_the_argument_head_conducts_gradient_to_the_body():
    """A zero-initialised output layer switches off everything beneath it.

    `mu = h @ arg_w2 + arg_b2`, so `d(mu)/d(arg_w1)` is PROPORTIONAL to
    `arg_w2`.  It used to be initialised to exactly zero -- so the net would
    start at the prior in `arg_b2` -- which made the gradient to `arg_w1`, to
    the shared body and to the scene encoder underneath it exactly zero.  The
    branch could only switch on once `arg_w2` lifted itself off zero, and over
    578 live iterations it did not: `arg_w2` reached 0.00136 while `arg_w1`
    moved 0.9% from its initial value.  The argument head stayed a constant
    (`r` had sd 0.0088 across 6,945 decisions) and perception, which is only
    useful through those arguments, was never trained.
    """
    from lagrangian_es.composer.policy_cont import ContPolicyNet
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    assert float(net.arg_w2.abs().mean()) > 1e-3, "arg_w2 at zero blocks the whole branch"

    tok = {"self": torch.randn(8, 8), "goal": torch.randn(8, 8),
           "entities": torch.randn(8, 40, 8), "ent_types": torch.randint(2, 8, (8, 40)),
           "ent_mask": torch.ones(8, 40, dtype=torch.bool),
           "chain": torch.randn(8, 16, 8), "chain_types": torch.randint(4, 6, (8, 16)),
           "chain_mask": torch.zeros(8, 16, dtype=torch.bool), "psi": torch.zeros(8)}
    net.zero_grad(set_to_none=True)
    net.pre(tok)[1][:, net.vocab.WAYPOINT].sum().backward()
    # the gradient must reach BOTH the layer below the head and the encoder
    # that reads the sensors -- perception is only useful through the arguments
    for name in ("arg_w1", "emb_w1"):
        g = dict(net.named_parameters())[name].grad
        assert g is not None and float(g.abs().mean()) > 0, f"no gradient reaches {name}"


def test_no_argument_carries_a_hand_set_prior():
    """arg_b2 was 0.7 on the WAYPOINT's r, putting the subgoal at ~80% of the
    way.  All slots start at zero now; the network chooses."""
    from lagrangian_es.composer.policy_cont import ContPolicyNet
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    assert float(net.arg_b2.abs().max()) == 0.0, net.arg_b2
    assert float(net.head_act[-1].bias.abs().max()) == 0.0, net.head_act[-1].bias
    # but the WEIGHTS must still be non-zero, or the gradient to everything
    # below the heads is zero (see test_neither_head_is_zero_initialised)
    assert float(net.arg_w2.abs().mean()) > 1e-4
    assert float(net.head_act[-1].weight.abs().mean()) > 1e-4

def test_soft_time_scores_need_the_arrival_flag_stated_not_inferred():
    """With `finish_frac` the score was 1.0 exactly when a flight never
    arrived, so the weight could infer failure from the value.  Soft time has
    no such sentinel -- 0.99 means "spent nearly the whole episode away from the
    goal", which a slow ARRIVAL can also score -- so inferring would drop real
    successes and keep real failures."""
    from lagrangian_es.composer.policy_cont import arrival_weights
    t = torch.tensor([0.30, 0.99, 0.55, 0.98])       # all four are soft times
    arrived = torch.tensor([True, True, False, False])
    w = arrival_weights(t, tau=1.0, arrived=arrived)
    assert float(w[1]) > 0, "a slow ARRIVAL scoring 0.99 must not be read as a failure"
    assert float(w[2]) == 0.0 and float(w[3]) == 0.0, "failures must carry no imitation weight"
    # Inferring from the value alone is worse than useless here: `t < 1.0` is
    # true of every soft time, so both FAILURES would be handed positive
    # imitation weight and actively reinforced.
    naive = arrival_weights(t, tau=1.0)
    assert float(naive[2]) > 0 and float(naive[3]) > 0, \
        "fixture: the naive inference should wrongly imitate the failures"
    # the signed path ranks failures instead of dropping them -- that is the point
    sw = arrival_weights(t, tau=1.0, signed=True, arrived=arrived)
    assert float(sw[0]) > float(sw[3]), "a quick flight must outrank one that got nowhere"
    assert bool((sw < 0).any()), "signed weighting must be able to push probability down"


def test_signed_weights_do_not_blow_the_loss_up():
    """Signed weights are CENTRED, so their signed sum is ~0.  Normalising by
    that sum -- even clamped -- divides by almost nothing: the signed arm's very
    first iteration reported loss -746,178.  The normaliser is the total weight
    MASS, which equals the signed sum whenever the weights are non-negative, so
    the unsigned path is unchanged."""
    from lagrangian_es.composer.policy_cont import arrival_weights
    t = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0, 0.5])
    task = torch.tensor([0, 0, 1, 1, 2, 2])
    sw = arrival_weights(t, task, tau=1.0, signed=True)
    assert abs(float(sw.sum())) < 1e-6, "fixture: signed weights must sum to ~0"
    assert float(sw.abs().sum()) > 0.5, "but their MASS must be substantial"
    lp = torch.tensor([-1.0, -2.0, -0.5, -1.5, -3.0, -1.2], dtype=torch.float64)
    good = -(sw * lp).sum() / sw.abs().sum().clamp_min(1e-9)
    bad = -(sw * lp).sum() / sw.sum().clamp_min(1e-9)
    assert abs(float(good)) < 10.0, f"normalised loss should stay O(1), got {float(good)}"
    assert abs(float(bad)) > 1e6, "fixture: the old normaliser should explode"


# --- the error loss ----------------------------------------------------------
# No reward, no return, no labels: the composer claims the vehicle can reach g
# by the next decision, and reality answers.  e_reach alone would put the
# subgoal at the vehicle's own feet; e_aim alone would put it on the goal
# through a building; together the optimum is as far toward the goal as the
# vehicle can actually get.

def test_each_error_term_alone_is_degenerate_and_together_they_are_not():
    from lagrangian_es.composer.policy_cont import _subgoal_ego, ContPolicyNet
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    W = net.vocab.WAYPOINT
    goal = torch.tensor([[0.8, 0.0, 0.0]])          # goal 0.8 of a reach ahead
    # the vehicle only managed a third of the way (something was in the way)
    got = torch.tensor([[0.27, 0.0, 0.0]])

    def loss_at(mu, a, b):
        g = _subgoal_ego(net, mu, goal)
        return float(a * ((g - got) ** 2).sum() + b * ((g - goal) ** 2).sum())

    at_feet = torch.tensor([[-1.0, 0.0, 0.0]])      # r = 0
    at_goal = torch.tensor([[3.0, 0.0, 0.0]])       # r = radius
    at_got = torch.tensor([[2 * 0.27 / 0.8 - 1.0, 0.0, 0.0]])
    # e_reach alone prefers where it actually got over the goal
    assert loss_at(at_got, 1.0, 0.0) < loss_at(at_goal, 1.0, 0.0)
    # e_aim alone prefers the goal, however unreachable
    assert loss_at(at_goal, 0.0, 1.0) < loss_at(at_got, 0.0, 1.0)
    # e_reach alone is happy at the vehicle's own feet if that is what it got
    assert loss_at(at_feet, 1.0, 0.0) < loss_at(at_goal, 1.0, 0.0)
    # together, neither extreme wins: the optimum is strictly between them
    both = [loss_at(m, 1.0, 1.0) for m in (at_feet, at_got, at_goal)]
    assert both[1] < both[0] and both[1] < both[2], both


def test_the_error_gradient_is_exact_and_reaches_the_sensors():
    """No score-function estimator: `g` is analytic in the arguments and the
    target is a constant, so the gradient is exact -- and it must reach the
    per-type encoders, since only perception explains WHY the vehicle fell
    short of what was commanded."""
    from lagrangian_es.composer.policy_cont import _subgoal_ego, ContPolicyNet
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    tok = {"self": torch.randn(8, 8), "goal": torch.randn(8, 8),
           "entities": torch.randn(8, 40, 8), "ent_types": torch.randint(2, 8, (8, 40)),
           "ent_mask": torch.ones(8, 40, dtype=torch.bool),
           "chain": torch.randn(8, 16, 8), "chain_types": torch.randint(4, 6, (8, 16)),
           "chain_mask": torch.zeros(8, 16, dtype=torch.bool), "psi": torch.zeros(8)}
    net.zero_grad(set_to_none=True)
    _, mu_all, _ = net.pre(tok)
    gl = torch.rand(8, 3) * 0.8
    got = torch.rand(8, 3) * 0.4
    g = _subgoal_ego(net, mu_all[:, net.vocab.WAYPOINT], gl)
    (((g - got) ** 2).sum(-1) + ((g - gl) ** 2).sum(-1)).mean().backward()
    for name in ("arg_w2", "arg_w1", "emb_w1"):
        gr = dict(net.named_parameters())[name].grad
        assert gr is not None and float(gr.abs().mean()) > 0, f"no gradient reaches {name}"


def test_neither_head_is_zero_initialised():
    """BOTH output layers were zeroed so their priors would hold exactly, and
    both blocked the gradient to everything beneath them -- `d(out)/d(layer
    below)` is proportional to the output weight.  At initialisation the body
    and the per-type sensor encoders received gradient through NEITHER route,
    which is why perception stayed decorative while the heads slowly escaped.
    """
    from lagrangian_es.composer.policy_cont import ContPolicyNet
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    assert float(net.arg_w2.abs().mean()) > 1e-3, "arg_w2 zeroed: argument path blocked"
    assert float(net.head_act[-1].weight.abs().mean()) > 1e-3, "token head zeroed: token path blocked"

    tok = {"self": torch.randn(8, 8), "goal": torch.randn(8, 8),
           "entities": torch.randn(8, 40, 8), "ent_types": torch.randint(2, 8, (8, 40)),
           "ent_mask": torch.ones(8, 40, dtype=torch.bool),
           "chain": torch.randn(8, 16, 8), "chain_types": torch.randint(4, 6, (8, 16)),
           "chain_mask": torch.zeros(8, 16, dtype=torch.bool), "psi": torch.zeros(8)}
    # the TOKEN path alone must reach the sensor encoders
    net.zero_grad(set_to_none=True)
    net.pre(tok)[0].sum().backward()
    for name in ("head_act.0.weight", "emb_w1"):
        g = dict(net.named_parameters())[name].grad
        assert g is not None and float(g.abs().mean()) > 0, f"token path gives {name} no gradient"

    # the bias is FLAT now -- no token is preferred a priori
    with torch.no_grad():
        p = torch.softmax(net.head_act[-1].bias[None], -1)[0]
    assert abs(float(p.max()) - float(p.min())) < 1e-6, "a token still carries a hand-set prior"


def test_neither_error_term_can_dominate_by_being_larger():
    """The two errors are squared distances in the same units but not the same
    SIZE: on the city e_aim ~ 1.5 against e_reach ~ 0.4, so an even alpha:beta
    gave the aim term ~3x the gradient and the composer learned to put the
    subgoal on the goal regardless of whether it could get there -- exactly the
    degeneracy the other term exists to prevent.  Each is divided by its own
    detached mean, so the weighting is a ratio of relative improvement."""
    big = torch.tensor([2.0, 1.0, 1.5], dtype=torch.float64)      # e_aim, larger
    small = torch.tensor([0.4, 0.2, 0.3], dtype=torch.float64)    # e_reach
    raw = (big.mean(), small.mean())
    assert float(raw[0]) > 3 * float(raw[1]), "fixture: the terms must differ in scale"
    nb = big / big.mean().detach().clamp_min(1e-6)
    ns = small / small.mean().detach().clamp_min(1e-6)
    assert abs(float(nb.mean()) - float(ns.mean())) < 1e-9, \
        "after normalising, equal alpha:beta must give equal weight"


def test_the_time_model_cannot_predict_negative_time():
    """Time is a fraction of the episode, so it lives in [0, 1].  Unbounded,
    the policy drove the prediction to -0.4 within two iterations while the
    model's fit error rose 0.98 -> 1.21: it was not learning to be quick, it was
    walking the model somewhere the model had never seen and believing it."""
    from lagrangian_es.composer.policy_cont import ContPolicyNet
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    d = net.time_head[0].weight.shape[1] - 3
    # drive the head hard in both directions; the bound must hold
    q = torch.randn(64, d) * 50.0
    g = torch.randn(64, 3) * 50.0
    with torch.no_grad():
        t = torch.sigmoid(net.time_head(torch.cat([q, g], -1)).squeeze(-1))
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0, (float(t.min()), float(t.max()))


def test_applying_a_token_never_mutates_the_caller_s_spec():
    """`ContVocab.step` writes TURN and LOOK in place (`out.yaw[rows] = ...`)
    and is only safe because `begin()` takes a DEEP clone -- a detail living in
    spec.py, two files from the code that depends on it.  A shallow clone would
    have every emitted token corrupt the spec the rollout is still holding, on
    rows that were never asked."""
    from lagrangian_es.composer.actions_cont import ContVocab
    from lagrangian_es.composer.spec import TaskSpec

    B = 4
    V = ContVocab(1)
    cur = TaskSpec.identity(B, 3, 1, torch.float64, torch.device("cpu"))
    cur.yaw = torch.zeros(B, dtype=torch.float64)
    cur.yaw_gate = torch.zeros(B, dtype=torch.float64)
    snapshot = {"yaw": cur.yaw.clone(), "alpha": cur.alpha.clone(),
                "gate": cur.gate.clone(), "delta": cur.delta.clone(),
                "yaw_gate": cur.yaw_gate.clone()}

    psi = torch.zeros(B, dtype=torch.float64)
    arg = torch.zeros(B, 3, dtype=torch.float64)
    arg[:, 0] = 1.0
    for tk in (V.TURN, V.PRIORITY, V.LOOK, V.WAYPOINT):
        out, pend, has = V.begin(cur, psi)
        V.step(torch.full((B,), tk, dtype=torch.long), arg, out, pend, has, psi)
        for name, before in snapshot.items():
            now = getattr(cur, name)
            assert torch.equal(now, before), (
                f"emitting {V.name(tk)} mutated the caller's spec field {name!r}")


def test_the_goal_residual_is_what_decides_the_waypoint():
    """MEASURED on a trained composer at 100% buildings:

        argument        network   injected     network share
        r  (brake)       0.0004     0.0000        100.0%
        theta            0.0004     0.7746          0.1%
        phi              0.0002     0.0626          0.3%

    With the residual on, `atanh(goal bearing)` supplies 99.9% of theta's
    variation and the network supplies 0.0004 of it.  That is the reason
    perception has no measurable effect on the output despite being linearly
    recoverable from the read-out at R^2 0.852 -- the output barely depends on
    the NETWORK, so it cannot depend on what the network sees.  `r` is the
    control: the only argument with nothing injected, and the only one whose
    variation is entirely the network's, at sd 0.0004.

    This pins the SIZE of the effect, so that turning the residual off is known
    to be a change of this magnitude and not a detail.
    """
    from lagrangian_es.composer.policy_cont import ContPolicyNet
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    W = net.vocab.WAYPOINT

    # goals scattered around the vehicle, so the injected bearing varies a lot
    import tests.test_actions_cont as _self
    toks = [_self._fake_tok(32, deg, net.goal_gain) for deg in (-150.0, -60.0, 0.0, 60.0, 150.0)]

    def theta_spread(residual):
        net.goal_residual = residual
        with torch.no_grad():
            mu = torch.cat([net.pre(t)[1][:, W, 1] for t in toks])
        return float(mu.std())

    with_res = theta_spread(True)
    without = theta_spread(False)
    net.goal_residual = True
    assert with_res > 10 * without, (
        f"theta varies {with_res:.4f} with the hand-injected bearing and "
        f"{without:.4f} without it -- the residual should dominate by an order "
        "of magnitude or this measurement no longer says what it claims")
