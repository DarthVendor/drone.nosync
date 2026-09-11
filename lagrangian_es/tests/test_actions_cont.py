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


def test_the_subgoal_collapses_onto_the_goal_so_arrival_stays_possible():
    """The failure this replaces: a Cartesian offset held in world coordinates
    sat a fixed distance from a goal with a 0.25 m tolerance, so the vehicle
    could never satisfy both and never arrived."""
    V = ContVocab(2)
    prev = float("inf")
    for L in (10.0, 4.0, 1.0, 0.25, 0.05):
        _, _, sp = _place(V, 1.0, 0.5, L=L)          # worst case: full radius, square to the goal
        off = float(sp.delta[0].norm())
        assert off <= prev + 1e-9, "the offset must not grow as the goal nears"
        prev = off
    assert prev < 0.25, f"at 0.05 m to go the subgoal is still {prev:.3f} m out"


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
    """EOS and LOOK carry none, so their density is pure classification; a
    token's unused argument slots must not enter its log-probability or the
    update would fit noise."""
    V = ContVocab(2); B = 5
    torch.manual_seed(0)
    lg = torch.randn(B, V.V, dtype=DT)
    mu = torch.randn(B, 2, dtype=DT); ls = torch.full((2,), -0.5, dtype=DT)
    u = torch.randn(B, 2, dtype=DT)
    toks = torch.tensor([V.EOS, V.WAYPOINT, V.TURN, V.PRIORITY, V.LOOK])
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


def test_the_prior_aims_at_the_goal_so_silence_and_a_waypoint_agree():
    """The identity has to be the straight line.  The bearing mean is a
    residual about the goal, so an untrained head places the subgoal ON the
    goal -- which is what the bare controller already flies.  Without this the
    prior fired waypoints off the nose and nothing arrived."""
    net = _cont_net()
    V = ContVocab(2)
    for deg in (0.0, 30.0, -120.0, 179.0):
        with torch.no_grad():
            _, mu, _ = net.pre(_fake_tok(3, deg, net.goal_gain))
        # the head emits a set of arguments PER TOKEN TYPE; the waypoint's are
        # the ones that must aim at the goal, and a turn's must not inherit them
        assert mu.shape[1:] == (V.V, V.n_args), mu.shape
        a = V.squash(mu[:, V.WAYPOINT])
        got = math.degrees(math.pi * float(a[0, 1]))
        # APPROXIMATE, not exact.  It used to be within 1 degree because
        # `arg_w2` was initialised to exactly zero, which made the head a pure
        # constant -- and that is precisely what switched off the gradient to
        # `arg_w1`, the body and the scene encoder (see
        # test_the_argument_head_conducts_gradient_to_the_body).  An exact
        # prior cost the whole perception pathway its learning signal, so the
        # prior is now approximate by design.  What must still hold is that it
        # TRACKS the goal rather than firing off the nose: the failure this
        # guards against is waypoints at uniformly random bearings.
        assert abs(((got - deg + 180) % 360) - 180) < 15.0, (deg, got)
        # It must place MOST of the way -- close enough that silence and a
        # waypoint roughly agree -- but NOT at the saturated end.  A prior of
        # tanh(2.0) = 0.96 read as the perfect default and was unlearnable: the
        # slope there is 0.07, the sampled range of r collapsed to 0.017, and
        # the braking end was 17 standard deviations away and never sampled in
        # 512 flights.  A prior that cannot be moved is worse than one slightly
        # off, so this pins RESPONSIVENESS, not closeness to 1.
        r0 = float(a[0, 0])
        assert 0.4 < r0 < 0.9, f"the prior r should be well inside the range, got {r0:.3f}"
        slope = 1.0 - float(mu[0, V.WAYPOINT, 0]) ** 2 / (1 + float(mu[0, V.WAYPOINT, 0]) ** 2) * 0 - r0 ** 2
        assert slope > 0.3, f"the prior sits in the flat region of tanh (slope {slope:.3f})"
    # a turn must not inherit the waypoint's prior: no default turn, no default
    # priority move, no goal bearing leaking into either
    with torch.no_grad():
        _, mu, _ = net.pre(_fake_tok(3, 40.0, net.goal_gain))
    for t in (V.EOS, V.TURN, V.PRIORITY, V.LOOK):
        # Not exactly zero any more: `arg_w2` must be non-zero at init or the
        # gradient to arg_w1, the body and the scene encoder is zero too (see
        # test_the_argument_head_conducts_gradient_to_the_body).  What this
        # assertion is really for is the old SHARED head, where the waypoint's
        # r bias commanded a 54-degree default turn and the goal-bearing
        # residual landed on a priority weight.  Per-token heads make that
        # structurally impossible; the requirement now is only that no other
        # token carries a MEANINGFUL default.
        assert float(mu[0, t].abs().max()) < 0.1, f"{V.name(t)} inherited a prior"
    # and silence is still the overwhelming default
    with torch.no_grad():
        lg, _, _ = net.pre(_fake_tok(3, 0.0, net.goal_gain))
    p = torch.softmax(lg, -1)[0]
    assert float(p[V.EOS]) > 0.8, float(p[V.EOS])


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


def test_phi_is_bounded_so_one_token_cannot_command_straight_up():
    """The argument is squashed into [-PHI_MAX, PHI_MAX].  Unbounded, the
    extreme puts the subgoal directly overhead with no horizontal progress."""
    from lagrangian_es.composer.actions_cont import ContVocab, PHI_MAX
    V = ContVocab(1)
    assert PHI_MAX <= math.pi / 2
    x, goal, sp = _place(V, r=1.0, th=0.0, phi=1.0)
    sub = goal + sp.delta
    horiz = (sub[:, :2] - x[:, :2]).norm(dim=-1)
    assert (horiz > 0).all(), "full climb must still make horizontal progress"


def test_the_subgoal_still_collapses_onto_the_goal_in_three_dimensions():
    """The arrival degeneracy, now in 3-D.  The radius is the 3-D distance to
    go, so r at its maximum aimed at the goal lands exactly ON it -- measuring
    the radius horizontally (as before) left the subgoal at the goal's range
    but the vehicle's height whenever the goal was above or below."""
    from lagrangian_es.composer.actions_cont import ContVocab, PHI_MAX
    V = ContVocab(1)
    L, dz, reach = 6.0, 3.0, 10.0
    elev = math.atan2(dz, L)
    assert elev < PHI_MAX, "fixture only meaningful when the goal is reachable in one token"
    x, goal, sp = _place(V, r=1.0, th=0.0, L=L, dz=dz, reach=reach, phi=elev / PHI_MAX)
    sub = goal + sp.delta
    assert torch.allclose(sub, goal, atol=1e-6), (sub[0].tolist(), goal[0].tolist())


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


def test_the_prior_survives_the_non_zero_init():
    """The reason arg_w2 was zeroed was to start at the hand-built prior, and
    that intent still has to hold: r near 0.60 of the radius, not scattered."""
    from lagrangian_es.composer.policy_cont import ContPolicyNet
    torch.manual_seed(0)
    net = ContPolicyNet(n_terms=1)
    W = net.vocab.WAYPOINT
    assert abs(float(net.arg_b2[W, 0]) - 0.7) < 1e-6, "the r prior itself must be untouched"
    tok = {"self": torch.randn(64, 8), "goal": torch.randn(64, 8),
           "entities": torch.randn(64, 40, 8), "ent_types": torch.randint(2, 8, (64, 40)),
           "ent_mask": torch.ones(64, 40, dtype=torch.bool),
           "chain": torch.randn(64, 16, 8), "chain_types": torch.randint(4, 6, (64, 16)),
           "chain_mask": torch.zeros(64, 16, dtype=torch.bool), "psi": torch.zeros(64)}
    with torch.no_grad():
        r_u = net.pre(tok)[1][:, W, 0]
    assert abs(float(r_u.mean()) - 0.7) < 0.25, f"the r prior drifted to {float(r_u.mean()):.3f}"
