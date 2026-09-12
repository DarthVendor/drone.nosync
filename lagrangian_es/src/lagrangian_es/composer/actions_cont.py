"""Tokens that carry continuous arguments: `[WAYPOINT(r, theta)] [EOS]`.

The same grammar as `actions.py` -- a chain of components closed by EOS, silence
is a bare EOS -- but the arguments are real numbers instead of a grid.  Five
token TYPES replace twenty-five components:

    EOS                     close the chain; alone, it is silence
    WAYPOINT(r, theta, phi) put the subgoal r of the way out, at bearing theta
                            and elevation phi
    TURN(theta)             command a yaw change
    PRIORITY(w0..w_{n-1})   push the constraint terms up or down
    LOOK                    point the camera

Why this and not twenty-five discrete components: the grid was coarse where it
mattered (30 degrees of bearing is a metre and a half across a four-metre
street) and it split the gradient twenty-five ways, so the rare components --
turn, priority -- never gathered enough evidence to improve and stayed rare.
One continuous head gets every sample instead.

POLAR, in the vehicle's OWN frame, and both halves of that matter.

`theta` is measured from where the drone is facing, which is the frame its
beams and camera patches already report their bearings in -- so "the beam at
+30 degrees is clear" and "put the waypoint at +30 degrees" are the same
number, and the composer never has to convert between frames to act on what it
sees.  `r` is the distance, which is also the only brake it has: a near
waypoint is how the task level tells a controller that flies at whatever the
pull commands to slow down.

**`r` is a FRACTION of the distance still to go, and that is not cosmetic.**
An earlier design held a Cartesian offset in world coordinates and arrival
became impossible: the subgoal sat a fixed distance from a goal with a 0.25 m
tolerance, so the vehicle could never be at both.  Here the radius is
`min(reach, distance to the goal)`, so every reachable subgoal collapses onto
the goal on approach.

The policy is Gaussian over the UNSQUASHED argument `u`, and the squash
(tanh) is applied here when the action is built.  That keeps the log-density
exact without a change-of-variables term, which is what the update needs.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from .actions import to_world
from .spec import TaskSpec

#: token types
EOS, WAYPOINT, TURN, PRIORITY, LOOK = 0, 1, 2, 3, 4
#: LOOK IS OUT OF THE VOCABULARY.  `V` stops at PRIORITY, so index 4 can never
#: be sampled or emitted; `step` still knows what a LOOK means, so turning it
#: back on is a one-line change to this number.
#:
#: Why: LOOK only sets `yaw_gate = 0`, handing yaw back to the plant's own
#: look-at -- which is what the plant does anyway unless a TURN took it.  It is
#: therefore a token that costs nothing and does nothing, and the policy found
#: it: measured on the live checkpoint, LOOK 78.4% / EOS 20.0% / WAYPOINT 1.6%
#: / TURN 0% / PRIORITY 0%, arriving 0.490 against a silence baseline of 0.521.
#: The composer had converged on saying something harmless rather than staying
#: quiet, and a no-op in the vocabulary is a free place for the update's noise
#: to accumulate.  An earlier run collapsed onto TURN 100% the same way, so the
#: pattern is "find the cheapest token", not anything specific to LOOK.
N_TOKENS = 2
#: TURN AND PRIORITY ARE OUT TOO, for the same reason LOOK was: at an argument
#: near zero each is an EXACT no-op, and a no-op is an absorbing state under
#: paired credit.
#:
#:   TURN commands `psi_at_decision + TURN_MAX * tanh(u)`.  With the argument
#:   head untrained tanh(u) ~ 0, so it commands the heading the vehicle already
#:   had: `dcmd ~ 0`, `des ~ cur`, nothing moves, and the Lagrangian's own yaw
#:   torque carries on underneath regardless.
#:   PRIORITY scales alpha by `1.5 ** arg`, which at arg 0 is exactly 1.0.
#:
#: A token that changes nothing produces a flight byte-identical to its muted
#: control, so its advantage is EXACTLY zero -- and zero-advantage samples are
#: dropped from the batch rather than penalised, which makes the no-op
#: invisible to the update and impossible to push back down.  MEASURED: the run
#: went subgoals 0.1 -> 0.0 by iteration 5, `heading 100%` at the judge, then
#: `signal 0 / 0 tokens / nan` from iteration 13 with the weights frozen --
#: judges 10 and 20 identical to the decimal.  An earlier run collapsed to
#: `heading 100%` the same way.
#:
#: What is left is the honest choice: say nothing, or place a waypoint.  EOS is
#: a no-op too, but it is the CONTROL -- it scores zero because it is the thing
#: everything else is measured against, which is the one case where zero is the
#: right answer rather than an escape hatch.

#: how many continuous arguments each type carries (PRIORITY is n_terms, set per instance)
BASE_ARGS = {EOS: 0, WAYPOINT: 3, TURN: 1, LOOK: 0}      # WAYPOINT is (r, theta, phi)

#: the widest argument vector any token uses; the head always emits this many
#: and the unused tail is ignored, so one Gaussian covers every token type
MAX_ARGS = 3

#: The action BOUNDS are the full range now, not a tuned slice of it.  A
#: squash needs some scale, so a bound cannot be removed -- but choosing pi/2
#: for a turn and pi/4 for a climb was a judgement about what the vehicle
#: should want to do, and that is the network's to make.
TURN_MAX = math.pi
#: the full vertical range; the network decides what it wants of it
PHI_MAX = math.pi / 2
PRIORITY_MAX = 1.0              # how far one PRIORITY token can push a term


class ContVocab:
    """Token types with continuous arguments."""

    EOS, WAYPOINT, TURN, PRIORITY, LOOK = EOS, WAYPOINT, TURN, PRIORITY, LOOK
    HOLD = EOS                   # the chain memory calls silence HOLD; here it is a bare EOS
    straight = WAYPOINT          # the rollout's opening action (see ContComposer._apply)
    V = N_TOKENS
    L_MAX = 4                    # components in a chain before EOS is forced

    def __init__(self, n_terms: int = 1):
        self.n_terms = int(n_terms)
        self.n_args = max(MAX_ARGS, self.n_terms)

    def name(self, t: int) -> str:
        return {EOS: "EOS", WAYPOINT: "WAYPOINT(r,theta,phi)", TURN: "TURN(theta)",
                PRIORITY: "PRIORITY(w)", LOOK: "LOOK"}[int(t)]

    def n_arg_of(self, t: int) -> int:
        return self.n_terms if int(t) == PRIORITY else BASE_ARGS[int(t)]

    # --- the squash: unbounded policy output -> a bounded action -------------
    @staticmethod
    def squash(u: Tensor) -> Tensor:
        """[-1, 1]^k.  The policy is Gaussian over `u`; this is the action."""
        return torch.tanh(u)

    # --- building the spec ---------------------------------------------------
    def begin(self, cur: TaskSpec, psi: Tensor) -> Tuple[TaskSpec, Tensor, Tensor]:
        """A working copy plus the pending-waypoint slots."""
        out = cur.clone() if hasattr(cur, "clone") else cur
        B = psi.shape[0]
        pend = torch.zeros(B, 3, dtype=psi.dtype, device=psi.device)     # (r, theta, phi), squashed
        has = torch.zeros(B, dtype=torch.bool, device=psi.device)
        return out, pend, has

    def step(self, tok: Tensor, arg: Tensor, out: TaskSpec, pend: Tensor, has: Tensor,
             psi: Tensor, rows: Optional[Tensor] = None) -> None:
        """One component, in place.

        `tok` [n] token type and `arg` [n, n_args] (already squashed to
        [-1, 1]) describe the rows named by `rows`, while `out`, `pend` and
        `has` are FULL width -- the same split the discrete `step` uses, so a
        decision that only some rows are still making writes through to the
        batch instead of into a copy.

        WAYPOINT is held until EOS (so a chain that also turns still places
        once); TURN, PRIORITY and LOOK apply immediately.
        """
        a = arg.to(psi.dtype)
        idx = rows if rows is not None else torch.arange(tok.shape[0], device=tok.device)
        psi_r = psi if psi.shape[0] == tok.shape[0] else psi[idx]
        w = tok == WAYPOINT
        if bool(w.any()):
            pend[idx[w]] = a[w, :3]
            has[idx[w]] = True
        t = tok == TURN
        if bool(t.any()) and out.yaw is not None:
            # absolute, like the discrete TURN: the current heading plus the
            # commanded offset, and the gate opened so the plant obeys it
            out.yaw[idx[t]] = psi_r[t] + TURN_MAX * a[t, 0]
            if out.yaw_gate is not None:
                out.yaw_gate[idx[t]] = 1.0
        p = tok == PRIORITY
        if bool(p.any()):
            # MULTIPLICATIVE, matching the discrete RAISE/LOWER exactly at the
            # extremes: an argument of +1 is one RAISE (x1.5), -1 one LOWER
            scale = torch.exp(math.log(1.5) * a[p][:, : self.n_terms])
            out.alpha = out.alpha.clone()
            out.alpha[idx[p]] = (out.alpha[idx[p]] * scale).clamp(0.05, 20.0)
        lk = tok == LOOK
        if bool(lk.any()) and out.yaw_gate is not None:
            out.yaw_gate[idx[lk]] = 0.0

    def finish(self, out: TaskSpec, pend: Tensor, has: Tensor, x: Tensor, goal: Tensor,
               psi: Tensor, g_ego: Tensor, reach: float, z_min: float) -> TaskSpec:
        """EOS for every row: realise the pending waypoint.

        The ball the argument indexes has radius `min(reach, |goal - x|)`, so a
        waypoint is always reachable AND collapses onto the goal as the vehicle
        arrives -- the property whose absence made arrival impossible when the
        offset was held in world coordinates.
        """
        dt = x.dtype
        if bool(has.any()):
            # `g_ego` is in units of the REACH, not metres -- the same convention
            # the discrete `finish` works in -- so everything here is in reach
            # units and the conversion happens once, at the end.  Mixing the two
            # put every waypoint a couple of metres from the vehicle instead of
            # out at its reach, and it crawled instead of flying.
            #
            # SPHERICAL, in the vehicle's frame: `theta` swings the subgoal
            # around the nose and `phi` tips it above or below the horizon.
            # Height used to be derived -- the subgoal simply took the goal's
            # own height scaled by how far out it sat -- which meant the
            # composer could not climb or descend at all.  It could route
            # around a building but never over one, and it could not lift away
            # from the floor, where most deaths happen.  `phi` is that missing
            # axis; nothing else commands altitude.
            # min(reach, |goal - x|).  This was briefly a FIXED reach ball, on
            # the argument that a goal-relative radius computes part of the
            # placement rather than letting the network choose it.  That
            # argument is wrong, and the run that tested it arrived 0.000 on an
            # EMPTY map for six iterations while the frozen low level alone
            # flies the same rung at 0.96-0.99.
            #
            # The radius is the action FRAME, not the answer: the network still
            # has to learn theta and phi, and nothing here tells it which way
            # the goal lies (that was `goal_residual`, which added
            # atanh(bearing) straight into mu, and is gone for good).  What the
            # frame decides is whether arrival is an ATTRACTOR.  Measured, the
            # chance a random placement lands inside the 0.25 m tolerance, as
            # the vehicle closes from 8 m to 0.5 m:
            #
            #     |goal-x|      fixed ball     goal-relative
            #          8.0        6.0e-06          1.5e-06
            #          1.0        3.4e-04          1.8e-03
            #          0.5        1.5e-03          1.7e-02
            #
            # The goal-relative ball SHRINKS onto the goal, so closing in makes
            # arriving easier and easier -- below the tolerance every placement
            # in the ball arrives.  The fixed ball does not: half a metre out
            # the vehicle is still being flung up to a full reach away, so the
            # error never settles and arrival is not reachable by approach.
            # That matters here beyond the flight itself, because the update is
            # cross-entropy on the composer's OWN successes: at zero arrivals
            # the filter admits nothing, the loss is nan, and the run cannot
            # bootstrap at all.  "0 tokens from 0/288", six iterations running.
            radius = g_ego.norm(dim=-1).to(dt).clamp(max=1.0)
            # THE ORIGIN MUST BE EXACTLY THE DO-NOTHING ACTION.
            #
            # `(a0+1)/2 * radius` puts mu = 0 at HALF the goal distance, so an
            # untrained router places a subgoal halfway to the goal -- not the
            # identity.  Silence is `delta = 0`, the subgoal BEING the goal, and
            # a router with EOS masked cannot decline to place: it is forced
            # into a slightly harmful move and can only choose which one.
            # MEASURED: on the judge the fresh router opens at 0.242 against
            # silence's 0.2692 +- 0.0045, and 30 uninterrupted iterations with
            # every diagnostic healthy (EV +0.37, kl 0.006-0.009 inside the cap,
            # clipfrac 0.17-0.34, speak flat) never recovered it: 0.199, 0.156,
            # 0.191.  A sound estimator cannot climb out of a floor set by the
            # action frame.
            #
            # `1 + min(0, tanh(a0))` is 1 at the origin (subgoal ON the goal =
            # silence), falls to 0 as a0 -> -inf, and is flat above -- r cannot
            # usefully exceed the goal anyway, so the flat side costs nothing.
            # `pend` is ALREADY squashed (the live path calls `V.squash(u)`
            # before `step`), so no tanh here -- `_subgoal_ego` squashes
            # internally instead.  Applying it twice made the rollout geometry
            # and the loss's mirror disagree; there is a test.
            r = (1.0 + torch.clamp(pend[:, 0].to(dt), max=0.0)) * radius
            # THE BEARING IS MEASURED FROM THE GOAL, not from the nose.
            #
            # From the nose, the full +-180 degrees is compressed into
            # a1 in [-1,1], so the network must reproduce atan2(g_y, g_x) to a
            # few degrees at EVERY decision or the flight fails -- and the error
            # is amplified by pi.  MEASURED: a student distilled onto a teacher
            # that arrives 0.576 reached MSE 0.0349 in mu (RMS 0.187, about 33
            # degrees of bearing) and still arrived 0.000, ruining 213 of 213
            # flights.  Nearly all the network's capacity went into the goal
            # geometry, leaving any beam-driven routing a rounding correction on
            # top of a large learned quantity -- which is why `sens` was ~0.
            #
            # Goal-relative, a1 = 0 IS straight at the goal: the identity sits
            # at the INTERIOR ORIGIN (no boundary optimum, so no |mu| runaway --
            # it reached 21.7 and tanh'(21.7) ~ 1e-16), and the network's whole
            # output becomes the DEVIATION, which is the routing decision
            # itself, with the beams at full leverage.
            #
            # This is NOT `goal_residual`, which added atanh(bearing) into mu
            # PRE-squash and was removed for supplying "99.9% of theta's
            # variation".  Variance share is not importance: theta mostly DOES
            # point at the goal, and obstacle deviations are rare and small in
            # variance while being decisive in effect.  After that removal
            # perception was still irrelevant (sens ~ 0) and the base competence
            # was gone too.  Here the correction is in ANGLE space and spans the
            # whole circle, so the learned part is not a rounding error, and
            # `sens` measures it in reach units rather than as a variance share.
            _gn = g_ego.norm(dim=-1).clamp_min(1e-9).to(dt)
            _gb = torch.atan2(g_ego[:, 1].to(dt), g_ego[:, 0].to(dt))
            _gp = torch.asin((g_ego[:, 2].to(dt) / _gn).clamp(-1.0, 1.0))
            theta = _gb + math.pi * pend[:, 1].to(dt)                  # 0 -> straight at the goal
            phi = (_gp + PHI_MAX * pend[:, 2].to(dt)).clamp(-PHI_MAX, PHI_MAX)
            # The radius is the 3-D distance now, so the arrival degeneracy
            # holds in three dimensions: r at its maximum, aimed at the goal,
            # puts the subgoal exactly ON the goal.  Measuring it horizontally
            # (as before) left a subgoal at the goal's range but the vehicle's
            # height whenever the goal was above or below.
            cphi = torch.cos(phi)
            sub_ego = torch.stack([r * cphi * torch.cos(theta),
                                   r * cphi * torch.sin(theta),
                                   r * torch.sin(phi)], -1)
            sub_world = x + to_world(sub_ego * reach, psi)             # reach units -> metres
            sub_world = torch.cat([sub_world[:, :2], sub_world[:, 2:].clamp_min(z_min)], -1)
            out.delta = torch.where(has[:, None], sub_world - goal, out.delta)
        out.moved = has
        return out

    def apply(self, tok: Tensor, arg: Tensor, cur: TaskSpec, x: Tensor, goal: Tensor, psi: Tensor,
              g_ego: Tensor, reach: float, z_min: float) -> TaskSpec:
        """One component then EOS -- the opening placement, and the tests."""
        out, pend, has = self.begin(cur, psi.to(x.dtype))
        self.step(tok, arg, out, pend, has, psi.to(x.dtype), rows=None)
        return self.finish(out, pend, has, x, goal, psi.to(x.dtype), g_ego, reach, z_min)


def log_prob(tok_logits: Tensor, mu: Tensor, log_std: Tensor, tok: Tensor, u: Tensor,
             n_args: Tensor, type_is_action: bool = True) -> Tensor:
    """Log-density of `[type, arguments]`, [B].

    Categorical over the type, plus a diagonal Gaussian over the UNSQUASHED
    arguments that type actually uses -- a token with no arguments contributes
    only its type term, so EOS and LOOK are pure classification.
    """
    # `type_is_action=False` drops the categorical term.  Under `route_only`
    # EOS is masked at rollout and every act is WAYPOINT, so the type is a
    # CONSTANT, not a decision -- and a constant in the likelihood ratio still
    # inflates the KL, eating the trust-region budget the real action (the
    # placement) needs, and still carries gradient into the type head.
    # MEASURED: `speak` drifting 0.401 -> 0.204 toward an action that is masked
    # and cannot be taken, while clipfrac sat at 0.43-0.61 with kl pinned at the
    # 0.02 cap -- the budget spent on a phantom choice.
    lp = (torch.log_softmax(tok_logits, -1).gather(-1, tok[:, None]).squeeze(-1)
          if type_is_action else torch.zeros(mu.shape[0], dtype=mu.dtype, device=mu.device))
    k = mu.shape[-1]
    idx = torch.arange(k, device=mu.device)[None, :]
    used = idx < n_args[:, None]                                   # [B, k]
    var = (2.0 * log_std).exp()
    g = -0.5 * (((u - mu) ** 2) / var + 2.0 * log_std + math.log(2 * math.pi))
    return lp + (g * used.to(g.dtype)).sum(-1)


def entropy(tok_logits: Tensor, log_std: Tensor) -> Tensor:
    """Type entropy plus the Gaussian's, [B]."""
    p = torch.softmax(tok_logits, -1)
    h_t = -(p * torch.log_softmax(tok_logits, -1)).sum(-1)
    h_g = (0.5 * math.log(2 * math.pi * math.e) + log_std).sum(-1)
    return h_t + h_g
