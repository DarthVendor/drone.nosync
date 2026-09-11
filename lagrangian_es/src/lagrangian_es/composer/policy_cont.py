"""The composer over tokens that carry continuous arguments.

Same shape as `policy.py` -- a transformer over the scene and the chain,
sampled on the recorded rows, fitted by a clipped surrogate -- but a decision
is now a token TYPE plus real-valued arguments rather than one of twenty-five
grid points.  See `actions_cont.py` for the grammar and the geometry.

The policy is a product of two pieces: a categorical over the five types, and
a diagonal Gaussian over the UNSQUASHED arguments.  The squash (tanh) is
applied when the action is built, not when it is sampled, so the density stays
exact and no change-of-variables term is needed.  A type that carries no
arguments -- EOS, LOOK -- contributes only its categorical term.

The trust region is ON here by default, and that is deliberate: the last
continuous composer this project ran was uncapped and reached a KL of 18 with
the judge flat.  Discrete tokens are far more forgiving of a large step than a
Gaussian whose mean can walk off.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from .actions_cont import PHI_MAX, ContVocab, entropy as cont_entropy, log_prob as cont_log_prob
from .base import COMPOSERS
from .policy import PolicyComposer, PolicyNet, collate_tok
from .tokens import drop_oldest_event
from .transformer import ACT_SCALE, INSTR


class ContPolicyNet(PolicyNet):
    """`PolicyNet` with a continuous argument head beside the type head."""

    def __init__(self, n_terms, **kw):
        # Deeper than the token composer's 2/2/1.  The scene is 78 entities of
        # beams, camera patches and map entries and the chain is now 128 events,
        # and two layers over each is thin for relating one to the other -- which
        # is the whole job: "the beam at +30 is clear AND I have not been there".
        kw.setdefault("n_scene", 4)
        kw.setdefault("n_chain", 4)
        kw.setdefault("n_read", 2)
        super().__init__(n_terms, **kw)
        d = self.embed.out_features
        self.vocab = ContVocab(n_terms)
        V, k = self.vocab.V, self.vocab.n_args
        self.act_emb = nn.Embedding(V, d)
        # A deeper token head (~103k) and a SEPARATE argument head per token
        # type (~19k each).  The previous heads were a single linear layer --
        # 325 and 130 parameters -- reading off a 64-wide pooled query, which is
        # very little capacity for deciding where to go and how far.
        H = 288
        self.head_act = nn.Sequential(nn.Linear(d, H), nn.GELU(),
                                      nn.Linear(H, H), nn.GELU(),
                                      nn.Linear(H, V))
        # PER TOKEN TYPE.  One shared head made slot 0 mean three different
        # things at once -- a waypoint's r, a turn's angle, a priority weight --
        # and emitted a single number for all of them, so the network could not
        # say "if I place, go 80% of the way; if I turn, turn 45 degrees".  The
        # priors collided too: the r bias commanded a 54-degree default turn and
        # the goal-bearing residual was being added to a priority weight.
        #
        # Held as stacked weights rather than a ModuleList so every token's head
        # runs in one batched matmul instead of V small ones.
        self.arg_w1 = nn.Parameter(torch.empty(V, d, H)); self.arg_b1 = nn.Parameter(torch.zeros(V, H))
        self.arg_w2 = nn.Parameter(torch.empty(V, H, k)); self.arg_b2 = nn.Parameter(torch.zeros(V, k))
        nn.init.normal_(self.arg_w1, std=(1.0 / d) ** 0.5)
        # `arg_w2` SMALL, NOT ZERO.  It was zeros, so that the net would start
        # exactly at the prior in `arg_b2` -- but mu = h @ arg_w2 + arg_b2, so
        # d(mu)/d(arg_w1) is PROPORTIONAL TO arg_w2, and at exactly zero the
        # gradient to arg_w1 -- and to the shared body and the scene encoder
        # underneath it -- is exactly zero too.  The branch is switched off
        # until arg_w2 lifts itself off zero, and measured over 578 iterations
        # it did not: arg_w2 reached 0.00136 while arg_w1 moved 0.9% from its
        # initial value (ratio 1.009 after ~11,000 optimizer steps).  So the
        # argument head stayed a constant -- r had sd 0.0088 across 6,945
        # decisions -- and the perception pathway received gradient ~500x
        # smaller than the heads, which is why it never learned to use a beam.
        #
        # Sized by measuring both ends of the trade.  At this scale arg_w2
        # starts at |w| 0.024 -- seventeen times the 0.00136 it reached after
        # 578 iterations from zero -- and the gradient reaching arg_w1 is 25x
        # what a timid 0.02 gives, while the prior moves by only 0.098 in u, so
        # r still sits near 0.60 of the radius where arg_b2 put it.
        # Sized on REALISTIC tokens, not synthetic ones -- the activations a
        # real scene produces are ~10x larger and a scale picked on random
        # inputs overshot badly (the bearing prior landed 88 degrees off).
        # At this scale the initial bearing is ~8 degrees off the goal (about
        # 1.1 m on an 8 m leg, re-decided every 0.4 s), r is unmoved at 0.608,
        # and the gradient reaching arg_w1 is 60x what the live run had.
        nn.init.normal_(self.arg_w2, std=0.05 * (1.0 / H) ** 0.5)
        # One std per argument slot, shared across states and learned.  A
        # state-dependent std is the usual next step and the usual way a
        # continuous policy collapses, so it is not the place to start.
        # Small on purpose.  The policy is Gaussian in the UNSQUASHED variable,
        # and near the goal bearing the squash is locally linear, so a std of
        # 0.35 there is about +-60 degrees of bearing on every waypoint -- the
        # prior was firing the vehicle off in random directions and nothing
        # arrived.  0.12 is about +-20 degrees.
        #
        # FIXED, not learned.  Under cross-entropy on its own successful flights
        # the maximum-likelihood fit drives the spread toward the residuals,
        # which shrinks exploration every update until the policy stops varying
        # and there is nothing left to select among.  Holding it constant keeps
        # the exploration the imitation depends on, and it is the exploration,
        # since nothing is injected on top.
        # TIME MODEL: how long to the goal if the vehicle goes via subgoal `g`.
        # The existing value head sees only the state, so it cannot say whether
        # one subgoal is quicker than another -- and that comparison is the
        # whole of navigation.  This one takes the read-out AND the subgoal.
        # Trained on the time actually measured, it is the only differentiable
        # path from a subgoal to a time, and the only reason perception becomes
        # necessary: nothing but the beams explains why a subgoal on the far
        # side of a building takes forever.
        #: Add the goal bearing/elevation to the WAYPOINT's arguments by hand.
        #: OFF.  Measured on a trained composer at 100% buildings, it supplied
        #: 99.9% of theta's variation and 99.7% of phi's, leaving the network
        #: deciding 0.0004 of a quantity that moved by 0.775 -- which is why
        #: perception had no effect on the output despite being linearly
        #: recoverable from the read-out at R^2 0.852.  The network now has to
        #: learn where the goal is from the goal token, like everything else.
        self.goal_residual = False
        self.time_head = nn.Sequential(nn.Linear(d + 3, d), nn.GELU(),
                                       nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.log_std = nn.Parameter(torch.full((k,), math.log(0.12)),
                                    requires_grad=False)
        # A remembered instruction IS its type together with its arguments.
        # Carrying them only as raw numbers in feature slots 2-3 conflated them
        # with what those slots mean for a MEASUREMENT token, where they are the
        # closest beam and whether it was alive -- one shared projection had to
        # serve both.  This gives every token type its own map from its own
        # arguments into the model's space, added to the token embedding, so
        # "I placed a waypoint 0.6 of the way at +30 degrees" is one vector.
        self.argin_w = nn.Parameter(torch.randn(V, k, d) * 0.02)
        self.argin_b = nn.Parameter(torch.zeros(V, d))
        nn.init.normal_(self.act_emb.weight, std=0.02)
        # SMALL, NOT ZERO -- the same trap `arg_w2` was in.  The prior lives in
        # this layer's bias, and zeroing the WEIGHT put the prior in place at
        # the cost of the gradient to everything beneath it: d(logits)/d(layer
        # below) is proportional to this weight, so at exactly zero the body and
        # the per-type sensor encoders got no gradient through the token path at
        # all.  Both heads were zeroed, so at initialisation the scene encoder
        # received nothing from either route.
        #
        # This head tolerates a much larger init than `arg_w2` did, because its
        # prior is a 2.0-logit gap between EOS and WAYPOINT rather than a
        # tanh-squashed value: measured on realistic tokens, this scale moves
        # `speak` from 0.125 to 0.114 while taking the gradient reaching
        # `head_act[0]` from 0 to 4.1e-01.  It also starts where the old run's
        # head took 578 iterations to reach (|w| 0.023 against 0.025).
        nn.init.normal_(self.head_act[-1].weight, std=0.5 * (1.0 / self.head_act[-1].weight.shape[1]) ** 0.5)
        with torch.no_grad():
            # NO HAND-SET PRIORS.  Both heads start flat and the network decides
            # everything: whether to speak, how far out to place, in what
            # direction.  What was here before, and why it went:
            #
            #   head bias EOS 6.0 / WAYPOINT 4.0 -- silence as the default, so
            #     an untrained composer was nearly harmless (0.977 against 0.992
            #     for no composer at 8 m legs).
            #   arg_b2[WAYPOINT, 0] = 0.7 -- the subgoal at ~80% of the way.
            #     Earlier it was 2.0, which read as the right answer (the
            #     identity) but sat at tanh slope 0.07: over 27,088 decisions
            #     every flight emitted r = 0.963 to three decimals and the
            #     braking end was 17 sigma away, never sampled once.
            #
            # Both were judgements about what the vehicle should want, and the
            # decomposition showed what that cost: the network was deciding
            # 0.1% of theta and 0.3% of phi, with the rest supplied by hand.
            # A composer that cannot choose its own behaviour cannot learn to
            # base it on what it sees.
            self.head_act[-1].bias.zero_()
            self.arg_b2.zero_()

    def _chain_in(self, tok):
        """The chain embedding, with an instruction's ARGUMENTS folded into the
        token's own vector rather than left as bare numbers in shared slots."""
        from .transformer import ACT_SCALE, INSTR
        # `embed_typed`, NOT the old shared `self.embed`.  This override
        # shadowed the base class, so while `encode_scene` was moved onto the
        # per-type encoders the CHAIN -- the drone's measurement stream and the
        # composer's own instruction history -- was still going through the one
        # 576-parameter Linear the per-type encoders replaced.  Half the input
        # path was left on the old encoder by an override nobody re-read.
        ch = self.embed_typed(tok["chain"], tok["chain_types"]) + self.type_emb(tok["chain_types"])
        is_i = (tok["chain_types"] == INSTR)
        ids = (tok["chain"][..., 0] * ACT_SCALE).round().long().clamp(0, self.vocab.V - 1)
        k = self.vocab.n_args
        args = tok["chain"][..., 2:2 + k]
        if args.shape[-1] < k:                                  # a chain from a narrower vocabulary
            args = torch.nn.functional.pad(args, (0, k - args.shape[-1]))
        w = self.argin_w[ids]                                   # [..., k, d], this token's own map
        vec = self.act_emb(ids) + (args.unsqueeze(-2) @ w).squeeze(-2) + self.argin_b[ids]
        return ch + vec * is_i.to(ch.dtype)[..., None]

    def pre(self, tok, store=None, scene=None):
        """Type logits [B, V], argument means [B, k], value [B].

        The bearing mean is a RESIDUAL about the goal: the head's output is
        added to whatever `u` would aim the waypoint straight at the goal, so a
        zero-weight head emits the straight line and everything it learns is a
        deviation from it.  The emitted argument is still the nose-relative
        bearing -- the same number the beams report theirs in -- this only
        moves where "no opinion" points.

        Without it the prior aimed every waypoint off the nose at half reach,
        which is a large detour, and flights stopped arriving at all.
        """
        q = self.read_out(tok, store, scene)
        B = q.shape[0]
        k = self.vocab.n_args
        h = torch.nn.functional.gelu(torch.einsum("bd,vdh->bvh", q, self.arg_w1) + self.arg_b1)
        mu = torch.einsum("bvh,vhk->bvk", h, self.arg_w2) + self.arg_b2    # [B, token, argument]
        g = self.goal_ego(tok).to(mu.dtype)
        bearing = torch.atan2(g[:, 1], g[:, 0]) / math.pi              # [-1, 1]
        # ELEVATION, the same residual one axis up.  `phi` spans [-PHI_MAX,
        # PHI_MAX], so the goal's own climb angle has to be expressed as a
        # fraction of that before it can be inverted through the squash -- and
        # a goal steeper than PHI_MAX simply saturates at full deflection,
        # which is the steepest the token can ask for anyway.
        elev = torch.atan2(g[:, 2], g[:, :2].norm(dim=-1).clamp_min(1e-9)) / PHI_MAX
        # each residual belongs to the WAYPOINT's own slot and nowhere else, so
        # a turn or a priority move is not offset by where the goal is
        mu = mu.clone()
        if self.goal_residual:
            # MEASURED, on a trained composer at 100% buildings: with these
            # residuals on, the network decides 0.1% of theta and 0.3% of phi.
            # The rest is this arithmetic -- a hand-written function of the goal,
            # recomputed every forward pass, that no amount of training can
            # remove.  theta varies by 0.775 across states and the head
            # contributes 0.0004 of it.  That is why perception has no effect on
            # the output despite being linearly recoverable from the read-out at
            # R^2 0.852: the output hardly depends on the NETWORK at all, so it
            # cannot depend on what the network sees.
            #
            # Off, the head must learn the bearing from the goal token itself.
            # It starts far worse -- the untrained composer no longer aims at
            # anything -- but every part of the output is then something the
            # network computes, and perception can compete on equal terms.
            mu[:, self.vocab.WAYPOINT, 1] = mu[:, self.vocab.WAYPOINT, 1] + \
                torch.atanh(bearing.clamp(-0.999, 0.999))
            mu[:, self.vocab.WAYPOINT, 2] = mu[:, self.vocab.WAYPOINT, 2] + \
                torch.atanh(elev.clamp(-0.999, 0.999))
        return self.head_act(q), mu, self.value(q).squeeze(-1)


class ContComposer(PolicyComposer):
    """Samples `[type, arguments]` per component and records both."""

    kind = "policy_cont"
    #: A beam return closer than this (metres) makes the row decide AT ONCE,
    #: without waiting for the report clock.  Decisions used to fire only every
    #: `measure_every` steps, so a fan that saw a wall at step 7 could not move
    #: the waypoint until step 20 -- 1.3 m at 5 m/s against a 6 m sensing range.
    #: Seeing an obstacle and being unable to act on it for a fifth of the
    #: horizon is the one thing a reactive navigator must not do.
    beam_trigger = 1.5
    #: steps that must pass since a row last decided, or a wall held in view
    #: re-fires every step and the chain fills with duplicates
    trigger_gap = 4

    def __init__(self, system, trainable, **kw):
        super().__init__(system, trainable, **kw)
        # `PolicyComposer` built the TOKEN net and loaded the weights file into
        # it.  Replacing the net here threw those weights away, so every freshly
        # built composer -- a restart, a worker, a diagnostic -- silently flew an
        # UNTRAINED policy.  It hid well: the heads are zero-initialised on
        # purpose, so an untrained net is input-independent, which reads as "the
        # model ignores its inputs" rather than as "the weights never loaded".
        self.net = ContPolicyNet(max(self.n_terms, 1), d=kw.get("d", 64),
                                 heads=kw.get("heads", 4),
                                 n_scene=kw.get("n_scene", 4), n_chain=kw.get("n_chain", 4),
                                 n_read=kw.get("n_read", 2)).to(self.net_dtype)
        self.net.goal_gain = self.tok.scale / self.reach
        # How much the policy VARIES when it flies.  Under imitation this is the
        # only engine producing new behaviour: every update sharpens the policy
        # toward what it already did, so without something widening the proposal
        # the loop converges to its own mean and stops.  It scales the token
        # logits and the argument spread together, and it applies only to
        # SAMPLING -- the likelihood the update fits is the untempered policy,
        # which is what makes this a proposal distribution rather than a change
        # of objective.
        self.temperature = float(kw.pop("temperature", 1.0)) if "temperature" in kw else \
            float(getattr(self, "temperature", 1.0))
        w = kw.get("weights", "") or getattr(self, "weights_path", "")
        if w:
            import os
            if os.path.exists(w):
                from .transformer import load_composer_weights
                load_composer_weights(self.net, w)
        self.net.eval()

    # --- the opening placement ------------------------------------------------
    def _apply(self, ids, ctx, tok):
        """The rollout opens every flight with `vocab.straight`.

        For the token composer that was a single component, polar about the
        GOAL direction, so "straight" needed no argument.  Here theta is
        measured from the nose, so aiming at the goal is an argument and has to
        be computed: `WAYPOINT(r=+1, theta=bearing of the goal)`.
        """
        V = self.net.vocab
        x, goal = ctx["x"], ctx["goal"]
        psi = tok["psi"].to(self.dtype)
        g_ego = self.net.goal_ego(tok).to(self.dtype)
        B = x.shape[0]
        arg = torch.zeros(B, V.n_args, dtype=x.dtype, device=x.device)
        arg[:, 0] = 1.0                                            # the whole reachable radius
        arg[:, 1] = torch.atan2(g_ego[:, 1], g_ego[:, 0]).to(x.dtype) / math.pi
        spec = V.apply(ids.to(torch.long), arg, self._current(ctx, B), x, goal, psi,
                       g_ego, self.reach, self.z_min)
        self._remember(ctx, ids.to(torch.long))
        return spec

    # --- the chained decision --------------------------------------------------
    @torch.no_grad()
    def _decide_cont(self, ctx, choose):
        """`_decide` for typed tokens with arguments.

        Kept separate from the token version rather than generalising it: the
        two differ in what a component IS, and threading an optional argument
        through the other one would make the hot path conditional.
        """
        x, goal = ctx["x"], ctx["goal"]
        B = x.shape[0]
        rows_full = getattr(self, "_rows", None)
        ids_full = rows_full if rows_full is not None else torch.arange(B, device=x.device)
        if rows_full is None:
            self._B = B
        V = self.net.vocab
        tok0 = self.tokens(ctx)
        psi = tok0["psi"].to(self.dtype)
        g_ego = self.net.goal_ego(tok0).to(self.dtype)
        out, pend, has = V.begin(self._current(ctx, B), psi)
        active = torch.ones(B, dtype=torch.bool, device=x.device)
        alive0 = ctx["alive"]
        t_now = ctx.get("t", 0)
        saved = getattr(self, "_rows", None)
        tok_cur, cur_pos, last_act, scene0 = tok0, torch.arange(B, device=x.device), None, None
        last_arg = None
        try:
            for step in range(V.L_MAX + 1):
                idx = active.nonzero().flatten()
                if idx.numel() == 0:
                    break
                if step == 0:
                    tok = tok0
                    scene0 = self.net.encode_scene(tok0)
                    sc = scene0
                else:
                    keep = torch.isin(cur_pos, idx)
                    n_prev = cur_pos.numel()
                    tok = {k: (v[keep] if torch.is_tensor(v) and v.ndim and v.shape[0] == n_prev else v)
                           for k, v in tok_cur.items()}
                    ch = tok["chain"]
                    n = idx.numel()
                    row = torch.zeros(n, 1, ch.shape[-1], dtype=ch.dtype, device=ch.device)
                    row[:, 0, 0] = last_act[keep].to(ch.dtype) / ACT_SCALE
                    if last_arg is not None:                       # and what it said it with
                        a_prev = last_arg[keep]
                        for j in range(min(a_prev.shape[-1], ch.shape[-1] - 2)):
                            row[:, 0, 2 + j] = a_prev[:, j].to(ch.dtype)
                    chain = torch.cat([ch, row], 1)
                    ctypes = torch.cat([tok["chain_types"],
                                        torch.full((n, 1), INSTR, dtype=torch.long, device=ch.device)], 1)
                    cmask = torch.cat([tok["chain_mask"],
                                       torch.zeros(n, 1, dtype=torch.bool, device=ch.device)], 1)
                    kc = self.tok.kc
                    while chain.shape[1] > kc:
                        chain, ctypes, cmask = drop_oldest_event(chain, ctypes, cmask)
                    tok["chain"], tok["chain_types"], tok["chain_mask"] = chain, ctypes, cmask
                    sc = (scene0[0][idx], scene0[1][idx])
                self._rows = ids_full[idx]
                logits, mu, _ = self.net.pre(tok, scene=sc)
                if step == V.L_MAX:                       # the cap: only EOS is left
                    forced = torch.full_like(logits, float("-inf"))
                    forced[:, V.EOS] = 0.0
                    logits = forced
                # `x` and `goal` travel with the sub-context for the error
                # loss: the composer's output is a claim about where the vehicle
                # can get, and the error needs the position it claimed FROM.
                ctx_s = {"alive": alive0[idx], "t": t_now, "x": x[idx], "goal": goal[idx]}
                # THE BEAM RANGES TRAVEL WITH IT TOO.  `ctx_s` is built fresh
                # per component step, so anything not put here is invisible to
                # `choose` -- the variational sampler read `ctx_s["obs"]`, found
                # nothing, and silently never fired: measured, var_temp 0 and
                # var_temp 2 gave byte-identical arrival (0.469 both).  Only the
                # forward ring is needed, so only that is carried.
                _rng0 = (ctx.get("obs") or {}).get("range")
                if torch.is_tensor(_rng0) and _rng0.ndim == 2:
                    ctx_s["range"] = _rng0[idx]
                act, u = choose(logits, mu, tok, ctx_s, ids_full[idx], step, has[idx])
                arg = V.squash(u)
                V.step(act, arg, out, pend, has, psi, rows=idx)
                self._remember_args(ctx_s, act, arg)
                last_arg = arg
                tok_cur, cur_pos, last_act = tok, idx, act
                active[idx[act == V.EOS]] = False
        finally:
            self._rows = saved
        return V.finish(out, pend, has, x, goal, psi, g_ego, self.reach, self.z_min)

    def _remember_args(self, ctx, ids, args):
        """`_remember`, plus the arguments the instruction carried."""
        rows = getattr(self, "_rows", None)
        B = ids.shape[0] if rows is None else self._B
        dev = ids.device
        full = torch.zeros(B, dtype=torch.long, device=dev)
        valid = torch.zeros(B, dtype=torch.bool, device=dev)
        fargs = torch.zeros(B, args.shape[-1], dtype=args.dtype, device=dev)
        spoke = ids != self.net.vocab.EOS                       # a bare EOS is silence
        sel = spoke.nonzero().flatten() if rows is None else rows[spoke]
        full[sel] = ids[spoke]
        valid[sel] = True
        fargs[sel] = args[spoke]
        self._instr.append((float(ctx.get("t", 0)), full, valid, fargs))
        if len(self._instr) > 4 * self.tok.kc:
            self._instr = self._instr[-2 * self.tok.kc:]

    def _beam_geom(self, device, dtype):
        """The forward ring's unit directions and max range, from the
        TOKENIZER's layout -- the same geometry the sensor casts with, so the
        action cannot disagree with what the beams reported."""
        cached = getattr(self, "_bg", None)
        if cached is not None:
            d, r = cached
            return d.to(device=device, dtype=dtype), r
        for name, kind, bear, rmax, _geom in getattr(self.tok, "layout", ()):
            if kind == "beam" and not str(name).endswith("_down"):
                az, el = bear[:, 0], bear[:, 1]
                ce = torch.cos(el)
                d = torch.stack([torch.cos(az) * ce, torch.sin(az) * ce, torch.sin(el)], -1)
                self._bg = (d, float(rmax))
                return d.to(device=device, dtype=dtype), float(rmax)
        self._bg = (None, 0.0)
        return None, 0.0

    @torch.no_grad()
    def emit(self, ctx):
        """Sample a decision per row and record it for the update."""
        eps = float(getattr(self, "explore_eps", 0.0))
        rec_rows = self.record_rows.to(ctx["x"].device) if self.record_rows is not None else None
        exploring = self.stochastic and eps > 0.0 and rec_rows is not None and len(rec_rows)

        def choose(logits, mu, tok, ctx_s, ids, step, placing):
            B = logits.shape[0]
            dev = logits.device
            rec = (torch.isin(ids, rec_rows) if rec_rows is not None
                   else torch.zeros(B, dtype=torch.bool, device=dev))
            b_logits = logits
            if exploring:                              # the type mixture, as in the token composer
                V = logits.shape[-1]
                mix = torch.log((1.0 - eps) * torch.softmax(logits, -1) + eps / V)
                mix = torch.where(torch.isfinite(logits), mix, logits)
                b_logits = torch.where(rec[:, None], mix, logits)
            # SPEAK FLOOR.  With probability `speak_floor` a decision may not
            # OPEN on EOS, so it has to say at least one thing; the components
            # after the first are untouched, so the chain still ends when the
            # policy wants it to.  This exists because silence is ABSORBING:
            # `progress_weights` credits only a decision that placed a subgoal,
            # so a policy that has stopped placing produces no samples at all
            # and no gradient can bring it back (measured: subgoals 0.3 -> 0.0,
            # tokens 18 -> 0 by iteration 38, and flat thereafter).  It biases
            # only what is SAMPLED; the update still judges whatever comes out
            # by the flight's outcome, so a forced token that hurts is pushed
            # down exactly like a volunteered one.
            if getattr(self, "mute", False):        # the control: say nothing
                b_logits = torch.full_like(b_logits, -1e9)
                b_logits[:, self.net.vocab.EOS] = 0.0
            p_sp = float(getattr(self, "speak_floor", 0.0))
            if self.stochastic and p_sp > 0.0 and step == 0:
                force = torch.rand(B, device=dev) < p_sp
                if bool(force.any()):
                    # PIN THE FORCED TOKEN TO WAYPOINT, not merely "not EOS".
                    # Excluding EOS alone does not help once the policy has
                    # collapsed onto one of the other types: measured, it went
                    # to heading 100% / place 0% by judge 40, so forcing
                    # non-silence just forced more TURN and the update still
                    # never saw a placement.  WAYPOINT is the only token that
                    # moves the subgoal, so it is the one that has to be kept
                    # in the sample if the update is ever to learn whether
                    # placing helps.  The ARGUMENTS are still drawn from the
                    # policy's own mu and std, so where it places is explored
                    # rather than dictated.
                    row = torch.full((b_logits.shape[-1],), float("-inf"),
                                     dtype=b_logits.dtype, device=dev)
                    row[self.net.vocab.WAYPOINT] = 0.0
                    b_logits = torch.where(force[:, None], row[None].expand_as(b_logits), b_logits)
            temp = float(getattr(self, "temperature", 1.0))
            std = self.net.log_std.exp().to(mu.dtype) * temp
            rows_ = torch.arange(B, device=dev)
            if not self.stochastic:
                act = logits.argmax(-1)
                u = mu[rows_, act]                       # the chosen token's own arguments
            else:
                act = torch.distributions.Categorical(logits=b_logits / temp).sample()
                mu = mu[rows_, act]
                # the arguments are explored by WIDENING the Gaussian on the
                # recorded rows, which keeps the density exact (a uniform
                # mixture over an unbounded variable does not)
                s = std[None].expand_as(mu)
                if exploring:
                    s = torch.where(rec[:, None], s * (1.0 + eps), s)
                u = mu + s * torch.randn_like(mu)
                # PROBABILISTIC WAYPOINTS FROM THE ACTION.  For the rows whose
                # token is a WAYPOINT, redraw the argument from exp(-S/T) over
                # `var_k` of the policy's OWN candidates, S being the two-leg
                # path action through what the beams see.  The network still
                # decides where to look; the action decides which of the places
                # it was already considering is worth going to.  Uniform
                # exploration in a 10 m ball measured a p95 advantage of
                # exactly ZERO over 256 paired flights -- nothing it tried ever
                # helped -- which is why this exists.
                if float(getattr(self, "var_temp", 0.0)) > 0.0:
                    wp = (act == self.net.vocab.WAYPOINT)
                    if bool(wp.any()):
                        dirs, rmax = self._beam_geom(dev, mu.dtype)
                        rng_all = ctx_s.get("range")
                        if dirs is not None and torch.is_tensor(rng_all):
                            gk = self.net.goal_ego(tok).to(mu.dtype)[wp]
                            u = u.clone()
                            u[wp] = variational_u(
                                self.net, mu[wp], std, gk, dirs, rng_all[wp].to(mu.dtype),
                                rmax, self.reach, k=int(getattr(self, "var_k", 16)),
                                temperature=float(self.var_temp),
                                lam=float(getattr(self, "var_lam", 2.0)))
            if self.stochastic and rec_rows is not None and bool(rec.any()):
                V = self.net.vocab
                n_args = torch.tensor([V.n_arg_of(int(t)) for t in act.tolist()],
                                      device=dev, dtype=torch.long)
                self.records.append({
                    "t": float(ctx_s.get("t", 0)), "act": act[rec].clone(), "u": u[rec].clone(),
                    "n_args": n_args[rec].clone(),
                    "moved": (act == self.net.vocab.WAYPOINT)[rec].clone(),
                    "logits": b_logits[rec].float().clone(),
                    "pi_logits": logits[rec].float().clone(),
                    "mu": mu[rec].float().clone(),
                    "log_std": self.net.log_std.detach().float().clone(),
                    "alive": ctx_s["alive"][rec].clone(), "rows": ids[rec].clone(),
                    # WHERE THE VEHICLE WAS, for the error loss.  The composer's
                    # output is a claim about where the drone can be by the next
                    # decision; the error needs the position it started from and
                    # the one it reached, and nothing else supplies them.
                    "x": ctx_s["x"][rec].float().clone(),
                    "goalw": ctx_s["goal"][rec].float().clone(),
                    "tok_keep": None,
                    "tok": {k: (v[rec].clone() if torch.is_tensor(v) and v.ndim and v.shape[0] == B
                                else (v.clone() if torch.is_tensor(v) else v)) for k, v in tok.items()}})
            return act, u

        return self._decide_cont(ctx, choose)


def arrival_weights(t: Tensor, task: Optional[Tensor] = None, tau: float = 1.0,
                    signed: bool = False, w_max: float = 2.0,
                    arrived: Optional[Tensor] = None) -> Tensor:
    """A weight per flight from its arrival time `t` (`finish_frac`; 1.0 = never).

    `task` says which flights flew the SAME task, and its absence is the whole
    reason the composer stalled: with one flight per task, ranking arrival
    times ranks the TASKS, so "the fastest 30%" means "the 30% nearest goals"
    and the update trains on the easy third of the distribution.  Centring
    within a task removes the task's own difficulty and leaves what the
    decisions changed -- the same argument as `center_by_task`, applied to the
    imitation filter instead of the return.

    `signed=False` is imitation: a flight that never arrived has no arrival
    time and takes weight ZERO, because a positive weight on a failure makes
    that failure more likely.  The consequence is that this can only move
    probability around among successes -- it cannot lower the failure rate.

    `signed=True` lets the weight go negative, which is what a failure needs:
    the weight is the negated, centred arrival time, so slower-than-its-task
    flights (a failure is the slowest possible) are pushed DOWN.  That is a
    policy gradient with a per-task baseline, written as a weight -- the loss
    is still the one cross-entropy term.  It is also unbounded below in
    principle, so the magnitude is clipped at `w_max`.
    """
    t = t.to(torch.float64)
    a = t.clone()
    grouped = False
    if task is not None and task.numel() == t.numel():
        _, inv = torch.unique(task, return_inverse=True)
        m = int(inv.max()) + 1
        cnt = torch.zeros(m, dtype=a.dtype).index_add_(0, inv, torch.ones_like(a))
        sm = torch.zeros(m, dtype=a.dtype).index_add_(0, inv, a)
        # One sample per task is no baseline at all: subtracting a task's own
        # mean from its single flight leaves exactly zero, every weight comes
        # out 1, and the update degenerates to plain imitation of everything.
        # That is the k=1 control, and it has to centre on the batch instead.
        if float(cnt.max()) > 1.0:
            a = a - (sm / cnt.clamp_min(1.0))[inv]
            grouped = True
    if not grouped:
        a = a - a.mean()
    # Which flights arrived, stated rather than inferred.  `t == 1.0` used to
    # mean "never arrived" because the score was `finish_frac`; with SOFT TIME
    # as the score that sentinel is gone -- 0.99 is a flight that spent nearly
    # the whole episode away from the goal, not a failure -- so the caller
    # passes the flag.  Only the unsigned path needs it: signed weighting ranks
    # failures too, which is the point of soft time.
    arr = (arrived.to(torch.bool) if arrived is not None else (t < 1.0))
    # SCALE ON THE ROWS THAT CARRY WEIGHT.  A flight that never arrived sits at
    # finish_frac 1.0, far above any real arrival time, and unsigned weighting
    # gives it zero weight -- but it was still inflating the spread everything
    # else is divided by.  Measured: at a 20% failure rate the arrivals' z
    # collapsed toward 0 and the effective sample size went to 0.93 of them
    # (0.37 by design), i.e. the weights turned uniform and the update stopped
    # selecting at all.  Exactly when obstacles appear and selection matters
    # most.  The signed path keeps the full spread, because there every row
    # carries weight and the failures are the point.
    ref = a if (signed or int(arr.sum()) < 2) else a[arr]
    a = a / ref.std().clamp_min(1e-9)
    if signed:
        return (-a / max(tau, 1e-9)).clamp(-w_max, w_max)
    w = torch.zeros_like(a)
    if int(arr.sum()) > 0:
        e = torch.exp((-a[arr] / max(tau, 1e-9)).clamp(max=10.0))
        w[arr] = e / e.mean().clamp_min(1e-12)       # mean 1 over the arrivals
    return w


def _subgoal_ego(net: ContPolicyNet, mu: Tensor, g_ego: Tensor) -> Tensor:
    """The commanded subgoal in the vehicle frame, in REACH units, differentiably.

    The same spherical geometry `ContVocab.finish` uses, rebuilt here from the
    head's mean so the error can be pushed back into it.  No sampling: this is
    where the policy is POINTING, which is the thing the error is about.
    """
    from .actions_cont import PHI_MAX
    a = torch.tanh(mu)                                   # [n, k] squashed arguments
    L0 = g_ego.norm(dim=-1)
    radius = L0.clamp(max=1.0)
    r = (a[:, 0] + 1.0) * 0.5 * radius
    theta = math.pi * a[:, 1]
    phi = PHI_MAX * a[:, 2] if a.shape[-1] > 2 else torch.zeros_like(theta)
    cphi = torch.cos(phi)
    return torch.stack([r * cphi * torch.cos(theta), r * cphi * torch.sin(theta),
                        r * torch.sin(phi)], -1)


def variational_u(net: "ContPolicyNet", mu: Tensor, std: Tensor, g_ego: Tensor,
                  dirs: Optional[Tensor], rng: Optional[Tensor], max_range: float,
                  reach: float, k: int = 16, temperature: float = 2.0,
                  lam: float = 2.0) -> Tensor:
    """Draw a WAYPOINT argument from `exp(-S/T)` over `k` of the policy's own draws.

    The candidates come from the policy -- `mu + std * randn` -- so the network
    still decides where to look; the action only decides which of the places it
    was already considering is worth going to.  At a large temperature this is
    the policy untouched.

    `g_ego` is in REACH units (the convention `_subgoal_ego` works in); the
    action is computed in metres, because the barrier compares against beam
    ranges.  Returns the chosen `u`, [n, k_args].
    """
    from .variational import beam_points, boltzmann_pick, path_action
    n, ka = mu.shape
    cand = mu[:, None, :] + std[None, None, :ka] * torch.randn(
        n, int(k), ka, dtype=mu.dtype, device=mu.device)
    flat = cand.reshape(n * int(k), ka)
    sub = _subgoal_ego(net, flat, g_ego.repeat_interleave(int(k), 0)).reshape(n, int(k), 3) * float(reach)
    if dirs is None or rng is None or rng.shape[0] != n:
        return cand[:, 0]                        # no beams this step: the plain draw
    pts_hit = beam_points(dirs.to(mu.dtype), rng.to(mu.dtype), max_range)
    S = path_action(sub, g_ego.to(mu.dtype) * float(reach), dirs.to(mu.dtype),
                    rng.to(mu.dtype), pts_hit[1], lam=lam)
    idx = boltzmann_pick(S, temperature=temperature)
    return cand[torch.arange(n, device=mu.device), idx]


def time_update(net: ContPolicyNet, records: List[Dict], t_left: List[Tensor],
                ep_steps: int = 1800, epochs: int = 2, batch: int = 1024, lr: float = 1e-4,
                opt=None, max_samples: int = 0, policy_w: float = 0.25) -> Dict[str, float]:
    """PURELY TIME.  Nothing else appears in it.

        T_hat(s, g)   the composer's own estimate of time-to-goal via subgoal g
        T             the time the flight actually took from that decision on

        L_model  = (T_hat(s, g_emitted) - T)^2      learn the time model
        L_policy = T_hat(s, g(theta))               choose the quickest subgoal

    Measured time cannot be differentiated with respect to the subgoal -- it
    comes out of the simulator -- and a hand-written prediction (distance over
    speed) is blind to obstacles.  Putting the prediction in a NETWORK resolves
    both: d(T_hat)/dg is exact so the policy can descend it, and T_hat only
    becomes accurate by reading the beams, because nothing else explains why a
    subgoal behind a building takes forever.  Perception stops being optional --
    it is the only thing that predicts time.

    The degeneracies that sank the other formulations close by themselves, with
    no balancing term:

      subgoal at the vehicle's own feet -> no progress -> time maximal -> the
        model learns it -> the policy avoids it
      subgoal through a building -> the flight never arrives -> time maximal

    A crash needs no penalty term either: a crash IS infinite time to the goal.

    The risk is the usual one for descending a learned model: the policy can
    find where T_hat is optimistically wrong.  `match` reports the signed
    model error for exactly that -- if it drifts negative, the model is being
    gamed and the policy step is too large relative to the fit.
    """
    groups = records if (records and isinstance(records[0], list)) else [records]
    lefts = list(t_left) if isinstance(t_left, (list, tuple)) else [t_left]
    toks, tt = [], []
    for gi, recs in enumerate(groups):
        left = lefts[gi] if gi < len(lefts) else None
        if left is None:
            continue
        for rec in recs:
            if not rec.get("tok"):
                continue
            rows = rec.get("rows")
            t_now = float(rec.get("t", 0))
            for j in range(int(rec["alive"].shape[0])):
                if not bool(rec["alive"][j]):
                    continue
                b = int(rows[j]) if rows is not None else j
                fin = float(left[b])          # finish_frac: 1.0 means it never arrived
                toks.append({k: (v[j] if torch.is_tensor(v) else v) for k, v in rec["tok"].items()})
                tt.append(min(1.0, max(0.0, (fin * ep_steps - t_now) / max(ep_steps, 1))))
    if not toks:
        return {"n": 0}
    from .policy import collate_tok
    gen = torch.Generator().manual_seed(0)
    if max_samples and len(toks) > max_samples:
        pick = torch.randperm(len(toks), generator=gen)[:max_samples].sort().values.tolist()
        toks = [toks[i] for i in pick]; tt = [tt[i] for i in pick]
    tok_all = collate_tok(toks)
    T = torch.tensor(tt, dtype=torch.float32)
    n = T.shape[0]
    opt = opt or torch.optim.Adam(net.parameters(), lr=lr)
    acc = {"model": 0.0, "pol": 0.0, "err": 0.0, "nb": 0}
    W = net.vocab.WAYPOINT
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        for s0 in range(0, n, batch):
            idx = order[s0:s0 + batch]
            tk = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in tok_all.items()}
            q = net.read_out(tk)
            _, mu_all, _ = net.pre(tk)
            if not (torch.isfinite(q).all() and torch.isfinite(mu_all).all()):
                continue
            g_ego = net.goal_ego(tk).to(mu_all.dtype)
            g = _subgoal_ego(net, mu_all[:, W], g_ego)
            # (1) fit the model to the time actually taken, on the subgoal
            #     actually emitted -- both held constant here
            # SIGMOID: time is a fraction of the episode, so T_hat must live in
            # (0, 1).  Unbounded, the policy drove it to -0.4 within two
            # iterations -- predicting negative time -- while the model's own
            # fit error ROSE from 0.98 to 1.21.  It was not learning to be
            # quick, it was walking the model into territory the model had
            # never seen and believing what it found there.  Bounded, the worst
            # it can claim is "instant", and the model saturates instead of
            # running away.
            t_hat = torch.sigmoid(net.time_head(torch.cat([q.detach(), g.detach()], -1)).squeeze(-1))
            l_model = ((t_hat - T[idx]) ** 2).mean()
            # (2) move the policy DOWN the model.  The model's own parameters
            #     still receive gradient from (1) only, because this term is
            #     what the policy is being scored by, not what the model is.
            t_pol = torch.sigmoid(net.time_head(torch.cat([q, g], -1)).squeeze(-1))
            l_pol = t_pol.mean()
            (l_model + policy_w * l_pol).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                acc["model"] += float(l_model); acc["pol"] += float(l_pol)
                acc["err"] += float((t_hat - T[idx]).mean())      # negative = optimistic
                acc["nb"] += 1
    nb = max(1, acc["nb"])
    with torch.no_grad():
        p = torch.softmax(net.pre({k: (v[:512] if torch.is_tensor(v) else v)
                                   for k, v in tok_all.items()})[0], -1).mean(0)
    return {"n": n, "flights": n, "kept_flights": n, "ess": 0.0,
            "ce": acc["pol"] / nb, "nll": acc["model"] / nb, "match": acc["err"] / nb,
            "speak": float(1.0 - p[net.vocab.EOS]), "std": float(net.log_std.detach().exp().mean()),
            "kl": 0.0, "clipfrac": 0.0, "entropy": 0.0, "ev": float("nan"), "w_up": 0.0, "w_abs": 0.0}


def error_update(net: ContPolicyNet, records: List[Dict], reach: float = 10.0,
                 epochs: int = 2, batch: int = 1024, lr: float = 1e-4, opt=None,
                 max_samples: int = 0, alpha: float = 1.0, beta: float = 1.0) -> Dict[str, float]:
    """The composer's own error, with no reward, no return and no labels.

    The low level is FROZEN and knows the mechanics of flying.  The composer's
    output is therefore a CLAIM -- "the vehicle can be at g by the next
    decision" -- and reality answers it.  Two errors, and they are the only two
    mistakes a navigator can make:

        e_reach = |x_next - g| / reach     it asked for somewhere unreachable
        e_aim   = |g - goal|   / reach     it did not ask to go toward the goal

        L = alpha * e_reach^2 + beta * e_aim^2

    Each alone is degenerate and each kills the other's degeneracy: e_reach
    alone puts the subgoal at the vehicle's own feet, always reachable and never
    moving; e_aim alone puts it on the goal through a building.  Together the
    optimum is to place the subgoal AS FAR TOWARD THE GOAL AS THE VEHICLE CAN
    ACTUALLY REACH -- which is navigation, stated entirely as error.

    Why this and not the weighted log-probability it replaces.  That handed ONE
    scalar to ~90 decisions: measured, the advantage separated crashed from
    surviving flights at z = -166 while correlating -0.013 with the braking
    argument.  It knew the flight was bad and not which decision made it so, so
    `r` sat frozen at sd 0.0088 and perception stayed decorative.  Here every
    decision carries its own vector error and the gradient is EXACT -- `g` is an
    analytic function of the arguments, `x_next` is a constant target, so there
    is no score-function estimator and no sampling variance.

    Crashes need no penalty term: a vehicle that died before the next decision
    is far from what was commanded, so e_reach is large for exactly the
    decisions that flew it in.
    """
    groups = records if (records and isinstance(records[0], list)) else [records]
    toks, mus_g, tgt, goals = [], [], [], []
    for recs in groups:
        # pair each decision with the NEXT one from the same flight: that is the
        # interval the subgoal was actually held for
        by_row = {}
        for rec in recs:
            if not rec.get("tok") or "x" not in rec:
                continue
            rows = rec.get("rows")
            for j in range(int(rec["alive"].shape[0])):
                if not bool(rec["alive"][j]):
                    continue
                b = int(rows[j]) if rows is not None else j
                by_row.setdefault(b, []).append((rec, j))
        for b, seq in by_row.items():
            for (r0, j0), (r1, j1) in zip(seq, seq[1:]):
                toks.append({k: (v[j0] if torch.is_tensor(v) else v) for k, v in r0["tok"].items()})
                tgt.append(r1["x"][j1] - r0["x"][j0])       # world displacement actually achieved
                goals.append(r0["goalw"][j0] - r0["x"][j0])  # world offset to the goal
                mus_g.append(r0["tok"]["psi"][j0] if torch.is_tensor(r0["tok"].get("psi")) else torch.zeros(()))
    if not toks:
        return {"n": 0}
    from .policy import collate_tok
    from .tokens import to_ego
    gen = torch.Generator().manual_seed(0)
    if max_samples and len(toks) > max_samples:
        pick = torch.randperm(len(toks), generator=gen)[:max_samples].sort().values.tolist()
        toks = [toks[i] for i in pick]; tgt = [tgt[i] for i in pick]
        goals = [goals[i] for i in pick]; mus_g = [mus_g[i] for i in pick]
    tok_all = collate_tok(toks)
    psi = torch.stack([p.reshape(()) for p in mus_g]).float()
    d_world = torch.stack(tgt).float()
    g_world = torch.stack(goals).float()
    # everything in the vehicle's own frame, in units of the reach
    d_ego = to_ego(d_world, psi) / reach
    gl_ego = to_ego(g_world, psi) / reach
    n = d_ego.shape[0]
    opt = opt or torch.optim.Adam(net.parameters(), lr=lr)
    acc = {"reach": 0.0, "aim": 0.0, "nb": 0}
    W = net.vocab.WAYPOINT
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        for s0 in range(0, n, batch):
            idx = order[s0:s0 + batch]
            tk = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in tok_all.items()}
            _, mu_all, _ = net.pre(tk)
            if not torch.isfinite(mu_all).all():
                continue
            g = _subgoal_ego(net, mu_all[:, W], gl_ego[idx])
            e_reach = ((g - d_ego[idx]) ** 2).sum(-1)
            e_aim = ((g - gl_ego[idx]) ** 2).sum(-1)
            # SCALE-FREE.  Both terms are squared distances in reach units, but
            # they are not the same size: measured on the city, e_aim ~ 1.5 and
            # e_reach ~ 0.4, so an even alpha:beta hands the aim term three
            # times the gradient and it simply wins -- the composer learns to
            # put the subgoal ON the goal and ignore whether it can get there,
            # which is the degenerate arm each term is supposed to prevent in
            # the other.  Dividing by each term's own detached batch mean makes
            # the balance a ratio of RELATIVE improvement, so neither can
            # dominate by being larger, and alpha:beta means what it says.
            loss = (alpha * e_reach / e_reach.mean().detach().clamp_min(1e-6)
                    + beta * e_aim / e_aim.mean().detach().clamp_min(1e-6)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                acc["reach"] += float(e_reach.mean()); acc["aim"] += float(e_aim.mean()); acc["nb"] += 1
    nb = max(1, acc["nb"])
    with torch.no_grad():
        p = torch.softmax(net.pre({k: (v[:512] if torch.is_tensor(v) else v)
                                   for k, v in tok_all.items()})[0], -1).mean(0)
    return {"n": n, "flights": n, "kept_flights": n, "ess": 0.0,
            "ce": acc["reach"] / nb + acc["aim"] / nb,
            "nll": acc["reach"] / nb, "match": acc["aim"] / nb,
            "speak": float(1.0 - p[net.vocab.EOS]), "std": float(net.log_std.detach().exp().mean()),
            "kl": 0.0, "clipfrac": 0.0, "entropy": 0.0, "ev": float("nan"), "w_up": 0.0, "w_abs": 0.0}


# `progress_weights` was DELETED here.
#
# It credited a decision with the distance closed between it and the NEXT
# decision of the same flight (`zip(seq, seq[1:])`).  That was written for the
# ~90 decisions a flight the composer used to make, where losing the last one
# costs 1% of the data.  Under sparse injection a flight has one or two
# decisions, so the last -- often the only -- is always uncredited, and most
# decisions fire after the drone has parked on its goal with no distance left
# to close.  MEASURED on the live checkpoint: the speak floor emitted 13
# WAYPOINTs (10.40% of components, exactly its 0.10 setting) and 0 of 13
# survived `progress_weights > 0`.  The exploration worked; the credit threw
# all of it away, and the policy drifted freely -- TURN 100%, then LOOK 78%,
# then EOS 99.9% -- because nothing that could have argued otherwise ever
# reached the update.
#
# It was also confounded even when it did fire: the distance closed in an
# interval is mostly the FROZEN LOW LEVEL flying to the goal, not the token's
# doing, so it paid the composer for work it did not do.
#
# Credit now comes from `paired_advantage`: the same task and seed flown twice,
# once with the injected token and once with the composer muted, and the token
# is worth the DIFFERENCE.  That cancels the low level's own progress exactly,
# gives every emitted token a weight, and scores silence at zero by
# construction, since silence IS the control.


def projected_time(finish_frac: Tensor, final_err: Tensor, success: Tensor,
                   span: float) -> Tensor:
    """Time to the goal, PROJECTED for the flights that never got there.

    A flight that arrives is scored by when it arrived.  One that does not has
    no arrival time, and `finish_frac` gives every such flight exactly 1.0 --
    so in a paired comparison two failures tie at zero and teach nothing, even
    when one crashed at two seconds and the other died a metre short.  MEASURED:
    with arrival at 0.30, ~70% of pairs were both-failures, and only 13% of
    flights produced any signal at all.

    So a failure is charged 1.0 plus the time the distance it still had to go
    WOULD have taken at cruise: `final_err / span`, where `span` is how far the
    vehicle travels in a whole episode.  That keeps one currency -- time -- and
    ranks failures by how close they got, while never letting a crash look
    fast: dying early leaves the most distance outstanding and therefore costs
    the most.  This is the property `soft_time` gets backwards, since it stops
    accumulating at death and so pays for crashing.
    """
    t = finish_frac.to(torch.float64).reshape(-1)
    extra = final_err.to(torch.float64).reshape(-1) / max(float(span), 1e-9)
    return torch.where(success.to(torch.bool).reshape(-1), t, t + extra)


def paired_advantage(t_arr: Tensor, t_ctl: Tensor) -> Tensor:
    """TIME SAVED against the muted twin: `t_ctl - t_arr`.

    `t_arr` is the injected flight's finish fraction and `t_ctl` its control's
    -- the same task, the same seed, the same initial state, flown by the
    frozen low level with the composer muted.  Positive means the tokens got
    there sooner.

    Time is the whole objective and nothing else is added to it.  It already
    contains arrival: a flight that never arrives finishes at 1.0, the worst
    score there is, so rescuing a flight the control lost is the largest
    possible gain and losing one the control won is the largest possible loss.
    A crash is simply a flight that never arrives. No bonus, no death charge,
    no shaping -- the fastest one there wins.

    It is a COUNTERFACTUAL, not a correlation: whatever the frozen low level
    would have done unaided happens in both flights and cancels, so this cannot
    pay the composer for distance it did not close.  That was the flaw in the
    per-decision progress credit this replaces, which measured the low level's
    own progress and attributed it to whatever token happened to be live.
    """
    return t_ctl.to(torch.float64).reshape(-1) - t_arr.to(torch.float64).reshape(-1)


def imitate_update(net: ContPolicyNet, records: List[Dict], reached: List[Tensor],
                   epochs: int = 2, batch: int = 1024, lr: float = 1e-4,
                   opt=None, max_samples: int = 0, keep_frac: float = 1.0,
                   score: Optional[List[Tensor]] = None, weight_tau: float = 0.0,
                   task: Optional[List[Tensor]] = None, signed: bool = False,
                   w_max: float = 2.0, advantage: Optional[List[Tensor]] = None,
                   reach: float = 10.0) -> Dict[str, float]:
    """Cross-entropy on the composer's OWN successful flights.

    The token model is trained the way a language model is post-trained: sample,
    keep what worked, and raise the likelihood of exactly those tokens.  No
    advantage, no value function, no ratio.

    Why this suits the problem.  A policy gradient needs to know WHICH decision
    in a flight mattered, and measured on a real batch it does not: the
    advantage separated crashed from surviving flights at z = -166 while its
    correlation with the braking argument was -0.013, because every decision in
    a flight carries essentially the same return (consecutive decisions differ
    by 3 against a spread of 241).  Imitation never asks that question.  The
    outcome is used as a FILTER over whole flights, and within a kept flight
    every token is simply made more likely.

    What it gives up: it cannot learn from failure, only from success, so it
    needs a supply of flights that already arrive -- which is what the
    difficulty curriculum provides, and why the two belong together.  It also
    reinforces whatever the successful flights happened to do, including the
    parts that were irrelevant.

    Selection is not optional polish.  Imitation only improves a policy when
    the kept set is better than the average, and with ~90% of flights arriving
    an arrival filter admits almost everything: cross-entropy on 90% of your
    own behaviour is a fixed point, the loss settles at the policy's own
    entropy and nothing moves.

    TWO ways to make the kept set better than average, both ranked by `score`
    (lower is better -- the fraction of the episode the flight needed to
    finish, which is 1.0 exactly when it never arrived):

    `keep_frac` < 1 is the HARD cut: keep the fastest fraction, discard the
    rest.  `weight_tau` > 0 is the SOFT version and takes precedence -- every
    arrival contributes, weighted by how quickly it arrived, and the weight
    multiplies its tokens' cross-entropy.  The weight is

        w_i = exp(-z_i / tau) / mean(exp(-z / tau)),   z = (t - mean t) / sd t

    in units of the BATCH's own spread of arrival times, so it needs no
    retuning as the policy gets faster.  At tau = 1 the effective sample size
    is e^-1 = 37% of the arrivals, so it is about as selective as the 30% cut
    it replaces, but continuous: a flight one standard deviation quicker counts
    2.7x, instead of everything above the 70th percentile counting the same and
    everything below counting zero.  The cliff was throwing away the gradient
    from two thirds of the arrivals and quantizing the rest.

    Either way this adds no TERM to the objective -- it is a choice of which
    successes to copy and how much, and arriving sooner is already implicit in
    the goal.

    What neither can do: a cross-entropy weight is non-negative, so the update
    can only move probability AROUND among flights that arrived.  A flight that
    never arrived has no arrival time and gets weight zero.  Nothing here can
    reduce the failure rate; it can only make the successes quicker.  Moving
    the failures needs them to become successes in the data (fly each task
    more than once and keep its best) or a loss that can push probability down.
    """
    groups = list(zip(records, reached)) if (records and isinstance(records[0], list)) else [(records, reached)]
    samples, acts, us, nargs, wts = [], [], [], [], []
    flight_w = []
    n_flights = n_kept = 0
    scores = score if score is not None else [None] * len(groups)
    if not isinstance(scores, (list, tuple)):
        scores = [scores]
    tasks = task if task is not None else [None] * len(groups)
    if not isinstance(tasks, (list, tuple)):
        tasks = [tasks]
    # The weights are computed over ALL groups at once, because a group is a
    # rollout SHARD (or one of k repeats of the same task list) and a task's
    # samples are scattered across them -- centring within a group would centre
    # within a shard, which is not a task.
    w_split = [None] * len(groups)
    if score is not None and (weight_tau > 0.0 or signed):
        lens = [int(scores[gi].numel()) for gi in range(len(groups))]
        t_all = torch.cat([scores[gi].reshape(-1).to(torch.float64) for gi in range(len(groups))])
        k_all = (torch.cat([tasks[gi].reshape(-1).to(torch.long) for gi in range(len(groups))])
                 if all(tk is not None for tk in tasks[:len(groups)]) else None)
        a_all = torch.cat([groups[gi][1].reshape(-1).to(torch.bool) for gi in range(len(groups))])
        w_split = list(torch.split(arrival_weights(t_all, k_all, tau=max(weight_tau, 1e-9),
                                                   signed=signed, w_max=w_max,
                                                   arrived=a_all), lens))
    for gi, (recs, win) in enumerate(groups):
        # `advantage`, when given, IS the weight: each flight's paired
        # counterfactual against its muted twin.  Every decision of that flight
        # carries it, because with one or two decisions a flight there is
        # nothing finer to attribute to and no honest way to split it.
        adv = advantage[gi] if (advantage is not None and gi < len(advantage)) else None
        w = win.to(torch.bool)
        sc = scores[gi] if gi < len(scores) else None
        wt = w.to(torch.float64)
        if adv is not None:
            wt = adv.to(torch.float64).reshape(-1)
            w = wt != 0                      # a token that changed nothing teaches nothing
        elif w_split[gi] is not None:
            wt = w_split[gi]
            w = wt != 0
        elif sc is not None and keep_frac < 1.0 and int(w.sum()) > 1:
            # HARD: among the flights that arrived, keep the fastest `keep_frac`
            idx = w.nonzero().flatten()
            k = max(1, int(round(keep_frac * idx.numel())))
            best = idx[sc[idx].argsort()[:k]]
            w = torch.zeros_like(w); w[best] = True
            wt = w.to(torch.float64)
        n_flights += int(win.numel()); n_kept += int(w.sum())
        flight_w.append(wt[w])
        for di, rec in enumerate(recs):
            al = rec["alive"]
            rows = rec.get("rows")
            for j in al.nonzero().flatten().tolist():
                b = int(rows[j]) if rows is not None else j
                if not bool(w[b]):
                    continue                      # weight zero: nothing to learn from it
                samples.append({kk: (v[j].float() if torch.is_tensor(v) and v.is_floating_point()
                                     else (v[j] if torch.is_tensor(v) else v))
                                for kk, v in rec["tok"].items()})
                acts.append(rec["act"][j]); us.append(rec["u"][j]); nargs.append(rec["n_args"][j])
                wts.append(float(wt[b]))
    if not samples:
        return {"n": 0, "kept_flights": 0, "flights": n_flights, "ess": 0.0}
    gen = torch.Generator().manual_seed(0)
    if max_samples and len(samples) > max_samples:
        pick = torch.randperm(len(samples), generator=gen)[:max_samples].sort().values.tolist()
        samples = [samples[i] for i in pick]
        acts = [acts[i] for i in pick]; us = [us[i] for i in pick]; nargs = [nargs[i] for i in pick]
        wts = [wts[i] for i in pick]
    acts = torch.stack(acts).long(); us = torch.stack(us).float(); nargs = torch.stack(nargs).long()
    wts = torch.tensor(wts, dtype=torch.float32)
    tok_all = collate_tok(samples)
    n = acts.shape[0]
    opt = opt or torch.optim.Adam(net.parameters(), lr=lr)
    acc = {"ce": 0.0, "nll": 0.0, "acc": 0.0, "nb": 0}
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        for s0 in range(0, n, batch):
            idx = order[s0:s0 + batch]
            tk = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in tok_all.items()}
            logits, mu_all, _ = net.pre(tk)
            if not (torch.isfinite(logits).all() and torch.isfinite(mu_all).all()):
                continue
            # the arguments belong to the token that was CHOSEN: the head emits
            # a set per token type, so pick that token's row
            mu = mu_all[torch.arange(mu_all.shape[0], device=mu_all.device), acts[idx]]
            # ONE term: the negative log-likelihood of the decision that was
            # taken -- cross-entropy over the token type and the density of its
            # arguments, which is the same quantity written once.  No value
            # loss, no entropy bonus, no KL penalty, no auxiliary anything.
            # Everything the composer should care about -- not crashing, not
            # dawdling, not placing needless subgoals -- is IMPLICIT in which
            # flights got kept to imitate.
            # ONE term still: the same negative log-likelihood, each decision
            # carrying its flight's arrival-time weight.  Normalised by the
            # weights in the minibatch, so the loss and its gradient keep the
            # scale an unweighted mean would have.
            lw = wts[idx]
            lp = cont_log_prob(logits, mu, net.log_std, acts[idx], us[idx], nargs[idx])
            # Normalise by the total WEIGHT MASS, not the signed sum.  Signed
            # weights are centred, so their sum is ~0 and dividing by it (even
            # clamped at 1e-9) sends the loss to infinity: measured -746,178 on
            # the very first iteration of the signed arm.  |w| is the sum for
            # unsigned weights, so the unsigned path is unchanged.
            loss = -(lw * lp).sum() / lw.abs().sum().clamp_min(1e-9)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                acc["ce"] += float(loss)
                acc["nll"] += float(nn.functional.cross_entropy(logits, acts[idx]))   # reported, not optimised
                _m = (logits.argmax(-1) == acts[idx]).float()
                acc["acc"] += float((lw.abs() * _m).sum() / lw.abs().sum().clamp_min(1e-9))
                acc["nb"] += 1
    nb = max(1, acc["nb"])
    with torch.no_grad():
        p = torch.softmax(net.pre({kk: (v[:512] if torch.is_tensor(v) else v)
                                   for kk, v in tok_all.items()})[0], -1).mean(0)
    fw = torch.cat(flight_w) if flight_w else torch.zeros(0)
    # effective flights behind the update.  For a SIGNED weight the sum is ~0
    # by construction, so that ratio says nothing; report the split instead.
    ess = float(fw.sum() ** 2 / (fw * fw).sum().clamp_min(1e-12)) if (fw.numel() and not signed) else 0.0
    return {"n": n, "flights": n_flights, "kept_flights": n_kept, "ess": ess,
            "w_up": float((fw > 0).double().mean()) if fw.numel() else 0.0,
            "w_abs": float(fw.abs().mean()) if fw.numel() else 0.0,
            "ce": acc["ce"] / nb, "nll": acc["nll"] / nb, "match": acc["acc"] / nb,
            "kl": 0.0, "clipfrac": 0.0, "entropy": float(cont_entropy(
                torch.log(p.clamp_min(1e-9))[None], net.log_std.detach()).mean()),
            "speak": float(1.0 - p[net.vocab.EOS]), "std": float(net.log_std.detach().exp().mean()),
            "ev": float("nan")}


def ppo_update_cont(net: ContPolicyNet, records: List[Dict], returns: Tensor, n_terms: int,
                    epochs: int = 2, batch: int = 1024, lr: float = 1e-4, clip: float = 0.2,
                    vcoef: float = 0.5, ent: float = 0.0, target_kl: float = 0.02,
                    opt=None, max_samples: int = 0) -> Dict[str, float]:
    """Clipped surrogate over `[type, arguments]`.

    The ratio is taken against the policy AT COLLECTION (`pi_logits`, `mu`,
    `log_std`), not against the behaviour distribution the exploration widened,
    which is the same anchoring the token composer needed: clipping around the
    behaviour let a widened draw move the mean much further than the clip
    suggests.

    `target_kl` is a real cap here rather than a formality -- the Gaussian's
    mean is unbounded and a single oversized step does not come back.
    """
    groups = list(zip(records, returns)) if (records and isinstance(records[0], list)) else [(records, returns)]
    samples, acts, us, nargs, rets, plog, pmu, plsd = [], [], [], [], [], [], [], []
    for recs, R in groups:
        for k, rec in enumerate(recs):
            al = rec["alive"]
            rows = rec.get("rows")
            for j in al.nonzero().flatten().tolist():
                b = int(rows[j]) if rows is not None else j
                samples.append({kk: (v[j].float() if torch.is_tensor(v) and v.is_floating_point()
                                     else (v[j] if torch.is_tensor(v) else v))
                                for kk, v in rec["tok"].items()})
                acts.append(rec["act"][j]); us.append(rec["u"][j]); nargs.append(rec["n_args"][j])
                rets.append(R[k, b]); plog.append(rec["pi_logits"][j]); pmu.append(rec["mu"][j])
                plsd.append(rec["log_std"])
    if not samples:
        return {"n": 0}
    gen = torch.Generator().manual_seed(0)
    if max_samples and len(samples) > max_samples:
        pick = torch.randperm(len(samples), generator=gen)[:max_samples].sort().values.tolist()
        take = lambda a: [a[i] for i in pick]
        samples, acts, us, nargs, rets, plog, pmu, plsd = map(
            take, (samples, acts, us, nargs, rets, plog, pmu, plsd))
    acts = torch.stack(acts).long(); us = torch.stack(us).float()
    nargs = torch.stack(nargs).long(); rets = torch.stack(rets).float()
    old_logits = torch.stack(plog).float(); old_mu = torch.stack(pmu).float()
    old_lsd = torch.stack(plsd).float()
    # the chain grows over a flight and is capped, so records carry different
    # chain lengths; `collate_tok` pads them into one batch (the same helper
    # the token update uses)
    tok_all = collate_tok(samples)
    n = acts.shape[0]
    # NORMALISED returns, and the value head is fit to THOSE.  Fitting it to raw
    # returns (spread ~230 here) left it at its initialisation -- explained
    # variance 0.000 -- so it predicted a constant and absorbed nothing.
    rets_n = (rets - rets.mean()) / rets.std().clamp_min(1e-6)
    old_lp = cont_log_prob(old_logits, old_mu, old_lsd, acts, us, nargs).detach()
    opt = opt or torch.optim.Adam(net.parameters(), lr=lr)
    acc = {"loss": 0.0, "v_loss": 0.0, "ent": 0.0, "clipfrac": 0.0, "nb": 0, "kl": 0.0}
    stop = False
    for _ in range(epochs):
        if stop:
            break
        order = torch.randperm(n, generator=gen)
        for s0 in range(0, n, batch):
            idx = order[s0:s0 + batch]
            tk = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in tok_all.items()}
            logits, mu_all, val = net.pre(tk)
            if not (torch.isfinite(logits).all() and torch.isfinite(mu_all).all()):
                continue
            mu = mu_all[torch.arange(mu_all.shape[0], device=mu_all.device), acts[idx]]
            lp = cont_log_prob(logits, mu, net.log_std, acts[idx], us[idx], nargs[idx])
            ratio = (lp - old_lp[idx]).clamp(-20.0, 20.0).exp()
            # THE STATE BASELINE.  Without subtracting V(s) the advantage is the
            # raw return, which is dominated by whether the flight eventually
            # crashed: measured on a real batch, the advantage separated crashed
            # from surviving flights at z = -166 while its correlation with the
            # braking argument was -0.013.  Every decision in a bad flight was
            # pushed down equally, so the gradient carried almost no information
            # about what the composer actually chose.
            a = rets_n[idx] - val.detach()
            a = (a - a.mean()) / a.std().clamp_min(1e-6)
            loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            v_loss = ((val - rets_n[idx]) ** 2).mean()
            h = cont_entropy(logits, net.log_std).mean()
            (loss + vcoef * v_loss - ent * h).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                kl = float((old_lp[idx] - lp).mean())
                acc["kl"] += kl
                acc["clipfrac"] += float(((ratio - 1).abs() > clip).float().mean())
                acc["loss"] += float(loss); acc["v_loss"] += float(v_loss); acc["ent"] += float(h)
                acc["nb"] += 1
            if target_kl > 0 and acc["nb"] and acc["kl"] / acc["nb"] > target_kl:
                stop = True                      # the cap the last continuous run did not have
                break
    nb = max(1, acc["nb"])
    with torch.no_grad():
        sel = torch.randperm(n, generator=gen)[:2048]
        tk = {k: (v[sel] if torch.is_tensor(v) else v) for k, v in tok_all.items()}
        v0 = net.pre(tk)[2]
        ev = float(1.0 - (rets_n[sel] - v0).var() / rets_n[sel].var().clamp_min(1e-9))
    p = torch.softmax(old_logits, -1).mean(0)
    return {"n": n, "kl": acc["kl"] / nb, "clipfrac": acc["clipfrac"] / nb,
            "loss": acc["loss"] / nb, "v_loss": acc["v_loss"] / nb, "entropy": acc["ent"] / nb,
            "speak": float(1.0 - p[net.vocab.EOS]), "std": float(net.log_std.detach().exp().mean()), "ev": ev,
            "stopped_early": bool(stop)}


COMPOSERS["policy_cont"] = ContComposer
