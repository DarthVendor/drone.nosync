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

from .actions_cont import ContVocab, entropy as cont_entropy, log_prob as cont_log_prob
from .base import COMPOSERS
from .policy import PolicyComposer, PolicyNet, collate_tok
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
        self.arg_w2 = nn.Parameter(torch.zeros(V, H, k)); self.arg_b2 = nn.Parameter(torch.zeros(V, k))
        nn.init.normal_(self.arg_w1, std=(1.0 / d) ** 0.5)
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
        nn.init.zeros_(self.head_act[-1].weight)      # the prior lives in the last layer's bias
        with torch.no_grad():
            # The prior is SILENCE, as in the token composer: the subgoal stays
            # at the goal, which is the bare controller flying the straight
            # line, and everything the composer does is a learned deviation
            # from it.  A fresh head that spoke every report killed every
            # flight the first time this was tried.
            self.head_act[-1].bias.zero_()
            self.head_act[-1].bias[self.vocab.EOS] = 6.0
            self.head_act[-1].bias[self.vocab.WAYPOINT] = 4.0
            # r: NOT at its maximum.  A bias of 2.0 put the subgoal on the goal,
            # which is the identity and reads as the right prior, but tanh(2.0)
            # sits in the flat region -- slope 0.07 instead of 1.0.  Measured
            # over 27,088 decisions, 100% of them landed there, the sampled
            # range of r collapsed to 0.017, and every flight emitted r = 0.963
            # to three decimals.  The braking end of the lever (r = -1, a full
            # stop) was 17 standard deviations away and was never sampled once,
            # so no filter could select it and no imitation could learn it.
            #
            # 0.7 puts the waypoint at ~80% of the distance to the goal -- still
            # essentially the straight line, since it re-decides every 0.4 s --
            # while the slope rises to 0.63 and the reachable range of r by a
            # factor of six.  A prior that cannot be moved is worse than a prior
            # that is slightly off.
            # only the WAYPOINT's own r slot; every other token starts at zero
            self.arg_b2.zero_()
            self.arg_b2[self.vocab.WAYPOINT, 0] = 0.7   # tanh(0.7)=0.60 -> 80% of the radius

    def _chain_in(self, tok):
        """The chain embedding, with an instruction's ARGUMENTS folded into the
        token's own vector rather than left as bare numbers in shared slots."""
        from .transformer import ACT_SCALE, INSTR
        ch = self.embed(tok["chain"]) + self.type_emb(tok["chain_types"])
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
        # the goal-bearing residual belongs to the WAYPOINT's theta and nowhere
        # else, so a turn or a priority move is not offset by where the goal is
        mu = mu.clone()
        mu[:, self.vocab.WAYPOINT, 1] = mu[:, self.vocab.WAYPOINT, 1] + \
            torch.atanh(bearing.clamp(-0.999, 0.999))
        return self.head_act(q), mu, self.value(q).squeeze(-1)


class ContComposer(PolicyComposer):
    """Samples `[type, arguments]` per component and records both."""

    kind = "policy_cont"

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
                    if chain.shape[1] > kc:
                        chain, ctypes, cmask = chain[:, -kc:], ctypes[:, -kc:], cmask[:, -kc:]
                    tok["chain"], tok["chain_types"], tok["chain_mask"] = chain, ctypes, cmask
                    sc = (scene0[0][idx], scene0[1][idx])
                self._rows = ids_full[idx]
                logits, mu, _ = self.net.pre(tok, scene=sc)
                if step == V.L_MAX:                       # the cap: only EOS is left
                    forced = torch.full_like(logits, float("-inf"))
                    forced[:, V.EOS] = 0.0
                    logits = forced
                ctx_s = {"alive": alive0[idx], "t": t_now}
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
                    "tok_keep": None,
                    "tok": {k: (v[rec].clone() if torch.is_tensor(v) and v.ndim and v.shape[0] == B
                                else (v.clone() if torch.is_tensor(v) else v)) for k, v in tok.items()}})
            return act, u

        return self._decide_cont(ctx, choose)


def imitate_update(net: ContPolicyNet, records: List[Dict], reached: List[Tensor],
                   epochs: int = 2, batch: int = 1024, lr: float = 1e-4,
                   opt=None, max_samples: int = 0, keep_frac: float = 1.0,
                   score: Optional[List[Tensor]] = None) -> Dict[str, float]:
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

    `keep_frac` < 1 keeps only the BEST fraction of the successful flights,
    ranked by `score` (lower is better -- the fraction of the episode they
    needed to finish).  This is not optional polish.  Imitation only improves a
    policy when the kept set is better than the average, and with ~90% of
    flights arriving the filter admitted almost everything: cross-entropy on
    90% of your own behaviour is a fixed point, the loss settles at the policy's
    own entropy and nothing moves.  Selecting the fastest arrivals gives the
    update something to climb toward.

    Ranking by time-to-arrive adds no term to the objective -- it is a choice
    of WHICH successes to copy, and arriving sooner is already implicit in the
    goal.
    """
    groups = list(zip(records, reached)) if (records and isinstance(records[0], list)) else [(records, reached)]
    samples, acts, us, nargs = [], [], [], []
    n_flights = n_kept = 0
    scores = score if score is not None else [None] * len(groups)
    if not isinstance(scores, (list, tuple)):
        scores = [scores]
    for gi, (recs, win) in enumerate(groups):
        w = win.to(torch.bool)
        sc = scores[gi] if gi < len(scores) else None
        if sc is not None and keep_frac < 1.0 and int(w.sum()) > 1:
            # among the flights that arrived, keep the fastest `keep_frac`
            idx = w.nonzero().flatten()
            k = max(1, int(round(keep_frac * idx.numel())))
            best = idx[sc[idx].argsort()[:k]]
            w = torch.zeros_like(w); w[best] = True
        n_flights += int(win.numel()); n_kept += int(w.sum())
        for rec in recs:
            al = rec["alive"]
            rows = rec.get("rows")
            for j in al.nonzero().flatten().tolist():
                b = int(rows[j]) if rows is not None else j
                if not bool(w[b]):
                    continue                      # only flights that arrived
                samples.append({kk: (v[j].float() if torch.is_tensor(v) and v.is_floating_point()
                                     else (v[j] if torch.is_tensor(v) else v))
                                for kk, v in rec["tok"].items()})
                acts.append(rec["act"][j]); us.append(rec["u"][j]); nargs.append(rec["n_args"][j])
    if not samples:
        return {"n": 0, "kept_flights": 0, "flights": n_flights}
    gen = torch.Generator().manual_seed(0)
    if max_samples and len(samples) > max_samples:
        pick = torch.randperm(len(samples), generator=gen)[:max_samples].sort().values.tolist()
        samples = [samples[i] for i in pick]
        acts = [acts[i] for i in pick]; us = [us[i] for i in pick]; nargs = [nargs[i] for i in pick]
    acts = torch.stack(acts).long(); us = torch.stack(us).float(); nargs = torch.stack(nargs).long()
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
            loss = -cont_log_prob(logits, mu, net.log_std, acts[idx], us[idx], nargs[idx]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                acc["ce"] += float(loss)
                acc["nll"] += float(nn.functional.cross_entropy(logits, acts[idx]))   # reported, not optimised
                acc["acc"] += float((logits.argmax(-1) == acts[idx]).float().mean())
                acc["nb"] += 1
    nb = max(1, acc["nb"])
    with torch.no_grad():
        p = torch.softmax(net.pre({kk: (v[:512] if torch.is_tensor(v) else v)
                                   for kk, v in tok_all.items()})[0], -1).mean(0)
    return {"n": n, "flights": n_flights, "kept_flights": n_kept,
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
