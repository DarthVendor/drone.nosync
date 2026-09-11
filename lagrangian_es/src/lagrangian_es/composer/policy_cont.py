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
        super().__init__(n_terms, **kw)
        d = self.embed.out_features
        self.vocab = ContVocab(n_terms)
        V, k = self.vocab.V, self.vocab.n_args
        self.act_emb = nn.Embedding(V, d)
        self.head_act = nn.Linear(d, V)
        self.head_arg = nn.Linear(d, k)
        # One std per argument slot, shared across states and learned.  A
        # state-dependent std is the usual next step and the usual way a
        # continuous policy collapses, so it is not the place to start.
        # Small on purpose.  The policy is Gaussian in the UNSQUASHED variable,
        # and near the goal bearing the squash is locally linear, so a std of
        # 0.35 there is about +-60 degrees of bearing on every waypoint -- the
        # prior was firing the vehicle off in random directions and nothing
        # arrived.  0.12 is about +-20 degrees; the update widens it if the
        # return asks for it.
        self.log_std = nn.Parameter(torch.full((k,), math.log(0.12)))
        nn.init.normal_(self.act_emb.weight, std=0.02)
        nn.init.zeros_(self.head_act.weight)
        nn.init.zeros_(self.head_arg.weight)
        with torch.no_grad():
            # The prior is SILENCE, as in the token composer: the subgoal stays
            # at the goal, which is the bare controller flying the straight
            # line, and everything the composer does is a learned deviation
            # from it.  A fresh head that spoke every report killed every
            # flight the first time this was tried.
            self.head_act.bias.zero_()
            self.head_act.bias[self.vocab.EOS] = 6.0
            self.head_act.bias[self.vocab.WAYPOINT] = 4.0
            # r near its maximum: with the radius already capped at the distance
            # to the goal, r=1 puts the subgoal ON the goal, which is the
            # identity -- the bare controller flying the straight line.
            self.head_arg.bias.zero_()
            self.head_arg.bias[0] = 2.0          # tanh(2) = 0.96 of the radius

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
        mu = self.head_arg(q)
        g = self.goal_ego(tok).to(mu.dtype)
        bearing = torch.atan2(g[:, 1], g[:, 0]) / math.pi              # [-1, 1]
        mu = mu + torch.nn.functional.pad(torch.atanh(bearing.clamp(-0.999, 0.999))[:, None],
                                          (1, max(0, mu.shape[-1] - 2)))
        return self.head_act(q), mu, self.value(q).squeeze(-1)


class ContComposer(PolicyComposer):
    """Samples `[type, arguments]` per component and records both."""

    kind = "policy_cont"

    def __init__(self, system, trainable, **kw):
        super().__init__(system, trainable, **kw)
        # `PolicyComposer` built the token net; replace it with the typed one
        # and re-apply the goal scaling it had set
        self.net = ContPolicyNet(max(self.n_terms, 1), d=kw.get("d", 64),
                                 heads=kw.get("heads", 4)).to(self.net_dtype)
        self.net.goal_gain = self.tok.scale / self.reach
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
                V.step(act, V.squash(u), out, pend, has, psi, rows=idx)
                self._remember(ctx_s, act)
                tok_cur, cur_pos, last_act = tok, idx, act
                active[idx[act == V.EOS]] = False
        finally:
            self._rows = saved
        return V.finish(out, pend, has, x, goal, psi, g_ego, self.reach, self.z_min)

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
            std = self.net.log_std.exp().to(mu.dtype)
            if not self.stochastic:
                act = logits.argmax(-1)
                u = mu
            else:
                act = torch.distributions.Categorical(logits=b_logits).sample()
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
    adv = (rets - rets.mean()) / rets.std().clamp_min(1e-6)
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
            logits, mu, val = net.pre(tk)
            if not (torch.isfinite(logits).all() and torch.isfinite(mu).all()):
                continue
            lp = cont_log_prob(logits, mu, net.log_std, acts[idx], us[idx], nargs[idx])
            ratio = (lp - old_lp[idx]).clamp(-20.0, 20.0).exp()
            a = adv[idx]
            loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            v_loss = ((val - rets[idx]) ** 2).mean()
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
    p = torch.softmax(old_logits, -1).mean(0)
    return {"n": n, "kl": acc["kl"] / nb, "clipfrac": acc["clipfrac"] / nb,
            "loss": acc["loss"] / nb, "v_loss": acc["v_loss"] / nb, "entropy": acc["ent"] / nb,
            "speak": float(1.0 - p[net.vocab.EOS]), "std": float(net.log_std.detach().exp().mean()),
            "stopped_early": bool(stop)}


COMPOSERS["policy_cont"] = ContComposer
