"""Training the composer against the objective, at its own timescale.

There is no teacher.  The composer acts every few steps on what the drone
perceives and what its stream reports, and is rewarded with the rollout's own
cost -- the same cost the low level was evolved on, read off the measurement
tokens -- so nothing about "how to navigate" is written into a reward.  The
decision problem is short (a few hundred ticks) and the low level absorbs the
dynamics, which is what makes a plain clipped policy gradient with a value
baseline enough.

`PolicyComposer` samples every head with a learned log-std and records what it
saw and did; `ppo_update` fits the recorded stream.  Deterministic mode (the
mean) is what gets judged.
"""
from __future__ import annotations

from typing import Dict, List

import math

import torch
from torch import Tensor, nn

from .base import COMPOSERS
from .distill import collate
from .spec import TaskSpec
from .tokens import to_world
from .transformer import ComposerNet, TransformerComposer


class PolicyNet(ComposerNet):
    def __init__(self, n_terms, **kw):
        super().__init__(n_terms, **kw)
        d = self.head_sub.in_features
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # Exploration scale, MEASURED: on identical city legs the identity prior
        # flies 0.354 reach / 0.646 crash deterministically and at a pre-
        # activation std of 0.37 collapses to 0.000 / 0.875, at 0.14 to
        # 0.042 / 0.708, at 0.05 keeps 0.354 / 0.646.  So it starts at 0.05;
        # it is a parameter, and may grow if exploring further ever pays.
        self.log_std = nn.Parameter(torch.full((3 + 2 * n_terms + 2,), -3.0))   # + heading delta, heading gate

    def pre(self, tok, store=None):
        """Pre-activations of every head, and the value, from the read tokens.
        `store` collects attention weights for the explainer."""
        B = tok["self"].shape[0]; te = self.type_emb
        scene = torch.cat([self.embed(tok["self"])[:, None] + te.weight[0],
                           self.embed(tok["goal"])[:, None] + te.weight[1],
                           self.embed(tok["entities"]) + te(tok["ent_types"])], 1)
        smask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=scene.device), ~tok["ent_mask"]], 1)
        for blk in self.scene:
            scene = blk(scene, mask=smask, store=None if store is None else store.setdefault("scene", {}))
        mem, mmask = scene, smask
        if tok["chain"].shape[1]:
            ch = self.embed(tok["chain"]) + te(tok["chain_types"])
            L = ch.shape[1]
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ch.device), 1)
            for i, blk in enumerate(self.chain):
                ch = blk(ch, attn_mask=causal, store=None if store is None else store.setdefault("chain", {}),
                         last=i == len(self.chain) - 1)
            mem = torch.cat([mem, ch[:, -1:]], 1)
            mmask = torch.cat([mmask, torch.zeros(B, 1, dtype=torch.bool, device=ch.device)], 1)
        q = torch.cat([self.pool.expand(B, -1, -1), self.constraint.expand(B, -1, -1)], 1)
        for blk in self.read:
            q = blk(q, mem=mem, mem_mask=mmask, store=None if store is None else store.setdefault("read", {}))
        q = self.ln(q)
        h_sub = self.head_sub(q[:, 0]) + tok["goal"][:, :3] * self.goal_gain   # residual on the goal
        aw = self.head_w(q[:, 1:])                           # [B, n, 2]
        hy = self.head_yaw(q[:, 0])                          # [B, 2]
        pre = torch.cat([h_sub, aw[..., 0], aw[..., 1], hy], -1)   # [B, 3 + 2n + 2]
        return pre, self.value(q[:, 0]).squeeze(-1)

    @staticmethod
    def activate(pre, n):
        h = pre[:, :3]
        nrm = h.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        sub = torch.tanh(nrm) * h / nrm
        alpha = nn.functional.softplus(pre[:, 3:3 + n]) + 1e-3
        gate = torch.sigmoid(pre[:, 3 + n:3 + 2 * n])
        yaw = (math.pi * torch.tanh(pre[:, 3 + 2 * n]), torch.sigmoid(pre[:, 3 + 2 * n + 1]))
        return sub, alpha, gate, yaw

    LOG_STD_MIN = -3.5      # a floor at the measured harmless scale: sigma may grow, not vanish

    @property
    def std(self):
        return self.log_std.clamp_min(self.LOG_STD_MIN).exp()

    def dist(self, pre):
        return torch.distributions.Normal(pre, self.std)


class PolicyComposer(TransformerComposer):
    """Stochastic in training, the mean when judged; records every decision."""
    kind = "policy"

    def __init__(self, system, trainable, **kw):
        # the checkpoint is for THIS net; the base class must not try to load
        # it into the plain ComposerNet it builds first (a policy checkpoint
        # carries the value head and log_std, and the base load rejected it)
        weights = kw.pop("weights", "")
        super().__init__(system, trainable, **kw)
        kw["weights"] = weights
        n = max(self.n_terms, 1)
        self.net = PolicyNet(n, d=kw.get("d", 64), heads=kw.get("heads", 4)).to(self.net_dtype)
        self.net.goal_gain = self.tok.scale / self.reach
        self.weights_path = kw.get("weights", "")
        if kw.get("weights"):
            import os
            self.net.load_state_dict(torch.load(kw["weights"], map_location="cpu"))
            self._weights_mtime = os.path.getmtime(kw["weights"])
        self.net.eval()
        self.stochastic = False
        self.records: List[Dict] = []          # per decision: tokens (per episode), action, t
        # Noise is HELD for `noise_hold` decisions (1 s at a 10-step interval):
        # a fresh draw every 0.2 s is jitter the hold smooths into a random
        # walk, while a held perturbation explores the same distance without
        # compounding into a crash.
        self.noise_hold = int(kw.get("noise_hold", 5))
        self._eps = None; self._eps_age = 0
        # Rows to record for the update, by full-batch id; None = every row.
        # Set by the worker before a batch: recording every live row's full
        # token set at every decision and slimming at the end held ~700 MB per
        # worker for nothing, on a machine with 2 GB to spare.
        self.record_rows = None

    def reset(self, B):
        super().reset(B)
        self._eps = None; self._eps_age = 0

    @torch.no_grad()
    def emit(self, ctx):
        tok = self.tokens(ctx)
        pre, _ = self.net.pre(tok)
        if self.stochastic:
            rows = getattr(self, "_rows", None)
            B = self._B if rows is not None else pre.shape[0]
            if self._eps is None or self._eps_age >= self.noise_hold or self._eps.shape[0] != B:
                self._eps = torch.randn(B, pre.shape[-1], dtype=pre.dtype, device=pre.device); self._eps_age = 0
            self._eps_age += 1
            eps = self._eps if rows is None else self._eps[rows]
            act = pre + self.net.std * eps
        else:
            act = pre
        sub, alpha, gate, (dpsi, yg) = self.net.activate(act, max(self.n_terms, 1))
        sub, alpha, gate, dpsi, yg = (v.to(self.dtype) for v in (sub, alpha, gate, dpsi, yg))
        x, goal, psi = ctx["x"], ctx["goal"], tok["psi"].to(self.dtype)
        sub_world = x + to_world(sub * self.reach, psi)
        sub_world = torch.cat([sub_world[:, :2], sub_world[:, 2:].clamp_min(self.z_min)], -1)
        spec = TaskSpec(delta=sub_world - goal, alpha=alpha[:, :self.n_terms], gate=gate[:, :self.n_terms],
                        yaw=psi + dpsi, yaw_gate=yg)
        self._remember(ctx, spec)
        if self.stochastic:
            rows = getattr(self, "_rows", None)
            ids = rows.clone() if rows is not None else torch.arange(act.shape[0])
            keep = torch.ones(ids.shape[0], dtype=torch.bool) if self.record_rows is None \
                else torch.isin(ids, self.record_rows)
            self.records.append({"t": float(ctx.get("t", 0)), "act": act[keep].clone(),
                                 "alive": ctx["alive"][keep].clone(), "rows": ids[keep],
                                 "tok": {k: (v[keep].clone() if torch.is_tensor(v) and v.ndim and v.shape[0] == act.shape[0]
                                             else (v.clone() if torch.is_tensor(v) else v)) for k, v in tok.items()}})
        return spec


def returns_from_stream(records: List[Dict], chain: List[Dict], gamma: float,
                        subgoal_cost: float = 0.0) -> Tensor:
    """Per decision and FULL-BATCH row, the discounted sum of the cost
    increments the stream reported after it -- negated, so lower cost is higher
    return.  Records may cover only the rows that were alive at that decision
    (`rows`); the return is computed for every row and the update picks its
    own rows out.  `subgoal_cost` is charged to a row for every decision it
    was given (every subgoal placed for it), so the composer is asked to
    reach the goal with the fewest."""
    ts = [r["t"] for r in records]
    cost_at = {m["t"]: m["cost"] for m in chain}
    times = sorted(cost_at)
    B = chain[0]["cost"].shape[0] if chain else records[0]["alive"].shape[0]
    zero = torch.zeros(B, dtype=chain[0]["cost"].dtype if chain else records[0]["act"].dtype)
    # cost increment attributed to decision k: cost(next decision time) - cost(this one)
    def cost_at_or_before(t):
        prev = [u for u in times if u <= t]
        return cost_at[prev[-1]] if prev else zero
    last = cost_at[times[-1]] if times else zero
    R = torch.zeros(len(records), B, dtype=zero.dtype)
    G = torch.zeros(B, dtype=zero.dtype)
    for k in range(len(records) - 1, -1, -1):
        c0 = cost_at_or_before(ts[k])
        c1 = cost_at_or_before(ts[k + 1]) if k + 1 < len(records) else last
        r = -(c1 - c0)
        if subgoal_cost:
            rows = records[k].get("rows")
            r = r - subgoal_cost if rows is None else \
                r.index_add(0, rows.to(torch.long), torch.full((rows.numel(),), -subgoal_cost, dtype=r.dtype))
        G = r + gamma * G
        R[k] = G
    return R


def ppo_update(net: PolicyNet, records: List[Dict], returns: Tensor, n_terms: int,
               epochs: int = 4, batch: int = 512, lr: float = 3e-4, clip: float = 0.2,
               vcoef: float = 0.5, ent: float = 1e-3, gen=None,
               target_kl: float = 0.02, backtracks: int = 8) -> Dict[str, float]:
    """Fit the recorded stream: clipped surrogate, value regression, entropy.

    The step is sized in POLICY space.  After the epochs the exact KL from the
    collecting policy is measured over the whole batch; if it is beyond the
    trust region the weights go back to where they started and the step is
    halved, up to `backtracks` times (a failed attempt is cheap: the early-stop
    ends it within a minibatch or two).  Measured need: at sigma 0.05 a single
    Adam step at lr 2e-4 moved the policy by KL 0.13-0.5 (the update's first
    step moves every parameter by the full rate, and the zero-initialised
    heads make the network more sensitive to that as they grow), and seven
    such iterations took the judged reach from 0.25 to 0.  The returned `lr`
    is the rate that fit, for the caller to start from next time."""
    gen = gen or torch.Generator().manual_seed(0)
    # the update runs in float32 whatever the inference dtype: half gradients underflow
    infer_dtype = next(net.parameters()).dtype
    net.float()
    # `records` may be one list with one returns tensor, or several groups --
    # one per worker shard, each with its own decision count and kept rows
    groups = list(zip(records, returns)) if (records and isinstance(records[0], list)) else [(records, returns)]
    samples, acts, rets = [], [], []
    for recs, R in groups:
        for k, rec in enumerate(recs):
            al = rec["alive"]; rows = rec.get("rows")
            for j in al.nonzero().flatten().tolist():
                b = int(rows[j]) if rows is not None else j         # the column of R this record's row is
                samples.append({kk: (v[j].float() if torch.is_tensor(v) and v.is_floating_point() else (v[j] if torch.is_tensor(v) else v))
                                for kk, v in rec["tok"].items()})
                acts.append(rec["act"][j]); rets.append(R[k, b])
    if not samples:
        return {"n": 0}
    acts = torch.stack(acts).float(); rets = torch.stack(rets).float()
    # a non-finite action or return is dropped and counted before it can reach
    # a log-probability, which rejects it outright, or a gradient
    ok = torch.isfinite(acts).all(-1) & torch.isfinite(rets)
    dropped = int((~ok).sum())
    if dropped:
        keep = ok.nonzero().flatten().tolist()
        samples = [samples[i] for i in keep]; acts = acts[ok]; rets = rets[ok]
        if not samples:
            return {"n": 0, "nonfinite": dropped}
    rets_n = (rets - rets.mean()) / rets.std().clamp_min(1e-6)
    # Collate ONCE into padded tensors and minibatch by indexing.  Re-collating
    # 512 Python dicts per minibatch was most of a 159 s update on 52k samples.
    ALL = collate_tok(samples)
    def take(idx):
        return {k: (v[idx] if torch.is_tensor(v) else v) for k, v in ALL.items()}
    with torch.no_grad():
        old_pre = torch.cat([net.pre(take(torch.arange(i, min(i + batch, len(samples)))))[0]
                             for i in range(0, len(samples), batch)])
        old_std = net.std.detach().clone()
        old_lp = torch.distributions.Normal(old_pre, old_std).log_prob(acts).sum(-1)

    def kl_to_old(idx, pre):
        # the exact KL(old || new) of the diagonal Gaussians, not a sampled estimate
        return torch.distributions.kl_divergence(torch.distributions.Normal(old_pre[idx], old_std),
                                                 net.dist(pre)).sum(-1).mean()

    def batch_kl():
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(samples), batch):
                idx = torch.arange(i, min(i + batch, len(samples)))
                tot += float(kl_to_old(idx, net.pre(take(idx))[0])) * len(idx)
            return tot / len(samples)

    # The exploration scale gets its own learning rate.  Adam moves a
    # parameter by about `lr` per step whatever the gradient, so at the
    # network's rate `log_std` could change by at most ~0.7% an iteration and
    # ~4x over a whole run -- nominally learned, unable to adapt in practice.
    # Ten times the rate lets it move ~7% an iteration; the floor in `std`
    # still keeps it from vanishing.
    head = [p for n, p in net.named_parameters() if n == "log_std"]
    body = [p for n, p in net.named_parameters() if n != "log_std"]
    net.train()
    stats = {"n": len(samples), "nonfinite": dropped, "backtracks": 0}
    # value-head fit BEFORE the update: how much of the return the composer's
    # situation explains.  Near zero means the return is not varying with what
    # the composer sees and does, and no update can find a gradient in it.
    with torch.no_grad():
        v0 = torch.cat([net.pre(take(torch.arange(i, min(i + batch, len(samples)))))[1] for i in range(0, len(samples), batch)])
        stats["ev"] = float(1.0 - (rets_n - v0).var() / rets_n.var().clamp_min(1e-9))
    start = {k: v.detach().clone() for k, v in net.state_dict().items()}
    lr_used = lr
    for attempt in range(backtracks + 1):
        opt = torch.optim.Adam([{"params": body, "lr": lr_used}, {"params": head, "lr": 10 * lr_used}])
        acc = {"loss": 0.0, "v_loss": 0.0, "ent": 0.0, "clipfrac": 0.0, "nb": 0, "epochs": 0}
        stop = False
        for _ in range(epochs):
            if stop:
                break
            acc["epochs"] += 1
            perm = torch.randperm(len(samples), generator=gen)
            for i in range(0, len(samples), batch):
                idx = perm[i:i + batch]
                b = take(idx)
                pre, v = net.pre(b)
                d = net.dist(pre)
                lp = d.log_prob(acts[idx]).sum(-1)
                adv = rets_n[idx] - v.detach()
                adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
                ratio = (lp - old_lp[idx]).clamp(-20.0, 20.0).exp()      # a stale sample cannot overflow the update
                pg = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                with torch.no_grad():
                    acc["clipfrac"] += float(((ratio - 1).abs() > clip).to(ratio.dtype).mean())
                    # A STEP-SIZE limit, like the clip and the sigma floor, not
                    # a loss factor: once the policy has moved `target_kl` from
                    # where the batch was collected, further steps are steps on
                    # stale samples.
                    if target_kl and float(kl_to_old(idx, pre)) > target_kl:
                        stop = True
                vl = ((v - rets_n[idx]) ** 2).mean()
                e = d.entropy().sum(-1).mean()
                loss = pg + vcoef * vl - ent * e
                if not torch.isfinite(loss):
                    # a non-finite minibatch is skipped and counted, never stepped:
                    # one bad sample must not poison the weights
                    stats["nonfinite"] = stats.get("nonfinite", 0) + 1
                    continue
                if stop:
                    break
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                acc["loss"] += pg.item(); acc["v_loss"] += vl.item(); acc["ent"] += e.item(); acc["nb"] += 1
        kl = batch_kl()
        # The early-stop above lands the batch AT the target, give or take a
        # minibatch; a step too large to be sized by stopping lands well past
        # it (measured: 6-25x).  Only the latter is backtracked.
        if not target_kl or kl <= 2.0 * target_kl or attempt == backtracks:
            break
        # beyond what the samples can vouch for: back to the start, half the step
        net.load_state_dict(start); lr_used *= 0.5; stats["backtracks"] += 1
    net.eval(); net.to(infer_dtype)
    nb = max(acc.pop("nb"), 1)
    stats.update({k: (v / nb if k != "epochs" else v) for k, v in acc.items()})   # per-minibatch means
    stats["kl"] = kl                                     # exact, whole batch, after the step
    stats["lr"] = lr_used
    stats["sigma"] = float(net.std.mean().detach())
    return stats


def collate_tok(samples):
    """`distill.collate` without the target fields."""
    fake = [dict(s, y_sub=torch.zeros(3, dtype=s["self"].dtype), y_alpha=torch.zeros(1, dtype=s["self"].dtype),
                 y_gate=torch.zeros(1, dtype=s["self"].dtype)) for s in samples]
    return collate(fake)


COMPOSERS["policy"] = PolicyComposer
