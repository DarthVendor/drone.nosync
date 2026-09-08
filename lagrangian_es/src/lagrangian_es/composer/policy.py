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

    def pre(self, tok):
        """Pre-activations of every head, and the value, from the read tokens."""
        B = tok["self"].shape[0]; te = self.type_emb
        scene = torch.cat([self.embed(tok["self"])[:, None] + te.weight[0],
                           self.embed(tok["goal"])[:, None] + te.weight[1],
                           self.embed(tok["entities"]) + te(tok["ent_types"])], 1)
        smask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=scene.device), ~tok["ent_mask"]], 1)
        for blk in self.scene:
            scene = blk(scene, mask=smask)
        mem, mmask = scene, smask
        if tok["chain"].shape[1]:
            ch = self.embed(tok["chain"]) + te(tok["chain_types"])
            L = ch.shape[1]
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ch.device), 1)
            for blk in self.chain:
                ch = blk(ch, attn_mask=causal)
            mem = torch.cat([mem, ch[:, -1:]], 1)
            mmask = torch.cat([mmask, torch.zeros(B, 1, dtype=torch.bool, device=ch.device)], 1)
        q = torch.cat([self.pool.expand(B, -1, -1), self.constraint.expand(B, -1, -1)], 1)
        for blk in self.read:
            q = blk(q, mem=mem, mem_mask=mmask)
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
        self.net = PolicyNet(n, d=kw.get("d", 64), heads=kw.get("heads", 4)).to(system.dtype)
        self.net.goal_gain = self.tok.scale / self.reach
        if kw.get("weights"):
            self.net.load_state_dict(torch.load(kw["weights"], map_location="cpu"))
        self.net.eval()
        self.stochastic = False
        self.records: List[Dict] = []          # per decision: tokens (per episode), action, t
        # Noise is HELD for `noise_hold` decisions (1 s at a 10-step interval):
        # a fresh draw every 0.2 s is jitter the hold smooths into a random
        # walk, while a held perturbation explores the same distance without
        # compounding into a crash.
        self.noise_hold = int(kw.get("noise_hold", 5))
        self._eps = None; self._eps_age = 0

    def reset(self, B):
        super().reset(B)
        self._eps = None; self._eps_age = 0

    @torch.no_grad()
    def emit(self, ctx):
        tok = self.tokens(ctx)
        pre, _ = self.net.pre(tok)
        if self.stochastic:
            if self._eps is None or self._eps_age >= self.noise_hold or self._eps.shape[0] != pre.shape[0]:
                self._eps = torch.randn_like(pre); self._eps_age = 0
            self._eps_age += 1
            act = pre + self.net.std * self._eps
        else:
            act = pre
        sub, alpha, gate, (dpsi, yg) = self.net.activate(act, max(self.n_terms, 1))
        x, goal, psi = ctx["x"], ctx["goal"], tok["psi"]
        sub_world = x + to_world(sub * self.reach, psi)
        sub_world = torch.cat([sub_world[:, :2], sub_world[:, 2:].clamp_min(self.z_min)], -1)
        spec = TaskSpec(delta=sub_world - goal, alpha=alpha[:, :self.n_terms], gate=gate[:, :self.n_terms],
                        yaw=psi + dpsi, yaw_gate=yg)
        self._instr.append((float(ctx.get("t", 0)), spec.delta.clone(), spec.weight.clone()))
        if len(self._instr) > 4 * self.tok.kc:
            self._instr = self._instr[-2 * self.tok.kc:]
        if self.stochastic:
            self.records.append({"t": float(ctx.get("t", 0)), "act": act.clone(),
                                 "alive": ctx["alive"].clone(),
                                 "tok": {k: (v.clone() if torch.is_tensor(v) else v) for k, v in tok.items()}})
        return spec


def returns_from_stream(records: List[Dict], chain: List[Dict], gamma: float) -> Tensor:
    """Per decision and episode, the discounted sum of the cost increments the
    stream reported after it -- negated, so that lower cost is higher return."""
    ts = [r["t"] for r in records]
    cost_at = {m["t"]: m["cost"] for m in chain}
    times = sorted(cost_at)
    B = records[0]["alive"].shape[0]
    zero = torch.zeros(B, dtype=records[0]["act"].dtype)
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
        G = r + gamma * G
        R[k] = G
    return R


def ppo_update(net: PolicyNet, records: List[Dict], returns: Tensor, n_terms: int,
               epochs: int = 4, batch: int = 512, lr: float = 3e-4, clip: float = 0.2,
               vcoef: float = 0.5, ent: float = 1e-3, gen=None) -> Dict[str, float]:
    """Fit the recorded stream: clipped surrogate, value regression, entropy."""
    gen = gen or torch.Generator().manual_seed(0)
    samples, acts, rets = [], [], []
    for k, rec in enumerate(records):
        al = rec["alive"]
        for b in al.nonzero().flatten().tolist():
            samples.append({kk: (v[b] if torch.is_tensor(v) else v) for kk, v in rec["tok"].items()})
            acts.append(rec["act"][b]); rets.append(returns[k, b])
    if not samples:
        return {"n": 0}
    acts = torch.stack(acts); rets = torch.stack(rets)
    rets_n = (rets - rets.mean()) / rets.std().clamp_min(1e-6)
    with torch.no_grad():
        old_lp = []
        for i in range(0, len(samples), batch):
            b = collate_tok(samples[i:i + batch])
            pre, _ = net.pre(b)
            old_lp.append(net.dist(pre).log_prob(acts[i:i + batch]).sum(-1))
        old_lp = torch.cat(old_lp)
    # The exploration scale gets its own learning rate.  Adam moves a
    # parameter by about `lr` per step whatever the gradient, so at the
    # network's rate `log_std` could change by at most ~0.7% an iteration and
    # ~4x over a whole run -- nominally learned, unable to adapt in practice.
    # Ten times the rate lets it move ~7% an iteration; the floor in `std`
    # still keeps it from vanishing.
    head = [p for n, p in net.named_parameters() if n == "log_std"]
    body = [p for n, p in net.named_parameters() if n != "log_std"]
    opt = torch.optim.Adam([{"params": body, "lr": lr}, {"params": head, "lr": 10 * lr}])
    net.train()
    stats = {"n": len(samples), "loss": 0.0, "v_loss": 0.0, "ent": 0.0, "nb": 0,
             "kl": 0.0, "clipfrac": 0.0}
    # value-head fit BEFORE the update: how much of the return the composer's
    # situation explains.  Near zero means the return is not varying with what
    # the composer sees and does, and no update can find a gradient in it.
    with torch.no_grad():
        v0 = torch.cat([net.pre(collate_tok(samples[i:i + batch]))[1] for i in range(0, len(samples), batch)])
        stats["ev"] = float(1.0 - (rets_n - v0).var() / rets_n.var().clamp_min(1e-9))
    for _ in range(epochs):
        perm = torch.randperm(len(samples), generator=gen)
        for i in range(0, len(samples), batch):
            idx = perm[i:i + batch].tolist()
            b = collate_tok([samples[j] for j in idx])
            pre, v = net.pre(b)
            d = net.dist(pre)
            lp = d.log_prob(acts[idx]).sum(-1)
            adv = rets_n[idx] - v.detach()
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
            ratio = (lp - old_lp[idx]).exp()
            pg = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            with torch.no_grad():
                stats["kl"] += float((old_lp[idx] - lp).mean())               # approx KL(old || new)
                stats["clipfrac"] += float(((ratio - 1).abs() > clip).to(ratio.dtype).mean())
            vl = ((v - rets_n[idx]) ** 2).mean()
            e = d.entropy().sum(-1).mean()
            loss = pg + vcoef * vl - ent * e
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
            stats["loss"] += pg.item(); stats["v_loss"] += vl.item(); stats["ent"] += e.item(); stats["nb"] += 1
    net.eval()
    nb = max(stats.pop("nb"), 1)
    stats = {k: (v / nb if k != "n" else v) for k, v in stats.items()}      # per-minibatch means
    stats["sigma"] = float(net.std.mean())
    return stats


def collate_tok(samples):
    """`distill.collate` without the target fields."""
    fake = [dict(s, y_sub=torch.zeros(3, dtype=s["self"].dtype), y_alpha=torch.zeros(1, dtype=s["self"].dtype),
                 y_gate=torch.zeros(1, dtype=s["self"].dtype)) for s in samples]
    return collate(fake)


COMPOSERS["policy"] = PolicyComposer
