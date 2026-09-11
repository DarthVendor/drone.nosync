"""Training the composer against the objective, at its own timescale.

There is no teacher.  The composer is a language model over the flight: at
every report it reads the chain -- the drone's measurement tokens and its own
earlier action tokens -- and emits one ACTION TOKEN (see `actions.Vocab`).
It is rewarded with the rollout's own settled cost, read off the measurement
stream, so nothing about "how to navigate" is written into a reward.

`PolicyComposer` samples the token on the rows the update will read and takes
the mode elsewhere; `ppo_update` fits the recorded stream with the clipped
policy gradient.  The categorical is exact: likelihoods, KL and entropy need
no scale to tune, and a wide move is one token away.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import math

import torch
from torch import Tensor, nn

from .base import COMPOSERS
from .distill import collate
from .spec import TaskSpec
from .transformer import ComposerNet, TransformerComposer


class PolicyNet(ComposerNet):
    def __init__(self, n_terms, **kw):
        super().__init__(n_terms, **kw)
        d = self.embed.out_features
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def pre(self, tok, store=None, scene=None):
        """Action logits [B, V] and the value [B].  `store` collects attention
        weights for the explainer."""
        q = self.read_out(tok, store, scene)
        return self.head_act(q), self.value(q).squeeze(-1)

    def dist(self, logits):
        return torch.distributions.Categorical(logits=logits)


class PolicyComposer(TransformerComposer):
    """Sampled on the explored rows, the mode elsewhere; records every report."""
    kind = "policy"

    def __init__(self, system, trainable, **kw):
        # the checkpoint is for THIS net; the base class must not try to load
        # it into the plain ComposerNet it builds first
        weights = kw.pop("weights", "")
        self.explore_eps = float(kw.pop("explore_eps", 0.0))     # uniform mixture on the RECORDED rows (see `emit`)
        self.follow_parent = bool(kw.pop("follow_parent", False))  # a population's policy rows take the parent's token (see `emit`)
        # The update reads a capped random subset of the recorded decisions
        # (`ppo_update(max_samples=...)`), yet every recorded row carried its
        # full scene tokens: about 150k samples' worth per iteration for a
        # 10k update -- 600 MB per worker, most of it pickled to the parent and
        # dropped there.  Records keep their small fields for EVERY recorded
        # row (the returns need the whole stream); the scene tokens are kept
        # for this fraction of the rows, drawn from a private generator so the
        # flights' own randomness is untouched.
        self.tok_frac = float(kw.pop("tok_frac", 1.0))
        self._tok_gen = None
        # Where the exploration mixture puts its mass.  Uniform gives every
        # token the same slice, which sounds fair and is not: the policy emits
        # PLACE 92% of the time and TURN/priority about 1%, so the rare tokens
        # end a batch with ~76 flights of evidence against a per-flight spread
        # of 415 -- far too little to resolve an effect worth ~25, so they can
        # never earn their way out of being rare.  Balanced exploration puts
        # the mixture's mass where the policy is NOT looking, in inverse
        # proportion to how often each token is actually emitted.  The update
        # already corrects for the behaviour distribution by importance
        # weight, so this changes what is sampled, not what is estimated.
        self.explore_balance = bool(kw.pop("explore_balance", False))
        self._tok_count = None
        self._tok_decay = float(kw.pop("explore_decay", 0.5))   # counts kept across ~2 batches
        super().__init__(system, trainable, **kw)
        self.net = PolicyNet(max(self.n_terms, 1), d=kw.get("d", 64), heads=kw.get("heads", 4)).to(self.net_dtype)
        self.net.goal_gain = self.tok.scale / self.reach
        self.weights_path = weights
        if weights:
            import os
            from .transformer import load_composer_weights
            load_composer_weights(self.net, weights)
            self._weights_mtime = os.path.getmtime(weights)
        self.net.eval()
        self.stochastic = False
        self.records: List[Dict] = []          # per report: tokens (per row), the token chosen, t
        # Rows to record for the update, by full-batch id; None = every row.
        self.record_rows = None

    @torch.no_grad()
    def pair(self, n_eps: int, seed: int) -> None:
        """Pair the token draws by episode across the population for this run
        (`crn_sample`); the rollout calls it at every start."""
        self._crn = (int(n_eps), int(seed)); self._crn_k = 0; self._n_emit = 0
        if self._tok_count is not None:
            self._tok_count *= self._tok_decay      # follow the policy as it moves

    @torch.no_grad()
    def emit(self, ctx):
        """A chained decision per row (see `TransformerComposer._decide`); each
        component is sampled from the policy -- with the exploration mixture on
        the recorded rows, paired draws across the population, the parent's
        token on its kids when asked -- and recorded for the update."""
        eps = float(getattr(self, "explore_eps", 0.0))
        exploring = self.stochastic and eps > 0.0 and self.record_rows is not None and len(self.record_rows)
        rec_rows = self.record_rows.to(ctx["x"].device) if self.record_rows is not None else None

        def choose(logits, tok, ctx_s, ids_full, step, placing):
            B = logits.shape[0]
            b_logits = logits
            rec = torch.isin(ids_full, rec_rows) if rec_rows is not None else torch.zeros(B, dtype=torch.bool, device=logits.device)
            if exploring:
                V = logits.shape[-1]
                w = None
                if self.explore_balance:
                    if self._tok_count is None or self._tok_count.numel() != V:
                        self._tok_count = torch.zeros(V, dtype=torch.float64)
                    w = explore_weights(self._tok_count, V, protect=(self.net.vocab.EOS,))
                w = (torch.full((V,), 1.0 / V, dtype=logits.dtype, device=logits.device)
                     if w is None else w.to(logits.dtype).to(logits.device))
                mix = torch.log((1.0 - eps) * torch.softmax(logits, -1) + eps * w)
                mix = torch.where(torch.isfinite(logits), mix, logits)            # a forced EOS stays forced
                b_logits = torch.where(rec[:, None], mix, logits)
            if not self.stochastic:
                act = logits.argmax(-1)
            elif getattr(self, "_crn", None) is None:
                act = self.net.dist(b_logits).sample()
            else:
                E, seed = self._crn; k = self._crn_k; self._crn_k += 1
                own = rec if getattr(self, "unpair_recorded", False) and exploring else None
                act = crn_sample(b_logits, ids_full, E, seed, k, independent=own)
            if getattr(self, "follow_parent", False) and getattr(self, "_crn", None) is not None:
                E = self._crn[0]; ids_c = ids_full.cpu(); n_max = int(ids_c.max()) + 1
                pos = torch.full((n_max,), -1, dtype=torch.long); pos[ids_c] = torch.arange(len(ids_c))
                par = pos[ids_c % E]
                m = (ids_c // E > 0) & (par >= 0) & ~rec.cpu()
                if bool(m.any()):
                    act = act.clone(); act[m.to(act.device)] = act[par[m].to(act.device)]
            if self.explore_balance and self._tok_count is not None:
                self._tok_count.index_add_(0, act.detach().flatten().cpu(),
                                           torch.ones(act.numel(), dtype=torch.float64))
            n_emit = getattr(self, "_n_emit", 0); self._n_emit = n_emit + 1
            ov = getattr(self, "override", None)
            if ov is not None and n_emit in ov:
                o = ov[n_emit][ids_full.cpu()].to(act.device)
                act = torch.where(o >= 0, o, act)
            if self.stochastic:
                keep = rec if rec_rows is not None else torch.ones(B, dtype=torch.bool, device=logits.device)
                if bool(keep.any()):
                    moved = placing & (act == self.net.vocab.EOS)                 # the placement is made at EOS
                    keep_idx = keep.nonzero().flatten(); tok_keep = None
                    tf = float(getattr(self, "tok_frac", 1.0))
                    if tf < 1.0:
                        if self._tok_gen is None:
                            self._tok_gen = torch.Generator().manual_seed(int(torch.initial_seed()) % (2 ** 31) + 7)
                        tok_keep = torch.rand(keep_idx.numel(), generator=self._tok_gen) < tf
                        keep_idx = keep_idx[tok_keep.to(keep_idx.device)]
                    self.records.append({"t": float(ctx_s.get("t", 0)), "act": act[keep].clone(), "moved": moved[keep].clone(),
                                         "logits": b_logits[keep].float().clone(),
                                         "pi_logits": logits[keep].float().clone(),
                                         "alive": ctx_s["alive"][keep].clone(), "rows": ids_full[keep].clone(),
                                         "tok_keep": tok_keep,                                     # None: tokens for every row
                                         "tok": {k: (v[keep_idx].clone() if torch.is_tensor(v) and v.ndim and v.shape[0] == B
                                                     else (v.clone() if torch.is_tensor(v) else v)) for k, v in tok.items()}})
            return act
        return self._decide(ctx, choose)


def crn_sample(logits: torch.Tensor, ids: torch.Tensor, n_eps: int, seed: int, k: int,
               independent: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A categorical sample with COMMON RANDOM NUMBERS across the population.

    The k-th decision of episode e draws the same uniform in every genome
    (batch index = member * n_eps + episode), so two genomes flown on the same
    task see the same composer decision wherever their logits agree, and the
    GA's ranking compares controllers rather than dice.  Measured on the
    corridor city: with independent draws two task draws ranked the same 16
    genomes at Spearman +0.02 -- noise -- and +0.51 with the tokens frozen;
    the crash count, which the fitness tracks at +0.76, was the composer's
    luck rather than the genome's.  Inverse-CDF, so it is still an exact
    sample of softmax(logits); the recorded logits and acts feed PPO as before.
    """
    gen = torch.Generator(device="cpu").manual_seed(int((seed * 1_000_003 + k) % (2 ** 63 - 1)))
    u_ep = torch.rand(int(n_eps), generator=gen, dtype=torch.float64)
    u = u_ep[(ids % int(n_eps)).cpu()].to(logits.device)
    if independent is not None and bool(independent.any()):
        # rows drawn on their own: a hash of (row, decision, seed) as a uniform,
        # so two genomes' recordings of one episode are different flights
        h = (ids.cpu().to(torch.int64) * 2_654_435_761 + (k + 1) * 40_503 + seed * 97) % (2 ** 31 - 1)
        u_own = (h.to(torch.float64) + 0.5) / (2 ** 31 - 1)
        u = torch.where(independent.cpu(), u_own, u.cpu()).to(logits.device)
    cdf = torch.softmax(logits.double(), dim=-1).cumsum(-1)
    return (cdf < u[:, None]).sum(-1).clamp(max=logits.shape[-1] - 1)


def returns_from_stream(records: List[Dict], chain: List[Dict], gamma: float,
                        subgoal_cost: float = 0.0, unit: float = 0.0,
                        speed_bonus: float = 0.0, speed_ref: float = 5.0) -> Tensor:
    """Per decision and FULL-BATCH row, the discounted sum of the cost
    increments the stream reported after it -- negated, so lower cost is higher
    return.  Records may cover only the rows that were alive at that decision
    (`rows`); the return is computed for every row and the update picks its
    own rows out.  `subgoal_cost` is charged to a row for every decision it
    was given (every subgoal placed for it), so the composer is asked to
    reach the goal with the fewest.  `unit` > 0 makes the discount a rate in
    TIME: gamma per `unit` steps between one record and the next, so the
    horizon does not shorten when decisions are dense and stretch when they
    are sparse (decisions are events, and near the end of a batch few rows
    are still deciding).  0 keeps gamma per record.

    `speed_bonus` credits the vehicle's own airspeed: that much per interval at
    `speed_ref`, pro rata below it.  This is the ONE shaped term in the
    composer's objective, and it lives here rather than in the cost so that the
    plant cost the low level is evolved on stays exactly what it was -- distance
    to go, the arrival bonus, the charge for dying.

    Keep it small.  Raw speed is gameable in principle: a vehicle circling at
    full speed collects it forever, and only the distance charge accumulating
    against it makes circling lose.  Well under the distance term and that
    balance holds; comparable to it and it tips.  A dead row earns nothing, so
    the credit cannot buy a cheap crash."""
    ts = [r["t"] for r in records]
    cost_at = {m["t"]: m["cost"] for m in chain}
    speed_at = {m["t"]: (m.get("speed"), m.get("alive")) for m in chain} if speed_bonus else {}
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
        if speed_bonus:
            # credited at the interval's END, the instant its cost increment is
            # read, and only while the row is still flying
            t1 = ts[k + 1] if k + 1 < len(records) else (times[-1] if times else ts[k])
            prev = [u for u in times if u <= t1]
            sp, al = speed_at.get(prev[-1], (None, None)) if prev else (None, None)
            if sp is not None:
                credit = speed_bonus * (sp.to(r.dtype) / speed_ref).clamp(0.0, 1.0)
                r = r + (credit if al is None else credit * al.to(r.dtype))
        if subgoal_cost:
            rows = records[k].get("rows"); mv = records[k].get("moved")
            if rows is not None and mv is not None:
                rows = rows[mv.to(torch.bool)]               # only the rows given a new subgoal at this report
            r = r - subgoal_cost if rows is None else \
                r.index_add(0, rows.to(torch.long), torch.full((rows.numel(),), -subgoal_cost, dtype=r.dtype))
        g = gamma if not unit else gamma ** ((ts[k + 1] - ts[k]) / unit) if k + 1 < len(records) else 0.0
        G = r + g * G
        R[k] = G
    return R


def explore_weights(count: Optional[Tensor], V: int, protect: Sequence[int] = ()) -> Optional[Tensor]:
    """Where the exploration mixture should put its mass, given how often each
    token has actually been emitted.

    Inverse to frequency, floored so a token the policy never emits gets at
    most V times a uniform share rather than everything: weight is
    1 / (frequency + 1/V), normalised.  A token at uniform frequency keeps a
    uniform share, so a policy that already spreads evenly is left alone.

    `protect` pins tokens to a plain uniform share and leaves them out of the
    rebalancing.  EOS belongs there, and leaving it out was a real bug: it is
    52% of emitted components because every chain ENDS with one and a bare EOS
    is silence -- structural frequency, not over-use.  Reweighted against, it
    fell to 0.09x a uniform share, so exploring chains stopped terminating,
    ran to the component limit, and turned every exploring decision into a
    placement.  Subgoals per flight climbed monotonically until it was pinned.

    None (or an empty count) means uniform -- the first batch, before anything
    has been seen.
    """
    if count is None or count.numel() != V or float(count.sum()) <= 0:
        return None
    keep = torch.ones(V, dtype=torch.bool)
    for i in protect:
        if 0 <= int(i) < V:
            keep[int(i)] = False
    n_free = int(keep.sum())
    if n_free == 0:
        return torch.full((V,), 1.0 / V, dtype=count.dtype)
    w = torch.full((V,), 1.0 / V, dtype=count.dtype)          # protected: a plain uniform share
    c = count[keep]
    tot = float(c.sum())
    if tot <= 0:
        return w
    f = c / tot
    wf = 1.0 / (f + 1.0 / n_free)
    w[keep] = wf / wf.sum() * (n_free / V)                    # the rest share what is left
    return w / w.sum()


def center_by_task(returns: List[Tensor], rows: List[Tensor], n_eps: int,
                   min_count: int = 2) -> None:
    """Subtract a per-(task, decision) baseline from the composer's returns, in
    place, pooled across shards.

    Why this and not the per-genome centring it replaces: most of the spread in
    a flight's return is not the policy's doing, it is the TASK's.  A flight's
    cost is dominated by whether it crashed, dying charges 40/s against a 36 s
    episode, and 55% of starts on this city have a building within 10 m on the
    line to the goal.  So the same policy scores ~1100 on a hard draw and ~250
    on an easy one, and centring per genome leaves all of that in.  Every
    genome flies the SAME task list, so the mean over genomes of one task is a
    clean estimate of that task's difficulty, and removing it leaves what the
    decisions actually changed.

    Unbiased: the baseline depends on the task and the decision index, never on
    the action taken.  Tasks with fewer than `min_count` samples are left
    alone, since a one-sample mean would subtract the very return being
    learned from and zero the gradient.

    `rows` are GLOBAL row indices (genome * n_eps + task); the shards split the
    population, so a task's samples live in different shards and the pooling
    has to happen across all of them.
    """
    if not returns:
        return
    K = max(int(R.shape[0]) for R in returns)
    dt = returns[0].dtype
    tot = torch.zeros(K, n_eps, dtype=dt)
    cnt = torch.zeros(K, n_eps, dtype=dt)
    tids = [(r % n_eps).to(torch.long) for r in rows]
    for R, tid in zip(returns, tids):
        k = int(R.shape[0])
        tot[:k].index_add_(1, tid, R.to(dt))
        cnt[:k].index_add_(1, tid, torch.ones_like(R, dtype=dt))
    mean = tot / cnt.clamp(min=1.0)
    enough = cnt >= float(min_count)
    for R, tid in zip(returns, tids):
        k = int(R.shape[0])
        b = torch.where(enough[:k][:, tid], mean[:k][:, tid], torch.zeros((), dtype=dt))
        R -= b.to(R.dtype)


def ppo_update(net: PolicyNet, records: List[Dict], returns: Tensor, n_terms: int,
               epochs: int = 4, batch: int = 512, lr: float = 3e-4, clip: float = 0.2,
               vcoef: float = 0.5, ent: float = 1e-3, gen=None,
               target_kl: float = 0.02, backtracks: int = 8, opt=None, max_samples: int = 0) -> Dict[str, float]:
    """Fit the recorded stream: the clipped surrogate over the action tokens
    (a categorical: exact likelihoods and KL), an optional value regression
    and entropy bonus.

    The step is sized in POLICY space when `target_kl` > 0: after the epochs
    the exact KL from the collecting policy is measured over the whole batch;
    beyond 2x the target the weights go back and the rate is halved, up to
    `backtracks` times.  `target_kl` = 0 runs every epoch at the given rate.
    `opt`: an Adam over the net kept ACROSS calls (a fresh Adam's first step
    moves every parameter by the full rate whatever the gradient).
    `max_samples` > 0: a random subset of the samples of that size -- one
    token per report per live row is ~200k samples when flights run long,
    and the update was measured at 10+ minutes on them."""
    gen = gen or torch.Generator().manual_seed(0)
    infer_dtype = next(net.parameters()).dtype
    net.float()
    groups = list(zip(records, returns)) if (records and isinstance(records[0], list)) else [(records, returns)]
    samples, acts, rets, blog, plog = [], [], [], [], []
    for recs, R in groups:
        for k, rec in enumerate(recs):
            al = rec["alive"]; rows = rec.get("rows"); bl = rec.get("logits"); pl = rec.get("pi_logits")
            tk = rec.get("tok_keep")                       # rows whose scene tokens were kept (None: all of them)
            if tk is None:
                js = al.nonzero().flatten().tolist(); tpos = None
            else:
                tk = tk.to(al.device); tpos = tk.long().cumsum(0) - 1; js = (al & tk).nonzero().flatten().tolist()
            for j in js:
                b = int(rows[j]) if rows is not None else j
                jt = j if tpos is None else int(tpos[j])
                samples.append({kk: (v[jt].float() if torch.is_tensor(v) and v.is_floating_point() else (v[jt] if torch.is_tensor(v) else v))
                                for kk, v in rec["tok"].items()})
                acts.append(rec["act"][j]); rets.append(R[k, b]); blog.append(None if bl is None else bl[j]); plog.append(None if pl is None else pl[j])
    if not samples:
        return {"n": 0}
    if max_samples and len(samples) > max_samples:
        pick = torch.randperm(len(samples), generator=gen)[:max_samples].sort().values.tolist()
        samples = [samples[i] for i in pick]; acts = [acts[i] for i in pick]; rets = [rets[i] for i in pick]; blog = [blog[i] for i in pick]; plog = [plog[i] for i in pick]
    acts = torch.stack(acts).long(); rets = torch.stack(rets).float()
    behaviour = torch.stack(blog).float() if all(b is not None for b in blog) else None
    policy_old = torch.stack(plog).float() if all(p_ is not None for p_ in plog) else None
    ok = torch.isfinite(rets)
    if behaviour is not None:
        ok = ok & torch.isfinite(behaviour).all(-1)
    if policy_old is not None:
        ok = ok & torch.isfinite(policy_old).all(-1)
    dropped = int((~ok).sum())
    if dropped:
        keep = ok.nonzero().flatten().tolist()
        samples = [samples[i] for i in keep]; acts = acts[ok]; rets = rets[ok]
        behaviour = None if behaviour is None else behaviour[ok]
        policy_old = None if policy_old is None else policy_old[ok]
        if not samples:
            return {"n": 0, "nonfinite": dropped}
    rets_n = (rets - rets.mean()) / rets.std().clamp_min(1e-6)
    ALL = collate_tok(samples)
    def take(idx):
        return {k: (v[idx] if torch.is_tensor(v) else v) for k, v in ALL.items()}
    with torch.no_grad():
        # the reference policy: the one that COLLECTED the samples when the
        # records say so (the batch may have flown while the last update ran),
        # else this net as it stands
        # The ratio and the KL are taken against the POLICY at collection time
        # (`pi_logits`).  When the recorded rows explored -- drew from the policy
        # mixed with a uniform -- that behaviour distribution starts far from the
        # policy; clipping the ratio around 1 against IT let a rare token be
        # raised fivefold before the clip bit (one update: KL 0.107, entropy
        # 2.98, the policy flattened).  Off-policy samples are reweighted by
        # pi_old / pi_behaviour, capped at 5, as a fixed per-sample weight.
        ref = policy_old if policy_old is not None else behaviour
        old_logits = ref if ref is not None else \
            torch.cat([net.pre(take(torch.arange(i, min(i + batch, len(samples)))))[0] for i in range(0, len(samples), batch)])
        old_lp = torch.distributions.Categorical(logits=old_logits).log_prob(acts)
        old_p = torch.softmax(old_logits, -1)
        if policy_old is not None and behaviour is not None:
            b_lp = torch.distributions.Categorical(logits=behaviour).log_prob(acts)
            iw = (old_lp - b_lp).clamp(max=math.log(5.0)).exp()
        else:
            iw = torch.ones_like(old_lp)

    def kl_to_old(idx, logits):
        # exact KL(old || new) of the categoricals
        return (old_p[idx] * (torch.log_softmax(old_logits[idx], -1) - torch.log_softmax(logits, -1))).sum(-1).mean()

    def batch_kl():
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(samples), batch):
                idx = torch.arange(i, min(i + batch, len(samples)))
                tot += float(kl_to_old(idx, net.pre(take(idx))[0])) * len(idx)
            return tot / len(samples)

    params = [p for p in net.parameters() if p.requires_grad]
    if opt is None:
        opt = torch.optim.Adam(params, lr=lr)
    import copy
    opt_start = copy.deepcopy(opt.state_dict())
    net.train()
    stats = {"n": len(samples), "nonfinite": dropped, "backtracks": 0}
    if vcoef:
        # the value head's fit before the update -- a forward pass over every
        # sample, so only when the head is trained at all
        with torch.no_grad():
            v0 = torch.cat([net.pre(take(torch.arange(i, min(i + batch, len(samples)))))[1] for i in range(0, len(samples), batch)])
            stats["ev"] = float(1.0 - (rets_n - v0).var() / rets_n.var().clamp_min(1e-9))
    start = {k: v.detach().clone() for k, v in net.state_dict().items()}
    lr_used = lr
    for attempt in range(backtracks + 1):
        for g in opt.param_groups: g["lr"] = lr_used
        acc = {"loss": 0.0, "v_loss": 0.0, "ent": 0.0, "clipfrac": 0.0, "nb": 0, "epochs": 0}
        stop = False
        for _ in range(epochs):
            if stop:
                break
            acc["epochs"] += 1
            perm = torch.randperm(len(samples), generator=gen)
            for i in range(0, len(samples), batch):
                idx = perm[i:i + batch]
                logits, v = net.pre(take(idx))
                d = torch.distributions.Categorical(logits=logits)
                lp = d.log_prob(acts[idx])
                adv = rets_n[idx] - v.detach()
                adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
                ratio = (lp - old_lp[idx]).clamp(-20.0, 20.0).exp()
                pg = -(iw[idx] * torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)).mean()
                with torch.no_grad():
                    acc["clipfrac"] += float(((ratio - 1).abs() > clip).to(ratio.dtype).mean())
                    kl_mb = float(kl_to_old(idx, logits))
                    if target_kl and kl_mb > target_kl:
                        stop = True
                vl = ((v - rets_n[idx]) ** 2).mean()
                e = d.entropy().mean()
                loss = pg + vcoef * vl - ent * e
                if not torch.isfinite(loss):
                    stats["nonfinite"] = stats.get("nonfinite", 0) + 1
                    continue
                acc["kl_mb"] = acc.get("kl_mb", 0.0) + kl_mb          # only the minibatches that stepped count toward the reported KL
                if stop:
                    break
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 0.5); opt.step()
                acc["loss"] += pg.item(); acc["v_loss"] += vl.item(); acc["ent"] += e.item(); acc["nb"] += 1
        # the whole-batch KL is a forward pass over every sample: taken when
        # it sizes the step, otherwise the minibatch mean stands in for it
        kl = batch_kl() if target_kl else acc.pop("kl_mb", 0.0) / max(acc.get("nb", 1), 1)
        if not target_kl or kl <= 2.0 * target_kl or attempt == backtracks:
            break
        net.load_state_dict(start); opt.load_state_dict(copy.deepcopy(opt_start)); lr_used *= 0.5; stats["backtracks"] += 1
    net.eval(); net.to(infer_dtype)
    nb = max(acc.pop("nb"), 1); acc.pop("kl_mb", None)
    stats.update({k: (v / nb if k != "epochs" else v) for k, v in acc.items()})
    stats["kl"] = kl; stats["lr"] = lr_used
    with torch.no_grad():
        # what the policy says, on this batch: how often it speaks (not HOLD) and its entropy
        p = torch.softmax(old_logits, -1)
        stats["speak"] = float(1.0 - p[:, net.vocab.HOLD].mean())
        stats["entropy"] = float(torch.distributions.Categorical(probs=p).entropy().mean())
    return stats


def collate_tok(samples):
    """`distill.collate` without the target fields."""
    fake = [dict(s, y_sub=torch.zeros(3, dtype=s["self"].dtype), y_alpha=torch.zeros(1, dtype=s["self"].dtype),
                 y_gate=torch.zeros(1, dtype=s["self"].dtype)) for s in samples]
    return collate(fake)


COMPOSERS["policy"] = PolicyComposer
