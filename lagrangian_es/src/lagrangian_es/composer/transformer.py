"""The learned composer: a small transformer that writes TaskSpecs.

Two attention stacks factor the problem.  Set attention over [self, goal,
entities] gives permutation invariance and variable cardinality over the scene;
causal attention over the last few (instruction, measurement) pairs gives phase
memory and lets the composer condition on what its own last instruction did.
Constraint tokens -- one learned embedding per term in the plant's library --
cross-attend to both, and one output is read per constraint token, so adding a
term adds a token rather than widening a head.

Every output channel is bounded by construction: priorities through softplus,
gates through a sigmoid, the subgoal delta through r_max * tanh in the ego
frame so it can never leave the low level's measured reach.  The hold, not this
module, is what makes the spec safe to apply; this module only promises not to
ask for anything the low level is not known to fly.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from .base import COMPOSERS, Composer
from .spec import TaskSpec
from .tokens import ACT_SCALE, F, INSTR, N_TYPES, Tokenizer, to_world


class Block(nn.Module):
    def __init__(self, d, heads, cross=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.lnc = nn.LayerNorm(d); self.xattn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    @staticmethod
    def _attn(mod: nn.MultiheadAttention, q: Tensor, kv: Tensor, key_pad=None, attn_mask=None) -> Tensor:
        """The module's own projections through the fused attention kernel.

        Same weights, same numbers (measured max |diff| 1e-7 against the
        module's forward), a third of the time gone: 15.6 ms against 23.2 ms
        for one scene block at worker width.  Masks are the module's
        convention (True = masked) and are inverted here for the kernel's."""
        B, Lq, d = q.shape; H = mod.num_heads; dh = d // H
        W, b = mod.in_proj_weight, mod.in_proj_bias
        if q is kv:
            qkv = nn.functional.linear(q, W, b).view(B, Lq, 3, H, dh).permute(2, 0, 3, 1, 4)
            Q, K, V = qkv[0], qkv[1], qkv[2]
        else:
            Q = nn.functional.linear(q, W[:d], b[:d]).view(B, Lq, H, dh).transpose(1, 2)
            kvp = nn.functional.linear(kv, W[d:], b[d:]).view(B, kv.shape[1], 2, H, dh).permute(2, 0, 3, 1, 4)
            K, V = kvp[0], kvp[1]
        m = None
        if attn_mask is not None:
            m = ~attn_mask                                   # [Lq, Lk], True = may attend
        if key_pad is not None:
            kp = ~key_pad[:, None, None, :]                  # [B, 1, 1, Lk]
            m = kp if m is None else (m & kp)
        o = nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=m)
        return nn.functional.linear(o.transpose(1, 2).reshape(B, Lq, d), mod.out_proj.weight, mod.out_proj.bias)

    def forward(self, x, mask=None, attn_mask=None, mem=None, mem_mask=None, store=None, last=False):
        """`store`, if given, receives the head-averaged attention weights --
        what each query drew from each key -- for the decision explainer.
        `last`: only the final position's output is wanted (the chain summary),
        so only that query is run -- against every key, the causal mask being
        moot for the last position.  Exact, and 4x cheaper for that block."""
        # A padding mask that masks nothing still forces attention off the fast
        # path; in a worker every row carries the same token count, so drop it.
        if mask is not None and not bool(mask.any()):
            mask = None
        if mem_mask is not None and not bool(mem_mask.any()):
            mem_mask = None
        if store is None:
            h = self.ln1(x)
            if last:
                x = x[:, -1:] + self._attn(self.attn, h[:, -1:], h, key_pad=mask)
            else:
                x = x + self._attn(self.attn, h, h, key_pad=mask, attn_mask=attn_mask)
            if self.cross and mem is not None:
                x = x + self._attn(self.xattn, self.lnc(x), mem, key_pad=mem_mask)
            return x + self.ff(self.ln2(x))
        # the explainer's path: the module's forward, which can return the weights
        h = self.ln1(x)
        a, w = self.attn(h, h, h, key_padding_mask=mask, attn_mask=attn_mask,
                         need_weights=True, average_attn_weights=True)
        x = x + a
        store.setdefault("self", []).append(w.detach())
        if self.cross and mem is not None:
            h = self.lnc(x)
            a, w = self.xattn(h, mem, mem, key_padding_mask=mem_mask,
                              need_weights=True, average_attn_weights=True)
            x = x + a
            store.setdefault("cross", []).append(w.detach())
        x = x + self.ff(self.ln2(x))
        return x[:, -1:] if last else x


class ComposerNet(nn.Module):
    """The composer as a language model over the flight: the scene tokens,
    the chain of measurement and action tokens, and one ACTION TOKEN out."""

    def __init__(self, n_terms: int, d: int = 64, heads: int = 4, n_scene: int = 2,
                 n_chain: int = 2, n_read: int = 1, n_types: int = N_TYPES):
        super().__init__()
        from .actions import Vocab
        self.vocab = Vocab(n_terms); V = self.vocab.V
        self.embed = nn.Linear(F, d); self.type_emb = nn.Embedding(n_types, d)
        self.act_emb = nn.Embedding(V, d)                            # an action token in the chain, by id
        self.scene = nn.ModuleList([Block(d, heads) for _ in range(n_scene)])
        self.chain = nn.ModuleList([Block(d, heads) for _ in range(n_chain)])
        self.constraint = nn.Parameter(torch.randn(n_terms, d) * 0.02)
        self.pool = nn.Parameter(torch.randn(1, d) * 0.02)          # the query the action is read from
        self.read = nn.ModuleList([Block(d, heads, cross=True) for _ in range(n_read)])
        self.ln = nn.LayerNorm(d)
        self.head_act = nn.Linear(d, V)                              # logits over the vocabulary
        self.n_terms = n_terms
        nn.init.normal_(self.act_emb.weight, std=0.02)
        # The prior: HOLD.  A fresh head would emit a random token every
        # report -- measured on the first day, a random composer killed every
        # flight.  So the head starts silent: the subgoal is the goal itself,
        # the straight line, and every other token is a learned deviation.
        nn.init.zeros_(self.head_act.weight)
        with torch.no_grad():
            # ~80% HOLD, ~11% "continue straight at full reach", the other 47
            # tokens ~0.2% each.  Measured with straight at the same prior as the
            # rest: a sampled flight speaks a random token every ~2 s and none
            # survived 36 s (0.000 / 1.000); with continuing the default word,
            # sampled flights mostly fly on and the update can learn WHEN to
            # say it, and what else to say, from flights that arrive.
            self.head_act.bias.zero_(); self.head_act.bias[self.vocab.HOLD] = 6.0; self.head_act.bias[self.vocab.straight] = 4.0

    goal_gain = 4.0    # scale/reach, so the goal token reads in reach units

    def goal_ego(self, tok):
        """The goal offset in the ego frame, in units of the reach."""
        return tok["goal"][:, :3] * self.goal_gain

    def _chain_in(self, tok):
        ch = self.embed(tok["chain"]) + self.type_emb(tok["chain_types"])
        ids = (tok["chain"][..., 0] * ACT_SCALE).round().long().clamp(0, self.vocab.V - 1)
        return ch + self.act_emb(ids) * (tok["chain_types"] == INSTR).to(ch.dtype)[..., None]

    def encode_scene(self, tok: Dict[str, Tensor], store=None):
        """The scene encoding (self, goal, entities through the scene blocks)
        and its padding mask.  It does not depend on the chain, so within one
        decision it is computed once and reused for every component step
        (measured: the chain loop ran the whole net five times per report)."""
        B = tok["self"].shape[0]; te = self.type_emb
        scene = torch.cat([self.embed(tok["self"])[:, None] + te.weight[0],
                           self.embed(tok["goal"])[:, None] + te.weight[1],
                           self.embed(tok["entities"]) + te(tok["ent_types"])], 1)
        smask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=scene.device), ~tok["ent_mask"]], 1)
        for blk in self.scene:
            scene = blk(scene, mask=smask, store=None if store is None else store.setdefault("scene", {}))
        return scene, smask

    def read_out(self, tok: Dict[str, Tensor], store=None, scene=None) -> Tensor:
        """The pool query after the read layer, [B, d].  `scene`: a
        precomputed (encoding, mask) from `encode_scene` for these rows."""
        B = tok["self"].shape[0]
        mem, mmask = scene if scene is not None else self.encode_scene(tok, store)
        if tok["chain"].shape[1]:
            ch = self._chain_in(tok)
            L = ch.shape[1]
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ch.device), 1)
            for i, blk in enumerate(self.chain):
                ch = blk(ch, attn_mask=causal, mask=tok.get("chain_mask"), store=None if store is None else store.setdefault("chain", {}),
                         last=i == len(self.chain) - 1)
            mem = torch.cat([mem, ch[:, -1:]], 1)                   # the latest chain state
            mmask = torch.cat([mmask, torch.zeros(B, 1, dtype=torch.bool, device=ch.device)], 1)
        q = torch.cat([self.pool.expand(B, -1, -1), self.constraint.expand(B, -1, -1)], 1)
        for blk in self.read:
            q = blk(q, mem=mem, mem_mask=mmask, store=None if store is None else store.setdefault("read", {}))
        return self.ln(q)[:, 0]

    def forward(self, tok: Dict[str, Tensor], store=None, scene=None) -> Tensor:
        """Logits over the action vocabulary, [B, V]."""
        return self.head_act(self.read_out(tok, store, scene))


class TransformerComposer(Composer):
    """A `ComposerNet` behind the composer interface, with its own chain memory."""
    kind = "transformer"
    live_only = True        # 1.68 s a call on 1152 rows; dead rows are not asked

    DTYPES = {"float64": torch.float64, "float32": torch.float32,
              "bfloat16": torch.bfloat16, "float16": torch.float16}

    def __init__(self, system, trainable, reach: float = 10.0, every: int = 5,
                 measure_every=None, d: int = 64, heads: int = 4, k_chain: int = 32,
                 weights: str = "", dtype: str = "float32", **kw):
        super().__init__(system, trainable)
        # `every`: how often the composer runs over the stream; `measure_every`:
        # how often the drone appends a measurement token.  Both default to
        # 0.1 s -- the composer monitors, it does not poll once a second.
        # `every` is the LONGEST a placed subgoal is held; decisions fall on
        # report steps when a subgoal is achieved, the leg changes, or the
        # hold runs out.  The report cadence defaults to the hold.
        self.reach, self.every = float(reach), int(every)
        self.measure_every = int(measure_every) if measure_every is not None else int(every)
        self.z_min = float(getattr(system, "z_floor", 0.0)) + 0.5
        span = float(getattr(system.env, "span", 32.0)) if hasattr(system, "env") else 32.0
        self.tok = Tokenizer(scale=span, reach=self.reach, k_chain=k_chain)
        self.k_chain = int(k_chain)
        # Inference dtype.  The composer's decisions need none of the plant's
        # float64.  Measured per call at 192 rows on this CPU: float64 332 ms,
        # float32 151 ms, bfloat16 510 ms, float16 453 ms -- the half formats
        # are emulated here, so float32 is the default; `dtype="float16"` is
        # one keyword away on hardware with the units.  Tokens are cast in,
        # outputs cast back to the plant's dtype; the policy update always runs
        # in float32, since half-precision gradients underflow.
        self.dtype = system.dtype
        self.net_dtype = self.DTYPES[dtype]
        self.net = ComposerNet(max(self.n_terms, 1), d=d, heads=heads).to(self.net_dtype)
        self.net.goal_gain = span / self.reach
        self.weights_path = weights
        if weights:
            import os
            load_composer_weights(self.net, weights)
            self._weights_mtime = os.path.getmtime(weights)
        self.net.eval()
        self._instr: List[Tuple[float, Tensor, Tensor]] = []     # (t, action token per row, valid rows)

    def attach(self, sensors):
        """The rollout hands over its sensors so bearings come from them."""
        self.tok.attach(sensors)

    def reset(self, B):
        self._instr = []; self._B = int(B); self._rows = None

    def tokens(self, ctx):
        rows = getattr(self, "_rows", None)
        instr = self._instr if rows is None else [tuple([e[0]] + [v[rows] for v in e[1:]]) for e in self._instr]
        tok = self.tok(ctx, instr)
        return {k: (v.to(self.net_dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in tok.items()}

    monitors = True         # asked at every report; a HOLD token changes nothing

    def _current(self, ctx, B):
        """The rows' current target spec, from the rollout when it has one."""
        cur = ctx.get("spec")
        if cur is None:
            cur = TaskSpec.identity(B, self.d, self.n_terms, self.dtype, ctx["x"].device)
        return cur

    def _apply(self, ids, ctx, tok):
        x, goal, psi = ctx["x"], ctx["goal"], tok["psi"].to(self.dtype)
        spec = self.net.vocab.apply(ids, self._current(ctx, x.shape[0]), x, goal, psi, self.net.goal_ego(tok).to(self.dtype),
                                    self.reach, self.z_min)
        self._remember(ctx, ids)
        return spec

    @staticmethod
    def _sub_ctx(ctx, idx, B):
        """The context restricted to rows `idx` (tensors with a leading dim
        of B, nested dicts and lists of them, and the TaskSpec)."""
        def take(v):
            if torch.is_tensor(v):
                return v[idx] if v.ndim and v.shape[0] == B else v
            if isinstance(v, TaskSpec):
                return TaskSpec(v.delta[idx], v.alpha[idx], v.gate[idx],
                                None if v.yaw is None else v.yaw[idx], None if v.yaw_gate is None else v.yaw_gate[idx],
                                None if getattr(v, "moved", None) is None else v.moved[idx])
            if isinstance(v, dict):
                return {k: take(w) for k, w in v.items()}
            if isinstance(v, list):
                return [take(w) for w in v]
            return v
        return {k: take(v) for k, v in ctx.items()}

    def _decide(self, ctx, choose):
        """A DECISION per row: components chained until EOS (or L_MAX, when
        EOS is the only token left), each chosen by `choose(logits, tok,
        ctx_s, rows_ids, step, placing)` over the rows still deciding.  A
        component is written to the chain as soon as it is made, so the next
        one is chosen in sight of it; at EOS the pending placement is made.

        Only step 0 tokenizes and encodes the scene.  Every later step keeps
        the previous token batch, drops the rows that emitted EOS, appends
        the component they just made as one chain row (age 0, valid), and
        reuses the scene encoding -- the scene, the measurement events and
        the older instructions cannot change inside a decision.  (Re-running
        the tokenizer and the whole net per step was 5 net calls a report.)"""
        x, goal = ctx["x"], ctx["goal"]; B = x.shape[0]
        rows_full = getattr(self, "_rows", None)
        ids_full = rows_full if rows_full is not None else torch.arange(B, device=x.device)
        if rows_full is None:
            self._B = B                                    # the chain memory's width, when called without a reset
        V = self.net.vocab
        tok0 = self.tokens(ctx); psi = tok0["psi"].to(self.dtype); g_ego = self.net.goal_ego(tok0).to(self.dtype)
        out, pb, pr = V.begin(self._current(ctx, B), psi)
        active = torch.ones(B, dtype=torch.bool, device=x.device)
        alive0 = ctx["alive"]; t_now = ctx.get("t", 0)
        saved = getattr(self, "_rows", None)
        tok_cur, cur_pos, last_act, scene0 = tok0, torch.arange(B, device=x.device), None, None
        try:
            for step in range(V.L_MAX + 1):
                idx = active.nonzero().flatten()
                if idx.numel() == 0:
                    break
                if step == 0:
                    tok = tok0; scene0 = self.net.encode_scene(tok0); sc = scene0
                else:
                    keep = torch.isin(cur_pos, idx)                          # rows still deciding, in `idx` order
                    n_prev = cur_pos.numel()
                    tok = {k: (v[keep] if torch.is_tensor(v) and v.ndim and v.shape[0] == n_prev else v) for k, v in tok_cur.items()}
                    ch = tok["chain"]; n = idx.numel()
                    row = torch.zeros(n, 1, ch.shape[-1], dtype=ch.dtype, device=ch.device)
                    row[:, 0, 0] = last_act[keep].to(ch.dtype) / ACT_SCALE       # the component just made: id, age 0
                    chain = torch.cat([ch, row], 1)
                    ctypes = torch.cat([tok["chain_types"], torch.full((n, 1), INSTR, dtype=torch.long, device=ch.device)], 1)
                    cmask = torch.cat([tok["chain_mask"], torch.zeros(n, 1, dtype=torch.bool, device=ch.device)], 1)
                    kc = self.tok.kc
                    if chain.shape[1] > kc:
                        chain, ctypes, cmask = chain[:, -kc:], ctypes[:, -kc:], cmask[:, -kc:]
                    tok["chain"], tok["chain_types"], tok["chain_mask"] = chain, ctypes, cmask
                    sc = (scene0[0][idx], scene0[1][idx])
                self._rows = ids_full[idx]
                logits = self.net.pre(tok, scene=sc)[0] if hasattr(self.net, "pre") else self.net(tok, scene=sc)
                if step == V.L_MAX:                                   # the cap: only EOS is left
                    forced = torch.full_like(logits, float("-inf")); forced[:, V.EOS] = 0.0; logits = forced
                placing = (pb[idx] >= 0) | (pr[idx] >= 0)
                ctx_s = {"alive": alive0[idx], "t": t_now}
                act = choose(logits, tok, ctx_s, ids_full[idx], step, placing)
                V.step(act, out, pb, pr, psi, rows=idx)
                self._remember(ctx_s, act)
                tok_cur, cur_pos, last_act = tok, idx, act
                active[idx[act == V.EOS]] = False
        finally:
            self._rows = saved
        return V.finish(out, pb, pr, x, goal, psi, g_ego, self.reach, self.z_min)

    @torch.no_grad()
    def emit(self, ctx):
        return self._decide(ctx, lambda logits, tok, ctx_s, rows, step, placing: logits.argmax(-1))

    def _remember(self, ctx, ids):
        """Append the action tokens to the per-row chain memory: one entry
        per report, valid for the rows that spoke (HOLD is silence).  When
        only some rows were asked (`_rows`), scatter into full width."""
        rows = getattr(self, "_rows", None)
        B = ids.shape[0] if rows is None else self._B
        dev = ids.device
        full = torch.zeros(B, dtype=torch.long, device=dev); valid = torch.zeros(B, dtype=torch.bool, device=dev)
        spoke = ids != self.net.vocab.HOLD
        sel = spoke.nonzero().flatten() if rows is None else rows[spoke]
        full[sel] = ids[spoke]; valid[sel] = True
        if not bool(valid.any()):
            return
        self._instr.append((float(ctx.get("t", 0)), full, valid))
        if len(self._instr) > 4 * self.tok.kc:
            self._instr = self._instr[-2 * self.tok.kc:]


def load_composer_weights(net: nn.Module, path: str) -> None:
    """Load a checkpoint into a `ComposerNet`.  One written before the action
    vocabulary carries continuous heads this net no longer has; its body
    (embeddings, blocks) is taken and the action head keeps its prior."""
    sd = torch.load(path, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith(("head_w", "head_sub", "head_yaw", "head_move", "log_std"))}
    own = net.state_dict()
    # a checkpoint from a different term count or vocabulary: the parameters
    # whose shape changed (constraint queries, action head, action embedding)
    # keep their prior; the body is what carries over
    # The type embedding GREW when the two map token types were added.  Its
    # existing rows are the trained meaning of self/goal/beam/pixel/instr/
    # measure and must carry over; the new rows start at their prior, which is
    # exactly right for a vehicle that has not yet been given a map.  Widening
    # it here is what lets one trained policy seed every arm of the map
    # experiment, so the arms differ by their INPUT and nothing else.
    tk = "type_emb.weight"
    if tk in sd and tk in own and sd[tk].shape[0] < own[tk].shape[0] and sd[tk].shape[1:] == own[tk].shape[1:]:
        grown = own[tk].clone()
        grown[: sd[tk].shape[0]] = sd[tk]
        sd[tk] = grown
    sd = {k: v for k, v in sd.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    # The heads may keep their prior (a vocabulary change, a checkpoint
    # without a value head); the BODY may not.  A body key that fails to
    # load means the checkpoint is for another net, and loading it silently
    # would fly the untrained prior while claiming the weights were loaded.
    heads = ("head_act", "value", "act_emb")          # vocabulary-sized: the action head, the value head, the action embedding
    body_skipped = [k for k in own if k not in sd and not k.startswith(heads)]
    if body_skipped:
        raise ValueError(f"{path}: {len(body_skipped)} body parameters do not match this net (e.g. {body_skipped[:3]}); "
                         f"a different width, depth or token layout -- refusing to load a checkpoint that would leave the body at its prior")
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected


COMPOSERS["transformer"] = TransformerComposer
