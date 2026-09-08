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
from .tokens import F, N_TYPES, Tokenizer, to_world


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
    def __init__(self, n_terms: int, d: int = 64, heads: int = 4, n_scene: int = 2,
                 n_chain: int = 2, n_read: int = 1, n_types: int = N_TYPES):
        super().__init__()
        self.embed = nn.Linear(F, d); self.type_emb = nn.Embedding(n_types, d)
        self.scene = nn.ModuleList([Block(d, heads) for _ in range(n_scene)])
        self.chain = nn.ModuleList([Block(d, heads) for _ in range(n_chain)])
        self.constraint = nn.Parameter(torch.randn(n_terms, d) * 0.02)
        self.pool = nn.Parameter(torch.randn(1, d) * 0.02)          # reads the subgoal
        self.read = nn.ModuleList([Block(d, heads, cross=True) for _ in range(n_read)])
        self.ln = nn.LayerNorm(d)
        self.head_w = nn.Linear(d, 2)                                # per constraint: alpha, gate
        self.head_sub = nn.Linear(d, 3)                              # from the pool token
        self.head_yaw = nn.Linear(d, 2)                              # heading delta (ego), gate logit
        self.n_terms = n_terms
        # The IDENTITY prior.  Fresh heads emit random term weights -- a gate
        # of 0.3 runs the low level at 30% strength -- and a random sub-goal,
        # and measured, that killed every flight (0.000 reach / 1.000 crash
        # against 0.227 / 0.773 with no composer).  So the heads start at the
        # composer Stage A trained with: unit priorities, open gates, and a
        # sub-goal toward the goal; learning is a deviation from that.
        nn.init.zeros_(self.head_w.weight); nn.init.zeros_(self.head_sub.weight); nn.init.zeros_(self.head_yaw.weight)
        with torch.no_grad():
            self.head_w.bias.copy_(torch.tensor([math.log(math.expm1(1.0)), 4.0]))   # softplus -> 1, sigmoid -> 0.98
            self.head_sub.bias.zero_()
            # heading: no delta, gate ~ 0.02 -> the plant keeps its look-at until the composer takes over
            self.head_yaw.bias.copy_(torch.tensor([0.0, -4.0]))

    goal_gain = 4.0    # scale/reach, so the residual base is the goal in reach units

    def forward(self, tok: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        B = tok["self"].shape[0]
        te = self.type_emb
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
            for i, blk in enumerate(self.chain):
                ch = blk(ch, attn_mask=causal, last=i == len(self.chain) - 1)
            mem = torch.cat([mem, ch[:, -1:]], 1)                   # the latest chain state
            mmask = torch.cat([mmask, torch.zeros(B, 1, dtype=torch.bool, device=ch.device)], 1)
        q = torch.cat([self.pool.expand(B, -1, -1), self.constraint.expand(B, -1, -1)], 1)
        for blk in self.read:
            q = blk(q, mem=mem, mem_mask=mmask)
        q = self.ln(q)
        # Radial tanh: a per-component tanh bounds a CUBE whose corner sits at
        # sqrt(3) * reach, and a sample was measured 5.6 m underground.  Bounding
        # the norm keeps the subgoal inside the reach BALL by construction.
        # residual on the goal direction: the goal token's first three
        # features are the ego-frame goal offset over the scale; at init the
        # head is zero and the sub-goal points at the goal
        g = tok["goal"][:, :3] * self.goal_gain
        h = self.head_sub(q[:, 0]) + g
        n = h.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        sub = torch.tanh(n) * h / n                                  # |sub| < 1, ego frame
        aw = self.head_w(q[:, 1:])
        alpha = nn.functional.softplus(aw[..., 0]) + 1e-3
        gate = torch.sigmoid(aw[..., 1])
        hy = self.head_yaw(q[:, 0])
        self.last_yaw = (math.pi * torch.tanh(hy[:, 0]), torch.sigmoid(hy[:, 1]))   # ego delta, gate
        return sub, alpha, gate


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
            self.net.load_state_dict(torch.load(weights, map_location="cpu"))
            self._weights_mtime = os.path.getmtime(weights)
        self.net.eval()
        self._instr: List[Tuple[float, Tensor, Tensor]] = []     # (t, delta, weight)

    def attach(self, sensors):
        """The rollout hands over its sensors so bearings come from them."""
        self.tok.attach(sensors)

    def reset(self, B):
        self._instr = []; self._B = int(B); self._rows = None

    def tokens(self, ctx):
        rows = getattr(self, "_rows", None)
        instr = self._instr if rows is None else [(t, d[rows], w[rows]) for t, d, w in self._instr]
        tok = self.tok(ctx, instr)
        return {k: (v.to(self.net_dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in tok.items()}

    @torch.no_grad()
    def emit(self, ctx):
        tok = self.tokens(ctx)
        sub, alpha, gate = self.net(tok)
        sub, alpha, gate = sub.to(self.dtype), alpha.to(self.dtype), gate.to(self.dtype)
        x, goal, psi = ctx["x"], ctx["goal"], tok["psi"].to(self.dtype)
        sub_world = x + to_world(sub * self.reach, psi)              # inside the reach ball
        # a subgoal below the floor is not in the reachable set; this is an
        # interlock like the ground release, not something to learn
        sub_world = torch.cat([sub_world[:, :2], sub_world[:, 2:].clamp_min(self.z_min)], -1)
        dpsi, yg = (v.to(self.dtype) for v in self.net.last_yaw)
        spec = TaskSpec(delta=sub_world - goal, alpha=alpha, gate=gate, yaw=psi + dpsi, yaw_gate=yg)
        self._remember(ctx, spec)
        return spec

    def _remember(self, ctx, spec):
        """Append this instruction to the per-row memory.  When the rollout
        asked only about live rows (`_rows` set), scatter into a full-batch
        record so every row's history keeps its own shape."""
        rows = getattr(self, "_rows", None)
        if rows is None:
            d, w = spec.delta.clone(), spec.weight.clone()
        else:
            if self._instr:
                _, d, w = self._instr[-1]; d, w = d.clone(), w.clone()
            else:
                d = torch.zeros(self._B, spec.delta.shape[-1], dtype=spec.delta.dtype, device=spec.delta.device)
                w = torch.ones(self._B, spec.weight.shape[-1], dtype=spec.weight.dtype, device=spec.weight.device)
            d[rows] = spec.delta; w[rows] = spec.weight
        self._instr.append((float(ctx.get("t", 0)), d, w))
        if len(self._instr) > 4 * self.tok.kc:
            self._instr = self._instr[-2 * self.tok.kc:]


COMPOSERS["transformer"] = TransformerComposer
