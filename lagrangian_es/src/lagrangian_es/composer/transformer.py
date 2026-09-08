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

    def forward(self, x, mask=None, attn_mask=None, mem=None, mem_mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, key_padding_mask=mask, attn_mask=attn_mask, need_weights=False)[0]
        if self.cross and mem is not None:
            h = self.lnc(x)
            x = x + self.xattn(h, mem, mem, key_padding_mask=mem_mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


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
            for blk in self.chain:
                ch = blk(ch, attn_mask=causal)
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

    def __init__(self, system, trainable, reach: float = 10.0, every: int = 5,
                 measure_every: int = 5, d: int = 64, heads: int = 4, k_chain: int = 32,
                 weights: str = "", **kw):
        super().__init__(system, trainable)
        # `every`: how often the composer runs over the stream; `measure_every`:
        # how often the drone appends a measurement token.  Both default to
        # 0.1 s -- the composer monitors, it does not poll once a second.
        self.reach, self.every, self.measure_every = float(reach), int(every), int(measure_every)
        self.z_min = float(getattr(system, "z_floor", 0.0)) + 0.5
        span = float(getattr(system.env, "span", 32.0)) if hasattr(system, "env") else 32.0
        self.tok = Tokenizer(scale=span, reach=self.reach, k_chain=k_chain)
        self.k_chain = int(k_chain)
        self.net = ComposerNet(max(self.n_terms, 1), d=d, heads=heads).to(system.dtype)
        self.net.goal_gain = span / self.reach
        if weights:
            self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.eval()
        self._instr: List[Tuple[float, Tensor, Tensor]] = []     # (t, delta, weight)

    def attach(self, sensors):
        """The rollout hands over its sensors so bearings come from them."""
        self.tok.attach(sensors)

    def reset(self, B):
        self._instr = []

    def tokens(self, ctx):
        return self.tok(ctx, self._instr)

    @torch.no_grad()
    def emit(self, ctx):
        tok = self.tokens(ctx)
        sub, alpha, gate = self.net(tok)
        x, goal, psi = ctx["x"], ctx["goal"], tok["psi"]
        sub_world = x + to_world(sub * self.reach, psi)              # inside the reach ball
        # a subgoal below the floor is not in the reachable set; this is an
        # interlock like the ground release, not something to learn
        sub_world = torch.cat([sub_world[:, :2], sub_world[:, 2:].clamp_min(self.z_min)], -1)
        dpsi, yg = self.net.last_yaw
        spec = TaskSpec(delta=sub_world - goal, alpha=alpha, gate=gate, yaw=psi + dpsi, yaw_gate=yg)
        self._instr.append((float(ctx.get("t", 0)), spec.delta.clone(), spec.weight.clone()))
        if len(self._instr) > 4 * self.tok.kc:
            self._instr = self._instr[-2 * self.tok.kc:]
        return spec


COMPOSERS["transformer"] = TransformerComposer
