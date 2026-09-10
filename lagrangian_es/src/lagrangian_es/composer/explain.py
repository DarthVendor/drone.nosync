"""Reading the composer's decisions.

A decision is legible when three things are on the table: what the composer
SAW (its tokens), what it CHOSE (the action token, its probability against the
rest of the vocabulary, the value), and WHICH inputs moved the choice.  The last comes three ways, because each
answers a different question:

  attention   what the sub-goal query and each constraint query drew from each
              key in the read layer -- beams, camera patches, the goal, the
              self token, the stream memory.  Where it LOOKED.
  saliency    how far the chosen token's logit moves per unit change of each
              input token (a gradient norm).  What it is SENSITIVE to.
  counterfactuals  how the chosen token's probability moves with the beams
              blanked, the camera blanked, the stream forgotten, the goal
              removed.  What it DEPENDS on.

`Tracer` collects all of that at every decision of a flight.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from .base import COMPOSERS
from .policy import PolicyComposer
from .tokens import BEAM, GOAL, MEASURE, INSTR, PIXEL, SELF, to_ego, to_world


def _labels(tok, b: int) -> List[str]:
    """Human names for the read layer's keys, in memory order."""
    names = ["self", "goal"]
    ty = tok["ent_types"][b].tolist(); nb = sum(1 for t in ty if t == BEAM)
    names += [f"beam {i}" for i in range(nb)] + [f"patch {i}" for i in range(len(ty) - nb)]
    if tok["chain"].shape[1]:
        names.append("stream memory")
    return names


@torch.no_grad()
def _outputs(comp: PolicyComposer, tok, ctx, b: Optional[int] = None):
    logits, value = comp.net.pre(tok)
    p = torch.softmax(logits.float(), -1)
    chosen = logits.argmax(-1)
    vocab = comp.net.vocab
    out = {"logits": logits, "probs": p, "chosen": chosen, "value": value, "p_hold": p[:, vocab.HOLD],
           "goal_ego": to_ego(ctx["goal"] - ctx["x"], tok["psi"].to(ctx["x"].dtype))}
    if b is None:
        return out
    row = {k: v[b] for k, v in out.items()}
    top = torch.topk(p[b], min(5, p.shape[-1]))
    row["chosen_name"] = vocab.name(int(chosen[b]))
    row["top"] = [(vocab.name(int(i)), float(q)) for q, i in zip(top.values, top.indices)]
    return row


def explain_decision(comp: PolicyComposer, ctx: Dict, b: int = 0) -> Dict:
    """Everything about one decision for episode `b` of the batch in `ctx`:
    the token distribution and the chosen token, where the read layer looked,
    what the chosen token's logit is sensitive to, and how its probability
    moves with the beams, the camera, the stream or the goal blanked."""
    tok = comp.tokens(ctx)
    net = comp.net
    store = {}
    with torch.no_grad():
        logits, value = net.pre(tok, store=store)
    outs = _outputs(comp, tok, ctx, b)
    keys = _labels(tok, b)
    read = store.get("read", {}).get("cross", [])
    attn = read[-1][b] if read else None                        # [1 + n_terms, n_keys]
    attention = None
    if attn is not None:
        attention = {"keys": keys,
                     "action_query": attn[0, :len(keys)].tolist(),
                     "constraint_queries": [attn[1 + i, :len(keys)].tolist() for i in range(attn.shape[0] - 1)]}
    chosen = int(outs["chosen"])
    one = {k: (v[b:b + 1].clone() if torch.is_tensor(v) else v) for k, v in tok.items()}
    with torch.enable_grad():                       # a rollout runs under no_grad; saliency needs it back
        for k in ("self", "goal", "entities", "chain"):
            one[k] = one[k].detach().requires_grad_(True)
        l1, _ = net.pre(one)
        grads = torch.autograd.grad(l1[0, chosen], [one[k] for k in ("self", "goal", "entities", "chain")], allow_unused=True)
    g_self, g_goal, g_ent, g_chain = [(g.norm(dim=-1) if g is not None else None) for g in grads]
    saliency = {"self": float(g_self[0]) if g_self is not None else 0.0,
                "goal": float(g_goal[0]) if g_goal is not None else 0.0,
                "entities": (g_ent[0].tolist() if g_ent is not None else []),
                "chain": (g_chain[0].tolist() if g_chain is not None else [])}
    base = float(outs["probs"][chosen])
    cf = {}
    ty = tok["ent_types"]
    def run_masked(mask_fn):
        t2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in tok.items()}
        mask_fn(t2)
        with torch.no_grad():
            l2, _ = net.pre(t2)
        return float(torch.softmax(l2.float(), -1)[b, chosen])
    cf["no_beams"] = run_masked(lambda t: t["ent_mask"].masked_fill_(ty == BEAM, False)) - base
    cf["no_camera"] = run_masked(lambda t: t["ent_mask"].masked_fill_(ty == PIXEL, False)) - base
    def drop_chain(t):
        t["chain"] = t["chain"][:, :0]; t["chain_types"] = t["chain_types"][:, :0]
        if "chain_mask" in t: t["chain_mask"] = t["chain_mask"][:, :0]
    cf["no_stream"] = run_masked(drop_chain) - base
    cf["no_goal"] = run_masked(lambda t: t["goal"].zero_()) - base
    return {"t": float(ctx.get("t", 0)), "tokens": {k: (v[b].detach() if torch.is_tensor(v) else v) for k, v in tok.items()},
            "outputs": {k: (v.detach() if torch.is_tensor(v) else v) for k, v in outs.items()},
            "attention": attention, "saliency": saliency, "counterfactual_shift_p": cf, "keys": keys}


class Tracer(PolicyComposer):
    """The policy composer, deterministic, explaining every decision of episode
    `watch` as it goes.  Slower than flying; meant for one flight at a time."""
    kind = "tracer"

    def __init__(self, system, trainable, watch: int = 0, **kw):
        super().__init__(system, trainable, **kw)
        self.watch = int(watch); self.trace: List[Dict] = []

    def reset(self, B):
        super().reset(B); self.trace = []

    def emit(self, ctx):
        self.stochastic = False
        if bool(ctx["alive"][self.watch]):
            self.trace.append(explain_decision(self, ctx, self.watch))
        return super().emit(ctx)


COMPOSERS["tracer"] = Tracer
