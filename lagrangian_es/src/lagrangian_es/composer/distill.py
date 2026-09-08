"""Distilling the learned composer from the oracle.

The oracle is a teacher.  `Recorder` wraps it: at every interval it tokenizes
the context exactly as the transformer would, lets the oracle answer, and stores
the pair with the target expressed in the ego frame and divided by the reach --
the same normalisation the transformer's head produces, so the student's loss is
on the quantity its output actually is.  `fit` then trains `ComposerNet` on
those pairs.  Gates and priorities are learned too; the oracle only ever says
"identity" for them, so this stage teaches the student where to go and leaves
the rest to the objective-driven stage that follows.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from .base import COMPOSERS, Composer
from .tokens import F, Tokenizer, to_ego
from .transformer import ComposerNet


class Recorder(Composer):
    """Delegates to a teacher and records (tokens, target) at every emit."""
    kind = "recorder"

    def __init__(self, system, trainable, teacher: Composer, reach: float = 10.0,
                 k_chain: int = 8, **kw):
        super().__init__(system, trainable)
        self.teacher, self.reach = teacher, float(reach)
        self.every = getattr(teacher, "every", 50)
        span = float(getattr(system.env, "span", 32.0)) if hasattr(system, "env") else 32.0
        self.tok = Tokenizer(scale=span, reach=self.reach, k_chain=k_chain)

    def attach(self, sensors):
        self.tok.attach(sensors)
        self._instr: List[Tuple[float, Tensor, Tensor]] = []
        self.samples: List[Dict[str, Tensor]] = []

    def reset(self, B):
        self.teacher.reset(B); self._instr = []; self._B = int(B); self._rows = None

    def emit(self, ctx):
        # asked about a subset of rows (`_rows`, as decisions are events):
        # the memory is kept at full width and read for those rows
        rows = getattr(self, "_rows", None)
        instr = self._instr if rows is None else [(t, d[rows], w[rows]) for t, d, w in self._instr]
        tok = self.tok(ctx, instr)
        spec = self.teacher.emit(ctx)
        sub_world = ctx["goal"] + spec.delta
        sub_ego = to_ego(sub_world - ctx["x"], tok["psi"]) / self.reach
        alive = ctx["alive"]
        for b in range(ctx["x"].shape[0]):
            if not bool(alive[b]):
                continue
            self.samples.append({k: (v[b].clone() if torch.is_tensor(v) else v) for k, v in tok.items()}
                                | {"y_sub": sub_ego[b].clone(), "y_alpha": spec.alpha[b].clone(),
                                   "y_gate": spec.gate[b].clone()})
        d, w = spec.delta.clone(), spec.weight.clone()
        if rows is not None:
            B = int(getattr(self, "_B", 0)) or int(rows.shape[0])
            if self._instr:
                _, d0, w0 = self._instr[-1]; d0, w0 = d0.clone(), w0.clone()
            else:
                d0 = torch.zeros(B, d.shape[-1], dtype=d.dtype, device=d.device)
                w0 = torch.ones(B, w.shape[-1], dtype=w.dtype, device=w.device)
            d0[rows] = d; w0[rows] = w; d, w = d0, w0
        self._instr.append((float(ctx.get("t", 0)), d, w))
        return spec


def collate(samples: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Pad entities and chain to the batch maximum; masks carry the truth."""
    B = len(samples)
    ke = max(s["entities"].shape[0] for s in samples)
    kc = max(s["chain"].shape[0] for s in samples)
    dt = samples[0]["self"].dtype
    out = {"self": torch.stack([s["self"] for s in samples]),
           "goal": torch.stack([s["goal"] for s in samples]),
           "entities": torch.zeros(B, ke, F, dtype=dt), "ent_mask": torch.zeros(B, ke, dtype=torch.bool),
           "ent_types": torch.zeros(B, ke, dtype=torch.long),
           "chain": torch.zeros(B, kc, F, dtype=dt), "chain_types": torch.zeros(B, kc, dtype=torch.long),
           "psi": torch.stack([s["psi"] for s in samples]),
           "y_sub": torch.stack([s["y_sub"] for s in samples]),
           "y_alpha": torch.stack([s["y_alpha"] for s in samples]),
           "y_gate": torch.stack([s["y_gate"] for s in samples])}
    for i, s in enumerate(samples):
        n = s["entities"].shape[0]; out["entities"][i, :n] = s["entities"]; out["ent_mask"][i, :n] = True
        out["ent_types"][i, :n] = s["ent_types"]
        m = s["chain"].shape[0]
        if m:
            out["chain"][i, :m] = s["chain"]; out["chain_types"][i, :m] = s["chain_types"]
    return out


def fit(net: ComposerNet, samples: List[Dict[str, Tensor]], epochs: int = 20,
        batch: int = 256, lr: float = 3e-4, seed: int = 0, log=None) -> List[float]:
    """Supervised distillation.  Returns the per-epoch mean loss."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCELoss()
    hist = []
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(samples), generator=gen)
        tot, n = 0.0, 0
        for i in range(0, len(samples), batch):
            b = collate([samples[j] for j in perm[i:i + batch].tolist()])
            sub, alpha, gate = net(b)
            loss = (((sub - b["y_sub"]) ** 2).sum(-1).mean()
                    + ((alpha.log() - b["y_alpha"].clamp_min(1e-3).log()) ** 2).mean()
                    + bce(gate.clamp(1e-6, 1 - 1e-6), b["y_gate"]))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tot += float(loss) * b["self"].shape[0]; n += b["self"].shape[0]
        hist.append(tot / n)
        if log:
            log(f"  epoch {ep:>3}  loss {hist[-1]:.4f}")
    net.eval()
    return hist


COMPOSERS["recorder"] = Recorder
