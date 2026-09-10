"""Ego-centric tokens for the composer -- from what the vehicle perceives.

Nothing here reads the map.  The scene reaches the composer only through the
sensors: one token per range beam (its bearing in the body frame and what it
returned) and one per camera column (its bearing and the nearest depth in that
column).  The other inputs are the vehicle's own motion, the relative goal the
task hands it, and the chain of its earlier instructions with what came of
each.  A composer that could read `state["boxes/*"]` would learn to read a map
the vehicle does not carry, and a test holds this module to never doing so.

Every quantity is expressed in the yaw-aligned, gravity-aligned body frame with
distances divided by a scale, so a scene rotated about the vehicle or shifted
with it tokenizes identically and the composer is equivariant by construction.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor

SELF, GOAL, BEAM, PIXEL, INSTR, MEASURE = 0, 1, 2, 3, 4, 5
N_TYPES = 6
F = 8                                     # feature width shared by every token
ACT_SCALE = 64.0                          # an action token's id rides in feature 0 of an INSTR token, over this


def yaw_of(R: Tensor) -> Tensor:
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def to_ego(v: Tensor, psi: Tensor) -> Tensor:
    c, s = torch.cos(psi), torch.sin(psi)
    x = c * v[..., 0] + s * v[..., 1]
    y = -s * v[..., 0] + c * v[..., 1]
    return torch.stack([x, y, v[..., 2]], -1) if v.shape[-1] == 3 else torch.stack([x, y], -1)


def to_world(v: Tensor, psi: Tensor) -> Tensor:
    return to_ego(v, -psi)


class Tokenizer:
    def __init__(self, scale: float, reach: float, sensors=(), k_chain: int = 32,
                 v_scale: float = 6.0, patch: int = 2):
        self.scale, self.reach, self.kc, self.vs = float(scale), float(reach), int(k_chain), float(v_scale)
        self.patch = int(patch)
        self.layout: List[Tuple[str, str, Tensor, float]] = []   # (obs key, kind, bearings, max range)
        self.attach(sensors)

    def attach(self, sensors) -> None:
        """Read each sensor's own geometry, so bearings are never hard-coded."""
        self.layout = []
        for sen in sensors or ():
            kind = getattr(sen, "kind", getattr(sen, "name", ""))
            if hasattr(sen, "n_beams") and hasattr(sen, "spread"):
                n = int(sen.n_beams); k = torch.arange(n, dtype=torch.float64)
                az = (k / n - 0.5) * float(sen.spread)
                el = torch.tensor(list(getattr(sen, "elevations", (0.0,))), dtype=torch.float64)
                bear = torch.stack([az.repeat(len(el)), el.repeat_interleave(n)], -1)
                self.layout.append((sen.name, "beam", bear, float(sen.max_range), None))
            elif hasattr(sen, "W") and hasattr(sen, "hfov"):
                # the image as PATCH tokens: every patch carries the bearing of
                # its centre and what it saw, so the composer reads pixels, not
                # a summary of them
                W, H = int(sen.W), int(getattr(sen, "H", 1)); P = self.patch
                pw, ph = max(1, W // P), max(1, H // P)
                u = ((torch.arange(pw, dtype=torch.float64) + 0.5) / pw - 0.5) * float(sen.hfov)
                v = ((torch.arange(ph, dtype=torch.float64) + 0.5) / ph - 0.5) * float(getattr(sen, "vfov", 0.0)) \
                    + float(getattr(sen, "pitch", 0.0))
                bear = torch.stack([u.repeat(ph), v.repeat_interleave(pw)], -1)      # row-major patches
                self.layout.append((sen.name, "pixel", bear, float(sen.max_range), (W, H, P)))
                continue


    def _perception(self, obs: Dict, B: int, dt, dev) -> Tuple[Tensor, Tensor, Tensor]:
        rows, types = [], []
        for key, kind, bear, rmax, geom in self.layout:
            if key not in obs:
                continue
            o = obs[key]
            if kind == "pixel":                       # [B, H*W] -> [B, ph*pw] patches: min, mean, hit fraction
                W, H, P = geom
                img = o.reshape(B, H, W)[:, : (H // P) * P, : (W // P) * P]
                pt = img.reshape(B, H // P, P, W // P, P).permute(0, 1, 3, 2, 4).reshape(B, -1, P * P)
                near = pt.min(-1).values; mean = pt.mean(-1); hit = (pt < 0.999 * rmax).to(dt).mean(-1)
            else:
                near = o.reshape(B, -1); mean = near; hit = (near < 0.999 * rmax).to(dt)
            n = near.shape[1]
            b = bear[:n].to(dtype=dt, device=dev)[None].expand(B, -1, -1)
            feat = torch.cat([torch.cos(b[..., :1]), torch.sin(b[..., :1]), torch.sin(b[..., 1:2]),
                              (near / rmax).clamp(0, 1)[..., None], (mean / rmax).clamp(0, 1)[..., None],
                              hit[..., None], torch.zeros(B, n, 2, dtype=dt, device=dev)], -1)
            rows.append(feat); types.append(torch.full((B, n), BEAM if kind == "beam" else PIXEL,
                                                       dtype=torch.long, device=dev))
        if not rows:
            return (torch.zeros(B, 0, F, dtype=dt, device=dev), torch.zeros(B, 0, dtype=torch.long, device=dev),
                    torch.zeros(B, 0, dtype=torch.bool, device=dev))
        feat = torch.cat(rows, 1); ty = torch.cat(types, 1)
        return feat, ty, torch.ones(feat.shape[:2], dtype=torch.bool, device=dev)

    def __call__(self, ctx: Dict, chain_instr: List[Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
        x, v, goal, s = ctx["x"], ctx["v"], ctx["goal"], ctx["state"]
        B = x.shape[0]; dt = x.dtype; dev = x.device
        psi = yaw_of(s["R"]) if "R" in s else torch.zeros(B, dtype=dt, device=dev)
        rel_goal = to_ego(goal - x, psi)
        prog = (1.0 - rel_goal.norm(dim=-1) / self.scale).clamp(0.0, 1.0)
        tilt = s["R"][..., 2, 2] if "R" in s else torch.ones(B, dtype=dt, device=dev)
        ve = to_ego(v, psi) / self.vs
        # view . travel: the cosine between the body's forward axis and the
        # direction of travel (0 at rest) -- is it flying where it is looking?
        sp = v.norm(dim=-1)
        fwd = s["R"][..., :, 0] if "R" in s else torch.stack([torch.cos(psi), torch.sin(psi), torch.zeros_like(psi)], -1)
        view_dot_travel = torch.where(sp > 1e-6, (fwd * v).sum(-1) / sp.clamp_min(1e-6), torch.zeros_like(sp))
        self_tok = torch.stack([ve[:, 0], ve[:, 1], ve[:, 2], x[:, 2] / self.scale, tilt, prog,
                                sp / self.vs, view_dot_travel], -1)
        goal_tok = torch.cat([rel_goal / self.scale, rel_goal.norm(dim=-1, keepdim=True) / self.scale,
                              torch.zeros(B, 4, dtype=dt, device=dev)], -1)
        per, per_types, per_mask = self._perception(ctx.get("obs", {}) or {}, B, dt, dev)
        # The chain: the drone's measurement stream and the composer's own
        # instructions, merged in time order, last `k_chain` entries.  The
        # composer runs over this window every tick, so it is monitoring the
        # stream rather than reading a summary once an interval.
        rmax = max(l[3] for l in self.layout) if self.layout else 1.0
        now = float(ctx.get("t", 0))
        # an action entry is (t, token id per row, valid per row): the rows
        # whose token it was see it, the others see padding
        events = [(m["t"], "m", m) for m in (ctx.get("chain", []) or [])] + \
                 [(e[0], "i", (e[1], e[2] if len(e) > 2 else None)) for e in chain_instr]
        events.sort(key=lambda e: (e[0], e[1] == "i"))
        ch_rows, ch_types, ch_valid = [], [], []
        for t_e, kind, val in events[-self.kc:]:
            age = torch.full((B,), (now - t_e) / 200.0, dtype=dt, device=dev)   # 4 s = 1.0
            if kind == "i":
                ids, valid = val
                ch_rows.append(torch.stack([ids.to(dt) / ACT_SCALE, age] + [torch.zeros(B, dtype=dt, device=dev)] * (F - 2), -1))
                ch_types.append(INSTR)
                ch_valid.append(torch.ones(B, dtype=torch.bool, device=dev) if valid is None else valid.to(torch.bool))
            else:
                mb = val["min_beam"] if val["min_beam"] is not None else torch.full((B,), rmax, dtype=dt, device=dev)
                ch_rows.append(torch.stack([val["progress"] / self.reach, val["remaining"] / self.reach, mb / rmax,
                                            val["alive"].to(dt), val["arrived"].to(dt), age,
                                            torch.zeros(B, dtype=dt, device=dev), torch.zeros(B, dtype=dt, device=dev)], -1))
                ch_types.append(MEASURE)
                ch_valid.append(torch.ones(B, dtype=torch.bool, device=dev))
        chain = torch.stack(ch_rows, 1) if ch_rows else torch.zeros(B, 0, F, dtype=dt, device=dev)
        ctypes = (torch.tensor(ch_types, dtype=torch.long, device=dev).expand(B, -1) if ch_rows
                  else torch.zeros(B, 0, dtype=torch.long, device=dev))
        cmask = ~torch.stack(ch_valid, 1) if ch_rows else torch.zeros(B, 0, dtype=torch.bool, device=dev)   # True = padding
        return {"self": self_tok, "goal": goal_tok, "entities": per, "ent_types": per_types, "ent_mask": per_mask,
                "chain": chain, "chain_types": ctypes, "chain_mask": cmask, "psi": psi}
