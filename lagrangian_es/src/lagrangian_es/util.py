"""Pytree helpers, seeded RNG, and config IO.

State is an opaque pytree (`dict[str, Tensor]`) whose leaves all carry a leading
batch dimension B.  Only the owning system interprets the keys; everything here
is key-agnostic on purpose -- that is what lets `rollout.py` stay plant-agnostic.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List

import torch
from torch import Tensor

from .config import Config, ESCfg, RolloutCfg

State = Dict[str, Tensor]

DTYPES = {"float32": torch.float32, "float64": torch.float64}


# --------------------------------------------------------------------------- #
# pytree ops
# --------------------------------------------------------------------------- #
def tree_where(mask: Tensor, a: State, b: State) -> State:
    """Select per batch element: `mask[i]` picks `a[i]`, else `b[i]`.

    `mask` is [B] bool; leaves may carry arbitrary trailing dims, so the mask is
    reshaped to broadcast against each leaf independently.  This is the whole
    reason state is a pytree rather than a flat vector.
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"tree_where expects a bool mask, got {mask.dtype}")
    out: State = {}
    for k, va in a.items():
        vb = b[k]
        if va.shape != vb.shape:
            raise ValueError(f"tree_where shape mismatch on '{k}': {va.shape} vs {vb.shape}")
        m = mask.reshape(mask.shape + (1,) * (va.ndim - mask.ndim))
        out[k] = torch.where(m, va, vb)
    return out


def tree_stack(states: Iterable[State], dim: int = 0) -> State:
    """Stack a sequence of states into one state whose leaves gain a new axis."""
    states = list(states)
    if not states:
        raise ValueError("tree_stack got an empty sequence")
    keys = states[0].keys()
    return {k: torch.stack([s[k] for s in states], dim=dim) for k in keys}


def tree_clone(s: State) -> State:
    return {k: v.clone() for k, v in s.items()}


def tree_detach(s: State) -> State:
    return {k: v.detach() for k, v in s.items()}


def tree_index(s: State, idx) -> State:
    """Index every leaf along its leading (batch) dimension."""
    return {k: v[idx] for k, v in s.items()}


def tree_repeat(s: State, n: int) -> State:
    """Tile every leaf `n` times along the batch dim, block-wise.

    `tree_repeat({'p': [a, b]}, 3)` -> `[a, b, a, b, a, b]`.  Combined with
    `Tensor.repeat_interleave` on the genomes this is what implements common
    random numbers: genome i, episode e lands at index i * n_eps + e.
    """
    return {k: v.repeat((n,) + (1,) * (v.ndim - 1)) for k, v in s.items()}


def tree_allclose(a: State, b: State, atol: float = 0.0) -> bool:
    return all(torch.allclose(a[k], b[k], atol=atol, rtol=0.0) for k in a)


# --------------------------------------------------------------------------- #
# rng
# --------------------------------------------------------------------------- #
def make_gen(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) % (2**63 - 1))
    return g


def gen_seed(base: int, *parts: int) -> int:
    """Deterministic sub-seed.  Mixes so that (0, 1) and (1, 0) differ."""
    h = int(base) & 0xFFFFFFFF
    for p in parts:
        h = (h * 1_000_003 + int(p) * 2_654_435_761 + 0x9E3779B9) & 0xFFFFFFFF
    return h


def randn(shape, gen: torch.Generator, dtype=torch.float64, device="cpu") -> Tensor:
    return torch.randn(*shape, generator=gen, dtype=dtype, device=device)


def rand(shape, gen: torch.Generator, dtype=torch.float64, device="cpu") -> Tensor:
    return torch.rand(*shape, generator=gen, dtype=dtype, device=device)


def uniform(shape, lo: float, hi: float, gen: torch.Generator,
            dtype=torch.float64, device="cpu") -> Tensor:
    return lo + (hi - lo) * rand(shape, gen, dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
# config IO  (kept out of config.py, which must stay logic-free)
# --------------------------------------------------------------------------- #
def config_from_dict(d: dict) -> Config:
    d = dict(d or {})
    ro = RolloutCfg(**d.pop("rollout", {}) or {})
    es = ESCfg(**d.pop("es", {}) or {})
    return Config(rollout=ro, es=es, **d)


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def load_config(path: str) -> Config:
    import yaml

    with open(path) as fh:
        return config_from_dict(yaml.safe_load(fh))


def replace_cfg(cfg: Config, **kw) -> Config:
    """Shallow override, with `rollout.*` / `es.*` dotted keys supported."""
    ro_kw = {k.split(".", 1)[1]: v for k, v in kw.items() if k.startswith("rollout.")}
    es_kw = {k.split(".", 1)[1]: v for k, v in kw.items() if k.startswith("es.")}
    top = {k: v for k, v in kw.items() if "." not in k}
    out = cfg
    if ro_kw:
        out = dataclasses.replace(out, rollout=dataclasses.replace(out.rollout, **ro_kw))
    if es_kw:
        out = dataclasses.replace(out, es=dataclasses.replace(out.es, **es_kw))
    return dataclasses.replace(out, **top) if top else out


def torch_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise KeyError(f"unknown dtype {name!r}; expected one of {sorted(DTYPES)}")
    return DTYPES[name]
