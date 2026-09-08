"""Composers: things that emit a `TaskSpec` from context, once an interval.

`FixedWeights` is the identity -- no subgoal, every term open at unit priority.
It exists so that Stage A (evolving the low level) runs with the composer in
place and provably out of the way: compiling its spec must reproduce the bare
controller bit for bit, and a test holds it to that.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from .spec import TaskSpec


class Composer:
    """Interface.  `emit` sees the context dict the rollout assembles once per
    interval and returns a target `TaskSpec` for every episode in the batch."""
    kind = "composer"
    # the rollout may ask only about rows still flying; cheap composers (the
    # identity, the oracle, the recorder) take the whole batch every time
    live_only = False

    def __init__(self, system, trainable, **kw):
        self.system, self.trainable = system, trainable
        self.d = int(system.task_dim)
        self.n_terms = len(getattr(trainable, "terms", []))

    def emit(self, ctx: Dict[str, Tensor]) -> TaskSpec:
        raise NotImplementedError

    def reset(self, B: int) -> None:
        """Called at the start of a batch; stateful composers clear here."""


class FixedWeights(Composer):
    kind = "fixed"

    def emit(self, ctx):
        B = ctx["x"].shape[0]
        return TaskSpec.identity(B, self.d, self.n_terms, ctx["x"].dtype,
                                 ctx["x"].device)


COMPOSERS = {"fixed": FixedWeights}


def make_composer(name: str, system, trainable, **kw) -> Composer:
    try:
        cls = COMPOSERS[name]
    except KeyError:
        raise KeyError(f"unknown composer {name!r}; have {sorted(COMPOSERS)}")
    return cls(system, trainable, **kw)
