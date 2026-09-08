"""A task-level layer above the evolved controller.

The low level flies legs of about 10 m at ~3% crash and falls apart at 20 m,
because routing AROUND a building is not something a gradient field can hold --
94% of its stalls are saddles.  The composer never asks it to.  It emits a
`TaskSpec` -- a bounded subgoal delta plus per-term gates and priorities -- and
the low level compiles that into a conic combination of the terms it already
has.  Navigation lives up here where a sequence model can learn it; the
certificate stays below, where it already holds.
"""
from .spec import TaskSpec, SpecHold, ground_release, stale_fallback
from .base import Composer, FixedWeights, make_composer
from .oracle import OracleSubgoal
from .transformer import ComposerNet, TransformerComposer
from .tokens import Tokenizer
from .distill import Recorder, collate, fit
from .policy import PolicyComposer, PolicyNet, ppo_update, returns_from_stream

__all__ = ["TaskSpec", "SpecHold", "ground_release", "stale_fallback",
           "Composer", "FixedWeights", "make_composer", "OracleSubgoal", "ComposerNet", "TransformerComposer", "Tokenizer", "Recorder", "collate", "fit", "PolicyComposer", "PolicyNet", "ppo_update", "returns_from_stream"]
