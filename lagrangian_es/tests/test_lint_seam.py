"""Mechanical guard against section 7's silent failure mode: a leaky abstraction.

If anything downstream of `rollout.py` names a concrete plant or reads a raw
state key, plant-swapping breaks quietly -- the quadrotor keeps working, so
nothing looks wrong, and the arm transfer simply never runs.  That is the failure
that makes the whole `systems/` split pointless, so it is worth a test rather
than a code-review habit.
"""
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "lagrangian_es"

#: modules that must stay plant-agnostic
GENERIC = ["rollout.py", "metric.py", "operators.py", "es.py", "evaluate.py", "viz.py"]

#: concrete plants, concrete controllers, and raw state keys
FORBIDDEN = {
    "concrete system": r"\b(QuadrotorSE3|PlanarQuadrotor|TwoLinkArm)\b",
    "concrete trainable": r"\b(EnergyShaping|MLPPolicy|FixedPD)\b",
    "concrete task": r"\b(WaypointPair|JointTarget|JointPair)\b",
    "plant module": r"\bfrom\s+\.systems\.(quadrotor|planar_quad|two_link_arm)\b",
    "raw state key": r"""\[\s*['"](p|v|R|om|q|dq|th)['"]\s*\]""",
}


def _strip_comments(text: str) -> str:
    """Docstrings legitimately mention plants; only executable code must be clean."""
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    return re.sub(r"#.*", "", text)


@pytest.mark.parametrize("fname", GENERIC)
@pytest.mark.parametrize("label,pattern", sorted(FORBIDDEN.items()))
def test_generic_modules_stay_plant_agnostic(fname, label, pattern):
    path = SRC / fname
    if not path.exists():
        pytest.skip(f"{fname} not present")
    code = _strip_comments(path.read_text())
    hits = [(i + 1, ln.strip()) for i, ln in enumerate(code.splitlines())
            if re.search(pattern, ln)]
    assert not hits, (
        f"{fname} leaks a {label}: " + "; ".join(f"line {n}: {ln}" for n, ln in hits)
        + "\nDownstream code must go through LagrangianSystem's accessors."
    )


def test_systems_never_import_trainables():
    """`systems/` must not depend on `trainables/`; if a cycle appears, the seam
    is in the wrong place."""
    for path in (SRC / "systems").glob("*.py"):
        code = _strip_comments(path.read_text())
        assert "trainables" not in code, f"{path.name} imports from trainables/"
        assert not re.search(r"\bfrom\s+\.\.(rollout|metric|operators|es)\b", code), \
            f"{path.name} imports from a module above it in the dependency order"


def test_trainables_never_import_upward():
    for path in (SRC / "trainables").glob("*.py"):
        code = _strip_comments(path.read_text())
        assert not re.search(r"\bfrom\s+\.\.(rollout|metric|operators|es|evaluate)\b", code), \
            f"{path.name} imports from a module above it in the dependency order"


def test_dependency_order_is_acyclic():
    """Build order: config, util -> systems -> trainables -> tasks -> rollout ->
    metric -> operators -> es -> evaluate -> viz."""
    order = ["config", "util", "systems", "trainables", "tasks", "rollout",
             "metric", "operators", "es", "evaluate", "viz"]
    rank = {name: i for i, name in enumerate(order)}
    for name, i in rank.items():
        path = SRC / f"{name}.py"
        if not path.exists():
            path = SRC / name / "__init__.py"
        if not path.exists():
            continue
        code = _strip_comments(path.read_text())
        for other, j in rank.items():
            if j <= i:
                continue
            assert not re.search(rf"\bfrom\s+\.{other}\b", code), \
                f"{name} imports {other}, which comes later in the dependency order"
