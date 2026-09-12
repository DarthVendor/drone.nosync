"""The co-training script is ~800 lines of MODULE-LEVEL code with no other
coverage, and `py_compile` cannot catch a name used before it is bound.

Measured cost of not having this: a constant was edited in at line 72 that read
`_os.environ`, while `import os as _os` sits at line 154.  The file compiled,
parsed, and passed every existing test; it died with `NameError: name '_os' is
not defined` the moment a run was launched, after the pool had forked and the
work directory had been prepared.

This walks the module's top level in order and fails on any name that is READ
before anything at the top level binds it -- which is exactly that bug, and is
cheap enough to run on every suite.
"""
import ast
import builtins
import pathlib

import pytest

SCRIPTS = sorted((pathlib.Path(__file__).resolve().parents[1] / "scripts" / "composer").glob("cotrain_v*.py"))


def _locals_of(node):
    """Names bound by comprehensions and lambdas inside this statement.

    Those have their own scope, so a load of one is never a module-level read;
    counting them produced false positives on every `[f(t) for t in xs]`.
    """
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, ast.Lambda):
            a = n.args
            for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                out.add(arg.arg)
            for extra in (a.vararg, a.kwarg):
                if extra is not None:
                    out.add(extra.arg)
    return out


def _bound_by(node):
    """Every name this top-level statement binds."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.alias):
            out.add((n.asname or n.name).split(".")[0])
    return out


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_module_level_name_is_read_before_it_is_bound(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    known = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    bad = []
    for stmt in tree.body:
        # a def/class body runs later, so only its decorators and defaults read now
        scan = stmt
        if isinstance(scan, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scan = ast.Module(body=list(getattr(scan, "decorator_list", [])), type_ignores=[])
        # a statement may bind a name and then read it -- a `for` loop body does
        # exactly that across iterations -- so its own bindings count as known
        # for reads inside it.  The bug this guards against is a read in ONE
        # top-level statement of a name bound only by a LATER one.
        here = known | _bound_by(stmt) | _locals_of(stmt)
        for n in ast.walk(scan):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in here:
                bad.append((n.lineno, n.id))
        known |= _bound_by(stmt)
    assert not bad, (
        f"{path.name}: name(s) read at module level before being bound -- "
        + ", ".join(f"{nm!r} at line {ln}" for ln, nm in bad[:5])
    )
