"""
Every ``assert`` in the package must be side-effect free. Asserts are stripped under ``python -O`` (CLAUDE.md relies on
that for invariant checks), so an assert whose expression MUTATES state silently skips the mutation under -O and
miscompiles. The canonical trap is ``assert d.setdefault(k, v) == v``, which both populates ``d`` and checks it: under
-O the populate vanishes. This guard caught exactly that in the register coalescer, where the unpopulated
pinned-register map then KeyError-ed the colorer under -O.
"""

import ast
from pathlib import Path

import holoso

# Container mutators whose call inside an assert is silently skipped under -O. ``NamedExpr`` (the walrus ``:=``) is the
# other side effect and is checked directly.
_MUTATORS = frozenset(
    {
        "setdefault", "append", "add", "pop", "popitem", "update", "discard",
        "remove", "insert", "extend", "clear", "reverse", "sort",
    }
)  # fmt: skip


def test_no_assert_has_a_side_effect() -> None:
    root = Path(holoso.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Assert):
                continue
            where = f"{path.relative_to(root.parent)}:{node.lineno}"
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.NamedExpr):
                    offenders.append(f"{where} -- a walrus binds inside an assert")
                elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in _MUTATORS:
                    offenders.append(f"{where} -- .{sub.func.attr}() mutates inside an assert")
    assert not offenders, "asserts with side effects are stripped under -O and silently miscompile:\n" + "\n".join(
        offenders
    )
