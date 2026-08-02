"""
Architectural guards for the Eel package: the two frontends never import each other, the desugarer's closure is
syntax-only, and numpy and HIR stay confined to their sanctioned homes. The two `_lib` trees diverge freely
(maintainer ruling): the Eel copy is the living one, the old frontend's dies at cutover.
"""

import ast
from pathlib import Path

from ._importguard import forbidden_imports, transitive_holoso_imports

_EEL_DIR = Path(__file__).resolve().parents[1] / "holoso" / "_eel"

# Modules allowed to import numpy / HIR within holoso._eel, as repo-relative posix paths under _eel.
# `_pe/_ops.py` is the partial evaluator's sole HIR-facing module: it selects operators and folds through their
# own `evaluate` (the one-answer mandate), while the rest of `_pe` stays HIR-free; `_lower.py` composes the
# stages and returns an `Hir`. `_pe/_state.py` is the reset side of the capture boundary: it walks raw instance
# attributes into slot specs and owns the A5 storage-overlap checks, so it shares the snapshot's numpy license.
# `_pe/_interpret.py` references np.ndarray for descriptor-identity method dispatch (maintainer ruling).
_NUMPY_HOMES = {"_lib", "_pe/_snapshot.py", "_pe/_state.py", "_pe/_interpret.py"}
_HIR_HOMES = {"_lib", "_emit", "_pe/_ops.py", "_lower.py"}


def test_eel_never_imports_old_frontend() -> None:
    assert forbidden_imports("holoso._eel", "holoso._frontend") == []


def test_old_frontend_never_imports_eel() -> None:
    assert forbidden_imports("holoso._frontend", "holoso._eel") == []


def test_desugar_closure_is_syntax_only() -> None:
    closure = transitive_holoso_imports("holoso._eel._desugar")
    allowed = ("holoso._eel._desugar", "holoso._eel._ir", "holoso._errors")
    stray = {m for m in closure if not any(m == p or m.startswith(p + ".") for p in allowed)}
    assert not stray, stray


def test_desugar_never_reaches_hir_or_backends() -> None:
    for forbidden in ("holoso._hir", "holoso._mir", "holoso._lir", "holoso._backend"):
        assert forbidden_imports("holoso._eel._desugar", forbidden) == []


def test_eel_never_reaches_below_hir() -> None:
    for forbidden in ("holoso._mir", "holoso._lir", "holoso._backend"):
        assert forbidden_imports("holoso._eel", forbidden) == []


def _unsanctioned_modules(homes: set[str]) -> list[Path]:
    return [
        path
        for path in sorted(_EEL_DIR.rglob("*.py"))
        if path.relative_to(_EEL_DIR).parts[0] not in homes and path.relative_to(_EEL_DIR).as_posix() not in homes
    ]


def _imported_names(path: Path) -> set[str]:
    """
    Imported module names: absolute for plain imports, the dotless suffix for relative ones. From-import
    aliases are recorded too, so `from .. import _hir` and `from holoso import _hir` cannot slip past.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names |= {f"{base}.{alias.name}" if base else alias.name for alias in node.names}
    return names


def test_numpy_only_in_sanctioned_homes() -> None:
    for path in _unsanctioned_modules(_NUMPY_HOMES):
        for name in _imported_names(path):
            assert name != "numpy" and not name.startswith("numpy."), f"numpy import in {path.name}"


def test_hir_only_in_sanctioned_homes() -> None:
    for path in _unsanctioned_modules(_HIR_HOMES):
        for name in _imported_names(path):
            assert "_hir" not in name.split("."), f"HIR import in {path.name}"
