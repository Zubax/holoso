"""
The compiler's layers depend one way: MIR on HIR and the operators, LIR on MIR and the operators, the backends on
LIR and MIR. The transitive closures pin the two edges that can be pinned; LIR and the backends reach HIR through MIR
by design, so they are pinned by their direct imports instead.
"""

import ast
from pathlib import Path

import holoso

from ._importguard import _imported_modules, _source_and_anchor, forbidden_imports

_ROOT = Path(holoso.__file__).parent


def _modules(package: str) -> list[str]:
    directory = _ROOT / package
    names = set()
    for path in directory.rglob("*.py"):
        parts = path.relative_to(directory).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.add(".".join(["holoso", package, *parts]))
    return sorted(names)


def _direct_imports(module: str) -> set[str]:
    source, anchor = _source_and_anchor(module)
    assert source is not None
    return set(_imported_modules(ast.parse(source.read_text(encoding="utf-8")), anchor))


def _offending(package: str, allowed: tuple[str, ...]) -> list[str]:
    offenders = []
    for module in _modules(package):
        for imported in _direct_imports(module):
            if imported.startswith("holoso") and not imported.startswith(allowed):
                offenders.append(f"{module} -> {imported}")
    return offenders


def test_mir_never_reaches_lir() -> None:
    offenders = {module: forbidden_imports(module, "holoso._lir") for module in _modules("_mir")}
    assert not any(offenders.values()), {k: v for k, v in offenders.items() if v}


def test_hir_never_reaches_mir_or_lir() -> None:
    offenders = {
        module: forbidden_imports(module, "holoso._mir") + forbidden_imports(module, "holoso._lir")
        for module in _modules("_hir")
    }
    assert not any(offenders.values()), {k: v for k, v in offenders.items() if v}


def test_lir_imports_only_mir_and_below() -> None:
    allowed: tuple[str, ...] = ("holoso._lir", "holoso._mir", "holoso._operators", "holoso._type", "holoso._value")
    allowed += ("holoso._util", "holoso._errors", "holoso._legal")
    assert not _offending("_lir", allowed)


def test_backends_import_only_lir_and_below() -> None:
    allowed: tuple[str, ...] = ("holoso._backend", "holoso._lir", "holoso._mir", "holoso._operators", "holoso._type")
    allowed += ("holoso._value", "holoso._util", "holoso._errors", "holoso._legal")
    assert not _offending("_backend", allowed)
