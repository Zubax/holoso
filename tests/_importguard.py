"""
Static import-layering analysis for architectural guards.

Computes the transitive closure of ``holoso.*`` modules a root module imports, by parsing source (not by importing,
which would pull the whole package via its ``__init__``). Lets a test assert a forbidden layering edge -- e.g. the LIR
never reaches HIR, the oracle never reaches the layer it verifies -- holds across the WHOLE dependency subtree, not just
one module's direct imports.
"""

import ast
import importlib.util
from pathlib import Path


def transitive_holoso_imports(root_module: str) -> set[str]:
    """
    Every ``holoso.*`` module reachable from ``root_module`` through static imports (the AST, resolving relative
    imports to absolute names), excluding the root itself. A module whose source is not a ``.py`` file contributes no
    edges. Raises ``ValueError`` if ``root_module`` itself does not resolve, so a typo'd guard root fails loudly rather
    than passing vacuously on an empty closure.
    """
    if _source_and_anchor(root_module)[0] is None:
        raise ValueError(f"import-guard root {root_module!r} did not resolve to a .py module")
    seen: set[str] = set()
    pending = [root_module]
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        source, anchor = _source_and_anchor(module)
        if source is None:
            continue
        for imported in _imported_modules(ast.parse(source.read_text(encoding="utf-8")), anchor):
            if imported.startswith("holoso") and imported not in seen:
                pending.append(imported)
    return seen - {root_module}


def forbidden_imports(root_module: str, forbidden_prefix: str) -> list[str]:
    return sorted(
        module
        for module in transitive_holoso_imports(root_module)
        if module == forbidden_prefix or module.startswith(forbidden_prefix + ".")
    )


def _source_and_anchor(module: str) -> tuple[Path | None, str]:
    """The module's source path and the package to resolve its relative imports against (itself if it is a package)."""
    try:
        spec = importlib.util.find_spec(module)
    except ImportError, ValueError:
        return None, module
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None, module
    anchor = module if spec.submodule_search_locations is not None else module.rpartition(".")[0]
    return Path(spec.origin), anchor


def _imported_modules(tree: ast.Module, anchor: str) -> list[str]:
    """Absolute module names imported in ``tree``; ``anchor`` is the package its relative imports resolve against."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (
                node.module or ""
                if node.level == 0
                else importlib.util.resolve_name("." * node.level + (node.module or ""), anchor)
            )
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)  # also a `from package import submodule`
    return names
