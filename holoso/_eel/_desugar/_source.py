"""
Source retrieval and location mapping for the desugarer.

Indented source (a method, a nested function) is parsed under a synthetic ``if 1:`` block instead of being
dedented: dedenting rewrites content inside multi-line string literals and breaks on continuation lines that
start at column zero, and it skews every reported column. Under the wrapper, node columns and line text index
the ORIGINAL source exactly; only line numbers carry a constant offset.
"""

import ast
import inspect
import types
from dataclasses import dataclass

from ..._errors import SourceLocation, SourceUnavailable, UnsupportedConstruct


@dataclass(frozen=True, slots=True)
class SourceUnit:
    fn: types.FunctionType
    fndef: ast.FunctionDef
    lines: list[str]
    start: int
    filename: str
    line_offset: int

    def location(self, node: ast.AST) -> SourceLocation:
        lineno = getattr(node, "lineno", 1 + self.line_offset) - self.line_offset
        col = getattr(node, "col_offset", 0)
        snippet = self.lines[lineno - 1] if 0 <= lineno - 1 < len(self.lines) else None
        return SourceLocation(self.filename, self.start + lineno - 1, _char_col(snippet, col), snippet)


def _char_col(line: str | None, byte_col: int) -> int:
    """AST column offsets count UTF-8 bytes; SourceLocation columns index characters of the line."""
    if line is None:
        return byte_col
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", "replace"))


def load(fn: types.FunctionType) -> SourceUnit:
    try:
        lines, start = inspect.getsourcelines(fn)
    except (OSError, TypeError) as exc:
        raise SourceUnavailable(
            f"cannot retrieve source for {getattr(fn, '__name__', '?')!r}; "
            "define it in an importable module (not a REPL/exec/lambda)"
        ) from exc
    filename = inspect.getsourcefile(fn) or "<unknown>"
    if fn.__name__ == "<lambda>":
        location = SourceLocation(filename, start, max(lines[0].find("lambda"), 0), lines[0])
        raise UnsupportedConstruct("lambdas are not supported; define the kernel with `def`", location)
    text = "".join(lines)
    if lines and lines[0][:1] in (" ", "\t"):
        body = _parse_block("if 1:\n" + text)
        line_offset = 1
    else:
        body = ast.parse(text).body
        line_offset = 0
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            return SourceUnit(fn, node, lines, start, filename, line_offset)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == fn.__name__:
            lineno = node.lineno - line_offset
            snippet = lines[lineno - 1]
            location = SourceLocation(filename, start + lineno - 1, _char_col(snippet, node.col_offset), snippet)
            raise UnsupportedConstruct("async functions are not supported", location)
    raise SourceUnavailable(f"could not locate a 'def {fn.__name__}' in the retrieved source")


def _parse_block(wrapped: str) -> list[ast.stmt]:
    module = ast.parse(wrapped)
    assert len(module.body) == 1 and isinstance(module.body[0], ast.If)
    return module.body[0].body
