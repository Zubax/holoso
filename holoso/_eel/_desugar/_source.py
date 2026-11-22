"""
Source retrieval and location mapping for the desugarer.

The source is the text the function's own code object was compiled from: its file, its first line, and the name it
was defined under. Neither a reassigned `__name__` nor a `__wrapped__` attribute redirects the lookup, so a wrapper
compiles as the wrapper it is.

Indented source (a method, a nested function) is parsed under a synthetic `if 1:` block instead of being
dedented: dedenting rewrites content inside multi-line string literals and breaks on continuation lines that
start at column zero, and it skews every reported column. Under the wrapper, node columns and line text index
the ORIGINAL source exactly; only line numbers carry a constant offset.
"""

import ast
import inspect
import linecache
import tokenize
import types
from dataclasses import dataclass

from ..._errors import SourceLocation, SourceUnavailable, UnsupportedConstruct
from ._classify import mangles


@dataclass(frozen=True, slots=True)
class SourceUnit:
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
    code = fn.__code__
    filename = inspect.getsourcefile(fn) or "<unknown>"
    try:
        # Not `inspect.getsourcelines`, which follows `__wrapped__` to another function's text.
        source, index = inspect.findsource(fn)
    except (OSError, TypeError) as exc:
        if linecache.getline(code.co_filename, 1):
            raise _changed(code, filename) from exc  # there is text, but it ends before the function's line
        raise SourceUnavailable(
            f"cannot retrieve source for {code.co_qualname!r} at {code.co_filename!r}; "
            "a runtime-generated function needs its text registered in linecache under that filename"
        ) from exc
    start = index + 1
    if code.co_name == "<lambda>":
        location = SourceLocation(filename, start, max(source[index].find("lambda"), 0), source[index])
        raise UnsupportedConstruct("lambdas are not supported; define the kernel with `def`", location)
    try:
        lines = inspect.getblock(source[index:])
        text = "".join(lines)
        if lines and lines[0][:1] in (" ", "\t"):
            body = _parse_block("if 1:\n" + text)
            line_offset = 1
        else:
            body = ast.parse(text).body
            line_offset = 0
    except (SyntaxError, tokenize.TokenError) as exc:
        raise _changed(code, filename) from exc
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == code.co_name and _declares(node, code):
            return SourceUnit(node, lines, start, filename, line_offset)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == code.co_name:
            lineno = node.lineno - line_offset
            snippet = lines[lineno - 1]
            location = SourceLocation(filename, start + lineno - 1, _char_col(snippet, node.col_offset), snippet)
            raise UnsupportedConstruct("async functions are not supported", location)
    raise _changed(code, filename)


def _changed(code: types.CodeType, filename: str) -> SourceUnavailable:
    """
    The text is read when the function is compiled for hardware, not when Python compiled it, so a file edited in
    between holds something else at the recorded line.
    """
    return SourceUnavailable(
        f"the text at {filename}:{code.co_firstlineno} is not what {code.co_qualname!r} was compiled from; "
        "the file changed after the function was loaded, so load it again"
    )


def _declares(fndef: ast.FunctionDef, code: types.CodeType) -> bool:
    """
    Whether the definition declares the code object's parameters, each of its own kind, a private one being stored
    mangled in a class.
    """
    args = fndef.args
    kinds = (
        len(args.posonlyargs),
        len(args.args),
        len(args.kwonlyargs),
        args.vararg is not None,
        args.kwarg is not None,
    )
    compiled = (
        code.co_posonlyargcount,
        code.co_argcount - code.co_posonlyargcount,
        code.co_kwonlyargcount,
        bool(code.co_flags & inspect.CO_VARARGS),
        bool(code.co_flags & inspect.CO_VARKEYWORDS),
    )
    starred = [arg for arg in (args.vararg, args.kwarg) if arg is not None]
    declared = [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, *starred)]
    return kinds == compiled and all(
        stored == name or (mangles(name) and stored.startswith("_") and stored.endswith(name))
        for name, stored in zip(declared, code.co_varnames, strict=False)
    )


def _parse_block(wrapped: str) -> list[ast.stmt]:
    module = ast.parse(wrapped)
    assert len(module.body) == 1 and isinstance(module.body[0], ast.If)
    return module.body[0].body
