"""The per-function lowering scope, its branch-arm result, and source-to-AST parsing for the front-end lowerer."""

import ast
import functools
import inspect
import textwrap
import types
from dataclasses import dataclass

from .._errors import SourceUnavailable
from ._aggregate import Value


@dataclass(frozen=True, slots=True)
class ArmResult:
    env: dict[str, Value]
    state: dict[str, Value]
    static_ints: dict[str, int]
    end_block: int


@dataclass(frozen=True, slots=True)
class Scope:
    """The per-function lowering state, captured and restored as a unit when a callee is inlined into a fresh scope."""

    fn: types.FunctionType
    env: dict[str, Value]
    static_ints: dict[str, int]
    return_: Value | None
    in_branch: int
    instance: object | None
    self_name: str | None
    snapshot: dict[str, object]
    state_order: list[str]
    state_env: dict[str, Value]
    lines: list[str]
    start: int
    filename: str

    @classmethod
    def fresh(
        cls,
        fn: types.FunctionType,
        env: dict[str, Value],
        lines: list[str],
        start: int,
        filename: str,
        *,
        context: "Scope | None" = None,
        self_name: str | None = None,
    ) -> "Scope":
        """
        A scope for lowering a callee: the given parameter bindings and source, with a fresh return slot and no
        inherited loop-counter bindings. A pure function gets NO state context (``context`` is None). An instance method
        passes the caller's scope as ``context`` to inherit its instance/snapshot/state -- so the method's
        ``self.<attr>`` reads resolve against the same module -- with ``self_name`` bound to the method's own receiver.
        The branch depth also restarts: whether the callee's own body guards a statement behind a data-dependent test is
        a property of the callee, not of the call site that happens to sit inside a branch arm.
        """
        return cls(
            fn=fn,
            env=env,
            static_ints={},
            return_=None,
            in_branch=0,
            instance=context.instance if context is not None else None,
            self_name=self_name,
            snapshot=context.snapshot if context is not None else {},
            state_order=context.state_order if context is not None else [],
            state_env=context.state_env if context is not None else {},
            lines=lines,
            start=start,
            filename=filename,
        )


@functools.lru_cache(maxsize=1024)  # bounded: a long-lived process must not pin every kernel it ever lowered
def parse_fndef(fn: types.FunctionType) -> tuple[ast.FunctionDef, list[str], int, str]:
    """
    Memoized because inlining reparses: one matrix product inlines its dot-product helper once per output element,
    so an unmemoized getsourcelines+parse would make lowering quadratic in the operand shapes. The result is treated
    as immutable by every caller. A raised SourceUnavailable is not cached, so a fixed source is picked up.
    """
    try:
        lines, start = inspect.getsourcelines(fn)
    except (OSError, TypeError) as exc:
        raise SourceUnavailable(
            f"cannot retrieve source for {getattr(fn, '__name__', '?')!r}; "
            "define it in an importable module (not a REPL/exec/lambda)"
        ) from exc
    module = ast.parse(textwrap.dedent("".join(lines)))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            return node, lines, start, inspect.getsourcefile(fn) or "<unknown>"
    raise SourceUnavailable(f"could not locate a 'def {fn.__name__}' in the retrieved source")
