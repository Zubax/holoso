"""Stateless AST/walrus helpers and naming utilities shared by the front-end lowerer."""

import ast
from collections.abc import Iterator

Path = list[int | str]

# A static ``for`` loop with at most this many trips fully unrolls; a larger count is rejected (a counted back-edge
# loop would need a runtime integer counter, which is not implemented -- use a ``while`` for a variable trip count).
UNROLL_THRESHOLD = 64


def range_trip_count(trips: range) -> int:
    """
    The number of iterations in a ``range`` as a Python integer. ``len(range(...))`` raises ``OverflowError`` once the
    count exceeds a C ``ssize_t`` (e.g. ``range(10**40)``); this computes it with big integers so an enormous static
    loop is cleanly rejected against the unroll threshold rather than crashing the compiler.
    """
    span = (trips.stop - trips.start) if trips.step > 0 else (trips.start - trips.stop)
    return max(0, (span + abs(trips.step) - 1) // abs(trips.step))


def port_name(path: Path) -> str:
    """Map a returned leaf path to its output-port name, e.g. ``[0, "x"]`` -> ``out_0_x``."""
    return "out" + "".join(f"_{key}" for key in path)


def state_port_name(slot: str) -> str:
    """Map a public state slot to its observable port name, e.g. ``"y"`` -> ``state_y``, ``"x_0"`` -> ``state_x_0``."""
    return f"state_{slot}"


def leaf_targets(target: ast.expr) -> Iterator[ast.expr]:
    """Yield an assignment target's leaf targets, descending through tuple/list/starred unpacking."""
    match target:
        case ast.Starred(value=value):
            yield from leaf_targets(value)
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            for elt in elts:
                yield from leaf_targets(elt)
        case _:
            yield target


def contains_walrus(node: ast.AST) -> bool:
    """
    Whether an expression subtree contains a walrus ``:=`` (the subset has no nested scope in expression position).
    """
    return any(isinstance(sub, ast.NamedExpr) for sub in ast.walk(node))


def walrus_target_names(node: ast.AST) -> set[str]:
    """
    The target names of every walrus ``(name := value)`` in an expression subtree (a walrus target is always a plain
    name -- Python forbids targeting an attribute or subscript).
    """
    return {
        sub.target.id for sub in ast.walk(node) if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name)
    }


def statement_walrus_names(stmt: ast.stmt) -> set[str]:
    """
    The walrus target names bound when ``stmt`` is reached: those in its OWN expressions (an ``if``/``while`` test, a
    ``for`` iterable, an assignment/return value) but NOT in the bodies of a nested ``if``/``for``/``while`` (those are
    scanned at their own reachability). Lowering binds a walrus target like any assignment, so the local-name and
    reachability scans register these to stay in lockstep with it.
    """
    match stmt:
        case ast.If(test=expr) | ast.While(test=expr) | ast.For(iter=expr):
            return walrus_target_names(expr)
        case ast.Assign(value=expr) | ast.AugAssign(value=expr) | ast.Expr(value=expr):
            return walrus_target_names(expr)
        case ast.AnnAssign(value=expr) | ast.Return(value=expr) if expr is not None:
            return walrus_target_names(expr)
        case _:
            return set()


def scope_local_walrus_targets(node: ast.AST) -> set[str]:
    """
    Every walrus target name bound in ``node``'s OWN scope, descending into nested statements/expressions but NOT into a
    nested function/lambda/comprehension/class (a walrus there binds in that scope, not this one). Unlike the
    reachability scans, this is purely syntactic: a walrus target is a function local throughout the body -- as in
    Python -- even inside a dead or out-of-subset statement, so it shadows a same-named global everywhere.
    """
    names: set[str] = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            # The body is a separate scope, but a nested def/lambda/class's default-argument, decorator, and base
            # expressions execute in THIS scope, so a walrus in one of them binds an enclosing local (comprehensions
            # are out of subset). The body itself is not descended into.
            enclosing: list[ast.expr] = []
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                enclosing += [d for d in (*child.args.defaults, *child.args.kw_defaults) if d is not None]
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                enclosing += child.decorator_list
            if isinstance(child, ast.ClassDef):
                enclosing += child.bases
            for expr in enclosing:
                names |= walrus_target_names(expr)
            continue
        if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
            names.add(child.target.id)
        names |= scope_local_walrus_targets(child)
    return names
