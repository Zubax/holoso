"""Lower a Python function object into HIR."""

import ast
import builtins
import inspect
import textwrap
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._errors import MissingIntrinsic, SourceLocation, SourceUnavailable, UnsupportedConstruct
from .._hir import *

_Path = list[int | str]

_ABSENT = object()  # sentinel distinguishing a missing global from one explicitly bound to None during name resolution
_NO_PARAMETER_ANNOTATION = object()

# A static ``for`` loop with at most this many trips fully unrolls; a larger count is rejected (a counted back-edge
# loop would need a runtime integer counter, which is not implemented -- use a ``while`` for a variable trip count).
_UNROLL_THRESHOLD = 64


def _range_trip_count(trips: range) -> int:
    """
    The number of iterations in a ``range`` as a Python integer. ``len(range(...))`` raises ``OverflowError`` once the
    count exceeds a C ``ssize_t`` (e.g. ``range(10**40)``); this computes it with big integers so an enormous static
    loop is cleanly rejected against the unroll threshold rather than crashing the compiler.
    """
    span = (trips.stop - trips.start) if trips.step > 0 else (trips.start - trips.stop)
    return max(0, (span + abs(trips.step) - 1) // abs(trips.step))


# numpy array constructors that take one array-like and preserve its elements: in this compile-time model the operand is
# already an aggregate, so they lower to identity. Recognizing them lets a kernel be ordinary executable numpy code.
_NUMPY_IDENTITY = frozenset({"array", "asarray", "asanyarray"})

# Standard numeric operators that are recognized but not yet implemented; calling them fails with a clear message.
_KNOWN_INTRINSICS = frozenset(
    {
        "sqrt",
        "cbrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sincos",
        "exp",
        "log",
        "log2",
        "log10",
        "hypot",
        "floor",
        "ceil",
        "pow",
    }
)


def _port_name(path: _Path) -> str:
    """Map a returned leaf path to its output-port name, e.g. ``[0, "x"]`` -> ``out_0_x``."""
    return "out" + "".join(f"_{key}" for key in path)


def _state_port_name(slot: str) -> str:
    """Map a public state slot to its observable port name, e.g. ``"y"`` -> ``state_y``, ``"x_0"`` -> ``state_x_0``."""
    return f"state_{slot}"


def _leaf_targets(target: ast.expr) -> Iterator[ast.expr]:
    """Yield an assignment target's leaf targets, descending through tuple/list/starred unpacking."""
    match target:
        case ast.Starred(value=value):
            yield from _leaf_targets(value)
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            for elt in elts:
                yield from _leaf_targets(elt)
        case _:
            yield target


def _contains_walrus(node: ast.AST) -> bool:
    """Whether an expression subtree contains a walrus ``:=`` (the subset has no nested scope in expression position)."""
    return any(isinstance(sub, ast.NamedExpr) for sub in ast.walk(node))


def _walrus_target_names(node: ast.AST) -> set[str]:
    """The target names of every walrus ``(name := value)`` in an expression subtree (a walrus target is always a plain
    name -- Python forbids targeting an attribute or subscript)."""
    return {
        sub.target.id for sub in ast.walk(node) if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name)
    }


def _statement_walrus_names(stmt: ast.stmt) -> set[str]:
    """
    The walrus target names bound when ``stmt`` is reached: those in its OWN expressions (an ``if``/``while`` test, a
    ``for`` iterable, an assignment/return value) but NOT in the bodies of a nested ``if``/``for``/``while`` (those are
    scanned at their own reachability). Lowering binds a walrus target like any assignment, so the local-name and
    reachability scans register these to stay in lockstep with it.
    """
    match stmt:
        case ast.If(test=expr) | ast.While(test=expr) | ast.For(iter=expr):
            return _walrus_target_names(expr)
        case ast.Assign(value=expr) | ast.AugAssign(value=expr) | ast.Expr(value=expr):
            return _walrus_target_names(expr)
        case ast.AnnAssign(value=expr) | ast.Return(value=expr) if expr is not None:
            return _walrus_target_names(expr)
        case _:
            return set()


def _scope_local_walrus_targets(node: ast.AST) -> set[str]:
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
                names |= _walrus_target_names(expr)
            continue
        if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
            names.add(child.target.id)
        names |= _scope_local_walrus_targets(child)
    return names


class _Value(ABC):
    """
    A compile-time lowering value: a single scalar HIR wire or an ordered aggregate of values. Aggregates (vectors,
    matrices, tuples) are pure frontend bookkeeping over scalar registers -- per DESIGN.md they never exist as hardware
    aggregates -- so they never enter HIR; only their scalar leaves do.
    """

    @abstractmethod
    def walk(self, path: _Path) -> Iterator[tuple[_Path, ValueId]]:
        """Yield ``(path, scalar)`` leaves row-major, extending ``path`` by the aggregate index at each level."""

    def leaves(self) -> list[ValueId]:
        return [vid for _, vid in self.walk([])]

    def flatten(self) -> "_Aggregate":
        """Collapse to a flat aggregate of all scalar leaves in row-major order (the ``.flatten()`` method)."""
        return _Aggregate(tuple(_Scalar(vid) for vid in self.leaves()))

    def output_leaves(self) -> list[tuple[_Path, ValueId]]:
        """The (path, scalar) pairs naming this returned value's output ports; an aggregate uses its indexed paths."""
        return list(self.walk([]))


@dataclass(frozen=True, slots=True)
class _Scalar(_Value):
    id: ValueId

    def walk(self, path: _Path) -> Iterator[tuple[_Path, ValueId]]:
        yield list(path), self.id

    def output_leaves(self) -> list[tuple[_Path, ValueId]]:
        # A bare scalar return is out_0 (leaf position 0), not the empty-path "out", to match the multi-output and
        # reference orderings; walking a lone scalar would otherwise yield the empty path.
        return [([0], self.id)]


@dataclass(frozen=True, slots=True)
class _Aggregate(_Value):
    items: tuple[_Value, ...]

    def walk(self, path: _Path) -> Iterator[tuple[_Path, ValueId]]:
        for index, item in enumerate(self.items):
            yield from item.walk([*path, index])


@dataclass(frozen=True, slots=True)
class _StateAttr:
    """
    The scalar-slot decomposition of one instance attribute, derived from the reset snapshot: a scalar occupies a single
    bare-named slot, a vector one indexed slot per element. It is the single source of an attribute's shape -- its slot
    names, its typed reset values, and whether an assigned value must be a scalar or a same-length flat aggregate.
    ``is_bool`` marks a single-bit boolean attribute (a scalar only; boolean vectors are not supported).
    """

    is_vector: bool
    is_bool: bool
    slots: list[str]
    resets: list[Const]

    def accepts(self, value: _Value) -> bool:
        """
        Whether an assigned value matches this shape: a scalar attribute accepts only a scalar, a vector only a flat
        aggregate of the same length. Checking the full shape -- not merely the leaf count -- keeps the assigned value
        consistent with the per-element slot layout that the next transaction reconstructs from the reset snapshot.
        """
        if not self.is_vector:
            return isinstance(value, _Scalar)
        return (
            isinstance(value, _Aggregate)
            and len(value.items) == len(self.slots)
            and all(isinstance(item, _Scalar) for item in value.items)
        )

    def compose(self, scalars: tuple[_Scalar, ...]) -> _Value:
        """A scalar attribute is its single wire; a vector attribute is the aggregate of its per-element wires."""
        return _Aggregate(scalars) if self.is_vector else scalars[0]


@dataclass(frozen=True, slots=True)
class _ArmResult:
    """One branch arm's outcome: its final locals, persistent state, compile-time-integer bindings, and end block."""

    env: dict[str, _Value]
    state: dict[str, _Value]
    static_ints: dict[str, int]
    end_block: int


@dataclass(frozen=True, slots=True)
class _Scope:
    """The per-function lowering state, captured and restored as a unit when a callee is inlined into a fresh scope."""

    fn: types.FunctionType
    env: dict[str, _Value]
    static_ints: dict[str, int]
    return_: _Value | None
    instance: object | None
    self_name: str | None
    snapshot: dict[str, object]
    state_order: list[str]
    state_env: dict[str, _Value]
    lines: list[str]
    start: int
    filename: str

    @classmethod
    def fresh(
        cls, fn: types.FunctionType, env: dict[str, _Value], lines: list[str], start: int, filename: str
    ) -> "_Scope":
        """A scope for lowering a pure function: the given parameter bindings and source, with no state context."""
        return cls(
            fn=fn,
            env=env,
            static_ints={},  # the callee starts with no inherited loop-counter bindings
            return_=None,
            instance=None,
            self_name=None,
            snapshot={},
            state_order=[],
            state_env={},
            lines=lines,
            start=start,
            filename=filename,
        )


def _parse_fndef(fn: types.FunctionType) -> tuple[ast.FunctionDef, list[str], int, str]:
    """Retrieve and parse a function's ``def`` node, returning its source lines, start line, and filename."""
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


class _Lowerer:
    def __init__(self, fn: types.FunctionType, instance: object | None = None) -> None:
        self._fn = fn
        self._entry_fndef, self._lines, self._start, self._filename = _parse_fndef(fn)
        self._builder = HirBuilder()
        self._env: dict[str, _Value] = {}
        # Stateful-class lowering context; all empty/None for a plain stateless function. The snapshot is the instance
        # as handed to the synthesizer (whatever __init__ and any later mutation produced); its values seed reset.
        self._instance = instance
        self._self_name: str | None = None
        self._snapshot: dict[str, object] = dict(vars(instance)) if instance is not None else {}
        self._shapes: dict[str, _StateAttr] = {}  # per-attribute decompositions, derived once from the snapshot
        self._state_order: list[str] = []
        self._state_env: dict[str, _Value] = {}
        # The single return value, buffered as a _Value rather than emitted on sight: dropping a return that carries a
        # public attribute's value needs that attribute's live-out, settled only once the body is fully lowered.
        self._return: _Value | None = None
        # Functions currently being inlined, to reject recursion (which cannot be unrolled to straight-line dataflow).
        self._inlining: set[types.FunctionType] = set()
        # Lexical depth of dynamic branch arms currently being lowered; a top-level return inside an arm is not yet
        # supported (the single-exit invariant), but an inlined callee's own return is consumed locally and allowed.
        self._in_branch = 0
        # Compile-time integer bindings for unrolled loop counters: a counter is a static value (it indexes constant
        # tables, sets shift exponents, bounds ranges), resolved by the static-int evaluator, not a runtime register.
        self._static_ints: dict[str, int] = {}
        # Instance attributes assigned anywhere in the body (syntactic, ignoring reachability). A boolean attribute NOT
        # in this set is read-only, so a branch on it has a compile-time-known condition (see ``_static_bool``).
        self._assigned_attrs: set[str] = set()
        # True only while that read-only set is itself being built: a static fold may then not consult an attribute's
        # read-only-ness (the set is incomplete), so attribute leaves read as opaque -- yet an absorbing connective
        # (``self.flag or True``) still folds, since the absorbing operand alone decides it.
        self._scanning_readonly_attrs: bool = False
        # The reachable assignments seen so far while building ``_assigned_attrs``. Integer range bounds may consult
        # this partial set: a write that has already been reached makes the attribute non-static from that point on,
        # while a read-only integer attribute remains usable for zero-trip and nested static-range reachability.
        self._readonly_scan_assigned_attrs: set[str] | None = None
        # The names each lowered function binds (parameters and assignment targets); shadow resolution consults these.
        self._local_names: dict[types.FunctionType, set[str]] = {}

    def _loc(self, node: ast.AST) -> SourceLocation:
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = self._lines[lineno - 1] if 0 <= lineno - 1 < len(self._lines) else None
        return SourceLocation(self._filename, self._start + lineno - 1, col, snippet)

    def run(self) -> Hir:
        self._builder.block()  # the entry block (id 0); subsequent blocks are created by branch lowering
        self._bind_parameters(self._entry_fndef)
        self._lower_body(self._entry_fndef)
        self._register_state_slots()
        self._emit_outputs()
        self._builder.ret()  # seal the current (function-exit) block; the frontend emits a single Ret
        return self._builder.finish()

    def _bind_parameters(self, fndef: ast.FunctionDef) -> None:
        args = fndef.args
        if args.vararg is not None or args.kwarg is not None:
            raise UnsupportedConstruct("variadic parameters (*args/**kwargs) are not supported", self._loc(fndef))
        self._local_names[self._fn] = self._collect_local_names(fndef)
        params = [*args.posonlyargs, *args.args]
        if self._instance is not None:
            if not params:
                raise UnsupportedConstruct(
                    "a synthesized method must take 'self' as its first parameter", self._loc(fndef)
                )
            self._self_name = params[0].arg
            params = params[1:]
            self._assigned_attrs = self._syntactically_assigned_attrs(fndef)
            self._collect_written_attrs(fndef)
            self._check_state_slot_names()
        # Positional then keyword-only parameters each become a scalar input port, in declaration order.
        for arg in [*params, *args.kwonlyargs]:
            self._env[arg.arg] = _Scalar(self._input(arg))

    def _input(self, arg: ast.arg) -> ValueId:
        annotation = self._fn.__annotations__.get(arg.arg, _NO_PARAMETER_ANNOTATION)
        if annotation is _NO_PARAMETER_ANNOTATION or annotation is float:
            return self._builder.float_input(arg.arg)
        if annotation is bool:
            return self._builder.bool_input(arg.arg)
        raise UnsupportedConstruct(
            f"unsupported parameter annotation for {arg.arg!r}: expected float or bool", self._loc(arg)
        )

    def _lower_body(self, fndef: ast.FunctionDef) -> None:
        returned = self._lower_stmts(fndef.body)
        # A stateful method need not return: its public state attributes are observable through their own ports.
        if not returned and self._instance is None:
            raise UnsupportedConstruct("function must end in a 'return'", self._loc(fndef))

    def _lower_stmts(self, stmts: list[ast.stmt]) -> bool:
        """Lower a straight-line statement list, returning True when a ``return`` was reached."""
        for stmt in stmts:
            if self._lower_stmt(stmt):
                return True
        return False

    def _lower_stmt(self, stmt: ast.stmt) -> bool:
        """Lower one statement. Returns True when a ``return`` was reached (no further statements are processed)."""
        match stmt:
            case ast.Expr(value=ast.Constant(value=str())):
                return False  # docstring
            case ast.Pass():
                return False
            case ast.Assign(targets=targets, value=value):
                # Lower the right-hand side once and bind it to every target; this covers single, chained, and
                # tuple/list-unpacking assignment uniformly (the swap ``x, y = y, x`` reads both sources first).
                rhs = self._lower_expr(value)
                for target in targets:
                    self._assign_target(target, rhs)
                return False
            case ast.AnnAssign(target=ast.Name(id=name), value=ast.expr() as value):
                self._bind_name(name, self._lower_expr(value), self._loc(stmt))
                return False
            case ast.AnnAssign(target=ast.Attribute() as target, value=ast.expr() as value):
                self._assign_attr(target, self._lower_expr(value))
                return False
            case ast.AugAssign(target=ast.Name(id=name), op=op, value=value):
                self._reject_self_rebinding(name, self._loc(stmt))
                if name not in self._env:
                    raise UnsupportedConstruct(f"augmented assignment to unknown name {name!r}", self._loc(stmt))
                self._bind_name(name, self._apply_binop(op, self._env[name], self._lower_expr(value), self._loc(stmt)))
                return False
            case ast.AugAssign(target=ast.Attribute() as target, op=op, value=value):
                current = self._read_attr(target)
                self._assign_attr(target, self._apply_binop(op, current, self._lower_expr(value), self._loc(stmt)))
                return False
            case ast.If(test=test, body=body, orelse=orelse):
                return self._lower_if(test, body, orelse)
            case ast.For(target=ast.Name(id=name), iter=iterable, body=body, orelse=[]):
                return self._lower_for(name, iterable, body, self._loc(stmt))
            case ast.For():
                raise UnsupportedConstruct(
                    "only 'for <name> in range(...)' over a static count (with no else clause) is supported",
                    self._loc(stmt),
                )
            case ast.While(test=test, body=body, orelse=[]):
                return self._lower_while(test, body, self._loc(stmt))
            case ast.While():
                raise UnsupportedConstruct("a 'while' loop with an else clause is not supported", self._loc(stmt))
            case ast.Return(value=None):
                self._reject_nested_return(stmt)
                return True
            case ast.Return(value=ast.expr() as value):
                self._reject_nested_return(stmt)
                self._return = self._lower_expr(value)
                return True
            case _:
                raise UnsupportedConstruct(f"unsupported statement {type(stmt).__name__}", self._loc(stmt))

    def _reject_nested_return(self, stmt: ast.stmt) -> None:
        # A return inside a branch arm would need the returns funneled to a single exit with an output phi; deferred.
        # An inlined callee's own return is consumed locally by _inline, so it is exempt.
        if self._in_branch > 0 and not self._inlining:
            raise UnsupportedConstruct("a 'return' inside a branch or loop is not yet supported", self._loc(stmt))

    def _lower_for(self, name: str, iterable: ast.expr, body: list[ast.stmt], loc: SourceLocation) -> bool:
        """
        Lower a ``for <name> in range(...)`` by fully unrolling it: the loop counter is a compile-time integer, so each
        trip lowers the body once with the counter bound (both as a static int for index/exponent positions and as a
        float-constant local for value positions). A trip count above the unroll threshold is rejected (a counted
        back-edge loop is not implemented; use a ``while`` for a variable count, lowered by ``_lower_while``). A
        ``return`` inside the body is rejected (the single-exit invariant), as in a branch arm.
        The counter leaks its final value after the loop, matching Python's ``for`` scoping (an empty range leaves any
        pre-loop binding untouched); restoring it instead would silently miscompile a nested loop that reuses the name.
        """
        self._reject_self_rebinding(name, loc)  # ``for self in ...`` rebinds the instance parameter -- rejected
        trips = self._static_range(iterable, loc)
        count = _range_trip_count(trips)  # big-int count: reject without materializing, even for range(10**40)
        if count > _UNROLL_THRESHOLD:
            raise UnsupportedConstruct(
                f"loop trip count {count} exceeds the unroll threshold {_UNROLL_THRESHOLD}; a counted back-edge "
                "for-loop is not supported (use a 'while' loop for a variable trip count)",
                loc,
            )
        for index in trips:
            self._static_ints[name] = index
            self._env[name] = _Scalar(self._builder.float_const(float(index)))
            if self._lower_stmts(body):
                raise UnsupportedConstruct("a 'return' inside a loop is not yet supported", loc)
        return False

    def _lower_while(self, test: ast.expr, body: list[ast.stmt], loc: SourceLocation) -> bool:
        """
        Lower a ``while`` as a real back-edge loop: preheader -> header(phis + exit branch) -> body -> back-edge to the
        header. Each scalar variable or persistent attribute the body reassigns becomes a loop-header phi merging its
        preheader value with the value at the body's end (a forward reference, closed once the body is lowered). A
        loop-invariant value dominates the loop and needs no phi. The condition is evaluated at the header each
        iteration. A ``return`` in the body and a loop-carried aggregate are rejected (the single-exit / scalar-merge
        invariants), as in a branch arm. A statically-false condition skips the loop entirely (its body never runs and
        contributes no state), mirroring a folded ``if``; a statically-true condition is lowered as a normal (infinite)
        loop, faithful to the source.
        """
        if _contains_walrus(test):
            # A walrus in the condition rebinds every iteration and its post-test value is what the name holds at the
            # loop exit (the loop leaves from the header after the test), which the header-phi exit value does not
            # capture. Rejected rather than miscompiled; bind it in the body instead.
            raise UnsupportedConstruct("a walrus ':=' in a while condition is not supported", loc)
        if self._static_condition(test) is False:
            # The loop never runs, but its condition is still type-checked here (a non-boolean operand is rejected, as
            # in an ``if``) -- a statically-false loop is the one path where the condition is otherwise never lowered in
            # a header. The lowered condition is dead and DCE-removed; environment and state are unchanged.
            skipped = self._scalar(self._lower_bool(test), test)
            if not isinstance(self._builder.type_of(skipped), BoolType):
                raise UnsupportedConstruct(
                    "a while condition must be a boolean value (a comparison or a boolean state/variable)",
                    self._loc(test),
                )
            return False
        # A counter the body reassigns is a runtime loop-header phi inside the loop, so it must be dropped from the
        # static-int map before the condition, body, and carried-set are folded/lowered (a leaked ``for`` counter the
        # loop rebinds is no longer a compile-time int; a folded comparison / static index / shift exponent must see it
        # as runtime), and the demotion persists past the loop. ``_loop_carried`` computes the reachable reassignments
        # and which of them are counters to demote -- a fixpoint, since demoting a counter turns a branch on it dynamic
        # and may expose further reassignments. A counter assigned only on a statically-dead path is NOT demoted.
        reassigned_names, carried_attrs, demoted = self._loop_carried(body)
        exit_static = {name: value for name, value in self._static_ints.items() if name not in demoted}
        self._static_ints = dict(exit_static)
        for attr in carried_attrs:
            self._ensure_state_loaded(attr)  # a state attr first written in the loop enters the header phi as live-in
        preheader_env, preheader_state = dict(self._env), dict(self._state_env)
        preheader = self._builder.current_block
        carried_names = reassigned_names & set(preheader_env)  # a pre-defined name is carried; a body-local is scoped

        header, body_block, exit_block = self._builder.block(), self._builder.block(), self._builder.block()
        self._builder.jump(header)
        self._builder.position_at(header)
        name_phis = {name: self._open_loop_phi(preheader, preheader_env[name], loc) for name in sorted(carried_names)}
        attr_phis = {attr: self._open_loop_phi(preheader, preheader_state[attr], loc) for attr in sorted(carried_attrs)}
        for name, (phi, _) in name_phis.items():
            self._env[name] = _Scalar(phi)
        for attr, (phi, _) in attr_phis.items():
            self._state_env[attr] = _Scalar(phi)

        cond = self._scalar(self._lower_bool(test), test)
        if not isinstance(self._builder.type_of(cond), BoolType):
            raise UnsupportedConstruct("a while condition must be a boolean value (a comparison or a boolean)", loc)
        self._builder.branch(cond, body_block, exit_block)

        self._builder.position_at(body_block)
        self._in_branch += 1  # a non-inlined return in the body is rejected by _reject_nested_return while this holds;
        returned = self._lower_stmts(body)  # an inlined callee's return is exempt there, so reject it explicitly here
        self._in_branch -= 1
        if returned:
            raise UnsupportedConstruct("a 'return' inside a loop is not yet supported", loc)
        latch = self._builder.current_block
        self._builder.jump(header)
        for name, (phi, init_id) in name_phis.items():
            self._close_loop_phi(phi, init_id, preheader, latch, self._env[name], loc)
        for attr, (phi, init_id) in attr_phis.items():
            self._close_loop_phi(phi, init_id, preheader, latch, self._state_env[attr], loc)

        self._builder.position_at(exit_block)
        self._env = dict(preheader_env)  # drop body-locals; the loop-carried names take their header-phi exit value
        self._env.update((name, _Scalar(phi)) for name, (phi, _) in name_phis.items())
        self._state_env = dict(preheader_state)
        self._state_env.update((attr, _Scalar(phi)) for attr, (phi, _) in attr_phis.items())
        self._static_ints = exit_static
        return False

    def _open_loop_phi(self, preheader: BlockId, init: _Value, loc: SourceLocation) -> tuple[ValueId, ValueId]:
        init_id = self._scalar(init, loc)  # a loop-carried value must be scalar (an aggregate merge is not supported)
        return self._builder.open_phi(self._builder.type_of(init_id), (preheader, init_id)), init_id

    def _close_loop_phi(
        self,
        phi: ValueId,
        init_id: ValueId,
        preheader: BlockId,
        latch: BlockId,
        latch_value: _Value,
        loc: SourceLocation,
    ) -> None:
        latch_id = self._scalar(latch_value, loc)
        if self._builder.type_of(latch_id) != self._builder.type_of(init_id):
            raise UnsupportedConstruct("a loop-carried variable must keep its type across the loop", loc)
        self._builder.set_phi_arms(phi, [(preheader, init_id), (latch, latch_id)])

    def _loop_carried(self, body: list[ast.stmt]) -> tuple[set[str], set[str], set[str]]:
        """
        The names and attributes a ``while`` body reassigns on a reachable path (its loop-carried set), plus the subset
        of static-int counters among the names that must be demoted inside (and after) the loop because the body
        rebinds them to a runtime value. A fixpoint: a reassigned counter is a runtime loop-header phi, so a branch on
        it is dynamic, which can expose further reassignments -- iterate until the demoted set is stable (it grows
        monotonically, bounded by the live counters). Unlike a fold-unaware scan this does not demote (and so does not
        reject the later static use of) a counter that is assigned only on a statically-dead, folded-away path.
        """
        demoted: set[str] = set()
        while True:
            saved = self._static_ints
            self._static_ints = {name: value for name, value in saved.items() if name not in demoted}
            names, attrs = self._loop_assigned(body)
            self._static_ints = saved
            next_demoted = demoted | (names & saved.keys())
            if next_demoted == demoted:
                return names, attrs, demoted
            demoted = next_demoted

    def _loop_assigned(self, stmts: list[ast.stmt]) -> tuple[set[str], set[str]]:
        """
        The local names and instance attributes a loop body reassigns on a reachable path (its loop-carried
        candidates). Reachability mirrors lowering precisely, including constant folding: a statically-known ``if``
        contributes only its taken arm, exactly as ``_scan_attr_writes`` and ``_lower_if`` resolve it. A non-fold-aware
        walk would over-approximate a write whose only occurrence is in a folded-away arm, opening a header phi for an
        attribute that is not persistent state (and never reassigned in the body) -- a self-referential, unwritten phi
        that crashes slot lookup. The static-int context is snapshot/restored across the two arms of a dynamic ``if``
        just as lowering does, so a counter bound in one arm does not leak into the sibling's fold decisions.
        """
        names: set[str] = set()
        attrs: set[str] = set()

        def record(targets: list[ast.expr]) -> None:
            for leaf in (leaf for target in targets for leaf in _leaf_targets(target)):
                if isinstance(leaf, ast.Name):
                    names.add(leaf.id)
                    self._invalidate_static_int(leaf.id)  # mirror lowering: a reassigned name is no longer static
                elif self._is_self_attr(leaf) and isinstance(leaf, ast.Attribute):
                    attrs.add(leaf.attr)

        def walk(body: list[ast.stmt]) -> None:
            for stmt in body:
                for name in _statement_walrus_names(stmt):  # a walrus in this statement's test/value rebinds a name
                    names.add(name)
                    self._invalidate_static_int(name)  # mirror lowering: a reassigned name is no longer static
                match stmt:
                    case ast.Return():
                        return
                    case ast.If(test=test, body=b, orelse=o):
                        constant = self._static_condition(test)
                        if constant is not None:
                            walk(b if constant else o)  # a folded ``if`` contributes only its taken arm
                        else:
                            saved = dict(self._static_ints)
                            walk(b)
                            then_static = dict(self._static_ints)
                            self._static_ints = dict(saved)
                            walk(o)
                            self._static_ints = self._merge_static_ints(then_static, self._static_ints)
                    case ast.For(target=ast.Name(id=counter), iter=iterable, body=b):
                        if self._for_counter_is_bound(
                            iterable
                        ):  # a for that runs >=1 trip binds (and leaks) its counter
                            names.add(counter)
                        self._walk_loop_assigned(counter, iterable, b, walk)
                    case ast.While(test=test, body=b):
                        if self._static_condition(test) is not False:  # a statically-false loop reassigns nothing
                            _, _, nested = self._loop_carried(b)  # counters the nested loop rebinds to runtime
                            saved = dict(self._static_ints)
                            self._static_ints = {n: v for n, v in saved.items() if n not in nested}
                            walk(b)
                            self._static_ints = {n: v for n, v in saved.items() if n not in nested}
                    case ast.For(body=b, orelse=o):
                        walk(b)
                        walk(o)
                    case ast.Assign(targets=targets):
                        record(targets)
                    case ast.AnnAssign(target=target) | ast.AugAssign(target=target):
                        record([target])

        outer_static = dict(self._static_ints)  # the fold-aware walk binds counters; do not perturb lowering's context
        try:
            walk(stmts)
        finally:
            self._static_ints = outer_static
        return names, attrs

    def _for_counter_is_bound(self, iterable: ast.expr) -> bool:
        """Whether a ``for <name> in <iterable>`` binds (and leaks) its counter: true when the static range runs at
        least once. ``for i in range(0)`` runs zero times and never binds ``i`` (matching Python), so it must not be
        recorded as a loop-carried reassignment of an outer leaked counter. A non-static / over-threshold range is
        rejected at lowering anyway, so be conservative (treat it as binding)."""
        try:
            trips = self._static_range(iterable, self._loc(iterable))
        except UnsupportedConstruct:
            return True
        return _range_trip_count(trips) >= 1

    def _walk_loop_assigned(
        self, counter: str, iterable: ast.expr, body: list[ast.stmt], walk: "Callable[[list[ast.stmt]], None]"
    ) -> None:
        """Unroll a static ``for`` inside a loop body exactly as lowering does, binding the counter per trip so a
        counter-dependent inner range is resolved consistently; a non-static / over-threshold range walks once."""
        try:
            trips = self._static_range(iterable, self._loc(iterable))
        except UnsupportedConstruct:
            walk(body)
            return
        if _range_trip_count(trips) > _UNROLL_THRESHOLD:
            walk(body)
            return
        for index in trips:
            self._static_ints[counter] = index
            walk(body)

    def _is_builtin_range(self) -> bool:
        """
        Whether bare ``range`` resolves to the builtin: not a local, and not shadowed by a module global (which Python
        resolves before the builtin). A shadowed ``range`` is not the unrollable builtin and is rejected.
        """
        if self._is_local("range"):
            return False
        return self._fn.__globals__.get("range", builtins.range) is builtins.range

    def _static_range(self, iterable: ast.expr, loc: SourceLocation) -> range:
        """
        Evaluate a ``for`` iterable to a ``range`` of compile-time integer counter values. The result is a lazy
        ``range`` object, never a materialized list: ``len()`` is O(1), so the caller can reject an enormous count
        against the unroll threshold without ever building the sequence (``range(10**9)`` must not OOM the compiler).
        """
        match iterable:
            case ast.Call(func=ast.Name(id="range"), args=args, keywords=[]) if self._is_builtin_range():
                bounds = [self._static_int(arg) for arg in args]
                if not 1 <= len(bounds) <= 3 or any(bound is None for bound in bounds):
                    raise UnsupportedConstruct("range(...) needs 1 to 3 static integer arguments", loc)
                try:
                    return range(*[bound for bound in bounds if bound is not None])
                except ValueError as exc:  # a zero step
                    raise UnsupportedConstruct(f"invalid range: {exc}", loc) from exc
            case _:
                raise UnsupportedConstruct("a for-loop must iterate over range(...)", loc)

    def _static_int(self, node: ast.expr) -> int | None:
        """Evaluate an expression to a compile-time integer (counter, index, or exponent), or None if it is not one."""
        match node:
            case ast.Constant(value=int(literal)) if not isinstance(literal, bool):
                return literal
            case ast.Name(id=name) if name in self._static_ints:
                return self._static_ints[name]
            case ast.Name(id=name) if not self._is_local(name):
                # A module-level integer constant (e.g. ITERATIONS = 12) used as a loop bound, index, or exponent.
                global_value = self._fn.__globals__.get(name, _ABSENT)
                return global_value if type(global_value) is int else None
            case ast.Attribute() if self._is_self_attr(node):
                assert isinstance(node, ast.Attribute)
                attr_value = self._snapshot.get(node.attr)
                if type(attr_value) is not int:
                    return None
                assigned = self._readonly_scan_assigned_attrs
                if assigned is None:
                    assigned = self._assigned_attrs
                return attr_value if node.attr not in assigned else None
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                inner = self._static_int(operand)
                return None if inner is None else -inner
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return self._static_int(operand)
            case ast.BinOp(left=left, op=op, right=right):
                a, b = self._static_int(left), self._static_int(right)
                if a is None or b is None:
                    return None
                match op:
                    case ast.Add():
                        return a + b
                    case ast.Sub():
                        return a - b
                    case ast.Mult():
                        return a * b
                    case _:
                        return None
            case _:
                return None

    def _resolves_to_builtin(self, name: str) -> bool:
        """
        Whether a bare ``name`` resolves to the actual Python builtin -- absent from the module globals, or explicitly
        rebound to the builtin itself -- rather than being shadowed by a user global (a local, or any other object).
        A shadow is what Python would call, so the name is not the builtin cast/intrinsic it spells.
        """
        if self._is_local(name):
            return False
        callee = self._fn.__globals__.get(name, _ABSENT)
        return callee is _ABSENT or callee is getattr(builtins, name, None)

    def _cast_call(self, node: ast.expr) -> tuple[str, ast.expr] | None:
        """
        If ``node`` is an unshadowed builtin ``bool(x)`` / ``float(x)`` call on a single positional argument, return
        ``(builtin name, argument)``; else None. Lets the static evaluators see through a cast exactly as lowering does.
        """
        if (
            not isinstance(node, ast.Call)
            or node.keywords
            or len(node.args) != 1
            or isinstance(node.args[0], ast.Starred)
        ):
            return None
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in ("bool", "float") or not self._resolves_to_builtin(func.id):
            return None
        return func.id, node.args[0]

    def _static_bool(self, test: ast.expr) -> bool | None:
        """
        Evaluate a branch condition known at compile time -- a literal ``True``/``False``, a read-only boolean
        attribute (a boolean instance attribute never assigned anywhere in the body, so it keeps its snapshot value),
        or a comparison of two compile-time floats (a literal, an unrolled loop counter, a read-only attribute, or
        arithmetic of these; e.g. ``if i > 0:``) -- or None for a runtime condition. The comparison fold is fast-math
        (float64, accepted per DESIGN.md), exactly as the constant folder evaluates a relational of constants. A
        statically-known condition takes one arm and the other is not lowered (no spurious persistent state from a
        write that can never execute). The scan and the lowering share this so their reachability agrees.
        """
        if isinstance(test, ast.Constant) and isinstance(test.value, bool):
            return test.value
        cast = self._cast_call(test)
        if cast is not None and cast[0] == "bool":
            # ``bool(<static bool>)`` is identity; ``bool(<static float>)`` is format-dependent (its argument is not a
            # static bool, so this returns None and defers to the format-aware hardware cast).
            return self._static_bool(cast[1])
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self._static_bool(test.operand)
            return None if inner is None else (not inner)
        if isinstance(test, ast.BoolOp):
            # Fold strictly left to right, exactly as Python evaluates: ``or`` stops at the first True operand, ``and``
            # at the first False one. An unknown operand reached before any absorbing operand leaves the result runtime
            # (returning None), so that operand is still lowered and type-checked by ``_lower_connective`` rather than
            # being silently folded away -- which is what rejects a non-boolean operand such as ``x or True``.
            absorbing = isinstance(test.op, ast.Or)
            for operand in test.values:
                folded = self._static_bool(operand)
                if folded is None:
                    return None
                if folded == absorbing:
                    return absorbing
            return not absorbing  # every operand folded to the identity
        if isinstance(test, ast.IfExp):
            condition = self._static_bool(test.test)
            if condition is None:
                return None
            return self._static_bool(test.body if condition else test.orelse)
        if isinstance(test, ast.Compare):
            # ``a OP1 b OP2 c`` is the conjunction of its consecutive pairs, short-circuiting on the first failing link.
            operands = [test.left, *test.comparators]
            for op, left, right in zip(test.ops, operands, operands[1:]):
                relation = self._RELATIONAL_OPS.get(type(op))
                if relation is None:
                    return None
                holds = self._static_relation(relation, left, right)
                if holds is None:
                    return None
                if not holds:
                    return False
            return True
        if self._is_self_attr(test) and not self._scanning_readonly_attrs:
            assert isinstance(test, ast.Attribute)
            value = self._snapshot.get(test.attr)
            if isinstance(value, (bool, np.bool_)) and test.attr not in self._assigned_attrs:
                return bool(value)  # a read-only boolean attribute keeps its snapshot value (never assigned)
        return None

    def _static_condition(self, test: ast.expr) -> bool | None:
        """
        The compile-time value of a branch or loop condition for REACHABILITY -- which arm runs -- folding a connective
        by its absorbing element: ``X or True`` is True and ``X and False`` is False whatever the other operands are.
        This assumes the operands are boolean; operand-type validity is enforced separately when the condition is
        lowered (``_lower_connective`` rejects a non-boolean operand), so this drives only reachability. Unlike
        ``_static_bool`` -- which folds strictly left to right so a not-yet-type-checked operand is still lowered and
        rejected -- it is sound here precisely because the lowering of the condition does the type-checking. It is the
        one predicate the lowering fold (``_lower_if`` / ``_lower_ifexp`` / ``_lower_while``) and the attribute and loop
        scans share, so they descend exactly the same arms; a divergence would make a folded-away arm's write a
        persistent-state slot with no value (a crash). A leaf or unrelated comparison defers to ``_static_bool``. This
        is a sound, deliberately incomplete approximation of the (complete) HIR constant folder -- it folds the forms a
        kernel realistically writes (connectives, casts, equal-arm ternaries, a ``float(<cond>)`` comparison); a
        constant condition buried under some other shape stays runtime, at worst rejecting a ``return`` or leaving an
        unused state register under a contrived tautology (Phase 2's early returns lift the return limit).
        """
        match test:
            case ast.BoolOp(op=op, values=values):
                absorbing = isinstance(op, ast.Or)
                saw_unknown = False
                for value in values:
                    folded = self._static_condition(value)
                    if folded is None:
                        saw_unknown = True
                    elif folded == absorbing:
                        return absorbing  # an absorbing operand fixes the result regardless of the rest
                return None if saw_unknown else (not absorbing)  # all operands folded to the identity element
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                inner = self._static_condition(operand)
                return None if inner is None else (not inner)
            case ast.IfExp(test=condition, body=body, orelse=orelse):
                chosen = self._static_condition(condition)
                if chosen is not None:
                    return self._static_condition(body if chosen else orelse)
                then_value = self._static_condition(body)  # a runtime test still folds when both arms agree
                return then_value if then_value is not None and then_value == self._static_condition(orelse) else None
            case ast.Compare(left=left, ops=[op], comparators=[right]) if (
                self._reach_float_cast(left) is not None or self._reach_float_cast(right) is not None
            ):
                # A comparison with a ``float(<connective>)`` operand: the strict ``_static_bool`` path defers the cast
                # argument and cannot fold it, so reachability folds it here (else the const condition leaks a branch
                # and a dead-arm state slot). Other comparisons fall through to ``_static_bool`` (which keeps int-exact
                # folding); a chained float-cast comparison is rare enough to leave runtime.
                relation = self._RELATIONAL_OPS.get(type(op))
                if relation is None:
                    return None
                left_value = self._reach_float_cast(left)
                left_value = self._static_float(left) if left_value is None else left_value
                right_value = self._reach_float_cast(right)
                right_value = self._static_float(right) if right_value is None else right_value
                if left_value is None or right_value is None or left_value != left_value or right_value != right_value:
                    return None
                return relation.holds((left_value > right_value) - (left_value < right_value))
            case _:
                cast = self._cast_call(test)
                if cast is not None and cast[0] == "bool":
                    return self._static_condition(cast[1])  # ``bool(<cond>)`` carries the truthiness of its argument
                return self._static_bool(test)

    def _reach_float_cast(self, node: ast.expr) -> float | None:
        """
        The compile-time float of an unshadowed ``float(<connective>)`` cast for reachability (``1.0``/``0.0``), or None
        if ``node`` is not a float() of a statically-known condition. Distinct from ``_static_float`` (whose cast case
        defers its argument to the strict ``_static_bool``), so a comparison like ``float(X or True) > 0.5`` folds.
        """
        cast = self._cast_call(node)
        if cast is None or cast[0] != "float":
            return None
        condition = self._static_condition(cast[1])
        return None if condition is None else (1.0 if condition else 0.0)

    def _static_relation(self, relation: RelationalOp, left: ast.expr, right: ast.expr) -> bool | None:
        """
        Fold one relational link of compile-time operands, or None if either is not compile-time. Two integers are
        compared exactly (a float64 fold would round operands beyond 2**53 and misfold, e.g.
        ``9007199254740993 == 9007199254740992``); otherwise the fast-math float64 fold is used (accepted per
        DESIGN.md), leaving a NaN operand to the comparator.
        """
        left_int, right_int = self._static_int(left), self._static_int(right)
        if left_int is not None and right_int is not None:
            return relation.holds((left_int > right_int) - (left_int < right_int))
        left_float, right_float = self._static_float(left), self._static_float(right)
        if (
            left_float is not None
            and right_float is not None
            and left_float == left_float
            and right_float == right_float
        ):
            return relation.holds((left_float > right_float) - (left_float < right_float))
        return None

    def _syntactically_assigned_attrs(self, fndef: ast.FunctionDef) -> set[str]:
        """
        The instance attributes the body could assign on a reachable path; used by ``_static_bool`` to recognize a
        read-only attribute (one never assigned, so it keeps its snapshot value). Reachability mirrors ``_lower_stmts``:
        it stops at a ``return`` and folds a literal-constant ``if`` (and a statically-false ``while``) to its live
        arm, so a write in statically-dead code does not mask a read-only attribute. A condition that depends on a
        read-only attribute is not resolved here -- the set is still being built -- so an attribute leaf reads as opaque
        and only an absorbing connective (``self.flag or True``) or an attribute-free condition folds; an unfoldable
        attribute condition is descended conservatively (a safe over-approximation: an attribute is only ever treated as
        written when it might not be, never the reverse).
        """
        attrs: set[str] = set()
        self._scanning_readonly_attrs = True
        self._readonly_scan_assigned_attrs = attrs
        try:
            self._collect_assigned(fndef.body, attrs)
        finally:
            self._scanning_readonly_attrs = False
            self._readonly_scan_assigned_attrs = None
        return attrs

    def _collect_assigned(self, stmts: list[ast.stmt], attrs: set[str]) -> bool:
        """
        Record the instance attributes assigned on a reachable path into ``attrs``; return True if a ``return`` is
        reached so the caller stops, exactly as lowering does. A folded ``if`` whose taken arm returns makes the rest
        of the enclosing list unreachable -- without propagating that, an attribute assigned after such an ``if`` would
        be wrongly counted as written and lose its read-only fold.
        """

        def record(targets: list[ast.expr]) -> None:
            for leaf in (leaf for target in targets for leaf in _leaf_targets(target)):
                if self._is_self_attr(leaf) and isinstance(leaf, ast.Attribute):
                    attrs.add(leaf.attr)

        for stmt in stmts:
            match stmt:
                case ast.Return():
                    return True  # statements after a return are unreachable, exactly as lowering stops here
                case ast.If(test=test, body=body, orelse=orelse):
                    # Fold a statically-known guard to its live arm, as lowering does, so a write in the dead arm is not
                    # counted as an assignment (which would wrongly mark a read-only attribute as written and suppress a
                    # later fold). Attribute leaves read as opaque while this set is being built (``_scanning_readonly
                    # _attrs``), so an absorbing connective still folds but a value-dependent attribute condition is
                    # conservatively descended on both arms.
                    constant = self._static_condition(test)
                    if constant is not None:
                        if self._collect_assigned(body if constant else orelse, attrs):
                            return True  # the taken arm returned; the rest of this list is unreachable
                    else:
                        self._collect_assigned(body, attrs)
                        self._collect_assigned(orelse, attrs)
                case ast.While(test=test, body=body) if self._static_condition(test) is False:
                    pass  # a statically-false while never runs; its body assigns nothing reachable (lowering skips it)
                case ast.For(iter=iterable, body=body, orelse=orelse):
                    # A zero-trip static range never runs its body, so a write there is not reachable; mirror only that.
                    # The counter is deliberately NOT bound here: folding a counter-dependent inner condition would
                    # require this scan to replicate the full static-int discipline of ``_scan_attr_writes``
                    # (invalidate-on-reassign, per-arm snapshot/restore), and binding without it risks a stale-counter
                    # miscompile. So a write reachable only on a counter value no trip takes is conservatively counted --
                    # a safe over-approximation (at worst an unused state register for a dead for-body write, never a
                    # wrong result). Unifying the three scans' loop traversal is tracked future work.
                    if self._for_counter_is_bound(iterable):
                        self._collect_assigned(body, attrs)
                    self._collect_assigned(orelse, attrs)
                case ast.While(body=body, orelse=orelse):
                    self._collect_assigned(body, attrs)
                    self._collect_assigned(orelse, attrs)
                case ast.Assign(targets=targets):
                    record(targets)
                case ast.AnnAssign(target=target) | ast.AugAssign(target=target):
                    record([target])
        return False

    def _lower_if(self, test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt]) -> bool:
        """
        Lower an ``if``/``else``. A compile-time-known condition (literal or read-only boolean attribute) takes one arm
        in place. A dynamic boolean test emits a ``branch`` terminator into fresh then/else blocks and a merge block
        whose phis reconcile the two arms' environments and persistent state. Returns True only if both arms returned
        (never, for now: a return inside an arm is rejected).
        """
        # Fold a nested if-without-else into one combined-``and`` branch: ``if A: (if B: S)`` with no ``else`` on either
        # is exactly ``if (A and B): S``, so it lowers to a single branch (one ``jump``) instead of two. A boolean test
        # in this subset is a pure combinational value (no side effects, no faulting), so evaluating B unconditionally
        # is equivalent, and the left-to-right ``and`` lowering preserves any walrus binding in A before B reads it.
        # The reachability scans recurse into both arms irrespective of nesting, so they record the same conditional
        # writes either way -- the fold does not desync them from lowering. Applied repeatedly, ``if A: if B: if C: S``
        # collapses to a single ``A and B and C`` branch.
        if not orelse:
            tests = [test]
            # Do not absorb an inner test that carries a walrus: in the nested form its binding is conditional on the
            # outer test, but ``A and B`` evaluates B unconditionally, so folding would over-bind the walrus.
            while (
                len(body) == 1
                and isinstance(body[0], ast.If)
                and not body[0].orelse
                and not _contains_walrus(body[0].test)
            ):
                tests.append(body[0].test)
                body = body[0].body
            if len(tests) > 1:
                test = ast.copy_location(ast.BoolOp(op=ast.And(), values=tests), tests[0])
        # Lower the condition first: this type-checks its operands (rejecting a non-boolean one). Then fold reachability
        # via ``_static_condition`` -- the same predicate the attribute/loop scans use, so a folded ``if X or True:``
        # takes one arm in place (no branch, no spurious return-inside-a-branch rejection) without the scans and the
        # lowering disagreeing about which arms exist.
        cond = self._scalar(self._lower_bool(test), test)
        if not isinstance(self._builder.type_of(cond), BoolType):
            raise UnsupportedConstruct(
                "an if condition must be a boolean value (a comparison or a boolean state/variable)", self._loc(test)
            )
        constant = self._static_condition(test)
        if constant is not None:
            return self._lower_stmts(body if constant else orelse)
        loc = self._loc(test)
        before_env, before_state = dict(self._env), dict(self._state_env)
        before_static = dict(self._static_ints)
        then_block, else_block, merge_block = self._builder.block(), self._builder.block(), self._builder.block()
        self._builder.branch(cond, then_block, else_block)
        then = self._lower_arm(then_block, before_env, before_state, before_static, body, merge_block, loc)
        else_ = self._lower_arm(else_block, before_env, before_state, before_static, orelse, merge_block, loc)
        self._builder.position_at(merge_block)
        self._env = self._merge_env(then.env, else_.env, then.end_block, else_.end_block, loc)
        self._state_env = self._merge_state(then.state, else_.state, then.end_block, else_.end_block, loc)
        self._static_ints = self._merge_static_ints(then.static_ints, else_.static_ints)
        return False

    def _lower_arm(
        self,
        block: BlockId,
        base_env: dict[str, _Value],
        base_state: dict[str, _Value],
        base_static: dict[str, int],
        stmts: list[ast.stmt],
        merge_block: BlockId,
        loc: SourceLocation,
    ) -> "_ArmResult":
        """
        Lower one branch arm from a copy of the pre-branch environment, then jump to the merge; return its final
        environment, state, compile-time-integer bindings, and end block (the arm may itself open nested blocks).
        """
        self._builder.position_at(block)
        self._env, self._state_env = dict(base_env), dict(base_state)
        self._static_ints = dict(base_static)  # the arm starts from the pre-branch counters, isolated from the sibling
        self._in_branch += 1
        if self._lower_stmts(stmts):
            raise UnsupportedConstruct("a 'return' inside a branch is not yet supported", loc)
        self._in_branch -= 1
        end_block = self._builder.current_block
        self._builder.jump(merge_block)
        return _ArmResult(self._env, self._state_env, self._static_ints, end_block)

    def _merge_static_ints(self, then_static: dict[str, int], else_static: dict[str, int]) -> dict[str, int]:
        """
        Keep a compile-time-integer binding past a branch only when both arms leave the same value: a counter that an
        arm leaks must not be trusted on the other path. A counter the two arms leave differing (e.g. nested loops with
        differing trip counts) is dropped, so a later static index/exponent use of it is rejected rather than silently
        compiled to one arm's value. Its float-constant binding in ``_env`` still merges to a runtime phi as usual.
        """
        return {
            name: then_static[name]
            for name in then_static.keys() & else_static.keys()
            if then_static[name] == else_static[name]
        }

    def _merge_env(
        self,
        then_env: dict[str, _Value],
        else_env: dict[str, _Value],
        pred_then: BlockId,
        pred_else: BlockId,
        loc: SourceLocation,
    ) -> dict[str, _Value]:
        """
        Merge the two arms' locals: a name bound in both arms becomes a phi when the arms disagree. A name bound in
        only one arm is conditionally defined and drops out of scope (using it afterwards is an unknown-name error).
        """
        return {
            name: self._merge_values(then_env[name], else_env[name], pred_then, pred_else, loc)
            for name in then_env.keys() & else_env.keys()
        }

    def _merge_state(
        self,
        then_state: dict[str, _Value],
        else_state: dict[str, _Value],
        pred_then: BlockId,
        pred_else: BlockId,
        loc: SourceLocation,
    ) -> dict[str, _Value]:
        """
        Merge persistent state across the arms: an attribute an arm did not touch carries its live-in there, so a
        write on only one path becomes a phi against the carried-over value.
        """
        merged: dict[str, _Value] = {}
        for attr in self._state_order:
            a = then_state.get(attr)
            b = else_state.get(attr)
            a = self._live_in(attr) if a is None else a
            b = self._live_in(attr) if b is None else b
            merged[attr] = self._merge_values(a, b, pred_then, pred_else, loc)
        return merged

    def _merge_values(self, a: _Value, b: _Value, pred_a: BlockId, pred_b: BlockId, loc: SourceLocation) -> _Value:
        """Reconcile two arm values into a phi per diverging scalar leaf (identical values need no phi)."""
        match (a, b):
            case (_Scalar(id=ia), _Scalar(id=ib)):
                if ia == ib:
                    return a
                if self._builder.type_of(ia) != self._builder.type_of(ib):
                    raise UnsupportedConstruct(
                        "the two branches produce values of different scalar types (a conditional's arms, and a "
                        "variable's value across an if, must have the same type)",
                        loc,
                    )
                return _Scalar(self._builder.phi(self._builder.type_of(ia), [(pred_a, ia), (pred_b, ib)]))
            case (_Aggregate(items=items_a), _Aggregate(items=items_b)) if len(items_a) == len(items_b):
                return _Aggregate(
                    tuple(self._merge_values(x, y, pred_a, pred_b, loc) for x, y in zip(items_a, items_b))
                )
            case _:
                raise UnsupportedConstruct("the two branches produce incompatible shapes for a merged value", loc)

    def _live_in(self, attr: str) -> _Value:
        """The slot's live-in value (state register content at initiation start), materialized from interned reads."""
        shape = self._shape(attr)
        return shape.compose(tuple(_Scalar(self._read_slot(shape, slot)) for slot in shape.slots))

    def _read_slot(self, shape: "_StateAttr", slot: str) -> ValueId:
        return self._builder.bool_state_read(slot) if shape.is_bool else self._builder.float_state_read(slot)

    def _bind_name(self, name: str, value: _Value, loc: SourceLocation | None = None) -> None:
        """
        Bind a local name to a (runtime) value. Crucially, this drops any compile-time-integer binding the name held:
        a ``for`` counter (or any name) reassigned to a runtime value is no longer a compile-time constant, so a later
        static-context use of it -- a folded branch condition, an array index, a shift exponent, a range bound -- must
        see it as runtime (rejected or lowered as such), never resolve to the stale counter value.
        """
        self._reject_self_rebinding(name, loc)
        self._env[name] = value
        self._invalidate_static_int(name)

    def _reject_self_rebinding(self, name: str, loc: SourceLocation | None) -> None:
        """
        Reject rebinding the instance parameter (``self``). It is the fixed instance the attributes resolve against, not
        a value: ``self.x`` always reads the original instance regardless of any later ``self = ...``, so allowing the
        rebinding would silently miscompile (Python instead makes ``self`` a plain local and ``self.x`` then faults).
        """
        if self._self_name is not None and name == self._self_name:
            raise UnsupportedConstruct(f"cannot assign to the instance parameter {name!r}", loc)

    def _invalidate_static_int(self, name: str) -> None:
        """
        Drop a name's compile-time-integer binding because it has been reassigned to a (runtime) value. The
        reachability scans (``_scan_attr_writes``, ``_loop_assigned``) must apply this exactly as lowering's
        ``_bind_name`` does: a value assignment never produces a static integer (those arise only from ``for`` counters
        and ``range`` bounds), so an assignment to a previously-static name demotes it everywhere. Keeping the scans in
        lockstep with lowering prevents a fold-reachability divergence -- e.g. the scan folding a branch on a stale
        counter while lowering treats it as runtime, which would desynchronize the persistent-state set from the phis.
        """
        self._static_ints.pop(name, None)

    def _assign_target(self, target: ast.expr, value: _Value) -> None:
        """Bind one assignment target to ``value``, recursing into tuple/list targets to unpack an aggregate."""
        match target:
            case ast.Name(id=name):
                self._bind_name(name, value, self._loc(target))
            case ast.Attribute():
                self._assign_attr(target, value)
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                for sub, item in self._unpack_targets(elts, value, target):
                    self._assign_target(sub, item)
            case _:
                raise UnsupportedConstruct(f"unsupported assignment target {type(target).__name__}", self._loc(target))

    def _unpack_targets(self, elts: list[ast.expr], value: _Value, node: ast.expr) -> list[tuple[ast.expr, _Value]]:
        """
        Pair each tuple-unpacking target with its value, mirroring Python: a single ``*rest`` target absorbs the
        surplus items as an aggregate, every other target takes one item, and the source must be an aggregate whose
        length matches the fixed (non-starred) targets.
        """
        if not isinstance(value, _Aggregate):
            raise UnsupportedConstruct("cannot unpack a scalar value in a tuple assignment", self._loc(node))
        items = value.items
        stars = [index for index, elt in enumerate(elts) if isinstance(elt, ast.Starred)]
        if not stars:
            if len(elts) != len(items):
                raise UnsupportedConstruct(
                    f"cannot unpack {len(items)} values into {len(elts)} targets", self._loc(node)
                )
            return list(zip(elts, items))
        if len(stars) > 1:
            raise UnsupportedConstruct("only one starred target is allowed in a tuple assignment", self._loc(node))
        star = stars[0]
        starred = elts[star]
        assert isinstance(starred, ast.Starred)
        after = elts[star + 1 :]
        if len(elts) - 1 > len(items):
            raise UnsupportedConstruct(
                f"cannot unpack {len(items)} values into at least {len(elts) - 1} targets", self._loc(node)
            )
        head = list(zip(elts[:star], items[:star]))
        rest = _Aggregate(tuple(items[star : len(items) - len(after)]))  # the starred target binds the surplus
        tail = list(zip(after, items[len(items) - len(after) :]))
        return [*head, (starred.value, rest), *tail]

    def _lower_expr(self, node: ast.expr) -> _Value:
        match node:
            case ast.Constant(value=value):
                if isinstance(value, bool):  # checked before int: bool is an int subclass
                    return _Scalar(self._builder.bool_const(value))
                if isinstance(value, (int, float)):
                    return _Scalar(self._builder.float_const(float(value)))
                raise UnsupportedConstruct(f"unsupported constant {value!r}", self._loc(node))
            case ast.Name(id=name):
                bound = self._env.get(name)
                if bound is None:
                    raise UnsupportedConstruct(
                        f"unknown name {name!r} (only parameters and locally-assigned names are in scope)",
                        self._loc(node),
                    )
                return bound
            case ast.List(elts=elts) | ast.Tuple(elts=elts):
                return _Aggregate(tuple(self._lower_elements(elts)))
            case ast.Subscript(value=value, slice=index):
                return self._lower_subscript(self._lower_expr(value), index, self._loc(node))
            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=(int() | float()) as value)) if not isinstance(
                value, bool
            ):
                return _Scalar(self._builder.float_const(-float(value)))
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return _Scalar(self._builder.operation(FloatNeg(), [self._scalar(self._lower_expr(operand), node)]))
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                # Unary plus is scalar identity; like negation, it rejects an aggregate operand.
                return _Scalar(self._scalar(self._lower_expr(operand), node))
            case ast.UnaryOp(op=ast.Not()):
                return self._lower_bool(node)
            case ast.BinOp(left=left, op=ast.Pow(), right=right):
                return _Scalar(self._lower_pow(left, right))
            case ast.BinOp(left=left, op=op, right=right):
                return self._apply_binop(op, self._lower_expr(left), self._lower_expr(right), self._loc(node))
            case ast.Compare() | ast.BoolOp():
                return self._lower_bool(node)
            case ast.IfExp(test=test, body=body, orelse=orelse):
                return self._lower_ifexp(test, body, orelse, self._loc(node))
            case ast.Call():
                return self._lower_call(node)
            case ast.Attribute():
                return self._read_attr(node)
            case ast.NamedExpr(target=ast.Name(id=name), value=value):
                # Walrus ``(name := value)``: evaluate the value, bind it to the name (visible to later code, as in
                # Python), and yield it. A bool- or float-valued value routes through the normal lowering (a Compare /
                # BoolOp value is delegated to _lower_bool from here), so a walrus is transparent to its value's type.
                bound = self._lower_expr(value)
                self._bind_name(name, bound, self._loc(node))
                return bound
            case _:
                raise UnsupportedConstruct(f"unsupported expression {type(node).__name__}", self._loc(node))

    def _lower_elements(self, elts: list[ast.expr]) -> Iterator[_Value]:
        """Lower the elements of a list/tuple literal, splicing any ``*aggregate`` element into place."""
        for elt in elts:
            if isinstance(elt, ast.Starred):
                yield from self._unpack(self._lower_expr(elt.value), elt)
            else:
                yield self._lower_expr(elt)

    def _unpack(self, value: _Value, node: ast.expr) -> tuple[_Value, ...]:
        if not isinstance(value, _Aggregate):
            raise UnsupportedConstruct("can only unpack an aggregate with '*'", self._loc(node))
        return value.items

    def _lower_subscript(self, value: _Value, index: ast.expr, loc: SourceLocation) -> _Value:
        if not isinstance(value, _Aggregate):
            raise UnsupportedConstruct("cannot index or slice a scalar value", loc)
        match index:
            case ast.Slice(lower=lower, upper=upper, step=None):
                start = 0 if lower is None else self._const_index(lower)
                stop = len(value.items) if upper is None else self._const_index(upper)
                return _Aggregate(value.items[start:stop])
            case ast.Slice():
                raise UnsupportedConstruct("a slice step is not supported", loc)
            case _:
                i = self._const_index(index)
                if not -len(value.items) <= i < len(value.items):
                    raise UnsupportedConstruct(f"index {i} is out of range for a {len(value.items)}-element value", loc)
                return value.items[i]

    def _const_index(self, node: ast.expr) -> int:
        index = self._static_int(node)
        if index is None:
            raise UnsupportedConstruct("array index/bound must be a compile-time integer", self._loc(node))
        return index

    def _lower_call(self, node: ast.Call) -> _Value:
        func = node.func
        # The only supported method call is ``.flatten()``, and only on an aggregate -- a scalar has no such method.
        if isinstance(func, ast.Attribute) and func.attr == "flatten" and not node.args and not node.keywords:
            receiver = self._lower_expr(func.value)
            if not isinstance(receiver, _Aggregate):
                raise UnsupportedConstruct(".flatten() is only supported on an aggregate value", self._loc(node))
            return receiver.flatten()
        numpy_fn = self._numpy_function(func)
        if numpy_fn in _NUMPY_IDENTITY:
            if node.keywords or len(node.args) != 1 or isinstance(node.args[0], ast.Starred):
                raise UnsupportedConstruct(f"np.{numpy_fn}() takes a single array-like argument", self._loc(node))
            return self._lower_expr(node.args[0])
        # Resolve a bare name as Python does: a locally bound name shadows any global (and is not callable); a
        # user-defined global function shadows the built-in ``abs``; ``abs`` is a bare-name builtin, so a method-style
        # call such as ``x.abs(...)`` is never mistaken for it and falls through to the unsupported-call error.
        if isinstance(func, ast.Name):
            if self._is_local(func.id):
                raise UnsupportedConstruct(f"{func.id!r} is a local name, not a callable function", self._loc(node))
            callee = self._fn.__globals__.get(func.id, _ABSENT)
            if isinstance(callee, types.FunctionType):
                return self._inline_call(callee, node)
            if callee is not _ABSENT and not callable(callee):
                # A non-callable global shadows the built-in (Python raises ``TypeError`` when calling it), so the name
                # is not the intrinsic it spells; reject rather than silently lowering, e.g., ``abs`` to a FloatAbs. The
                # _ABSENT sentinel distinguishes a missing global from one explicitly bound to None, which also shadows.
                raise UnsupportedConstruct(
                    f"{func.id!r} is shadowed by a non-callable global; it cannot be called", self._loc(node)
                )
            # A bare name is one of the recognized builtins (abs/list/tuple/bool/float) only when it is the actual
            # builtin. A callable GLOBAL of the same name (a class, partial, or callable instance) is a shadow Python
            # would call instead, so it must not be mistaken for the builtin cast/abs; it falls through to the
            # unsupported-call rejection below.
            builtin_unshadowed = self._resolves_to_builtin(func.id)
            if func.id == "abs" and not node.keywords and builtin_unshadowed:
                operands = self._lower_args(node)
                if len(operands) == 1:
                    return _Scalar(self._builder.operation(FloatAbs(), [self._scalar(operands[0], node)]))
            if func.id in ("list", "tuple") and not node.keywords and builtin_unshadowed:
                # list(seq)/tuple(seq) of an aggregate is identity here: it carries the element order the model holds,
                # and the front-end already treats list and tuple aggregates co-equally (the list/tuple-literal case).
                operands = self._lower_args(node)
                if len(operands) == 1 and isinstance(operands[0], _Aggregate):
                    return operands[0]
            if func.id == "bool" and not node.keywords and builtin_unshadowed:
                operands = self._lower_args(node)
                if len(operands) != 1:
                    raise UnsupportedConstruct("bool() takes a single scalar argument", self._loc(node))
                operand = self._scalar(operands[0], node)  # an aggregate argument is rejected here
                if isinstance(self._builder.type_of(operand), BoolType):
                    return _Scalar(operand)  # bool(<bool>) is identity
                return _Scalar(self._builder.operation(FloatToBool(), [operand]))
            if func.id == "float" and not node.keywords and builtin_unshadowed:
                operands = self._lower_args(node)
                if len(operands) != 1:
                    raise UnsupportedConstruct("float() takes a single scalar argument", self._loc(node))
                operand = self._scalar(operands[0], node)  # an aggregate argument is rejected here
                if isinstance(self._builder.type_of(operand), BoolType):
                    return _Scalar(self._builder.operation(BoolToFloat(), [operand]))
                return _Scalar(operand)  # float(<float>) is identity
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in _KNOWN_INTRINSICS:
            raise MissingIntrinsic(f"implement this operator: {name}", self._loc(node))
        raise UnsupportedConstruct(f"unsupported call to {name or '<expr>'!r}", self._loc(node))

    def _numpy_function(self, func: ast.expr) -> str | None:
        """
        The function name if ``func`` is a ``<numpy>.<name>`` access (under any alias), else None. A locally bound name
        shadows the global, mirroring Python scoping, so a local ``np`` is not mistaken for the numpy module.
        """
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if not self._is_local(func.value.id) and self._fn.__globals__.get(func.value.id) is np:
                return func.attr
        return None

    def _lower_args(self, node: ast.Call) -> list[_Value]:
        """Lower a call's positional arguments left to right, splicing any ``*aggregate`` argument into place."""
        args: list[_Value] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                args.extend(self._unpack(self._lower_expr(arg.value), arg))
            else:
                args.append(self._lower_expr(arg))
        return args

    def _inline_call(self, callee: types.FunctionType, node: ast.Call) -> _Value:
        """Inline a pure global function: bind its parameters to the arguments and lower its body in a fresh scope."""
        if node.keywords:
            raise UnsupportedConstruct(
                f"inlined call to {callee.__name__}() takes no keyword arguments", self._loc(node)
            )
        return self._inline(callee, self._lower_args(node), self._loc(node))

    def _inline(self, callee: types.FunctionType, args: list[_Value], loc: SourceLocation) -> _Value:
        if callee in self._inlining:
            raise UnsupportedConstruct(f"recursive inlining of {callee.__name__}() is not supported", loc)
        fndef, lines, start, filename = _parse_fndef(callee)
        decl = fndef.args
        if decl.vararg is not None or decl.kwarg is not None or decl.kwonlyargs:
            raise UnsupportedConstruct(
                f"cannot inline {callee.__name__}(): variadic or keyword-only parameters are not supported", loc
            )
        params = [*decl.posonlyargs, *decl.args]
        if len(params) != len(args):
            raise UnsupportedConstruct(
                f"{callee.__name__}() takes {len(params)} positional arguments but {len(args)} were given", loc
            )
        # The callee is pure: lower it in a fresh scope with no state context (an attribute access would fail name
        # resolution), sharing the one HirBuilder so its ops intern/CSE into the same DAG. The caller's scope is
        # captured and reinstalled as a unit, so no field can be silently saved-but-not-restored.
        outer = self._capture()
        self._inlining.add(callee)
        bindings = {param.arg: arg for param, arg in zip(params, args)}
        self._install(_Scope.fresh(callee, bindings, lines, start, filename))
        self._local_names[callee] = self._collect_local_names(fndef)
        try:
            self._lower_body(fndef)
            if self._return is None:
                raise UnsupportedConstruct(f"inlined {callee.__name__}() must end in a 'return'", loc)
            return self._return
        finally:
            self._inlining.discard(callee)
            self._install(outer)

    def _capture(self) -> _Scope:
        return _Scope(
            fn=self._fn,
            env=self._env,
            static_ints=self._static_ints,
            return_=self._return,
            instance=self._instance,
            self_name=self._self_name,
            snapshot=self._snapshot,
            state_order=self._state_order,
            state_env=self._state_env,
            lines=self._lines,
            start=self._start,
            filename=self._filename,
        )

    def _install(self, scope: _Scope) -> None:
        self._fn = scope.fn
        self._env = scope.env
        self._static_ints = scope.static_ints
        self._return = scope.return_
        self._instance = scope.instance
        self._self_name = scope.self_name
        self._snapshot = scope.snapshot
        self._state_order = scope.state_order
        self._state_env = scope.state_env
        self._lines = scope.lines
        self._start = scope.start
        self._filename = scope.filename

    def _lower_pow(self, base: ast.expr, exponent: ast.expr) -> ValueId:
        n = self._static_int(exponent)
        if n is None:
            raise UnsupportedConstruct("exponent must be a compile-time integer", self._loc(exponent))
        if n < 0:
            # A negative power is only meaningful for a constant base (e.g. the per-iteration shift 2**-i); it folds to
            # a constant, then strength reduction turns a multiply by it into an exact power-of-two scale.
            base_value = self._static_float(base)
            if base_value is None:
                raise UnsupportedConstruct("a negative power is only supported for a constant base", self._loc(base))
            return self._builder.float_const(base_value**n)
        base_id = self._scalar(self._lower_expr(base), base)
        if n == 0:
            return self._builder.float_const(1.0)
        result = base_id
        for _ in range(n - 1):
            result = self._builder.operation(FloatMul(), [result, base_id])
        return result

    def _static_float(self, node: ast.expr) -> float | None:
        """
        Evaluate an expression to a compile-time float, or None if it is not one: a literal, a static integer (an
        unrolled loop counter), a read-only float attribute (a float instance attribute never assigned in the body, so
        it keeps its snapshot value), or ``+``/``-``/``*`` arithmetic of these. The fold is fast-math (float64),
        matching the constant folder and accepted per DESIGN.md; it drives compile-time branch decisions and the
        negative-power base.
        """
        cast = self._cast_call(node)
        if cast is not None and cast[0] == "float":
            condition = self._static_bool(cast[1])  # float(<static bool>) -> 1.0 / 0.0; float(<static float>) identity
            return (1.0 if condition else 0.0) if condition is not None else self._static_float(cast[1])
        match node:
            case ast.Constant(value=(int() | float()) as value) if not isinstance(value, bool):
                return float(value)
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                inner = self._static_float(operand)
                return None if inner is None else -inner
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return self._static_float(operand)
            case ast.BinOp(left=left, op=op, right=right):
                a, b = self._static_float(left), self._static_float(right)
                if a is None or b is None:
                    return None
                match op:
                    case ast.Add():
                        return a + b
                    case ast.Sub():
                        return a - b
                    case ast.Mult():
                        return a * b
                    case _:
                        return None
            case _:
                if self._is_self_attr(node) and not self._scanning_readonly_attrs:
                    assert isinstance(node, ast.Attribute)
                    attr_value = self._snapshot.get(node.attr)
                    if isinstance(attr_value, (int, float)) and not isinstance(attr_value, bool):
                        return float(attr_value) if node.attr not in self._assigned_attrs else None
                integer = self._static_int(node)
                return None if integer is None else float(integer)

    def _apply_binop(self, op: ast.operator, a: _Value, b: _Value, loc: SourceLocation) -> _Value:
        match op:
            case ast.Mult():
                # Scalar*scalar, or the elementwise broadcast of a scalar over an aggregate's leaves (vector*scalar).
                if isinstance(a, _Aggregate) and isinstance(b, _Scalar):
                    return self._broadcast(a, b.id)
                if isinstance(a, _Scalar) and isinstance(b, _Aggregate):
                    return self._broadcast(b, a.id)
                if isinstance(a, _Aggregate) or isinstance(b, _Aggregate):
                    raise UnsupportedConstruct(
                        "elementwise aggregate-by-aggregate multiplication is not supported", loc
                    )
                return _Scalar(self._builder.operation(FloatMul(), [self._scalar(a, loc), self._scalar(b, loc)]))
            case ast.Add():
                return _Scalar(self._builder.operation(FloatAdd(), [self._scalar(a, loc), self._scalar(b, loc)]))
            case ast.Sub():
                negated = self._builder.operation(FloatNeg(), [self._scalar(b, loc)])
                return _Scalar(self._builder.operation(FloatAdd(), [self._scalar(a, loc), negated]))
            case ast.Div():
                return _Scalar(self._builder.operation(FloatDiv(), [self._scalar(a, loc), self._scalar(b, loc)]))
            case _:
                raise UnsupportedConstruct(f"unsupported binary operator {type(op).__name__}", loc)

    _RELATIONAL_OPS: dict[type[ast.cmpop], RelationalOp] = {
        ast.Lt: RelationalOp.LT,
        ast.LtE: RelationalOp.LE,
        ast.Gt: RelationalOp.GT,
        ast.GtE: RelationalOp.GE,
        ast.Eq: RelationalOp.EQ,
        ast.NotEq: RelationalOp.NE,
    }

    def _lower_compare(self, op: ast.cmpop, left: _Value, right: _Value, loc: SourceLocation) -> _Value:
        relop = self._RELATIONAL_OPS.get(type(op))
        if relop is None:
            raise UnsupportedConstruct(f"unsupported comparison operator {type(op).__name__}", loc)
        left_id, right_id = self._scalar(left, loc), self._scalar(right, loc)
        if isinstance(self._builder.type_of(left_id), BoolType) or isinstance(
            self._builder.type_of(right_id), BoolType
        ):
            raise UnsupportedConstruct("comparison operands must be floating-point, not boolean", loc)
        return _Scalar(self._builder.operation(FloatRelational(relop), [left_id, right_id]))

    def _lower_bool(self, node: ast.expr) -> _Value:
        """
        Lower a boolean-valued expression to a bool scalar: a comparison (single or chained), a connective
        (``and``/``or``/``not``), a boolean literal/variable/read-only attribute, or (cross-bank) a cast. A connective
        builds a combinational ``BoolAnd``/``BoolOr``/``BoolNot`` over its operands (both operands always evaluated --
        the operands here are pure booleans); a compile-time-known result folds to a constant with no operation, and a
        statically-known connective operand is dropped (its identity) or short-circuits the whole (its absorbing value),
        matching Python. A non-boolean operand in a boolean position is rejected.
        """
        constant = self._static_bool(node)
        if constant is not None:
            return _Scalar(self._builder.bool_const(constant))
        match node:
            case ast.BoolOp(op=ast.And(), values=values):
                return self._lower_connective(values, BoolAnd(), absorbing=False)
            case ast.BoolOp(op=ast.Or(), values=values):
                return self._lower_connective(values, BoolOr(), absorbing=True)
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                return _Scalar(self._builder.operation(BoolNot(), [self._bool_scalar(operand)]))
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                return self._lower_compare_chain(left, ops, comparators, self._loc(node))
            case _:
                # A value-position expression (a bool literal, a bool variable, a read-only attribute, ...): lower it
                # plainly; whether the result must be boolean is enforced by the caller with a context-specific message.
                return self._lower_expr(node)

    def _bool_scalar(self, node: ast.expr) -> ValueId:
        """Lower a boolean sub-expression to a single bool ValueId (rejecting an aggregate or a non-boolean value)."""
        scalar = self._scalar(self._lower_bool(node), node)
        if not isinstance(self._builder.type_of(scalar), BoolType):
            raise UnsupportedConstruct("expected a boolean value here (a comparison or a boolean)", self._loc(node))
        return scalar

    def _lower_connective(self, values: list[ast.expr], op: Operator, absorbing: bool) -> _Value:
        # ``and`` has absorbing False / identity True; ``or`` is the dual. A statically-absorbing operand short-circuits
        # the whole connective (later operands are not evaluated, as in Python); a statically-identity operand is
        # dropped; the remaining dynamic operands are reduced left-to-right by the combinational logic operator.
        dynamic: list[ValueId] = []
        for value in values:
            static = self._static_bool(value)
            if static is None:
                dynamic.append(self._bool_scalar(value))
            elif static == absorbing:
                return _Scalar(self._builder.bool_const(absorbing))
            # else: the identity value -- drop it and continue
        if not dynamic:
            return _Scalar(self._builder.bool_const(not absorbing))  # every operand folded to the identity
        result = dynamic[0]
        for operand in dynamic[1:]:
            result = self._builder.operation(op, [result, operand])
        return _Scalar(result)

    def _lower_compare_chain(
        self, left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr], loc: SourceLocation
    ) -> _Value:
        # ``a OP1 b OP2 c`` is ``(a OP1 b) and (b OP2 c)`` with each operand evaluated exactly once (the shared middle
        # operand feeds two comparisons). The conjunction is the combinational ``BoolAnd``; a single comparison needs no
        # ``and`` at all.
        operands = [self._lower_expr(left), *(self._lower_expr(comparator) for comparator in comparators)]
        comparisons = [
            self._scalar(self._lower_compare(op, operands[i], operands[i + 1], loc), loc) for i, op in enumerate(ops)
        ]
        result = comparisons[0]
        for comparison in comparisons[1:]:
            result = self._builder.operation(BoolAnd(), [result, comparison])
        return _Scalar(result)

    def _lower_ifexp(self, test: ast.expr, body: ast.expr, orelse: ast.expr, loc: SourceLocation) -> _Value:
        """
        Lower a conditional expression ``body if test else orelse``. The test is lowered first (type-checking its
        operands and folding a connective/cast to a constant where it can, including ``x or True`` -> True); a test
        that reduces to a constant -- or two arms that share one compile-time value -- selects the value with no branch,
        otherwise a ``branch`` into fresh arm blocks lowers each arm there (only the taken arm computes at run time) and
        merges the two values in a phi.
        """
        if _contains_walrus(body) or _contains_walrus(orelse):
            # An arm is evaluated only when selected, but ``_branch_value`` lowers each from a shared environment it does
            # not snapshot/merge, so a walrus binding in an arm would leak across arms. The test may carry a walrus (it
            # always evaluates); an arm walrus is rejected rather than silently mis-scoped.
            raise UnsupportedConstruct("a walrus ':=' in a conditional-expression arm is not supported", loc)
        cond = self._bool_scalar(test)
        constant = self._static_condition(test)
        if constant is not None:
            return self._lower_expr(body if constant else orelse)
        # Equal compile-time arms make the value independent of the test, so no branch is needed. This is a VALUE proof,
        # so it must use the strict ``_static_bool`` / ``_static_float`` (which fold only genuinely-static operands),
        # NOT the reachability-only ``_static_condition`` -- the latter assumes operands are boolean and would elide the
        # branch (and the arm lowering that type-checks them) for an arm like ``float(x or True) > 0.5``.
        both_bool = self._static_bool(body)
        if both_bool is not None and both_bool == self._static_bool(orelse):
            return _Scalar(self._builder.bool_const(both_bool))
        both_float = self._static_float(body)
        if both_float is not None and both_float == self._static_float(orelse):
            return _Scalar(self._builder.float_const(both_float))
        return self._branch_value(cond, lambda: self._lower_expr(body), lambda: self._lower_expr(orelse), loc)

    def _branch_value(
        self, cond: ValueId, then_value: Callable[[], _Value], else_value: Callable[[], _Value], loc: SourceLocation
    ) -> _Value:
        """
        Branch on ``cond`` into fresh then/else blocks, evaluate a value in each (a pure expression, so it mutates no
        environment), and merge the two results into a phi at the merge block, leaving the builder positioned there so
        the enclosing expression resumes after the merge. Unlike a statement arm this never carries a ``return``, so the
        branch-nesting guard is not raised.
        """
        then_block, else_block, merge_block = self._builder.block(), self._builder.block(), self._builder.block()
        self._builder.branch(cond, then_block, else_block)
        self._builder.position_at(then_block)
        then = then_value()
        then_end = self._builder.current_block
        self._builder.jump(merge_block)
        self._builder.position_at(else_block)
        else_ = else_value()
        else_end = self._builder.current_block
        self._builder.jump(merge_block)
        self._builder.position_at(merge_block)
        return self._merge_values(then, else_, then_end, else_end, loc)

    def _broadcast(self, value: _Value, scalar: ValueId) -> _Value:
        """Multiply every scalar leaf of ``value`` by ``scalar``, preserving shape (the one elementwise vector op)."""
        if isinstance(value, _Aggregate):
            return _Aggregate(tuple(self._broadcast(item, scalar) for item in value.items))
        assert isinstance(value, _Scalar)
        return _Scalar(self._builder.operation(FloatMul(), [value.id, scalar]))

    def _scalar(self, value: _Value, where: ast.AST | SourceLocation) -> ValueId:
        if isinstance(value, _Scalar):
            return value.id
        loc = where if isinstance(where, SourceLocation) else self._loc(where)
        raise UnsupportedConstruct(f"expected a scalar value here, got a {len(value.leaves())}-element aggregate", loc)

    def _reject_shortcircuit_walrus(self, fndef: ast.FunctionDef) -> None:
        """
        Reject a walrus inside an ``and``/``or`` operand or a chained comparison. Such an operand may be short-circuited
        -- statically dropped by the connective fold, or unevaluated in Python -- so whether its binding happens cannot
        be reconciled between the reachability scans (which see the syntactic walrus) and lowering (which may never
        evaluate it). A walrus is supported only where it is evaluated unconditionally: a single comparison or bare
        test, an assignment/return value, an ``and``/``or``-free ``if``/``while`` test.
        """
        for node in ast.walk(fndef):
            operands: list[ast.expr] = []
            if isinstance(node, ast.BoolOp):
                operands = node.values
            elif isinstance(node, ast.Compare) and len(node.ops) > 1:
                operands = [node.left, *node.comparators]
            for operand in operands:
                if _contains_walrus(operand):
                    raise UnsupportedConstruct(
                        "a walrus ':=' inside an 'and'/'or' or a chained comparison is not supported "
                        "(its operand may be short-circuited)",
                        self._loc(operand),
                    )

    def _collect_local_names(self, fndef: ast.FunctionDef) -> set[str]:
        """
        Every name the function binds: its parameters and the targets it assigns or iterates (including inside nested
        ``if``/``for``/``while`` blocks). Python treats such a name as local throughout the body, so it shadows a
        same-named global (function, builtin, or numpy alias) even at a use that precedes its assignment -- where
        Python itself raises ``UnboundLocalError`` rather than seeing the global. A nested ``def``/``lambda``/``class``
        is a SEPARATE scope: its bound names are not local here, so the traversal does not descend into one.
        """
        self._reject_shortcircuit_walrus(fndef)  # a walrus in a short-circuitable operand is rejected before any scan
        names = {arg.arg for arg in (*fndef.args.posonlyargs, *fndef.args.args, *fndef.args.kwonlyargs)}
        names |= _scope_local_walrus_targets(fndef)  # a walrus ``(name := ...)`` binds ``name`` as a function local

        def walk(body: list[ast.stmt]) -> None:
            for stmt in body:
                match stmt:
                    case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                        pass  # a nested scope; its names belong to it, not to the function being lowered
                    case ast.Assign(targets=targets):
                        leaves = (leaf for target in targets for leaf in _leaf_targets(target))
                        names.update(leaf.id for leaf in leaves if isinstance(leaf, ast.Name))
                    case ast.AnnAssign(target=ast.Name(id=name)) | ast.AugAssign(target=ast.Name(id=name)):
                        names.add(name)
                    case ast.For(target=target, body=b, orelse=o):
                        names.update(leaf.id for leaf in _leaf_targets(target) if isinstance(leaf, ast.Name))
                        walk(b)
                        walk(o)
                    case ast.If(body=b, orelse=o) | ast.While(body=b, orelse=o):
                        walk(b)
                        walk(o)

        walk(fndef.body)
        return names

    def _is_local(self, name: str) -> bool:
        """Whether ``name`` is bound (parameter or assignment target) in the function currently being lowered."""
        return name in self._local_names[self._fn]

    def _collect_written_attrs(self, fndef: ast.FunctionDef) -> None:
        """
        Find the instance attributes the method assigns on any reachable path; these become persistent state, the rest
        stay constant. Scanning must mirror lowering's static reachability exactly: it descends into both arms of a
        DYNAMIC ``if`` (an attribute written on only one path is still state, carrying its live-in on the other) but
        only the taken arm of a literal-constant ``if``, and the body of a ``for`` once per unrolled trip with the
        counter bound (so a counter-dependent inner range that is empty on every trip -- e.g. ``for i in range(1): for
        j in range(i): self.s = ...`` -- is never scanned). A write that lowering never reaches must not be classified
        as state: it would crash slot registration or silently add a spurious state port. Stops at the first top-level
        return, like ``_lower_stmts``.
        """
        self._scan_attr_writes(fndef.body)
        self._static_ints.clear()  # the scan binds loop counters to mirror the unroll; lowering re-binds from scratch

    def _scan_attr_writes(self, stmts: list[ast.stmt]) -> bool:
        """
        Scan one statement list for attribute writes, returning True if a ``return`` was reached (so the caller stops,
        exactly as ``_lower_stmts`` does). Control flow and counter scoping mirror lowering precisely: a literal ``if``
        scans only its taken arm and propagates that arm's return; a dynamic ``if`` scans both arms with the
        compile-time counters snapshot/restored per arm and merged afterward (a counter bound in one arm must not leak
        into the sibling -- the same hazard branch lowering guards); a ``for`` unrolls via ``_scan_loop_attr_writes``.
        """
        for stmt in stmts:
            for name in _statement_walrus_names(stmt):
                self._invalidate_static_int(name)  # mirror lowering: a walrus-reassigned name is no longer static
            if isinstance(stmt, ast.Return):
                return True
            if isinstance(stmt, ast.If):
                constant = self._static_condition(stmt.test)
                if constant is not None:
                    if self._scan_attr_writes(stmt.body if constant else stmt.orelse):
                        return True  # the taken arm returned; statements after the if are unreachable
                else:
                    before = dict(self._static_ints)
                    self._scan_attr_writes(stmt.body)
                    then_static = dict(self._static_ints)
                    self._static_ints = dict(before)
                    self._scan_attr_writes(stmt.orelse)
                    self._static_ints = self._merge_static_ints(then_static, self._static_ints)
                continue
            if isinstance(stmt, ast.For):
                self._scan_loop_attr_writes(stmt)
                continue
            if isinstance(stmt, ast.While):
                # The body runs an unknown number of times; any attribute it writes is persistent state. A statically
                # -false loop never runs (its writes are not state), mirroring lowering. A name the body reassigns is a
                # runtime loop phi inside it, so demote it from the static-int map before scanning (so a folded branch
                # there agrees with lowering). The loop does not end the enclosing scan (it may run zero times); a
                # return inside it is rejected at lowering.
                if self._static_condition(stmt.test) is not False:
                    _, _, demoted = self._loop_carried(stmt.body)
                    saved = self._static_ints
                    self._static_ints = {n: v for n, v in saved.items() if n not in demoted}
                    self._scan_attr_writes(stmt.body)
                    self._static_ints = {n: v for n, v in saved.items() if n not in demoted}
                continue
            targets: list[ast.expr] = []
            match stmt:
                case ast.Assign(targets=ts):
                    targets = [leaf for t in ts for leaf in _leaf_targets(t)]
                case ast.AnnAssign(target=t) | ast.AugAssign(target=t):
                    targets = list(_leaf_targets(t))
            for target in targets:
                if isinstance(target, ast.Name):
                    self._invalidate_static_int(target.id)  # mirror lowering: a reassigned name is no longer static
                if not self._is_self_attr(target):
                    continue
                assert isinstance(target, ast.Attribute)
                attr = target.attr
                if attr not in self._snapshot:
                    raise UnsupportedConstruct(
                        f"attribute self.{attr} is assigned but not initialized on the instance "
                        "(all persistent state must have an initial value)",
                        self._loc(target),
                    )
                if attr not in self._state_order:
                    self._state_order.append(attr)
        return False

    def _scan_loop_attr_writes(self, stmt: ast.For) -> None:
        """
        Scan a ``for`` loop's attribute writes exactly as ``_lower_for`` would unroll it: bind the compile-time counter
        and scan the body once per trip (enclosing counters in scope), so the discovered state set matches precisely
        what the unroller lowers -- a counter-dependent inner range that is empty on every trip contributes nothing. A
        non-``Name`` target or a non-static / over-threshold range is rejected at lowering; scan its body once so a
        real write is not missed before that rejection.
        """
        if not isinstance(stmt.target, ast.Name):
            self._scan_attr_writes(stmt.body)
            return
        try:
            trips = self._static_range(stmt.iter, self._loc(stmt))
        except UnsupportedConstruct:
            self._scan_attr_writes(stmt.body)
            return
        if _range_trip_count(trips) > _UNROLL_THRESHOLD:
            self._scan_attr_writes(stmt.body)
            return
        for index in trips:
            self._static_ints[stmt.target.id] = index
            self._scan_attr_writes(stmt.body)

    def _is_self_attr(self, node: ast.expr) -> bool:
        return (
            self._self_name is not None
            and isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self._self_name
        )

    def _attr_of(self, target: ast.Attribute) -> str:
        if not self._is_self_attr(target):
            raise UnsupportedConstruct(
                "only direct self.<attr> access is supported (no nested or foreign attributes)", self._loc(target)
            )
        if target.attr not in self._snapshot:
            raise UnsupportedConstruct(f"unknown instance attribute self.{target.attr}", self._loc(target))
        return target.attr

    def _ensure_state_loaded(self, attr: str) -> None:
        """Load a persistent attribute's live-in (its slot register content at the initiation start) into the state
        environment if it has not been read or written yet, so its first use -- including the entry arm of a loop
        header phi for an attribute first written in the loop -- sees the value carried over from the previous call."""
        if attr in self._state_order and attr not in self._state_env:
            shape = self._shape(attr)
            reads = tuple(_Scalar(self._read_slot(shape, slot)) for slot in shape.slots)
            self._state_env[attr] = shape.compose(reads)

    def _read_attr(self, target: ast.Attribute) -> _Value:
        attr = self._attr_of(target)
        shape = self._shape(attr)
        if attr in self._state_order:
            # Persistent state: first read before any write is the slot's live-in; later reads see the written value.
            self._ensure_state_loaded(attr)
            return self._state_env[attr]
        consts = tuple(_Scalar(self._builder.const_node(reset)) for reset in shape.resets)
        return shape.compose(consts)

    def _assign_attr(self, target: ast.Attribute, value: _Value) -> None:
        attr = self._attr_of(target)
        shape = self._shape(attr)
        if not shape.accepts(value):
            kind = f"a {len(shape.slots)}-element vector" if shape.is_vector else "a scalar"
            raise UnsupportedConstruct(
                f"self.{attr} is {kind}; the assigned value has an incompatible shape", self._loc(target)
            )
        self._state_env[attr] = value

    def _shape(self, attr: str) -> _StateAttr:
        """
        The scalar-slot decomposition of an instance attribute, derived once from the reset snapshot and memoized so it
        is the single source of the attribute's shape. A list/tuple or 1-D numpy array is a vector; a real number is a
        scalar. A jaxtyping array annotation, when present, must agree with the value; a shape-less annotation
        (``list``, ``numpy.typing.NDArray``) leaves the shape to the value.
        """
        if attr not in self._shapes:
            self._shapes[attr] = self._derive_shape(attr)
        return self._shapes[attr]

    def _derive_shape(self, attr: str) -> _StateAttr:
        value = self._snapshot[attr]
        self._check_annotation(attr, value)
        if isinstance(value, (bool, np.bool_)):  # checked before the numeric paths: bool is an int subclass
            return _StateAttr(is_vector=False, is_bool=True, slots=[attr], resets=[BoolConst(bool(value))])
        elements = self._aggregate_elements(attr, value)
        if elements is None:
            return _StateAttr(False, False, [attr], [FloatConst(self._coerce_real(attr, value))])
        slots = [f"{attr}_{index}" for index in range(len(elements))]
        return _StateAttr(True, False, slots, [FloatConst(self._coerce_real(attr, element)) for element in elements])

    def _aggregate_elements(self, attr: str, value: object) -> list[object] | None:
        """The ordered elements of a 1-D aggregate value (list, tuple, or numpy array), or None for a scalar."""
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise UnsupportedConstruct(
                    f"instance attribute self.{attr} must be a scalar or 1-D array, got a {value.ndim}-D array"
                )
            return list(value)
        if isinstance(value, (list, tuple)):
            return list(value)
        return None

    def _check_annotation(self, attr: str, value: object) -> None:
        """
        Enforce a jaxtyping array annotation against the reset value, so an explicitly declared shape cannot silently
        disagree with it. A jaxtyping type is a class that exposes ``dims`` and validates shape and dtype via
        ``isinstance``; a generic alias (``list[float]``, ``numpy.typing.NDArray``) is not a class and states no shape.
        """
        annotation = self._annotation_of(attr)
        if isinstance(annotation, type) and hasattr(annotation, "dims") and not isinstance(value, annotation):
            raise UnsupportedConstruct(f"self.{attr} value does not satisfy its declared array type {annotation}")

    def _annotation_of(self, attr: str) -> Any:
        if self._instance is None:
            return None
        for klass in type(self._instance).__mro__:
            annotations = getattr(klass, "__annotations__", {})
            if attr in annotations:
                return annotations[attr]
        return None

    def _coerce_real(self, attr: str, value: object) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise UnsupportedConstruct(
                f"instance attribute self.{attr} must be a real number or a sequence of reals, "
                f"got {type(value).__name__}"
            )
        return float(value)

    def _check_state_slot_names(self) -> None:
        """
        A vector attribute decomposes into slots ``attr_0, attr_1, ...``; guard against such a slot name coinciding with
        another attribute's slot (e.g. a vector ``v`` and a scalar ``v_0``), which would otherwise alias distinct state.
        """
        owner: dict[str, str] = {}
        for attr in self._state_order:
            for slot in self._shape(attr).slots:
                if slot in owner:
                    raise UnsupportedConstruct(
                        f"state slot {slot!r} is produced by both self.{owner[slot]} and self.{attr}; "
                        "rename one to avoid an aliasing collision"
                    )
                owner[slot] = attr

    @staticmethod
    def _is_public(attr: str) -> bool:
        """A public attribute (no leading underscore) drives state_<attr> ports; an underscored one stays internal."""
        return not attr.startswith("_")

    def _register_state_slots(self) -> None:
        """Register each written attribute as persistent state: one scalar slot per element, reset from the snapshot."""
        for attr in self._state_order:
            shape = self._shape(attr)
            for slot, reset, live_out in zip(shape.slots, shape.resets, self._state_env[attr].leaves()):
                self._builder.state_slot(slot, reset, live_out)

    def _emit_outputs(self) -> None:
        """
        Emit the returned outputs as out_<path> ports and the public state attributes as state_<slot> ports. A returned
        leaf is dropped when its value equals a public slot's live-out: that wire is already exposed through the slot's
        port, so deduping it loses nothing. The key is the value (ValueId), not the spelling, so an alias and a
        coincidentally-equal expression collapse alike.
        """
        public_live_outs: set[ValueId] = set()
        for attr in self._state_order:
            if self._is_public(attr):
                public_live_outs.update(self._state_env[attr].leaves())
        if self._return is not None:
            for path, vid in self._return.output_leaves():
                if vid not in public_live_outs:
                    self._builder.output(_port_name(path), vid)
        for attr in self._state_order:
            if self._is_public(attr):
                for slot, live_out in zip(self._shape(attr).slots, self._state_env[attr].leaves()):
                    self._builder.output(_state_port_name(slot), live_out)


def lower(target: object) -> Hir:
    """
    Lower a synthesis target into HIR.

    A plain function lowers to a stateless module. A bound method lowers to a stateful module: its ``__self__`` is the
    constructed instance whose attribute snapshot seeds the reset state, and ``__func__`` is the analyzed method.
    """
    if inspect.ismethod(target):
        func = target.__func__
        if not isinstance(func, types.FunctionType):
            raise UnsupportedConstruct("only a pure-Python method can be synthesized")
        if func.__name__ == "__init__":
            raise UnsupportedConstruct("the synthesized method must not be __init__")
        return _Lowerer(func, instance=target.__self__).run()
    if isinstance(target, types.FunctionType):
        return _Lowerer(target).run()
    if inspect.isclass(target):
        raise UnsupportedConstruct("pass a bound method (instance.method), not a class, to synthesize stateful logic")
    raise UnsupportedConstruct(f"unsupported synthesis target of type {type(target).__name__!r}")
