"""Lower a Python function object into HIR."""

import ast
import inspect
import textwrap
import types
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._errors import MissingIntrinsic, SourceLocation, SourceUnavailable, UnsupportedConstruct
from .._hir import FloatAbs, FloatAdd, FloatDiv, FloatMul, FloatNeg, Hir, HirBuilder, ValueId

_Path = list[int | str]

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
    names, its reset values, and whether an assigned value must be a scalar or a same-length flat aggregate.
    """

    is_vector: bool
    slots: list[str]
    resets: list[float]

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
class _Scope:
    """The per-function lowering state, captured and restored as a unit when a callee is inlined into a fresh scope."""

    fn: types.FunctionType
    env: dict[str, _Value]
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
        # The names each lowered function binds (parameters and assignment targets); shadow resolution consults these.
        self._local_names: dict[types.FunctionType, set[str]] = {}

    def _loc(self, node: ast.AST) -> SourceLocation:
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = self._lines[lineno - 1] if 0 <= lineno - 1 < len(self._lines) else None
        return SourceLocation(self._filename, self._start + lineno - 1, col, snippet)

    def run(self) -> Hir:
        self._bind_parameters(self._entry_fndef)
        self._lower_body(self._entry_fndef)
        self._register_state_slots()
        self._emit_outputs()
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
            self._collect_written_attrs(fndef)
            self._check_state_slot_names()
        # Positional then keyword-only parameters each become a scalar input port, in declaration order.
        for arg in [*params, *args.kwonlyargs]:
            self._env[arg.arg] = _Scalar(self._builder.float_input(arg.arg))

    def _lower_body(self, fndef: ast.FunctionDef) -> None:
        for stmt in fndef.body:
            if self._lower_stmt(stmt):
                return
        # A stateful method need not return: its public state attributes are observable through their own ports.
        if self._instance is None:
            raise UnsupportedConstruct("function must end in a 'return'", self._loc(fndef))

    def _lower_stmt(self, stmt: ast.stmt) -> bool:
        """Lower one statement. Returns True when a ``return`` was reached (no further statements are processed)."""
        match stmt:
            case ast.Expr(value=ast.Constant(value=str())):
                return False  # docstring
            case ast.Pass():
                return False
            case ast.Assign(targets=[ast.Name(id=name)], value=value):
                self._env[name] = self._lower_expr(value)
                return False
            case ast.Assign(targets=[ast.Attribute() as target], value=value):
                self._assign_attr(target, self._lower_expr(value))
                return False
            case ast.AnnAssign(target=ast.Name(id=name), value=ast.expr() as value):
                self._env[name] = self._lower_expr(value)
                return False
            case ast.AnnAssign(target=ast.Attribute() as target, value=ast.expr() as value):
                self._assign_attr(target, self._lower_expr(value))
                return False
            case ast.AugAssign(target=ast.Name(id=name), op=op, value=value):
                if name not in self._env:
                    raise UnsupportedConstruct(f"augmented assignment to unknown name {name!r}", self._loc(stmt))
                self._env[name] = self._apply_binop(op, self._env[name], self._lower_expr(value), self._loc(stmt))
                return False
            case ast.AugAssign(target=ast.Attribute() as target, op=op, value=value):
                current = self._read_attr(target)
                self._assign_attr(target, self._apply_binop(op, current, self._lower_expr(value), self._loc(stmt)))
                return False
            case ast.Return(value=None):
                return True
            case ast.Return(value=ast.expr() as value):
                self._return = self._lower_expr(value)
                return True
            case _:
                raise UnsupportedConstruct(f"unsupported statement {type(stmt).__name__}", self._loc(stmt))

    def _lower_expr(self, node: ast.expr) -> _Value:
        match node:
            case ast.Constant(value=value):
                if isinstance(value, bool):
                    raise UnsupportedConstruct("boolean values are not supported", self._loc(node))
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
            case ast.BinOp(left=left, op=ast.Pow(), right=right):
                return _Scalar(self._lower_pow(left, right))
            case ast.BinOp(left=left, op=op, right=right):
                return self._apply_binop(op, self._lower_expr(left), self._lower_expr(right), self._loc(node))
            case ast.Call():
                return self._lower_call(node)
            case ast.Attribute():
                return self._read_attr(node)
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
        match node:
            case ast.Constant(value=int(i)) if not isinstance(i, bool):
                return i
            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=int(i))) if not isinstance(i, bool):
                return -i
            case _:
                raise UnsupportedConstruct("array index/bound must be a constant integer literal", self._loc(node))

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
            callee = self._fn.__globals__.get(func.id)
            if isinstance(callee, types.FunctionType):
                return self._inline_call(callee, node)
            if func.id == "abs" and not node.keywords:
                operands = self._lower_args(node)
                if len(operands) == 1:
                    return _Scalar(self._builder.operation(FloatAbs(), [self._scalar(operands[0], node)]))
            if func.id in ("list", "tuple") and not node.keywords:
                # list(seq)/tuple(seq) of an aggregate is identity here: it carries the element order the model holds,
                # and the front-end already treats list and tuple aggregates co-equally (the list/tuple-literal case).
                operands = self._lower_args(node)
                if len(operands) == 1 and isinstance(operands[0], _Aggregate):
                    return operands[0]
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
        match exponent:
            case ast.Constant(value=int(n)) if not isinstance(n, bool) and n >= 0:
                base_id = self._scalar(self._lower_expr(base), base)
                if n == 0:
                    return self._builder.float_const(1.0)
                result = base_id
                for _ in range(n - 1):
                    result = self._builder.operation(FloatMul(), [result, base_id])
                return result
            case _:
                raise UnsupportedConstruct("exponent must be a non-negative integer literal", self._loc(exponent))

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

    def _collect_local_names(self, fndef: ast.FunctionDef) -> set[str]:
        """
        Every name the function binds: its parameters and the targets it assigns. Python treats such a name as local
        throughout the body, so it shadows a same-named global (function, builtin, or numpy alias) even at a use that
        precedes its assignment -- where Python itself raises ``UnboundLocalError`` rather than seeing the global.
        """
        names = {arg.arg for arg in (*fndef.args.posonlyargs, *fndef.args.args, *fndef.args.kwonlyargs)}
        for stmt in fndef.body:
            match stmt:
                case ast.Assign(targets=targets):
                    names.update(target.id for target in targets if isinstance(target, ast.Name))
                case ast.AnnAssign(target=ast.Name(id=name)) | ast.AugAssign(target=ast.Name(id=name)):
                    names.add(name)
        return names

    def _is_local(self, name: str) -> bool:
        """Whether ``name`` is bound (parameter or assignment target) in the function currently being lowered."""
        return name in self._local_names[self._fn]

    def _collect_written_attrs(self, fndef: ast.FunctionDef) -> None:
        """
        Find the instance attributes the method assigns in its reachable basic block; these become persistent state,
        the rest stay constant. Scanning stops at the first return, mirroring ``_lower_body``: statements after it are
        unreachable and never lowered, so collecting their writes would later look up an attribute that has no value.
        """
        for stmt in fndef.body:
            if isinstance(stmt, ast.Return):
                break
            targets: list[ast.expr] = []
            match stmt:
                case ast.Assign(targets=ts):
                    targets = list(ts)
                case ast.AnnAssign(target=t) | ast.AugAssign(target=t):
                    targets = [t]
            for target in targets:
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

    def _read_attr(self, target: ast.Attribute) -> _Value:
        attr = self._attr_of(target)
        shape = self._shape(attr)
        if attr in self._state_order:
            # Persistent state: first read before any write is the slot's live-in; later reads see the written value.
            if attr not in self._state_env:
                reads = tuple(_Scalar(self._builder.float_state_read(slot)) for slot in shape.slots)
                self._state_env[attr] = shape.compose(reads)
            return self._state_env[attr]
        consts = tuple(_Scalar(self._builder.float_const(reset)) for reset in shape.resets)
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
        elements = self._aggregate_elements(attr, value)
        if elements is None:
            return _StateAttr(False, [attr], [self._coerce_real(attr, value)])
        slots = [f"{attr}_{index}" for index in range(len(elements))]
        return _StateAttr(True, slots, [self._coerce_real(attr, element) for element in elements])

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
