"""Lower a Python function object into HIR."""

import ast
import inspect
import textwrap
import types

from .._errors import MissingIntrinsic, SourceLocation, SourceUnavailable, UnsupportedConstruct
from .._hir import FloatAbs, FloatAdd, FloatDiv, FloatMul, FloatNeg, Hir, HirBuilder, ValueId

_Path = list[int | str]

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
    """Map a leaf path to its output-port name, e.g. ``[0, "x"]`` -> ``out_0_x``."""
    return "out" + "".join(f"_{key}" for key in path)


class _Lowerer:
    def __init__(self, fn: types.FunctionType, instance: object | None = None) -> None:
        self._fn = fn
        try:
            lines, start = inspect.getsourcelines(fn)
        except (OSError, TypeError) as exc:
            raise SourceUnavailable(
                f"cannot retrieve source for {getattr(fn, '__name__', '?')!r}; "
                "define it in an importable module (not a REPL/exec/lambda)"
            ) from exc
        self._lines = lines
        self._start = start
        self._filename = inspect.getsourcefile(fn) or "<unknown>"
        self._builder = HirBuilder()
        self._env: dict[str, ValueId] = {}
        # Stateful-class lowering context; all empty/None for a plain stateless function. The snapshot is the instance
        # as handed to the synthesizer (whatever __init__ and any later mutation produced); its values seed reset.
        self._instance = instance
        self._self_name: str | None = None
        self._snapshot: dict[str, object] = dict(vars(instance)) if instance is not None else {}
        self._written_attrs: set[str] = set()
        self._state_order: list[str] = []
        self._state_env: dict[str, ValueId] = {}
        # Returned leaves are buffered rather than emitted on sight: dropping a return that carries a public attribute's
        # value needs that attribute's live-out ValueId, which is only settled once the whole body has been lowered.
        self._returns: list[tuple[_Path, ValueId]] = []

    def _loc(self, node: ast.AST) -> SourceLocation:
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = self._lines[lineno - 1] if 0 <= lineno - 1 < len(self._lines) else None
        return SourceLocation(self._filename, self._start + lineno - 1, col, snippet)

    def run(self) -> Hir:
        source = textwrap.dedent("".join(self._lines))
        module = ast.parse(source)
        fndef: ast.FunctionDef | None = None
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == self._fn.__name__:
                fndef = node
                break
        if fndef is None:
            raise SourceUnavailable(f"could not locate a 'def {self._fn.__name__}' in the retrieved source")
        self._bind_parameters(fndef)
        self._lower_body(fndef)
        self._register_state_slots()
        self._emit_outputs()
        return self._builder.finish()

    def _bind_parameters(self, fndef: ast.FunctionDef) -> None:
        args = fndef.args
        if args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
            raise UnsupportedConstruct("variadic and keyword-only parameters are not supported in v0", self._loc(fndef))
        params = [*args.posonlyargs, *args.args]
        if self._instance is not None:
            if not params:
                raise UnsupportedConstruct(
                    "a synthesized method must take 'self' as its first parameter", self._loc(fndef)
                )
            self._self_name = params[0].arg
            params = params[1:]
            self._collect_written_attrs(fndef)
        for arg in params:
            self._env[arg.arg] = self._builder.float_input(arg.arg)

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
                self._lower_return(value)
                return True
            case _:
                raise UnsupportedConstruct(f"unsupported statement {type(stmt).__name__}", self._loc(stmt))

    def _lower_return(self, value: ast.expr) -> None:
        for path, expr in _flatten_return(value):
            self._returns.append((path, self._lower_expr(expr)))

    def _lower_expr(self, node: ast.expr) -> ValueId:
        match node:
            case ast.Constant(value=value):
                if isinstance(value, bool):
                    raise UnsupportedConstruct("boolean values are not supported in v0", self._loc(node))
                if isinstance(value, (int, float)):
                    return self._builder.float_const(float(value))
                raise UnsupportedConstruct(f"unsupported constant {value!r}", self._loc(node))
            case ast.Name(id=name):
                vid = self._env.get(name)
                if vid is None:
                    raise UnsupportedConstruct(
                        f"unknown name {name!r} (globals/closures are not resolved in v0)", self._loc(node)
                    )
                return vid
            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=(int() | float()) as value)):
                return self._builder.float_const(-float(value))
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return self._builder.operation(FloatNeg(), [self._lower_expr(operand)])
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return self._lower_expr(operand)
            case ast.BinOp(left=left, op=ast.Pow(), right=right):
                return self._lower_pow(left, right)
            case ast.BinOp(left=left, op=op, right=right):
                return self._apply_binop(op, self._lower_expr(left), self._lower_expr(right), self._loc(node))
            case ast.Call():
                return self._lower_call(node)
            case ast.Attribute():
                return self._read_attr(node)
            case _:
                raise UnsupportedConstruct(f"unsupported expression {type(node).__name__}", self._loc(node))

    def _lower_call(self, node: ast.Call) -> ValueId:
        func = node.func
        name: str | None = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "abs" and len(node.args) == 1 and not node.keywords:
            return self._builder.operation(FloatAbs(), [self._lower_expr(node.args[0])])
        if name in _KNOWN_INTRINSICS:
            raise MissingIntrinsic(f"implement this operator: {name}", self._loc(node))
        raise UnsupportedConstruct(f"unsupported call to {name or '<expr>'!r}", self._loc(node))

    def _lower_pow(self, base: ast.expr, exponent: ast.expr) -> ValueId:
        match exponent:
            case ast.Constant(value=int(n)) if not isinstance(n, bool) and n >= 0:
                base_id = self._lower_expr(base)
                if n == 0:
                    return self._builder.float_const(1.0)
                result = base_id
                for _ in range(n - 1):
                    result = self._builder.operation(FloatMul(), [result, base_id])
                return result
            case _:
                raise UnsupportedConstruct("exponent must be a non-negative integer literal in v0", self._loc(exponent))

    def _apply_binop(self, op: ast.operator, a: ValueId, b: ValueId, loc: SourceLocation) -> ValueId:
        match op:
            case ast.Add():
                return self._builder.operation(FloatAdd(), [a, b])
            case ast.Sub():
                return self._builder.operation(FloatAdd(), [a, self._builder.operation(FloatNeg(), [b])])
            case ast.Mult():
                return self._builder.operation(FloatMul(), [a, b])
            case ast.Div():
                return self._builder.operation(FloatDiv(), [a, b])
            case _:
                raise UnsupportedConstruct(f"unsupported binary operator {type(op).__name__}", loc)

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
                if attr not in self._written_attrs:
                    self._written_attrs.add(attr)
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

    def _read_attr(self, target: ast.Attribute) -> ValueId:
        attr = self._attr_of(target)
        if attr in self._written_attrs:
            # Persistent state: first read before any write is the slot's live-in; later reads see the written value.
            if attr not in self._state_env:
                self._state_env[attr] = self._builder.float_state_read(attr)
            return self._state_env[attr]
        return self._builder.float_const(self._coerce_scalar(attr, self._snapshot[attr]))

    def _assign_attr(self, target: ast.Attribute, value: ValueId) -> None:
        self._state_env[self._attr_of(target)] = value

    def _coerce_scalar(self, attr: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsupportedConstruct(
                f"instance attribute self.{attr} must be a real number, got {type(value).__name__}"
            )
        return float(value)

    @staticmethod
    def _is_public(attr: str) -> bool:
        """A public attribute (no leading underscore) drives an out_<attr> port; an underscored one stays internal."""
        return not attr.startswith("_")

    def _register_state_slots(self) -> None:
        """Register each written attribute as a persistent state slot, reset to its instance-snapshot value."""
        for attr in self._state_order:
            reset_value = self._coerce_scalar(attr, self._snapshot[attr])
            self._builder.state_slot(attr, reset_value, self._is_public(attr), self._state_env[attr])

    def _emit_outputs(self) -> None:
        """
        A returned leaf is dropped when its value equals a public slot's live-out: that wire is already exposed through
        the slot's out_<attr> port, so deduping it loses nothing. The key is the value (ValueId), not the spelling, so
        an alias and a coincidentally-equal expression collapse alike.
        """
        public_live_outs = {self._state_env[attr] for attr in self._state_order if self._is_public(attr)}
        for path, vid in self._returns:
            if vid not in public_live_outs:
                self._builder.output(_port_name(path), vid)
        for attr in self._state_order:
            if self._is_public(attr):
                self._builder.output(_port_name([attr]), self._state_env[attr])


def _flatten_return(node: ast.expr) -> list[tuple[_Path, ast.expr]]:
    """Flatten a return expression tree into ``(path, scalar-expr)`` pairs."""
    leaves: list[tuple[_Path, ast.expr]] = []

    def walk(expr: ast.expr, path: _Path) -> None:
        if isinstance(expr, (ast.List, ast.Tuple)):
            for index, item in enumerate(expr.elts):
                walk(item, [*path, index])
        else:
            leaves.append((path, expr))

    if isinstance(node, (ast.List, ast.Tuple)):
        walk(node, [])
    else:
        leaves.append(([0], node))
    return leaves


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
