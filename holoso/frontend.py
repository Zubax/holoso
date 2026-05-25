"""Lower a Python function object into HIR."""

import ast
import inspect
import textwrap
import types

from ._shape import Path, port_name
from .errors import MissingIntrinsic, SourceLocation, SourceUnavailable, UnsupportedConstruct
from .format import FloatFormat
from .hir import ArithOp, Hir, HirBuilder, SignOp, ValueId

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


class _Lowerer:
    def __init__(self, fn: types.FunctionType, fmt: FloatFormat) -> None:
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
        self._builder = HirBuilder(fmt)
        self._env: dict[str, ValueId] = {}

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
        return self._builder.finish()

    def _bind_parameters(self, fndef: ast.FunctionDef) -> None:
        args = fndef.args
        if args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
            raise UnsupportedConstruct("variadic and keyword-only parameters are not supported in v0", self._loc(fndef))
        for arg in (*args.posonlyargs, *args.args):
            self._env[arg.arg] = self._builder.input(arg.arg)

    def _lower_body(self, fndef: ast.FunctionDef) -> None:
        for stmt in fndef.body:
            if self._lower_stmt(stmt):
                return
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
            case ast.AnnAssign(target=ast.Name(id=name), value=ast.expr() as value):
                self._env[name] = self._lower_expr(value)
                return False
            case ast.AugAssign(target=ast.Name(id=name), op=op, value=value):
                if name not in self._env:
                    raise UnsupportedConstruct(f"augmented assignment to unknown name {name!r}", self._loc(stmt))
                self._env[name] = self._apply_binop(op, self._env[name], self._lower_expr(value), self._loc(stmt))
                return False
            case ast.Return(value=ast.expr() as value):
                for path, expr in _flatten_return(value):
                    self._builder.output(port_name(path), self._lower_expr(expr))
                return True
            case _:
                raise UnsupportedConstruct(f"unsupported statement {type(stmt).__name__}", self._loc(stmt))

    def _lower_expr(self, node: ast.expr) -> ValueId:
        match node:
            case ast.Constant(value=value):
                if isinstance(value, bool):
                    raise UnsupportedConstruct("boolean values are not supported in v0", self._loc(node))
                if isinstance(value, (int, float)):
                    return self._builder.const(float(value))
                raise UnsupportedConstruct(f"unsupported constant {value!r}", self._loc(node))
            case ast.Name(id=name):
                vid = self._env.get(name)
                if vid is None:
                    raise UnsupportedConstruct(
                        f"unknown name {name!r} (globals/closures are not resolved in v0)", self._loc(node)
                    )
                return vid
            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=(int() | float()) as value)):
                return self._builder.const(-float(value))
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return self._builder.signfix(SignOp.NEG, self._lower_expr(operand))
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return self._lower_expr(operand)
            case ast.BinOp(left=left, op=ast.Pow(), right=right):
                return self._lower_pow(left, right)
            case ast.BinOp(left=left, op=op, right=right):
                return self._apply_binop(op, self._lower_expr(left), self._lower_expr(right), self._loc(node))
            case ast.Call():
                return self._lower_call(node)
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
            return self._builder.signfix(SignOp.ABS, self._lower_expr(node.args[0]))
        if name in _KNOWN_INTRINSICS:
            raise MissingIntrinsic(f"implement this operator: {name}", self._loc(node))
        raise UnsupportedConstruct(f"unsupported call to {name or '<expr>'!r}", self._loc(node))

    def _lower_pow(self, base: ast.expr, exponent: ast.expr) -> ValueId:
        match exponent:
            case ast.Constant(value=int() as n) if not isinstance(n, bool) and n >= 0:
                base_id = self._lower_expr(base)
                if n == 0:
                    return self._builder.const(1.0)
                result = base_id
                for _ in range(n - 1):
                    result = self._builder.arith(ArithOp.MUL, result, base_id)
                return result
            case _:
                raise UnsupportedConstruct("exponent must be a non-negative integer literal in v0", self._loc(exponent))

    def _apply_binop(self, op: ast.operator, a: ValueId, b: ValueId, loc: SourceLocation) -> ValueId:
        match op:
            case ast.Add():
                return self._builder.arith(ArithOp.ADD, a, b)
            case ast.Sub():
                return self._builder.arith(ArithOp.ADD, a, self._builder.signfix(SignOp.NEG, b))
            case ast.Mult():
                return self._builder.arith(ArithOp.MUL, a, b)
            case ast.Div():
                return self._builder.arith(ArithOp.DIV, a, b)
            case _:
                raise UnsupportedConstruct(f"unsupported binary operator {type(op).__name__}", loc)


def _flatten_return(node: ast.expr) -> list[tuple[Path, ast.expr]]:
    """Flatten a return expression tree into ``(path, scalar-expr)`` pairs, mirroring ``_shape.flatten_value``."""
    leaves: list[tuple[Path, ast.expr]] = []

    def walk(expr: ast.expr, path: Path) -> None:
        if isinstance(expr, (ast.List, ast.Tuple)):
            for index, item in enumerate(expr.elts):
                walk(item, (*path, index))
        else:
            leaves.append((path, expr))

    if isinstance(node, (ast.List, ast.Tuple)):
        walk(node, ())
    else:
        leaves.append(((0,), node))
    return leaves


def lower(target: object, float_format: FloatFormat) -> Hir:
    """Lower a function object into HIR. Classes/other targets raise an explicit error in v0."""
    if isinstance(target, types.FunctionType):
        return _Lowerer(target, float_format).run()
    if inspect.isclass(target):
        raise UnsupportedConstruct("class/stateful targets are not supported in v0 (state lands in a later milestone)")
    raise UnsupportedConstruct(f"unsupported synthesis target of type {type(target).__name__!r}")
