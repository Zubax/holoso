"""
The specializing interpreter: Eel -> residual Eel. Binding time is its whole product; the rest follows
CPython or the ratified rulings, of which the non-obvious ones are recorded here.

Static folds go through the selected HIR operator's own ``evaluate``, and a fold naming no number
residualizes with constant operands: one expression, one answer, refusal survivor-based in HIR. No algebra
over residual operands -- that is HIR strength reduction's. Laziness is CPython's: a dead arm resolves no
names and desugars no callees; captures are judged at use, never at capture.

A name bound on only one side of a residual branch drops (CPython would raise UnboundLocalError); divergent
unmergeable values stay bound to a marker so the read, in any spelling, rejects truthfully. ``raise`` is
judged within its own function: unconditional there is a compile-time diagnostic even under a caller's
residual arm. Library stubs bind positionally only -- deliberately stricter than a host spelling like
``pow(base=..)``. All pow spellings share the ``**`` lowering; a fully static non-integer power folds
THROUGH the pow_ stub so the answer cannot depend on binding time -- hence a static ``(-2.0) ** 6.0``
refuses exactly like its runtime counterpart misbehaves, while ``(-2.0) ** 6`` chains exact. Recorded edge,
do not "fix": a power chain over a static base saturates to infinity where CPython raises OverflowError,
identical to HIR folding the same chain, covered by the fastmath charter.
"""

import dataclasses
import inspect
import logging
import math
import types
import typing
from dataclasses import dataclass

from ..._errors import SourceLocation, SynthesisError
from .._desugar import desugar
from .._ir import *
from .._lib import Intrinsic, Library, resolve
from . import _ops
from ._reject import reattribute, reject
from ._values import ExpansionBudget, Opaque, ResidualScalar, Scalar, StaticScalar, TupleValue, Value, same

_logger = logging.getLogger(__name__)

type _EnvKey = str | int


@dataclass(frozen=True, slots=True)
class _Unjoinable:
    """
    A binding whose two branch values cannot merge (captured objects with different identities, divergent
    tuples). The binding itself stays in the environment so that a later read draws a truthful located
    rejection at the read site -- CPython would have carried either value happily, so the honest report is
    "the compiler cannot merge these", never "unbound".
    """

    description: str


type _Env = dict[_EnvKey, Value | _Unjoinable]


@dataclass(slots=True)
class _Frame:
    """One function under interpretation; ``result`` is where a non-root ``return`` deposits the call's value."""

    fn: types.FunctionType
    annotations: dict[str, object]
    env: _Env
    root: bool
    result: Value | None = None


_MISSING = object()

_SCALAR_ANNOTATIONS: list[tuple[type, ScalarType]] = [
    (bool, ScalarType.BOOL),
    (int, ScalarType.INT),
    (float, ScalarType.FLOAT),
]


def partial_evaluate(eel: EelFunction, fn: types.FunctionType) -> EelFunction:
    return _Interpreter(fn).run(eel)


def _annotation_stype(annotation: object) -> ScalarType | None:
    return next((stype for ty, stype in _SCALAR_ANNOTATIONS if annotation is ty), None)


def _key_order(key: _EnvKey) -> tuple[bool, str]:
    return isinstance(key, str), str(key)


def _describe_opaque(value: Opaque) -> str:
    # An Opaque never carries an admissible float, so a float payload here is necessarily a NaN.
    if type(value.value) is float:
        return f"the captured value of {value.name!r} is NaN, which the compiler cannot represent"
    return f"the captured value of {value.name!r} is not a bool, int, or float scalar"


def _rechained(node: object, frames: tuple[CallFrame, ...]) -> object:
    """
    An inlined callee's tree with every origin carrying the frame chain, so residual nodes and rejections
    alike re-attribute; shape-generic over the frozen node dataclasses so IR growth needs no maintenance here.
    """
    if isinstance(node, Origin):
        return Origin(node.location, frames)
    if isinstance(node, SourceLocation):
        return node
    if isinstance(node, tuple):
        return tuple(_rechained(item, frames) for item in node)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return dataclasses.replace(
            node, **{field.name: _rechained(getattr(node, field.name), frames) for field in dataclasses.fields(node)}
        )
    return node


class _Interpreter:
    def __init__(self, fn: types.FunctionType) -> None:
        self._fn = fn
        self._budget = ExpansionBudget()
        self._next_temp = 0
        self._outputs: tuple[OutputDecl, ...] = ()
        self._return_values: tuple[Atom, ...] = ()
        self._eels: dict[types.FunctionType, EelFunction] = {}
        self._meta: dict[types.FunctionType, dict[str, object]] = {}
        self._inlining: set[types.FunctionType] = {fn}
        anchor = resolve(pow)
        assert isinstance(anchor, Library), "the builtin pow key anchors the ** lowering"
        self._pow_stub = anchor.stub

    def run(self, eel: EelFunction) -> EelFunction:
        frame = _Frame(fn=self._fn, annotations=self._annotations_of(eel.origin, self._fn), env={}, root=True)
        params = tuple(self._param(param, frame) for param in eel.params)
        sink: list[Stmt] = []
        returned = self._block(eel.body, frame, sink, in_branch=False)
        if not returned:
            self._conclude_bare(eel.origin, frame)
        sink.append(ResidualReturn(eel.origin, self._return_values))
        body, live = _prune(sink)
        assert not live, "a residual temp is read before any assignment"
        residual = EelFunction(eel.origin, eel.name, params, tuple(body), slots=(), outputs=self._outputs)
        _logger.info(
            "%s: partial evaluation: %d residual statement(s), %d output(s), %d budget unit(s) spent",
            eel.name,
            len(body),
            len(self._outputs),
            self._budget.spent,
        )
        return residual

    def _desugared(self, fn: types.FunctionType) -> EelFunction:
        found = self._eels.get(fn)
        if found is None:
            found = desugar(fn)
            self._eels[fn] = found
        return found

    def _annotations_of(self, origin: Origin, fn: types.FunctionType) -> dict[str, object]:
        found = self._meta.get(fn)
        if found is None:
            try:
                # eval_str accepts quoted annotations; lazy PEP 649 annotations execute user expressions
                # here, so any exception is the user's -- the broad catch is sanctioned alongside the shadow's.
                found = inspect.get_annotations(fn, eval_str=True)
            except Exception as error:
                reject(origin, f"the type annotations cannot be evaluated: {error}")
            self._meta[fn] = found
        return found

    def _param(self, param: Param, frame: _Frame) -> Param:
        annotation = frame.annotations.get(param.name, _MISSING)
        if annotation is _MISSING:
            reject(param.origin, f"the parameter {param.name!r} requires a type annotation")
        stype = _annotation_stype(annotation)
        if stype is None:
            reject(param.origin, f"the annotation of parameter {param.name!r} is not supported yet")
        frame.env[param.name] = ResidualScalar(stype, LocalRef(param.origin, param.name))
        return Param(param.origin, param.name, param.kind, stype)

    def _fresh(self) -> int:
        index = self._next_temp
        self._next_temp += 1
        return index

    # ------------------------------------------------------------------ statements

    def _block(self, stmts: tuple[Stmt, ...], frame: _Frame, sink: list[Stmt], in_branch: bool) -> bool:
        for stmt in stmts:
            match stmt:
                case Assign(target=target, value=value):
                    frame.env[self._binding_key(target)] = self._expr(value, frame, sink)
                case AugAssign(origin=origin, target=target, op=op, value=value):
                    current = frame.env.get(target.name)
                    if current is None:
                        reject(origin, f"the local name {target.name!r} is not bound on every path reaching this read")
                    rhs = self._expr(value, frame, sink)
                    frame.env[target.name] = self._binary(origin, op, self._readable(current, origin), rhs, sink)
                case If():
                    if self._if(stmt, frame, sink, in_branch):
                        return True
                case Return(origin=origin):
                    if in_branch:
                        reject(origin, "a return inside a runtime branch is not supported yet")
                    if frame.root:
                        self._conclude(stmt, frame, sink)
                    elif stmt.value is not None:
                        result = self._expr(stmt.value, frame, sink)
                        declared = _annotation_stype(frame.annotations.get("return"))
                        if declared is not None:
                            # At the return itself, so a mismatch points into the callee.
                            result = self._conform(result, declared, origin, sink, "the returned value")
                        frame.result = result
                    return True
                case Raise():
                    self._raise(stmt, frame, sink, in_branch)
                case While(origin=origin) | For(origin=origin):
                    reject(origin, "loops are not supported yet")
                case Break(origin=origin) | Continue(origin=origin):
                    reject(origin, "loops are not supported yet")
                case Unpack(origin=origin):
                    reject(origin, "aggregate values are not supported yet")
                case Store(origin=origin) | AugStore(origin=origin):
                    reject(origin, "stores through attributes or elements are not supported yet")
                case _:
                    raise AssertionError(stmt)
        return False

    def _binding_key(self, binding: Binding) -> _EnvKey:
        match binding:
            case LocalBind(name=name):
                return name
            case TempBind(index=index):
                return index

    def _if(self, stmt: If, frame: _Frame, sink: list[Stmt], in_branch: bool) -> bool:
        cond = self._scalar(self._expr(stmt.cond, frame, sink), stmt.origin)
        if cond.stype is not ScalarType.BOOL:
            reject(stmt.origin, "the branch condition must be a bool; Python truthiness is not supported")
        if isinstance(cond, StaticScalar):
            decided = _ops.const_value(cond.const)
            assert isinstance(decided, bool)
            taken = stmt.then if decided else stmt.orelse
            return self._block(taken, frame, sink, in_branch)
        then_frame = _Frame(frame.fn, frame.annotations, dict(frame.env), frame.root)
        else_frame = _Frame(frame.fn, frame.annotations, dict(frame.env), frame.root)
        then_sink: list[Stmt] = []
        else_sink: list[Stmt] = []
        returned = self._block(stmt.then, then_frame, then_sink, in_branch=True)
        assert not returned
        returned = self._block(stmt.orelse, else_frame, else_sink, in_branch=True)
        assert not returned
        joined = self._join(stmt.origin, then_frame.env, then_sink, else_frame.env, else_sink)
        sink.append(If(stmt.origin, self._materialize(cond, stmt.origin), tuple(then_sink), tuple(else_sink)))
        frame.env.clear()
        frame.env.update(joined)
        return False

    def _join(
        self, origin: Origin, then_env: _Env, then_sink: list[Stmt], else_env: _Env, else_sink: list[Stmt]
    ) -> _Env:
        joined: _Env = {}
        for key in sorted(set(then_env) & set(else_env), key=_key_order):
            a, b = then_env[key], else_env[key]
            if not isinstance(a, _Unjoinable) and not isinstance(b, _Unjoinable) and same(a, b):
                joined[key] = a
                continue
            if not (isinstance(a, (StaticScalar, ResidualScalar)) and isinstance(b, (StaticScalar, ResidualScalar))):
                joined[key] = _Unjoinable(self._describe_key(key))
                continue
            if {a.stype, b.stype} == {ScalarType.INT, ScalarType.FLOAT}:
                a = self._as_float(a, origin, then_sink)
                b = self._as_float(b, origin, else_sink)
            elif a.stype is not b.stype:
                reject(
                    origin,
                    f"the branches bind {self._describe_key(key)} to incompatible types "
                    f"({a.stype.value} vs {b.stype.value})",
                )
            if same(a, b):
                joined[key] = a
                continue
            stype = a.stype
            index = self._fresh()
            then_sink.append(Assign(origin, TempBind(origin, index), self._materialize(a, origin), stype))
            else_sink.append(Assign(origin, TempBind(origin, index), self._materialize(b, origin), stype))
            joined[key] = ResidualScalar(stype, TempRef(origin, index))
        return joined

    def _describe_key(self, key: _EnvKey) -> str:
        return f"the local {key!r}" if isinstance(key, str) else "the conditional result"

    def _readable(self, binding: Value | _Unjoinable, origin: Origin) -> Value:
        if isinstance(binding, _Unjoinable):
            reject(
                origin,
                f"{binding.description} holds branch values the compiler cannot merge; "
                "only bool, int, and float values join branches",
            )
        return binding

    def _raise(self, stmt: Raise, frame: _Frame, sink: list[Stmt], in_branch: bool) -> None:
        if in_branch:
            reject(stmt.origin, "a raise on a data-dependent path (a runtime branch arm) cannot be lowered")
        pieces: list[str] = []
        for part in stmt.parts:
            if isinstance(part, str):
                pieces.append(part)
                continue
            value = self._expr(part, frame, sink)
            if isinstance(value, StaticScalar):
                pieces.append(format(_ops.const_value(value.const)))
            elif isinstance(value, Opaque) and type(value.value) is str:
                pieces.append(value.value)
            else:
                reject(stmt.origin, "the raise message interpolates a value that is not a compile-time constant")
        text = "".join(pieces)
        reject(stmt.origin, f"{stmt.exc_type}: {text}" if text else stmt.exc_type)

    # ------------------------------------------------------------------ the module boundary

    def _conclude(self, stmt: Return, frame: _Frame, sink: list[Stmt]) -> None:
        annotation = frame.annotations.get("return", _MISSING)
        if annotation is _MISSING:
            reject(stmt.origin, "the return type annotation is required")
        if stmt.value is None:
            if annotation is not None:
                reject(stmt.origin, "the kernel returns no value but its annotation declares one")
            return
        value = self._expr(stmt.value, frame, sink)
        match value:
            case TupleValue(items=items):
                self._conclude_tuple(stmt.origin, annotation, items, sink)
            case StaticScalar() | ResidualScalar():
                declared = _annotation_stype(annotation)
                if declared is None:
                    reject(stmt.origin, "the return annotation does not match the returned scalar value")
                leaf = self._conform(value, declared, stmt.origin, sink, "the returned value")
                self._outputs = (OutputDecl((0,), leaf.stype),)
                self._return_values = (self._materialize(leaf, stmt.origin),)
            case Opaque(name=name):
                reject(stmt.origin, f"the captured object {name!r} cannot be returned")

    def _conclude_tuple(self, origin: Origin, annotation: object, items: tuple[Scalar, ...], sink: list[Stmt]) -> None:
        if typing.get_origin(annotation) is not tuple:
            reject(origin, "the return annotation does not match the returned tuple")
        args = typing.get_args(annotation)
        if any(arg is Ellipsis for arg in args):
            reject(origin, "variadic tuple return annotations are not supported yet")
        declared = [_annotation_stype(arg) for arg in args]
        if any(stype is None for stype in declared):
            reject(origin, "the return annotation is not supported yet; only flat scalar tuples lower for now")
        if len(declared) != len(items):
            reject(
                origin, f"the annotation declares {len(declared)} returned value(s), the kernel returns {len(items)}"
            )
        leaves: list[Scalar] = []
        for item, stype in zip(items, declared, strict=True):
            assert stype is not None
            leaves.append(self._conform(item, stype, origin, sink, "the returned value"))
        self._outputs = tuple(OutputDecl((i,), leaf.stype) for i, leaf in enumerate(leaves))
        self._return_values = tuple(self._materialize(leaf, origin) for leaf in leaves)

    def _conclude_bare(self, origin: Origin, frame: _Frame) -> None:
        annotation = frame.annotations.get("return", _MISSING)
        if annotation is _MISSING:
            reject(origin, "the return type annotation is required")
        if annotation is not None:
            reject(origin, "the kernel can complete without returning a value but its annotation declares one")

    def _conform(self, value: Value, declared: ScalarType, origin: Origin, sink: list[Stmt], what: str) -> Scalar:
        if isinstance(value, Opaque):
            reject(origin, _describe_opaque(value))
        if not isinstance(value, (StaticScalar, ResidualScalar)):
            reject(origin, f"{what} is not a {declared.value} scalar")
        if value.stype is declared:
            return value
        if value.stype is ScalarType.INT and declared is ScalarType.FLOAT:
            return self._as_float(value, origin, sink)
        reject(origin, f"{what} has type {value.stype.value} where the annotation declares {declared.value}")

    # ------------------------------------------------------------------ expressions

    def _expr(self, expr: Expr, frame: _Frame, sink: list[Stmt]) -> Value:
        match expr:
            case Const(value=value):
                return StaticScalar(_ops.make_const(value))
            case TempRef(origin=origin, index=index):
                bound = frame.env.get(index)
                assert bound is not None, "desugar binds every temp before its first read"
                return self._readable(bound, origin)
            case LocalRef(origin=origin, name=name):
                found = frame.env.get(name)
                if found is None:
                    reject(origin, f"the local name {name!r} is not bound on every path reaching this read")
                return self._readable(found, origin)
            case EnvRead():
                return self._env_read(expr, frame)
            case Unary(origin=origin, op=op, operand=operand):
                return self._unary(origin, op, self._expr(operand, frame, sink), sink)
            case Binary(origin=origin, op=op, left=left, right=right):
                lv = self._expr(left, frame, sink)
                rv = self._expr(right, frame, sink)
                return self._binary(origin, op, lv, rv, sink)
            case Compare(origin=origin, op=op, left=left, right=right):
                lv = self._expr(left, frame, sink)
                rv = self._expr(right, frame, sink)
                return self._compare(origin, op, lv, rv, sink)
            case Call():
                return self._call(expr, frame, sink)
            case TupleExpr(origin=origin, items=items):
                scalars: list[Scalar] = []
                for item in items:
                    if isinstance(item, StarArg):
                        reject(origin, "aggregate values are not supported yet")
                    scalars.append(self._scalar(self._expr(item, frame, sink), origin))
                return TupleValue(tuple(scalars))
            case AttrRead(origin=origin, base=base, attr=attr):
                base_value = self._expr(base, frame, sink)
                if isinstance(base_value, Opaque) and isinstance(base_value.value, (types.ModuleType, type)):
                    # A module/class attribute is a metadata read (class access unwraps staticmethod and
                    # plain functions); an instance would run live descriptors, so it waits for snapshots.
                    try:
                        raw = getattr(base_value.value, attr)
                    except AttributeError:
                        reject(origin, f"{base_value.name!r} has no attribute {attr!r}")
                    return self._admit(origin, f"{base_value.name}.{attr}", raw)
                reject(origin, "attribute access is not supported yet")
            case IndexRead(origin=origin) | SliceRead(origin=origin) | MultiIndexRead(origin=origin):
                reject(origin, "aggregate values are not supported yet")
            case ListExpr(origin=origin):
                reject(origin, "aggregate values are not supported yet")
            case Comp(origin=origin):
                reject(origin, "comprehensions are not supported yet")
            case _:
                raise AssertionError(expr)

    def _env_read(self, node: EnvRead, frame: _Frame) -> Value:
        if node.free:
            code = frame.fn.__code__
            assert node.name in code.co_freevars
            closure = frame.fn.__closure__
            assert closure is not None
            cell = closure[code.co_freevars.index(node.name)]
            try:
                raw = cell.cell_contents
            except ValueError:
                reject(node.origin, f"the free name {node.name!r} is unbound in its enclosing scope")
        else:
            if node.name in frame.fn.__globals__:
                raw = frame.fn.__globals__[node.name]
            else:
                # The function's own builtins mapping, which is what CPython would consult -- absent from
                # typeshed's FunctionType, hence the getattr.
                builtins_ns = getattr(frame.fn, "__builtins__")
                assert isinstance(builtins_ns, dict)
                if node.name not in builtins_ns:
                    reject(node.origin, f"the name {node.name!r} is not defined")
                raw = builtins_ns[node.name]
        return self._admit(node.origin, node.name, raw)

    def _admit(self, origin: Origin, name: str, raw: object) -> Value:
        if type(raw) is bool or type(raw) is int:
            return StaticScalar(_ops.make_const(raw))
        if type(raw) is float and not math.isnan(raw):
            return StaticScalar(_ops.make_const(raw))
        # A NaN rides as an opaque capture: binding one is CPython-legal; only a scalar use rejects.
        return Opaque(name, raw)

    def _scalar(self, value: Value, origin: Origin) -> Scalar:
        match value:
            case StaticScalar() | ResidualScalar():
                return value
            case Opaque():
                reject(origin, _describe_opaque(value))
            case TupleValue():
                reject(origin, "a tuple is only supported as the returned value for now")

    def _materialize(self, scalar: Scalar, origin: Origin) -> Atom:
        match scalar:
            case StaticScalar(const=const):
                return Const(origin, _ops.const_value(const))
            case ResidualScalar(atom=atom):
                return atom

    def _apply(self, operator: _ops.Operator, operands: list[Scalar], origin: Origin, sink: list[Stmt]) -> Scalar:
        if all(isinstance(operand, StaticScalar) for operand in operands):
            consts = [operand.const for operand in operands if isinstance(operand, StaticScalar)]
            try:
                return StaticScalar(operator.evaluate(consts))
            except _ops.NoNumber:
                pass  # the graph re-derives the fault and the survivor sweep judges it; never convict here
        stype = _ops.result_stype(operator)
        atoms = tuple(self._materialize(operand, origin) for operand in operands)
        index = self._fresh()
        sink.append(Assign(origin, TempBind(origin, index), IntrinsicCall(origin, operator, atoms), stype))
        return ResidualScalar(stype, TempRef(origin, index))

    def _as_float(self, scalar: Scalar, origin: Origin, sink: list[Stmt]) -> Scalar:
        if scalar.stype is ScalarType.FLOAT:
            return scalar
        assert scalar.stype is ScalarType.INT
        return self._apply(_ops.CONVERT[(ScalarType.INT, ScalarType.FLOAT)], [scalar], origin, sink)

    def _unary(self, origin: Origin, op: UnaryOp, value: Value, sink: list[Stmt]) -> Scalar:
        operand = self._scalar(value, origin)
        match op:
            case UnaryOp.NEG | UnaryOp.POS:
                if operand.stype is ScalarType.BOOL:
                    reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
                if op is UnaryOp.POS:
                    return operand
                negate = _ops.INT_NEG if operand.stype is ScalarType.INT else _ops.FLOAT_NEG
                return self._apply(negate, [operand], origin, sink)
            case UnaryOp.NOT:
                if operand.stype is not ScalarType.BOOL:
                    reject(origin, "`not` requires a bool operand; Python truthiness is not supported")
                return self._apply(_ops.BOOL_NOT, [operand], origin, sink)
            case UnaryOp.INVERT:
                if operand.stype is not ScalarType.INT:
                    reject(origin, "`~` is integer-only")
                return self._apply(_ops.INT_NOT, [operand], origin, sink)

    def _binary(self, origin: Origin, op: BinaryOp, lv: Value, rv: Value, sink: list[Stmt]) -> Scalar:
        left = self._scalar(lv, origin)
        right = self._scalar(rv, origin)
        both_bool = left.stype is ScalarType.BOOL and right.stype is ScalarType.BOOL
        both_int = left.stype is ScalarType.INT and right.stype is ScalarType.INT
        match op:
            case BinaryOp.AND | BinaryOp.OR:
                if not both_bool:
                    reject(origin, "the boolean gates require bool operands; Python truthiness is not supported")
                return self._apply(_ops.BOOL_GATES[op], [left, right], origin, sink)
            case BinaryOp.BITAND | BinaryOp.BITOR | BinaryOp.BITXOR:
                if both_bool:
                    return self._apply(_ops.BOOL_GATES[op], [left, right], origin, sink)
                if both_int:
                    return self._apply(_ops.INT_BINARY[op], [left, right], origin, sink)
                reject(origin, f"the bitwise operator `{op.value}` requires two ints or two bools")
            case BinaryOp.POW:
                return self._pow(origin, left, right, sink)
            case BinaryOp.MATMUL:
                reject(origin, "aggregate values are not supported yet")
            case _:
                pass
        if ScalarType.BOOL in (left.stype, right.stype):
            reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
        match op:
            case BinaryOp.FLOORDIV | BinaryOp.MOD | BinaryOp.LSHIFT | BinaryOp.RSHIFT:
                if not both_int:
                    reject(origin, f"the operator `{op.value}` is integer-only")
                return self._apply(_ops.INT_BINARY[op], [left, right], origin, sink)
            case BinaryOp.DIV:
                dividend = self._as_float(left, origin, sink)
                divisor = self._as_float(right, origin, sink)
                return self._apply(_ops.FLOAT_DIV, [dividend, divisor], origin, sink)
            case BinaryOp.ADD | BinaryOp.SUB | BinaryOp.MUL:
                if both_int:
                    return self._apply(_ops.INT_BINARY[op], [left, right], origin, sink)
                left = self._as_float(left, origin, sink)
                right = self._as_float(right, origin, sink)
                if op is BinaryOp.SUB:
                    negated = self._apply(_ops.FLOAT_NEG, [right], origin, sink)
                    return self._apply(_ops.FLOAT_ADD, [left, negated], origin, sink)
                operator = _ops.FLOAT_ADD if op is BinaryOp.ADD else _ops.FLOAT_MUL
                return self._apply(operator, [left, right], origin, sink)
            case _:
                raise AssertionError(op)

    def _compare(self, origin: Origin, op: CompareOp, lv: Value, rv: Value, sink: list[Stmt]) -> Scalar:
        left = self._scalar(lv, origin)
        right = self._scalar(rv, origin)
        if left.stype is ScalarType.BOOL and right.stype is ScalarType.BOOL:
            if op is CompareOp.EQ:
                distinct = self._apply(_ops.BOOL_XOR, [left, right], origin, sink)
                return self._apply(_ops.BOOL_NOT, [distinct], origin, sink)
            if op is CompareOp.NE:
                return self._apply(_ops.BOOL_XOR, [left, right], origin, sink)
            reject(origin, "ordering comparisons of booleans are not supported")
        if ScalarType.BOOL in (left.stype, right.stype):
            reject(origin, "a boolean cannot be compared with a number; cast explicitly")
        if left.stype is ScalarType.INT and right.stype is ScalarType.INT:
            return self._apply(_ops.int_relational(op), [left, right], origin, sink)
        left = self._as_float(left, origin, sink)
        right = self._as_float(right, origin, sink)
        return self._apply(_ops.float_relational(op), [left, right], origin, sink)

    # ------------------------------------------------------------------ calls

    def _call(self, node: Call, frame: _Frame, sink: list[Stmt]) -> Value:
        callee = self._expr(node.callee, frame, sink)
        if not isinstance(callee, Opaque):
            reject(node.origin, "the callee is not a callable object")
        raw = callee.value
        match resolve(raw):
            case Intrinsic(operator=operator):
                return self._intrinsic(node, callee.name, operator, frame, sink)
            case Library(stub=stub):
                values = self._positional_arguments(node, callee.name, frame, sink)
                if stub is self._pow_stub:
                    # All pow spellings share the ** lowering, so the staged int/int rejection cannot be
                    # dodged by spelling.
                    if len(values) != 2:
                        reject(node.origin, f"{callee.name}() takes 2 argument(s), got {len(values)}")
                    base = self._scalar(values[0], node.origin)
                    exponent = self._scalar(values[1], node.origin)
                    return self._pow(node.origin, base, exponent, sink)
                return self._inline(node.origin, callee.name, stub, values, {}, sink, positional_only=True)
            case None:
                pass
        if raw is float or raw is int or raw is bool:
            target = raw
            assert isinstance(target, type)
            return self._cast(node, target, frame, sink)
        if isinstance(raw, types.FunctionType):
            # Ingested like user code wherever it lives: substitution is the registry's decision alone
            # (maintainer ruling), and the interpreter knows no library names.
            positional, keywords = self._signature_arguments(node, frame, sink)
            return self._inline(node.origin, callee.name, raw, positional, keywords, sink, positional_only=False)
        if inspect.ismethod(raw):
            reject(node.origin, "bound methods are not supported yet")
        if callable(raw):
            reject(node.origin, f"calls to {callee.name!r} are not supported yet")
        reject(node.origin, f"the captured object {callee.name!r} is not callable")

    def _positional_arguments(self, node: Call, display: str, frame: _Frame, sink: list[Stmt]) -> list[Value]:
        values: list[Value] = []
        for arg in node.args:
            match arg:
                case PosArg(value=value):
                    values.append(self._expr(value, frame, sink))
                case StarArg():
                    reject(node.origin, "aggregate values are not supported yet")
                case KwArg():
                    reject(node.origin, f"{display}() takes no keyword arguments")
        return values

    def _signature_arguments(self, node: Call, frame: _Frame, sink: list[Stmt]) -> tuple[list[Value], dict[str, Value]]:
        positional: list[Value] = []
        keywords: dict[str, Value] = {}
        for arg in node.args:
            match arg:
                case PosArg(value=value):
                    positional.append(self._expr(value, frame, sink))
                case StarArg():
                    reject(node.origin, "aggregate values are not supported yet")
                case KwArg(name=name, value=value):
                    keywords[name] = self._expr(value, frame, sink)
        return positional, keywords

    def _intrinsic(self, node: Call, display: str, operator: _ops.Operator, frame: _Frame, sink: list[Stmt]) -> Value:
        values = self._positional_arguments(node, display, frame, sink)
        stypes = _ops.operand_stypes(operator)
        if len(values) != len(stypes):
            reject(node.origin, f"{display}() takes {len(stypes)} argument(s), got {len(values)}")
        operands = [
            self._conform(value, stype, node.origin, sink, f"argument {i + 1} of {display}()")
            for i, (value, stype) in enumerate(zip(values, stypes, strict=True))
        ]
        return self._apply(operator, operands, node.origin, sink)

    def _inline(
        self,
        origin: Origin,
        display: str,
        fn: types.FunctionType,
        positional: list[Value],
        keywords: dict[str, Value],
        sink: list[Stmt],
        *,
        positional_only: bool,
    ) -> Value:
        site = Origin(origin.location, origin.frames + (CallFrame(display, origin.location),))
        if fn in self._inlining:
            reject(site, "recursive inlining is not supported")
        try:
            eel = self._desugared(fn)
        except SynthesisError as error:
            reattribute(site, error)
        annotations = self._annotations_of(site, fn)
        self._budget.spend(1, site, "the inlined call")
        callee = _rechained(eel, site.frames)
        assert isinstance(callee, EelFunction)
        if positional_only:
            assert not keywords
            if len(positional) != len(callee.params):
                reject(site, f"{display}() takes {len(callee.params)} argument(s), got {len(positional)}")
            bindings = {param.name: value for param, value in zip(callee.params, positional, strict=True)}
        else:
            bindings = self._bind_signature(site, fn, callee.params, positional, keywords)
        env: _Env = {}
        for param in callee.params:
            value = bindings[param.name]
            declared = _annotation_stype(annotations.get(param.name))
            if declared is not None:
                value = self._conform(value, declared, site, sink, f"the argument {param.name!r}")
            env[param.name] = value
        inner = _Frame(fn=fn, annotations=annotations, env=env, root=False)
        self._inlining.add(fn)
        try:
            returned = self._block(callee.body, inner, sink, in_branch=False)
        finally:
            self._inlining.discard(fn)
        if not returned or inner.result is None:
            reject(site, "the call returns no value, so it cannot be used in an expression")
        return inner.result

    def _bind_signature(
        self,
        site: Origin,
        fn: types.FunctionType,
        params: tuple[Param, ...],
        positional: list[Value],
        keywords: dict[str, Value],
    ) -> dict[str, Value]:
        signature = inspect.signature(fn)
        assert [param.name for param in params] == list(signature.parameters), "desugared params mirror the signature"
        try:
            bound = signature.bind(*positional, **keywords)
        except TypeError as error:
            reject(site, f"the arguments do not bind: {error}")
        bound.apply_defaults()
        bindings: dict[str, Value] = {}
        for name, value in bound.arguments.items():
            if isinstance(value, (StaticScalar, ResidualScalar, Opaque, TupleValue)):
                bindings[name] = value
            else:
                bindings[name] = self._admit(site, name, value)  # an injected default: a plain Python object
        return bindings

    def _cast(self, node: Call, target: type, frame: _Frame, sink: list[Stmt]) -> Value:
        if len(node.args) != 1 or not isinstance(node.args[0], PosArg):
            reject(node.origin, f"{target.__name__}() takes exactly one positional argument here")
        operand = self._scalar(self._expr(node.args[0].value, frame, sink), node.origin)
        declared = _annotation_stype(target)
        assert declared is not None
        if operand.stype is declared:
            return operand
        return self._apply(_ops.CONVERT[(operand.stype, declared)], [operand], node.origin, sink)

    # ------------------------------------------------------------------ powers

    def _pow(self, origin: Origin, base: Scalar, exponent: Scalar, sink: list[Stmt]) -> Scalar:
        if ScalarType.BOOL in (base.stype, exponent.stype):
            reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
        if isinstance(exponent, StaticScalar) and exponent.stype is ScalarType.INT:
            n = _ops.const_value(exponent.const)
            assert isinstance(n, int)
            if isinstance(base, StaticScalar):
                if (folded := self._fold_pow(base, n)) is not None:
                    return folded
            return self._pow_chain(origin, base, n, sink)
        if base.stype is ScalarType.INT and exponent.stype is ScalarType.INT:
            reject(
                origin,
                "a runtime integer exponent of an integer base is not supported yet; "
                "use a float base or a compile-time exponent",
            )
        promoted: list[Value] = [self._as_float(base, origin, sink), self._as_float(exponent, origin, sink)]
        value = self._inline(origin, "pow", self._pow_stub, promoted, {}, sink, positional_only=True)
        return self._scalar(value, origin)

    def _fold_pow(self, base: StaticScalar, n: int) -> StaticScalar | None:
        """
        The whole-power host fold for a fully static power with an integer exponent; an int power is exact at
        arbitrary precision (the budget carve-out demands better than a linear chain of huge folds). A refused
        value (overflow, zero to a negative power) falls back to the chain so the graph re-derives the fault
        survivor-based.
        """
        value = _ops.const_value(base.const)
        assert isinstance(value, (int, float))
        try:
            folded = value**n
        except (OverflowError, ZeroDivisionError, ValueError):
            return None
        assert type(folded) in (int, float), "an integer exponent cannot yield a complex power"
        return StaticScalar(_ops.make_const(folded))

    def _pow_chain(self, origin: Origin, base: Scalar, n: int, sink: list[Stmt]) -> Scalar:
        if n < 0:
            base = self._as_float(base, origin, sink)
            chain = self._chain(origin, base, -n, sink)
            one = StaticScalar(_ops.make_const(1.0))
            return self._apply(_ops.FLOAT_DIV, [one, chain], origin, sink)
        if n == 0:
            return StaticScalar(_ops.make_const(1 if base.stype is ScalarType.INT else 1.0))
        return self._chain(origin, base, n, sink)

    def _chain(self, origin: Origin, base: Scalar, n: int, sink: list[Stmt]) -> Scalar:
        assert n >= 1
        multiply = _ops.INT_BINARY[BinaryOp.MUL] if base.stype is ScalarType.INT else _ops.FLOAT_MUL
        result = base
        for _ in range(n - 1):
            self._budget.spend(1, origin, "the power chain")
            result = self._apply(multiply, [result, base], origin, sink)
        return result


# ---------------------------------------------------------------------- residual pruning


def _prune(stmts: list[Stmt] | tuple[Stmt, ...], live_after: set[int] | None = None) -> tuple[list[Stmt], set[int]]:
    """
    One reverse-liveness sweep; returns the surviving statements and the temp reads they need live-in.
    A branch whose arms both empty is dropped whole, de-rooting its condition so the condition's producer can
    disappear too. An arm is swept against the post-branch live set, so a join temp it assigns stays exactly
    when something after the branch reads it.
    """
    kept_reversed: list[Stmt] = []
    live = set(live_after) if live_after is not None else set()
    for stmt in reversed(stmts):
        match stmt:
            case Assign(target=TempBind(index=index), value=value):
                if index not in live:
                    continue
                live.discard(index)
                live |= _reads(value)
                kept_reversed.append(stmt)
            case If(origin=origin, cond=cond, then=then, orelse=orelse):
                then_kept, then_live = _prune(then, live)
                else_kept, else_live = _prune(orelse, live)
                if not then_kept and not else_kept:
                    continue
                live = then_live | else_live | _reads(cond)
                kept_reversed.append(If(origin, cond, tuple(then_kept), tuple(else_kept)))
            case ResidualReturn(values=values):
                for atom in values:
                    live |= _reads(atom)
                kept_reversed.append(stmt)
            case _:
                raise AssertionError(stmt)
    return list(reversed(kept_reversed)), live


def _reads(expr: Expr) -> set[int]:
    match expr:
        case TempRef(index=index):
            return {index}
        case IntrinsicCall(args=args):
            reads: set[int] = set()
            for arg in args:
                if isinstance(arg, TempRef):
                    reads.add(arg.index)
            return reads
        case LocalRef() | Const():
            return set()
        case _:
            raise AssertionError(expr)
