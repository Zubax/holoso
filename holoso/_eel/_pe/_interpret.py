"""
The specializing interpreter: Eel -> residual Eel, threading an environment of :mod:`._values` values.

Binding time is the interpreter's whole product. An operation whose operands are all static folds immediately
through the selected HIR operator's own ``evaluate`` (one expression, one answer -- see ``_ops``); when the
fold names no number the operation residualizes with constant operands instead, so refusal stays survivor-based
in HIR and the evaluator never convicts. No algebra is performed over residual operands -- ``x * 0`` and
friends belong to HIR strength reduction. Environment reads resolve lazily at evaluation time, so a
statically-dead arm never resolves its names nor judges its semantics -- exactly CPython's laziness -- while
the desugar whitelist has already judged its syntax.

Control: an ``if`` condition must be a bool at either binding time (no truthiness). A static condition folds
into the taken arm inline; a residual condition forks the environment, evaluates both arms into their own
statement sinks, and joins: every name bound on both sides is unified by the one join rule (int meeting float
promotes with conversions inserted inside the int arm; bool joins only with bool), materializing a join temp
assigned as the tail of both arms wherever the two values are not semantically identical. Names bound on one
side only are dropped from the joined environment, and a later read is a located rejection -- CPython would
raise ``UnboundLocalError`` on such a path. Divergent values that cannot merge at all (captured objects,
tuples) stay bound to an explicit unjoinable marker, so a later read draws a truthful located rejection in
every spelling, temps included, rather than an internal assertion.

The residual program is fully resolved, typed, and branch-normalized: every operation is an ``IntrinsicCall``
carrying its HIR operator, every residual assignment binds a fresh interpreter-numbered temp with a mandatory
``stype``, and parameters are the only named references. The module interface is validated here: parameters
and the return value require annotations, and the declared return type joins the inferred one (a declared
float accepts an int lane through promotion; any other divergence rejects the kernel).

A final reverse-liveness sweep drops residual pure temps nothing reads (a statically-decided consumer can
orphan its eagerly-hoisted operands, e.g. ``(a * b) ** 0``) and empty branches, mirroring HIR's own
DCE-before-refusal ordering. Recorded edge: a power chain over a static base saturates to infinity through
per-step folds where CPython's ``**`` raises ``OverflowError`` (``2.0 ** 10000``); this is identical to HIR
folding the same chain and is covered by the fastmath charter -- do not "fix" it.
"""

import inspect
import logging
import math
import types
import typing
from dataclasses import dataclass
from typing import NoReturn

from ..._errors import UnsupportedConstruct
from .._ir import *
from . import _ops
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

_MISSING = object()

_SCALAR_ANNOTATIONS: list[tuple[type, ScalarType]] = [
    (bool, ScalarType.BOOL),
    (int, ScalarType.INT),
    (float, ScalarType.FLOAT),
]


def partial_evaluate(eel: EelFunction, fn: types.FunctionType) -> EelFunction:
    return _Interpreter(fn).run(eel)


def _reject(origin: Origin, message: str) -> NoReturn:
    raise UnsupportedConstruct(message, origin.location)


def _annotation_stype(annotation: object) -> ScalarType | None:
    return next((stype for ty, stype in _SCALAR_ANNOTATIONS if annotation is ty), None)


def _key_order(key: _EnvKey) -> tuple[bool, str]:
    return isinstance(key, str), str(key)


class _Interpreter:
    def __init__(self, fn: types.FunctionType) -> None:
        self._fn = fn
        self._budget = ExpansionBudget()
        self._next_temp = 0
        self._outputs: tuple[OutputDecl, ...] = ()
        self._return_values: tuple[Atom, ...] = ()
        self._annotations: dict[str, str] = {}

    def run(self, eel: EelFunction) -> EelFunction:
        try:
            # eval_str accepts quoted annotations (`from __future__ import annotations` modules). Lazy PEP 649
            # annotations execute arbitrary user expressions here, so any exception is the user's -- a broad
            # catch into a located rejection is the honest containment, sanctioned alongside the shadow's.
            annotations = inspect.get_annotations(self._fn, eval_str=True)
        except Exception as error:
            _reject(eel.origin, f"the type annotations cannot be evaluated: {error}")
        self._annotations = annotations
        env: _Env = {}
        params = tuple(self._param(param, env) for param in eel.params)
        sink: list[Stmt] = []
        returned = self._block(eel.body, env, sink, in_branch=False)
        if not returned:
            self._conclude_bare(eel.origin)
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

    def _param(self, param: Param, env: _Env) -> Param:
        annotation = self._annotations.get(param.name, _MISSING)
        if annotation is _MISSING:
            _reject(param.origin, f"the parameter {param.name!r} requires a type annotation")
        stype = _annotation_stype(annotation)
        if stype is None:
            _reject(param.origin, f"the annotation of parameter {param.name!r} is not supported yet")
        env[param.name] = ResidualScalar(stype, LocalRef(param.origin, param.name))
        return Param(param.origin, param.name, param.kind, stype)

    def _fresh(self) -> int:
        index = self._next_temp
        self._next_temp += 1
        return index

    # ------------------------------------------------------------------ statements

    def _block(self, stmts: tuple[Stmt, ...], env: _Env, sink: list[Stmt], in_branch: bool) -> bool:
        for stmt in stmts:
            match stmt:
                case Assign(target=target, value=value):
                    env[self._binding_key(target)] = self._expr(value, env, sink)
                case AugAssign(origin=origin, target=target, op=op, value=value):
                    current = env.get(target.name)
                    if current is None:
                        _reject(origin, f"the local name {target.name!r} is not bound on every path reaching this read")
                    rhs = self._expr(value, env, sink)
                    env[target.name] = self._binary(origin, op, self._readable(current, origin), rhs, sink)
                case If():
                    if self._if(stmt, env, sink, in_branch):
                        return True
                case Return(origin=origin):
                    if in_branch:
                        _reject(origin, "a return inside a runtime branch is not supported yet")
                    self._conclude(stmt, env, sink)
                    return True
                case While(origin=origin) | For(origin=origin):
                    _reject(origin, "loops are not supported yet")
                case Break(origin=origin) | Continue(origin=origin):
                    _reject(origin, "loops are not supported yet")
                case Raise(origin=origin):
                    _reject(origin, "raise statements are not supported yet")
                case Unpack(origin=origin):
                    _reject(origin, "aggregate values are not supported yet")
                case Store(origin=origin) | AugStore(origin=origin):
                    _reject(origin, "stores through attributes or elements are not supported yet")
                case _:
                    raise AssertionError(stmt)
        return False

    def _binding_key(self, binding: Binding) -> _EnvKey:
        match binding:
            case LocalBind(name=name):
                return name
            case TempBind(index=index):
                return index

    def _if(self, stmt: If, env: _Env, sink: list[Stmt], in_branch: bool) -> bool:
        cond = self._scalar(self._expr(stmt.cond, env, sink), stmt.origin)
        if cond.stype is not ScalarType.BOOL:
            _reject(stmt.origin, "the branch condition must be a bool; Python truthiness is not supported")
        if isinstance(cond, StaticScalar):
            decided = _ops.const_value(cond.const)
            assert isinstance(decided, bool)
            taken = stmt.then if decided else stmt.orelse
            return self._block(taken, env, sink, in_branch)
        then_env, else_env = dict(env), dict(env)
        then_sink: list[Stmt] = []
        else_sink: list[Stmt] = []
        returned = self._block(stmt.then, then_env, then_sink, in_branch=True)
        assert not returned
        returned = self._block(stmt.orelse, else_env, else_sink, in_branch=True)
        assert not returned
        joined = self._join(stmt.origin, then_env, then_sink, else_env, else_sink)
        sink.append(If(stmt.origin, self._materialize(cond, stmt.origin), tuple(then_sink), tuple(else_sink)))
        env.clear()
        env.update(joined)
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
                _reject(
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
            _reject(
                origin,
                f"{binding.description} holds branch values the compiler cannot merge; "
                "only bool, int, and float values join branches",
            )
        return binding

    # ------------------------------------------------------------------ the module boundary

    def _conclude(self, stmt: Return, env: _Env, sink: list[Stmt]) -> None:
        annotation = self._annotations.get("return", _MISSING)
        if annotation is _MISSING:
            _reject(stmt.origin, "the return type annotation is required")
        if stmt.value is None:
            if annotation is not None:
                _reject(stmt.origin, "the kernel returns no value but its annotation declares one")
            return
        value = self._expr(stmt.value, env, sink)
        match value:
            case TupleValue(items=items):
                self._conclude_tuple(stmt.origin, annotation, items, sink)
            case StaticScalar() | ResidualScalar():
                declared = _annotation_stype(annotation)
                if declared is None:
                    _reject(stmt.origin, "the return annotation does not match the returned scalar value")
                leaf = self._conform(value, declared, stmt.origin, sink)
                self._outputs = (OutputDecl((0,), leaf.stype),)
                self._return_values = (self._materialize(leaf, stmt.origin),)
            case Opaque(name=name):
                _reject(stmt.origin, f"the captured object {name!r} cannot be returned")

    def _conclude_tuple(self, origin: Origin, annotation: object, items: tuple[Scalar, ...], sink: list[Stmt]) -> None:
        if typing.get_origin(annotation) is not tuple:
            _reject(origin, "the return annotation does not match the returned tuple")
        args = typing.get_args(annotation)
        if any(arg is Ellipsis for arg in args):
            _reject(origin, "variadic tuple return annotations are not supported yet")
        declared = [_annotation_stype(arg) for arg in args]
        if any(stype is None for stype in declared):
            _reject(origin, "the return annotation is not supported yet; only flat scalar tuples lower for now")
        if len(declared) != len(items):
            _reject(
                origin, f"the annotation declares {len(declared)} returned value(s), the kernel returns {len(items)}"
            )
        leaves: list[Scalar] = []
        for item, stype in zip(items, declared, strict=True):
            assert stype is not None
            leaves.append(self._conform(item, stype, origin, sink))
        self._outputs = tuple(OutputDecl((i,), leaf.stype) for i, leaf in enumerate(leaves))
        self._return_values = tuple(self._materialize(leaf, origin) for leaf in leaves)

    def _conclude_bare(self, origin: Origin) -> None:
        annotation = self._annotations.get("return", _MISSING)
        if annotation is _MISSING:
            _reject(origin, "the return type annotation is required")
        if annotation is not None:
            _reject(origin, "the kernel can complete without returning a value but its annotation declares one")

    def _conform(self, scalar: Scalar, declared: ScalarType, origin: Origin, sink: list[Stmt]) -> Scalar:
        if scalar.stype is declared:
            return scalar
        if scalar.stype is ScalarType.INT and declared is ScalarType.FLOAT:
            return self._as_float(scalar, origin, sink)
        _reject(
            origin,
            f"the returned value has type {scalar.stype.value} where the annotation declares {declared.value}",
        )

    # ------------------------------------------------------------------ expressions

    def _expr(self, expr: Expr, env: _Env, sink: list[Stmt]) -> Value:
        match expr:
            case Const(value=value):
                return StaticScalar(_ops.make_const(value))
            case TempRef(origin=origin, index=index):
                bound = env.get(index)
                assert bound is not None, "desugar binds every temp before its first read"
                return self._readable(bound, origin)
            case LocalRef(origin=origin, name=name):
                found = env.get(name)
                if found is None:
                    _reject(origin, f"the local name {name!r} is not bound on every path reaching this read")
                return self._readable(found, origin)
            case EnvRead():
                return self._env_read(expr)
            case Unary(origin=origin, op=op, operand=operand):
                return self._unary(origin, op, self._expr(operand, env, sink), sink)
            case Binary(origin=origin, op=op, left=left, right=right):
                lv = self._expr(left, env, sink)
                rv = self._expr(right, env, sink)
                return self._binary(origin, op, lv, rv, sink)
            case Compare(origin=origin, op=op, left=left, right=right):
                lv = self._expr(left, env, sink)
                rv = self._expr(right, env, sink)
                return self._compare(origin, op, lv, rv, sink)
            case Call():
                return self._call(expr, env, sink)
            case TupleExpr(origin=origin, items=items):
                scalars: list[Scalar] = []
                for item in items:
                    if isinstance(item, StarArg):
                        _reject(origin, "aggregate values are not supported yet")
                    scalars.append(self._scalar(self._expr(item, env, sink), origin))
                return TupleValue(tuple(scalars))
            case AttrRead(origin=origin):
                _reject(origin, "attribute access is not supported yet")
            case IndexRead(origin=origin) | SliceRead(origin=origin) | MultiIndexRead(origin=origin):
                _reject(origin, "aggregate values are not supported yet")
            case ListExpr(origin=origin):
                _reject(origin, "aggregate values are not supported yet")
            case Comp(origin=origin):
                _reject(origin, "comprehensions are not supported yet")
            case _:
                raise AssertionError(expr)

    def _env_read(self, node: EnvRead) -> Value:
        if node.free:
            code = self._fn.__code__
            assert node.name in code.co_freevars
            closure = self._fn.__closure__
            assert closure is not None
            cell = closure[code.co_freevars.index(node.name)]
            try:
                raw = cell.cell_contents
            except ValueError:
                _reject(node.origin, f"the free name {node.name!r} is unbound in its enclosing scope")
        else:
            if node.name in self._fn.__globals__:
                raw = self._fn.__globals__[node.name]
            else:
                # The function's own builtins mapping, which is what CPython would consult -- absent from
                # typeshed's FunctionType, hence the getattr.
                builtins_ns = getattr(self._fn, "__builtins__")
                assert isinstance(builtins_ns, dict)
                if node.name not in builtins_ns:
                    _reject(node.origin, f"the name {node.name!r} is not defined")
                raw = builtins_ns[node.name]
        return self._admit(node.origin, node.name, raw)

    def _admit(self, origin: Origin, name: str, raw: object) -> Value:
        if type(raw) is bool or type(raw) is int:
            return StaticScalar(_ops.make_const(raw))
        if type(raw) is float:
            if math.isnan(raw):
                _reject(origin, f"the captured value of {name!r} is NaN, which the compiler cannot represent")
            return StaticScalar(_ops.make_const(raw))
        return Opaque(name, raw)

    def _scalar(self, value: Value, origin: Origin) -> Scalar:
        match value:
            case StaticScalar() | ResidualScalar():
                return value
            case Opaque(name=name):
                _reject(origin, f"the captured value of {name!r} is not a bool, int, or float scalar")
            case TupleValue():
                _reject(origin, "a tuple is only supported as the returned value for now")

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
                    _reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
                if op is UnaryOp.POS:
                    return operand
                negate = _ops.INT_NEG if operand.stype is ScalarType.INT else _ops.FLOAT_NEG
                return self._apply(negate, [operand], origin, sink)
            case UnaryOp.NOT:
                if operand.stype is not ScalarType.BOOL:
                    _reject(origin, "`not` requires a bool operand; Python truthiness is not supported")
                return self._apply(_ops.BOOL_NOT, [operand], origin, sink)
            case UnaryOp.INVERT:
                if operand.stype is not ScalarType.INT:
                    _reject(origin, "`~` is integer-only")
                return self._apply(_ops.INT_NOT, [operand], origin, sink)

    def _binary(self, origin: Origin, op: BinaryOp, lv: Value, rv: Value, sink: list[Stmt]) -> Scalar:
        left = self._scalar(lv, origin)
        right = self._scalar(rv, origin)
        both_bool = left.stype is ScalarType.BOOL and right.stype is ScalarType.BOOL
        both_int = left.stype is ScalarType.INT and right.stype is ScalarType.INT
        match op:
            case BinaryOp.AND | BinaryOp.OR:
                if not both_bool:
                    _reject(origin, "the boolean gates require bool operands; Python truthiness is not supported")
                return self._apply(_ops.BOOL_GATES[op], [left, right], origin, sink)
            case BinaryOp.BITAND | BinaryOp.BITOR | BinaryOp.BITXOR:
                if both_bool:
                    return self._apply(_ops.BOOL_GATES[op], [left, right], origin, sink)
                if both_int:
                    return self._apply(_ops.INT_BINARY[op], [left, right], origin, sink)
                _reject(origin, f"the bitwise operator `{op.value}` requires two ints or two bools")
            case BinaryOp.POW:
                return self._pow(origin, left, right, sink)
            case BinaryOp.MATMUL:
                _reject(origin, "aggregate values are not supported yet")
            case _:
                pass
        if ScalarType.BOOL in (left.stype, right.stype):
            _reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
        match op:
            case BinaryOp.FLOORDIV | BinaryOp.MOD | BinaryOp.LSHIFT | BinaryOp.RSHIFT:
                if not both_int:
                    _reject(origin, f"the operator `{op.value}` is integer-only")
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
            _reject(origin, "ordering comparisons of booleans are not supported")
        if ScalarType.BOOL in (left.stype, right.stype):
            _reject(origin, "a boolean cannot be compared with a number; cast explicitly")
        if left.stype is ScalarType.INT and right.stype is ScalarType.INT:
            return self._apply(_ops.int_relational(op), [left, right], origin, sink)
        left = self._as_float(left, origin, sink)
        right = self._as_float(right, origin, sink)
        return self._apply(_ops.float_relational(op), [left, right], origin, sink)

    # ------------------------------------------------------------------ calls and powers

    def _call(self, node: Call, env: _Env, sink: list[Stmt]) -> Value:
        callee = self._expr(node.callee, env, sink)
        if not isinstance(callee, Opaque):
            _reject(node.origin, "the callee is not a callable object")
        if callee.value is float or callee.value is int or callee.value is bool:
            target = callee.value
            assert isinstance(target, type)
            return self._cast(node, target, env, sink)
        if callable(callee.value):
            _reject(node.origin, f"calls are not supported yet (cannot lower the call to {callee.name!r})")
        _reject(node.origin, f"the captured object {callee.name!r} is not callable")

    def _cast(self, node: Call, target: type, env: _Env, sink: list[Stmt]) -> Value:
        if len(node.args) != 1 or not isinstance(node.args[0], PosArg):
            _reject(node.origin, f"{target.__name__}() takes exactly one positional argument here")
        operand = self._scalar(self._expr(node.args[0].value, env, sink), node.origin)
        declared = _annotation_stype(target)
        assert declared is not None
        if operand.stype is declared:
            return operand
        return self._apply(_ops.CONVERT[(operand.stype, declared)], [operand], node.origin, sink)

    def _pow(self, origin: Origin, base: Scalar, exponent: Scalar, sink: list[Stmt]) -> Scalar:
        # STAGING STUB, due for replacement -- the residual surface below is deliberately narrow because the
        # machinery the ruled design needs does not exist yet. The target semantics (maintainer ruling at M3):
        # heterogeneous operands C-promote to float, and a residual float**float lowers by inlining the
        # registry's `_lib._numpy.pow_` stub -- which requires M4's call dispatch and stub inlining. Once that
        # lands, the static-base-2.0 exp2 shortcut below must DISSOLVE rather than survive: pow_(2.0, x)
        # static-folds through the stub body (log2(2.0) -> 1.0) into plain exp2(x). A runtime INTEGER exponent
        # of an integer base additionally waits for a loop-carrying `pow_int_` stub (M7). Only the fully
        # static folds and the static-integer-exponent chains here are the permanent semantics.
        if ScalarType.BOOL in (base.stype, exponent.stype):
            _reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
        if isinstance(exponent, StaticScalar) and exponent.stype is ScalarType.INT:
            n = _ops.const_value(exponent.const)
            assert isinstance(n, int)
            if isinstance(base, StaticScalar):
                if (folded := self._fold_pow(base, n)) is not None:
                    return folded
            return self._pow_chain(origin, base, n, sink)
        if isinstance(base, StaticScalar) and isinstance(exponent, StaticScalar):
            return self._fold_general_pow(origin, base, exponent)
        if base.stype is ScalarType.INT and exponent.stype is ScalarType.INT:
            _reject(
                origin,
                "a runtime integer exponent of an integer base is not supported yet; "
                "use 2.0 as the base or a compile-time exponent",
            )
        if isinstance(base, StaticScalar) and base.stype in (ScalarType.INT, ScalarType.FLOAT):
            base_value = _ops.const_value(base.const)
            assert isinstance(base_value, (int, float))
            # Exact cross-type equality: only the int 2 and the float 2.0 compare equal, and unlike a float()
            # image test it cannot overflow on a huge integer base.
            if base_value == 2:
                return self._apply(_ops.FLOAT_EXP2, [self._as_float(exponent, origin, sink)], origin, sink)
        _reject(
            origin,
            "this power form is not supported yet; only compile-time integer exponents and the base 2 "
            "(2 ** x is exp2) lower for now",
        )

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

    def _fold_general_pow(self, origin: Origin, base: StaticScalar, exponent: StaticScalar) -> Scalar:
        """
        A fully static power with a non-integer exponent folds over the C-promoted float images of both
        operands. There is no residual spelling for a refused fold until M4's pow_ inlining, so a value the
        fold cannot name (overflow, an unpromotable base, a complex result from a negative base) is a located
        rejection here rather than a survivor-based one; M4 must lift these onto the pow_ residual form.
        """
        base_value = _ops.const_value(base.const)
        exponent_value = _ops.const_value(exponent.const)
        assert isinstance(base_value, (int, float)) and isinstance(exponent_value, (int, float))
        try:
            folded = float(base_value) ** float(exponent_value)
        except (OverflowError, ZeroDivisionError, ValueError):
            _reject(origin, f"the power {base_value} ** {exponent_value} names no number")
        if isinstance(folded, complex):
            _reject(origin, f"the power {base_value} ** {exponent_value} names a complex value")
        if math.isnan(folded):
            _reject(origin, f"the power {base_value} ** {exponent_value} names no number")
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
