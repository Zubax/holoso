"""The expression side of the specializing interpreter: operators, attribute reads, and calls."""

import dataclasses
import inspect
import types
from typing import TYPE_CHECKING

import numpy as np

from .._annotations import annotation_stype
from .._ir import *
from .._lib import Array, Conversion, Factory, Operand, ScalarFunction, resolve
from . import _aggregate, _ops
from ._ownership import share
from ._reject import reject
from ._snapshot import describe_opaque as _describe_opaque, nan_payload, tensor_of
from ._values import (
    Allocation,
    Opaque,
    ResidualScalar,
    Scalar,
    SequenceValue,
    StaticScalar,
    TensorMethod,
    TensorValue,
    Value,
)

if TYPE_CHECKING:
    from ._interpret import Frame, Interpreter, Sink

_AGGREGATE = (SequenceValue, TensorValue)

# ------------------------------------------------------------------ attribute reads


def attr_read(interp: Interpreter, origin: Origin, base_value: Value, attr: str, frame: Frame, sink: Sink) -> Value:
    match base_value:
        case TensorValue():
            return _tensor_attr(interp, origin, base_value, attr, frame, sink)
        case SequenceValue():
            if attr in ("shape", "ndim") or resolve(getattr(np.ndarray, attr, None)) is not None:
                reject(
                    origin,
                    f"`.{attr}` on a Python sequence is not supported; " "build a numpy array with np.array([...])",
                )
            reject(origin, f"a sequence has no supported attribute {attr!r}")
        case StaticScalar() | ResidualScalar():
            if attr == "ndim":
                return StaticScalar(_ops.make_const(0))
            if attr == "shape":
                return SequenceValue((), Allocation())
            reject(origin, f"a scalar has no supported attribute {attr!r}")
        case TensorMethod():
            reject(origin, "a bound array method can only be called")
        case Opaque(name=name, value=value) if isinstance(value, (types.ModuleType, type)):
            # A module/class attribute is a metadata read (class access unwraps staticmethod and
            # plain functions); an instance never runs live descriptors, below.
            try:
                raw = getattr(value, attr)
            except AttributeError:
                reject(origin, f"{name!r} has no attribute {attr!r}")
            return interp.snapshot.admit(f"{name}.{attr}", raw, origin)
        case Opaque():
            return interp.instance_attr(origin, base_value, attr, frame, sink)
        case _:
            raise AssertionError(base_value)


def _tensor_attr(
    interp: Interpreter, origin: Origin, tensor: TensorValue, attr: str, frame: Frame, sink: Sink
) -> Value:
    if attr == "ndim":
        return StaticScalar(_ops.make_const(len(tensor.shape)))
    if attr == "shape":
        dims = tuple(StaticScalar(_ops.make_const(dim)) for dim in tensor.shape)
        return SequenceValue(dims, Allocation())
    descriptor = getattr(np.ndarray, attr, None)
    found = resolve(descriptor)
    if isinstance(found, Array):
        if inspect.isdatadescriptor(descriptor):
            return _array_call(interp, origin, f".{attr}", found, [tensor], frame, sink)
        share(tensor)
        return TensorMethod(tensor, attr)
    reject(origin, f"an array has no supported attribute {attr!r}")


# ------------------------------------------------------------------ scalars and operators


def scalar(value: Value, origin: Origin) -> Scalar:
    match value:
        case StaticScalar() | ResidualScalar():
            return value
        case Opaque():
            reject(origin, _describe_opaque(value))
        case SequenceValue() | TensorValue():
            reject(origin, f"{_aggregate.a_kind(value)} cannot be used as a scalar here")
        case TensorMethod():
            reject(origin, "a bound array method can only be called")


def materialize(scalar: Scalar, origin: Origin) -> Atom:
    match scalar:
        case StaticScalar(const=const):
            return Const(origin, _ops.const_value(const))
        case ResidualScalar(atom=atom):
            return atom


def apply(interp: Interpreter, operator: _ops.Operator, operands: list[Scalar], origin: Origin, sink: Sink) -> Scalar:
    if all(isinstance(operand, StaticScalar) for operand in operands):
        consts = [operand.const for operand in operands if isinstance(operand, StaticScalar)]
        try:
            return StaticScalar(operator.evaluate(consts))
        except _ops.NoNumber:
            pass  # the graph re-derives the fault and the survivor sweep judges it; never convict here
    stype = _ops.result_stype(operator)
    atoms = tuple(materialize(operand, origin) for operand in operands)
    index = interp.fresh()
    sink.append(Assign(origin, TempBind(origin, index), IntrinsicCall(origin, operator, atoms), stype))
    return ResidualScalar(stype, TempRef(origin, index))


def unary(interp: Interpreter, origin: Origin, op: UnaryOp, value: Value, sink: Sink) -> Value:
    if isinstance(value, TensorValue) and op in (UnaryOp.NEG, UnaryOp.POS):
        leaves = tuple(_unary_leaf(interp, origin, op, leaf, sink) for leaf in value.leaves)
        interp.budget.spend(len(leaves), origin, "the elementwise operation")
        return TensorValue(value.shape, value.family, leaves, Allocation())
    if isinstance(value, _AGGREGATE):
        if op is UnaryOp.NOT:
            reject(origin, "the truthiness of an aggregate is not supported")
        reject(origin, f"`{op.value}` is not supported on {_aggregate.a_kind(value)}")
    operand = scalar(value, origin)
    match op:
        case UnaryOp.NEG | UnaryOp.POS:
            if operand.stype is ScalarType.BOOL:
                reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
            if op is UnaryOp.POS:
                return operand
            negate = _ops.INT_NEG if operand.stype is ScalarType.INT else _ops.FLOAT_NEG
            return apply(interp, negate, [operand], origin, sink)
        case UnaryOp.NOT:
            if operand.stype is not ScalarType.BOOL:
                reject(origin, "`not` requires a bool operand; Python truthiness is not supported")
            return apply(interp, _ops.BOOL_NOT, [operand], origin, sink)
        case UnaryOp.INVERT:
            if operand.stype is not ScalarType.INT:
                reject(origin, "`~` is integer-only")
            return apply(interp, _ops.INT_BW_NOT, [operand], origin, sink)


def _unary_leaf(interp: Interpreter, origin: Origin, op: UnaryOp, leaf: Scalar | Opaque, sink: Sink) -> Scalar | Opaque:
    if op is UnaryOp.POS:
        return leaf
    result = unary(interp, origin, op, _scalar_leaf(origin, leaf), sink)
    assert isinstance(result, (StaticScalar, ResidualScalar))
    return result


def _scalar_leaf(origin: Origin, leaf: Scalar | Opaque) -> Scalar:
    if isinstance(leaf, Opaque):
        reject(origin, _describe_opaque(leaf))
    return leaf


def binary(interp: Interpreter, origin: Origin, op: BinaryOp, lv: Value, rv: Value, frame: Frame, sink: Sink) -> Value:
    if op is BinaryOp.MATMUL:
        for operand in (lv, rv):
            if isinstance(operand, _AGGREGATE):
                share(operand)
        return _operator_call(interp, origin, op, [lv, rv], frame, sink)
    tensors = isinstance(lv, TensorValue) or isinstance(rv, TensorValue)
    sequences = isinstance(lv, SequenceValue) or isinstance(rv, SequenceValue)
    if (tensors or sequences) and op in (BinaryOp.AND, BinaryOp.OR):
        reject(origin, "the truthiness of an aggregate is not supported")
    if tensors and sequences:
        reject(origin, "cannot mix an array with a Python list/tuple; convert with np.array([...])")
    if tensors:
        return elementwise(interp, origin, op, lv, rv, frame, sink)
    if sequences:
        return _sequence_binary(origin, op)
    return _binary_scalars(interp, origin, op, lv, rv, frame, sink)


def elementwise(
    interp: Interpreter, origin: Origin, op: BinaryOp, lv: Value, rv: Value, frame: Frame, sink: Sink
) -> TensorValue:
    if op not in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV):
        reject(origin, f"the operator `{op.value}` is not supported on arrays yet")
    pairs: list[tuple[Scalar, Scalar]]
    shape: tuple[int, ...]
    if isinstance(lv, TensorValue) and isinstance(rv, TensorValue):
        if lv.shape != rv.shape:
            reject(origin, f"array shapes {lv.shape} and {rv.shape} do not match; broadcasting is not supported")
        shape = lv.shape
        pairs = [(_scalar_leaf(origin, a), _scalar_leaf(origin, b)) for a, b in zip(lv.leaves, rv.leaves, strict=True)]
    elif isinstance(lv, TensorValue):
        operand = scalar(rv, origin)
        shape = lv.shape
        pairs = [(_scalar_leaf(origin, a), operand) for a in lv.leaves]
    else:
        assert isinstance(rv, TensorValue)
        operand = scalar(lv, origin)
        shape = rv.shape
        pairs = [(operand, _scalar_leaf(origin, b)) for b in rv.leaves]
    leaves: list[Scalar | Opaque] = []
    for a, b in pairs:
        leaves.append(_binary_scalars(interp, origin, op, a, b, frame, sink))
    interp.budget.spend(len(leaves), origin, "the elementwise operation")
    family = next(leaf.stype for leaf in leaves if not isinstance(leaf, Opaque))
    return TensorValue(shape, family, tuple(leaves), Allocation())


def _sequence_binary(origin: Origin, op: BinaryOp) -> Value:
    reject(origin, f"the operator `{op.value}` is not supported on a sequence; build a numpy array instead")


def _binary_scalars(
    interp: Interpreter, origin: Origin, op: BinaryOp, lv: Value, rv: Value, frame: Frame, sink: Sink
) -> Scalar:
    left = scalar(lv, origin)
    right = scalar(rv, origin)
    both_bool = left.stype is ScalarType.BOOL and right.stype is ScalarType.BOOL
    both_int = left.stype is ScalarType.INT and right.stype is ScalarType.INT
    match op:
        case BinaryOp.AND | BinaryOp.OR:
            if not both_bool:
                reject(origin, "the boolean gates require bool operands; Python truthiness is not supported")
            return apply(interp, _ops.BOOL_GATES[op], [left, right], origin, sink)
        case BinaryOp.BITAND | BinaryOp.BITOR | BinaryOp.BITXOR:
            if both_bool:
                return apply(interp, _ops.BOOL_GATES[op], [left, right], origin, sink)
            if both_int:
                return apply(interp, _ops.INT_BINARY[op], [left, right], origin, sink)
            reject(origin, f"the bitwise operator `{op.value}` requires two ints or two bools")
        case BinaryOp.POW:
            return scalar(_operator_call(interp, origin, op, [left, right], frame, sink), origin)
        case _:
            pass
    if ScalarType.BOOL in (left.stype, right.stype):
        reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
    match op:
        case BinaryOp.FLOORDIV | BinaryOp.MOD | BinaryOp.LSHIFT | BinaryOp.RSHIFT:
            if not both_int:
                reject(origin, f"the operator `{op.value}` is integer-only")
            return apply(interp, _ops.INT_BINARY[op], [left, right], origin, sink)
        case BinaryOp.DIV:
            dividend = interp.as_float(left, origin, sink)
            divisor = interp.as_float(right, origin, sink)
            return apply(interp, _ops.FLOAT_DIV, [dividend, divisor], origin, sink)
        case BinaryOp.ADD | BinaryOp.SUB | BinaryOp.MUL:
            if both_int:
                return apply(interp, _ops.INT_BINARY[op], [left, right], origin, sink)
            left = interp.as_float(left, origin, sink)
            right = interp.as_float(right, origin, sink)
            if op is BinaryOp.SUB:
                negated = apply(interp, _ops.FLOAT_NEG, [right], origin, sink)
                return apply(interp, _ops.FLOAT_ADD, [left, negated], origin, sink)
            operator = _ops.FLOAT_ADD if op is BinaryOp.ADD else _ops.FLOAT_MUL
            return apply(interp, operator, [left, right], origin, sink)
        case _:
            raise AssertionError(op)


def compare(interp: Interpreter, origin: Origin, op: CompareOp, lv: Value, rv: Value, sink: Sink) -> Scalar:
    if isinstance(lv, _AGGREGATE) or isinstance(rv, _AGGREGATE):
        reject(origin, "aggregate comparison is not supported")
    left = scalar(lv, origin)
    right = scalar(rv, origin)
    if left.stype is ScalarType.BOOL and right.stype is ScalarType.BOOL:
        if op is CompareOp.EQ:
            distinct = apply(interp, _ops.BOOL_XOR, [left, right], origin, sink)
            return apply(interp, _ops.BOOL_NOT, [distinct], origin, sink)
        if op is CompareOp.NE:
            return apply(interp, _ops.BOOL_XOR, [left, right], origin, sink)
        reject(origin, "ordering comparisons of booleans are not supported")
    if ScalarType.BOOL in (left.stype, right.stype):
        reject(origin, "a boolean cannot be compared with a number; cast explicitly")
    if left.stype is ScalarType.INT and right.stype is ScalarType.INT:
        return apply(interp, _ops.INT_COMPARE[op], [left, right], origin, sink)
    left = interp.as_float(left, origin, sink)
    right = interp.as_float(right, origin, sink)
    return apply(interp, _ops.FLOAT_COMPARE[op], [left, right], origin, sink)


# ------------------------------------------------------------------ calls


def call(interp: Interpreter, node: Call, frame: Frame, sink: Sink) -> Value:
    callee = interp.expr(node.callee, frame, sink)
    if isinstance(callee, TensorMethod):
        return _tensor_method(interp, node, callee, frame, sink)
    if not isinstance(callee, Opaque):
        reject(node.origin, "the callee is not a callable object")
    raw = callee.value
    match resolve(raw) if callable(raw) else None:
        case ScalarFunction() as match:
            values = _operand_arguments(interp, node, callee.name, frame, sink)
            return _scalar_call(interp, node.origin, callee.name, match, values, frame, sink)
        case Array() as match:
            values = _positional_arguments(interp, node, callee.name, frame, sink)
            return _array_call(interp, node.origin, callee.name, match, values, frame, sink)
        case Factory() as match:
            return _factory(interp, node, callee.name, match, frame, sink)
        case Conversion(copies=copies):
            values = _operand_arguments(interp, node, callee.name, frame, sink)
            if len(values) != 1:
                reject(node.origin, f"{callee.name}() takes a single array-like argument")
            return _to_tensor(interp, node.origin, callee.name, values[0], sink, copies=copies)
        case None:
            pass
    if raw is float or raw is int or raw is bool:
        target = raw
        assert isinstance(target, type)
        return _cast(interp, node, target, frame, sink)
    if raw is len:
        return _len(interp, node, frame, sink)
    if raw is range:
        return _range(interp, node, frame, sink)
    if raw is list or raw is tuple:
        return _rebuild_sequence(interp, node, callee.name, frame, sink)
    if isinstance(raw, types.FunctionType):
        # Ingested like user code wherever it lives: substitution is the registry's decision alone,
        # and the interpreter knows no library names.
        positional, keywords = _signature_arguments(interp, node, frame, sink)
        return interp.inline(node.origin, callee.name, raw, positional, keywords, frame, sink, positional_only=False)
    if inspect.ismethod(raw):
        # A helper method of any snapshotted object: the receiver rides as the first argument, sharing
        # the memoized identity, so its frozen attributes fold and its state reads see the current slots;
        # a receiver write inside the helper draws the helper-write rejection.
        inner = raw.__func__
        if not isinstance(inner, types.FunctionType):
            reject(node.origin, f"calls to {callee.name!r} are not supported yet")
        receiver = interp.snapshot.admit(callee.name, raw.__self__, node.origin)
        positional, keywords = _signature_arguments(interp, node, frame, sink)
        return interp.inline(
            node.origin,
            callee.name,
            inner,
            [receiver, *positional],
            keywords,
            frame,
            sink,
            positional_only=False,
        )
    if callable(raw):
        # A component instance defines __call__ as a plain Python function; a C-level one (a numpy dispatcher or
        # ufunc, a partial) is simply a callee the registry does not carry.
        if isinstance(getattr(type(raw), "__call__", None), types.FunctionType):
            reject(
                node.origin,
                f"cannot call {callee.name!r}: it is a separate component instance "
                f"({type(raw).__name__}); hierarchical state is not supported yet",
            )
        reject(node.origin, f"calls to {callee.name!r} are not supported yet")
    reject(node.origin, f"the captured object {callee.name!r} is not callable")


def _argument(interp: Interpreter, atom: Atom, frame: Frame, sink: Sink) -> Value:
    value = interp.expr(atom, frame, sink)
    if isinstance(value, _AGGREGATE) and (isinstance(atom, LocalRef) or interp.alias_conduit(frame, atom)):
        share(value)
    return value


def _operand_arguments(interp: Interpreter, node: Call, display: str, frame: Frame, sink: Sink) -> list[Value]:
    """Arguments for a callee that retains no handle: no aliasing event, unlike parameter binding."""
    values: list[Value] = []
    for arg in node.args:
        match arg:
            case PosArg(value=value):
                values.append(interp.expr(value, frame, sink))
            case StarArg(value=value):
                values.extend(_aggregate.splice_items(node.origin, interp.expr(value, frame, sink)))
            case KwArg():
                reject(node.origin, f"{display}() takes no keyword arguments")
    return values


def _positional_arguments(interp: Interpreter, node: Call, display: str, frame: Frame, sink: Sink) -> list[Value]:
    values: list[Value] = []
    for arg in node.args:
        match arg:
            case PosArg(value=value):
                values.append(_argument(interp, value, frame, sink))
            case StarArg(value=value):
                values.extend(_aggregate.splice_items(node.origin, interp.expr(value, frame, sink)))
            case KwArg():
                reject(node.origin, f"{display}() takes no keyword arguments")
    return values


def _signature_arguments(
    interp: Interpreter, node: Call, frame: Frame, sink: Sink
) -> tuple[list[Value], dict[str, Value]]:
    positional: list[Value] = []
    keywords: dict[str, Value] = {}
    for arg in node.args:
        match arg:
            case PosArg(value=value):
                positional.append(_argument(interp, value, frame, sink))
            case StarArg(value=value):
                positional.extend(_aggregate.splice_items(node.origin, interp.expr(value, frame, sink)))
            case KwArg(name=name, value=value):
                keywords[name] = _argument(interp, value, frame, sink)
    return positional, keywords


def _operator_call(
    interp: Interpreter, origin: Origin, op: BinaryOp, values: list[Value], frame: Frame, sink: Sink
) -> Value:
    """The spelled call resolves this same entry, so an operator and its spellings cannot drift apart."""
    found = resolve(op)
    assert found is not None, op
    display = op.value
    match found:
        case ScalarFunction() as match:
            return _scalar_call(interp, origin, display, match, values, frame, sink)
        case Array() as match:
            return _array_call(interp, origin, display, match, values, frame, sink)
        case _:
            raise AssertionError(found)


def _scalar_call(
    interp: Interpreter,
    origin: Origin,
    display: str,
    match: ScalarFunction,
    values: list[Value],
    frame: Frame,
    sink: Sink,
) -> Value:
    if len(values) != match.arity:
        reject(origin, f"{display}() takes {match.arity} argument(s), got {len(values)}")
    operands = [scalar(value, origin) for value in values]
    chosen = match.select([_operand(operand) for operand in operands])
    if chosen is None:
        served = " or ".join(stype.value for stype in match.domains)
        got = ", ".join(operand.stype.value for operand in operands)
        boolean = any(operand.stype is ScalarType.BOOL for operand in operands)
        note = "; a boolean is not a number, cast explicitly with int(...) or float(...)" if boolean else ""
        reject(origin, f"{display}() takes {served} operands, got {got}{note}")
    # Both kinds of lowering, so inlining judges base types alone and never learns what a refinement is.
    conformed = [
        interp.conform(operand, declared.stype, origin, sink, f"argument {i + 1} of {display}()")
        for i, (operand, declared) in enumerate(zip(operands, chosen.operands, strict=True))
    ]
    if chosen.operator is None:
        promoted: list[Value] = list(conformed)
        return interp.inline(origin, display, chosen.stub, promoted, {}, frame, sink, positional_only=True)
    return apply(interp, chosen.operator, conformed, origin, sink)


def _operand(value: Scalar) -> Operand:
    if isinstance(value, StaticScalar):
        const = _ops.const_value(value.const)
        assert isinstance(const, (bool, int, float))
        return Operand(value.stype, const)
    return Operand(value.stype)


def _array_call(
    interp: Interpreter,
    origin: Origin,
    display: str,
    match: Array,
    values: list[Value],
    frame: Frame,
    sink: Sink,
) -> Value:
    if any(isinstance(value, SequenceValue) for value in values):
        # Before the stub, so the rejection reads the same however the operation was spelled.
        reject(origin, "an array operation on a Python list/tuple is not supported; build one with np.array([...])")
    result = interp.inline(origin, display, match.stub, values, {}, frame, sink, positional_only=True)
    if match.derives and isinstance(values[0], TensorValue) and isinstance(result, TensorValue):
        share(values[0])
        result = dataclasses.replace(result, allocation=values[0].allocation)
    return result


def _tensor_method(interp: Interpreter, node: Call, method: TensorMethod, frame: Frame, sink: Sink) -> Value:
    found = resolve(getattr(np.ndarray, method.name))
    assert isinstance(found, Array), "a minted method stays resolvable"
    display = f".{method.name}"
    values = _positional_arguments(interp, node, display, frame, sink)
    arity = found.stub.__code__.co_argcount - 1
    if len(values) != arity:
        reject(node.origin, f"{display}() takes {arity} argument(s), got {len(values)}")
    return _array_call(interp, node.origin, display, found, [method.receiver, *values], frame, sink)


def _factory(interp: Interpreter, node: Call, display: str, match: Factory, frame: Frame, sink: Sink) -> Value:
    values = _operand_arguments(interp, node, display, frame, sink)
    host = [_static_argument(node.origin, display, value) for value in values]
    try:
        built = match.build(*host)
    except Exception as error:  # a registered library builder, not user code; its refusal is a diagnostic
        reject(node.origin, f"{display}() rejects its arguments: {error}")
    tensor = tensor_of(built, display)
    if tensor is None:
        reject(node.origin, f"{display}() must build a non-empty 1-D or 2-D numeric array")
    interp.budget.spend(len(tensor.leaves), node.origin, "the array factory")
    return tensor


def _static_argument(origin: Origin, display: str, value: Value) -> object:
    match value:
        case StaticScalar(const=const):
            return _ops.const_value(const)
        case SequenceValue(items=items):
            return tuple(_static_argument(origin, display, item) for item in items)
        case _:
            reject(origin, f"the arguments of {display}() must be compile-time constants")


def _len(interp: Interpreter, node: Call, frame: Frame, sink: Sink) -> Value:
    values = _operand_arguments(interp, node, "len", frame, sink)
    if len(values) != 1:
        reject(node.origin, f"len() takes exactly one argument, got {len(values)}")
    match values[0]:
        case SequenceValue(items=items):
            return StaticScalar(_ops.make_const(len(items)))
        case TensorValue(shape=shape):
            return StaticScalar(_ops.make_const(shape[0]))
        case _:
            reject(node.origin, f"len() requires an aggregate, not {_aggregate.a_kind(values[0])}")


def _range(interp: Interpreter, node: Call, frame: Frame, sink: Sink) -> SequenceValue:
    values = _operand_arguments(interp, node, "range", frame, sink)
    if not 1 <= len(values) <= 3:
        reject(node.origin, f"range() takes 1 to 3 arguments, got {len(values)}")
    bounds = [_aggregate.static_index(node.origin, value, "a range argument") for value in values]
    try:
        span = range(*bounds)
    except ValueError as error:
        reject(node.origin, f"range() rejects its arguments: {error}")
    # The length in exact arithmetic: len() of a huge range overflows Py_ssize_t, and the charge must
    # land as the located budget rejection before any materialization.
    length = max(0, (span.stop - span.start + span.step - (1 if span.step > 0 else -1)) // span.step)
    interp.budget.spend(max(length, 1), node.origin, "the range materialization")
    return SequenceValue(tuple(StaticScalar(_ops.make_const(i)) for i in span), Allocation())


def _rebuild_sequence(interp: Interpreter, node: Call, display: str, frame: Frame, sink: Sink) -> Value:
    values = _operand_arguments(interp, node, display, frame, sink)
    if len(values) != 1:
        reject(node.origin, f"{display}() takes exactly one aggregate argument here")
    source = values[0]
    if not isinstance(source, _AGGREGATE):
        reject(node.origin, f"{display}() requires an aggregate argument")
    children = _aggregate.splice_items(node.origin, source)
    interp.budget.spend(max(len(children), 1), node.origin, "the sequence conversion")
    return SequenceValue(tuple(children), Allocation())


def _to_tensor(
    interp: Interpreter, origin: Origin, display: str, value: Value, sink: Sink, *, copies: bool
) -> TensorValue:
    match value:
        case TensorValue():
            if copies:
                return TensorValue(value.shape, value.family, value.leaves, Allocation())
            # The non-copying conversion reuses the source allocation as a storage-equivalence token,
            # so the state-install disjointness checks see the derivation for free.
            share(value)
            return TensorValue(value.shape, value.family, value.leaves, value.allocation)
        case SequenceValue(items=items):
            if not items:
                reject(origin, f"{display}() of an empty sequence is not supported")
            shape, leaves = _tensor_rows(origin, display, items)
            for leaf in leaves:
                if not isinstance(leaf, Opaque) and leaf.stype is ScalarType.BOOL:
                    reject(origin, "an array must hold numbers, not booleans")
            if any(isinstance(leaf, Opaque) or leaf.stype is ScalarType.FLOAT for leaf in leaves):
                family = ScalarType.FLOAT
            else:
                family = ScalarType.INT
            conformed = [tensor_leaf(interp, origin, family, leaf, sink) for leaf in leaves]
            interp.budget.spend(len(conformed), origin, "the array conversion")
            return TensorValue(shape, family, tuple(conformed), Allocation())
        case _:
            reject(origin, f"{display}() requires a sequence or array argument, not {_aggregate.a_kind(value)}")


def _tensor_rows(
    origin: Origin, display: str, items: tuple[Value, ...]
) -> tuple[tuple[int, ...], list[Scalar | Opaque]]:
    """The scalar leaves are copied, so the source keeps sole ownership of itself."""
    if all(isinstance(item, (StaticScalar, ResidualScalar, Opaque)) for item in items):
        flat = [item for item in items if isinstance(item, (StaticScalar, ResidualScalar, Opaque))]
        return (len(items),), flat
    rows: list[list[Scalar | Opaque]] = []
    for item in items:
        match item:
            case SequenceValue(items=row_items):
                row: list[Scalar | Opaque] = []
                for element in row_items:
                    if not isinstance(element, (StaticScalar, ResidualScalar, Opaque)):
                        reject(origin, f"{display}() supports only 1-D and 2-D rectangular constructions")
                    row.append(element)
                rows.append(row)
            case TensorValue(shape=shape, leaves=leaves) if len(shape) == 1:
                rows.append(list(leaves))
            case _:
                reject(origin, f"{display}() supports only 1-D and 2-D rectangular constructions")
    widths = {len(row) for row in rows}
    if len(widths) != 1 or 0 in widths:
        reject(origin, f"{display}() requires rectangular rows of equal nonzero length")
    return (len(rows), widths.pop()), [leaf for row in rows for leaf in row]


def tensor_leaf(
    interp: Interpreter, origin: Origin, family: ScalarType, leaf: Scalar | Opaque, sink: Sink
) -> Scalar | Opaque:
    if isinstance(leaf, Opaque):
        if family is not ScalarType.FLOAT or not nan_payload(leaf.value):
            reject(origin, _describe_opaque(leaf))
        return leaf
    if leaf.stype is ScalarType.BOOL:
        reject(origin, "an array must hold numbers, not booleans")
    if family is ScalarType.FLOAT:
        return interp.as_float(leaf, origin, sink)
    if leaf.stype is ScalarType.FLOAT:
        reject(origin, "storing a float into an integer array truncates on the host; rebind a float array instead")
    return leaf


def bind_signature(
    interp: Interpreter,
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
        if isinstance(value, (StaticScalar, ResidualScalar, Opaque, SequenceValue, TensorValue, TensorMethod)):
            bindings[name] = value
        else:
            bindings[name] = interp.snapshot.admit(name, value, site)  # an injected default: a plain Python object
    return bindings


def _cast(interp: Interpreter, node: Call, target: type, frame: Frame, sink: Sink) -> Value:
    if len(node.args) != 1 or not isinstance(node.args[0], PosArg):
        reject(node.origin, f"{target.__name__}() takes exactly one positional argument here")
    operand = scalar(interp.expr(node.args[0].value, frame, sink), node.origin)
    declared = annotation_stype(target)
    assert declared is not None
    if operand.stype is declared:
        return operand
    return apply(interp, _ops.CONVERT[(operand.stype, declared)], [operand], node.origin, sink)
