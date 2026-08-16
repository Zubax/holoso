"""The expression side of the specializing interpreter: operators, attribute reads, and calls."""

import dataclasses
import inspect
import math
import types
from typing import TYPE_CHECKING

import numpy as np

from .._annotations import annotation_stype, host_type
from .._ir import *
from .._lib import Array, Conversion, Factory, Lifted, Operand, Reshape, ScalarFunction, resolve
from . import _aggregate, _ops
from ._ownership import share
from ._record import inadmissible_reason as record_inadmissible
from ._reject import reject
from ._snapshot import describe_opaque as _describe_opaque, nan_payload, tensor_of
from ._state import mro_attr
from ._values import (
    AGGREGATES,
    VALUE_KINDS,
    Allocation,
    BoundMethod,
    IteratorValue,
    Opaque,
    RangeValue,
    RecordValue,
    ResidualScalar,
    Scalar,
    SequenceValue,
    StaticScalar,
    TensorValue,
    Value,
)

if TYPE_CHECKING:
    from ._interpret import Frame, Interpreter, Sink

_LIST_ADVICE = "an array operation on a Python list/tuple is not supported; build one with np.array([...])"

# ------------------------------------------------------------------ attribute reads


def attr_read(interp: Interpreter, origin: Origin, base_value: Value, attr: str, frame: Frame, sink: Sink) -> Value:
    match base_value:
        case TensorValue():
            return _tensor_attr(interp, origin, base_value, attr, frame, sink)
        case RecordValue(cls=cls, fields=fields):
            names = [field.name for field in dataclasses.fields(cls)]
            if attr not in names:
                if callable(getattr(cls, attr, None)):
                    reject(origin, f"calling methods on a record value is not supported; {attr!r} is not a field")
                reject(origin, f"the record {cls.__name__} has no field {attr!r}")
            item = fields[names.index(attr)]
            if isinstance(item, AGGREGATES):
                share(base_value)
                share(item)
            return item
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
            found = resolve(getattr(host_type(base_value.stype), attr, None))
            if isinstance(found, ScalarFunction):
                return BoundMethod(base_value, attr)
            reject(origin, f"a scalar has no supported attribute {attr!r}")
        case BoundMethod():
            reject(origin, f"{_aggregate.a_kind(base_value)} can only be called")
        case IteratorValue() | RangeValue():
            reject(origin, f"{_aggregate.a_kind(base_value)} has no supported attribute {attr!r}")
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
    if isinstance(found, Array) and inspect.isdatadescriptor(descriptor):
        return _array_call(interp, origin, f".{attr}", found, [tensor], frame, sink)
    if isinstance(found, (Array, Reshape)):
        share(tensor)
        return BoundMethod(tensor, attr)
    reject(origin, f"an array has no supported attribute {attr!r}")


# ------------------------------------------------------------------ scalars and operators


def scalar(value: Value, origin: Origin) -> Scalar:
    match value:
        case StaticScalar() | ResidualScalar():
            return value
        case Opaque():
            reject(origin, _describe_opaque(value))
        case SequenceValue() | TensorValue() | RecordValue() | IteratorValue() | RangeValue():
            reject(origin, f"{_aggregate.a_kind(value)} cannot be used as a scalar here")
        case BoundMethod():
            reject(origin, f"{_aggregate.a_kind(value)} can only be called")


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
            pass  # the graph re-derives the fault and the refusal gate judges it; never convict here
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
    if isinstance(value, AGGREGATES):
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
            if isinstance(operand, AGGREGATES):
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
    if isinstance(lv, AGGREGATES) or isinstance(rv, AGGREGATES):
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
    if isinstance(callee, BoundMethod):
        return _bound_method(interp, node, callee, frame, sink)
    if not isinstance(callee, Opaque):
        reject(node.origin, "the callee is not a callable object")
    raw = callee.value
    match resolve(raw) if callable(raw) else None:
        case ScalarFunction() as match:
            values = _operand_arguments(interp, node, callee.name, frame, sink)
            return _scalar_call(interp, node.origin, callee.name, match, values, frame, sink)
        case Lifted(scalar=lifted):
            values = _operand_arguments(interp, node, callee.name, frame, sink)
            if len(values) == 1 and isinstance(values[0], TensorValue):
                return _lifted_call(interp, node.origin, callee.name, lifted, values[0], frame, sink)
            if any(isinstance(value, (SequenceValue, RangeValue)) for value in values):
                reject(
                    node.origin,
                    _LIST_ADVICE,
                )
            return _scalar_call(interp, node.origin, callee.name, lifted, values, frame, sink)
        case Array() as match:
            values = _positional_arguments(interp, node, callee.name, frame, sink)
            _demand_descriptor_receiver(node.origin, callee.name, raw, values)
            return _array_call(interp, node.origin, callee.name, match, values, frame, sink)
        case Factory() as match:
            return _factory(interp, node, callee.name, match, frame, sink)
        case Conversion(copies=copies):
            source, family = _conversion_arguments(interp, node, callee.name, frame, sink)
            return _to_tensor(interp, node.origin, callee.name, source, sink, copies=copies, family=family)
        case Reshape():
            if getattr(raw, "__objclass__", None) is np.ndarray:
                values = _operand_arguments(interp, node, callee.name, frame, sink)
                _demand_descriptor_receiver(node.origin, callee.name, raw, values)
                return _reshape(node.origin, callee.name, values[0], values[1:])
            base, shape = _option_arguments(interp, node, callee.name, frame, sink, option="shape")
            if shape is None:
                reject(node.origin, f"{callee.name}() takes an array and a shape (an int or a tuple of ints)")
            return _reshape(node.origin, callee.name, base, [shape])
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
    if raw is enumerate:
        return _enumerate(interp, node, frame, sink)
    if raw is list or raw is tuple:
        return _rebuild_sequence(interp, node, callee.name, frame, sink)
    if isinstance(raw, type) and dataclasses.is_dataclass(raw):
        return _construct_record(interp, node, callee.name, raw, frame, sink)
    resolved = _inlinable(interp, node.origin, callee.name, raw)
    if resolved is not None:
        fn, leading = resolved
        positional, keywords = _signature_arguments(interp, node, frame, sink)
        return interp.inline(
            node.origin, callee.name, fn, [*leading, *positional], keywords, frame, sink, positional_only=False
        )
    if callable(raw):
        reject(node.origin, f"calls to {callee.name!r} are not supported yet")
    reject(node.origin, f"the captured object {callee.name!r} is not callable")


def _inlinable(
    interp: Interpreter, origin: Origin, display: str, raw: object
) -> tuple[types.FunctionType, list[Value]] | None:
    """
    The plain-Python callee behind any calling spelling, with the receiver arguments CPython's descriptor
    protocol would prepend; an admitted receiver's frozen attributes fold and its state reads see the current
    slots. A C-level callable (a numpy dispatcher or ufunc, a partial) resolves to None: a callee the
    registry does not carry.
    """
    if isinstance(raw, types.FunctionType):
        return raw, []
    if inspect.ismethod(raw):
        if not isinstance(raw.__func__, types.FunctionType):
            return None
        return raw.__func__, [interp.snapshot.admit(display, raw.__self__, origin)]
    if callable(raw):
        call_attr = mro_attr(type(raw), "__call__")
        if isinstance(call_attr, staticmethod) and isinstance(call_attr.__func__, types.FunctionType):
            return call_attr.__func__, []
        if isinstance(call_attr, types.FunctionType):
            return call_attr, [interp.snapshot.admit(display, raw, origin)]
    return None


def _argument(interp: Interpreter, atom: Atom, frame: Frame, sink: Sink) -> Value:
    value = interp.expr(atom, frame, sink)
    if isinstance(value, AGGREGATES) and (isinstance(atom, LocalRef) or interp.alias_conduit(frame, atom)):
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
                spliced = _aggregate.decay(interp.budget, interp.expr(value, frame, sink), node.origin)
                values.extend(_aggregate.splice_items(node.origin, spliced, interp.loop_passes()))
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
                spliced = _aggregate.decay(interp.budget, interp.expr(value, frame, sink), node.origin)
                values.extend(_aggregate.splice_items(node.origin, spliced, interp.loop_passes()))
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
                spliced = _aggregate.decay(interp.budget, interp.expr(value, frame, sink), node.origin)
                positional.extend(_aggregate.splice_items(node.origin, spliced, interp.loop_passes()))
            case KwArg(name=name, value=value):
                keywords[name] = _argument(interp, value, frame, sink)
    return positional, keywords


def _demand_descriptor_receiver(origin: Origin, display: str, raw: object, values: list[Value]) -> None:
    """
    CPython rejects a non-ndarray receiver on an unbound ndarray method; a spelling that could never run
    as its own reference must refuse rather than quietly apply the stub.
    """
    if getattr(raw, "__objclass__", None) is not np.ndarray:
        return
    if not values or not isinstance(values[0], TensorValue):
        reject(origin, f"{display} is an unbound ndarray method, so its first argument must be an array")


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


def _lifted_call(
    interp: Interpreter,
    origin: Origin,
    display: str,
    match: ScalarFunction,
    tensor: TensorValue,
    frame: Frame,
    sink: Sink,
) -> TensorValue:
    leaves: list[Scalar] = []
    for leaf in tensor.leaves:
        result = _scalar_call(interp, origin, display, match, [_scalar_leaf(origin, leaf)], frame, sink)
        leaves.append(scalar(result, origin))
    interp.budget.spend(len(leaves), origin, "the elementwise operation")
    return TensorValue(tensor.shape, leaves[0].stype, tuple(leaves), Allocation())


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
    if any(
        isinstance(value, RangeValue) or (isinstance(value, SequenceValue) and position not in match.sequences)
        for position, value in enumerate(values)
    ):
        # Before the stub, so the rejection reads the same however the operation was spelled.
        reject(origin, _LIST_ADVICE)
    result = interp.inline(origin, display, match.stub, values, {}, frame, sink, positional_only=True)
    if match.derives and isinstance(values[0], TensorValue) and isinstance(result, TensorValue):
        share(values[0])
        result = dataclasses.replace(result, allocation=values[0].allocation)
    return result


def _bound_method(interp: Interpreter, node: Call, method: BoundMethod, frame: Frame, sink: Sink) -> Value:
    receiver = method.receiver
    display = f".{method.name}"
    if isinstance(receiver, TensorValue):
        found = resolve(getattr(np.ndarray, method.name))
        if isinstance(found, Reshape):
            values = _operand_arguments(interp, node, display, frame, sink)
            return _reshape(node.origin, display, receiver, values)
        assert isinstance(found, Array), "a minted method stays resolvable"
        values = _positional_arguments(interp, node, display, frame, sink)
        high = found.stub.__code__.co_argcount - 1
        low = high - len(found.stub.__defaults__ or ())
        if not low <= len(values) <= high:
            expected = str(low) if low == high else f"{low} to {high}"
            reject(node.origin, f"{display}() takes {expected} argument(s), got {len(values)}")
        return _array_call(interp, node.origin, display, found, [method.receiver, *values], frame, sink)
    scalar_found = resolve(getattr(host_type(receiver.stype), method.name))
    assert isinstance(scalar_found, ScalarFunction), "a minted method stays resolvable"
    values = _operand_arguments(interp, node, display, frame, sink)
    arity = scalar_found.arity - 1
    if len(values) != arity:
        reject(node.origin, f"{display}() takes {arity} argument(s), got {len(values)}")
    return _scalar_call(interp, node.origin, display, scalar_found, [receiver, *values], frame, sink)


def _option_arguments(
    interp: Interpreter, node: Call, display: str, frame: Frame, sink: Sink, *, option: str
) -> tuple[Value, Value | None]:
    """
    The non-aliasing one-option binder: exactly one positional subject, plus `option` given as the second
    positional or as its keyword -- the shape numpy's own conversion/reshape signatures share.
    """
    positional: list[Value] = []
    named: Value | None = None
    for arg in node.args:
        match arg:
            case PosArg(value=value):
                positional.append(interp.expr(value, frame, sink))
            case StarArg(value=value):
                spliced = _aggregate.decay(interp.budget, interp.expr(value, frame, sink), node.origin)
                positional.extend(_aggregate.splice_items(node.origin, spliced, interp.loop_passes()))
            case KwArg(name=name, value=value) if name == option:
                named = interp.expr(value, frame, sink)
            case KwArg(name=name):
                reject(node.origin, f"{display}() takes no keyword argument {name!r} (only {option})")
    if len(positional) == 2 and named is None:
        named = positional.pop()
    if len(positional) != 1:
        reject(node.origin, f"{display}() takes one positional argument plus an optional {option}")
    return positional[0], named


def _reshape(origin: Origin, display: str, base: Value, dim_values: list[Value]) -> TensorValue:
    """
    C-order reshape preserves the flat row-major leaf sequence: the same leaves over new dims, a derivation
    sharing the source allocation since the host MAY answer a view.
    """
    if isinstance(base, SequenceValue):
        reject(origin, _LIST_ADVICE)
    if not isinstance(base, TensorValue):
        reject(origin, f"{display}() requires an array, not {_aggregate.a_kind(base)}")
    if len(dim_values) == 1 and isinstance(dim_values[0], SequenceValue):
        dim_values = list(dim_values[0].items)
    if not dim_values:
        reject(origin, f"{display}() requires a shape (an int or a tuple of ints)")
    dims = [_aggregate.static_index(origin, value, "a reshape dimension") for value in dim_values]
    if any(dim < 0 for dim in dims):
        reject(origin, f"{display}() does not support dimension inference (-1); spell the dimension explicitly")
    if len(dims) > 2 or 0 in dims:
        reject(origin, f"{display}() supports only non-empty 1-D and 2-D shapes, got {tuple(dims)}")
    if math.prod(dims) != len(base.leaves):
        reject(origin, f"cannot reshape an array of size {len(base.leaves)} into shape {tuple(dims)}")
    share(base)
    return TensorValue(tuple(dims), base.family, base.leaves, base.allocation)


def _construct_record(interp: Interpreter, node: Call, display: str, cls: type, frame: Frame, sink: Sink) -> Value:
    """
    Structural construction: the generated __init__ never runs; arguments bind by its signature and conform
    strictly to the field annotations.
    """
    reason = record_inadmissible(cls)
    if reason is not None:
        reject(node.origin, reason)
    positional, keywords = _signature_arguments(interp, node, frame, sink)
    try:
        bound = inspect.signature(cls).bind(*positional, **keywords)
    except TypeError as error:
        reject(node.origin, f"the arguments do not bind: {error}")
    bound.apply_defaults()
    annotations = interp.record_annotations(cls, node.origin)
    fields: list[Value] = []
    for field in dataclasses.fields(cls):
        value = bound.arguments[field.name]
        if not isinstance(value, VALUE_KINDS):
            value = interp.snapshot.admit(f"{display}.{field.name}", value, node.origin)
        fields.append(
            interp.conform_annotation(
                value, annotations[field.name], node.origin, sink, f"the field {field.name!r} of {display}()"
            )
        )
    return RecordValue(cls, tuple(fields), Allocation())


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
        case RangeValue() as found:
            span = _aggregate.static_range(found)
            if span is None:
                reject(node.origin, "len() of a range with a runtime bound is not supported")
            try:
                return StaticScalar(_ops.make_const(len(span)))
            except OverflowError:
                reject(node.origin, "len() of this range overflows, exactly as it does in CPython")
        case _:
            reject(node.origin, f"len() requires an aggregate, not {_aggregate.a_kind(values[0])}")


def _range(interp: Interpreter, node: Call, frame: Frame, sink: Sink) -> RangeValue:
    values = _operand_arguments(interp, node, "range", frame, sink)
    if not 1 <= len(values) <= 3:
        reject(node.origin, f"range() takes 1 to 3 arguments, got {len(values)}")
    bounds: list[Scalar] = []
    for value in values:
        match value:
            case StaticScalar() | ResidualScalar() if value.stype is ScalarType.INT:
                bounds.append(value)
            case StaticScalar() | ResidualScalar():
                reject(node.origin, f"a range argument must be an int, not a {value.stype.value}")
            case _:
                reject(node.origin, f"a range argument must be an int, not {_aggregate.a_kind(value)}")
    start = bounds[0] if len(bounds) >= 2 else StaticScalar(_ops.make_const(0))
    stop = bounds[1] if len(bounds) >= 2 else bounds[0]
    match bounds[2] if len(bounds) == 3 else StaticScalar(_ops.make_const(1)):
        case StaticScalar(const=const):
            step = _ops.const_value(const)
            assert isinstance(step, int)
        case ResidualScalar():
            reject(
                node.origin,
                "the range step must be a compile-time constant int: its sign selects the loop "
                "direction, and a zero step is a ValueError at range() construction",
            )
    if step == 0:
        reject(node.origin, "range() rejects its arguments: range() arg 3 must not be zero")
    return RangeValue(start, stop, step)


def _enumerate(interp: Interpreter, node: Call, frame: Frame, sink: Sink) -> IteratorValue:
    """
    The pairs are an eager snapshot where CPython's iterator is lazy, so the source is shared (a mid-iteration
    store would read stale leaves -- the conservative refusal mirrors the borrow on a directly iterated
    aggregate); the iterator kind itself carries the exhausted-after-one-pass semantics.
    """
    source, start = _option_arguments(interp, node, "enumerate", frame, sink, option="start")
    begin = 0 if start is None else _aggregate.static_index(node.origin, start, "the enumerate start")
    decayed = _aggregate.decay(interp.budget, source, node.origin)
    items = _aggregate.splice_items(node.origin, decayed, interp.loop_passes())
    share(decayed)
    pairs = tuple(
        SequenceValue((StaticScalar(_ops.make_const(begin + position)), item), Allocation())
        for position, item in enumerate(items)
    )
    interp.budget.spend(max(len(pairs), 1), node.origin, "the enumerate expansion")
    return IteratorValue(pairs, tuple(interp.loop_passes()))


def _rebuild_sequence(interp: Interpreter, node: Call, display: str, frame: Frame, sink: Sink) -> Value:
    values = _operand_arguments(interp, node, display, frame, sink)
    if len(values) != 1:
        reject(node.origin, f"{display}() takes exactly one aggregate argument here")
    source = _aggregate.decay(interp.budget, values[0], node.origin)
    if not isinstance(source, (*AGGREGATES, IteratorValue)):
        reject(node.origin, f"{display}() requires an aggregate argument")
    children = _aggregate.splice_items(node.origin, source, interp.loop_passes())
    interp.budget.spend(max(len(children), 1), node.origin, "the sequence conversion")
    return SequenceValue(tuple(children), Allocation())


def _conversion_arguments(
    interp: Interpreter, node: Call, display: str, frame: Frame, sink: Sink
) -> tuple[Value, ScalarType | None]:
    source, dtype = _option_arguments(interp, node, display, frame, sink, option="dtype")
    if dtype is None:
        return source, None
    family = annotation_stype(dtype.value) if isinstance(dtype, Opaque) else None
    if family is ScalarType.BOOL:
        reject(node.origin, "an array must hold numbers, not booleans")
    if family is None:
        reject(node.origin, f"the dtype of {display}() must be the Python type float or int")
    return source, family


def _to_tensor(
    interp: Interpreter,
    origin: Origin,
    display: str,
    value: Value,
    sink: Sink,
    *,
    copies: bool,
    family: ScalarType | None = None,
) -> TensorValue:
    value = _aggregate.decay(interp.budget, value, origin)
    match value:
        case TensorValue():
            if family is not None and family is not value.family:
                # A family change copies on the host even under asarray, hence the fresh allocation.
                converted = tuple(
                    tensor_leaf(interp, origin, family, leaf, sink, explicit=True) for leaf in value.leaves
                )
                interp.budget.spend(len(converted), origin, "the array conversion")
                return TensorValue(value.shape, family, converted, Allocation())
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
            explicit = family is not None
            if family is None:
                floaty = any(isinstance(leaf, Opaque) or leaf.stype is ScalarType.FLOAT for leaf in leaves)
                family = ScalarType.FLOAT if floaty else ScalarType.INT
            conformed = [tensor_leaf(interp, origin, family, leaf, sink, explicit=explicit) for leaf in leaves]
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
    interp: Interpreter,
    origin: Origin,
    family: ScalarType,
    leaf: Scalar | Opaque,
    sink: Sink,
    *,
    explicit: bool = False,
) -> Scalar | Opaque:
    """
    `explicit` marks a spelled dtype request, which converts (bool widens, float truncates, as numpy defines)
    where the implicit paths refuse.
    """
    if isinstance(leaf, Opaque):
        if family is not ScalarType.FLOAT or not nan_payload(leaf.value):
            reject(origin, _describe_opaque(leaf))
        return leaf
    if leaf.stype is family:
        return leaf
    if leaf.stype is ScalarType.BOOL:
        if not explicit:
            reject(origin, "an array must hold numbers, not booleans")
        return apply(interp, _ops.CONVERT[(ScalarType.BOOL, family)], [leaf], origin, sink)
    if family is ScalarType.FLOAT:
        return interp.as_float(leaf, origin, sink)
    assert leaf.stype is ScalarType.FLOAT and family is ScalarType.INT
    if not explicit:
        reject(origin, "storing a float into an integer array truncates on the host; rebind a float array instead")
    return apply(interp, _ops.CONVERT[(ScalarType.FLOAT, ScalarType.INT)], [leaf], origin, sink)


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
        if isinstance(value, VALUE_KINDS):
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
