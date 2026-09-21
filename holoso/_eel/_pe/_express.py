"""The expression side of the specializing interpreter: operators, attribute reads, and calls."""

import dataclasses
import inspect
import math
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, NoReturn

import numpy as np

from .._annotations import annotation_stype, host_type
from .._ir import *
from .._decorator import plain_function
from .._lib import (
    Array,
    Conversion,
    Factory,
    Operand,
    Reshape,
    ScalarLowering,
    Spelling,
    VariadicFunction,
    resolve,
)
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


_ABSENT = object()

# What a scalar-only spelling can only refuse, a range among them though it is no aggregate value.
_AGGREGATE_OPERANDS = AGGREGATES + (RangeValue,)


def _demand_an_array(origin: Origin, value: Value) -> NoReturn:
    # np.array() takes a range as readily as a sequence, so one piece of advice covers both.
    reject(origin, f"{_aggregate.a_kind(value)} is not an array operand; build one with np.array(...)")


def _admit_operands(
    origin: Origin, display: str, values: list[Value], *, arrays: bool, sequences: frozenset[int] = frozenset()
) -> None:
    """The one gate on aggregate operands: an array where the callee takes arrays, a sequence where it names one."""
    for position, value in enumerate(values):
        if not isinstance(value, _AGGREGATE_OPERANDS) or (arrays and isinstance(value, TensorValue)):
            continue
        if isinstance(value, SequenceValue) and position in sequences:
            continue
        if arrays and isinstance(value, (SequenceValue, RangeValue)):
            _demand_an_array(origin, value)
        reject(origin, f"{display} does not support {_aggregate.kind_label(value)} operands")


# ATTRIBUTE READS


def attr_read(interp: Interpreter, origin: Origin, base_value: Value, attr: str, frame: Frame, sink: Sink) -> Value:
    match base_value:
        case TensorValue():
            return _tensor_attr(interp, origin, base_value, attr, frame, sink)
        case RecordValue(cls=cls, fields=fields):
            names = [field.name for field in dataclasses.fields(cls)]
            if attr not in names:
                found = mro_attr(cls, attr, _ABSENT)
                if isinstance(found, property) and (fget := plain_function(found.fget)) is not None:
                    return interp.inline(
                        origin, f"{cls.__name__}.{attr}", fget, [base_value], {}, frame, sink, stub=False
                    )
                if callable(getattr(cls, attr, None)):
                    reject(origin, f"calling methods on a record value is not supported; {attr!r} is not a field")
                if found is not _ABSENT and not hasattr(type(found), "__get__"):
                    return interp.snapshot.admit(f"{cls.__name__}.{attr}", found, origin)  # a class constant
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
            if isinstance(found, Spelling):
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


# SCALARS AND OPERATORS


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


# CALLS


def call(interp: Interpreter, node: Call, frame: Frame, sink: Sink) -> Value:
    callee = interp.expr(node.callee, frame, sink)
    if isinstance(callee, BoundMethod):
        return _bound_method(interp, node, callee, frame, sink)
    if not isinstance(callee, Opaque):
        reject(node.origin, "the callee is not a callable object")
    raw = callee.value
    display = f"{callee.name}()"
    match resolve(raw) if callable(raw) else None:
        case Spelling() as match:
            values = _positional_arguments(interp, node, display, frame, sink, shares=False)
            return _spelling_call(interp, node.origin, display, match, values, frame, sink)
        case VariadicFunction() as match:
            values = _positional_arguments(interp, node, display, frame, sink, shares=False)
            return _variadic_call(interp, node.origin, display, match, values, sink)
        case Array() as match:
            values = _positional_arguments(interp, node, display, frame, sink, shares=True)
            _demand_descriptor_receiver(node.origin, display, raw, values)
            return _array_call(interp, node.origin, display, match, values, frame, sink)
        case Factory() as match:
            return _factory(interp, node, display, match, frame, sink)
        case Conversion(copies=copies):
            source, family = _conversion_arguments(interp, node, display, frame, sink)
            return _to_tensor(interp, node.origin, display, source, sink, copies=copies, family=family)
        case Reshape():
            if getattr(raw, "__objclass__", None) is np.ndarray:
                values = _positional_arguments(interp, node, display, frame, sink, shares=False)
                _demand_descriptor_receiver(node.origin, display, raw, values)
                return _reshape(node.origin, display, values[0], values[1:])
            base, shape = _option_arguments(interp, node, display, frame, sink, option="shape")
            if shape is None:
                reject(node.origin, f"{display} takes an array and a shape (an int or a tuple of ints)")
            return _reshape(node.origin, display, base, [shape])
        case None:
            pass
    if raw is len:
        return _len(interp, node, frame, sink)
    if raw is range:
        return _range(interp, node, frame, sink)
    if raw is enumerate:
        return _enumerate(interp, node, frame, sink)
    if raw is list or raw is tuple:
        return _rebuild_sequence(interp, node, display, frame, sink)
    if isinstance(raw, type) and dataclasses.is_dataclass(raw):
        return _construct_record(interp, node, display, raw, frame, sink)
    resolved = _inlinable(interp, node.origin, callee.name, raw)
    if resolved is not None:
        fn, leading = resolved
        positional, keywords = _arguments(interp, node, frame, sink, shares=True)
        return interp.inline(node.origin, display, fn, [*leading, *positional], keywords, frame, sink, stub=False)
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
    registry does not carry, save for a wrapper the compiler reads through.
    """
    if (fn := plain_function(raw)) is not None:
        return fn, []
    if inspect.ismethod(raw):
        fn = plain_function(raw.__func__)
        return None if fn is None else (fn, [interp.snapshot.admit(display, raw.__self__, origin)])
    if callable(raw):
        call_attr = mro_attr(type(raw), "__call__")
        if isinstance(call_attr, staticmethod) and (fn := plain_function(call_attr.__func__)) is not None:
            return fn, []
        if (fn := plain_function(call_attr)) is not None:
            return fn, [interp.snapshot.admit(display, raw, origin)]
    return None


def _arguments(
    interp: Interpreter, node: Call, frame: Frame, sink: Sink, *, shares: bool
) -> tuple[list[Value], dict[str, Value]]:
    """
    The call's arguments as written. `shares` tells a callee that may retain a handle, where binding a named
    aggregate is an aliasing event, from one that only reads its operands.
    """

    def read(atom: Atom) -> Value:
        value = interp.expr(atom, frame, sink)
        if (
            shares
            and isinstance(value, AGGREGATES)
            and (isinstance(atom, LocalRef) or interp.alias_conduit(frame, atom))
        ):
            share(value)
        return value

    positional: list[Value] = []
    keywords: dict[str, Value] = {}
    for arg in node.args:
        match arg:
            case PosArg(value=value):
                positional.append(read(value))
            case StarArg(value=value):
                spliced = _aggregate.decay(interp.budget, interp.expr(value, frame, sink), node.origin)
                positional.extend(_aggregate.splice_items(node.origin, spliced, interp.loop_passes()))
            case KwArg(name=name, value=value):
                keywords[name] = read(value)
    return positional, keywords


def _positional_arguments(
    interp: Interpreter, node: Call, display: str, frame: Frame, sink: Sink, *, shares: bool
) -> list[Value]:
    if any(isinstance(arg, KwArg) for arg in node.args):
        reject(node.origin, f"{display} takes no keyword arguments")
    return _arguments(interp, node, frame, sink, shares=shares)[0]


def _demand_descriptor_receiver(origin: Origin, display: str, raw: object, values: list[Value]) -> None:
    """
    CPython rejects a non-ndarray receiver on an unbound ndarray method; a spelling that could never run
    as its own reference must refuse rather than quietly apply the stub.
    """
    if getattr(raw, "__objclass__", None) is not np.ndarray:
        return
    if not values or not isinstance(values[0], TensorValue):
        reject(origin, f"{display} is an unbound ndarray method, so its first argument must be an array")


def maps_over_arrays(op: BinaryOp) -> bool:
    found = resolve(op)
    return isinstance(found, Spelling) and found.elementwise


def operator_call(
    interp: Interpreter,
    origin: Origin,
    op: BinaryOp | CompareOp | UnaryOp,
    values: list[Value],
    frame: Frame,
    sink: Sink,
) -> Value:
    """An operator is a spelling like any other, so it and the calls that spell it cannot drift apart."""
    found = resolve(op)
    assert isinstance(found, (Spelling, Array)), op
    display = f"`{op.value}`"
    if isinstance(found, Array):
        # A spelled call shares a named aggregate it binds, since a composite may derive its result from one. An
        # operator has no argument atom to judge, so every aggregate is shared: conservative, never unsound.
        for value in values:
            if isinstance(value, AGGREGATES):
                share(value)
        return _array_call(interp, origin, display, found, values, frame, sink)
    return _spelling_call(interp, origin, display, found, values, frame, sink)


def _spelling_call(
    interp: Interpreter,
    origin: Origin,
    display: str,
    match: Spelling,
    values: list[Value],
    frame: Frame,
    sink: Sink,
) -> Value:
    """One path for an operator and for each of its spellings, over scalars and, elementwise, over arrays."""
    meaning = match.meaning
    if len(values) != meaning.arity:
        reject(origin, f"{display} takes {meaning.arity} argument(s), got {len(values)}")
    _admit_operands(origin, display, values, arrays=match.elementwise)
    operands = [value if isinstance(value, TensorValue) else scalar(value, origin) for value in values]
    tensors = [operand for operand in operands if isinstance(operand, TensorValue)]
    assert match.elementwise or not tensors
    # One selection serves every leaf, so an operand whose value selects the lowering must be the same for all of them.
    for i, operand in enumerate(operands):
        if isinstance(operand, TensorValue) and meaning.selects_by_value(i):
            reject(origin, f"argument {i + 1} of {display} must be a scalar")
    # Selected before the shapes are compared, so an unsupported operation is named even where the shapes also differ.
    chosen = meaning.select([Operand(o.family) if isinstance(o, TensorValue) else _operand(o) for o in operands])
    if chosen is None:
        given = [_Given(o.family, True) if isinstance(o, TensorValue) else _Given(o.stype) for o in operands]
        _refuse_domains(origin, display, given, meaning.signatures)
    if not tensors:
        scalars = [operand for operand in operands if not isinstance(operand, TensorValue)]
        return _lowering_call(interp, origin, display, chosen, scalars, frame, sink)
    shape = tensors[0].shape
    for tensor in tensors[1:]:
        if tensor.shape != shape:
            reject(origin, f"array shapes {shape} and {tensor.shape} do not match; broadcasting is not supported")
    assert all(len(tensor.leaves) == len(tensors[0].leaves) for tensor in tensors)
    # Charged before the leaves are built, where a composite lowering would otherwise inline once per leaf first.
    interp.budget.spend(len(tensors[0].leaves), origin, "the elementwise operation")
    leaves: list[Scalar] = []
    for leaf in range(len(tensors[0].leaves)):
        row = [scalar(o.leaves[leaf], origin) if isinstance(o, TensorValue) else o for o in operands]
        leaves.append(scalar(_lowering_call(interp, origin, display, chosen, row, frame, sink), origin))
    assert all(leaf.stype is leaves[0].stype for leaf in leaves)
    return TensorValue(shape, leaves[0].stype, tuple(leaves), Allocation())


def _lowering_call(
    interp: Interpreter,
    origin: Origin,
    display: str,
    chosen: ScalarLowering,
    operands: list[Scalar],
    frame: Frame,
    sink: Sink,
) -> Value:
    # Both kinds of lowering, so inlining judges base types alone and never learns what a refinement is.
    conformed = [
        interp.conform(operand, declared.stype, origin, sink, f"argument {i + 1} of {display}")
        for i, (operand, declared) in enumerate(zip(operands, chosen.operands, strict=True))
    ]
    if chosen.operator is None:
        arguments: list[Value] = list(conformed)
        return interp.inline(origin, display, chosen.stub, arguments, {}, frame, sink, stub=True)
    return apply(interp, chosen.operator, conformed, origin, sink)


@dataclasses.dataclass(frozen=True, slots=True)
class _Given:
    """An operand as a refusal names it: `array` tells a leaf family from a scalar operand of the same family."""

    stype: ScalarType
    array: bool = False


def _refuse_domains(
    origin: Origin, display: str, operands: list[_Given], signatures: list[list[ScalarType]]
) -> NoReturn:
    served = " or ".join("(" + ", ".join(stype.value for stype in one) + ")" for one in signatures)
    got = ", ".join(given.stype.value + (" array" if given.array else "") for given in operands)
    stypes = [given.stype for given in operands]
    accepted = {stype for one in signatures for stype in one}
    boolean = ScalarType.BOOL in stypes
    number = any(stype is not ScalarType.BOOL for stype in stypes)
    serves_boolean = ScalarType.BOOL in accepted
    serves_number = accepted != {ScalarType.BOOL}
    casts = " or ".join(f"{stype.value}(...)" for stype in (ScalarType.INT, ScalarType.FLOAT) if stype in accepted)
    if boolean and not serves_boolean:
        note = f"; a boolean is not a number, cast explicitly with {casts}"
    elif number and not serves_number:
        note = "; Python truthiness is not supported, compare explicitly (e.g. x != 0)"
    elif boolean and number:
        note = f"; a boolean and a number cannot mix, cast explicitly with {casts}"
    else:
        note = ""
    reject(origin, f"{display} takes {served}, got ({got}){note}")


def _variadic_call(
    interp: Interpreter,
    origin: Origin,
    display: str,
    match: VariadicFunction,
    values: list[Value],
    sink: Sink,
) -> Value:
    if len(values) < match.minimum:
        reject(origin, f"{display} takes at least {match.minimum} argument(s), got {len(values)}")
    _admit_operands(origin, display, values, arrays=False)
    operands = [scalar(value, origin) for value in values]
    if not all(match.domain.accepts(_operand(operand)) for operand in operands):
        _refuse_domains(origin, display, [_Given(operand.stype) for operand in operands], [[match.domain.stype]])
    conformed = [
        interp.conform(operand, match.domain.stype, origin, sink, f"argument {i + 1} of {display}")
        for i, operand in enumerate(operands)
    ]
    return apply(interp, match.operator(len(conformed)), conformed, origin, sink)


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
    _admit_operands(origin, display, values, arrays=True, sequences=match.sequences)
    result = interp.inline(origin, display, match.stub, values, {}, frame, sink, stub=True)
    if match.derives and isinstance(values[0], TensorValue) and isinstance(result, TensorValue):
        share(values[0])
        result = dataclasses.replace(result, allocation=values[0].allocation)
    return result


def _bound_method(interp: Interpreter, node: Call, method: BoundMethod, frame: Frame, sink: Sink) -> Value:
    receiver = method.receiver
    display = f".{method.name}()"
    if isinstance(receiver, TensorValue):
        found = resolve(getattr(np.ndarray, method.name))
        if isinstance(found, Reshape):
            values = _positional_arguments(interp, node, display, frame, sink, shares=False)
            return _reshape(node.origin, display, receiver, values)
        assert isinstance(found, Array), "a minted method stays resolvable"
        values = _positional_arguments(interp, node, display, frame, sink, shares=True)
        return _array_call(interp, node.origin, display, found, [method.receiver, *values], frame, sink)
    scalar_found = resolve(getattr(host_type(receiver.stype), method.name))
    assert isinstance(scalar_found, Spelling), "a minted method stays resolvable"
    values = _positional_arguments(interp, node, display, frame, sink, shares=False)
    arity = scalar_found.meaning.arity - 1
    if len(values) != arity:
        reject(node.origin, f"{display} takes {arity} argument(s), got {len(values)}")
    return _spelling_call(interp, node.origin, display, scalar_found, [receiver, *values], frame, sink)


def _option_arguments(
    interp: Interpreter, node: Call, display: str, frame: Frame, sink: Sink, *, option: str
) -> tuple[Value, Value | None]:
    """
    The non-aliasing one-option binder: exactly one positional subject, plus `option` given as the second
    positional or as its keyword -- the shape numpy's own conversion/reshape signatures share.
    """
    for arg in node.args:
        if isinstance(arg, KwArg) and arg.name != option:
            reject(node.origin, f"{display} takes no keyword argument {arg.name!r} (only {option})")
    positional, keywords = _arguments(interp, node, frame, sink, shares=False)
    named = keywords.get(option)
    if len(positional) == 2 and named is None:
        named = positional.pop()
    if len(positional) != 1:
        reject(node.origin, f"{display} takes one positional argument plus an optional {option}")
    return positional[0], named


def _reshape(origin: Origin, display: str, base: Value, dim_values: list[Value]) -> TensorValue:
    """
    C-order reshape preserves the flat row-major leaf sequence: the same leaves over new dims, a derivation
    sharing the source allocation since the host MAY answer a view.
    """
    if isinstance(base, SequenceValue):
        _demand_an_array(origin, base)
    if not isinstance(base, TensorValue):
        reject(origin, f"{display} requires an array, not {_aggregate.a_kind(base)}")
    if len(dim_values) == 1 and isinstance(dim_values[0], SequenceValue):
        dim_values = list(dim_values[0].items)
    if not dim_values:
        reject(origin, f"{display} requires a shape (an int or a tuple of ints)")
    dims = [_aggregate.static_index(origin, value, "a reshape dimension") for value in dim_values]
    if any(dim < -1 for dim in dims) or dims.count(-1) > 1:
        reject(origin, f"{display} infers at most one dimension, spelled -1, got {tuple(dims)}")
    if len(dims) > 2 or 0 in dims:
        reject(origin, f"{display} supports only non-empty 1-D and 2-D shapes, got {tuple(dims)}")
    if -1 in dims:
        known = math.prod(dim for dim in dims if dim != -1)
        assert known > 0
        if len(base.leaves) % known == 0:
            dims[dims.index(-1)] = len(base.leaves) // known
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
    positional, keywords = _arguments(interp, node, frame, sink, shares=True)
    bindings = bind_signature(interp, node.origin, cls, positional, keywords, prefix=f"{cls.__name__}.")
    annotations = interp.record_annotations(cls, node.origin)
    fields: list[Value] = [
        interp.conform_annotation(
            bindings[field.name], annotations[field.name], node.origin, sink, f"the field {field.name!r} of {display}"
        )
        for field in dataclasses.fields(cls)
    ]
    return RecordValue(cls, tuple(fields), Allocation())


def bind_signature(
    interp: Interpreter,
    site: Origin,
    target: Callable[..., object],
    positional: list[Value],
    keywords: dict[str, Value],
    prefix: str = "",
) -> dict[str, Value]:
    """An injected default is a plain Python object, admitted under `prefix` and its parameter's name."""
    try:
        bound = inspect.signature(target).bind(*positional, **keywords)
    except TypeError as error:
        reject(site, f"the arguments do not bind: {error}")
    bound.apply_defaults()
    return {
        name: value if isinstance(value, VALUE_KINDS) else interp.snapshot.admit(prefix + name, value, site)
        for name, value in bound.arguments.items()
    }


def _factory(interp: Interpreter, node: Call, display: str, match: Factory, frame: Frame, sink: Sink) -> Value:
    values = _positional_arguments(interp, node, display, frame, sink, shares=False)
    host = [_static_argument(node.origin, display, value) for value in values]
    try:
        built = match.build(*host)
    except Exception as error:  # a registered library builder, not user code; its refusal is a diagnostic
        reject(node.origin, f"{display} rejects its arguments: {error}")
    tensor = tensor_of(built, display)
    if tensor is None:
        reject(node.origin, f"{display} must build a non-empty 1-D or 2-D numeric array")
    interp.budget.spend(len(tensor.leaves), node.origin, "the array factory")
    return tensor


def _static_argument(origin: Origin, display: str, value: Value) -> object:
    match value:
        case StaticScalar(const=const):
            return _ops.const_value(const)
        case SequenceValue(items=items):
            return tuple(_static_argument(origin, display, item) for item in items)
        case _:
            reject(origin, f"the arguments of {display} must be compile-time constants")


def _len(interp: Interpreter, node: Call, frame: Frame, sink: Sink) -> Value:
    values = _positional_arguments(interp, node, "len()", frame, sink, shares=False)
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
    values = _positional_arguments(interp, node, "range()", frame, sink, shares=False)
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
    source, start = _option_arguments(interp, node, "enumerate()", frame, sink, option="start")
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
    values = _positional_arguments(interp, node, display, frame, sink, shares=False)
    if len(values) != 1:
        reject(node.origin, f"{display} takes exactly one aggregate argument here")
    source = _aggregate.decay(interp.budget, values[0], node.origin)
    if not isinstance(source, (*AGGREGATES, IteratorValue)):
        reject(node.origin, f"{display} requires an aggregate argument")
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
        reject(node.origin, f"the dtype of {display} must be the Python type float or int")
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
                reject(origin, f"{display} of an empty sequence is not supported")
            shape, leaves = _tensor_rows(origin, display, items)
            explicit = family is not None
            if family is None:
                floaty = any(isinstance(leaf, Opaque) or leaf.stype is ScalarType.FLOAT for leaf in leaves)
                family = ScalarType.FLOAT if floaty else ScalarType.INT
            conformed = [tensor_leaf(interp, origin, family, leaf, sink, explicit=explicit) for leaf in leaves]
            interp.budget.spend(len(conformed), origin, "the array conversion")
            return TensorValue(shape, family, tuple(conformed), Allocation())
        case _:
            reject(origin, f"{display} requires a sequence or array argument, not {_aggregate.a_kind(value)}")


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
                        reject(origin, f"{display} supports only 1-D and 2-D rectangular constructions")
                    row.append(element)
                rows.append(row)
            case TensorValue(shape=shape, leaves=leaves) if len(shape) == 1:
                rows.append(list(leaves))
            case _:
                reject(origin, f"{display} supports only 1-D and 2-D rectangular constructions")
    widths = {len(row) for row in rows}
    if len(widths) != 1 or 0 in widths:
        reject(origin, f"{display} requires rectangular rows of equal nonzero length")
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
