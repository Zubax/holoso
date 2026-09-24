"""
Statement-free aggregate structure: everything here maps values to values, while anything that must emit
residual statements lives with the interpreter.

Subscript semantics are CPython's, delegated to host indexing over the children where possible. An array
subscript validates EVERY integer axis against the full shape before refusing an empty selection, so an empty
slice on one axis cannot mask a bounds fault on another.
"""

import dataclasses

import numpy as np

from .._ir import Origin, ScalarType
from .._names import element_index
from ._ownership import share
from ._reject import reject
from ._snapshot import describe_opaque
from ._values import (
    AGGREGATES,
    Allocation,
    BoundMethod,
    ExpansionBudget,
    IteratorValue,
    LoopPass,
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

type LeafPath = tuple[int | str, ...]

type ResolvedAxis = Value | tuple[int | None, int | None]


def a_kind(value: Value) -> str:
    label = kind_label(value)
    return ("an " if label[0] in "aeiou" else "a ") + label


def kind_label(value: Value) -> str:
    match value:
        case SequenceValue():
            return "sequence"
        case TensorValue():
            return "array"
        case RecordValue():
            return "record"
        case IteratorValue():
            return "enumerate iterator"
        case Opaque():
            return "captured object"
        case BoundMethod(receiver=TensorValue()):
            return "bound array method"
        case BoundMethod():
            return "bound scalar method"
        case RangeValue():
            return "range"
        case _:
            return "scalar"


def static_range(value: RangeValue) -> range | None:
    match value.start, value.stop:
        case StaticScalar(value=int() as start), StaticScalar(value=int() as stop):
            return range(start, stop, value.step)
        case _:
            return None


def range_length(span: range) -> int:
    """Exact arithmetic: host `len()` of a huge range overflows Py_ssize_t."""
    return max(0, (span.stop - span.start + span.step - (1 if span.step > 0 else -1)) // span.step)


def decay(budget: ExpansionBudget, value: Value, origin: Origin) -> Value:
    """Wherever an aggregate is demanded: a static range materializes, a runtime one drives only a counted for."""
    if not isinstance(value, RangeValue):
        return value
    span = static_range(value)
    if span is None:
        reject(origin, "a range with a runtime bound can only drive a for loop")
    budget.spend(max(range_length(span), 1), origin, "the range materialization")
    return SequenceValue(tuple(StaticScalar.of(i) for i in span), Allocation())


def static_index(origin: Origin, value: Value, what: str) -> int:
    match value:
        case StaticScalar(value=int() as index) if value.stype is ScalarType.INT:
            return index
        case StaticScalar() | ResidualScalar() if value.stype is not ScalarType.INT:
            reject(origin, f"{what} must be an int, not a {value.stype.value}")
        case _:
            reject(origin, f"{what} must be a compile-time constant int")


def index_read(origin: Origin, base: Value, axis: ResolvedAxis) -> Value:
    """`b[i]` or `b[lo:hi]`; a sliced axis arrives as its resolved `(lo, hi)` bounds pair."""
    match base:
        case SequenceValue(items=items):
            if isinstance(axis, tuple):
                lo, hi = axis
                taken = items[lo:hi]
                for item in taken:
                    if isinstance(item, AGGREGATES):
                        share(item)
                return SequenceValue(taken, Allocation())
            position = static_index(origin, axis, "a subscript index")
            item = items[resolved_index(origin, position, len(items), a_kind(base))]
            if isinstance(item, AGGREGATES):
                share(base)
                share(item)
            return item
        case TensorValue(shape=shape):
            return _tensor_read(origin, base, (axis, *[(None, None)] * (len(shape) - 1)))
        case _ if isinstance(axis, tuple):
            reject(origin, f"{a_kind(base)} cannot be sliced")
        case Opaque(name=name):
            reject(origin, f"cannot index {name!r}: the captured value is not a supported aggregate")
        case _:
            reject(origin, f"{a_kind(base)} is not subscriptable")


def multi_index_read(origin: Origin, base: Value, axes: tuple[ResolvedAxis, ...]) -> Value:
    """The `m[i, j]` / `m[:, k]` / `m[a:b, c:d]` read, one resolved axis per dimension."""
    if not isinstance(base, TensorValue):
        reject(origin, f"too many indices: a multi-axis subscript works only on an array, not {a_kind(base)}")
    if len(axes) != len(base.shape):
        reject(
            origin, f"a multi-axis subscript must name every axis: the array is {len(base.shape)}-D, got {len(axes)}"
        )
    return _tensor_read(origin, base, axes)


def axis_position(origin: Origin, index: Value, dim: int) -> int:
    if isinstance(index, SequenceValue):
        reject(origin, "a sequence index on an array is not supported; spell the axes directly (m[i, j])")
    return resolved_index(origin, static_index(origin, index, "a subscript index"), dim, "an axis")


def resolved_index(origin: Origin, position: int, length: int, what: str) -> int:
    resolved = position + length if position < 0 else position
    if not 0 <= resolved < length:
        reject(origin, f"index {position} is out of bounds for {what} of length {length}")
    return resolved


def _tensor_read(origin: Origin, base: TensorValue, axes: tuple[ResolvedAxis, ...]) -> Value:
    """An indexed axis drops out of the result and a sliced one stays, so indexing every axis reads a leaf."""
    selections: list[list[int]] = []
    kept: list[int] = []
    for dim, axis in zip(base.shape, axes, strict=True):
        if isinstance(axis, tuple):
            lo, hi = axis
            selections.append(list(range(dim))[lo:hi])
            kept.append(len(selections[-1]))
        else:
            selections.append([axis_position(origin, axis, dim)])
    if not all(selections):
        reject(origin, "the slice selects no elements; an empty array is not supported")
    columns = selections[1] if len(base.shape) == 2 else [0]
    width = base.shape[1] if len(base.shape) == 2 else 1
    picked = tuple(base.leaves[i * width + j] for i in selections[0] for j in columns)
    if not kept:
        assert len(picked) == 1
        return picked[0]
    return derive(base, tuple(kept), picked)


def unpack_items(origin: Origin, value: Value, count: int, loops: list[LoopPass]) -> list[Value]:
    items = splice_items(origin, value, loops)
    if len(items) > count:
        reject(origin, f"too many values to unpack (expected {count})")
    if len(items) < count:
        reject(origin, f"not enough values to unpack (expected {count}, got {len(items)})")
    return items


def flatten(origin: Origin, value: Value) -> list[tuple[LeafPath, Scalar]]:
    leaves: list[tuple[LeafPath, Scalar]] = []

    def walk(node: Value, path: LeafPath) -> None:
        match node:
            case SequenceValue(items=items):
                if not items:
                    reject(origin, "an empty aggregate cannot be returned")
                for position, item in enumerate(items):
                    walk(item, (*path, position))
            case RecordValue(cls=cls, fields=fields):
                if not fields:
                    reject(origin, "an empty aggregate cannot be returned")
                for field, value in zip(dataclasses.fields(cls), fields, strict=True):
                    walk(value, (*path, field.name))
            case TensorValue(shape=shape, leaves=tensor_leaves):
                for position, leaf in enumerate(tensor_leaves):
                    walk(leaf, (*path, *element_index(shape, position)))
            case StaticScalar() | ResidualScalar():
                leaves.append((path, node))
            case Opaque():
                reject(origin, describe_opaque(node))
            case _:
                reject(origin, f"{a_kind(node)} cannot be returned")

    walk(value, ())
    return leaves


def splice_items(origin: Origin, value: Value, loops: list[LoopPass]) -> list[Value]:
    """Top-level items as iteration/unpacking yields them; aggregate extractions share parent and item."""
    found: list[Value]
    match value:
        case IteratorValue(items=items, made_in=made_in):
            if list(made_in[: len(loops)]) != loops:
                reject(
                    origin,
                    "an enumerate iterator made outside this data-dependent loop would be re-consumed on "
                    "every hardware iteration where Python drains it once; call enumerate inside the loop",
                )
            if value.spent:
                reject(origin, "an enumerate iterator can only be consumed once; call enumerate again")
            value.spent = True
            found = list(items)
        case SequenceValue(items=items):
            found = list(items)
        case TensorValue(shape=shape):
            if len(shape) == 1:
                found = list(value.leaves)
            else:
                rows, width = shape
                found = [derive(value, (width,), value.leaves[row * width : (row + 1) * width]) for row in range(rows)]
        case _:
            reject(origin, f"cannot unpack {a_kind(value)}: it is not iterable")
    for item in found:
        if isinstance(item, AGGREGATES):
            share(value)
            share(item)
    return found


def derive(base: TensorValue, shape: tuple[int, ...], leaves: tuple[Scalar | Opaque, ...]) -> TensorValue:
    # The derivation reuses the source allocation as a storage-equivalence token: store blocking is identical
    # (one shared allocation either way), and the state-install disjointness checks see views for free.
    result = TensorValue(shape, base.family, leaves, base.allocation)
    share(base)
    return result


def array_annotation_shape(annotation: object, origin: Origin, what: str) -> tuple[tuple[int, ...], ScalarType] | None:
    """
    Detected structurally (a type carrying `dims`), so the annotation library stays a dependency of the
    user's code only.
    """
    if not (isinstance(annotation, type) and hasattr(annotation, "dims")):
        return None
    if getattr(annotation, "array_type", None) is not np.ndarray:
        reject(origin, f"{what}: only numpy array containers are supported in shaped annotations")
    dims = getattr(annotation, "dims", None)
    if not isinstance(dims, tuple):
        reject(origin, f"{what}: not a valid fixed-shape array annotation")
    sizes: list[int] = []
    for dim in dims:
        size = getattr(dim, "size", None)
        if not isinstance(size, int) or getattr(dim, "broadcastable", False):
            reject(origin, f'{what}: array dimensions must be fixed integers (e.g. Float64[np.ndarray, "3 3"])')
        if size < 1:
            reject(origin, f"{what}: array dimensions must be at least 1")
        sizes.append(size)
    if len(sizes) not in (1, 2):
        reject(origin, f"{what}: only 1-D and 2-D arrays are supported, got {len(sizes)}-D")
    dtypes = getattr(annotation, "dtypes", None)
    if not isinstance(dtypes, (tuple, list)) or not dtypes:
        reject(origin, f"{what}: the array element type must be a float or integer family (e.g. Float64)")
    if all(isinstance(name, str) and name.startswith(("float", "bfloat")) for name in dtypes):
        family = ScalarType.FLOAT
    elif all(isinstance(name, str) and name.startswith(("int", "uint")) for name in dtypes):
        family = ScalarType.INT
    else:
        reject(origin, f"{what}: the array element type must be a float or integer family (e.g. Float64)")
    return tuple(sizes), family
