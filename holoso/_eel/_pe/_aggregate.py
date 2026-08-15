"""
Statement-free aggregate structure: everything here maps values to values, while anything that must emit
residual statements lives with the interpreter.

Subscript semantics are CPython's, delegated to host indexing over the children where possible. A
multi-axis subscript validates the rank and EVERY integer axis against the full shape before any selection,
so an empty slice on one axis cannot mask a bounds fault on another.
"""

import dataclasses

from .._ir import Origin, ScalarType
from ._ops import const_value, make_const
from ._ownership import share
from ._reject import reject
from ._snapshot import describe_opaque, ndarray_annotation
from ._values import (
    AGGREGATES,
    Allocation,
    BoundMethod,
    ExpansionBudget,
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

_EXTRACTED = AGGREGATES

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
        case StaticScalar(const=start), StaticScalar(const=stop):
            a, b = const_value(start), const_value(stop)
            assert isinstance(a, int) and isinstance(b, int)
            return range(a, b, value.step)
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
    return SequenceValue(tuple(StaticScalar(make_const(i)) for i in span), Allocation())


def static_index(origin: Origin, value: Value, what: str) -> int:
    match value:
        case StaticScalar(const=const) if value.stype is ScalarType.INT:
            index = const_value(const)
            assert isinstance(index, int)
            return index
        case StaticScalar() | ResidualScalar() if value.stype is not ScalarType.INT:
            reject(origin, f"{what} must be an int, not a {value.stype.value}")
        case _:
            reject(origin, f"{what} must be a compile-time constant int")


def index_read(origin: Origin, base: Value, index: Value) -> Value:
    match base:
        case SequenceValue(items=items, allocation=allocation):
            if allocation.one_shot:
                reject(origin, "an enumerate iterator supports only iteration, as in Python")
            position = static_index(origin, index, "a subscript index")
            item = _select(origin, items, position, base)
            if isinstance(item, _EXTRACTED):
                share(base)
                share(item)
            return item
        case TensorValue(shape=shape, leaves=leaves):
            if isinstance(index, SequenceValue):
                reject(origin, "a sequence index on an array is not supported; spell the axes directly (m[i, j])")
            position = static_index(origin, index, "a subscript index")
            if len(shape) == 1:
                return _select(origin, leaves, position, base)
            return _derive_row(origin, base, position)
        case Opaque(name=name):
            reject(origin, f"cannot index {name!r}: the captured value is not a supported aggregate")
        case _:
            reject(origin, f"{a_kind(base)} is not subscriptable")


def slice_read(origin: Origin, base: Value, lo: int | None, hi: int | None) -> Value:
    match base:
        case SequenceValue(items=items, allocation=allocation):
            if allocation.one_shot:
                reject(origin, "an enumerate iterator supports only iteration, as in Python")
            taken = list(items)[lo:hi]
            for item in taken:
                if isinstance(item, _EXTRACTED):
                    share(item)
            return SequenceValue(tuple(taken), Allocation())
        case TensorValue(shape=shape, leaves=leaves):
            selected = _axis_range(shape[0], lo, hi)
            if not selected:
                reject(origin, "the slice selects no elements; an empty array is not supported")
            if len(shape) == 1:
                return _derived(base, (len(selected),), tuple(leaves[i] for i in selected))
            width = shape[1]
            picked = tuple(leaves[i * width + j] for i in selected for j in range(width))
            return _derived(base, (len(selected), width), picked)
        case _:
            reject(origin, f"{a_kind(base)} cannot be sliced")


def multi_index_read(origin: Origin, base: Value, axes: tuple[ResolvedAxis, ...]) -> Value:
    """
    The `m[i, j]` / `m[:, k]` / `m[a:b, c:d]` read; a sliced axis arrives as its resolved
    `(lo, hi)` bounds pair, an indexed axis as its value.
    """
    if not isinstance(base, TensorValue):
        reject(origin, f"too many indices: a multi-axis subscript works only on an array, not {a_kind(base)}")
    shape = base.shape
    if len(axes) != len(shape):
        reject(origin, f"a multi-axis subscript must name every axis: the array is {len(shape)}-D, got {len(axes)}")
    selections: list[list[int]] = []
    kept_dims: list[int | None] = []
    for dim, axis in zip(shape, axes, strict=True):
        if isinstance(axis, tuple):
            lo, hi = axis
            selection = _axis_range(dim, lo, hi)
            selections.append(selection)
            kept_dims.append(len(selection))
        else:
            position = static_index(origin, axis, "a subscript index")
            resolved = position + dim if position < 0 else position
            if not 0 <= resolved < dim:
                reject(origin, f"index {position} is out of bounds for an axis of length {dim}")
            selections.append([resolved])
            kept_dims.append(None)
    if any(not selection for selection in selections):
        reject(origin, "the slice selects no elements; an empty array is not supported")
    columns = selections[1] if len(shape) == 2 else [0]
    width = shape[1] if len(shape) == 2 else 1
    picked = tuple(base.leaves[i * width + j] for i in selections[0] for j in columns)
    kept = tuple(dim for dim in kept_dims if dim is not None)
    if not kept:
        assert len(picked) == 1
        return picked[0]
    return _derived(base, kept, picked)


def unpack_items(origin: Origin, value: Value, count: int) -> list[Value]:
    items = _iterated(origin, value)
    if len(items) > count:
        reject(origin, f"too many values to unpack (expected {count})")
    if len(items) < count:
        reject(origin, f"not enough values to unpack (expected {count}, got {len(items)})")
    return items


def splice_items(origin: Origin, value: Value) -> list[Value]:
    return _iterated(origin, value)


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
                    if len(shape) == 1:
                        walk(leaf, (*path, position))
                    else:
                        walk(leaf, (*path, position // shape[1], position % shape[1]))
            case StaticScalar() | ResidualScalar():
                leaves.append((path, node))
            case Opaque():
                reject(origin, describe_opaque(node))
            case _:
                reject(origin, f"{a_kind(node)} cannot be returned")

    walk(value, ())
    return leaves


def _iterated(origin: Origin, value: Value) -> list[Value]:
    """Top-level items as iteration/unpacking yields them; aggregate extractions share parent and item."""
    found: list[Value]
    match value:
        case SequenceValue(items=items, allocation=allocation):
            if allocation.one_shot:
                if allocation.spent:
                    reject(origin, "this enumerate iterator is already exhausted, as it would be in Python")
                allocation.spent = True
            found = list(items)
        case TensorValue(shape=shape):
            if len(shape) == 1:
                found = list(value.leaves)
            else:
                found = [_derive_row(origin, value, row) for row in range(shape[0])]
        case _:
            reject(origin, f"cannot unpack {a_kind(value)}: it is not iterable")
    for item in found:
        if isinstance(item, _EXTRACTED):
            share(value)
            share(item)
    return found


def _select(
    origin: Origin, items: tuple[Value, ...] | tuple[Scalar | Opaque, ...], position: int, base: Value
) -> Value:
    try:
        selected = list(items)[position]
    except IndexError:
        reject(origin, f"index {position} is out of bounds for {a_kind(base)} of length {len(items)}")
    return selected


def _axis_range(dim: int, lo: int | None, hi: int | None) -> list[int]:
    return list(range(dim))[lo:hi]


def _derive_row(origin: Origin, base: TensorValue, position: int) -> TensorValue:
    rows, width = base.shape
    resolved = position + rows if position < 0 else position
    if not 0 <= resolved < rows:
        reject(origin, f"index {position} is out of bounds for an array of length {rows}")
    return _derived(base, (width,), tuple(base.leaves[resolved * width : (resolved + 1) * width]))


def _derived(base: TensorValue, shape: tuple[int, ...], leaves: tuple[Scalar | Opaque, ...]) -> TensorValue:
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
    if not ndarray_annotation(annotation):
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
