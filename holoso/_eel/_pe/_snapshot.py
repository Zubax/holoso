"""
Captured host values -> partial-evaluation values; the single place that decides what a captured object means.
Everything environment-rooted (globals, closure cells, injected defaults, attribute reads of frozen objects)
enters ESCAPED per the ownership model, and conversion is MEMOIZED BY OBJECT IDENTITY,
so two captures sharing a sub-object map to the same allocation and every read of one object yields one identity.
Anything inadmissible stays a lazily-judged Opaque -- binding it is CPython-legal; only a use rejects.
The optional `guard` is the overlap check against the state trees, called on every captured
aggregate with the capturing read's origin; the state trees are built before any capture, so this one
direction establishes the disjointness invariant.
"""

import math
from collections.abc import Callable

import numpy as np

from .._names import element_index
from ._ownership import escape
from .._ir import Origin, ScalarType
from ._values import Allocation, Opaque, Scalar, SequenceValue, StaticScalar, TensorValue, Value

CAPTURED_AGGREGATES = (list, tuple, np.ndarray)


def dtype_family(kind: str) -> ScalarType | None:
    return {"f": ScalarType.FLOAT, "i": ScalarType.INT, "u": ScalarType.INT}.get(kind)


def nan_payload(raw: object) -> bool:
    return isinstance(raw, (float, np.floating)) and math.isnan(float(raw))


def describe_opaque(value: Opaque) -> str:
    # An Opaque never carries an admissible float, so a float-typed payload here is necessarily a NaN.
    if nan_payload(value.value):
        return f"the captured value of {value.name!r} is NaN, which the compiler cannot represent"
    return f"the captured value of {value.name!r} is not a bool, int, or float scalar"


def scalar_of(raw: object, name: str) -> StaticScalar | Opaque | None:
    if isinstance(raw, (bool, np.bool_)):
        return StaticScalar.of(bool(raw))
    if isinstance(raw, (int, np.integer)):
        return StaticScalar.of(int(raw))
    if type(raw) is float or isinstance(raw, np.floating):
        if nan_payload(raw):
            return Opaque(name, raw)
        return StaticScalar.of(float(raw))
    return None


def tensor_of(array: object, name: str) -> TensorValue | None:
    if type(array) is not np.ndarray:  # a subclass (np.matrix, a masked array) redefines its own operators
        return None
    if array.ndim not in (1, 2) or 0 in array.shape:
        return None
    family = dtype_family(array.dtype.kind)
    if family is None:
        return None
    leaves: list[Scalar | Opaque] = []
    for position, element in enumerate(array.flatten().tolist()):
        leaf = scalar_of(element, f"{name}[{', '.join(map(str, element_index(array.shape, position)))}]")
        assert leaf is not None
        leaves.append(leaf)
    return TensorValue(tuple(array.shape), family, tuple(leaves), Allocation())


class Snapshotter:
    """One capture boundary per interpretation; the memo pins object identity for the interpreter's lifetime."""

    def __init__(self, guard: Callable[[str, object, Origin], None] | None = None) -> None:
        self._memo: dict[int, tuple[object, Value]] = {}
        self._guard = guard

    def admit(self, name: str, raw: object, origin: Origin) -> Value:
        scalar = scalar_of(raw, name)
        if scalar is not None:
            return scalar
        if isinstance(raw, CAPTURED_AGGREGATES):
            found = self._memo.get(id(raw))
            if found is not None:
                return found[1]
            if self._guard is not None:
                self._guard(name, raw, origin)
            value = self._aggregate(name, raw, origin)
            escape(value)
            self._memo[id(raw)] = (raw, value)
            return value
        return Opaque(name, raw)

    def _aggregate(self, name: str, raw: list[object] | tuple[object, ...] | np.ndarray, origin: Origin) -> Value:
        if isinstance(raw, np.ndarray):
            tensor = tensor_of(raw, name)
            return Opaque(name, raw) if tensor is None else tensor
        items = tuple(self.admit(f"{name}[{position}]", item, origin) for position, item in enumerate(raw))
        return SequenceValue(items, Allocation())
