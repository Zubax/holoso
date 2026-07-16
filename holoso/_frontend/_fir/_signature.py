"""
Parameter and return contracts parsed from the kernel's annotations. Array contracts are detected structurally (a
jaxtyping-style class carrying ``dims``), so jaxtyping stays a dependency of the user's code only. Parsing is
total over the supported grammar and raises :class:`ContractError` otherwise; the builder locates the error at the
offending parameter or return annotation. Fulfillment is a separate concern: the aggregate stages lower the array
and aggregate contracts, and until then their ports are honest located rejections rather than silent scalar seeds.
"""

import math
import types
import typing
from dataclasses import dataclass

from ._value import SemType

_SCALAR_KINDS: dict[object, SemType] = {float: SemType.FLOAT, bool: SemType.BOOL, int: SemType.INT}


class ContractError(ValueError):
    """A malformed or unsupported annotation; the caller locates and rethrows it."""


@dataclass(frozen=True, slots=True)
class ScalarParameter:
    kind: SemType


@dataclass(frozen=True, slots=True)
class ArrayParameter:
    shape: tuple[int, ...]


type ParameterContract = ScalarParameter | ArrayParameter


@dataclass(frozen=True, slots=True)
class VoidReturn:
    pass


@dataclass(frozen=True, slots=True)
class ScalarReturn:
    kind: SemType


@dataclass(frozen=True, slots=True)
class TupleReturn:
    items: tuple["ReturnContract", ...]


@dataclass(frozen=True, slots=True)
class VariadicTupleReturn:
    item: "ReturnContract"


@dataclass(frozen=True, slots=True)
class ListReturn:
    item: "ReturnContract"


@dataclass(frozen=True, slots=True)
class ArrayReturn:
    shape: tuple[int, ...]


type ReturnContract = VoidReturn | ScalarReturn | TupleReturn | VariadicTupleReturn | ListReturn | ArrayReturn


def is_array_annotation(annotation: object) -> bool:
    """A jaxtyping-style array annotation: a class carrying ``dims``, detected structurally."""
    return isinstance(annotation, type) and hasattr(annotation, "dims")


# A resource limit distinct from the loop-unrolling threshold: every array-annotation leaf becomes a physical
# scalar port, so the cap bounds port fan-out, name generation, and per-leaf plan sizes.
MAX_ARRAY_PORT_LEAVES = 4096


def array_shape(annotation: object) -> tuple[int, ...]:
    """
    The fixed shape of a jaxtyping-style array annotation. Everything must be static -- a symbolic, variadic, or
    broadcastable dimension has no memory to size at run time. The dtype category must be floating-point, and the
    annotated carrier must be np.ndarray itself (Float64[list, "2"] would seed LIST semantics as an array); the
    concrete hardware format still comes from the operator configuration, not the annotation.
    """
    import numpy as np

    dims = getattr(annotation, "dims", None)
    if not isinstance(dims, tuple):  # a real jaxtyping type always carries a dims tuple; anything else is not one
        raise ContractError("not a valid fixed-shape array annotation")
    if getattr(annotation, "array_type", None) is not np.ndarray:
        raise ContractError('the array annotation must be over np.ndarray (e.g. Float64[np.ndarray, "3"])')
    sizes: list[int] = []
    for dim in dims:
        size = getattr(dim, "size", None)
        if not isinstance(size, int) or getattr(dim, "broadcastable", False):
            raise ContractError('array dimensions must be fixed integers (e.g. Float64[np.ndarray, "3 3"])')
        if size < 1:
            raise ContractError("array dimensions must be at least 1")
        sizes.append(size)
    if len(sizes) not in (1, 2):
        raise ContractError(f"only 1-D and 2-D arrays are supported, got {len(sizes)}-D")
    if math.prod(sizes) > MAX_ARRAY_PORT_LEAVES:
        raise ContractError(f"the array decomposes into {math.prod(sizes)} ports, beyond {MAX_ARRAY_PORT_LEAVES}")
    dtypes = getattr(annotation, "dtypes", None)  # e.g. jaxtyping Shaped carries a non-iterable any-dtype marker
    if not isinstance(dtypes, (tuple, list)) or not all(
        isinstance(name, str) and name.startswith(("float", "bfloat")) for name in dtypes
    ):
        raise ContractError("the array element type must be floating-point (e.g. Float64)")
    return tuple(sizes)


def parameter_contract(annotation: object) -> ParameterContract:
    kind = _SCALAR_KINDS.get(annotation)
    if kind is not None:
        return ScalarParameter(kind)
    if is_array_annotation(annotation):
        return ArrayParameter(array_shape(annotation))
    raise ContractError("expected float, bool, int, or a fixed-shape jaxtyping array")


def return_contract(hint: object) -> ReturnContract:
    # ``X | None``: the None arm is the implicit fall-off of an early-return kernel, so unwrap it.
    args = typing.get_args(hint)
    if typing.get_origin(hint) in (typing.Union, types.UnionType) and type(None) in args:
        remainder = [arg for arg in args if arg is not type(None)]
        if len(remainder) == 1:
            hint = remainder[0]
    if hint is None or hint is type(None):
        return VoidReturn()
    kind = _SCALAR_KINDS.get(hint)
    if kind is not None:
        return ScalarReturn(kind)
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return VariadicTupleReturn(return_contract(args[0]))
        if args in ((), ((),)):  # ``tuple[()]``: the canonical empty-tuple annotation (a zero-output bundle)
            return TupleReturn(())
        return TupleReturn(tuple(return_contract(arg) for arg in args))
    if origin is list:
        if len(args) == 1:
            return ListReturn(return_contract(args[0]))
        raise ContractError("a list return annotation must carry exactly one element type (e.g. list[float])")
    if is_array_annotation(hint):
        return ArrayReturn(array_shape(hint))
    raise ContractError("expected float, bool, int, None, a tuple/list of them, or a fixed-shape jaxtyping array")
