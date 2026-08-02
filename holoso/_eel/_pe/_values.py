"""
The partial-evaluation value domain: what an environment name can hold.

A static scalar carries the typed HIR ``Const``: its dataclass equality is type-discriminating where Python
equality is not (``True == 1 == 1.0`` would silently unify a bool arm with an int arm at a join), and its
construction normalizes negative zero and refuses NaN, so the compiler's numeric invariants hold in the
evaluator's own state for free. An opaque value is a captured object that is not an admitted scalar (NaN
floats included); it is judged at its USE site, never at capture -- desugar hoists every callee through a
temp, and binding an unused NaN default is CPython-legal.

Aggregate dtype widths are not modeled: leaves live in the subset's width-less int / binary64 float value
model, the same ratified deviation scalars carry. The ``Allocation`` is the identity the ownership model
tracks: values are frozen and rebound while allocations persist and carry the monotone sharing state, so a
store under a residual branch is a new value over the same allocation and joins leafwise.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from .._ir import LocalRef, Origin, ScalarType, TempRef
from ._ops import Const, scalar_type
from ._reject import reject


@dataclass(frozen=True, slots=True)
class StaticScalar:
    const: Const

    @property
    def stype(self) -> ScalarType:
        return scalar_type(self.const)


@dataclass(frozen=True, slots=True)
class ResidualScalar:
    stype: ScalarType
    atom: TempRef | LocalRef


type Scalar = StaticScalar | ResidualScalar


@dataclass(frozen=True, slots=True)
class Opaque:
    name: str
    value: object


class AllocationState(Enum):
    UNIQUE = "unique"
    SHARED = "shared"
    ESCAPED = "escaped"


@dataclass(eq=False, slots=True)
class Allocation:
    """
    One runtime container's identity; the state only ever moves forward (never back toward UNIQUE), while
    ``borrows`` is the scoped overlay counting the active loops/comprehensions iterating this allocation --
    a counter rather than a flag because nested loops may iterate the same allocation. ``joined`` marks an
    allocation minted by a branch join of differing arm allocations: the value is ONE of the two runtime
    objects, so the fresh identity erases provenance -- installing such a tree into a state attribute is
    rejected, since the disjointness checks would judge the wrong object.
    """

    state: AllocationState = AllocationState.UNIQUE
    borrows: int = 0
    joined: bool = False


@dataclass(frozen=True, slots=True)
class SequenceValue:
    items: tuple["Value", ...]
    allocation: Allocation


@dataclass(frozen=True, slots=True)
class TensorValue:
    shape: tuple[int, ...]
    family: ScalarType
    leaves: tuple[Scalar | Opaque, ...]
    allocation: Allocation

    def __post_init__(self) -> None:
        assert self.family in (ScalarType.FLOAT, ScalarType.INT)
        assert 1 <= len(self.shape) <= 2 and all(dim >= 1 for dim in self.shape)
        assert len(self.leaves) == math.prod(self.shape)
        assert all(isinstance(leaf, Opaque) or leaf.stype is self.family for leaf in self.leaves)
        assert self.family is ScalarType.FLOAT or not any(isinstance(leaf, Opaque) for leaf in self.leaves)


@dataclass(frozen=True, slots=True)
class TensorMethod:
    """A read of a registered array method; only a call consumes it."""

    receiver: TensorValue
    name: str


type Value = StaticScalar | ResidualScalar | Opaque | SequenceValue | TensorValue | TensorMethod


def same(a: Value, b: Value) -> bool:
    """
    Semantic identity for join folding: typed constant equality for statics, binding identity for residuals
    (atom origins differ between reads of one binding and must not matter), object identity for opaques, and
    allocation identity plus leafwise sameness for aggregates (equal-looking distinct allocations are NOT the
    same value: mutating one would not touch the other).
    """
    if a is b:
        return True
    match a, b:
        case StaticScalar(), StaticScalar():
            return a.const == b.const
        case ResidualScalar(), ResidualScalar():
            return a.stype is b.stype and same_atom(a.atom, b.atom)
        case Opaque(), Opaque():
            return a.value is b.value
        case SequenceValue(), SequenceValue():
            return (
                a.allocation is b.allocation
                and len(a.items) == len(b.items)
                and all(same(x, y) for x, y in zip(a.items, b.items))
            )
        case TensorValue(), TensorValue():
            return (
                a.allocation is b.allocation
                and a.shape == b.shape
                and a.family is b.family
                and all(same(x, y) for x, y in zip(a.leaves, b.leaves))
            )
        case TensorMethod(), TensorMethod():
            return a.name == b.name and same(a.receiver, b.receiver)
        case _:
            return False


def same_atom(a: TempRef | LocalRef, b: TempRef | LocalRef) -> bool:
    match a, b:
        case TempRef(), TempRef():
            return a.index == b.index
        case LocalRef(), LocalRef():
            return a.name == b.name
        case _:
            return False


class ExpansionBudget:
    """
    The one graph-expansion bound of the whole lowering: every structure-producing expansion spends units at
    its expansion site, regardless of whether the produced structure later folds away, so a blow-up is a
    located rejection, never a hang. Exact constant arithmetic never spends: folding is not expansion.
    """

    def __init__(self, limit: int = 100_000) -> None:
        assert limit > 0
        self._remaining = limit
        self.spent = 0

    def spend(self, units: int, origin: Origin, construct: str) -> None:
        assert units > 0
        self.spent += units
        self._remaining -= units
        if self._remaining < 0:
            reject(origin, f"the graph expansion budget is exhausted while expanding {construct}")


def allocations_match(a: Value, b: Value) -> bool:
    match a, b:
        case SequenceValue(), SequenceValue():
            return (
                a.allocation is b.allocation
                and len(a.items) == len(b.items)
                and all(allocations_match(x, y) for x, y in zip(a.items, b.items))
            )
        case TensorValue(), TensorValue():
            return a.allocation is b.allocation
        case (SequenceValue() | TensorValue(), _) | (_, SequenceValue() | TensorValue()):
            return False
        case _:
            return True


def tree_leaves(value: Value) -> list[Scalar | Opaque]:
    """The scalar leaves of a slot tree in declaration order (depth-first, row-major)."""
    match value:
        case SequenceValue(items=items):
            return [leaf for item in items for leaf in tree_leaves(item)]
        case TensorValue(leaves=leaves):
            return list(leaves)
        case StaticScalar() | ResidualScalar() | Opaque():
            return [value]
        case _:
            raise AssertionError(value)


def tensor_occurrences(value: Value) -> list[Allocation]:
    """Every OCCURRENCE of a tensor allocation, walking through repeated sequence nodes each time."""
    match value:
        case SequenceValue(items=items):
            return [allocation for item in items for allocation in tensor_occurrences(item)]
        case TensorValue(allocation=allocation):
            return [allocation]
        case _:
            return []


def tree_rebuild(value: Value, leaves: Iterator[Scalar]) -> Value:
    """The same structure over the SAME allocations with the scalar leaves replaced in declaration order."""
    match value:
        case SequenceValue(items=items, allocation=allocation):
            return SequenceValue(tuple(tree_rebuild(item, leaves) for item in items), allocation)
        case TensorValue(shape=shape, family=family, allocation=allocation):
            replaced = tuple(next(leaves) for _ in value.leaves)
            return TensorValue(shape, family, replaced, allocation)
        case StaticScalar() | ResidualScalar() | Opaque():
            return next(leaves)
        case _:
            raise AssertionError(value)


def same_structure(a: Value, b: Value) -> bool:
    match a, b:
        case SequenceValue(), SequenceValue():
            return (
                a.allocation is b.allocation
                and len(a.items) == len(b.items)
                and all(same_structure(x, y) for x, y in zip(a.items, b.items, strict=True))
            )
        case TensorValue(), TensorValue():
            return a.allocation is b.allocation and a.shape == b.shape and a.family is b.family
        case (StaticScalar() | ResidualScalar() | Opaque()), (StaticScalar() | ResidualScalar() | Opaque()):
            return True
        case _:
            return False
