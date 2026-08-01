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
    """One runtime container's identity; the state only ever moves forward (never back toward UNIQUE)."""

    state: AllocationState = AllocationState.UNIQUE


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
    """A read of a tensor's bound method (``.flatten``); only a call consumes it."""

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
