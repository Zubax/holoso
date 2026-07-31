"""
The partial-evaluation value domain: what an environment name can hold.

A static scalar carries the typed HIR ``Const``: its dataclass equality is type-discriminating where Python
equality is not (``True == 1 == 1.0`` would silently unify a bool arm with an int arm at a join), and its
construction normalizes negative zero and refuses NaN, so the compiler's numeric invariants hold in the
evaluator's own state for free. An opaque value is a captured object that is not an admitted scalar (NaN
floats included); it is judged at its USE site, never at capture -- desugar hoists every callee through a
temp, and binding an unused NaN default is CPython-legal. A tuple value is the flat scalar row of a return;
tuples and opaques also ride unread across call boundaries. Nothing else consumes tuples until aggregates
land (M5).
"""

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class TupleValue:
    items: tuple[Scalar, ...]


type Value = StaticScalar | ResidualScalar | Opaque | TupleValue


def same(a: Value, b: Value) -> bool:
    """
    Semantic identity for join folding: typed constant equality for statics, binding identity for residuals
    (atom origins differ between reads of one binding and must not matter), object identity for opaques.
    """
    match a, b:
        case StaticScalar(), StaticScalar():
            return a.const == b.const
        case ResidualScalar(), ResidualScalar():
            return a.stype is b.stype and same_atom(a.atom, b.atom)
        case Opaque(), Opaque():
            return a.value is b.value
        case TupleValue(), TupleValue():
            return len(a.items) == len(b.items) and all(same(x, y) for x, y in zip(a.items, b.items))
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
    The one graph-expansion bound of the whole lowering: every structure-producing expansion (power chains now;
    inlining, unrolling, comprehensions, repetition in later milestones) spends units here, at the expansion
    site, regardless of whether the produced structure later folds away -- so a blow-up is a located rejection,
    never a hang. Exact constant arithmetic never spends: folding is not expansion.
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
