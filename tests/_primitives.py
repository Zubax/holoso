"""
Every pooled primitive an operator serves, for the tests that must reach every way an operator is driven; the
completeness guard ties the catalogue to the primitive classes the package defines, so a new primitive cannot slip it.
"""

import inspect
import random

from holoso._operators import (
    FAddPrimitive,
    FAtan2Primitive,
    FCmpPrimitive,
    FCordicOperator,
    FCosPrimitive,
    FDivPrimitive,
    FExp2Primitive,
    FFmaOperator,
    FFmaPrimitive,
    FFromIntPrimitive,
    FHypotPrimitive,
    FILog2Primitive,
    FLog2Primitive,
    FMaxPrimitive,
    FMinPrimitive,
    FMulILog2Operator,
    FMulILog2Primitive,
    FMulPrimitive,
    FRintPrimitive,
    FRintToIntPrimitive,
    FSinPrimitive,
    FSqrtPrimitive,
    HardwareOperator,
    IAbsPrimitive,
    IAddPrimitive,
    ICmpPrimitive,
    IFloorDivPrimitive,
    IModPrimitive,
    IMulPrimitive,
    IPopcntPrimitive,
    IShlPrimitive,
    IShrPrimitive,
    ISubPrimitive,
    PooledPrimitive,
)
from holoso._operators._common import TotalOrdering
from holoso._operators._float import (
    FAddOperator,
    FCmpOperator,
    FDivsqrtOperator,
    FExp2Operator,
    FFromIntOperator,
    FILog2Operator,
    FLog2Operator,
    FMulOperator,
    FRintOperator,
)
from holoso._operators._int import (
    IAbsOperator,
    IAddOperator,
    IDivOperator,
    IMulOperator,
    IPopcntOperator,
    IShftOperator,
)
from holoso._type import FloatType, IntType, ScalarType
from holoso._value import FloatValue, IntValue, RoundMode


def primitives_of(operator: HardwareOperator) -> list[PooledPrimitive]:
    """Every primitive `operator` serves: a comparing one at each ordering, a rounding one at each mode."""
    match operator:
        case FAddOperator():
            return [FAddPrimitive(operator)]
        case FMulOperator():
            return [FMulPrimitive(operator)]
        case FDivsqrtOperator():
            return [FDivPrimitive(operator), FSqrtPrimitive(operator)]
        case FILog2Operator():
            return [FILog2Primitive(operator)]
        case FMulILog2Operator():
            return [FMulILog2Primitive(operator)]
        case FCmpOperator():
            orderings: list[PooledPrimitive] = [FCmpPrimitive(operator, ordering) for ordering in TotalOrdering]
            return [*orderings, FMinPrimitive(operator), FMaxPrimitive(operator)]
        case FRintOperator():
            floats: list[PooledPrimitive] = [FRintPrimitive(operator, mode) for mode in RoundMode]
            return [*floats, *(FRintToIntPrimitive(operator, mode) for mode in RoundMode)]
        case FFmaOperator():
            return [FFmaPrimitive(operator)]
        case FExp2Operator():
            return [FExp2Primitive(operator)]
        case FLog2Operator():
            return [FLog2Primitive(operator)]
        case FCordicOperator():
            return [
                FSinPrimitive(operator),
                FCosPrimitive(operator),
                FAtan2Primitive(operator),
                FHypotPrimitive(operator),
            ]
        case FFromIntOperator():
            return [FFromIntPrimitive(operator)]
        case IAddOperator():
            sums: list[PooledPrimitive] = [IAddPrimitive(operator), ISubPrimitive(operator)]
            return [*sums, *(ICmpPrimitive(operator, ordering) for ordering in TotalOrdering)]
        case IMulOperator():
            return [IMulPrimitive(operator)]
        case IDivOperator():
            return [IFloorDivPrimitive(operator), IModPrimitive(operator)]
        case IAbsOperator():
            return [IAbsPrimitive(operator)]
        case IShftOperator():
            return [IShlPrimitive(operator), IShrPrimitive(operator)]
        case IPopcntOperator():
            return [IPopcntPrimitive(operator)]
    assert False, f"no primitives catalogued for {type(operator).__name__}"


def concrete_primitive_kinds() -> set[str]:
    kinds: set[str] = set()
    pending: list[type] = [PooledPrimitive]
    while pending:
        kind = pending.pop()
        pending.extend(kind.__subclasses__())
        if not inspect.isabstract(kind) and kind.__module__.startswith("holoso."):
            kinds.add(kind.__name__)
    return kinds


def random_operand(rng: random.Random, scalar_type: ScalarType) -> FloatValue | IntValue:
    """A float in [-3, 3], or an integer within a byte, its edges as likely as the rest of it together."""
    if isinstance(scalar_type, FloatType):
        return scalar_type.fmt.encode(rng.uniform(-3, 3))
    assert isinstance(scalar_type, IntType)
    return IntValue.from_int(scalar_type.fmt, rng.choice([rng.choice([0, 1, -1, -128, 127]), rng.randint(-128, 127)]))
