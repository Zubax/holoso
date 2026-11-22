"""
The integer operators and primitives. Every operator is one of Holoso's own ZKF blocks, Amaranth hardware with a
bit-exact `IntValue` reference, configured from its options; it answers its timing and evaluates every primitive running
on it, saturating wherever the operation can leave the format. Saturation is what the integer type does at its extremes
rather than a failure, and HIR marks the saturating operations speculatable, so no block reports it: an if-converted arm
that saturates must not raise the machine's error flag. The inline primitives are native Verilog over the whole wide
bank, sound because an integer fills that register exactly (DESIGN.md, Types).
"""

from ._iabss import IAbsOperator as IAbsOperator, IAbsPrimitive as IAbsPrimitive, IAbssOptions as IAbssOptions
from ._iadds import (
    IAddOperator as IAddOperator,
    IAddPrimitive as IAddPrimitive,
    IAddsOptions as IAddsOptions,
    ICmpPrimitive as ICmpPrimitive,
    ISubPrimitive as ISubPrimitive,
)
from ._idivs import (
    IDivOperator as IDivOperator,
    IDivsOptions as IDivsOptions,
    IFloorDivPrimitive as IFloorDivPrimitive,
    IModPrimitive as IModPrimitive,
)
from ._imuls import IMulOperator as IMulOperator, IMulPrimitive as IMulPrimitive, IMulsOptions as IMulsOptions
from ._inline import (
    BoolToIntPrimitive as BoolToIntPrimitive,
    IntBwAndPrimitive as IntBwAndPrimitive,
    IntBwNotPrimitive as IntBwNotPrimitive,
    IntBwOrPrimitive as IntBwOrPrimitive,
    IntBwXorPrimitive as IntBwXorPrimitive,
    IntShiftConstPrimitive as IntShiftConstPrimitive,
    IntToBoolPrimitive as IntToBoolPrimitive,
)
from ._ipopcnt import (
    IPopcntOperator as IPopcntOperator,
    IPopcntOptions as IPopcntOptions,
    IPopcntPrimitive as IPopcntPrimitive,
)
from ._ishft import (
    IShftOperator as IShftOperator,
    IShftOptions as IShftOptions,
    IShlPrimitive as IShlPrimitive,
    IShrPrimitive as IShrPrimitive,
)
