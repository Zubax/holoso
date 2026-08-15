"""
Stubs that map 1:1 onto HIR operators. Stub names are irrelevant to dispatch.
The bodies delegate to library functions so each stub doubles as a plain-Python numerical reference.
Beware of subtle differences in return types; e.g., `round(float) -> int` while `np.round(float) -> float`.
"""

import math
import numpy as np
from ..._hir import *
from ._registry import intrinsic, lift


@intrinsic(FloatFloor, np.floor)  # math.floor() etc are excluded because they return int and require special handling.
def floor(x: float) -> float:
    return float(np.floor(x))


@intrinsic(FloatCeil, np.ceil)
def ceil(x: float) -> float:
    return float(np.ceil(x))


@intrinsic(FloatTrunc, np.trunc, np.fix)
def trunc(x: float) -> float:
    return float(np.trunc(x))


# An entry is per key, so naming np.rint (and below math.fabs/np.fabs) only here withholds the integer entry
# their groups carry, their own answer on an integer being a float.
@intrinsic(FloatRound, np.round, np.around, np.rint)
def round_(x: float) -> float:
    return float(np.round(x))


@intrinsic(FloatAbs, abs, np.abs, np.absolute, math.fabs, np.fabs)
def abs_float(x: float) -> float:
    return math.fabs(x)


# Exact at arbitrary precision, so abs(-2**63) is 2**63 like CPython, where numpy wraps under int64.
@intrinsic(IntAbs, abs, np.abs, np.absolute)
def abs_int(x: int) -> int:
    return abs(x)


# The one array-capable scalar family; math.fabs/np.fabs stay scalar-only (their own integer answer is a float).
lift(abs, np.abs, np.absolute)


@intrinsic(IntPopcount, int.bit_count, np.bitwise_count)
def popcount(x: int) -> int:
    return x.bit_count()


# The numpy binary elementwise spellings (np.minimum/np.maximum and the NaN-suppressing np.fmin/np.fmax) are array
# composites that delegate each element pair back to these scalar entries; np.min/np.max are reductions.
@intrinsic(FloatMin, min)
def min_float(a: float, b: float) -> float:
    return min(a, b)


@intrinsic(FloatMax, max)
def max_float(a: float, b: float) -> float:
    return max(a, b)


@intrinsic(FloatFma, math.fma)
def fma(a: float, b: float, c: float) -> float:
    return math.fma(a, b, c)


@intrinsic(FloatExp2, math.exp2, np.exp2)
def exp2(x: float) -> float:
    return float(np.exp2(x))


@intrinsic(FloatLog2, math.log2, np.log2)
def log2(x: float) -> float:
    return float(np.log2(x))  # -inf at the pole and nan off the domain, like the hardware; math.log2 raises instead


@intrinsic(FloatSqrt, math.sqrt, np.sqrt)
def sqrt(x: float) -> float:
    return math.sqrt(x)


@intrinsic(FloatSin, math.sin, np.sin)
def sin(x: float) -> float:
    return math.sin(x)


@intrinsic(FloatCos, math.cos, np.cos)
def cos(x: float) -> float:
    return math.cos(x)


@intrinsic(FloatAtan2, math.atan2, np.arctan2, np.atan2)
def atan2(y: float, x: float) -> float:
    return math.atan2(y, x)


@intrinsic(FloatHypot2, math.hypot, np.hypot)
def hypot(x: float, y: float) -> float:
    return math.hypot(x, y)


@intrinsic(FloatIsFinite, math.isfinite, np.isfinite)
def isfinite(x: float) -> bool:
    return math.isfinite(x)


@intrinsic(FloatIsInf, math.isinf, np.isinf)
def isinf(x: float) -> bool:
    return math.isinf(x)


@intrinsic(FloatIsPosInf, np.isposinf)
def isposinf(x: float) -> bool:
    return bool(np.isposinf(x))


@intrinsic(FloatIsNegInf, np.isneginf)
def isneginf(x: float) -> bool:
    return bool(np.isneginf(x))
