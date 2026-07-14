"""
Stubs that map 1:1 onto HIR operators. Stub names are irrelevant to dispatch.
The bodies delegate to library functions so each stub doubles as a plain-Python numerical reference.
"""

import math
import numpy as np
from ..._hir import *
from ._registry import intrinsic


# The math spellings return a Python int (int-returning), so a floored value stays an exact integer and downstream
# integer arithmetic does not round in the float datapath; the numpy spellings return float. Both run the same float
# rounding operator, so a float-context use (float(math.floor(x))) elides the int round-trip back to the operator.
@intrinsic(FloatFloor, np.floor)
def floor_(x: float) -> float:
    return float(np.floor(x))


@intrinsic(FloatFloor, math.floor, returns_int=True)
def math_floor_(x: float) -> int:
    return math.floor(x)


@intrinsic(FloatCeil, np.ceil)
def ceil_(x: float) -> float:
    return float(np.ceil(x))


@intrinsic(FloatCeil, math.ceil, returns_int=True)
def math_ceil_(x: float) -> int:
    return math.ceil(x)


@intrinsic(FloatTrunc, np.trunc, np.fix)
def trunc_(x: float) -> float:
    return float(np.trunc(x))


@intrinsic(FloatTrunc, math.trunc, returns_int=True)
def math_trunc_(x: float) -> int:
    return math.trunc(x)


@intrinsic(FloatRound, np.round, np.rint, np.around)
def round_(x: float) -> float:
    return float(np.round(x))


@intrinsic(FloatRound, round, returns_int=True)
def builtin_round_(x: float) -> int:
    return round(x)


@intrinsic(FloatAbs, abs, math.fabs, np.abs, np.absolute, np.fabs)
def abs_(x: float) -> float:
    return math.fabs(x)


# np.minimum/maximum (and NaN-suppressing np.fmin/fmax) are the binary elementwise forms; np.min/np.max are reductions
# and are deliberately unregistered. NaN-propagation differences between the spellings are moot under the fast-math /
# no-NaN policy, so all binary spellings collapse onto one operator.
@intrinsic(FloatMin, min, np.minimum, np.fmin)
def min_(a: float, b: float) -> float:
    return min(a, b)


@intrinsic(FloatMax, max, np.maximum, np.fmax)
def max_(a: float, b: float) -> float:
    return max(a, b)


@intrinsic(FloatFma, math.fma)
def fma_(a: float, b: float, c: float) -> float:
    return math.fma(a, b, c)


@intrinsic(FloatExp2, math.exp2, np.exp2)
def exp2_(x: float) -> float:
    return float(np.exp2(x))


@intrinsic(FloatLog2, math.log2, np.log2)
def log2_(x: float) -> float:
    return math.log2(x)


@intrinsic(FloatSqrt, math.sqrt, np.sqrt)
def sqrt_(x: float) -> float:
    return math.sqrt(x)


@intrinsic(FloatSin, math.sin, np.sin)
def sin_(x: float) -> float:
    return math.sin(x)


@intrinsic(FloatCos, math.cos, np.cos)
def cos_(x: float) -> float:
    return math.cos(x)


@intrinsic(FloatAtan2, math.atan2, np.arctan2, np.atan2)
def atan2_(y: float, x: float) -> float:
    return math.atan2(y, x)


@intrinsic(FloatHypot2, math.hypot, np.hypot)
def hypot_(x: float, y: float) -> float:
    return math.hypot(x, y)


@intrinsic(FloatIsFinite, math.isfinite, np.isfinite)
def isfinite_(x: float) -> bool:
    return math.isfinite(x)


@intrinsic(FloatIsInf, math.isinf, np.isinf)
def isinf_(x: float) -> bool:
    return math.isinf(x)


@intrinsic(FloatIsPosInf, np.isposinf)
def isposinf_(x: float) -> bool:
    return bool(np.isposinf(x))


@intrinsic(FloatIsNegInf, np.isneginf)
def isneginf_(x: float) -> bool:
    return bool(np.isneginf(x))
