"""
Stubs that map 1:1 onto HIR operators. Stub names are irrelevant to dispatch.
The bodies delegate to library functions so each stub doubles as a plain-Python numerical reference.
"""

import math
import numpy as np
from ..._hir import *
from ._registry import IntegerImplementation, IntrinsicResultRule, intrinsic

_INT_OVERLOAD = IntrinsicResultRule.INT_OVERLOAD
_ALWAYS_INT = IntrinsicResultRule.ALWAYS_INT
_IDENTITY = IntegerImplementation.IDENTITY


# Rounding an integer keeps it integer: the numpy spellings preserve the operand kind (numpy floors an int64 to that
# int64), the math spellings always return a Python int; both run the float rounding operator on a float operand. An
# integer operand is thus an identity (contained at MIR), never promoted and rounded in the float datapath. np.rint is
# the exception -- it always returns float -- so it stays the default SIGNATURE (float-forcing) rule.
@intrinsic(FloatFloor, np.floor, result_rule=_INT_OVERLOAD, integer_implementation=_IDENTITY)
def floor_(x: float) -> float:
    return float(np.floor(x))


@intrinsic(FloatFloor, math.floor, result_rule=_ALWAYS_INT, integer_implementation=_IDENTITY)
def math_floor_(x: float) -> int:
    return math.floor(x)


@intrinsic(FloatCeil, np.ceil, result_rule=_INT_OVERLOAD, integer_implementation=_IDENTITY)
def ceil_(x: float) -> float:
    return float(np.ceil(x))


@intrinsic(FloatCeil, math.ceil, result_rule=_ALWAYS_INT, integer_implementation=_IDENTITY)
def math_ceil_(x: float) -> int:
    return math.ceil(x)


@intrinsic(FloatTrunc, np.trunc, np.fix, result_rule=_INT_OVERLOAD, integer_implementation=_IDENTITY)
def trunc_(x: float) -> float:
    return float(np.trunc(x))


@intrinsic(FloatTrunc, math.trunc, result_rule=_ALWAYS_INT, integer_implementation=_IDENTITY)
def math_trunc_(x: float) -> int:
    return math.trunc(x)


@intrinsic(FloatRound, np.round, np.around, result_rule=_INT_OVERLOAD, integer_implementation=_IDENTITY)
def round_(x: float) -> float:
    return float(np.round(x))


@intrinsic(FloatRound, np.rint)  # np.rint returns float even for an integer input: the default float-forcing rule
def rint_(x: float) -> float:
    return float(np.rint(x))


@intrinsic(FloatRound, round, result_rule=_ALWAYS_INT, integer_implementation=_IDENTITY)
def builtin_round_(x: float) -> int:
    return round(x)


# abs/np.abs preserve the operand kind (an integer uses IntAbs, contained at MIR); math.fabs/np.fabs always return
# float, so an integer operand promotes at the call (the default SIGNATURE rule).
@intrinsic(
    FloatAbs, abs, np.abs, np.absolute, result_rule=_INT_OVERLOAD, integer_implementation=IntegerImplementation.ABS
)
def abs_(x: float) -> float:
    return math.fabs(x)


@intrinsic(FloatAbs, math.fabs, np.fabs)
def fabs_(x: float) -> float:
    return math.fabs(x)


# min/max in every spelling share one numeric rule: an all-integer form uses IntRelational + IntSelect (contained
# at MIR), any float operand promotes the integer side and selects in float. Builtin min/max thus do NOT preserve the
# winning operand's Python type on a mixed call -- the documented C-style deviation. np.min/np.max are reductions and
# stay unregistered.
@intrinsic(FloatMin, min, result_rule=_INT_OVERLOAD, integer_implementation=IntegerImplementation.MIN)
def min_(a: float, b: float) -> float:
    return min(a, b)


@intrinsic(
    FloatMin,
    np.minimum,
    np.fmin,
    result_rule=_INT_OVERLOAD,
    integer_implementation=IntegerImplementation.MIN,
)
def numpy_min_(a: float, b: float) -> float:
    return min(a, b)


@intrinsic(FloatMax, max, result_rule=_INT_OVERLOAD, integer_implementation=IntegerImplementation.MAX)
def max_(a: float, b: float) -> float:
    return max(a, b)


@intrinsic(
    FloatMax,
    np.maximum,
    np.fmax,
    result_rule=_INT_OVERLOAD,
    integer_implementation=IntegerImplementation.MAX,
)
def numpy_max_(a: float, b: float) -> float:
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
