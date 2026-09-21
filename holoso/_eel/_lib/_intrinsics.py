"""
Stubs that map 1:1 onto HIR operators.
Each body is its operator's plain-Python numerical reference.
"""

import math

import numpy as np
from ..._hir import *
from ._registry import intrinsic, variadic


@intrinsic(FloatFloor())
def floor(x: float) -> float:
    return float(np.floor(x))


@intrinsic(FloatCeil())
def ceil(x: float) -> float:
    return float(np.ceil(x))


@intrinsic(FloatTrunc())
def trunc(x: float) -> float:
    return float(np.trunc(x))


@intrinsic(FloatRound())
def round_(x: float) -> float:
    return float(np.round(x))


@intrinsic(FloatAbs())
def abs_float(x: float) -> float:
    return math.fabs(x)


# Exact at arbitrary precision, so abs(-2**63) is 2**63 like CPython, where numpy wraps under int64.
@intrinsic(IntAbs())
def abs_int(x: int) -> int:
    return abs(x)


@intrinsic(IntPopcount())
def popcount(x: int) -> int:
    return x.bit_count()


@intrinsic(FloatMin())
def min_float(a: float, b: float) -> float:
    return min(a, b)


@intrinsic(FloatMax())
def max_float(a: float, b: float) -> float:
    return max(a, b)


# Each integer reference is exact at arbitrary precision, where the hardware saturates at the native width.
@intrinsic(IntAdd())
def add_int(a: int, b: int) -> int:
    return a + b


@intrinsic(FloatAdd())
def add_float(a: float, b: float) -> float:
    return a + b


@intrinsic(IntSub())
def subtract_int(a: int, b: int) -> int:
    return a - b


@intrinsic(IntMul())
def multiply_int(a: int, b: int) -> int:
    return a * b


@intrinsic(FloatMul())
def multiply_float(a: float, b: float) -> float:
    return a * b


@intrinsic(FloatDiv())
def divide(a: float, b: float) -> float:
    return a / b


@intrinsic(IntDivFloor())
def floor_divide(a: int, b: int) -> int:
    return a // b


@intrinsic(IntMod())
def remainder(a: int, b: int) -> int:
    return a % b


# The shifters follow Python, which has no negative count: a count known to be negative is refused rather than folded,
# while the hardware reverses direction on one it meets at run time.
@intrinsic(IntShiftLeft())
def left_shift(a: int, n: int) -> int:
    return a << n


@intrinsic(IntShiftRight())
def right_shift(a: int, n: int) -> int:
    return a >> n


@intrinsic(BoolAnd())
def and_bool(a: bool, b: bool) -> bool:
    return a & b


@intrinsic(IntBwAnd())
def and_int(a: int, b: int) -> int:
    return a & b


@intrinsic(BoolOr())
def or_bool(a: bool, b: bool) -> bool:
    return a | b


@intrinsic(IntBwOr())
def or_int(a: int, b: int) -> int:
    return a | b


@intrinsic(BoolXor())
def xor_bool(a: bool, b: bool) -> bool:
    return a != b


@intrinsic(IntBwXor())
def xor_int(a: int, b: int) -> int:
    return a ^ b


@intrinsic(BoolNot())
def not_bool(x: bool) -> bool:
    return not x


@intrinsic(IntNeg())
def negative_int(x: int) -> int:
    return -x


@intrinsic(FloatNeg())
def negative_float(x: float) -> float:
    return -x


@intrinsic(IntBwNot())
def invert(x: int) -> int:
    return ~x


@intrinsic(IntLess())
def less_int(a: int, b: int) -> bool:
    return a < b


@intrinsic(FloatLess())
def less_float(a: float, b: float) -> bool:
    return a < b


@intrinsic(IntLessOrEqual())
def less_equal_int(a: int, b: int) -> bool:
    return a <= b


@intrinsic(FloatLessOrEqual())
def less_equal_float(a: float, b: float) -> bool:
    return a <= b


@intrinsic(IntGreater())
def greater_int(a: int, b: int) -> bool:
    return a > b


@intrinsic(FloatGreater())
def greater_float(a: float, b: float) -> bool:
    return a > b


@intrinsic(IntGreaterOrEqual())
def greater_equal_int(a: int, b: int) -> bool:
    return a >= b


@intrinsic(FloatGreaterOrEqual())
def greater_equal_float(a: float, b: float) -> bool:
    return a >= b


@intrinsic(IntEqual())
def equal_int(a: int, b: int) -> bool:
    return a == b


@intrinsic(FloatEqual())
def equal_float(a: float, b: float) -> bool:
    return a == b


@intrinsic(IntNotEqual())
def not_equal_int(a: int, b: int) -> bool:
    return a != b


@intrinsic(FloatNotEqual())
def not_equal_float(a: float, b: float) -> bool:
    return a != b


@intrinsic(FloatFma())
def fma(a: float, b: float, c: float) -> float:
    return math.fma(a, b, c)


@intrinsic(FloatExp2())
def exp2(x: float) -> float:
    return float(np.exp2(x))


@intrinsic(FloatLog2())
def log2(x: float) -> float:
    return float(np.log2(x))  # -inf at the pole and nan off the domain, like the hardware; math.log2 raises instead


@intrinsic(FloatSqrt())
def sqrt(x: float) -> float:
    return math.sqrt(x)


@intrinsic(FloatSin())
def sin(x: float) -> float:
    return math.sin(x)


@intrinsic(FloatCos())
def cos(x: float) -> float:
    return math.cos(x)


@intrinsic(FloatAtan2())
def atan2(y: float, x: float) -> float:
    return math.atan2(y, x)


@variadic(FloatHypot, math.hypot, minimum=1)
def hypot(*coords: float) -> float:
    return math.hypot(*coords)


# numpy's is a strictly binary ufunc -- `np.hypot(a, b, c)` is a TypeError on the host -- so its entry stays fixed.
@intrinsic(FloatHypot(2))
def hypot_pair(x: float, y: float) -> float:
    return math.hypot(x, y)


@intrinsic(FloatIsFinite())
def isfinite(x: float) -> bool:
    return math.isfinite(x)


@intrinsic(FloatIsInf())
def isinf(x: float) -> bool:
    return math.isinf(x)


@intrinsic(FloatIsPosInf())
def isposinf(x: float) -> bool:
    return bool(np.isposinf(x))


@intrinsic(FloatIsNegInf())
def isneginf(x: float) -> bool:
    return bool(np.isneginf(x))


@intrinsic(FloatToInt())
def int_from_float(x: float) -> int:
    return int(x)


@intrinsic(BoolToInt())
def int_from_bool(x: bool) -> int:
    return int(x)


@intrinsic(BoolToFloat())
def float_from_bool(x: bool) -> float:
    return float(x)


@intrinsic(IntToBool())
def bool_from_int(x: int) -> bool:
    return bool(x)


@intrinsic(FloatToBool())
def bool_from_float(x: float) -> bool:
    return bool(x)


@intrinsic(IntToFloat())
def float_from_int(x: int) -> float:
    return float(x)
