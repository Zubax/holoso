"""
Round-trip codec between Python floats and Zubax Kulibin float (ZKF) bit patterns, for any ``FloatFormat``.

Layout (matches ``holoso_support.v`` / the Kulibin model): ``[sign | exponent(wexp) | stored-fraction(wman-1)]`` with a
hidden leading significand bit. ``exp == 0`` is zero (ZKF has no subnormals and no negative zero); the all-ones
exponent is infinity. Encoding rounds to nearest, ties to even, using exact rational arithmetic.
"""

import math
from fractions import Fraction

from ..format import FloatFormat


def _bias(fmt: FloatFormat) -> int:
    return (1 << (fmt.wexp - 1)) - 1


def _wfrac(fmt: FloatFormat) -> int:
    return fmt.wman - 1


def _exp_inf(fmt: FloatFormat) -> int:
    return (1 << fmt.wexp) - 1


def _pow2(exp: int) -> Fraction:
    return Fraction(1 << exp, 1) if exp >= 0 else Fraction(1, 1 << -exp)


def _floor_log2(value: Fraction) -> int:
    exp = value.numerator.bit_length() - value.denominator.bit_length()
    while _pow2(exp + 1) <= value:
        exp += 1
    while _pow2(exp) > value:
        exp -= 1
    return exp


def _pack(fmt: FloatFormat, sign: int, exp: int, frac: int) -> int:
    wfrac = _wfrac(fmt)
    return ((sign & 1) << (fmt.width - 1)) | ((exp & _exp_inf(fmt)) << wfrac) | (frac & ((1 << wfrac) - 1))


def decode(fmt: FloatFormat, bits: int) -> float:
    """Exactly decode a ZKF bit pattern into the Python float it represents."""
    wfrac = _wfrac(fmt)
    sign = (bits >> (fmt.width - 1)) & 1
    exp = (bits >> wfrac) & _exp_inf(fmt)
    frac = bits & ((1 << wfrac) - 1)
    if exp == 0:
        return 0.0
    if exp == _exp_inf(fmt):
        return -math.inf if sign else math.inf
    significand = (1 << wfrac) | frac
    value = math.ldexp(float(significand), exp - _bias(fmt) - wfrac)
    return -value if sign else value


def encode(fmt: FloatFormat, x: float) -> int:
    """Encode a Python float to the nearest ZKF bit pattern (ties to even). NaN is rejected; ZKF has no NaN."""
    if math.isnan(x):
        raise ValueError("NaN is not representable in ZKF")
    sign = 1 if math.copysign(1.0, x) < 0 else 0
    magnitude = abs(x)
    if magnitude == 0.0:
        return 0  # canonical positive zero
    if math.isinf(magnitude):
        return _pack(fmt, sign, _exp_inf(fmt), 0)

    wfrac = _wfrac(fmt)
    value = Fraction(magnitude)
    exp = _floor_log2(value)
    min_exp = 1 - _bias(fmt)
    max_exp = _exp_inf(fmt) - 1 - _bias(fmt)
    if exp < min_exp:
        # Underflow: round against the half-MIN_NORMAL boundary (ZKF has no subnormals).
        return _pack(fmt, sign, 1, 0) if value >= _pow2(min_exp - 1) else 0

    scaled = value / _pow2(exp) * (1 << wfrac)
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (twice == scaled.denominator and (quotient & 1)):
        quotient += 1
    if quotient >= (1 << fmt.wman):
        quotient >>= 1
        exp += 1
    if exp > max_exp:
        return _pack(fmt, sign, _exp_inf(fmt), 0)
    return _pack(fmt, sign, exp + _bias(fmt), quotient & ((1 << wfrac) - 1))


def is_legal(fmt: FloatFormat, bits: int) -> bool:
    """Whether ``bits`` is a legal ZKF value (rejects subnormals and negative zero)."""
    wfrac = _wfrac(fmt)
    sign = (bits >> (fmt.width - 1)) & 1
    exp = (bits >> wfrac) & _exp_inf(fmt)
    frac = bits & ((1 << wfrac) - 1)
    if exp == 0:
        return frac == 0 and sign == 0  # only canonical +0
    return True


def is_finite(fmt: FloatFormat, bits: int) -> bool:
    wfrac = _wfrac(fmt)
    return ((bits >> wfrac) & _exp_inf(fmt)) != _exp_inf(fmt)
