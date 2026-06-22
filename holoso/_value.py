"""Runtime values and exact arithmetic for Zubax Kulibin float."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Self

from ._type import FloatFormat, pow2


@dataclass(frozen=True, slots=True, init=False)
class FloatValue:
    """A concrete ZKF value: a format plus the exact bits carried by hardware."""

    fmt: FloatFormat
    bits: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use FloatValue.from_bits() or FloatValue.from_float()")

    @classmethod
    def from_bits(cls, fmt: FloatFormat, bits: int) -> Self:
        """Construct a value from the exact bit pattern on a ZKF data port."""
        _check_format(fmt)
        if isinstance(bits, bool) or not isinstance(bits, int):
            raise TypeError(f"bits must be int, got {type(bits).__name__}")
        if not 0 <= bits < (1 << fmt.width):
            raise ValueError(f"bits 0x{bits:x} do not fit in {fmt.width} bits")
        return cls._new(fmt, bits)

    @classmethod
    def from_float(cls, fmt: FloatFormat, value: float) -> Self:
        """Encode a native Python float to the nearest ZKF value."""
        _check_format(fmt)
        if type(value) is not float:
            raise TypeError(f"value must be float, got {type(value).__name__}")
        return cls.from_bits(fmt, fmt.encode(value))

    @classmethod
    def _new(cls, fmt: FloatFormat, bits: int) -> Self:
        self = object.__new__(cls)
        object.__setattr__(self, "fmt", fmt)
        object.__setattr__(self, "bits", bits)
        return self

    def __float__(self) -> float:
        return self.fmt.decode(self.bits)

    @property
    def sign(self) -> int:
        return (self.bits >> (self.fmt.width - 1)) & 1

    @property
    def exponent(self) -> int:
        return (self.bits >> self.fmt.wfrac) & self.fmt.exp_inf

    @property
    def significand(self) -> int:
        return (1 << self.fmt.wfrac) | (self.bits & self.fmt.frac_mask)

    def apply_sign(self, *, negate: bool, absolute: bool) -> "FloatValue":
        """Apply the bit-level sign conditioner used by ``holoso_fsgnop``."""
        sign_shift = self.fmt.width - 1
        body = self.bits & ((1 << sign_shift) - 1)
        out_sign = (self.sign & (0 if absolute else 1)) ^ (1 if negate else 0)
        return FloatValue.from_bits(self.fmt, (out_sign << sign_shift) | body)


@dataclass(frozen=True, slots=True)
class _Decoded:
    bits: int
    sign: int
    exp: int
    frac: int
    significand: int
    is_zero: bool
    is_inf: bool


def add_float_values(a: FloatValue, b: FloatValue) -> FloatValue:
    """Exact ZKF add matching ``zkf_add``."""
    fmt = _matching_format(a, b)
    da = _decode(a)
    db = _decode(b)

    if da.is_inf and db.is_inf:
        return _canonical_inf(fmt, da.sign) if da.sign == db.sign else _zero(fmt)
    if da.is_inf:
        return _canonical_inf(fmt, da.sign)
    if db.is_inf:
        return _canonical_inf(fmt, db.sign)

    result = _finite_fraction(fmt, da) + _finite_fraction(fmt, db)
    return FloatValue.from_bits(fmt, fmt.pack(result))


def mul_float_values(a: FloatValue, b: FloatValue) -> FloatValue:
    """Exact ZKF multiply matching ``zkf_mul``."""
    fmt = _matching_format(a, b)
    da = _decode(a)
    db = _decode(b)
    result_zero = da.is_zero or db.is_zero
    result_inf = (not result_zero) and (da.is_inf or db.is_inf)

    product = da.significand * db.significand
    product_high = (product >> ((2 * fmt.wman) - 1)) & 1
    exp_unbiased_base = da.exp + db.exp - (fmt.bias << 1)

    if product_high:
        exp_unbiased = exp_unbiased_base + 1
        significand = (product >> fmt.wman) & _mask(fmt.wman)
        guard = (product >> (fmt.wman - 1)) & 1
        round_bit = (product >> (fmt.wman - 2)) & 1
        sticky = _sticky_below(product, fmt.wman - 3)
    else:
        exp_unbiased = exp_unbiased_base
        significand = (product >> (fmt.wman - 1)) & _mask(fmt.wman)
        guard = (product >> (fmt.wman - 2)) & 1
        round_bit = (product >> (fmt.wman - 3)) & 1
        sticky = _sticky_below(product, fmt.wman - 4)

    return _pack_reference(
        fmt,
        da.sign ^ db.sign,
        force_zero=result_zero,
        force_inf=result_inf,
        exp_unbiased=exp_unbiased,
        significand_value=significand,
        guard=guard,
        round_bit=round_bit,
        sticky=sticky,
    )


def div_float_values(a: FloatValue, b: FloatValue) -> FloatValue:
    """Exact ZKF divide quotient matching ``zkf_div``; error sidebands are intentionally ignored."""
    fmt = _matching_format(a, b)
    da = _decode(a)
    db = _decode(b)

    if da.is_zero or db.is_inf:
        return _zero(fmt)

    result_sign = da.sign if db.is_zero else (da.sign ^ db.sign)
    if db.is_zero or da.is_inf:
        return _canonical_inf(fmt, result_sign)

    value = Fraction(da.significand, db.significand) * pow2(da.exp - db.exp)
    return FloatValue.from_bits(fmt, fmt.pack(-value if result_sign else value))


def compare_float_values(a: FloatValue, b: FloatValue) -> int:
    """
    Exact total-order comparison of two values matching ``zkf_cmp``: -1 if ``a < b``, 0 if equal, +1 if ``a > b``.
    ZKF has no NaN (the order is total) and no negative zero. Comparison is on exact values, not on a lossy float
    decode, so it stays bit-exact always.
    """
    fmt = _matching_format(a, b)

    def key(value: FloatValue) -> tuple[int, Fraction]:
        decoded = _decode(value)
        if decoded.is_inf:  # -inf sorts below every finite value, +inf above; the fraction tier is unused
            return -1 if decoded.sign else 1, Fraction(0, 1)
        return 0, _finite_fraction(fmt, decoded)

    ka, kb = key(a), key(b)
    return (ka > kb) - (ka < kb)


def mul_ilog2_float_value(a: FloatValue, k: int) -> FloatValue:
    """Exact ZKF scaling by ``2**k`` matching ``zkf_mul_ilog2_const``."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    fmt = a.fmt
    da = _decode(a)
    if da.is_zero:
        return _zero(fmt)
    if da.is_inf:
        return _canonical_inf(fmt, da.sign)
    new_exp = da.exp + k
    if new_exp < 0:
        return _zero(fmt)
    if new_exp == 0:
        return _normal(fmt, da.sign, 1, 0)
    if new_exp > fmt.exp_max_finite:
        return _canonical_inf(fmt, da.sign)
    return _normal(fmt, da.sign, new_exp, da.frac)


def _check_format(fmt: FloatFormat) -> None:
    if not isinstance(fmt, FloatFormat):
        raise TypeError(f"fmt must be FloatFormat, got {type(fmt).__name__}")


def _matching_format(a: FloatValue, b: FloatValue) -> FloatFormat:
    if not isinstance(a, FloatValue):
        raise TypeError(f"left operand must be FloatValue, got {type(a).__name__}")
    if not isinstance(b, FloatValue):
        raise TypeError(f"right operand must be FloatValue, got {type(b).__name__}")
    if a.fmt != b.fmt:
        raise ValueError(f"operand format mismatch: {a.fmt} != {b.fmt}")
    return a.fmt


def _decode(value: FloatValue) -> _Decoded:
    fmt = value.fmt
    sign = value.sign
    exp = value.exponent
    frac = value.bits & fmt.frac_mask
    return _Decoded(
        bits=value.bits,
        sign=sign,
        exp=exp,
        frac=frac,
        significand=(1 << fmt.wfrac) | frac,
        is_zero=exp == 0,
        is_inf=exp == fmt.exp_inf,
    )


def _finite_fraction(fmt: FloatFormat, value: _Decoded) -> Fraction:
    if value.is_zero:
        return Fraction(0, 1)
    result = Fraction(value.significand, 1) * pow2(value.exp - fmt.bias - fmt.wfrac)
    return -result if value.sign else result


def _pack_reference(
    fmt: FloatFormat,
    sign: int,
    *,
    force_zero: bool,
    force_inf: bool,
    exp_unbiased: int,
    significand_value: int,
    guard: int,
    round_bit: int,
    sticky: int,
) -> FloatValue:
    exp_biased = exp_unbiased + fmt.bias
    exp_underflow_zero = exp_unbiased < (fmt.min_exp_unbiased - 1)
    exp_one_below_min = exp_unbiased == (fmt.min_exp_unbiased - 1)
    exp_overflow = exp_unbiased > fmt.max_exp_unbiased

    round_increment = bool(guard and (round_bit or sticky or (significand_value & 1)))
    rounded_ext = (significand_value & _mask(fmt.wman)) + (1 if round_increment else 0)
    round_carry = (rounded_ext >> fmt.wman) & 1
    rounded_significand = (rounded_ext >> 1) if round_carry else (rounded_ext & _mask(fmt.wman))
    exp_round_overflow = exp_biased == fmt.exp_max_finite and bool(round_carry)
    infinity = bool(force_inf or exp_overflow or exp_round_overflow)

    result_zero = bool(force_zero or ((not force_inf) and exp_underflow_zero))
    result_infinity = (not result_zero) and infinity
    result_min_normal = (not result_zero) and (not result_infinity) and (not force_inf) and exp_one_below_min

    if result_zero:
        return _zero(fmt)
    if result_infinity:
        return _canonical_inf(fmt, sign)
    if result_min_normal:
        return _normal(fmt, sign, 1, 0)

    exp_rounded = (exp_biased + round_carry) & _mask(fmt.wexp)
    return _from_parts(fmt, sign, exp_rounded, rounded_significand & fmt.frac_mask)


def _zero(fmt: FloatFormat) -> FloatValue:
    return _from_parts(fmt, 0, 0, 0)


def _canonical_inf(fmt: FloatFormat, sign: int) -> FloatValue:
    return _from_parts(fmt, sign, fmt.exp_inf, 0)


def _normal(fmt: FloatFormat, sign: int, exp: int, frac: int) -> FloatValue:
    return _from_parts(fmt, sign, exp, frac)


def _from_parts(fmt: FloatFormat, sign: int, exp: int, frac: int) -> FloatValue:
    return FloatValue.from_bits(
        fmt, ((sign & 1) << (fmt.width - 1)) | ((exp & fmt.exp_inf) << fmt.wfrac) | (frac & fmt.frac_mask)
    )


def _sticky_below(value: int, high_bit: int) -> int:
    if high_bit < 0:
        return 0
    return 1 if (value & _mask(high_bit + 1)) != 0 else 0


def _mask(width: int) -> int:
    return (1 << width) - 1
