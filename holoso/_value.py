"""Runtime values and exact arithmetic for Zubax Kulibin float."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Self

from ._type import FloatFormat, pow2


@dataclass(frozen=True, slots=True, init=False)
class FloatValue:
    """
    A concrete ZKF value: a format plus the exact bits carried by hardware.
    ``__eq__`` stays the dataclass's structural ``(fmt, bits)`` equality (used for hashing/interning);
    numeric ordering goes through :meth:`compare`.
    """

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

    def __add__(self, other: "FloatValue") -> "FloatValue":
        """Exact ZKF add matching ``zkf_add``."""
        fmt = _matching_format(self, other)
        da = _decode(self)
        db = _decode(other)

        if da.is_inf and db.is_inf:
            return _canonical_inf(fmt, da.sign) if da.sign == db.sign else _zero(fmt)
        if da.is_inf:
            return _canonical_inf(fmt, da.sign)
        if db.is_inf:
            return _canonical_inf(fmt, db.sign)

        result = _finite_fraction(fmt, da) + _finite_fraction(fmt, db)
        return FloatValue.from_bits(fmt, fmt.pack(result))

    def __mul__(self, other: "FloatValue") -> "FloatValue":
        """Exact ZKF multiply matching ``zkf_mul``."""
        fmt = _matching_format(self, other)
        da = _decode(self)
        db = _decode(other)
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

    def __truediv__(self, other: "FloatValue") -> "FloatValue":
        """Exact ZKF divide quotient matching ``zkf_div``; error sidebands are intentionally ignored."""
        fmt = _matching_format(self, other)
        da = _decode(self)
        db = _decode(other)

        if da.is_zero or db.is_inf:
            return _zero(fmt)

        result_sign = da.sign if db.is_zero else (da.sign ^ db.sign)
        if db.is_zero or da.is_inf:
            return _canonical_inf(fmt, result_sign)

        value = Fraction(da.significand, db.significand) * pow2(da.exp - db.exp)
        return FloatValue.from_bits(fmt, fmt.pack(-value if result_sign else value))

    def compare(self, other: "FloatValue") -> int:
        """
        Exact total-order comparison matching ``zkf_cmp``: -1 if ``self < other``, 0 if equal, +1 if ``self > other``.
        ZKF has no NaN (the order is total) and no negative zero. Comparison is on exact values, not on a lossy float
        decode, so it stays bit-exact always. A method rather than overloaded ordering operators: ``__eq__`` is the
        frozen-dataclass structural (bit) equality used for hashing, and numeric ordering would be inconsistent with it
        where equal values differ in bits, so the relations could not form a coherent order with ``==``; one three-way
        result also mirrors the single-firing hardware comparator.
        """
        fmt = _matching_format(self, other)

        def key(value: FloatValue) -> tuple[int, Fraction]:
            decoded = _decode(value)
            if decoded.is_inf:  # -inf sorts below every finite value, +inf above; the fraction tier is unused
                return -1 if decoded.sign else 1, Fraction(0, 1)
            return 0, _finite_fraction(fmt, decoded)

        ka, kb = key(self), key(other)
        return (ka > kb) - (ka < kb)

    def scale_pow2(self, k: int) -> "FloatValue":
        """Exact ZKF scaling by ``2**k`` matching ``zkf_mul_ilog2_const``."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be int, got {type(k).__name__}")
        fmt = self.fmt
        da = _decode(self)
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

    @staticmethod
    def fma(a: "FloatValue", b: "FloatValue", c: "FloatValue") -> "FloatValue":
        """
        Exact fused multiply-add ``a*b + c``, rounded once (ties to even), matching ``zkf_fma``. Specials mirror the
        core: an infinite product or ``c`` forces infinity, except opposite-sign ``inf + inf``, which cancels to +0.
        """
        fmt = _matching_format(a, b)
        if not isinstance(c, FloatValue):
            raise TypeError(f"fma addend must be FloatValue, got {type(c).__name__}")
        if c.fmt != fmt:
            raise ValueError(f"operand format mismatch: {c.fmt} != {fmt}")
        da, db, dc = _decode(a), _decode(b), _decode(c)
        product_zero = da.is_zero or db.is_zero
        product_inf = (not product_zero) and (da.is_inf or db.is_inf)
        product_sign = da.sign ^ db.sign
        if product_inf and dc.is_inf and product_sign != dc.sign:
            return _zero(fmt)
        if product_inf or dc.is_inf:
            return _canonical_inf(fmt, product_sign if product_inf else dc.sign)
        product = Fraction(0, 1) if product_zero else _finite_fraction(fmt, da) * _finite_fraction(fmt, db)
        return FloatValue.from_bits(fmt, fmt.pack(product + _finite_fraction(fmt, dc)))

    @staticmethod
    def sort(a: "FloatValue", b: "FloatValue") -> tuple["FloatValue", "FloatValue"]:
        """
        The operands sorted ascending as ``(min, max)``, matching ``zkf_sort``: a bit-preserving selection by the
        total order of :meth:`compare` (ZKF has no NaN). Equal values are bit-identical, so the tie direction is
        unobservable.
        """
        return (a, b) if a.compare(b) < 0 else (b, a)

    def round(self) -> "FloatValue":
        """Round to the nearest integral-valued float, ties to even."""
        return self._round_to_integral(round)

    def floor(self) -> "FloatValue":
        return self._round_to_integral(math.floor)

    def ceil(self) -> "FloatValue":
        return self._round_to_integral(math.ceil)

    def trunc(self) -> "FloatValue":
        return self._round_to_integral(math.trunc)

    def _round_to_integral(self, to_integer: Callable[[Fraction], int]) -> "FloatValue":
        """
        Apply ``to_integer`` to the exact value. Infinity passes through; zero and any zero-magnitude result
        canonicalize to +0 (so ``(-0.3).ceil()`` is +0); an overflowing integer becomes signed inf.
        """
        fmt = self.fmt
        da = _decode(self)
        if da.is_inf:
            return _canonical_inf(fmt, da.sign)
        if da.is_zero:
            return _zero(fmt)
        integer = to_integer(_finite_fraction(fmt, da))
        if integer == 0:
            return _zero(fmt)
        return FloatValue.from_bits(fmt, fmt.pack(Fraction(integer)))


@dataclass(frozen=True, slots=True)
class _Decoded:
    bits: int
    sign: int
    exp: int
    frac: int
    significand: int
    is_zero: bool
    is_inf: bool


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
