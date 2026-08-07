"""Runtime values and exact arithmetic for Zubax Kulibin float and for the native saturating integer."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Self

import zkf
from zkf import RoundMode as RoundMode

from ._type import FloatFormat, IntFormat


class SortResult(NamedTuple):
    min: FloatValue
    max: FloatValue


class SinCos(NamedTuple):
    sin: FloatValue
    cos: FloatValue


class Atan2Result(NamedTuple):
    theta: FloatValue
    magnitude: FloatValue


@dataclass(frozen=True, slots=True, init=False, repr=False)
class FloatValue:
    """A concrete ZKF value. Results match the ``zkf_*`` RTL bit-for-bit."""

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
        self = object.__new__(cls)  # __init__ is disabled, so build the frozen instance directly
        object.__setattr__(self, "fmt", fmt)
        object.__setattr__(self, "bits", bits)
        return self

    @classmethod
    def from_float(cls, fmt: FloatFormat, value: float) -> Self:
        """Encode a native Python float to the nearest ZKF value."""
        _check_format(fmt)
        if type(value) is not float:
            raise TypeError(f"value must be float, got {type(value).__name__}")
        return cls.from_bits(fmt, fmt.encode(value))

    def __float__(self) -> float:
        return self.fmt.decode(self.bits)

    def __repr__(self) -> str:
        """If the value is too large to fit in the native double, it is rendered as a fraction instead."""
        number = float(self)
        if math.isinf(number) and self.fmt.is_finite(self.bits):
            return str(self._zval.to_fraction())
        return repr(number)

    @property
    def negative(self) -> bool:
        return self._zval.negative

    @property
    def exponent(self) -> int:
        return self._zval.exp

    def apply_sign(self, *, negate: bool, absolute: bool) -> FloatValue:
        """Apply the sign conditioner of ``holoso_fsgnop``: absolute value first, then optional negation."""
        value = self._zval
        if absolute:
            value = abs(value)
        if negate:
            value = -value
        return FloatValue.from_bits(self.fmt, value.bits)

    def __add__(self, other: FloatValue) -> FloatValue:
        fmt = _matching_format(self, other)
        return FloatValue.from_bits(fmt, (self._zval + other._zval).bits)

    def __mul__(self, other: FloatValue) -> FloatValue:
        fmt = _matching_format(self, other)
        return FloatValue.from_bits(fmt, (self._zval * other._zval).bits)

    def __truediv__(self, other: FloatValue) -> FloatValue:
        """``zkf_div``'s error sidebands are intentionally not modeled."""
        fmt = _matching_format(self, other)
        return FloatValue.from_bits(fmt, (self._zval / other._zval).bits)

    def compare(self, other: FloatValue) -> int:
        """
        -1/0/+1 three-way compare (ZKF's order is total: no NaN). A method, not ordering operators, because ``__eq__``
        is the frozen-dataclass bit equality used for hashing -- numeric ordering would be inconsistent with it where
        equal values differ in bits.
        """
        _matching_format(self, other)
        result = self._zval.cmp(other._zval)
        return result.gt - result.lt

    def scale_pow2(self, k: int) -> FloatValue:
        """Matches ``zkf_mul_ilog2_const``."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be int, got {type(k).__name__}")
        return FloatValue.from_bits(self.fmt, self._zval.mul_ilog2(k).bits)

    @staticmethod
    def fma(a: FloatValue, b: FloatValue, c: FloatValue) -> FloatValue:
        """Fused multiply-add ``a*b + c``, rounded once (ties to even)."""
        fmt = _matching_format(a, b)
        if not isinstance(c, FloatValue):
            raise TypeError(f"fma addend must be FloatValue, got {type(c).__name__}")
        if c.fmt != fmt:
            raise ValueError(f"operand format mismatch: {c.fmt} != {fmt}")
        return FloatValue.from_bits(fmt, a._zval.fma(b._zval, c._zval).bits)

    @staticmethod
    def sort(a: FloatValue, b: FloatValue) -> SortResult:
        fmt = _matching_format(a, b)
        lo, hi = a._zval.sort(b._zval)
        return SortResult(FloatValue.from_bits(fmt, lo.bits), FloatValue.from_bits(fmt, hi.bits))

    def round(self) -> FloatValue:
        return FloatValue.from_bits(self.fmt, self._zval.round().bits)

    def floor(self) -> FloatValue:
        return FloatValue.from_bits(self.fmt, self._zval.floor().bits)

    def ceil(self) -> FloatValue:
        return FloatValue.from_bits(self.fmt, self._zval.ceil().bits)

    def trunc(self) -> FloatValue:
        return FloatValue.from_bits(self.fmt, self._zval.trunc().bits)

    def exp2(self) -> FloatValue:
        return FloatValue.from_bits(self.fmt, self._zval.exp2().bits)

    def log2(self) -> FloatValue:
        """``zkf_log2``'s domain-error/pole sidebands are intentionally not modeled (as with ``zkf_div``'s div0)."""
        return FloatValue.from_bits(self.fmt, self._zval.log2().value.bits)

    def sincos(self) -> SinCos:
        """``(sin(2*pi*self), cos(2*pi*self))`` -- turn-native, as ``zkf_sincos``; the quadrant sideband is dropped."""
        r = self._zval.sincos()
        return SinCos(FloatValue.from_bits(self.fmt, r.sin.bits), FloatValue.from_bits(self.fmt, r.cos.bits))

    @staticmethod
    def atan2(y: FloatValue, x: FloatValue) -> Atan2Result:
        """``(theta, magnitude)`` of ``atan2(y, x)`` -- theta in turns, magnitude ``hypot(y, x)`` (``zkf_atan2``)."""
        fmt = _matching_format(y, x)
        r = y._zval.atan2(x._zval)
        return Atan2Result(FloatValue.from_bits(fmt, r.theta.bits), FloatValue.from_bits(fmt, r.magnitude.bits))

    @property
    def _zval(self) -> zkf.Zkf:
        return _zkf_format(self.fmt).wrap(self.bits)


class DivResult(NamedTuple):
    quo: IntValue
    rem: IntValue


class ShiftResult(NamedTuple):
    shft: IntValue  # the raw bit shift, letting a left shift push bits past the word
    prod: IntValue  # the same shift as a multiplication by a power of two, saturating instead


@dataclass(frozen=True, slots=True, init=False, repr=False)
class IntValue:
    """
    A concrete native integer, signed and saturating. Results match the ``holoso_i*`` RTL bit-for-bit, edge cases
    included: ``abs(MIN)`` is ``MAX``, ``MIN // -1`` is ``(MAX, 0)``, and a division by zero answers a rail by the
    numerator's sign while keeping the numerator as the remainder. The saturation sidebands those modules also raise
    are intentionally not modeled, as ``zkf_div``'s ``div0`` is not.
    """

    fmt: IntFormat
    value: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use IntValue.from_bits() or IntValue.from_int()")

    @classmethod
    def from_bits(cls, fmt: IntFormat, bits: int) -> Self:
        """Construct a value from the exact bit pattern on a two's-complement data port."""
        _check_int_format(fmt)
        if isinstance(bits, bool) or not isinstance(bits, int):
            raise TypeError(f"bits must be int, got {type(bits).__name__}")
        if not 0 <= bits < (1 << fmt.width):
            raise ValueError(f"bits 0x{bits:x} do not fit in {fmt.width} bits")
        return cls._wrap(fmt, fmt.decode(bits))

    @classmethod
    def from_int(cls, fmt: IntFormat, value: int) -> Self:
        """Out of range is rejected: saturation is what the arithmetic does, not what construction does."""
        _check_int_format(fmt)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"value must be int, got {type(value).__name__}")
        if not fmt.fits(value):
            raise ValueError(f"{value} is out of range for {fmt}: [{fmt.min}, {fmt.max}]")
        return cls._wrap(fmt, value)

    @classmethod
    def from_float(cls, fmt: IntFormat, value: FloatValue, mode: RoundMode) -> Self:
        """Matches ``holoso_ftoint``: rounds by ``mode``, saturating at the rails, an infinity reaching one of them."""
        _check_int_format(fmt)
        if not isinstance(value, FloatValue):
            raise TypeError(f"value must be FloatValue, got {type(value).__name__}")
        return cls._wrap(fmt, _TO_INT[RoundMode(mode)](_zkf_format(value.fmt).wrap(value.bits), fmt.width))

    def to_float(self, fmt: FloatFormat) -> FloatValue:
        """Matches ``holoso_ffromint``: nearest, ties to even; a magnitude past the finite range becomes an infinity."""
        _check_format(fmt)
        return FloatValue.from_bits(fmt, _zkf_format(fmt).from_int(self.fmt.width, self.value).bits)

    @property
    def bits(self) -> int:
        return self.fmt.encode(self.value)

    def __int__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return repr(self.value)

    def __add__(self, other: IntValue) -> IntValue:
        _matching_int_format(self, other)
        return self._saturate(self.value + other.value)

    def __sub__(self, other: IntValue) -> IntValue:
        _matching_int_format(self, other)
        return self._saturate(self.value - other.value)

    def __mul__(self, other: IntValue) -> IntValue:
        _matching_int_format(self, other)
        return self._saturate(self.value * other.value)

    def __abs__(self) -> IntValue:
        return self._saturate(abs(self.value))

    def divmod_floor(self, other: IntValue) -> DivResult:
        """Python's ``//`` and ``%``; both degenerate cases answer as ``holoso_idivs`` does."""
        fmt = _matching_int_format(self, other)
        if other.value == 0:
            return DivResult(self._wrap(fmt, fmt.min if self.value < 0 else fmt.max), self._wrap(fmt, self.value))
        if self.value == fmt.min and other.value == -1:
            return DivResult(self._wrap(fmt, fmt.max), self._wrap(fmt, 0))
        quotient, remainder = divmod(self.value, other.value)
        assert fmt.fits(quotient) and fmt.fits(remainder)
        return DivResult(self._wrap(fmt, quotient), self._wrap(fmt, remainder))

    def shift(self, count: IntValue) -> ShiftResult:
        """Arithmetic shift, left for a positive count and right for a negative one, as ``holoso_ishift``."""
        fmt = _matching_int_format(self, count)
        if count.value < 0:  # Any amount past the word is indistinguishable from the word itself.
            shifted = self._wrap(fmt, self.value >> min(-count.value, fmt.width))
            return ShiftResult(shifted, shifted)
        exact = self.value << min(count.value, fmt.width)
        truncated = fmt.decode(exact & ((1 << fmt.width) - 1))
        return ShiftResult(self._wrap(fmt, truncated), self._wrap(fmt, fmt.saturate(exact)))

    # Bitwise combination cannot leave the range a two's-complement word already spans (``~min == max``), so these
    # wrap where the arithmetic operators saturate; ``_wrap``'s assertion is what makes that a checked claim.
    def __and__(self, other: IntValue) -> IntValue:
        return self._wrap(_matching_int_format(self, other), self.value & other.value)

    def __or__(self, other: IntValue) -> IntValue:
        return self._wrap(_matching_int_format(self, other), self.value | other.value)

    def __xor__(self, other: IntValue) -> IntValue:
        return self._wrap(_matching_int_format(self, other), self.value ^ other.value)

    def __invert__(self) -> IntValue:
        return self._wrap(self.fmt, ~self.value)

    def compare(self, other: IntValue) -> int:
        """-1/0/+1 three-way compare, the exact-arithmetic dual of :meth:`FloatValue.compare`."""
        _matching_int_format(self, other)
        return (self.value > other.value) - (self.value < other.value)

    def _saturate(self, exact: int) -> IntValue:
        return self._wrap(self.fmt, self.fmt.saturate(exact))

    @classmethod
    def _wrap(cls, fmt: IntFormat, value: int) -> Self:
        assert fmt.fits(value)
        self = object.__new__(cls)  # __init__ is disabled, so build the frozen instance directly
        object.__setattr__(self, "fmt", fmt)
        object.__setattr__(self, "value", value)
        return self


# The value dual of ScalarType: what a port of any scalar family may carry at run time.
type ScalarValue = FloatValue | IntValue | bool

# What the shared wide bank may hold: the two families that occupy a whole wide register, never the single-bit one.
type WideValue = FloatValue | IntValue


_TO_INT: dict[RoundMode, Callable[[zkf.Zkf, int], int]] = {
    RoundMode.NEAREST_EVEN: zkf.Zkf.round_int,
    RoundMode.FLOOR: zkf.Zkf.floor_int,
    RoundMode.CEIL: zkf.Zkf.ceil_int,
    RoundMode.TRUNC: zkf.Zkf.trunc_int,
}


def _zkf_format(fmt: FloatFormat) -> zkf.ZkfFormat:
    return zkf.ZkfFormat(fmt.wexp, fmt.wman)


def _check_format(fmt: FloatFormat) -> None:
    if not isinstance(fmt, FloatFormat):
        raise TypeError(f"fmt must be FloatFormat, got {type(fmt).__name__}")


def _check_int_format(fmt: IntFormat) -> None:
    if not isinstance(fmt, IntFormat):
        raise TypeError(f"fmt must be IntFormat, got {type(fmt).__name__}")


def _matching_int_format(a: IntValue, b: IntValue) -> IntFormat:
    if not isinstance(a, IntValue):
        raise TypeError(f"left operand must be IntValue, got {type(a).__name__}")
    if not isinstance(b, IntValue):
        raise TypeError(f"right operand must be IntValue, got {type(b).__name__}")
    if a.fmt != b.fmt:
        raise ValueError(f"operand format mismatch: {a.fmt} != {b.fmt}")
    return a.fmt


def _matching_format(a: FloatValue, b: FloatValue) -> FloatFormat:
    if not isinstance(a, FloatValue):
        raise TypeError(f"left operand must be FloatValue, got {type(a).__name__}")
    if not isinstance(b, FloatValue):
        raise TypeError(f"right operand must be FloatValue, got {type(b).__name__}")
    if a.fmt != b.fmt:
        raise ValueError(f"operand format mismatch: {a.fmt} != {b.fmt}")
    return a.fmt
