"""Runtime values and exact arithmetic for Zubax Kulibin float."""

from dataclasses import dataclass
from typing import Self

from ._type import FloatFormat
from ._zkf import Zkf, ZkfFormat


@dataclass(frozen=True, slots=True, init=False)
class FloatValue:
    """
    A concrete ZKF value: a format plus the exact bits carried by hardware.
    ``__eq__`` stays the dataclass's structural ``(fmt, bits)`` equality (used for hashing/interning); numeric
    ordering goes through :meth:`compare`. Every operation delegates to the vendored bit-exact ZKF model, so the
    results match the ``zkf_*`` RTL bit-for-bit.
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

    def _zval(self) -> Zkf:
        return ZkfFormat(self.fmt.wexp, self.fmt.wman).wrap(self.bits)

    def __float__(self) -> float:
        return self.fmt.decode(self.bits)

    @property
    def negative(self) -> bool:
        return self._zval().negative

    @property
    def exponent(self) -> int:
        return self._zval().exp

    def apply_sign(self, *, negate: bool, absolute: bool) -> "FloatValue":
        """Apply the sign conditioner of ``holoso_fsgnop``: absolute value first, then optional negation."""
        value = self._zval()
        if absolute:
            value = abs(value)
        if negate:
            value = -value
        return FloatValue.from_bits(self.fmt, value.bits)

    def __add__(self, other: "FloatValue") -> "FloatValue":
        fmt = _matching_format(self, other)
        return FloatValue.from_bits(fmt, (self._zval() + other._zval()).bits)

    def __mul__(self, other: "FloatValue") -> "FloatValue":
        fmt = _matching_format(self, other)
        return FloatValue.from_bits(fmt, (self._zval() * other._zval()).bits)

    def __truediv__(self, other: "FloatValue") -> "FloatValue":
        """``zkf_div``'s error sidebands are intentionally not modeled."""
        fmt = _matching_format(self, other)
        return FloatValue.from_bits(fmt, (self._zval() / other._zval()).bits)

    def compare(self, other: "FloatValue") -> int:
        """
        -1 if ``self < other``, 0 if equal, +1 if ``self > other``. ZKF has no NaN (the order is total) and no
        negative zero. A method rather than overloaded ordering operators: ``__eq__`` is the frozen-dataclass
        structural (bit) equality used for hashing, and numeric ordering would be inconsistent with it where equal
        values differ in bits, so the relations could not form a coherent order with ``==``; one three-way result
        also mirrors the single-firing hardware comparator.
        """
        _matching_format(self, other)
        result = self._zval().cmp(other._zval())
        return result.gt - result.lt

    def scale_pow2(self, k: int) -> "FloatValue":
        """Matches ``zkf_mul_ilog2_const``."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be int, got {type(k).__name__}")
        return FloatValue.from_bits(self.fmt, self._zval().mul_ilog2(k).bits)

    @staticmethod
    def fma(a: "FloatValue", b: "FloatValue", c: "FloatValue") -> "FloatValue":
        """Fused multiply-add ``a*b + c``, rounded once (ties to even)."""
        fmt = _matching_format(a, b)
        if not isinstance(c, FloatValue):
            raise TypeError(f"fma addend must be FloatValue, got {type(c).__name__}")
        if c.fmt != fmt:
            raise ValueError(f"operand format mismatch: {c.fmt} != {fmt}")
        return FloatValue.from_bits(fmt, a._zval().fma(b._zval(), c._zval()).bits)

    @staticmethod
    def sort(a: "FloatValue", b: "FloatValue") -> tuple["FloatValue", "FloatValue"]:
        """
        Ascending as ``(min, max)``: a bit-preserving selection by the total order of :meth:`compare` (ZKF has no
        NaN). Equal values are bit-identical, so the tie direction is unobservable.
        """
        fmt = _matching_format(a, b)
        lo, hi = a._zval().sort(b._zval())
        return FloatValue.from_bits(fmt, lo.bits), FloatValue.from_bits(fmt, hi.bits)

    def round(self) -> "FloatValue":
        return FloatValue.from_bits(self.fmt, self._zval().round().bits)

    def floor(self) -> "FloatValue":
        return FloatValue.from_bits(self.fmt, self._zval().floor().bits)

    def ceil(self) -> "FloatValue":
        return FloatValue.from_bits(self.fmt, self._zval().ceil().bits)

    def trunc(self) -> "FloatValue":
        return FloatValue.from_bits(self.fmt, self._zval().trunc().bits)

    def exp2(self) -> "FloatValue":
        return FloatValue.from_bits(self.fmt, self._zval().exp2().bits)

    def log2(self) -> "FloatValue":
        """``zkf_log2``'s domain-error/pole sidebands are intentionally not modeled (as with ``zkf_div``'s div0)."""
        return FloatValue.from_bits(self.fmt, self._zval().log2().value.bits)


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
