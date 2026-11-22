"""Runtime values: a float is ZKF's own value; the native saturating integer has its exact arithmetic here."""

from dataclasses import dataclass
from typing import NamedTuple, Self

from zkf import RoundMode as RoundMode, Zkf

from ._type import BoolType, FloatType, IntFormat, IntType, ScalarType

FloatValue = Zkf
"""A float value is ZKF's own: its format and its bits."""


class DivResult(NamedTuple):
    quo: IntValue
    rem: IntValue


@dataclass(frozen=True, slots=True, init=False, repr=False)
class IntValue:
    """
    A concrete native integer, signed and saturating: the arithmetic the integer operators compute, edge cases
    included: `abs(MIN)` is `MAX`, `MIN // -1` is `(MAX, 0)`, and a division by zero answers a rail by the
    numerator's sign while keeping the numerator as the remainder.
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

    @property
    def bits(self) -> int:
        return self.fmt.encode(self.value)

    def __int__(self) -> int:
        return self.value

    def __float__(self) -> float:
        return float(self.value)

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
        """Python's `//` and `%`; both degenerate cases answer as the integer divider does."""
        fmt = _matching_int_format(self, other)
        if other.value == 0:
            return DivResult(self._wrap(fmt, fmt.min if self.value < 0 else fmt.max), self._wrap(fmt, self.value))
        if self.value == fmt.min and other.value == -1:
            return DivResult(self._wrap(fmt, fmt.max), self._wrap(fmt, 0))
        quotient, remainder = divmod(self.value, other.value)
        assert fmt.fits(quotient) and fmt.fits(remainder)
        return DivResult(self._wrap(fmt, quotient), self._wrap(fmt, remainder))

    def shift_left(self, count: IntValue) -> IntValue:
        """
        Arithmetic shift, left for a positive count and right for a negative one, as the shifter with `right` low.
        It is the raw bit shift: what a left shift pushes past the word is dropped, so nothing saturates.
        """
        fmt = _matching_int_format(self, count)
        magnitude = min(abs(count.value), fmt.width)  # any amount past the word is the word itself
        if count.value >= 0:
            return self._wrap(fmt, fmt.decode((self.value << magnitude) & ((1 << fmt.width) - 1)))
        return self._wrap(fmt, self.value >> magnitude)

    def shift_right(self, count: IntValue) -> IntValue:
        """The mirror of shift_left, as the shifter with `right` high: right for a positive count, left otherwise."""
        fmt = _matching_int_format(self, count)
        magnitude = min(abs(count.value), fmt.width)  # any amount past the word is the word itself
        if count.value >= 0:
            return self._wrap(fmt, self.value >> magnitude)
        return self._wrap(fmt, fmt.decode((self.value << magnitude) & ((1 << fmt.width) - 1)))

    # Bitwise combination cannot leave the range a two's-complement word already spans (`~min == max`), so these
    # wrap where the arithmetic operators saturate; `_wrap`'s assertion is what makes that a checked claim.
    def __and__(self, other: IntValue) -> IntValue:
        return self._wrap(_matching_int_format(self, other), self.value & other.value)

    def __or__(self, other: IntValue) -> IntValue:
        return self._wrap(_matching_int_format(self, other), self.value | other.value)

    def __xor__(self, other: IntValue) -> IntValue:
        return self._wrap(_matching_int_format(self, other), self.value ^ other.value)

    def __invert__(self) -> IntValue:
        return self._wrap(self.fmt, ~self.value)

    def compare(self, other: IntValue) -> int:
        """-1/0/+1 three-way compare."""
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


def type_of(value: WideValue) -> FloatType | IntType:
    """The scalar type a wide value is of, which is where its width is read."""
    return FloatType(value.fmt) if isinstance(value, FloatValue) else IntType(value.fmt)


# What coerce_scalar admits: a typed scalar, or the plain Python value the port's family encodes.
type ScalarLike = ScalarValue | float | int


def coerce_scalar(scalar_type: ScalarType, value: object, what: str) -> ScalarValue:
    """
    The one scalar port codec: admit a value onto a port of the given family, accepting the family's own value type
    (format-checked) or the plain Python type it encodes. Exact-type checks keep a bool off an integer port.
    """
    match scalar_type:
        case FloatType(fmt=ffmt):
            if isinstance(value, FloatValue):
                if value.fmt != ffmt:
                    raise ValueError(f"{what} has {value.fmt}, expected {ffmt}")
                return value
            if type(value) is float:
                try:
                    return ffmt.encode(value)
                except ValueError as error:
                    raise ValueError(f"{what}: {error}") from None
            raise TypeError(f"{what} must be FloatValue or float, got {type(value).__name__}")
        case IntType(fmt=ifmt):
            if isinstance(value, IntValue):
                if value.fmt != ifmt:
                    raise ValueError(f"{what} has {value.fmt}, expected {ifmt}")
                return value
            if type(value) is int:
                try:
                    return IntValue.from_int(ifmt, value)
                except ValueError as error:
                    raise ValueError(f"{what}: {error}") from None
            raise TypeError(f"{what} must be IntValue or int, got {type(value).__name__}")
        case BoolType():
            if type(value) is bool:
                return value
            raise TypeError(f"{what} must be bool, got {type(value).__name__}")
        case _:
            assert False, f"no scalar family for {scalar_type!r}"


def _check_int_format(fmt: IntFormat) -> None:
    if not isinstance(fmt, IntFormat):
        raise TypeError(f"fmt must be IntFormat, got {type(fmt).__name__}")


def _matching_int_format(a: IntValue, b: IntValue) -> IntFormat:
    assert isinstance(a, IntValue), a
    if not isinstance(b, IntValue):
        raise TypeError(f"right operand must be IntValue, got {type(b).__name__}")
    if a.fmt != b.fmt:
        raise ValueError(f"operand format mismatch: {a.fmt} != {b.fmt}")
    return a.fmt
