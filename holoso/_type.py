"""Scalar data types."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import zkf


@dataclass(frozen=True, slots=True)
class ScalarType(ABC):
    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    def is_wide(self) -> bool:
        """In the wide data register bank, not the 1-bit boolean bank."""
        return self.width > 1


@dataclass(frozen=True, slots=True)
class FloatFormat:
    """
    A Zubax Kulibin float (ZKF) format: `wexp` exponent bits and `wman` significand bits.

    `wman` counts the significand *including* the hidden leading bit, matching the `WMAN` convention of
    `holoso_support.v`. The total port width is `wexp + wman` (a sign bit, `wexp` exponent bits, and
    `wman - 1` stored significand bits).

    Engine-agnostic float facade: the codec delegates to the bit-exact ZKF model, which is the single source
    of numeric truth. `exp == 0` is zero and the all-ones exponent is infinity; ZKF has no subnormals.
    """

    wexp: int
    wman: int

    def __post_init__(self) -> None:
        if self.wexp < 2:
            raise ValueError(f"wexp must be >= 2, got {self.wexp}")
        if self.wman < 4:
            raise ValueError(f"wman must be >= 4, got {self.wman}")

    @property
    def _zfmt(self) -> zkf.ZkfFormat:
        return zkf.ZkfFormat(self.wexp, self.wman)

    def encode(self, value: float) -> int:
        """NaN is rejected; ZKF has no NaN."""
        return self._zfmt.encode(value).bits

    def decode(self, bits: int) -> float:
        """
        The value as the nearest Python double, correctly rounded in a single step. Formats wider than IEEE double
        (`wman > 53`, reaching the double-subnormal range) round up to 1 ULP tighter than a naive `ldexp` decode
        that double-rounds; no float32/float64-class ZKF format reaches that regime, so this is invisible in practice.
        A finite value past the double range saturates to an infinity, which is what correct rounding means there.
        """
        value = self._zfmt.wrap(bits)
        try:
            return float(value)
        except OverflowError:
            return -math.inf if value.negative else math.inf

    def round(self, value: float) -> float:
        """Rounds exactly as the hardware packer does after each operator. NaN is rejected (ZKF has no NaN)."""
        return self.decode(self.encode(value))

    def is_legal(self, bits: int) -> bool:
        """Rejects subnormals and negative zero."""
        value = self._zfmt.wrap(bits)
        return (value.frac == 0 and not value.negative) if value.is_zero else True

    def is_finite(self, bits: int) -> bool:
        return self._zfmt.wrap(bits).is_finite

    @property
    def width(self) -> int:
        return self.wexp + self.wman


@dataclass(frozen=True, slots=True)
class IntFormat:
    """
    The native signed integer format: `width` bits in two's complement. Arithmetic saturates at the extremes
    rather than wrapping, so the representable range is exactly `[min, max]`.
    """

    width: int

    def __post_init__(self) -> None:
        if self.width < 2:
            raise ValueError(f"width must be >= 2, got {self.width}")

    @property
    def min(self) -> int:
        return -(1 << (self.width - 1))

    @property
    def max(self) -> int:
        return (1 << (self.width - 1)) - 1

    def encode(self, value: int) -> int:
        """An out-of-range value is rejected; pass it through saturate first if clamping is the intent."""
        if not self.fits(value):
            raise ValueError(f"{value} is out of range for {self}: [{self.min}, {self.max}]")
        bits = value & ((1 << self.width) - 1)
        assert self.decode(bits) == value
        return bits

    def decode(self, bits: int) -> int:
        assert 0 <= bits < (1 << self.width)
        value = bits - (1 << self.width) if bits > self.max else bits
        assert self.fits(value)
        return value

    def saturate(self, value: int) -> int:
        return min(max(value, self.min), self.max)

    def fits(self, value: int) -> bool:
        return self.min <= value <= self.max

    def __str__(self) -> str:
        return f"int{self.width}"


@dataclass(frozen=True, slots=True)
class FloatType(ScalarType):
    fmt: FloatFormat

    @property
    def width(self) -> int:
        return self.fmt.width

    def __str__(self) -> str:
        return f"float{self.fmt.width}"


@dataclass(frozen=True, slots=True)
class IntType(ScalarType):
    fmt: IntFormat

    @property
    def width(self) -> int:
        return self.fmt.width

    def __str__(self) -> str:
        return str(self.fmt)


@dataclass(frozen=True, slots=True)
class BoolType(ScalarType):
    """The storage type of branch conditions and boolean state."""

    @property
    def width(self) -> int:
        return 1

    def __str__(self) -> str:
        return "bool"


@dataclass(frozen=True, slots=True)
class LogicalPort:
    """
    One logical I/O port of a synthesized kernel: a parameter or output name paired with its scalar type. Both oracles
    speak this signature -- the numerical model and the MIR interpreter expose their inputs/outputs as these, so the two
    are directly comparable. Distinct from the RTL data ports, which carry a port-name prefix and explicit direction;
    here the name carries no port-name prefix and direction is implicit in the inputs/outputs split. It is the name
    the hardware carries, not necessarily the Python one: a kernel's Greek identifiers are spelled out on the way in.
    """

    name: str
    scalar_type: ScalarType
