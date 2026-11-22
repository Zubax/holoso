"""Scalar data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from zkf import ZkfFormat

FloatFormat = ZkfFormat
"""A float format is ZKF's own: `wexp` exponent bits and `wman` significand bits, the hidden one included."""


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
        return self.fmt.wfull

    def __str__(self) -> str:
        return f"float{self.fmt.wfull}"


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
