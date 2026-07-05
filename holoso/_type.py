"""Scalar data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ._zkf import ZkfFormat


@dataclass(frozen=True, slots=True)
class ScalarType(ABC):
    @property
    @abstractmethod
    def width(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ScalarSignature:
    """
    Operand- and result-port types for a concrete hardware operator. An operator may produce several results (e.g. a
    comparator's three one-hot order flags, a sorter's min and max), one per output port, each independently typed.
    """

    operand_types: tuple[ScalarType, ...]
    result_types: tuple[ScalarType, ...]

    @property
    def arity(self) -> int:
        return len(self.operand_types)


@dataclass(frozen=True, slots=True)
class FloatFormat:
    """
    A Zubax Kulibin float (ZKF) format: ``wexp`` exponent bits and ``wman`` significand bits.

    ``wman`` counts the significand *including* the hidden leading bit, matching the ``WMAN`` convention of
    ``holoso_support.v``. The total port width is ``wexp + wman`` (a sign bit, ``wexp`` exponent bits, and
    ``wman - 1`` stored significand bits).

    Engine-agnostic float facade: the codec delegates to the vendored bit-exact ZKF model, which is the single source
    of numeric truth. ``exp == 0`` is zero and the all-ones exponent is infinity; ZKF has no subnormals.
    """

    wexp: int
    wman: int

    def __post_init__(self) -> None:
        if self.wexp < 2:
            raise ValueError(f"wexp must be >= 2, got {self.wexp}")
        if self.wman < 4:
            raise ValueError(f"wman must be >= 4, got {self.wman}")

    @property
    def _zfmt(self) -> ZkfFormat:
        return ZkfFormat(self.wexp, self.wman)

    def encode(self, value: float) -> int:
        """NaN is rejected; ZKF has no NaN."""
        return self._zfmt.encode(value).bits

    def decode(self, bits: int) -> float:
        """
        The value as the nearest Python double, correctly rounded in a single step. Formats wider than IEEE double
        (``wman > 53``, reaching the double-subnormal range) round up to 1 ULP tighter than a naive ``ldexp`` decode
        that double-rounds; no float32/float64-class ZKF format reaches that regime, so this is invisible in practice.
        """
        return float(self._zfmt.wrap(bits))

    def round(self, value: float) -> float:
        """Rounds exactly as the hardware packer does after each operator. NaN is rejected (ZKF has no NaN)."""
        return float(self._zfmt.encode(value))

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
class FloatType(ScalarType):
    fmt: FloatFormat

    @property
    def width(self) -> int:
        return self.fmt.width

    def __str__(self) -> str:
        return f"float{self.fmt.width}"


@dataclass(frozen=True, slots=True)
class BoolType(ScalarType):
    """The storage type of branch conditions and boolean state."""

    @property
    def width(self) -> int:
        return 1

    def __str__(self) -> str:
        return "bool"


def is_wide_type(scalar_type: ScalarType) -> bool:
    """
    Whether ``scalar_type`` lives in the WIDE data register bank (as opposed to the 1-bit boolean bank): the single
    storage-bank predicate the timing model, the scheduler, and the backends share instead of open-coding
    ``isinstance(x, FloatType)`` at each site. Float is the only wide tenant today; a future fixed-width int joins it
    here, so a wide value is routed correctly everywhere without revisiting every dispatch.
    """
    return isinstance(scalar_type, FloatType)


@dataclass(frozen=True, slots=True)
class LogicalPort:
    """
    One logical I/O port of a synthesized kernel: a parameter or output name paired with its scalar type. Both oracles
    speak this signature -- the numerical model and the MIR interpreter expose their inputs/outputs as these, so the two
    are directly comparable. Distinct from the RTL data ports, which carry a port-name prefix and explicit direction;
    here the name is the logical one (as written in the kernel) and direction is implicit in the inputs/outputs split.
    """

    name: str
    scalar_type: ScalarType
