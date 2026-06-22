"""Scalar data types and the Zubax Kulibin float format."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class ScalarType(ABC):
    """A scalar value type carried by a data port or an internal typed storage resource."""

    @property
    @abstractmethod
    def width(self) -> int:
        """The number of bits needed to represent one scalar value."""


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

    ``encode``/``decode`` are the bit-exact round-trip codec between Python floats and ZKF bit patterns. The layout is
    ``[sign | exponent(wexp) | stored-fraction(wman-1)]`` with a hidden leading significand bit: ``exp == 0`` is zero
    (ZKF has no subnormals and no negative zero) and the all-ones exponent is infinity. Encoding rounds to nearest,
    ties to even, using exact rational arithmetic.
    """

    wexp: int
    wman: int

    def __post_init__(self) -> None:
        if self.wexp < 2:
            raise ValueError(f"wexp must be >= 2, got {self.wexp}")
        if self.wman < 4:
            raise ValueError(f"wman must be >= 4, got {self.wman}")

    @property
    def width(self) -> int:
        """Total bit width of one scalar (``WFULL = wexp + wman``)."""
        return self.wexp + self.wman

    def encode(self, value: float) -> int:
        """Encode a Python float to the nearest ZKF bit pattern (ties to even). NaN is rejected; ZKF has no NaN."""
        if math.isnan(value):
            raise ValueError("NaN is not representable in ZKF")
        sign = 1 if math.copysign(1.0, value) < 0 else 0
        magnitude = abs(value)
        if magnitude == 0.0:
            return 0  # canonical positive zero
        if math.isinf(magnitude):
            return self._pack(sign, self.exp_inf, 0)
        return self.pack(Fraction(value))

    def pack(self, value: Fraction) -> int:
        """Round a signed exact rational to the nearest float (round-to-nearest, ties to even) and assemble it."""
        if value == 0:
            return 0  # canonical positive zero
        sign = 1 if value < 0 else 0
        magnitude = abs(value)
        wfrac = self.wfrac
        exp = _floor_log2(magnitude)
        min_exp = 1 - self.bias
        max_exp = self.exp_inf - 1 - self.bias
        if exp < min_exp:  # Underflow: round against the half-MIN_NORMAL boundary (no subnormals).
            return self._pack(sign, 1, 0) if magnitude >= pow2(min_exp - 1) else 0
        scaled = magnitude / pow2(exp) * (1 << wfrac)
        quotient, remainder = divmod(scaled.numerator, scaled.denominator)
        twice = 2 * remainder
        if twice > scaled.denominator or (twice == scaled.denominator and (quotient & 1)):
            quotient += 1
        if quotient >= (1 << self.wman):
            quotient >>= 1
            exp += 1
        if exp > max_exp:
            return self._pack(sign, self.exp_inf, 0)
        return self._pack(sign, exp + self.bias, quotient & self.frac_mask)

    def decode(self, bits: int) -> float:
        """Exactly decode a ZKF bit pattern into the Python float it represents."""
        wfrac = self.wfrac
        sign = (bits >> (self.width - 1)) & 1
        exp = (bits >> wfrac) & self.exp_inf
        frac = bits & self.frac_mask
        if exp == 0:
            return 0.0
        if exp == self.exp_inf:
            return -math.inf if sign else math.inf
        significand = (1 << wfrac) | frac
        value = math.ldexp(float(significand), exp - self.bias - wfrac)
        return -value if sign else value

    def round(self, value: float) -> float:
        """
        Snap a real-valued result to the nearest value representable in this format (round-to-nearest, ties to even),
        exactly as the hardware packer rounds after each operator. NaN may be rejected depending on the format
        (e.g., ZKF has no NaN).
        """
        return self.decode(self.encode(value))

    def is_legal(self, bits: int) -> bool:
        """Whether ``bits`` is a legal ZKF value (rejects subnormals and negative zero)."""
        wfrac = self.wfrac
        sign = (bits >> (self.width - 1)) & 1
        exp = (bits >> wfrac) & self.exp_inf
        frac = bits & self.frac_mask
        if exp == 0:
            return frac == 0 and sign == 0  # only canonical +0
        return True

    def is_finite(self, bits: int) -> bool:
        """Whether ``bits`` encodes a finite value rather than an infinity."""
        return ((bits >> self.wfrac) & self.exp_inf) != self.exp_inf

    @property
    def bias(self) -> int:
        """The exponent bias."""
        return (1 << (self.wexp - 1)) - 1

    @property
    def wfrac(self) -> int:
        """The number of stored fraction bits (the significand width minus the hidden leading bit)."""
        return self.wman - 1

    @property
    def exp_inf(self) -> int:
        """The all-ones biased exponent reserved for infinity."""
        return (1 << self.wexp) - 1

    @property
    def exp_max_finite(self) -> int:
        """The largest finite biased exponent (one below infinity)."""
        return self.exp_inf - 1

    @property
    def frac_mask(self) -> int:
        """A mask selecting the stored fraction bits."""
        return (1 << self.wfrac) - 1

    @property
    def min_exp_unbiased(self) -> int:
        """The smallest unbiased exponent of a normal value (ZKF has no subnormals)."""
        return 1 - self.bias

    @property
    def max_exp_unbiased(self) -> int:
        """The largest unbiased exponent of a finite value."""
        return self.exp_max_finite - self.bias

    def _pack(self, sign: int, exp: int, frac: int) -> int:
        """Simply assemble raw sign/exponent/fraction fields without rounding."""
        wfrac = self.wfrac
        return ((sign & 1) << (self.width - 1)) | ((exp & self.exp_inf) << wfrac) | (frac & self.frac_mask)


@dataclass(frozen=True, slots=True)
class FloatType(ScalarType):
    """A ZKF floating-point scalar with the given bit-exact format."""

    fmt: FloatFormat

    @property
    def width(self) -> int:
        return self.fmt.width

    def __str__(self) -> str:
        return f"float{self.fmt.width}"


@dataclass(frozen=True, slots=True)
class BoolType(ScalarType):
    """A single-bit boolean scalar; the storage type of branch conditions and boolean state."""

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


def pow2(exp: int) -> Fraction:
    """``2**exp`` as an exact rational, for either sign of ``exp``. Shared by the codec and the value arithmetic."""
    return Fraction(1 << exp, 1) if exp >= 0 else Fraction(1, 1 << -exp)


def _floor_log2(value: Fraction) -> int:
    exp = value.numerator.bit_length() - value.denominator.bit_length()
    while pow2(exp + 1) <= value:
        exp += 1
    while pow2(exp) > value:
        exp -= 1
    return exp
