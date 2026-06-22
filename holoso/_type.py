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
            return self._pack(sign, self._exp_inf, 0)
        return self.pack(Fraction(value))

    def pack(self, value: Fraction) -> int:
        """Round a signed exact rational to the nearest float (round-to-nearest, ties to even) and assemble it."""
        if value == 0:
            return 0  # canonical positive zero
        sign = 1 if value < 0 else 0
        magnitude = abs(value)
        wfrac = self._wfrac
        exp = _floor_log2(magnitude)
        min_exp = 1 - self._bias
        max_exp = self._exp_inf - 1 - self._bias
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
            return self._pack(sign, self._exp_inf, 0)
        return self._pack(sign, exp + self._bias, quotient & ((1 << wfrac) - 1))

    def decode(self, bits: int) -> float:
        """Exactly decode a ZKF bit pattern into the Python float it represents."""
        wfrac = self._wfrac
        sign = (bits >> (self.width - 1)) & 1
        exp = (bits >> wfrac) & self._exp_inf
        frac = bits & ((1 << wfrac) - 1)
        if exp == 0:
            return 0.0
        if exp == self._exp_inf:
            return -math.inf if sign else math.inf
        significand = (1 << wfrac) | frac
        value = math.ldexp(float(significand), exp - self._bias - wfrac)
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
        wfrac = self._wfrac
        sign = (bits >> (self.width - 1)) & 1
        exp = (bits >> wfrac) & self._exp_inf
        frac = bits & ((1 << wfrac) - 1)
        if exp == 0:
            return frac == 0 and sign == 0  # only canonical +0
        return True

    def is_finite(self, bits: int) -> bool:
        """Whether ``bits`` encodes a finite value rather than an infinity."""
        return ((bits >> self._wfrac) & self._exp_inf) != self._exp_inf

    @property
    def _bias(self) -> int:
        return bias(self)

    @property
    def _wfrac(self) -> int:
        return wfrac(self)

    @property
    def _exp_inf(self) -> int:
        return exp_inf(self)

    def _pack(self, sign: int, exp: int, frac: int) -> int:
        """Simply assemble raw sign/exponent/fraction fields without rounding."""
        wfrac = self._wfrac
        return ((sign & 1) << (self.width - 1)) | ((exp & self._exp_inf) << wfrac) | (frac & ((1 << wfrac) - 1))


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


# ZKF field geometry -- the single source of truth, shared by ``FloatFormat`` (the codec) and ``_value`` (the
# arithmetic), so a layout change cannot drift between them. Package-internal (``_type`` is a private module, and these
# are not re-exported by ``holoso/__init__``), so they do not widen the public API.


def bias(fmt: FloatFormat) -> int:
    """The exponent bias of ``fmt``."""
    return (1 << (fmt.wexp - 1)) - 1


def wfrac(fmt: FloatFormat) -> int:
    """The number of stored fraction bits (the significand width minus the hidden leading bit)."""
    return fmt.wman - 1


def exp_inf(fmt: FloatFormat) -> int:
    """The all-ones biased exponent reserved for infinity."""
    return (1 << fmt.wexp) - 1


def exp_max_finite(fmt: FloatFormat) -> int:
    """The largest finite biased exponent (one below infinity)."""
    return exp_inf(fmt) - 1


def frac_mask(fmt: FloatFormat) -> int:
    """A mask selecting the stored fraction bits."""
    return (1 << wfrac(fmt)) - 1


def min_exp_unbiased(fmt: FloatFormat) -> int:
    """The smallest unbiased exponent of a normal value (ZKF has no subnormals)."""
    return 1 - bias(fmt)


def max_exp_unbiased(fmt: FloatFormat) -> int:
    """The largest unbiased exponent of a finite value."""
    return exp_max_finite(fmt) - bias(fmt)


def _floor_log2(value: Fraction) -> int:
    exp = value.numerator.bit_length() - value.denominator.bit_length()
    while pow2(exp + 1) <= value:
        exp += 1
    while pow2(exp) > value:
        exp -= 1
    return exp
