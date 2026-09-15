"""
A constant multiplier read as sign, significand and exponent, and the one HIR shape it takes over a value, shared by the
reducer, the linear pass and the reader.
"""

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

from .._util import ValueId
from ._ir import Hir, Operation
from ._operators import FloatMul, FloatMulPow2, FloatNeg


@dataclass(frozen=True, slots=True)
class Identity:
    """The value itself, standing as no node at all."""


@dataclass(frozen=True, slots=True)
class Exponent:
    """`k` is never zero, and no host float is asked whatever it is."""

    k: int


@dataclass(frozen=True, slots=True)
class Product:
    """The sign is peeled into a negation over the product, so `x*3.0` and `x*-3.0` share one multiply."""

    magnitude: float


type Magnitude = Identity | Exponent | Product


@dataclass(frozen=True, slots=True)
class Rendering:
    """The negation is a sideband the reader folds; the identity is no node at all."""

    magnitude: Magnitude
    negative: bool


@dataclass(frozen=True, slots=True)
class Scaling:
    """
    Held apart, the three parts compose without ever leaving the range the compiler's own arithmetic can carry;
    multiplied together they may not be a number at all, which is why the product is formed once, by whoever
    materializes it, and declined there rather than guarded against at every composition.

    The significand lies in `[1, 2)`, which every float format represents exactly, so a scaling that cannot be
    one constant can always be two.
    """

    significand: float
    k: int
    negative: bool

    def __post_init__(self) -> None:
        assert 1.0 <= self.significand < 2.0

    @property
    def is_power_of_two(self) -> bool:
        return self.significand == 1.0

    def magnitude(self) -> float | None:
        """None where no NORMAL host float names it."""
        try:
            product = math.ldexp(self.significand, self.k)
        except OverflowError:
            return None
        return product if sys.float_info.min <= product < math.inf else None

    def coefficient(self) -> float | None:
        """None where no host float names it."""
        magnitude = self.magnitude()
        return None if magnitude is None else (-magnitude if self.negative else magnitude)

    def ratio(self) -> Fraction:
        """Exact, as the significand is a binary fraction."""
        magnitude = Fraction(self.significand) * Fraction(2) ** self.k
        return -magnitude if self.negative else magnitude

    def rendering(self) -> Rendering | None:
        """
        None where it is a product no host float names; an exponent never asks the host, which is what lets
        `x * 2**1200` compose at any width.
        """
        if self.is_power_of_two:
            return Rendering(Identity() if self.k == 0 else Exponent(self.k), self.negative)
        magnitude = self.magnitude()
        return None if magnitude is None else Rendering(Product(magnitude), self.negative)

    def compose(self, other: "Scaling") -> "Scaling":
        """Fuse two scalings of one value into the one they amount to; total, hence no failure to answer for."""
        significand, k = self.significand * other.significand, self.k + other.k
        if significand >= 2.0:  # the product lies in [1, 4), so at most one exact halving renormalizes it
            significand, k = significand * 0.5, k + 1
        return Scaling(significand, k, self.negative != other.negative)


def scaling_of(c: float) -> Scaling | None:
    """
    A constant read as a scaling. Zero and the infinities are excluded: they are the absorbing elements of
    multiplication rather than scalings of it, and `frexp` cannot normalize them, so composition is total only
    over what this admits.
    """
    if c == 0.0 or not math.isfinite(c):
        return None
    fraction, exponent = math.frexp(abs(c))  # in [0.5, 1), so one doubling puts the significand in [1, 2)
    return Scaling(2.0 * fraction, exponent - 1, c < 0.0)


def rendering_of(c: float) -> Rendering | None:
    """
    A written constant's shape over a value: a literal is exact as written, so a subnormal keeps its magnitude; zero and
    the infinities render as nothing since they absorb rather than scale.
    """
    scaling = scaling_of(c)
    if scaling is None:
        return None
    rendering = scaling.rendering()
    return Rendering(Product(abs(c)), c < 0.0) if rendering is None else rendering


def scaling_of_ratio(ratio: Fraction) -> Scaling | None:
    """
    Plus or minus a power of two is read exactly off the numerator and the denominator, never through a host float; any
    other ratio must be a host float, and a ratio that underflows on the way to one is refused rather than answered 0.
    """
    assert ratio
    numerator, denominator = abs(ratio.numerator), ratio.denominator
    if numerator & (numerator - 1) == 0 and denominator & (denominator - 1) == 0:
        return Scaling(1.0, numerator.bit_length() - denominator.bit_length(), ratio < 0)
    try:
        return scaling_of(float(ratio))
    except OverflowError:
        return None


def scaled_node(magnitude: Magnitude, base: ValueId, constant: Callable[[float], ValueId]) -> Operation | None:
    """None for the identity, which stands as no node at all."""
    match magnitude:
        case Identity():
            return None
        case Exponent(k=k):
            return Operation(FloatMulPow2(k), (base,))
        case Product(magnitude=value):
            return Operation(FloatMul(), (base, constant(value)))


@dataclass(frozen=True, slots=True)
class Reading:
    """`layers` are outermost first."""

    base: ValueId
    scaling: Scaling
    layers: tuple[ValueId, ...]


def scaling_layer(
    hir: Hir, vid: ValueId, constant: Callable[[ValueId], float | None]
) -> tuple[ValueId, Scaling] | None:
    """The one constant scaling `vid` applies directly to its operand, which it also names."""
    match hir.nodes[vid]:
        case Operation(operator=FloatMulPow2(k=k), operands=(x,)):
            return x, Scaling(1.0, k, False)
        case Operation(operator=FloatNeg(), operands=(x,)):
            return x, Scaling(1.0, 0, True)
        case Operation(operator=FloatMul(), operands=(x, y)):
            for base, other in ((x, y), (y, x)):
                factor = constant(other)
                if factor is not None and (found := scaling_of(factor)) is not None:
                    return base, found
    return None


def read_scaling(
    hir: Hir,
    vid: ValueId,
    constant: Callable[[ValueId], float | None],
    exclusive: Callable[[ValueId], bool] = lambda _: True,
) -> Reading | None:
    """
    Composition stops at the first layer whose composition with what lies above it no host float names, the boundary the
    reducer itself leaves; that layer is the base. `constant` says what a node is worth as a number, so a caller can
    read the shape off the old graph and constant-ness off the new.
    """
    layers: list[ValueId] = []
    total = Scaling(1.0, 0, False)
    current = vid
    while (layer := scaling_layer(hir, current, constant)) is not None and exclusive(current):
        operand, scaling = layer
        composed = total.compose(scaling)
        if layers and composed.rendering() is None:
            break
        layers.append(current)
        total, current = composed, operand
    return Reading(current, total, tuple(layers)) if layers else None
