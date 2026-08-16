"""
A constant multiplier read as sign, significand and exponent. Shared, because the layer that composes scalings and
the layer that materializes them must agree on what one is.
"""

import math
import sys
from dataclasses import dataclass


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

    @property
    def signed_significand(self) -> float:
        return -self.significand if self.negative else self.significand

    def coefficient(self) -> float | None:
        """The one constant this scaling multiplies by, or None where no host float names it."""
        try:
            product = math.ldexp(self.signed_significand, self.k)
        except OverflowError:
            return None
        return product if sys.float_info.min <= abs(product) < math.inf else None

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
