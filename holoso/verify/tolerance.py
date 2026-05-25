"""
The numerical tolerance model for equivalence checking.

The generated FSM is allowed to differ from the original Python in the last bits (ZKF rounding plus fast-math
reassociation), so equivalence is checked with a combined relative + absolute band. The relative term scales with the
format's unit roundoff and the operation count; the absolute term guards values near zero.
"""

import math

from ..format import FloatFormat


def unit_roundoff(fmt: FloatFormat) -> float:
    """The format's unit roundoff, ``2**-(wman-1)`` (relative spacing of representable values)."""
    return 2.0 ** -(fmt.wman - 1)


def default_tolerance(
    fmt: FloatFormat, op_count: int, magnitude: float = 1.0, rel_factor: float = 16.0
) -> tuple[float, float]:
    """A defensible (rtol, atol) for a kernel of ``op_count`` operations evaluated over operands up to ``magnitude``."""
    u = unit_roundoff(fmt)
    rtol = rel_factor * max(op_count, 1) * u
    atol = rtol * abs(magnitude)
    return rtol, atol


def within(actual: float, expected: float, rtol: float, atol: float) -> bool:
    """Whether ``actual`` is within ``atol + rtol*|expected|`` of ``expected`` (infinities must match exactly)."""
    if math.isinf(expected) or math.isinf(actual) or math.isnan(expected) or math.isnan(actual):
        return actual == expected
    return abs(actual - expected) <= atol + rtol * abs(expected)
