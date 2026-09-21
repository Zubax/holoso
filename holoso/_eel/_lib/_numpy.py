"""
Composite math/numpy stubs the frontend inlines like ordinary user functions, expressed via the intrinsics,
that is, bare-hardware operators. Fast-math transforms are legal here, which may alter semantics.

Math domain violations raise in plain Python but garbage-in-garbage-out in hardware,
with the error output signals asserted; refer to the operator RTL for details. A violation the compiler happens to
prove statically is refused at build time instead -- a liberty it takes where a fold reaches, never a guarantee.

Composites compute through the intrinsic stubs (exp2, atan2, ...) instead of the library functions directly even
though the library spellings would lower identically (they select the same lowerings). `clip` is the
exception: its bounds are arrays as readily as scalars, so it composes `np.maximum`/`np.minimum`, which map over one.
The intrinsic stub pins each primitive to the numpy/math variant matching the hardware behavior -- e.g. exp2 saturates
to inf like the hardware where math.exp2 would raise -- so a composite built on the stubs inherits that
hardware-faithful behavior, and its plain-Python run uses exactly the primitives (and fast-math choices) it lowers to.
"""

import math
from typing import Any

import numpy as np

from ._intrinsics import atan2, ceil, cos, exp2, floor, isinf, log2, round_, sin, sqrt
from ._registry import array, lib

_LOG2E = math.log2(math.e)
_LN2 = math.log(2.0)
_LOG10_2 = math.log10(2.0)
_DEG_PER_RAD = 180.0 / math.pi
_RAD_PER_DEG = math.pi / 180.0


@lib
def identity_bool(x: bool) -> bool:
    return x


@lib
def identity_int(x: int) -> int:
    """The identity every spelling whose own answer on an integer IS that integer lowers to, unary plus included."""
    return x


# A float consumer reduces the cast back to the bare rounder.
@lib
def floor_to_int(x: float) -> int:
    return int(floor(x))


@lib
def ceil_to_int(x: float) -> int:
    return int(ceil(x))


@lib
def round_to_int(x: float) -> int:
    return int(round_(x))


@lib
def min_int(a: int, b: int) -> int:
    """There is no hardware min/max operator for integers."""
    return a if a <= b else b  # the tie answers a, as CPython's min does


@lib
def max_int(a: int, b: int) -> int:
    return a if b <= a else b


@lib
def subtract_float(a: float, b: float) -> float:
    """HIR has no float subtraction."""
    return a + -b


@lib
def identity_float(x: float) -> float:
    return x


@lib
def equal_bool(a: bool, b: bool) -> bool:
    """HIR has no boolean equality. The body avoids `==`, which would resolve back to this meaning."""
    return not (a != b)


@lib
def square_int(x: int) -> int:
    return x * x


@lib
def square_float(x: float) -> float:
    return x * x


@array(np.ndarray.clip)
def clip(x: np.ndarray, lo: Any = None, hi: Any = None) -> Any:
    """
    Saturation as the maximum/minimum composition, following numpy: the lower bound applies first, so an inverted
    pair answers `hi`, and an omitted bound skips that side (the subset admits no `None` argument, so one-sided
    saturation from above is spelled `np.minimum`). The bounds broadcast as the extrema do, which on the host
    reaches further than the compiler admits, so this stub is a reference for the values and not for the shapes.
    """
    r = x
    if lo is not None:
        r = np.maximum(r, lo)
    if hi is not None:
        r = np.minimum(r, hi)
    return r


@array(np.clip)
def clip_free(x: np.ndarray, lo: Any, hi: Any) -> Any:
    """The free function demands both bounds -- numpy raises on a lone `a_min`, where the method accepts one."""
    return clip(x, lo, hi)


@array(np.polyval, sequences=(0,))
def polyval(p: np.ndarray, x: Any) -> Any:
    """
    numpy's own shape: convert the coefficients first (so a heterogeneous sequence promotes BEFORE the fold,
    never saturating an integer partial product), then Horner over a zero seed -- which also answers the
    empty polynomial and broadcasts a short one over an array x. A 2-D p against a broadcastable array x is
    refused where numpy row-broadcasts (a rank guard cannot be spelled over a sequence p).
    """
    acc = 0 * x
    if len(p) > 0:
        for c in np.asarray(p):
            acc = acc * x + c
    return acc


@lib
def sign_int(x: int) -> int:
    if x > 0:
        r = 1
    elif x < 0:
        r = -1
    else:
        r = 0
    return r


@lib
def sign_float(x: float) -> float:
    if x > 0.0:
        r = +1.0
    elif x < 0.0:
        r = -1.0
    else:
        r = x
    return r


@lib
def cbrt(x: float) -> float:
    """Fastmath: cbrt(−0.0) may return +0.0"""
    return sign_float(x) * exp2(log2(abs(x)) / 3.0) if bool(x) else 0.0


@lib
def tan(x: float) -> float:
    """
    tan = sin/cos may diverge to +-inf at a pole where cos rounds to zero (the format-nearest pi/2),
    whereas the float64 reference stays finite there.
    """
    s, c = sin(x), cos(x)
    if c == 0.0:  # a real branch (div is unspeculatable), so the pole skips the divide and asserts no div-by-zero flag
        r = math.inf if s >= 0.0 else -math.inf
    else:
        r = s / c
    return r


@lib
def atan(x: float) -> float:
    return atan2(x, 1.0)


# `1 - x*x` cancels as |x| approaches 1, exactly where these are steepest, and `ffma` answers it by not rounding
# the product first. Spelling it `(1-x)*(1+x)` would buy the same without `ffma`, but it destroys the `a*b + c`
# shape and costs a cycle everywhere -- so an accuracy-sensitive build configures `ffma` instead.
@lib
def asin(x: float) -> float:
    return atan2(x, sqrt(1.0 - x * x))


@lib
def acos(x: float) -> float:
    return atan2(sqrt(1.0 - x * x), x)


@lib
def exp(x: float) -> float:
    return exp2(x * _LOG2E)


@lib
def log(x: float) -> float:
    return log2(x) * _LN2


@lib
def log10(x: float) -> float:
    return log2(x) * _LOG10_2


@lib
def expm1(x: float) -> float:
    """FIXME Loses the small-argument precision the reference exists to preserve."""
    return exp(x) - 1.0


@lib
def log1p(x: float) -> float:
    """FIXME Loses the small-argument precision the reference exists to preserve."""
    return log(1.0 + x)


@lib
def sinh(x: float) -> float:
    return exp(x - _LN2) - exp(-x - _LN2)  # the /2 folded into the exponent, so exp does not overflow before it


@lib
def cosh(x: float) -> float:
    return exp(x - _LN2) + exp(-x - _LN2)


@lib
def tanh(x: float) -> float:
    """Stable sigmoid form: no exp overflow for large |x|. FIXME Loses precision to cancellation near zero."""
    return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0


@lib
def asinh(x: float) -> float:
    """
    Sign/abs form avoids the large-negative-x cancellation; the branch avoids x*x overflowing (to +inf, over a huge
    in-range band) before the sqrt recovers -- there sqrt(x*x + 1) == |x|, so asinh(x) == sign(x)*ln(2|x|).
    """
    t = x * x
    if isinf(t):
        r = sign_float(x) * (log(abs(x)) + _LN2)
    else:
        r = sign_float(x) * log(abs(x) + sqrt(t + 1.0))
    return r


@lib
def acosh(x: float) -> float:
    """The branch avoids x*x overflowing before the sqrt; there sqrt(x*x - 1) == x, so acosh(x) == ln(2x)."""
    t = x * x
    if isinf(t):
        r = log(x) + _LN2
    else:
        r = log(x + sqrt(t - 1.0))
    return r


@lib
def atanh(x: float) -> float:
    return 0.5 * log((1.0 + x) / (1.0 - x))


@lib
def degrees(x: float) -> float:
    return x * _DEG_PER_RAD


@lib
def radians(x: float) -> float:
    return x * _RAD_PER_DEG
