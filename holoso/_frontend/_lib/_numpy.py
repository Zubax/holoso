"""
Composite math/numpy stubs the frontend inlines like ordinary user functions, expressed via the intrinsics,
that is, bare-hardware operators. Fast-math transforms are legal here, which may alter semantics.

Math domain violations raise in plain Python but garbage-in-garbage-out in hardware,
with the error output signals asserted; refer to the operator RTL for details.

Composites compute through the intrinsic stubs (exp2_, atan2_, ...) instead of the library functions directly even
though the library spellings would lower identically (they are registered intrinsic keys too). The intrinsic stub
pins each primitive to the numpy/math variant matching the hardware behavior -- e.g. exp2_ saturates to inf like
the hardware where math.exp2 would raise -- so a composite built on the stubs inherits that hardware-faithful behavior,
and its plain-Python run uses exactly the primitives (and fast-math choices) it lowers to.

Stub names are irrelevant to dispatch; underscores dodge the builtin names.
"""

import math

import numpy as np

from ._intrinsics import atan2_, cos_, exp2_, isinf_, log2_, sin_, sqrt_
from ._registry import lib

_LOG2E = math.log2(math.e)
_LN2 = math.log(2.0)
_LOG10_2 = math.log10(2.0)
_DEG_PER_RAD = 180.0 / math.pi
_RAD_PER_DEG = math.pi / 180.0
_INF = math.inf


@lib(np.sign)
def sign_(x: float) -> float:
    if x > 0.0:
        r = +1.0
    elif x < 0.0:
        r = -1.0
    else:
        r = x
    return r


@lib(math.cbrt, np.cbrt)
def cbrt_(x: float) -> float:
    """Fastmath: cbrt(−0.0) may return +0.0"""
    return sign_(x) * exp2_(log2_(abs(x)) / 3.0) if bool(x) else 0.0


@lib(math.tan, np.tan)
def tan_(x: float) -> float:
    """
    tan = sin/cos may diverge to +-inf at a pole where cos rounds to zero (the format-nearest pi/2),
    whereas the float64 reference stays finite there.
    """
    s, c = sin_(x), cos_(x)
    if c == 0.0:  # a real branch (div is unspeculatable), so the pole skips the divide and asserts no div-by-zero flag
        r = _INF if s >= 0.0 else -_INF
    else:
        r = s / c
    return r


@lib(math.atan, np.arctan, np.atan)
def atan_(x: float) -> float:
    return atan2_(x, 1.0)


@lib(math.asin, np.arcsin, np.asin)
def asin_(x: float) -> float:
    return atan2_(x, sqrt_(1.0 - x * x))


@lib(math.acos, np.arccos, np.acos)
def acos_(x: float) -> float:
    return atan2_(sqrt_(1.0 - x * x), x)


@lib(math.exp, np.exp)
def exp_(x: float) -> float:
    return exp2_(x * _LOG2E)


@lib(math.log, np.log)
def log_(x: float) -> float:
    return log2_(x) * _LN2


@lib(math.log10, np.log10)
def log10_(x: float) -> float:
    return log2_(x) * _LOG10_2


@lib(math.expm1, np.expm1)
def expm1_(x: float) -> float:
    """FIXME Loses the small-argument precision the reference exists to preserve."""
    return exp_(x) - 1.0


@lib(math.log1p, np.log1p)
def log1p_(x: float) -> float:
    """FIXME Loses the small-argument precision the reference exists to preserve."""
    return log_(1.0 + x)


@lib(math.sinh, np.sinh)
def sinh_(x: float) -> float:
    return exp_(x - _LN2) - exp_(-x - _LN2)  # the /2 folded into the exponent, so exp does not overflow before it


@lib(math.cosh, np.cosh)
def cosh_(x: float) -> float:
    return exp_(x - _LN2) + exp_(-x - _LN2)


@lib(math.tanh, np.tanh)
def tanh_(x: float) -> float:
    """Stable sigmoid form: no exp overflow for large |x|. FIXME Loses precision to cancellation near zero."""
    return 2.0 / (1.0 + exp2_(-2.0 * x * _LOG2E)) - 1.0


@lib(math.asinh, np.asinh, np.arcsinh)
def asinh_(x: float) -> float:
    """
    Sign/abs form avoids the large-negative-x cancellation; the branch avoids x*x overflowing (to +inf, over a huge
    in-range band) before the sqrt recovers -- there sqrt(x*x + 1) == |x|, so asinh(x) == sign(x)*ln(2|x|).
    """
    t = x * x
    if isinf_(t):
        r = sign_(x) * (log_(abs(x)) + _LN2)
    else:
        r = sign_(x) * log_(abs(x) + sqrt_(t + 1.0))
    return r


@lib(math.acosh, np.acosh, np.arccosh)
def acosh_(x: float) -> float:
    """The branch avoids x*x overflowing before the sqrt; there sqrt(x*x - 1) == x, so acosh(x) == ln(2x)."""
    t = x * x
    if isinf_(t):
        r = log_(x) + _LN2
    else:
        r = log_(x + sqrt_(t - 1.0))
    return r


@lib(math.atanh, np.atanh, np.arctanh)
def atanh_(x: float) -> float:
    return 0.5 * log_((1.0 + x) / (1.0 - x))


@lib(math.degrees, np.degrees, np.rad2deg)
def degrees_(x: float) -> float:
    return x * _DEG_PER_RAD


@lib(math.radians, np.radians, np.deg2rad)
def radians_(x: float) -> float:
    return x * _RAD_PER_DEG


@lib(pow, math.pow, np.power, np.pow, np.float_power)
def pow_(b: float, e: float) -> float:
    """
    General path is the a>0 identity exp2(e*log2(b)); a negative base is only honored on small-integer e.
    TODO generalize.
    """
    if e == 0.0 or b == 1.0:  # b==1 avoids IEEE 754 divergence on non-finite e
        r = 1.0
    elif e == 1.0 or b == 0.0:  # b==0 avoids the pole error signal on log2(0)
        r = b
    elif e == 2.0:
        r = b * b
    elif e == 3.0:
        r = b * b * b
    elif e == 4.0:
        r = (b * b) * (b * b)
    elif e == 5.0:
        r = (b * b) * (b * b * b)
    else:
        r = exp2_(e * log2_(b))
    return r
