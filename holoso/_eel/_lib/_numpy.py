"""
Composite math/numpy stubs the frontend inlines like ordinary user functions, expressed via the intrinsics,
that is, bare-hardware operators. Fast-math transforms are legal here, which may alter semantics.

Math domain violations raise in plain Python but garbage-in-garbage-out in hardware,
with the error output signals asserted; refer to the operator RTL for details. A violation the compiler happens to
prove statically is refused at build time instead -- a liberty it takes where a fold reaches, never a guarantee.

Composites compute through the intrinsic stubs (exp2, atan2, ...) instead of the library functions directly even
though the library spellings would lower identically (they are registered intrinsic keys too). The intrinsic stub
pins each primitive to the numpy/math variant matching the hardware behavior -- e.g. exp2 saturates to inf like
the hardware where math.exp2 would raise -- so a composite built on the stubs inherits that hardware-faithful behavior,
and its plain-Python run uses exactly the primitives (and fast-math choices) it lowers to.

Stub names are irrelevant to dispatch.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ._intrinsics import atan2, ceil, cos, exp2, floor, isinf, log2, round_, sin, sqrt
from ._registry import array, lib, lift

_LOG2E = math.log2(math.e)
_LN2 = math.log(2.0)
_LOG10_2 = math.log10(2.0)
_DEG_PER_RAD = 180.0 / math.pi
_RAD_PER_DEG = math.pi / 180.0
_INF = math.inf


@lib(math.floor, np.floor, math.ceil, np.ceil, math.trunc, np.trunc, np.fix, round, np.round, np.around)
def integral(x: int) -> int:
    """An integer entry exists exactly where the Python spelling's own answer is an integer."""
    return x


# On a float the math spellings answer an int where the numpy ones answer a float, so the two share no symbol.
# A float consumer reduces the cast back to the bare rounder.
@lib(math.floor)
def floor_to_int(x: float) -> int:
    return int(floor(x))


@lib(math.ceil)
def ceil_to_int(x: float) -> int:
    return int(ceil(x))


@lib(math.trunc)
def trunc_to_int(x: float) -> int:
    return int(x)  # truncation toward zero is what the cast already is


@lib(round)
def round_to_int(x: float) -> int:
    return int(round_(x))


@lib(min)
def min_int(a: int, b: int) -> int:
    """There is no hardware min/max operator for integers."""
    return a if a <= b else b  # the tie answers a, as CPython's min does


@lib(max)
def max_int(a: int, b: int) -> int:
    return a if b <= a else b


def _elementwise(f: Callable[[Any, Any], Any], a: np.ndarray, b: np.ndarray) -> Any:
    """
    Rank <= 2 with scalar broadcast against either side; the general vector-against-matrix broadcast is not
    supported. Operands are numpy-typed, as everywhere in this library: a scalar answers ndim 0, which a native
    Python number does not. A mixed-dtype result carries the winning elements' own types, and each comparison is
    exact where numpy compares after promotion -- so past the float's exact-integer range a mixed pair can answer
    the unpromoted neighbor of numpy's value. The compiled path conforms mixed operands C-style and follows numpy.
    """
    if a.ndim > 2 or b.ndim > 2:
        raise ValueError(f"elementwise operands must be at most 2-D, got {a.ndim}-D and {b.ndim}-D")
    if a.ndim == 0 and b.ndim == 0:
        return f(a, b)
    if b.ndim == 0:
        if a.ndim == 1:
            return np.array([f(a[i], b) for i in range(len(a))])
        return np.array([[f(a[i][j], b) for j in range(len(a[0]))] for i in range(len(a))])
    if a.ndim == 0:
        if b.ndim == 1:
            return np.array([f(a, b[j]) for j in range(len(b))])
        return np.array([[f(a, b[i][j]) for j in range(len(b[0]))] for i in range(len(b))])
    if a.ndim != b.ndim:
        raise ValueError(f"elementwise shape mismatch: {a.ndim}-D against {b.ndim}-D; broadcasting is not supported")
    if len(a) != len(b):
        raise ValueError(f"elementwise shape mismatch: length {len(a)} against {len(b)}")
    if a.ndim == 1:
        return np.array([f(a[i], b[i]) for i in range(len(a))])
    if len(a[0]) != len(b[0]):
        raise ValueError(f"elementwise shape mismatch: rows of length {len(a[0])} against {len(b[0])}")
    return np.array([[f(a[i][j], b[i][j]) for j in range(len(a[0]))] for i in range(len(a))])


# The NaN-propagation difference between np.minimum/np.fmin (and np.maximum/np.fmax) is moot under the no-NaN policy.
@array(np.minimum, np.fmin)
def minimum(a: np.ndarray, b: np.ndarray) -> Any:
    return _elementwise(min, a, b)


@array(np.maximum, np.fmax)
def maximum(a: np.ndarray, b: np.ndarray) -> Any:
    return _elementwise(max, a, b)


@array(np.ndarray.clip)
def clip(x: np.ndarray, lo: Any = None, hi: Any = None) -> Any:
    """
    Saturation as the maximum/minimum composition, following numpy: the lower bound applies first, so an inverted
    pair answers `hi`, and an omitted bound skips that side (the subset admits no `None` argument, so one-sided
    saturation from above is spelled `np.minimum`). Bounds broadcast per the elementwise rules.
    """
    r = x
    if lo is not None:
        r = maximum(r, lo)
    if hi is not None:
        r = minimum(r, hi)
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


@lib(np.sign)
def sign_int(x: int) -> int:
    if x > 0:
        r = 1
    elif x < 0:
        r = -1
    else:
        r = 0
    return r


@lib(np.sign)
def sign_float(x: float) -> float:
    if x > 0.0:
        r = +1.0
    elif x < 0.0:
        r = -1.0
    else:
        r = x
    return r


@lib(math.cbrt, np.cbrt)
def cbrt(x: float) -> float:
    """Fastmath: cbrt(−0.0) may return +0.0"""
    return sign_float(x) * exp2(log2(abs(x)) / 3.0) if bool(x) else 0.0


@lib(math.tan, np.tan)
def tan(x: float) -> float:
    """
    tan = sin/cos may diverge to +-inf at a pole where cos rounds to zero (the format-nearest pi/2),
    whereas the float64 reference stays finite there.
    """
    s, c = sin(x), cos(x)
    if c == 0.0:  # a real branch (div is unspeculatable), so the pole skips the divide and asserts no div-by-zero flag
        r = _INF if s >= 0.0 else -_INF
    else:
        r = s / c
    return r


@lib(math.atan, np.arctan, np.atan)
def atan(x: float) -> float:
    return atan2(x, 1.0)


@lib(math.asin, np.arcsin, np.asin)
def asin(x: float) -> float:
    return atan2(x, sqrt(1.0 - x * x))


@lib(math.acos, np.arccos, np.acos)
def acos(x: float) -> float:
    return atan2(sqrt(1.0 - x * x), x)


@lib(math.exp, np.exp)
def exp(x: float) -> float:
    return exp2(x * _LOG2E)


@lib(math.log, np.log)
def log(x: float) -> float:
    return log2(x) * _LN2


@lib(math.log10, np.log10)
def log10(x: float) -> float:
    return log2(x) * _LOG10_2


@lib(math.expm1, np.expm1)
def expm1(x: float) -> float:
    """FIXME Loses the small-argument precision the reference exists to preserve."""
    return exp(x) - 1.0


@lib(math.log1p, np.log1p)
def log1p(x: float) -> float:
    """FIXME Loses the small-argument precision the reference exists to preserve."""
    return log(1.0 + x)


@lib(math.sinh, np.sinh)
def sinh(x: float) -> float:
    return exp(x - _LN2) - exp(-x - _LN2)  # the /2 folded into the exponent, so exp does not overflow before it


@lib(math.cosh, np.cosh)
def cosh(x: float) -> float:
    return exp(x - _LN2) + exp(-x - _LN2)


@lib(math.tanh, np.tanh)
def tanh(x: float) -> float:
    """Stable sigmoid form: no exp overflow for large |x|. FIXME Loses precision to cancellation near zero."""
    return 2.0 / (1.0 + exp2(-2.0 * x * _LOG2E)) - 1.0


@lib(math.asinh, np.asinh, np.arcsinh)
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


@lib(math.acosh, np.acosh, np.arccosh)
def acosh(x: float) -> float:
    """The branch avoids x*x overflowing before the sqrt; there sqrt(x*x - 1) == x, so acosh(x) == ln(2x)."""
    t = x * x
    if isinf(t):
        r = log(x) + _LN2
    else:
        r = log(x + sqrt(t - 1.0))
    return r


@lib(math.atanh, np.atanh, np.arctanh)
def atanh(x: float) -> float:
    return 0.5 * log((1.0 + x) / (1.0 - x))


@lib(math.degrees, np.degrees, np.rad2deg)
def degrees(x: float) -> float:
    return x * _DEG_PER_RAD


@lib(math.radians, np.radians, np.deg2rad)
def radians(x: float) -> float:
    return x * _RAD_PER_DEG


# The array-capable spellings, declared here because this module and the intrinsics it imports register them all.
# Only numpy keys qualify: their `math` twins raise on an array, and a lifted predicate or bit count would mint a
# boolean or unsigned array, families the subset does not have.
lift(
    abs, np.abs, np.absolute, np.fabs, np.sign,
    np.floor, np.ceil, np.trunc, np.fix, np.round, np.around, np.rint,
    np.sqrt, np.cbrt, np.exp, np.exp2, np.expm1, np.log, np.log2, np.log10, np.log1p,
    np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan,
    np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh,
    np.degrees, np.rad2deg, np.radians, np.deg2rad,
)  # fmt: skip
