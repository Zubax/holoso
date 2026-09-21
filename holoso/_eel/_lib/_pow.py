"""
The five lowerings of a power; the exponent's binding time, sign, wholeness and -- for the root -- value, declared
on the stubs, are what select between them. Wholeness is judged by value, not by type, because a whole exponent is
often spelled as a float: SymPy's numpy printer emits a reciprocal as `x ** (-1.0)`, which would otherwise buy the
exp2/log2 pair for a divide. There is no integer power hardware, so an integral result needs a whole exponent, and
the spellings split by what they do with one: `**`, `pow`, and numpy's power keep it integral (so the chain
saturates as any int expression does), while `math.pow` and `np.float_power` compute in floating point regardless.

Exponentiation is square-and-multiply rather than a linear chain: shorter, and free of any residual branch, at an
accuracy cost the chain did not pay -- reassociating the products drifts to tens of ULP around `n = 100`, which
the fast-math charter allows. The stubs compute through the intrinsic stubs like the rest of the library, so a fold
saturates where the host raises: `2.0 ** 10000` is `inf`, as `exp2(1e30)` already is.
"""

import math
from typing import TypeVar

from ._registry import StaticOneHalf, StaticWholeNegative, StaticWholeNonNegative, lib
from ._intrinsics import exp2, isinf, log2, round_, sqrt

_N = TypeVar("_N", int, float)

# The float chain needs no transcendental hardware and measures at least as accurate as the general rung at every
# exponent both can express, so this bound is deliberately a length cap and nothing more: 128 keeps the chain inside
# a couple of dozen cycles while covering every exponent a kernel spells by hand.
_CHAIN_MAX = 128


def _chain(acc: _N, base: _N, k: int) -> _N:
    """
    The seed is what types the whole power, so it is the caller's to choose. The leading `acc * base` is a
    multiply by one, left for the HIR's identity elision rather than dodged with a flag here.
    """
    while k > 0:  # a while, not a range: unrolling a static test costs no materialized sequence
        if k % 2 == 1:
            acc = acc * base
        k = k // 2
        if k > 0:
            base = base * base
    return acc


@lib
def pow_chain_int(b: int, e: StaticWholeNonNegative[int]) -> int:
    """The base's sign needs no case of its own: it rides the multiplies."""
    return _chain(1, b, e)


@lib
def pow_chain_float(b: float, n: StaticWholeNonNegative[float]) -> float:
    if n > _CHAIN_MAX:
        return pow_(b, n)
    return _chain(1.0, b, int(n))


@lib
def pow_reciprocal(b: float, n: StaticWholeNegative[float]) -> float:
    if n < -_CHAIN_MAX:
        return pow_(b, n)
    return 1.0 / pow_chain_float(b, -n)


@lib
def pow_root(b: float, e: StaticOneHalf[float]) -> float:
    """
    One correctly-rounded root instead of the general path's exp2/log2 pair, and none of its guards: the root
    answers zero and infinity itself, and a negative base is off its domain as it is off the power's.
    """
    return sqrt(b)


@lib
def pow_(b: float, e: float) -> float:
    """
    Optimized for exactly one exp2 and one log2 as they dominate the hardware cost.
    The parity test is exact over the whole float range: every float >= 2**53 is even.
    """
    # Schedule the speculable general-case ops early so they overlap with the guards.
    integral = round_(e) == e
    half = e * 0.5  # from e, not round_(e): the two rounds then schedule in parallel; equal when it matters
    odd = round_(half) != half
    if e == 0.0 or b == 1.0 or (b == -1.0 and isinf(e)):  # |b|==1 with non-finite e: exp2(inf*0), IEEE says 1
        r = 1.0
    elif b == 0.0:  # keeps the log2 pole (and its error sideband) away from the degenerate base
        r = 0.0 if e > 0.0 else math.inf if e < 0.0 else e  # e == 0 is unreachable here, so the last arm is NaN
    else:
        t = exp2(e * log2(abs(b) if integral else b))
        r = -t if b < 0.0 and integral and odd else t
    return r
