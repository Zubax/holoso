"""
The five lowerings of a power; the exponent's binding time, sign and -- for the root -- value, declared on the
stubs, are what select between them. There is no integer power hardware, so only a compile-time base stays
integral. The spellings split by what they do with an integer base: `**`, `pow`, and numpy's power keep it
integral (so the chain saturates as any int expression does), while `math.pow` and `np.float_power` compute in
floating point regardless and never reach the integer chain.

Exponentiation is square-and-multiply rather than a linear chain: shorter, and free of any residual branch, at an
accuracy cost the chain did not pay -- reassociating the products drifts to tens of ULP around `n = 100`, which
the fast-math charter allows. The stubs compute through the intrinsic stubs like the rest of the library, so a fold
saturates where the host raises: `2.0 ** 10000` is `inf`, as `exp2(1e30)` already is.
"""

import math
from typing import TypeVar

import numpy as np

from .._ir import BinaryOp
from ._registry import StaticNegative, StaticNonNegative, StaticOneHalf, lib
from ._intrinsics import exp2, isinf, log2, round_, sqrt

_INF = math.inf
_N = TypeVar("_N", int, float)
_POW_INT_PRESERVING = (pow, np.power, np.pow, BinaryOp.POW)
_POW = (*_POW_INT_PRESERVING, math.pow, np.float_power)


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


@lib(*_POW_INT_PRESERVING)
def pow_chain_int(b: int, e: StaticNonNegative[int]) -> int:
    """The base's sign needs no case of its own: it rides the multiplies."""
    return _chain(1, b, e)


@lib(*_POW)
def pow_chain_float(b: float, n: StaticNonNegative[int]) -> float:
    return _chain(1.0, b, n)


@lib(*_POW)
def pow_reciprocal(b: float, n: StaticNegative[int]) -> float:
    return 1.0 / pow_chain_float(b, -n)


@lib(*_POW)
def pow_root(b: float, e: StaticOneHalf[float]) -> float:
    """
    One correctly-rounded root instead of the general path's exp2/log2 pair, and none of its guards: the root
    answers zero and infinity itself, and a negative base is off its domain as it is off the power's.
    """
    return sqrt(b)


@lib(*_POW)
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
        r = 0.0 if e > 0.0 else _INF if e < 0.0 else e  # e == 0 is unreachable here, so the last arm is NaN
    else:
        t = exp2(e * log2(abs(b) if integral else b))
        r = -t if b < 0.0 and integral and odd else t
    return r
