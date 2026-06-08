#!/usr/bin/env python3
"""
The IEEE 754 floating-point remainder ``remainder(x, y) = x - n*y``, where ``n`` is ``x/y`` rounded to the nearest
integer (ties to even), so the result lies in ``[-|y|/2, +|y|/2]``.

Computed by the standard iterative reduction: scale ``|y|`` up to the largest ``2**k * |y|`` not exceeding ``|x|``,
then subtract scaled divisors while halving back down to ``|y|``, which leaves the truncated remainder
``fmod(|x|, |y|)`` in ``[0, |y|)``; a final round-to-nearest-even step centers it. Each subtraction is exact (Sterbenz),
so the result is the exact IEEE remainder whenever that remainder is representable in the configured float format --
which it is for any normal-magnitude result.

The trip count of both loops is the binary magnitude ratio of ``x`` to ``y``, hence data-dependent --
the machine runs a variable number of cycles per input.
"""

from pathlib import Path
import holoso


def remainder(x: float, y: float) -> float:
    ax = abs(x)
    ay = abs(y)
    scaled = ay
    while scaled + scaled <= ax:  # largest 2**k * |y| not exceeding |x|
        scaled = scaled + scaled
    r = ax
    quotient_is_odd = 0.0  # low bit of the integer quotient, for round-to-even tie-breaking TODO FIXME make it boolean!
    while scaled > ay:  # subtract scaled divisors, halving down to -- but not past -- the unit place |y|
        if r >= scaled:
            r = r - scaled
        scaled = scaled * 0.5
    # The unit place (scaled == ay) is handled explicitly rather than by halving once more: in ZKF there are no
    # subnormals, so |y| * 0.5 of a tiny |y| can clamp back to |y| and never fall below it, so the loop must not depend
    # on scaled dropping under |y| to stop.
    if r >= ay:
        r = r - ay
        quotient_is_odd = 1.0  # subtracting at the unit place sets the quotient's least-significant bit
    # r is now fmod(|x|, |y|) in [0, |y|). Round the quotient to nearest, ties to even: pull r down by |y| when it is
    # past the half, or exactly at the half with an odd quotient.
    twice_r = r + r
    if twice_r > ay:
        r = r - ay
    elif twice_r == ay:
        if quotient_is_odd > 0.5:
            r = r - ay
    # Apply x's sign to a nonzero result (r is the centered remainder, which the rounding step may have made negative);
    # an exact-zero remainder of a negative x must stay the canonical +0.0, since ZKF has no negative zero.
    return -r if x < 0.0 and r != 0.0 else r


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    ops = holoso.OpConfig(
        holoso.FAddOperator(float_format),
        holoso.FMulOperator(float_format),
        holoso.FDivOperator(float_format),
        holoso.FMulILog2OperatorFamily(float_format),
        holoso.FCmpOperator(float_format),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(remainder, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()
