#!/usr/bin/env python3
"""
Rotation-mode CORDIC computing ``(cos theta, sin theta)`` with no multiplier on the rotation: each iteration is an
add/subtract and a power-of-two scale ``2**-i`` (an exact shift), with a per-iteration sign decision driving the
direction of micro-rotation. The fixed arctan table and the aggregate gain fold at compile time; the loop unrolls
because the shift amount ``2**-i`` is a per-iteration compile-time constant (a runtime variable shift is not a Holoso
operator). The result is a flat sequence of sign-branch diamonds -- comparisons feeding branches over a small datapath.
"""

import math
from pathlib import Path

import holoso

ITERATIONS = 12


class CordicSinCos:
    def __init__(self, *, iterations: int = ITERATIONS) -> None:
        self.iterations: int = iterations
        gain = 1.0
        for i in range(iterations):
            gain *= 1.0 / math.sqrt(1.0 + 2.0 ** (-2 * i))
        self.gain: float = gain  # the CORDIC scale factor, folded in as the seed for x
        self.angles: tuple[float, ...] = tuple(math.atan(2.0**-i) for i in range(iterations))

    def __call__(self, theta: float, /) -> tuple[float, float]:
        x = self.gain
        y = 0.0
        z = theta  # residual angle, driven toward zero
        for i in range(ITERATIONS):
            if z >= 0.0:
                x_next = x - y * (2.0**-i)
                y_next = y + x * (2.0**-i)
                z = z - self.angles[i]
            else:
                x_next = x + y * (2.0**-i)
                y_next = y - x * (2.0**-i)
                z = z + self.angles[i]
            x = x_next
            y = y_next
        return x, y  # (cos theta, sin theta)


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
    result = holoso.synthesize(CordicSinCos().__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()
