#!/usr/bin/env python3
"""
Reciprocal by Newton-Raphson iteration, with no hardware divider: ``y <- y * (2 - x*y)`` converges quadratically to
``1/x``. A linear seed ``y0 = 1.5 - 0.5*x`` is accurate on the restricted domain ``x in [0.5, 2.0]``, and the loop
runs until the update ``delta = y_next - y`` falls below a tolerance -- a variable, convergence-tested iteration count.
This is the real back-edge loop: the body lowers once to a basic block whose latch jumps back to the loop header, and
the header re-tests the condition each iteration (unlike the fixed-count loops, e.g. CORDIC, which fully unroll).
"""

from pathlib import Path

import holoso


class NewtonReciprocal:
    def __init__(self, *, tolerance: float = 2.0**-12) -> None:
        self.tolerance: float = tolerance

    def __call__(self, x: float, /) -> float:
        y = 1.5 - 0.5 * x  # linear initial guess; one Newton step roughly doubles the count of correct digits
        delta = 1.0  # seed the update above the tolerance so the convergence test runs at least once
        while abs(delta) > self.tolerance:  # a read-only attribute folds to its snapshot constant in the condition
            y_next = y * (2.0 - x * y)
            delta = y_next - y
            y = y_next
        return y


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
    result = holoso.synthesize(NewtonReciprocal().__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()
