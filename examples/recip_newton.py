#!/usr/bin/env python3
"""
Reciprocal by Newton-Raphson iteration, with no hardware divider: `y <- y * (2 - x*y)` converges quadratically to 1/x.
A linear seed `y0 = 1.5 - 0.5*x` is accurate on the restricted domain `x in [0.5, 2.0]`, and the loop runs until the
update `delta = y_next - y` falls below a tolerance.

This is illustrative; obviously Holoso can compute the same in hardware natively.
"""

from pathlib import Path
import holoso


class NewtonReciprocal:
    """Computes 1/x with the given tolerance."""

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
    options = holoso.Options(
        operator=holoso.OperatorOptions(
            fadd=holoso.FAddOptions(
                stage_input=0,
                stage_core=holoso.FAddOptions.CoreStages.OPERANDS
                | holoso.FAddOptions.CoreStages.ALIGNED
                | holoso.FAddOptions.CoreStages.NORMALIZE
                | holoso.FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=3, stage_output=0),
            fdivsqrt=holoso.FDivsqrtOptions(
                stage_input=0, stage_core=holoso.FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
            ),
            fmul_ilog2=holoso.FMulILog2Options(
                stage_input=0, stage_core=holoso.FMulILog2Options.CoreStages.NONE, stage_output=0
            ),
            fcmp=holoso.FCmpOptions(stage_input=0, stage_output=0),
        ),
        ffmt=float_format,
        wmultiplier=18,
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(NewtonReciprocal().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()
