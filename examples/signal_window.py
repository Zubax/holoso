#!/usr/bin/env python3
"""A stateless signal-window conditioner exercising various boolean/float expression forms."""

from pathlib import Path

import holoso


def signal_window(x: float, lo: float, hi: float) -> tuple[float, bool, bool, bool, float]:
    """
    Outputs (matching the return tuple `(float, bool, bool, bool, float)`):
    - clamped (float): x clamped into [lo, hi] with a nested conditional (ternary) expression;
    - inside (bool): whether x lies strictly inside the open window, straight from a chained comparison `lo < x < hi`;
    - outside (bool): the boundary/outside region, an or of two boundary comparisons `x <= lo or x >= hi`;
    - live (bool): a "live" sample -- nonzero and inside -- a float->bool cast `bool(x)` and-ed with a comparison;
    - gated (float): the input passed through only inside the window -- a cross-domain chain
      (comparison -> bool -> float cast -> float multiply) that the bool->float result must feed on time.
    """
    # The two ternary arms of `clamped` if-convert to selects, leaving a single straight-line block.
    # The comparisons, connectives, and casts are combinational options within it; the kernel is branchless.
    clamped = hi if x > hi else (lo if x < lo else x)
    inside = lo < x < hi
    outside = x <= lo or x >= hi
    live = bool(x) and lo < x < hi
    gated = float(lo < x < hi) * x
    return clamped, inside, outside, live, gated


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    options = holoso.Options(
        operator=holoso.OperatorOptions(
            fadd=holoso.FAddOptions(stage_input=0, stage_core=holoso.FAddOptions.CoreStages.NONE, stage_output=0),
            fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=3, stage_output=1),
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
    result = holoso.synthesize(signal_window, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()
