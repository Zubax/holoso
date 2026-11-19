#!/usr/bin/env python3
"""
The order of magnitude of a number in octaves: how many halvings (or, for a magnitude below unity, doublings) bring
|x| into the unit interval (0.5, 1]. This is ceil(|log2(|x|)|) computed with only compares, a reciprocal,
and a halving loop -- a pure-arithmetic exponent estimator for designs without bit-level access to the float field
(auto-ranging front-ends, gain staging, coarse logarithms).

This is illustrative; Holoso supports transcendental functions natively in hardware.
"""

from pathlib import Path
import holoso


def octave_index(x: float) -> int:
    magnitude = abs(x)
    if magnitude >= 1.0:
        scaled = magnitude
    else:
        scaled = 1.0 / magnitude  # the lone non-speculatable op: keeps the diamond a real branch
    octaves = 0  # rides the integer datapath while the magnitude stays float
    while scaled > 1.0:
        scaled = scaled * 0.5  # absorbed by the power-of-two scaler, so no float multiplier is built
        octaves = octaves + 1
    return octaves


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    options = holoso.Options(
        operator=holoso.OperatorOptions(
            fdivsqrt=holoso.FDivsqrtOptions(
                stage_input=0, stage_core=holoso.FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
            ),
            fmul_ilog2=holoso.FMulILog2Options(
                stage_input=0, stage_core=holoso.FMulILog2Options.CoreStages.NONE, stage_output=1
            ),
            fcmp=holoso.FCmpOptions(stage_input=0, stage_output=1),
        ),
        ffmt=float_format,
        wmultiplier=18,
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(octave_index, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()
